# FishLab runbook

One file for running the system: which command, why, and what happens inside. Specs and history live in
[PRD.md](PRD.md), [progress.md](../backend/prepds/docs/progress.md) and
[classifier_progress.md](../backend/dcs/docs/classifier_progress.md); you should not need them to run anything.

**Find your task**

| I want to... | Go to |
|---|---|
| Start or stop the app, review videos | [1. Daily use](#1-daily-use-start-review-stop) |
| Know what a command does before I run it | [2. Command cheat sheet](#2-command-cheat-sheet) |
| Understand how a video becomes a label | [3. How the system works](#3-how-the-system-works) |
| Run one `prepds` step by hand | [4. prepds](#4-prepds-video-to-labeled-data) |
| Train or use the compound classifier | [5. dcs](#5-dcs-compound-and-dose-classifier) |
| Ask the research chat | [6. Research chat](#6-research-chat) |
| Know how far to trust the output | [7. What to trust](#7-what-to-trust) |
| Fix an error | [8. When something fails](#8-when-something-fails) |
| Retrain the fish detector | [9. Fish detector (optional)](#9-fish-detector-optional) |

---

## 1. Daily use: start, review, stop

You need Docker with Compose. No Python or Node on the host.

```sh
cp .env.example .env    # once. Set PDS_VIDEO_DIR, PDS_DB_PATH, FISHLAB_HOST=127.0.0.1, FISHLAB_PORT=<free port>
./start.sh --setup      # once per machine, and again when videos or the workbook change; then it stops
./start.sh              # start the app and print the link to open
./stop.sh               # stop the app and remove the containers (saved work stays on disk)
```

`./start.sh` lists what is missing from `.env` and starts nothing until it is fixed. `--full` redoes every step.
Do not run `prepds` commands by hand while the app is running: two `run`s at once overwrite each other.

### Reviewing a video (what the pharmacy/lab reviewer does)

Why: the computer's labels are not ground truth. Only videos a person has accepted become trusted data for the
classifier. When you and the computer disagree, you are right.

1. Open the printed link. Click **Review**. Type your name in **Reviewer** (needed to save anything).
2. Pick a video in the left list (filter: **Needs review**).
3. Play it and watch the colored timeline. **Prepds review flags** (e.g. "no movement until the end") and
   **Possible Listing/LORR** (guesses, often wrong) jump to places worth checking. **Why this label?** explains one second.
4. To fix a stretch: **Relabel a time range** → set start/end → pick the behavior → **Stage range** (repeat) →
   **Save staged edits**. **Restore labels** returns to the computer's version.
5. **Review decision**: **Accept review** if you agree with the whole video, **Reject review** if it is unusable.

| Behavior | Look for | Who finds it |
|---|---|---|
| Controlled Swim | steady, purposeful swimming | computer (speed) |
| Erratic Movement | fast darting, sudden turns | computer (speed) |
| Freezing/Drift | nearly still or drifting | computer (speed) |
| Listing/LORR | tilted, on its side or upside down | **reviewer only** |
| Surface Breach | at or through the water surface | **reviewer only** |
| Dead | motionless from some point to the end | **reviewer only** |
| Undetermined | tracker lost the fish | **reviewer must fix** |

Rules the app enforces: **Accept is impossible while any Undetermined remains** (fix it or reject the video);
**Dead needs an explicit confirmation** (a still fish may be a long Listing, not dead); **an accepted video is
final**; unsaved staged edits block Accept. Accept means "the labels are right", not "the model is validated".
If unsure what the fish is doing: do not accept, reject and note video, time and what you saw. A wrong accepted
video is worse than a missing one. The tracking tools (waterline, ROI, tracking point, boxes, keypoints) are for
developers; ignore them. Watch for the fish's reflection at the surface, which the tracker can follow instead.

---

## 2. Command cheat sheet

Docker users only need the first four rows. Everything else is for developers and runs inside the project venv
(`source .venv/bin/activate`), from the repository root.

| Command | What it does | When |
|---|---|---|
| `./start.sh --setup` | Build, check config, catalog, track every new video, extract features, train if no model, then stop | first time, or new inputs |
| `./start.sh` | Start the app, print the link | every day |
| `./stop.sh` | Stop the app | when done |
| `python -m prepds check-config` | Print resolved paths and settings; exit 1 if a path is missing | first sign of trouble |
| `python -m prepds catalog` | Match workbook trial rows to video files | after new videos or workbook |
| `python -m prepds run --dry-run` | **List** the videos that would be processed, change nothing (there is no `--preview`) | before a long run |
| `python -m prepds run [--workers N] [--limit N]` | Track, label, export every unprocessed video | normally via `--setup` |
| `python -m prepds run --tracker model:<run dir> --stride 5 --device cuda` | Same, with the fine-tuned detector | only if you have a trained model |
| `python -m prepds review` | Review app without Docker (loopback only) | development |
| `python -m prepds export-index` | Rebuild the accepted index from accepted videos | after accepting videos, before `dcs` |
| `python -m dcs check-config` | Print classifier paths and settings | before `dcs` |
| `python -m dcs featurize` | One row per fish → `training_table.parquet` | after prepds output changes |
| `python -m dcs audit` | Data-quality report and go/wait verdict (`audit.md`) | before training |
| `python -m dcs train` | Train all models, ablations, dose models; save the best | after `featurize` |
| `python -m dcs predict --model <run> --input <table>` | Class probabilities for fish from a saved model | scoring new fish |
| `python -m dcs ask "question"` | Research chat in the terminal | any time a model server runs |

Global options go **before** the subcommand: `python -m dcs --config my.yaml featurize`.

---

## 3. How the system works

```
videos + trial workbook
   │  prepds catalog ─ match each trial row to a video (unmatched → exceptions_report.json)
   ▼
prepds run: track fish → features → label each frame → vote per 1 s → review flags → strip.png → export
   ▼                                                                 (outputs/processed/<video>/)
review app: a person fixes labels, resolves Undetermined, Accepts  → accepted/<video>/ + accepted_index.parquet
   ▼
dcs featurize → training_table.parquet (one row per fish) → dcs train → run folder + saved model
   ▼                                                              ▼
dcs audit → audit.md                                       dcs predict / dcs ask (chat)
```

The two packages are separate on purpose: `prepds` makes labeled data, `dcs` learns from it and never imports
`prepds` or changes its files.

**Why this design.** Behavior labels must be checkable by a human (the review app), and the classifier must only
learn from checked data (the accepted folder). Until enough videos are accepted, `dcs` runs on the unreviewed
output, marked **UNREVIEWED**, so no result is a claim about drugs yet.

### How a video becomes a label

1. **Track.** Classical OpenCV: KNN background subtraction finds the moving fish each frame. A motionless fish is
   absorbed into the background and becomes "undetected". The optional fine-tuned detector (section 9) finds
   the fish in a box even when still.
2. **Features.** Speed (~1 s displacement), acceleration, turning, depth, from the track.
3. **Label.** A speed band: at or below a floor → Freezing/Drift; at or above a threshold → Erratic; between →
   Controlled; no speed (fish not found) → Undetermined. Precedence: undetected > Dead > Surface Breach > Listing
   (off) > Freezing > Erratic > Controlled.
4. **Vote.** Frame labels are majority-voted in 1 s bins.
5. **Flag.** Suspicious cases (e.g. fish still to the end of the video) get a review flag.
6. **Calibration.** The three thresholds were fitted once against two reference figures (frozen profile, e.g.
   `cal-2026-09-24-r4-model-m3`). Only 7 compound/dose groups could be calibrated, so labels for other
   compounds are extrapolations.

Per-video files in `outputs/processed/<video>/`: `frames.parquet` (per-frame track, features, state),
`segments.csv` (run-length encoding), `strip.png` (colored strip), `manifest.json` (status, provenance, flags).
Accepting copies them to `accepted/<video>/` and updates `accepted/accepted_index.parquet` (the gold dataset).

### What `./start.sh` runs

| # | Step | Detail |
|---|---|---|
| 1 | Build | Docker image with frontend and Python (minutes the first time, then cached) |
| 2 | Check + catalog | `check-config`, `catalog` |
| 3 | Track | `prepds run`, ~35 s per 20-min video per worker (328 videos ≈ 50 min on 22 workers); resumable; edited or accepted videos are never regenerated |
| 4 | Verify | every video has its output files |
| 5 | DCS | `featurize`, `check-config`, `audit`, train only if no saved model, `predict` |
| 6 | Chat | only if `FISHLAB_DCS_CHAT=1`: start the local model server |
| 7 | Serve | start app, wait for health, print the link |

It stops at the first failed step and names it; details in `<PDS_OUTPUT_DIR>/docker/startup.json`.

---

## 4. prepds: video to labeled data

| Command | Why it exists | How it works |
|---|---|---|
| `check-config` | Catch a wrong path before a 50-minute run | Prints each setting and path, exit 1 if a configured path is missing |
| `catalog` | The video name does not carry the compound; the workbook does | Matches trial rows to files (compound folders found at any depth under `PDS_VIDEO_DIR`); checks durations; writes `trials_catalog.parquet` and `exceptions_report.json` |
| `run` | The main pipeline | One video per worker process; resumable; `--force` regenerates **including edited and accepted work**, so avoid it; a failing video goes to `run_report.json` and the run continues (exit 1 if any failed) |
| `run --tracker model:<dir>` | Better detection than the classical tracker | Runs the detector every Nth frame (`--stride`, default 5), interpolates only between two detections; single process on GPU, ~75 s per video; needs the `ml` extra and its own `--config` profile and output folders (never over `outputs/processed`) |
| `review` | Human check | Review web app; refuses non-loopback hosts (no login) |
| `export-index` | `dcs` reads the index, not the folders | Rebuilds `accepted_index.parquet` from ACCEPTED videos on disk |
| `annotate` | Label frames for the detector | Frame-labeling page (box, 5 keypoints, Listing) over `outputs/phase15/` |

Settings come from environment, then `.env`, then defaults. Main ones: `PDS_VIDEO_DIR`, `PDS_DB_PATH`,
`PDS_OUTPUT_DIR` (default `outputs`), `PDS_ACCEPTED_DIR`, `PDS_WORKERS`, `PDS_CONFIG` (a thresholds/profile YAML).

Without Docker (developers): `uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"`,
then `pytest` (synthetic data only) or `python backend/shared/scripts/changed_tests.py` (only the units you changed).

---

## 5. dcs: compound and dose classifier

Goal: tell which compound and dose a fish received from its behavior. Reads prepds output (`DCS_PROCESSED_DIR`,
default `outputs`) or the accepted folder (`DCS_ACCEPTED_DIR`) depending on `training.gold_source`.

```
featurize → training_table.parquet + schema → audit → train → <run>/ (report, metrics, model/) → predict
```

| Command | Why | How it works |
|---|---|---|
| `featurize` | Turn each fish folder into one feature row | Per fish: state shares, bouts, latencies, transitions, kinematics (from `segments.csv`, `frames.parquet`); optional depth and workbook (NTT) features. `Undetermined` is unknown time, not a behavior. Dropped fish are listed in the schema file with a reason, never silently |
| `audit` | Know if the data can support a result before training | Reads only the table and schema; writes `audit.md`: counts per compound/date, framing differences between dates, flagged fish, and the **G1 verdict**: GO if ≥3 non-vehicle compounds have ≥6 accepted fish over ≥2 dates, else WAIT |
| `train` | Fit and honestly score models | See below |
| `predict` | Score new fish | Loads `<run>/model/`, uses only the saved preprocessing, writes `predicted` and `p:<class>` per fish; checks the copy against saved reference predictions |
| `featurize --videos <dir>` | Features for fish that have no catalog or index | Reads per-video folders directly; marks fish unreviewed; table is for `predict` only |
| `synth --out <dir>` | Try everything without real data | Writes a synthetic gold dataset (placeholder names) with a planted signal |

**How `train` stays honest.**
- **Same folds for every model.** Preprocessing is fitted on each fold's training fish only (no leakage).
- **Scheme A (main score): date held out.** All fish from one recording date sit in the same fold, so the model is
  tested on days it never saw. **Scheme B: random folds** (reference). B − A is the **date effect**.
- **Baselines always run:** `majority` and `date_only` (predicts from the date alone), plus logistic regression,
  random forest, boosting and a PyTorch MLP.
- **`B_permuted`:** labels shuffled within each date; if the model still scores well, it learned the date.
- **Decision rule:** a model is *useful* only if its scheme-A balanced accuracy beats `majority` and `date_only` by
  more than the spread across repeats and beats its own `B_permuted`. Plain accuracy is never reported.
- **Why the date matters here:** camera setups fall into three epochs that line up with compounds, so a model can
  recognize the camera instead of the drug. On the current unreviewed data no model beats `date_only`.

```bash
python -m dcs train                          # both stages (compound, then dose within compound)
python -m dcs train --stage compound --models logreg,random_forest --no-ablations   # quick look
python -m dcs train --device cuda            # MLP on GPU; stops if torch sees none
```

Each run writes a new folder `outputs/dcs/training/<run_id>/`, never overwritten. **Read `report.md` first**; also
`metrics.csv`, `decisions.csv`, `predictions.csv`, `folds.csv`, `confusion_<stage>.png`, `config_used.yaml`,
`run_info.json` and `model/` (the best compound model).

**Settings.** `.env`: `DCS_PROCESSED_DIR`, `DCS_ACCEPTED_DIR`, `DCS_DB_PATH` (workbook, for the NTT columns),
`DCS_OUTPUT_DIR`, `DCS_TABLE`, `DCS_CONFIG`. Change training settings by writing only the changed keys in a YAML
file (misspelled keys are errors) and pointing `DCS_CONFIG` at it:

```yaml
training:
  gold_source: accepted          # use reviewed videos instead of the unreviewed output (default: processed)
  vehicle_compound: Veh          # the vehicle's name as the workbook writes it (also in my_training.yaml)
chat:
  model: qwen3:30b-a3b
```

**Try it (10 s, synthetic):**
```bash
python -m dcs synth --out /tmp/dcs_try
printf 'training:\n  gold_source: accepted\n' > /tmp/dcs_try/accepted.yaml
export DCS_ACCEPTED_DIR=/tmp/dcs_try/accepted DCS_DB_PATH=/tmp/dcs_try/synthetic_db.xlsx DCS_OUTPUT_DIR=/tmp/dcs_try/out
python -m dcs --config /tmp/dcs_try/accepted.yaml featurize && python -m dcs --config /tmp/dcs_try/accepted.yaml audit
python -m dcs --config /tmp/dcs_try/accepted.yaml train --models logreg --no-ablations
```

### Training on the GB10 (from a clone)

Copying files to the box may be blocked, so it rebuilds from a clone and trains on its own prepds output (read as
temporarily accepted, UNREVIEWED). It needs the raw videos and the workbook readable on the box.

```bash
git clone <repo url> fish-detection && cd fish-detection && git checkout <branch>
python3 -m venv .venv && source .venv/bin/activate && pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu130      # CUDA build for aarch64 first
pip install -e ".[dev,train]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
cp .env.example .env     # set PDS_VIDEO_DIR, PDS_DB_PATH, DCS_DB_PATH; keep PDS_OUTPUT_DIR=outputs, DCS_PROCESSED_DIR=outputs
tmux new -s fish         # so a dropped SSH session does not stop the run
python -m prepds catalog && python -m prepds run --workers <cores-2>
python -m dcs featurize && python -m dcs audit && python -m dcs train 2>&1 | tee train.log
```

Never put real names, dates or results into git.

---

## 6. Research chat

Researchers ask in English ("Did COMPOUND_A fish freeze more than vehicle?"). A local language model picks
**query tools**, the tools compute every number from the data, and the model writes the answer with their caveats.
**Nothing is trained**: a few hundred fish cannot teach a language model facts, and tool results can be checked.
Under each answer the app lists the lookups behind it.

```bash
# .env: FISHLAB_DCS_CHAT=1   DCS_CHAT_BASE_URL=http://127.0.0.1:11435/v1   FISHLAB_CHAT_URL=http://127.0.0.1:8010
docker compose --profile chat up -d model                 # model server (Ollama, GPU); data in .ollama/models
docker compose exec model ollama pull qwen3:30b-a3b       # once, needs internet (~18 GB)
python -m dcs ask "question" [--show-tools] [--run <run dir>]   # empty `ask` = a session
python -m dcs serve-chat                                  # JSON API on 127.0.0.1:8010 (loopback only, no login)
```

Name the model in the override YAML (`chat: model: ...`). Expect 15-60 s per answer; the app gives up after 240 s.
Data goes only to `DCS_CHAT_BASE_URL`; a non-local address is refused unless `chat.allow_remote: true`. A key goes
in `DCS_CHAT_API_KEY` in `.env`, never in a YAML file.

**Fix a bad answer** in `backend/dcs/config/chat_guidance.yaml` (applies to the next question, no restart). Sections:
`format` (layout), `rules` (how to read statistics; prefer a rule over an example), `glossary` (word → measure),
`examples` (question + wanted answer, style only). Use placeholder names and made-up numbers; the file is committed.

**Check a change:** `docker compose exec -T app env PYTHONPATH=/classifier/src python -m dcs.chat_eval qwen3:30b-a3b --show-answers > outputs/dcs/chat_eval_<model>.txt`
asks 27 fixed questions and scores right tool, right arguments, right number, invented numbers and latency. Read the failed answers yourself.

---

## 7. What to trust

- **Detection (fine-tuned model):** recall about 0.90-0.92 at the production setting on a small held-out set. It can
  report a fish on empty frames or follow the reflection. There is no accuracy metric for the track itself.
- **Controlled/Erratic/Freezing:** thresholds fitted on 5 reference groups (7 groups usable); extrapolated for
  other compounds. Agreement with the reference figures is a group-level match, not per-frame accuracy.
- **Listing/LORR, Dead:** not produced automatically; reviewer labels only. The Listing hints are high-recall and
  low-precision.
- **Surface Breach:** the current rule cannot fire (0.0% in both full runs); reviewer labels only.
- **Classifier:** no model yet beats the recording date on the unreviewed data. Behavior does separate vehicle
  from drug. Nothing is a drug claim until videos are accepted and the audit says GO.
- **Not ready for live streaming:** it needs accepted data and a measured tracker first.

---

## 8. When something fails

`dcs` errors print one line starting `Configuration error:` (exit 2). `./start.sh` names the failed step.

| Message contains | Fix |
|---|---|
| `DCS_PROCESSED_DIR is not set` | Add `DCS_PROCESSED_DIR=outputs` to `.env`, or set `gold_source: accepted` |
| `No trials_catalog.parquet` / `No processed/ folder` | Run `prepds catalog` and `prepds run`, or fix the path |
| `No accepted_index.parquet` | `prepds export-index`, or fix `DCS_ACCEPTED_DIR` |
| `The set mixes calibration profiles` | `featurize --profile <version>` |
| `fish have no date` | `prepds catalog`, then `prepds export-index` |
| `Undetermined found in an Accepted video` | Review that video again |
| `class(es) left for the ... stage` | Too few fish per class; wait for more, or lower `training.min_class_size` |
| `The training table has no fish` | Look at `dropped` in `training_table_schema.json` |
| `--device cuda, but PyTorch sees no CUDA GPU` | Install the CUDA torch build, or use `--device auto` |
| `Cannot reach the chat server` | `docker compose ps model` must be healthy; own server: start it or fix `DCS_CHAT_BASE_URL` |
| `chat.model is not available` | `docker compose exec model ollama pull <model>` |
| `No saved model in ...` | `--model` must be a run folder written by `dcs train` |
| `does not match the checksum saved with the model` | Model file changed or is from an older `dcs`: copy again or retrain |
| Page says "Could not connect to the API" | **Reload from backend**; if it stays, `docker compose logs -f app` |

The audit's **notes** are not errors; `audit.md` is always written. Note `vehicle ... is not in the table` means
set `training.vehicle_compound`.

---

## 9. Fish detector (optional)

Not needed to run anything above. It only improves tracking (undetected frames median 21.5% → 0.4% on the 328
videos; it did not improve speed measurement). Its files are not in git: `outputs/phase15/` is restricted. Retrain
from the repo root (needs `uv sync --extra ml` and a CUDA GPU):

```bash
python backend/prepds/scripts/phase15_prepare.py                 # split + 350 frames to label → outputs/phase15/{split,sample}.json, frames/
python backend/prepds/scripts/phase15_prefill.py                 # optional: OWLv2 box suggestions (slower labeling without them)
python -m prepds annotate                                        # a person labels box, 5 keypoints, Listing (needs ~100+ train, 30+ held-out)
python backend/prepds/scripts/phase15_finetune.py --run-name m3  # → outputs/phase15/models/m3/{model.pt,run.json,metrics.json}
python -m prepds run --tracker model:outputs/phase15/models/m3 --stride 5 --device cuda --dry-run
```

`prepare` writes the split once and never overwrites; run names are never reused. Fine-tune trains only on labeled
train frames and scores on labeled held-out frames; with none it scores on training frames and says so (a plumbing
check, not an evaluation). Details and results: Phase 15 in [progress.md](../backend/prepds/docs/progress.md).

**Data handling.** Videos, the workbook and reference images are restricted and never committed (`data/`, `outputs/`,
`accepted/` are git-ignored; only synthetic fixtures under `backend/shared/tests/fixtures/` are tracked).
