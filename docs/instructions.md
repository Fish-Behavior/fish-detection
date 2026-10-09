# Instructions: Preprocessing Dataset System

Turns each zebrafish drug-exposure trial video plus its trial-metadata row into a
reviewed, per-subject 5-state ethogram strip and machine-readable labels — the
labeled dataset a later, separate drug-classification project will train on. Full
specification: [`PRD.md`](PRD.md). Build status/progress:
[`progress.md`](progress.md).

## Run it: `./start.sh` and `./stop.sh`

Everyone runs the project the same way, from the repository root. Only Docker (with Compose)
is needed; no host Python or Node.
That later project lives in the same repository: see
[Classifier (`dcs`)](#classifier-dcs-compound-and-dose-from-behavior) at the end of this file.

## Setup

```bash
cp .env.example .env     # once: set PDS_VIDEO_DIR (synced video folder), PDS_DB_PATH (00_NTT_DataBase.xlsx),
                         # FISHLAB_HOST=127.0.0.1 and a free FISHLAB_PORT (the link you open);
                         # optional PDS_OUTPUT_DIR, PDS_ACCEPTED_DIR, PDS_WORKERS, PDS_CONFIG, DCS_* (see the file)
./start.sh               # build, prepare everything, start the app, print the local link
./stop.sh                # stop the app, remove the containers
```

If `.env`, a required path, `FISHLAB_HOST` or `FISHLAB_PORT` is not set up, `./start.sh` lists what is missing and starts nothing.
Nothing outside `.env` needs configuring. Data and `.env` stay on your machine and are never committed.

**What `./start.sh` does (it prints this list when it runs, and stops at the first failed step):**

| # | Step | Details | Average wait |
|---|---|---|---|
| 1 | Build | Docker image with the frontend and Python code | a few minutes the first time, cached after |
| 2 | Check | `prepds check-config`, then `catalog` (match trial rows to videos; unmatched go to `exceptions_report.json`) | seconds |
| 3 | Track | `prepds run`: track, label, consolidate and export every video not yet processed. Resumable; edited or accepted videos are never regenerated | ~35 s per 20-min video per worker; 328 videos took ~50 min on 22 workers (`PDS_WORKERS`). Reruns skip finished videos |
| 4 | Verify | every processed video has its output files | seconds |
| 5 | DCS | `featurize`, `check-config`, `audit`, model (trains only if none is saved), `predict` | training is slow and untimed; skipped if DCS is off |
| 6 | Chat | optional research chat, only if enabled in `.env` | seconds |
| 7 | Serve | start frontend + API, wait for health, print the link | seconds |

Failed and unfinished steps are shown in the terminal and in `<PDS_OUTPUT_DIR>/docker/startup.json`.
The accepted index is not exported by `./start.sh`.

**What `./stop.sh` does:** stops the app (waiting for in-flight saves) and removes the containers. Saved work stays on the host and is overwritten in place; no backups are made.

Then open the printed link and use the review UI below. Do not start `prepds` commands by hand next to a running
stack; two `run`s at once overwrite each other.

## Development without Docker (contributors only)

Not needed to use the project. For changing the code or running the tests:

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # add "[dev,ocr]" for optional pytesseract OCR support
cp .env.example .env
python -m prepds check-config   # confirms every configured path actually exists
pytest                          # synthetic data only
```

`opencv-python-headless` is used deliberately (not `opencv-python`): the project targets headless/WSL2-style
environments where the full wheel's `libGL.so.1` fails at import time.

CLI reference (the launcher runs the first four for you):

- `check-config` - show resolved paths/settings; exit 1 if a configured path is missing.
- `catalog` - match trial rows to local videos (compound folders are found at any depth under `PDS_VIDEO_DIR`).
- `run [--workers N] [--limit N] [--dry-run] [--force]` - one video per worker: track -> features -> label ->
  1 s consolidation -> review flag -> strip PNG -> export. **`--force` also overwrites edited/accepted work.** A failing
  video is reported in `run_report.json` and the run continues; exit 1 if any failed. Default workers `max(1, cpu_count - 2)`.
- `run --tracker model:<run dir> [--stride 5] [--device cuda]` - fine-tuned detector instead of the classical
  tracker (needs the `ml` extra and a GPU; single process, ~75 s/video). Use its own calibration profile
  (`--config`, e.g. `cal-2026-09-24-r4-model-m3.yaml`) and its own `PDS_OUTPUT_DIR` / `PDS_ACCEPTED_DIR`. Not part of `./start.sh`.
- `review [--host H] [--port P]` (defaults: `FISHLAB_HOST` / `FISHLAB_PORT` from `.env`, required) - review app without Docker; refuses non-loopback hosts (no authentication).
- `annotate [--port P]` (default `FISHLAB_ANNOTATE_PORT` from `.env`) - frame-labeling page (box, 5 keypoints, Listing tag) for the Phase 15 detector; needs
  `outputs/phase15/sample.json` (see `scripts/phase15_prepare.py`).
- `export-index` - rebuild `accepted_index.parquet` from ACCEPTED videos on disk. Run it after reviewing.

## Outputs per video

`outputs/processed/<sex>_<subject>/` holds `frames.parquet` (per-frame track, features, state, source),
`segments.csv` (exact run-length encoding of the frames), `strip.png` (one colour strip per video) and
`manifest.json` (status, provenance, review flags). Accepting copies these plus `provenance.json` to
`accepted/<video>/` and updates `accepted/accepted_index.parquet` (the gold dataset, directly consumable).

## Model tracker and Listing hints (Phase 15)

The fine-tuned detector (`scripts/phase15_finetune.py`, run `m3`, trained on ~330 human-labeled frames) replaces the
classical tracker in `outputs_r3/` (328 videos: undetected frames median 21.5% -> 0.4%, Undetermined median 13.7% ->
0.0%). Read [`progress.md`](progress.md) ("Review corrections") before relying on it: it is not human-reviewed, detection gain is
not proof of track accuracy, it can report a fish on empty frames, and its calibration is weakly identified.
Listing/LORR is still a manual label. `scripts/phase15_flag_listing.py` writes advisory `listing_flags.json` hints
(crop classifier, high recall, low precision) that the review app shows next to the video; they never change a state
or the accept rules.

## Review UI

Open the link `./start.sh` printed, pick a video, watch it with the timeline synced to playback, drag on the timeline to select a
time range, choose a state, **Stage relabel**, then **Save edits**. Accept is blocked while any `Undetermined`
remains (no override) and asks for confirmation when `Dead` is present. Reject clears edits and re-queues the video.
Keys on the timeline: arrows seek, `[` / `]` set the selection start/end at the playhead. Review flags (e.g. no
movement until the end of the video) are shown above the player.

- **Large video, controls below it:** play/pause, seek bar, clock and speed sit under the picture, so nothing covers the fish.
- **Markers on the video** (toggle each): track point, last-2-s path, and, for model-run output only, the detector box
  and 5 keypoints. Classical output has no box; the page says so. A dashed blue **waterline** can be set by hand
  ("Set waterline", then click the water surface; stored as `waterline.json`). It is advisory: no labeling rule reads it.
- **Why this label:** for the second under the playhead, the labeling rules in order with the measured value next to each
  threshold, the frame votes, and whether a reviewer changed the automatic label.
- **Time per state:** seconds, percent and bouts for all seven states.
- **Possible Listing hints:** advisory ranges (model run + `scripts/phase15_flag_listing.py`) with a "Go to" button.

## How labels are produced (and what to trust)

Tracking is classical OpenCV (KNN background subtraction; a motionless fish is absorbed and becomes undetected).
The label rule is a **speed band** on the ~1 s displacement: Freezing/Drift at or below a floor, Erratic at or above a
threshold, Controlled between; frames with no computable speed are `Undetermined`. Frame labels are then majority-
voted in 1 s bins. Precedence: undetected > Dead > Surface Breach > Listing (off) > Freezing > Erratic > Controlled.

- **Listing/LORR is not detected automatically** (body angle does not separate it on this side-view footage) and
  **Dead** cannot be told apart from prolonged LORR from video alone; both come from reviewers. Videos whose fish is
  still to the end carry a `terminal_no_action` review flag. The Dead rule did not fire on any of the 328 videos.
- **Surface Breach and Dead thresholds are uncalibrated placeholders** (20 px, 60 s): the reference figures show ~0%
  of both.
- **Undetermined is common where the tracker loses the fish:** median 14% per video, 29/328 videos above 50%.

## Calibration (run once; frozen profile `cal-2026-09-23-r2`)

The three speed thresholds were searched against proportions digitized from the two restricted reference figures
(`src/prepds/calibration/reference_targets.json`), minimizing the mean total-variation distance per compound/dose
group, scored on the 1 s-consolidated output, Listing excluded. In-sample distance 0.198 (placeholder thresholds:
0.418); **leave-one-group-out 0.246**, so expect roughly that gap on unseen groups. Reproduce with:

```bash
python scripts/build_track_cache.py            # track the calibration videos once (~40 s each)
python scripts/calibrate_thresholds.py         # search + per-group + leave-one-group-out (no file written)
```

Freeze results with `prepds.calibration.profile.write_calibration_profile` (profiles are immutable; add a new
revision) and select it via `PDS_CONFIG`. These scripts are not covered by the test suite.

**Known limitations (matches PRD section 10):**
- The references cover only 5 of the 17 compound families and 2 of up to 4 concentration levels; **only 7 groups
  could be calibrated** (Fentanyl 100 uM, MDMA 100 uM and one Veh cohort have no local videos). Auto labels for other
  compounds (e.g. the FD-2-* series, mostly Undetermined) are extrapolations: review them accordingly.
- Weakest groups: Fentanyl 30 uM (held-out 0.35), methylone 100 uM (0.33), MDMA 30 uM (0.29).
- Agreement with a reference figure is a group-level proportion match, not per-frame accuracy: there is no per-frame
  ground truth until reviewers accept videos.
- Frame rate is fixed by the recordings (~29.84 fps); a better tracker or pose model, not a higher frame rate, is the
  route to fewer Undetermined frames (see Phase 15 in [`progress.md`](progress.md)).

## Data handling

Videos, the trial workbook, and the two reference images are restricted research
data (PRD FR-018 / NFR-004) and are **never** committed — `data/`, `outputs/`,
`accepted/`, and the raw-data file extensions are gitignored (see `.gitignore`,
which also covers the raw-data extensions). Only synthetic fixtures under
`tests/fixtures/` are committed, via narrow negations in this package's own
`.gitignore` — verified with `git check-ignore -v <path>` before each fixture is
added, not assumed.

Implementation is tracked task-by-task in [`progress.md`](progress.md) (PRD §12, Phase 0-15).
## Status

Implementation is tracked task-by-task in [`progress.md`](progress.md),
mirroring the PRD §12 phase breakdown (Phase 0-15).

---

# Classifier (`dcs`): compound and dose from behavior

`dcs` reads what `prepds` wrote (labels per frame and per segment, one folder per fish) and learns to tell
which compound, and which dose, a fish received from its behavior; a research chat answers questions about it. Specification:
[`PRD.md`](PRD.md), Part II (cited in the code as "PRD §n" = `II.n`); open items and roadmap: Part III; status and
decisions (D-nnn): [`classifier_progress.md`](classifier_progress.md). It never imports `prepds` and never changes its files.

**Where it stands.** Everything in the plan except the optional 1D-CNN is built (U1-U20): the **training table**
(one row per fish), the **audit**, the **training set** and shared **folds**, per-fold **preprocessing**, five
**baselines** and the **MLP**, the **evaluation** with its leakage checks, **ablations** and vehicle normalization,
**per-compound dose models**, the **report**, the **saved model**, **predict**, and the **research chat**. Every unit
is tested on synthetic data. Until reviewers Accept videos, the real input is the **unreviewed** prepds output
(temporarily accepted), so no result is a claim about drugs yet.

```
prepds output ──► dcs featurize ──► training_table.parquet + schema ──► dcs train ──► <run>/report.md, metrics, predictions
   (+ workbook for NTT)                  │                              │   trainset -> folds -> preprocess per fold
                                         │                              │   -> models -> evaluation, ablations, dose models
                                         ├──► dcs audit ──► audit.md    └──► <run>/model/ ──► dcs predict (new fish)
                                         └──► dcs ask / serve-chat (research chat, local language model + query tools)
```

## Setup

Same virtual environment as `prepds` (see [Setup](#setup)). Add the `DCS_*` lines to `.env` (copy them from
`.env.example`); `dcs` ignores the `PDS_*` variables.

| Variable | What it points at | Needed by |
|---|---|---|
| `DCS_PROCESSED_DIR` | prepds output folder (`trials_catalog.parquet` + `processed/<video_id>/`), e.g. `outputs` | `featurize` while `training.gold_source` is `processed` (the default for now) |
| `DCS_ACCEPTED_DIR` | gold folder (`accepted_index.parquet` + one folder per Accepted fish) | `featurize` when `gold_source` is `accepted` |
| `DCS_DB_PATH` | the trial workbook (same file as `PDS_DB_PATH`) | `featurize`, for the 8 NTT columns; unset → NTT left empty, `has_ntt = 0` |
| `DCS_OUTPUT_DIR` | where `dcs` writes (default `outputs/dcs`, git-ignored) | everything |
| `DCS_TABLE` | the training table (default `<DCS_OUTPUT_DIR>/training_table.parquet`) | `featurize` writes it; `audit`, `train` and the chat read it |
| `DCS_CONFIG` | a YAML file overriding `config/default_training.yaml` | optional |
| `DCS_CHAT_BASE_URL` | the chat server's OpenAI-compatible address, e.g. Ollama on this machine (`http://127.0.0.1:11434/v1`); never in a YAML file | `ask`, `serve-chat` |
| `DCS_CHAT_API_KEY` | key for a hosted chat server; in your own `.env` (git-ignored) or the environment, never in `.env.example` or a YAML file; `check-config` shows it as `***` | `ask`, `serve-chat`, only for a hosted server |

Settings come from the environment first, then `.env`, then the defaults. Check them before touching data:

```bash
python -m dcs check-config      # every path with [ok] / [MISSING] / [not set], then every training setting; exit 1 if a path you set is missing
```

**Changing a setting.** Write only the keys you change into a YAML file and pass it with `--config` (before the
command) or `DCS_CONFIG`. A misspelled key or a wrong type is an error, never silently ignored.

```yaml
# my_training.yaml
training:
  gold_source: accepted          # read the reviewed gold folder instead of the unreviewed output
  duration_range_s: [590, 610]   # flag recordings outside this length (default: no check)
  vehicle_compound: CONTROL_X    # the vehicle's name as the workbook writes it (default VEHICLE, the synthetic name)
```

Set `vehicle_compound` to your lab's vehicle name before reading an audit of real data: the audit leaves the vehicle
out of the G1 stop rule and counts vehicle fish per date, and says so in its notes when the name is not in the table.

## Commands

| Command | What it does |
|---|---|
| `python -m dcs check-config` | Shows the paths and settings `dcs` would use. Reads no data. |
| `python -m dcs synth --out <new folder> [--seed N]` | Writes a **synthetic** gold dataset: `<out>/accepted/` (index + one folder per fish) and `<out>/synthetic_db.xlsx` (workbook). Placeholder names only (`COMPOUND_A`, `F_0001`). Use it to try `dcs` without real data. Refuses a folder that already has files. |
| `python -m dcs featurize [--profile <version>]` | Reads the gold source, computes the per-fish features and writes `training_table.parquet` and `training_table_schema.json` to `DCS_OUTPUT_DIR`. Prints how many fish were kept and dropped. `--profile` keeps one calibration profile when the set mixes several. |
| `python -m dcs audit` | Reads the two featurize files (nothing else), writes `audit.md` to `DCS_OUTPUT_DIR` and prints the G1 verdict and the notes. Trains nothing and changes no file it reads ([The audit](#the-audit)). |
| `python -m dcs featurize --videos <dir> [--out <table>]` | Rows for fish read from their folders alone (no index or catalog), for `predict` ([Predicting new fish](#predicting-new-fish-dcs-predict)) |
| `python -m dcs train [--stage] [--models] [--seed] [--repeats] [--device] [--no-ablations]` | Every model on shared folds, ablations, dose models, the report and the saved model in a new run folder ([Training](#training-dcs-train)) |
| `python -m dcs predict --model <run> --input <table> [--out] [--device]` | Class probabilities per fish from a saved model; checks the reference predictions |
| `python -m dcs ask ["question"] [--run] [--show-tools]` | The research chat in the terminal ([Research chat](#research-chat-dcs-ask-dcs-serve-chat)) |
| `python -m dcs serve-chat [--host 127.0.0.1] [--port 8010] [--run]` | The research chat as a loopback JSON API for a web frontend |

Global options go **before** the command: `python -m dcs --config my.yaml featurize`, `--env-file other.env`.

### Try it on synthetic data (about 10 seconds)

```bash
python -m dcs synth --out /tmp/dcs_try
printf 'training:\n  gold_source: accepted\n' > /tmp/dcs_try/accepted.yaml     # synth writes the accepted layout
DCS_ACCEPTED_DIR=/tmp/dcs_try/accepted DCS_DB_PATH=/tmp/dcs_try/synthetic_db.xlsx DCS_OUTPUT_DIR=/tmp/dcs_try/out \
  python -m dcs --config /tmp/dcs_try/accepted.yaml featurize
# Featurized 48 fish (48 Accepted, the rest UNREVIEWED; profile cal-synthetic), 0 dropped
DCS_OUTPUT_DIR=/tmp/dcs_try/out python -m dcs --config /tmp/dcs_try/accepted.yaml audit
# Audited 48 fish (Accepted gold folder): G1 stop rule GO      (report: /tmp/dcs_try/out/audit.md)
DCS_OUTPUT_DIR=/tmp/dcs_try/out python -m dcs --config /tmp/dcs_try/accepted.yaml train --models logreg --no-ablations
# Run folder: /tmp/dcs_try/out/training/<run_id>         (about 10 s; all models and ablations: a few minutes)
```

### Run it on the real data

```bash
# .env: DCS_PROCESSED_DIR=outputs  and  DCS_DB_PATH=<same file as PDS_DB_PATH>
python -m dcs check-config
python -m dcs featurize          # Featurized <n> fish (UNREVIEWED, profile cal-...), <m> dropped
python -m dcs --config my_training.yaml audit     # my_training.yaml sets vehicle_compound; read outputs/dcs/audit.md
```

Fish that the catalog matched to a video but `prepds run` never processed are listed as dropped (`missing_file`),
not silently skipped.

## What each part does

| Unit | Tool | What it does | How to use it |
|---|---|---|---|
| U1 Config | `check-config`, `config/default_training.yaml`, `dcs.config.load_settings()` | Loads paths and training settings with the precedence above; rejects unknown keys, wrong types and out-of-range values with a message naming the key | Run `check-config` after editing `.env` or an override YAML |
| U2 Synthetic data | `synth`, `dcs.synthetic.make_gold_dataset(out, SynthConfig(...))` | Writes fish that follow the real prepds file formats, with a planted compound signal and a date effect. `SynthConfig` has one switch per edge case (e.g. `low_detected_fish=2`, `undetermined=True`, `missing_file="frames"`), and `result.targets` names the fish each switch hit | CLI for trying the tools; the Python form for tests |
| U3 Gold reader | `dcs.gold.read_gold(settings)` (inside `featurize`) | Reads either source and checks the prepds file contract. **Drops** a fish (listed with a reason) when a file is missing or unreadable, or its status is REJECTED / NOT_PROCESSED. **Stops** with one clear message when the data breaks the contract: mixed calibration profiles, a fish without a date, a duplicated fish, `Undetermined` in an Accepted video, a missing column or wrong type, tracker evidence that disagrees with the profile | Nothing to call by hand; its messages tell you what to fix (table below) |
| U4 Workbook | `dcs.workbook.read_ntt(path, videos)` (inside `featurize`) | Reads the 8 NTT columns per fish from the workbook (`<Sex>_<Subject:04d>`). `has_ntt = 1` when all 8 are filled. A `-` in a half's cell means the fish never entered that half: distance 0, speed empty (still `has_ntt = 1`). Compounds are compared with the gold set after trimming, collapsing spaces and ignoring case. Stops if the workbook disagrees with the gold set on compound or date, or has two different rows for one fish | Set `DCS_DB_PATH`; leave it unset to train without NTT |
| U5 Featurize | `featurize`, `dcs.featurize.featurize(gold, workbook, settings.training)` | Builds the training table (next section) | `python -m dcs featurize` |
| U6 Training set | `dcs.trainset.load_table(path)`, `build_trainset(table, schema, settings.training, stage)` | Cleans the labels, drops small classes, rare-state, constant and switched-off features, refuses forbidden columns; returns what one stage trains on and lists everything dropped ([The training set](#the-training-set)) | No command of its own: `audit` and `train` will call it. Call it from Python to see what a setting change does |
| U7 Folds | `dcs.folds.make_folds(y, groups, ids, settings.training)` | Splits a training set into scheme A (whole dates held out) and scheme B (random) folds, `repeats` times, and pins classes seen on one date to training in scheme A ([The folds](#the-folds)) | No command of its own: `train` will call it once and give the same folds to every model |
| U8 Audit | `audit`, `dcs.audit.build_audit(table, schema, settings.training)`, `render(result)` | Runs the class filter, training set and folds of both stages as `train` will, and reports what the data holds, what was dropped and why, and what could let a model recognize the day, the labeling or the camera instead of the compound ([The audit](#the-audit)) | `python -m dcs audit` after every `featurize`; read `audit.md` at gate G1 |

## The training table

One row per fish, in the gold set's order. Columns, by group:

| Group | Columns | From | Notes |
|---|---|---|---|
| Identity, labels, folds | `video_id`, `subject_id`; `compound`, `concentration_mM`; `date` | gold set | Labels are as written; the training set cleans them. `date` only groups the folds |
| States | per state: `state_<s>_share`, `_bouts`, `_mean_bout_s`, `_latency_s` (6 states) | `segments.csv` | Share of the **known** time. A state never shown: 0 bouts, mean bout 0, latency = recording length |
| Transitions | `trans_<a>_to_<b>` (30 ordered pairs) | `segments.csv` | Counts of a state directly followed by another |
| Kinematics | `velocity_mean`, `_median`, `_cv`, `abs_acceleration_mean`, `abs_angular_velocity_mean`, `meander_mean`, `immobile_share`, `detected_share` | `frames.parquet` | Detected frames only, minus the first frame of each detected run (speed, turning) or first two (acceleration, meander), where prepds writes 0 for lack of history; so patchy tracking does not slow a fish down. Pixels and seconds |
| Depth | `depth_mean`, `depth_min`, `depth_p01` … `depth_p99` | `frames.parquet` | Pixel row from the frame top; **off for training by default** (camera framing can differ by date) |
| NTT | `tdm_*`, `velocity_*`, `time_top_s`, `time_bottom_s`, `has_ntt` | workbook | Empty when absent; filled per fold later (U9) |
| Demographics | `sex`, `strain`, `age` | gold set | Off by default; an ablation only |
| Meta (never features) | `undetermined_share`, `manual_share`, `flag_low_detected`, `flag_odd_duration`, `recording_s`, review fields, fps, profile | both | For the audit. Flags mark a fish, never drop it |

**`Undetermined`** (time the tracker could not label; every unreviewed video has some) is treated as unknown time,
not as a behavior: it gets no features, state shares are computed over the remaining time, and a transition across
it is not counted. Its amount is kept as `undetermined_share`.

**Dropped fish** (listed in the schema file with a reason, never silent): files missing or unreadable, review status,
other calibration profile, no detected frame at all (`no_detected_frames`), or no known state time
(`no_known_state_time`).

`training_table_schema.json` describes every column: `role` (`feature`, `label`, `group`, `id`, `meta`), `source`,
`kind` (`count` and `duration` get `log1p` later), `feature_group` and, for state features, `states`. It also records
the source, profile, tracker, frame-rate check and every dropped fish, so these two files are all that has to be
copied to the training machine. To look at them:

```python
import json, pandas as pd
table = pd.read_parquet("outputs/dcs/training_table.parquet")
schema = json.load(open("outputs/dcs/training_table_schema.json"))
features = [c["name"] for c in schema["columns"] if c["role"] == "feature"]
pd.DataFrame(schema["dropped"]).groupby("reason").size()      # why fish were left out
```

## The training set

`build_trainset` takes the two featurize files and one **stage**: `compound` (Stage 1) or `dose` (Stage 2, dose within
compound). It changes no file. In order:

1. **Labels.** Compound: spaces trimmed and collapsed, upper case, so `compound_a ` and `COMPOUND_A` are one class.
   Dose: the written text without spaces, so `0.03 + 0.01` and `0.03+0.01` match (combination doses are not numbers).
   A dose-stage label reads `COMPOUND_A @ 0.1`.
2. **Class filter.** Classes with fewer than `training.min_class_size` fish (default 6) are dropped. Classes of exactly
   that size stay but are flagged **very small**. In the dose stage, a compound left with one dose (vehicle always) is
   dropped too (`single_dose`).
3. **Feature groups.** States, transitions and kinematics are always used. `depth` only with `use_depth` (off: camera
   framing), `ntt` only with `use_ntt` (off by default: it failed its keep rule and follows the camera epoch, D-076), `demographics` (`sex`, `strain`, `age`) only with `use_demographics` (off).
4. **Forbidden columns.** Labels, ids, `date`, paths, review and edit fields, manual-frame share, profile and pipeline
   versions, protocol fields (`agent_exposure_min`, `ntt_min`, `uv_min`, `h2o_*`, tissue), recording length, fps and
   resolution never become features. If the schema file marks one as a feature, the run stops.
5. **Rare states.** A state shown (at least one bout) by fewer than `training.min_state_fish` kept fish (default 10)
   loses all its features, transitions to or from it included.
6. **Nothing to learn from.** A feature that is constant or empty over the kept fish is dropped (on the real NTT data
   `has_ntt` is always 1, so it goes).

Missing values are **not** filled here; that happens per fold (U9). The result (`TrainSet`) holds `X` (features),
`y` (label), `groups` (date) and `ids` (`video_id`) on the same rows, `kinds` (which features get `log1p` later),
`classes` (fish per kept class), `very_small`, and the drop lists `dropped_classes` (label, fish, reason),
`dropped_states` and `dropped_features` (name, reason). To see what a setting does:

```python
from dcs.config import load_settings
from dcs.trainset import build_trainset, load_table

settings = load_settings()                              # or load_settings(config_file="my.yaml")
table, schema = load_table(settings.paths.table)
result = build_trainset(table, schema, settings.training, "compound")   # or "dose"
result.classes, result.very_small                       # what is kept
result.dropped_classes, result.dropped_states           # what is left out, and why
```

On the real unreviewed set (143 fish when this unit was written; default settings): compound stage keeps every fish; Listing/LORR, Surface Breach
and Dead are shown by too few fish, so their 36 features go; the dose stage keeps 104 fish in 12 classes.

## The folds

`make_folds` takes a training set's `y`, `groups` (date) and `ids` and the training settings. It changes no file and
never stops a run; what it could not do goes into `notes`. Two schemes, both repeated `training.repeats` times
(default 5) with `training.folds` folds (default 5):

- **Scheme A, date held out** (the main score). All fish of one date sit in the same fold, so every test fish comes
  from a day the model never saw. Classes stay as balanced across folds as whole dates allow. A class on two dates
  gets its dates in two different folds, so it is never missing from the training side.
- **Scheme B, random** (the reference). Fish are split at random, keeping class shares equal in every fold. Same-date
  fish land on both sides, so B is expected to score higher than A; the gap shows how much the date helps.

Rules, each with a note in `notes` when it applies:

| Situation | What happens |
|---|---|
| A class seen on **one date only** | Cannot be held out. In scheme A its fish get fold `-1` (always training, never scored) and the class is listed in `date_confounded`. Scheme B scores it as usual |
| Fewer dates than folds (pinned classes' dates not counted) | Scheme A uses one fold per date (`k["A"]` says how many) |
| Fewer than 2 dates left to hold out | Scheme A is skipped (`k["A"] = 0`, no A rows). This is normal for the **dose** stage: doses of one compound never share a date, so every dose class is pinned |
| A class's dates all land in one scheme-A fold | Possible only when dates are too entangled to split; the note names the repeat and the class, and that fold trains without it |
| The largest class has fewer fish than folds | Scheme B uses fewer folds |

The result (`Folds`) holds `table` (one row per fish, scheme and repeat: `video_id`, `label`, `date`, `scheme`,
`repeat`, `fold`; this is what `train` will save as `folds.csv`), `k` (folds per scheme), `date_confounded` and
`notes`. The same `training.seed` always gives the same folds; `repeat_seed(seed, repeat)` gives each repeat's random state, which the models of that repeat use too. To look at them:

```python
from dcs.folds import make_folds
folds = make_folds(result.y, result.groups, result.ids, settings.training)   # result from build_trainset above
folds.k, folds.date_confounded, folds.notes
folds.table.query("scheme == 'A' and repeat == 0").groupby("fold")["label"].value_counts()   # who is tested where
```

sklearn may print `The least populated class in y has only N members, which is less than n_splits`: a class smaller
than the number of folds is simply absent from some test folds. It is a warning, not an error.

On the real unreviewed set (default settings): the compound stage gets 5 folds in both schemes with no notes; the dose
stage pins all 12 dose classes (each on one date), so only scheme B runs there.

## The audit

`python -m dcs audit` reads only `training_table.parquet` and its schema file, so it also runs on the training machine.
It writes `<DCS_OUTPUT_DIR>/audit.md` and prints the G1 verdict and the notes. It never stops because the data is thin:
a stage that cannot be built (for example, too few classes) becomes a note.

`audit.md` starts with the **facts**, then the **notes** (things to act on), then one table per section:

| Section | What it shows | What to look for |
|---|---|---|
| Facts: source, profile, tracker | Accepted or UNREVIEWED; calibration profile; tracker and how many fish its evidence was checked for; frame-rate range; resolution (prepds does not record it) | UNREVIEWED means no result is a drug claim; fps not uniform (EC-26) means pixel features are not comparable |
| Facts: PRD §2.3 statistics | Fish, Accepted fish, dates, fish per date, compounds, vehicle fish and dates, compound+dose classes (on one date, below `min_class_size`), compounds on one date, dates where one compound has 2+ doses, fish with NTT, video length range, EC-11 flag counts | Dose classes on one date and "2+ doses" = 0 mean dose is confounded with date (PRD §6.5) |
| Facts: per stage | Classes kept, folds per scheme, date-confounded classes, share of fish on dates holding 2+ classes (D-016: the within-date permutation test only learns from those) | A low share weakens the permutation test |
| Facts: G1 stop rule | **GO** when at least 3 compounds besides vehicle have `min_class_size` (6) Accepted fish spread over 2+ dates, else **WAIT** (PRD §10). On unreviewed data it also gives the verdict counting every fish (Q17) | The go / wait decision at G1 |
| Facts: framing | Whether camera framing differs between dates (EC-31), how many framing setups, dates without depth data | See Camera framing |
| Filter steps | Fish dropped before the table (by reason), in the table, then per stage: dropped by the class filter (by reason), kept, scored in scheme A | Where fish are lost |
| Compounds | Per compound: fish, Accepted fish, dates, Accepted dates (a compound with 0 Accepted fish or one date shows here, EC-24), doses, mean manual-frame and Undetermined share, flag counts, `agent_exposure_min` values | Manual share differing by compound = labeling effort differs (reported, never a feature); exposure differing by compound = protocol tracks the label |
| Classes: compound / dose | Per class: fish, Accepted, dates, status (`kept`, `very small`, or the drop reason), scheme A (`scored`, `pinned` = one date only, `skipped`) | Which classes the main score can say anything about |
| Dates, Compound by date | Fish, Accepted, compounds and vehicle fish per date; the compound × date count table | Compounds concentrated on few dates |
| Missing values | Columns with empty cells and how many | NTT gaps are expected (filled per fold later); anything else is a surprise |
| Flagged fish | Fish with a low detected share or an odd recording length (kept, EC-11) | Many flags on one compound or date |
| States | Fish showing each state; in which stage its features were dropped (`min_state_fish`) | Rare states |
| Camera framing | Per date: where the typical fish's top and bottom sit in the frame (median of each fish's 1st and 99th depth percentile), the height between them, its `setup`, and the extremes (`top_min`, `bottom_max`). Dates whose top, bottom and height agree within 10 % of the median height, directly or through other dates, share a setup. A date without depth data is listed with 0 fish and no setup, and a note says it was not checked | Framing differs when any two dates are further apart than that (several setups, or one setup that drifts). Medians, so one fish at the surface cannot move a date. Then keep `use_depth` off, and treat pixel speeds as date-dependent |

On the real unreviewed set (328 fish, 41 dates, `vehicle_compound` set): G1 stop rule WAIT on Accepted fish (none yet),
GO when unreviewed fish count; frame rates uniform; tracker checked for every fish; **framing differs between dates**:
the recordings fall into three camera epochs (fish vertical range about 71 px in epoch 1, about 31 px in epoch 2,
about 34 px in epoch 3; 18 framing setups, none spanning two epochs, one stable setup through most of epoch 3), and most
compounds were recorded in one epoch only. Depth stays off, and
pixel speeds, state shares and `has_ntt` track the epoch (about 93 / 71 / 39 % NTT coverage), so scheme A cannot rule out
that a model recognizes the camera instead of the compound. Handle this before models are designed (per-epoch
normalization, FR-9).

## Preprocessing (per fold)

Models never see the raw feature matrix. Inside every fold, `fit_preprocess(X_train, kinds)` learns three things from
the **training rows of that fold only**, and `transform` applies them to both sides:

1. **`log1p`** on features of kind `count` or `duration` (bouts, latencies, transitions, NTT times), which are heavy-tailed.
2. **Fill gaps** with the fold's median (after `log1p`). This is where a fish without NTT gets its values (EC-2,
   `has_ntt` stays 0 so the model can tell), and where a fish that never entered the NTT top half gets a top-half velocity
   (its time in that half, 0, already says it never went there; D-051). A column empty in the whole fold becomes 0.
3. **Standardize** with the fold's mean and standard deviation. A column constant in the fold gets a spread of 1, so
   nothing divides by zero.

Test rows never move these numbers (EC-9): a fold with extreme test fish is scaled by the training fish's spread.
The fitted numbers go to JSON and back unchanged (`as_json`, `Preprocess.from_json`); the model folder will store
them as `preprocess.json` (U16). Category features (`sex`, `strain` with `use_demographics: true`) become one 0/1
column per value seen in the training fold (`sex=F`, `sex=M`); a value the fold never saw, or a gap, is all 0 (D-062).
To look at one fold:

```python
from dcs.preprocess import fit_preprocess
train = result.X.iloc[:100]                              # result from build_trainset above
fitted = fit_preprocess(train, result.kinds)
fitted.log1p, fitted.fill["velocity_mean"]               # what was logged; the fill value of one feature
fitted.transform(result.X.iloc[100:])                    # the other fish, scaled with the training fish's numbers
```

## Baseline models

`make_model(name, seed)` returns a fresh model; all of them work the same way: `fit(X, y, dates)`, then
`predict_proba(X, dates)` gives one probability per class, in the order of `model.classes_` (sorted labels). `X` is
the preprocessed matrix; only `date_only` reads `dates`. The five (PRD §6.2, `BASELINES`):

| Name | What it is | Why it is there |
|---|---|---|
| `majority` | Always the class shares of the training fish (the largest class wins) | The floor: a model must beat guessing the biggest class |
| `date_only` | The label mix of the test fish's date in training (scheme B); for a date never seen in training (scheme A), the nearest training date, the earlier one on a tie; a date that is not an ISO date gets the overall class shares | The leakage ceiling: how much the day alone explains, with no behavior at all (D-011) |
| `logreg` | Logistic regression, L2, classes weighted by their size | Simple linear model |
| `random_forest` | 500 trees, classes balanced, all CPU cores | Non-linear, robust on small tables |
| `hist_gb` | Histogram gradient boosting (scikit-learn), classes balanced | Strong tabular default, no extra dependency |

The settings are fixed (D-053); there is no tuning, because a few hundred fish cannot support a tuning loop on top of
the date-held-out folds. The same seed gives identical probabilities. The MLP (next section) has the same interface.
To try one:

```python
from dcs.models import make_model
model = make_model("logreg", seed=0).fit(X_train, y_train, dates_train)   # X from fitted.transform(...)
model.classes_, model.predict_proba(X_test, dates_test)
```

## The MLP (PyTorch)

`make_model("mlp", seed, settings.training["mlp"], device)` builds the neural network of PRD §6.3 from `training.mlp`:

| Setting | Default | Meaning |
|---|---|---|
| `hidden_sizes` | `[128, 64]` | Two hidden layers, each Linear -> ReLU -> Dropout |
| `dropout` | 0.3 | Share of units switched off in each training step |
| `learning_rate`, `weight_decay` | 0.001, 0.01 | AdamW optimizer |
| `batch_size` | 32 | Fish per optimizer step |
| `max_epochs`, `patience` | 300, 20 | Stop after `patience` epochs without a better validation loss; keep the best epoch's weights |
| `inner_val_fraction` | 0.2 | Share of each training fold held back to decide when to stop (stratified when every class has 2+ fish) |
| `seeds` | 5 | Networks per fold, each with its own random start; their probabilities are averaged |

The loss weights classes by their size, like the baselines' `class_weight="balanced"`. Inside cross-validation the
MLP sees the same preprocessed matrix as every other model, so it gets no extra information. On the same CPU, the
same seed gives the same probabilities (EC-14); on a GPU the last digits may differ, and the report says so.

PyTorch is optional (`pip install -e ".[train]"`). Without it, `train` runs the baselines and prints
`mlp: skipped, PyTorch is not installed` (EC-16). `dcs train --device` picks where the MLP trains: `auto` (default; the
GPU when PyTorch sees one, else the CPU), `cpu`, or `cuda`. `cuda` without a GPU stops with a message (EC-15). Cost on
the real unreviewed set: about 1.3 s per fold for 5 seeds on a laptop CPU, so about 2 min per stage.

## Evaluation

`evaluate(trainset, folds, models, training)` cross-validates every model on the **same folds** (`make_folds`), so
models are compared on identical splits. `majority` and `date_only` are always added first, because the decision rule
compares every model with them. For each scheme and repeat:

1. **Per fold:** preprocessing is fitted on that fold's training fish (EC-9) and shared by all models of the fold; each
   model is trained and gives probabilities for the test fish. Pinned fish (scheme A fold `-1`) train in every fold
   and are never scored. A class missing from one training fold gets probability 0 there.
2. **Within-date permutation** (scheme B only, PRD §6.6): the labels are shuffled among the fish of each date and
   scheme B is run again (`B_permuted`, same folds, a new shuffle per repeat). A date holding one class cannot be
   shuffled, which is why the audit gives the share of fish on dates with 2+ classes (D-016).
3. **Metrics** (PRD §6.7) on the **pooled** out-of-fold predictions of the repeat (D-017): `balanced_accuracy`,
   `macro_f1`, `log_loss`, `top3_accuracy`. Class averages run over the classes present among the scored fish. Plain
   accuracy is never reported.

The result (`Evaluation`) holds `predictions` (one row per model, scheme, repeat and scored fish: `fold`, `video_id`,
`date`, `label`, `predicted`, and `p:<class>` per class; in `B_permuted` rows `label` is the shuffled label the model
was trained and scored on), `metrics` (one row per repeat), `summary` (mean and `std` across repeats, the "spread"),
`per_class` (precision, recall, F1, fish; pooled over repeats), `decision` and `vehicle`:

| Output | Rule |
|---|---|
| `decision` | A model is **useful** when its scheme-A balanced accuracy beats `majority` and `date_only` by more than the spread (the larger of the two spreads compared, D-055) and beats its own `B_permuted` score. Otherwise **not useful**, with the reason. **undecided** when scheme A was skipped (every class on one date); the two references are marked `baseline` |
| `vehicle` | Compound stage only: vehicle against any drug, from the compound model's scheme-A probabilities (drug score = 1 − P(vehicle)): AUROC and balanced accuracy, mean and spread across repeats. Empty when `training.vehicle_compound` is not among the kept classes |

`log=print` prints one line per scheme and repeat with its time, so a long run shows progress. To run it by hand:

```python
from dcs.evaluate import evaluate
from dcs.models import BASELINES
ev = evaluate(result, folds, BASELINES, settings.training, log=print)   # result, folds from the sections above
ev.summary["balanced_accuracy"]                                          # mean and spread per model and scheme
ev.decision                                                              # useful / not useful, and why
```

How to read it: scheme A is the main score. **Date effect** = scheme B − scheme A. When `date_only` beats the
behavior models in scheme A, the dates alone (experiment campaigns) tell the compounds apart better than the
behavior does. When `B_permuted` is as high as `B`, the model learned the date, not the compound. On the real
unreviewed set (compound stage, default settings, dry run with nothing written) every behavior model was **not useful**:
`date_only` scored well above them in scheme A. That is a statement about the unreviewed data and the camera epochs
(see Camera framing), not a final result. Runtime there: about 2 min per stage on a 10-core laptop, most of it the
forest and boosting in scheme B and `B_permuted`.

## Ablations (compound stage)

After the compound stage, `dcs train` re-runs every model with one change at a time, on **the main run's folds** (PRD §6.8; a switch that changes which fish are kept gets new folds, and its row in the report says so;
`--no-ablations` skips them, about 5 times faster):

| Ablation | Change | Question it answers |
|---|---|---|
| `use_ntt=true` | NTT features on (or off, when your settings have them on) | Does the novel-tank test add anything? The **NTT keep rule** in the report: keep NTT only if scheme A is better with it by more than the spread |
| `use_demographics=true` | `sex`, `strain`, `age` on | Do fish characteristics help, or do they track the date (strains bought per campaign)? |
| `use_depth=true` | Depth features on | Depth in pixels follows the camera framing (D-015, EC-31): a gain in B but not in A means the model reads the camera |
| `non-vehicle, raw` | Vehicle fish left out | The baseline of the FR-9 pair |
| `non-vehicle, vehicle-normalized` | Vehicle fish left out; every feature (after `log1p`) minus the median of the reference vehicle fish | FR-9: does the model still separate compounds once the day's (or the camera's) vehicle level is removed? Compared with the raw row on the **same classes and folds** (D-008) |

**The vehicle reference (`training.camera_epochs`, D-061).** By default (`null`) a fish's reference is the vehicle fish
of its own date when that date has 2 or more, else of its camera framing setup (the audit's EC-31 groups), else all
vehicle fish; the report says how many dates used each. With 1 to 3 vehicle fish per date this reference is noisy,
and the noise is the same for every fish of the date: it stamps each date with a fingerprint. On the real unreviewed set
that made things worse (scheme A down, scheme B up to its permuted score: pure date recognition). Setting the camera
epochs uses the much larger vehicle groups of each epoch instead, which removed part of the camera effect there
(scheme A up by more than the spread). Read the epoch start dates off `audit.md` ("Camera framing": where the tank top
and height jump) and put them in your override file:

```yaml
training:
  vehicle_compound: <vehicle name>
  camera_epochs: [<first day of epoch 2>, <first day of epoch 3>]    # YYYY-MM-DD, increasing
```

The ablation rows go to `metrics.csv` (column `ablation`; `main` for the main run) and to a section of `report.md`
(scheme A and B per model and ablation, and `delta A`, the change in scheme A). An ablation that cannot be built (for
example a group whose features are all constant) is listed with the reason. To run them by hand:

```python
from dcs.ablations import run_ablations
from dcs.report import StageResult
main = StageResult(result, folds, ev)                       # compound stage, from the sections above
for a in run_ablations(table, schema, settings.training, main, ["logreg"], "cpu", {}, print):
    print(a.name, a.note or a.result.evaluation.summary.loc[("logreg", "A"), "balanced_accuracy"].to_dict())
```

## Stage 2: dose within compound

With `stage: both` or `dose`, `dcs train` builds the dose training set (classes `<compound> @ <dose>`; small doses
and compounds left with one dose are dropped, which removes vehicle), then trains **one model set per compound** with
two or more doses (PRD FR-7). Each compound is its own stage in the run folder, named `dose <compound>`: its own
folds, rows in `folds.csv`, `metrics.csv`, `predictions.csv`, a report section and `confusion_dose_<compound>.png`.

Doses of one compound are rarely recorded on the same date (PRD §6.5), so scheme A is usually skipped and a dose
model can win scheme B by recognizing the date. Every dose section therefore opens with the **Exploratory** caveat,
and the report adds a **Stage 2 summary** table: per compound and model, scheme B next to `B permuted`, `date_only`,
and **B vehicle-normalized**, the same models on the same folds after the vehicle reference is subtracted (FR-9, same
reference rules as the ablations, `camera_epochs` included). Only that column can say anything about dose: if a model
still separates doses there, the day's vehicle level does not explain it. In `metrics.csv` those rows have
`ablation = non-vehicle, vehicle-normalized`. When no compound has two eligible doses, the report says why and the
compound stage still runs.

## Training (`dcs train`)

```bash
python -m dcs train                                   # both stages, every model in training.models
python -m dcs train --stage compound                  # Stage 1 only (compound, dose or both)
python -m dcs train --models logreg,random_forest     # subset; majority and date_only are always added
python -m dcs train --seed 1 --repeats 10             # other seed / more repeats for this run only
python -m dcs train --device cuda                     # MLP on the GPU (stops if PyTorch sees none)
python -m dcs train --no-ablations                    # main run only (quick look)
```

It reads only `DCS_TABLE` (the training table and its schema file, both from `featurize`) and, per stage, builds the
training set, the shared folds and the evaluation above. It prints one line per scheme and repeat, so a long run shows
progress (`tee train.log` keeps it). The command-line values are checked by the same rules as the settings file; they
apply to this run only and are saved in `config_used.yaml`. `--device auto|cpu|cuda` (default `auto`) is where the
MLP trains; the device used is in `run_info.json`.

When `stage` is `both` and one stage cannot be built (for example the dose stage with too few fish per dose), the
other still runs and the report says why the first is missing. When no stage can be built, nothing is written and
the command stops with the reason (exit code 2).

Each run gets its own folder `<DCS_OUTPUT_DIR>/training/<run_id>/` (`run_id` = UTC start time + short git commit,
`-2`, `-3` ... when two runs start in the same second). Nothing in it is ever overwritten:

| File | What it holds |
|---|---|
| `report.md` | **Read this first.** Source (Accepted, or UNREVIEWED = temporarily accepted), caveats, how to read the scores, then per stage: classes kept and dropped, folds, the score table (every model on one row with `majority`, `date_only`, `B permuted` and the date effect B − A beside it), the decision and its reason, vehicle against drug, per-class scores and the confusion matrix of the best model |
| `metrics.csv` | One row per stage, model, scheme (`A`, `B`, `B_permuted`) and repeat: `fish`, `balanced_accuracy`, `macro_f1`, `log_loss`, `top3_accuracy` |
| `decisions.csv` | Per stage and model: the decision rule's verdict and its reason (what the chat and a frontend read) |
| `predictions.csv` | Out-of-fold predictions: stage, model, scheme, repeat, fold, `video_id`, date, label, predicted, `p:<class>` per class |
| `folds.csv` | The folds every model used (stage, `video_id`, label, date, scheme, repeat, fold; `-1` = pinned) |
| `confusion_<stage>.png` | Best model (highest scheme-A balanced accuracy), share of each true class predicted as each class, all repeats |
| `audit.md` | `dcs audit` of the same table |
| `config_used.yaml` | Every `training:` setting of this run, command-line values included |
| `run_info.json` | Run id, git commit and whether tracked files had uncommitted changes, command, seed, folds, repeats, stages, models (and skipped ones), source, table path, versions (Python, scikit-learn, numpy, pandas, torch), hardware, start time, seconds |

| `model/` | The final model (next section) |

Runtime on the real unreviewed set: about 4 to 5 min for both stages and the five baselines on a 10-core laptop.

## The saved model (`model/`)

After the evaluation, `dcs train` takes the **best compound model** (highest scheme-A balanced accuracy; a reference
baseline only when nothing else ran), refits it, preprocessing included, on **all** fish of the compound stage, and
saves it in `<run>/model/` (PRD §6.9, §7.4). The dose models are exploratory and are not saved.

| File | What it holds |
|---|---|
| `model_info.json` | Model name, stage, the decision-rule verdict it earned in cross-validation, fish, features, seed, device, versions (Python, scikit-learn, numpy, pandas, torch), and the SHA-256 of the model file |
| `preprocess.json` | Feature order, `log1p` list, fill values, means and spreads, category values |
| `classes.json` | Class names in the order of the probability columns |
| `sklearn.joblib` / `mlp.pt` | The model: a scikit-learn model (joblib), or the MLP's settings and weights (`torch.save`, read back with `weights_only`, so the file cannot run code). Loading checks the file against the SHA-256 in `model_info.json` first, so a damaged or swapped file is refused; it does not protect against someone who can rewrite the whole folder, so load only folders you trust |
| `reference_predictions.csv` | The final model's probabilities for its own training fish (`video_id`, `predicted`, `p:<class>`); a reloaded copy must give exactly these (AC-9, D-006) |

The model's expected performance is the cross-validation estimate in `report.md`, never its score on the fish it was
refit on. A model whose verdict is *not useful* is still saved (the report and `model_info.json` say so), so the
pipeline can be checked end to end; do not draw conclusions from its predictions.

```python
from dcs.artifact import load_model
model = load_model("outputs/dcs/training/<run_id>")      # device="cuda" to run an MLP on the GPU
model.version_warnings                                   # set when scikit-learn/torch differ from the saved versions
model.predict(table)                                     # video_id, predicted, p:<class>; any table from `featurize`
```

A missing feature column stops with the names of the missing columns; extra columns are ignored. A different
scikit-learn or torch version gives a warning, not an error: the predictions may differ in the last digits.

## Predicting new fish (`dcs predict`)

```bash
python -m dcs predict --model outputs/dcs/training/<run_id> --input outputs/dcs/training_table.parquet
python -m dcs featurize --videos <folder of video folders> [--out new_fish.parquet]   # fish without index/catalog
python -m dcs predict --model outputs/dcs/training/<run_id> --input outputs/dcs/videos_table.parquet --out scores.csv
```

`predict` loads `<run>/model/`, prints a warning per library whose version differs from the saved one, and writes one
row per fish: `video_id`, `predicted` and `p:<class>` per class (default file `<DCS_OUTPUT_DIR>/<input name>_predictions.csv`).
It uses only the saved preprocessing; the input needs every feature column the model was trained with (a missing one
stops with its name), extra columns are ignored. Input: a `featurize` table (`.parquet`) or the same columns as `.csv`.
When some input fish are the model's own training fish, it compares them with `reference_predictions.csv` and prints
`matches the saved reference predictions for N fish` (or `DIFFERS`, with the largest gap): run it on the training
table after copying a run folder to another machine to check the copy (AC-9). `--device` (default `cpu`) matters for
an MLP only.

`featurize --videos <dir>` reads per-video folders (`<dir>/<video_id>/` with `manifest.json`, `frames.parquet`,
`segments.csv`, as `prepds run` writes under `outputs/processed/`) **without** `trials_catalog.parquet` or the
accepted index (D-018). Labels come from each manifest; the date stays empty; rejected and unprocessed videos are
left out; every fish not Accepted is marked unreviewed (`reviewed = false`). With `DCS_DB_PATH` set, NTT values come
from the workbook as usual. Its table goes to `<DCS_OUTPUT_DIR>/videos_table.parquet` (or `--out`) and never replaces
the training table. Such a table is for `predict` only; `train` needs dates.

## Research query tools (what the chat looks things up with)

`dcs.chat_tools` (data) and `dcs.chat_results` (training-run results) hold the functions the research chat calls, `dcs.chat_core` the registry they share; every number in a chat answer comes from one of them.
They read the training table and its schema, the per-video folders (frames and segments, from the gold source in the
settings) and a run folder (`ResearchData.from_settings(settings)` takes the latest one). They change nothing.

| Tool | Answers | Built from |
|---|---|---|
| `list_compounds` | Which compounds, how many fish, Accepted fish, dates and doses each; which is vehicle | Table |
| `find_features` | What a measure is called and what it means ("freezing" -> `state_freezing_drift_share`, ...) | Schema + glossary |
| `compare_to_vehicle` | One measure, compound (or one dose) vs vehicle: n, mean, sd, median, quartiles, Hedges' g, Mann-Whitney p | Table |
| `top_differences` | Every measure ranked by effect size for a compound, with p and Benjamini-Hochberg q | Table |
| `feature_by_compound` | One measure across all compounds, each against vehicle | Table |
| `fish_profile` | One fish: labels, date, flags, every measure with its percentile in its compound and in vehicle, out-of-fold predictions | Table + run |
| `fish_timeline` | One fish over time: state shares and mean speed per bin, the bouts, first time each state appeared | `frames.parquet`, `segments.csv` |
| `model_results` | Scores per model (scheme A, B, permuted, date effect), verdict and reason, saved model | `metrics.csv`, `decisions.csv`, `run_info.json` |
| `class_scores` | Per-class precision, recall, F1 and the most common confusions | `predictions.csv` |
| `ablation_results` | Each ablation next to the main run | `metrics.csv` |
| `audit_facts` | The audit's facts and notes | Table (same as `dcs audit`) |

**The control is vehicle fish from the same dates** (D-071): compounds were recorded on different days and cameras,
so a plain compound-vs-all-vehicle difference can be a date or camera difference. With fewer than 3 vehicle fish on
those dates the tool uses all vehicle fish and says so in `caveats`. Every comparison also carries the UNREVIEWED
caveat (temporarily accepted data) and, for pixel measures, the camera caveat. A bad argument (unknown compound,
feature or fish) returns `{"error": ...}` listing the valid choices instead of stopping. To call one by hand:

```python
from dcs.chat_tools import ResearchData, call_tool
data = ResearchData.from_settings(settings)                 # settings from load_settings()
call_tool(data, "top_differences", {"compound": "COMPOUND_A", "n": 5})
call_tool(data, "fish_timeline", {"video_id": "F_0042", "bin_s": 60})
```

## Research chat (`dcs ask`, `dcs serve-chat`)

Researchers ask questions in English ("Did COMPOUND_A fish freeze more than vehicle?", "When did F_0042 start
swimming erratically?", "Which compound changed speed the most?", "How well can the model tell vehicle from drug?").
A **language model running on the box** reads the question, calls the query tools above for every number, and writes
the answer with the tools' caveats. **Nothing is trained** (D-071): a few hundred fish cannot teach a language model
facts, and answers built from tool results can be checked. Use an existing open-weights *instruct* model that supports
tool calling (for example a Qwen or Llama instruct model); larger models follow the rules better, and the GB10's
128 GB of unified memory fits large ones.

**1. Start a chat server on the box** (once). Any server that speaks the OpenAI chat-completions protocol with tools
works: Ollama, a llama.cpp server, vLLM. With Ollama (check its install page for Linux ARM64):

```bash
ollama serve &                      # listens on 127.0.0.1:11434
ollama pull <model>                 # once; `ollama list` shows the exact name
```

**2. Point `dcs` at the server and name the model.** The server's address is machine-specific, so it goes in `.env`
(git-ignored), never in a YAML file (a `chat.base_url` there is refused):

```bash
DCS_CHAT_BASE_URL=http://127.0.0.1:11434/v1   # Ollama on this machine; any http(s) address of an OpenAI-compatible server
```

The model's name goes in your override file (`DCS_CONFIG`), next to the training settings:

```yaml
chat:
  model: <name as `ollama list` shows it>
```

**3. Ask.** `python -m dcs ask "question"` answers one question; `python -m dcs ask` starts a session that keeps the
conversation (empty line or `exit` ends it); `--show-tools` prints every tool call and whether it worked;
`--run <run folder>` picks a run other than the latest. The chat reads the same settings as the other commands:
`DCS_TABLE`, the video folders of the gold source (for timelines), the latest run under `<DCS_OUTPUT_DIR>/training`.

**4. For a web frontend:** `python -m dcs serve-chat` serves a JSON API on `127.0.0.1:8010` (loopback only: it has no
login, like `prepds review`). From your PC: `ssh -L 8010:127.0.0.1:8010 <user>@<box>`, then the frontend talks to
`http://127.0.0.1:8010`. The API answers only requests whose `Host` is a loopback name (127.0.0.1, localhost, [::1]);
a POST with an `Origin` from another site gets 403, so a web page you visit cannot reach it through DNS rebinding. A
dev server on `localhost:<port>` (e.g. Vite) may call it. `/api/ask` accepts only `user` and `assistant` turns in
`history`, a question of at most 4,000 characters and at most 50 earlier turns:

| Endpoint | Body | Returns |
|---|---|---|
| `GET /api/health` | | model, run, fish |
| `GET /api/tools` | | the query tools (OpenAI function format) |
| `POST /api/ask` | `{"question": "...", "history": [...]}` | `{"answer", "tool_calls", "history"}`; send `history` back with the next question |
| `POST /api/tool` | `{"name": "fish_timeline", "arguments": {"video_id": "F_0042"}}` | the tool's JSON, without the language model (for charts and tables) |

**Where the data goes.** Only to the chat server in `DCS_CHAT_BASE_URL`. An address on another machine is refused
unless `chat.allow_remote: true`, because questions and tool results (numbers per compound and per fish) would leave the
box; set it only if your data agreement allows it. A hosted server's key goes in `DCS_CHAT_API_KEY` in your `.env`
(git-ignored) or the environment, never in `.env.example`, a YAML file or anything committed; `check-config` prints it as
`***`. Where to host the model later is the lab's choice: only `DCS_CHAT_BASE_URL` (and maybe the key) changes.

| Setting (`chat:`) | Default | Meaning |
|---|---|---|
| `model` | `null` (must be set) | Model name on that server |
| `temperature` | 0.1 | Low = steadier answers |
| `max_tool_rounds` | 8 | Tool calls per question before the model must answer |
| `timeout_s` | 300 | Seconds to wait for one reply |
| `allow_remote` | `false` | Allow a server that is not this machine |

## Training on the GB10 (temporarily accepted data)

Copying files onto the GB10 (WinSCP, `pscp`) may not be allowed for this account. The box therefore builds everything
itself from a clone: it re-runs `prepds` on the raw videos and trains on that output, read as **temporarily accepted**.
This is `training.gold_source: processed`, the default (D-033). Every fish counts as data, every report says
UNREVIEWED, and `reviewed` stays false. Once enough videos are Accepted and the results justify it, the accepted
folder is moved to the box and `gold_source: accepted` is switched on (D-057).

What the box needs that git does not carry (NFR-2): the **raw videos** (`PDS_VIDEO_DIR`) and the **trial workbook**
(`PDS_DB_PATH`, also `DCS_DB_PATH` for the NTT columns), readable from the box. The calibration profiles are in git, so
a clone re-runs with the same profile as the PC (`dcs audit` prints it; it must match the PC's run).

```bash
# 1. once: code and environment (Python 3.11 or newer)
git clone <repository url> fish-detection && cd fish-detection
git checkout <branch to run>
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu130   # CUDA build for aarch64 first (PRD Appendix B step 5)
pip install -e ".[dev,train]"                      # torch already satisfied, so it is not replaced
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
pytest tests/dcs -q                                # synthetic data only; must pass on aarch64 (EC-19)

# 2. once: paths on the box (.env is git-ignored)
cp .env.example .env    # set PDS_VIDEO_DIR, PDS_DB_PATH, DCS_DB_PATH; keep PDS_OUTPUT_DIR=outputs, DCS_PROCESSED_DIR=outputs
printf 'training:\n  vehicle_compound: <vehicle name as the workbook writes it>\n' > my_training.yaml
#   after the first `dcs audit`, add camera_epochs: [<YYYY-MM-DD>, ...] from its framing table (see Ablations)
echo 'DCS_CONFIG=my_training.yaml' >> .env
python -m prepds check-config && python -m dcs check-config

# 3. preprocessing, then training; inside tmux so a dropped SSH session does not stop it
tmux new -s fish                                    # detach: Ctrl-b d   reattach: tmux attach -t fish
python -m prepds catalog
python -m prepds run --workers <cores - 2>         # resumable; videos already done are skipped
python -m dcs featurize
python -m dcs audit
python -m dcs train 2>&1 | tee train.log
python -m dcs ask                                   # research chat on the results (see Research chat; needs a chat server)
```

Read the result on the box: `less outputs/dcs/training/<run_id>/report.md` (the last lines of `train.log` name the
folder). Nothing has to leave the machine to read the verdict. If a file must come back and copying is blocked, the
report is plain text: open it in the SSH window and copy the table. **Never put real names, dates or results into
git** (NFR-2).

After a `git pull` with new `dcs` code: run `python -m dcs featurize` again if `featurize` changed (the progress file
says so), then `dcs train`. `prepds run` only needs repeating when `prepds` or the calibration profile changed.

## When `dcs` stops: what to do

Errors print one line starting with `Configuration error:` (exit code 2).

| Message contains | Fix |
|---|---|
| `DCS_PROCESSED_DIR is not set` | Add `DCS_PROCESSED_DIR=outputs` to `.env`, or set `training.gold_source: accepted` |
| `No trials_catalog.parquet` / `No processed/ folder` | Run `prepds catalog` and `prepds run`, or fix `DCS_PROCESSED_DIR` |
| `No accepted_index.parquet` | Run `prepds export-index`, or fix `DCS_ACCEPTED_DIR` |
| `The set mixes calibration profiles` | Choose one: `featurize --profile <version>` |
| `fish have no date` | Run `prepds catalog`, then `prepds export-index` |
| `Undetermined found in an Accepted video` | Review that video again in `prepds review` |
| `column(s) ... missing` / `has dtype` | prepds output from a different version: re-run `prepds run` for those videos |
| `workbook compound ... disagrees` / `different rows for the same fish` | Correct the workbook row named in the message |
| `class(es) left for the ... stage` | Too few fish per class; the message lists them. Wait for more fish, or lower `training.min_class_size` in an override file. In the dose stage a compound also needs 2 kept doses |
| `The training table has no fish` | `featurize` kept nobody: look at `dropped` in the schema file |
| `No schema file ... beside the table` / `schema file ... is damaged` | Copy both featurize files together, or run `python -m dcs featurize` again |
| `marks forbidden column(s)` / `in the schema but not in the table` | The two files come from different runs or were edited: run `featurize` again |
| `No feature left` | Every usable feature is constant or of a rare state: switch on a group (`use_ntt`, `use_depth`) or lower `min_state_fish` |
| `--device cuda, but PyTorch sees no CUDA GPU` (train) | Install the CUDA build of torch (GB10 runbook, step 1), or run with `--device auto` or `cpu` |
| `--models must be` / `--stage must be` / `--seed must be` / `--repeats must be` (train) | Fix the command-line value; the message lists what is allowed |
| `<stage> stage cannot be built: ...` (train) | No stage could be built; the rest of the message is one of the training-set errors above |
| `DCS_CHAT_BASE_URL is not set` | Add `DCS_CHAT_BASE_URL=<server>/v1` to `.env` (see `.env.example`) |
| `DCS_CHAT_BASE_URL must be an http:// or https:// address` | Fix the address in `.env` or the environment |
| `chat.base_url` / `chat.api_key in ... is not read from YAML` | Remove it from your override file and set `DCS_CHAT_BASE_URL` / `DCS_CHAT_API_KEY` in `.env` |
| `chat.model is not set` | Put the model's name in your override file under `chat: model:` |
| `Cannot reach the chat server at ...` | Start it (`ollama serve`, `ollama pull <model>`) or fix `DCS_CHAT_BASE_URL` |
| `The chat server at ... answered 4xx/5xx` | Usually a wrong model name (`ollama list`) or a model without tool calling |
| `DCS_CHAT_BASE_URL ... is not this machine` | Use a local server, or set `chat.allow_remote: true` if your data agreement allows it |
| `the chat API has no authentication: --host must be a loopback address` | Keep `serve-chat` on 127.0.0.1 and reach it through an SSH tunnel |
| `--input ... does not exist` (predict) | Point `--input` at a table from `dcs featurize` |
| `--videos ... is not a folder` | Point it at the folder holding one `<video_id>/` folder per fish |
| `... does not match the checksum saved with the model` / `has no checksum for ...` | The model file changed after training (bad copy, edit) or the folder is from an older `dcs`: copy it again or re-run `python -m dcs train` |
| `No saved model in ...` | `--model` must be a run folder written by `dcs train` (it holds `model/model_info.json`) |
| `The input lacks ... feature column(s) the model needs` | Build the input with `python -m dcs featurize` from the same `dcs` version as the model |
| `DCS_TABLE points to ..., which does not exist` (audit) | Run `python -m dcs featurize` first, or point `DCS_TABLE` at a copied table (copy its schema file with it) |

The audit's **notes** are not errors: it always writes `audit.md`. A note such as `vehicle ... is not in the table`
means set `training.vehicle_compound`; `... stage cannot be built: ...` carries the same message the table above explains.

## Tests

```bash
pytest tests/dcs -q        # about 25 s on an idle machine, synthetic data only; run after every change
pytest tests/ -q           # everything, before a commit
```

| File | Checks |
|---|---|
| `test_dcs_config.py`, `test_dcs_config_params.py` | Settings precedence, override YAML, rejected keys and values, `check-config` |
| `test_dcs_synthetic*.py` | Synthetic files match the prepds formats, same seed gives the same data, each edge-case switch does what it says |
| `test_dcs_gold.py`, `_gold_files.py`, `_gold_processed.py` | What the gold reader drops and what stops it, for both sources |
| `test_dcs_workbook.py` | NTT columns: header cleanup, `-` rule, duplicates, disagreements (compound compared by the training set's spelling rule), blank-compound rows ignored, missing fish |
| `test_dcs_featurize.py` | Each feature on small hand-built segments and frames (the expected numbers can be checked by hand), including run-start frames left out of the kinematics; a detected fish with no depth values gets NaN depth features |
| `test_dcs_featurize_table.py` | The table and schema on a synthetic set, flags, drops, and `dcs featurize` itself |
| `test_dcs_trainset.py` | Label cleaning, class filter and very-small flag (both stages), forbidden columns never in the matrix, group switches, rare-state and constant features dropped, the stop messages, and a synthetic set end to end |
| `test_dcs_audit.py` | Each audit table and fact on small hand-built tables: filter steps, class status and scheme A, per-date and compound × date counts, missing values, manual share and flags per compound, §2.3 statistics, a compound with no Accepted fish, the D-016 share, the G1 stop rule (vehicle, one-date and unaccepted compounds never count; the unreviewed count), vehicle name from the settings, fps and tracker notes, framing setups (a shift, a zoom, two equally common setups, slow drift, one fish at the surface, a date without depth data), dose spelling variants counted once, a stage that cannot be built; `dcs audit` end to end on synthetic data and without a table |
| `test_dcs_preprocess.py` | Training rows come out with mean 0 and spread 1; an extreme test fold leaves the fitted numbers alone (EC-9); gaps get the training median (EC-2); `log1p` only on count and duration kinds; JSON round trip; constant or empty columns in a fold; column order; category features one-hot, unseen or missing values all 0 |
| `test_dcs_models.py` | Every baseline has the same interface and probabilities that sum to 1; majority gives the class shares; date-only: same date, nearest date, tie to the earlier date, non-date text; logreg, forest and boosting find a planted signal and weight classes; same seed, same probabilities; `mlp` and unknown names are errors |
| `test_dcs_evaluate.py` | Every model sees the same folds; majority and date-only always run; preprocessing is fitted without the test fold (EC-9, by spying on every fit); metrics equal scikit-learn's on the pooled predictions of a repeat, no plain accuracy (D-017); per-class scores pool the repeats; spread = standard deviation across repeats; the permutation stays inside each date; a planted signal is judged useful; **leakage canary** (label depends only on the date: scheme B perfect, scheme A below chance, permuted = real, not useful); pinned fish never scored in scheme A; no scheme A means undecided; vehicle vs drug; same seed, same metrics (EC-14) |
| `test_dcs_mlp.py` | (runs only where torch is installed) Interface and a planted signal; built from the settings; layer sizes, ReLU and dropout from the config; one network per seed; early stopping on noise; balanced loss weights; same seed same CPU probabilities, other seed different (EC-14); a one-fish class still trains; `--device` rules (EC-15) |
| `test_dcs_no_torch.py` | With torch hidden: `torch_available()` is false and `dcs train --models logreg,mlp --device cuda` runs the baselines and skips the MLP with a message (EC-16) |
| `test_dcs_ablations.py` | Vehicle normalization: vehicle fish removed, counts logged once; date reference with 2+ vehicle fish; fallback to the framing setup, then all vehicle fish; camera-epoch reference skips the date level; `epoch_of`; the NTT keep rule (gain vs spread, undecided without scheme A); every ablation reported; switch ablations reuse the main folds, or say `new folds` when the switch changes the fish; the FR-9 pair shares classes and folds; FR-9 is a note when the vehicle is not a kept class; `camera_epochs` switches the reference |
| `test_dcs_stage2.py` | One dose model per compound with two doses, never the vehicle; a compound left with one dose gets none; date-only and the permuted score beside every dose model; the FR-9 variant on the same folds; without the vehicle it is a note; no dose class left is a note, not a stop; `dcs train` writes one section, metrics rows and confusion PNG per compound |
| `test_dcs_artifact.py` | `model/` holds the files of PRD §7.4 plus the reference predictions, classes in probability order, model named in the report and `run_info.json`; reload in a **fresh process** reproduces the reference (EC-18) for logistic regression and (with torch) the MLP; a version mismatch is a warning; a missing feature column is an error naming it; no model folder is an error; a changed model file or a missing checksum is refused |
| `test_dcs_predict.py` | `dcs predict` on the training table reproduces `reference_predictions.csv` and says so (AC-9); default output file and CSV input; missing columns stop with their names; version warnings printed; `featurize --videos` needs no index, marks unreviewed fish, leaves the date empty and the training table untouched; with the workbook the NTT values are kept; folders -> `featurize --videos` -> `predict` gives the same predictions as the reference (D-018) |
| `test_dcs_privacy.py` | `git ls-files` holds no `.parquet`, `.csv`, `.xlsx`, `.pt`, `.joblib`, pickle, video or `.env` file outside the two named synthetic fixtures (AC-8); the outputs of `dcs` and `prepds` and the restricted inputs are git-ignored; the pattern itself catches what it should. Skipped outside a git checkout |
| `test_dcs_chat_tools.py` | Tool schemas valid and unique; compounds and vehicle; feature search; `compare_to_vehicle` equals hand-computed n, mean, Hedges' g and Mann-Whitney p with the same-date control; fallback to all vehicle fish with a caveat; bad arguments return errors naming the choices; `top_differences` ordered with q >= p; fish profile percentiles and out-of-fold predictions; timeline bins cover the recording and shares sum to 1; model results, class scores, ablations and audit facts read from a real `dcs train` run; no run -> a hint; the latest run found from the settings; `fish_timeline` refuses a zero, negative, NaN, infinite, boolean, text or too-small `bin_s` with an error; a tool that crashes comes back as `{"error"}` instead of ending the chat |
| `test_dcs_chat.py` | The answer is built from a tool result (scripted engine); the system prompt sets the rules and names the data; history carries earlier turns; dict arguments and tool errors go back to the model; too many tool rounds force an answer without tools; local servers allowed, a remote one only with `allow_remote`; no model name stops; the client speaks the OpenAI protocol to a local stub server (path, body, API key header); an unreachable server says how to start one; `dcs ask` one question; the terminal session keeps history; `serve-chat` refuses a public address; the API (`health`, `tools`, `ask`, `tool`) and a 503 on engine failure; the server address and key come only from `DCS_CHAT_BASE_URL` and `DCS_CHAT_API_KEY` (environment over `.env`; a non-http(s) address and a `chat.base_url` or `chat.api_key` in a YAML file are refused; no address stops with a hint; the key prints as `***` in `check-config`); a foreign `Host` (DNS rebinding) and a foreign `Origin` are refused; `/api/ask` refuses `system`/`tool` history turns, an empty or over-long question and too many turns |
| `test_dcs_report.py` | Baselines and the permuted score beside every model (AC-5); date effect = B − A on the fish scheme A scores (a date-confounded class is left out of B for this); no plain accuracy column; unreviewed data called temporarily accepted; very small and date-confounded classes named; dose-stage caveat; camera and FR-9 caveats; notes and skipped models shown; skipped scheme A; best model choice; confusion PNG |
| `test_dcs_train.py` | `dcs train` on a synthetic table: run folder and files, `run_id` form, `run_info.json` fields, command-line values in `config_used.yaml`, `folds.csv` equals the folds, stage and scheme columns, `mlp` skipped with a message, same seed gives the same metrics (two runs in one second get two folders), bad values and an unbuildable lone stage stop with a message and write nothing, missing table |
| `test_dcs_folds.py` | Single-date class pinned and flagged; scheme A skipped when every class is on one date; a two-date class's dates in different folds, and a note when that cannot be done; over 200 seeds no fish in two folds, no date in two scheme-A folds, no empty fold; same seed, same folds; fewer dates than folds; a synthetic set end to end |

## Keeping this section current

Whoever changes `dcs` (a new unit or a small fix) also does these, in the same branch:

- [ ] **Instructions:** the new or changed tool, command, setting, column or error message is described above (what it does, how to run it, what to do when it stops), and any new test file is in the Tests table.
- [ ] **Unused code:** nothing left that no one calls or reads (functions, constants, settings, duplicated constants). Search by name in `src/dcs` and `tests/dcs`; delete what is dead.
- [ ] **Tests:** `pytest tests/dcs -q` is green.
- [ ] **Progress file:** [`classifier_progress.md`](classifier_progress.md) lists the unit, its decisions (D-nnn) and what was removed.
