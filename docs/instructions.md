# Instructions: Preprocessing Dataset System

Turns each zebrafish drug-exposure trial video plus its trial-metadata row into a
reviewed, per-subject 5-state ethogram strip and machine-readable labels — the
labeled dataset a later, separate drug-classification project will train on. Full
specification: [`PRD.md`](PRD.md). Build status/progress:
[`progress.md`](progress.md).

## Run it: `./start.sh` and `./stop.sh`

Everyone runs the project the same way, from the repository root. Only Docker (with Compose)
is needed; no host Python or Node.

```bash
cp .env.example .env     # once: set PDS_VIDEO_DIR (synced video folder), PDS_DB_PATH (00_NTT_DataBase.xlsx),
                         # FISHLAB_HOST=127.0.0.1 and a free FISHLAB_PORT (the link you open);
                         # optional PDS_OUTPUT_DIR, PDS_ACCEPTED_DIR, PDS_WORKERS, PDS_CONFIG, DCS_* (see the file)
./start.sh               # build, prepare everything, start the app, print the local link
./stop.sh                # stop the app, verify a backup of saved work, remove the containers
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

**What `./stop.sh` does:** stops the app, writes and verifies a private dated archive of outputs, accepted data and
configuration under `backups/`, then removes the containers. If the backup fails, nothing is removed. Raw videos
and unsaved browser drafts are not in the backup.

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
