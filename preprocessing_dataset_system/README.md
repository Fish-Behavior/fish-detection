# Preprocessing Dataset System

Turns each zebrafish drug-exposure trial video plus its trial-metadata row into a
reviewed, per-subject 5-state ethogram strip and machine-readable labels — the
labeled dataset a later, separate drug-classification project will train on. Full
specification: [`docs/PRD.md`](docs/PRD.md). Build status/progress:
[`docs/progress.md`](docs/progress.md).

Independent top-level package from `src/fishbehavior/` in the parent repo (see PRD
Clarification C10) — no shared runtime dependency between the two.

## Setup

```bash
cd preprocessing_dataset_system
uv venv --python 3.11 .venv     # or: python3.11 -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"      # add "[dev,ocr]" for optional pytesseract OCR support
cp .env.example .env            # then fill in PDS_VIDEO_DIR / PDS_DB_PATH / PDS_REFERENCE_DIR
python -m prepds check-config   # confirms every configured path actually exists
```

`opencv-python-headless` is used deliberately (not `opencv-python`) — this project
targets headless/WSL2-style environments with no display server, where the full
wheel's `libGL.so.1` dependency fails at import time.

## Quick start

```bash
# 1. configure (paths to the restricted data; nothing here is ever committed)
cp .env.example .env     # PDS_VIDEO_DIR (folder holding Phase_1/, Phase_2/ ...), PDS_DB_PATH (00_NTT_DataBase.xlsx),
                         # PDS_OUTPUT_DIR, PDS_ACCEPTED_DIR, optional PDS_WORKERS / PDS_CONFIG
python -m prepds check-config

# 2. match trial rows to videos (writes outputs/trials_catalog.parquet + exceptions_report.json)
python -m prepds catalog

# 3. label every matched video (resumable; ~50 min for 328 videos on 22 workers)
python -m prepds run --dry-run      # list what would be processed
python -m prepds run                # writes outputs/processed/<sex>_<subject>/ + outputs/run_report.json

# 4. review in the browser (local, loopback only), then rebuild the gold index
python -m prepds review             # http://127.0.0.1:8000/
python -m prepds export-index       # accepted_index.parquet from the ACCEPTED videos
```

## Commands

- `check-config` - show resolved paths/settings; exit 1 if a configured path is missing.
- `catalog` - FR-001..004: match trial rows to local videos (compound folders are found at any depth under
  `PDS_VIDEO_DIR`). Unmatched trials/videos, duplicates and duration mismatches go to `exceptions_report.json`, not errors.
- `run [--workers N] [--limit N] [--dry-run] [--force]` - one video per worker process: track -> features -> label
  -> 1 s consolidation -> review flag -> strip PNG -> export. Resumable: videos already processed, edited or accepted
  are skipped; **`--force` also overwrites edited/accepted work**. A failing video (including a crashed worker) is
  reported in `run_report.json` and the run continues; exit code 1 if any failed. Default workers
  `max(1, cpu_count - 2)`; `PDS_WORKERS` or `--workers` override. Measured: 328 videos (108.8 h) in 50 min on 22 workers
  (the machine was oversubscribed, load ~37 on 24 cores; fewer workers may be as fast). Run only one `run` at a time.
- `run --tracker model:<run dir> [--stride 5] [--device cuda]` - track with a fine-tuned fish detector instead of the classical
  tracker (needs the `ml` extra and a GPU; single process, about 75 s/video). A model run must use its own calibration
  profile (`--config`, e.g. `cal-2026-09-24-r4-model-m3.yaml`) and should write to its own `PDS_OUTPUT_DIR` /
  `PDS_ACCEPTED_DIR` so it never overwrites the classical output. It also writes `detections.parquet` per video.
- `review [--host 127.0.0.1] [--port 8000]` - review web app; refuses non-loopback hosts (no authentication).
- `annotate [--port 8001]` - frame-labeling page (box, 5 keypoints, Listing tag) for the Phase 15 detector; needs
  `outputs/phase15/sample.json` (see `scripts/phase15_prepare.py`).
- `export-index` - rebuild `accepted_index.parquet` from ACCEPTED videos on disk, adding workbook fields from the catalog.
- Calibration is not a CLI command: see below.

## Outputs per video

`outputs/processed/<sex>_<subject>/` holds `frames.parquet` (per-frame track, features, state, source),
`segments.csv` (exact run-length encoding of the frames), `strip.png` (one colour strip per video) and
`manifest.json` (status, provenance, review flags). Accepting copies these plus `provenance.json` to
`accepted/<video>/` and updates `accepted/accepted_index.parquet` (the gold dataset, directly consumable).

## Model tracker and Listing hints (Phase 15)

The fine-tuned detector (`scripts/phase15_finetune.py`, run `m3`, trained on ~330 human-labeled frames) replaces the
classical tracker in `outputs_r3/` (328 videos: undetected frames median 21.5% -> 0.4%, Undetermined median 13.7% ->
0.0%). Read `docs/progress.md` ("Review corrections") before relying on it: it is not human-reviewed, detection gain is
not proof of track accuracy, it can report a fish on empty frames, and its calibration is weakly identified.
Listing/LORR is still a manual label. `scripts/phase15_flag_listing.py` writes advisory `listing_flags.json` hints
(crop classifier, high recall, low precision) that the review app shows next to the video; they never change a state
or the accept rules.

## Review UI

`prepds review`, pick a video, watch it with the timeline synced to playback, drag on the timeline to select a
time range, choose a state, **Stage relabel**, then **Save edits**. Accept is blocked while any `Undetermined`
remains (no override) and asks for confirmation when `Dead` is present. Reject clears edits and re-queues the video.
Keys on the timeline: arrows seek, `[` / `]` set the selection start/end at the playhead. Review flags (e.g. no
movement until the end of the video) are shown above the player.

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
  route to fewer Undetermined frames (see Phase 15 in `docs/progress.md`).

## Data handling

Videos, the trial workbook, and the two reference images are restricted research
data (PRD FR-018 / NFR-004) and are **never** committed — `data/`, `outputs/`,
`accepted/`, and the raw-data file extensions are gitignored (see `.gitignore`,
which extends the repo root's policy). Only synthetic fixtures under
`tests/fixtures/` are committed, via narrow negations in this package's own
`.gitignore` — verified with `git check-ignore -v <path>` before each fixture is
added, not assumed.

## Status

Implementation is tracked task-by-task in [`docs/progress.md`](docs/progress.md),
mirroring the PRD §12 phase breakdown (Phase 0-14).
