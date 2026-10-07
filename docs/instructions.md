# Instructions: Preprocessing Dataset System

Turns each zebrafish drug-exposure trial video plus its trial-metadata row into a
reviewed, per-subject 5-state ethogram strip and machine-readable labels — the
labeled dataset a later, separate drug-classification project will train on. Full
specification: [`PRD.md`](PRD.md). Build status/progress:
[`progress.md`](progress.md).

That later project lives in the same repository: see
[Classifier (`dcs`)](#classifier-dcs-compound-and-dose-from-behavior) at the end of this file.

## Setup

```bash
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
0.0%). Read [`progress.md`](progress.md) ("Review corrections") before relying on it: it is not human-reviewed, detection gain is
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

## Status

Implementation is tracked task-by-task in [`progress.md`](progress.md),
mirroring the PRD §12 phase breakdown (Phase 0-15).

---

# Classifier (`dcs`): compound and dose from behavior

`dcs` reads what `prepds` wrote (labels per frame and per segment, one folder per fish) and will learn to tell
which compound, and later which dose, a fish received from its behavior. Specification:
[`classifier_PRD.md`](classifier_PRD.md); plan: [`plans/classifier_plan.md`](plans/classifier_plan.md); status and
decisions (D-nnn): [`classifier_progress.md`](classifier_progress.md). It never imports `prepds` and never changes its files.

**Where it stands.** The tools that exist today (units U1-U8) build the **training table**, one row per fish, turn it
into the **training set** one stage learns from, split that set into the **folds** every model will share, and
**audit** the table before anything is trained. Training models (U9 and later) comes next. Every unit is tested on synthetic data. Until
reviewers Accept videos, the real input is the **unreviewed** prepds output, so no result is a claim about drugs yet.

```
prepds output ──► dcs featurize ──► training_table.parquet + training_table_schema.json ──► trainset ──► folds ──► (train: later)
   (+ workbook for NTT)                     one row per fish        what every column is      features,    scheme A / B
                                                     │                                         labels, dates
                                                     └──► dcs audit ──► audit.md (counts, drops, confounds, G1 verdict)
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
| `DCS_TABLE` | the training table (default `<DCS_OUTPUT_DIR>/training_table.parquet`) | `featurize` writes it; `audit` reads it (train will) |
| `DCS_CONFIG` | a YAML file overriding `config/default_training.yaml` | optional |

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

Global options go **before** the command: `python -m dcs --config my.yaml featurize`, `--env-file other.env`.

### Try it on synthetic data (about 10 seconds)

```bash
python -m dcs synth --out /tmp/dcs_try
printf 'training:\n  gold_source: accepted\n' > /tmp/dcs_try/accepted.yaml     # synth writes the accepted layout
DCS_ACCEPTED_DIR=/tmp/dcs_try/accepted DCS_DB_PATH=/tmp/dcs_try/synthetic_db.xlsx DCS_OUTPUT_DIR=/tmp/dcs_try/out \
  python -m dcs --config /tmp/dcs_try/accepted.yaml featurize
# Featurized 48 fish (reviewed, profile cal-synthetic), 0 dropped
DCS_OUTPUT_DIR=/tmp/dcs_try/out python -m dcs --config /tmp/dcs_try/accepted.yaml audit
# Audited 48 fish (Accepted gold folder): G1 stop rule GO      (report: /tmp/dcs_try/out/audit.md)
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
   framing), `ntt` only with `use_ntt` (on), `demographics` (`sex`, `strain`, `age`) only with `use_demographics` (off).
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
`notes`. The same `training.seed` always gives the same folds. To look at them:

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
the recordings fall into three camera epochs (fish vertical range about 71 px for the early 2024 dates, about 31 px for
late 2024 to early Feb 2025, about 34 px after; 18 framing setups, none spanning two epochs, one stable setup from
2025-02-18), and most compounds were recorded in one epoch only. Depth stays off, and
pixel speeds, state shares and `has_ntt` track the epoch (about 93 / 71 / 39 % NTT coverage), so scheme A cannot rule out
that a model recognizes the camera instead of the compound. Handle this before models are designed (per-epoch
normalization, FR-9).

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
| `test_dcs_featurize.py` | Each feature on small hand-built segments and frames (the expected numbers can be checked by hand), including run-start frames left out of the kinematics |
| `test_dcs_featurize_table.py` | The table and schema on a synthetic set, flags, drops, and `dcs featurize` itself |
| `test_dcs_trainset.py` | Label cleaning, class filter and very-small flag (both stages), forbidden columns never in the matrix, group switches, rare-state and constant features dropped, the stop messages, and a synthetic set end to end |
| `test_dcs_audit.py` | Each audit table and fact on small hand-built tables: filter steps, class status and scheme A, per-date and compound × date counts, missing values, manual share and flags per compound, §2.3 statistics, a compound with no Accepted fish, the D-016 share, the G1 stop rule (vehicle, one-date and unaccepted compounds never count; the unreviewed count), vehicle name from the settings, fps and tracker notes, framing setups (a shift, a zoom, two equally common setups, slow drift, one fish at the surface, a date without depth data), dose spelling variants counted once, a stage that cannot be built; `dcs audit` end to end on synthetic data and without a table |
| `test_dcs_folds.py` | Single-date class pinned and flagged; scheme A skipped when every class is on one date; a two-date class's dates in different folds, and a note when that cannot be done; over 200 seeds no fish in two folds, no date in two scheme-A folds, no empty fold; same seed, same folds; fewer dates than folds; a synthetic set end to end |

## Keeping this section current

Whoever changes `dcs` (a new unit or a small fix) also does these, in the same branch:

- [ ] **Instructions:** the new or changed tool, command, setting, column or error message is described above (what it does, how to run it, what to do when it stops), and any new test file is in the Tests table.
- [ ] **Unused code:** nothing left that no one calls or reads (functions, constants, settings, duplicated constants). Search by name in `src/dcs` and `tests/dcs`; delete what is dead.
- [ ] **Tests:** `pytest tests/dcs -q` is green.
- [ ] **Progress file:** [`classifier_progress.md`](classifier_progress.md) lists the unit, its decisions (D-nnn) and what was removed.
