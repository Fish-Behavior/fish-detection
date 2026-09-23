# Zebrafish Drug-Response Detection System

Advisor: Dr. Ashish Kharel

Collaborator: Dr. Scott Hall

## Project Overview

This project develops an AI system that analyzes zebrafish behavior to estimate
which psychoactive compound and dose produced an observed response. The work is a
collaboration between the University of Toledo research team and Pharmacy
Department.

The project addresses the overdose and drug-abuse crisis, where current methods
cannot rapidly classify newly emerging psychoactive compounds. Behavioral
fingerprinting focuses on measurable features including distance, velocity,
anxiety, exploration, and mobility.

## Scientific Objective

The system will combine computer-vision tracking, supervised learning, and
unsupervised behavior-state discovery. For each session, it is intended to:

- extract movement and behavioral features from recorded or live video;
- estimate probabilities across known compound-and-dose classes;
- identify behavior that does not match known compound profiles;
- produce a timestamped behavior log and natural-language explanation; and
- support researcher queries about previous sessions.

The initial ethogram contains five target states: Controlled Swim, Erratic
Movement, Freezing/Drift, Listing/LORR, and Surface Breach.

## Current Scope and Status

Scoping is complete and Phase 1 is ready to begin. The committed near-term scope
is offline analysis of recorded video. Phase 2 is planned for live camera
integration, streaming inference, and real-time narration.

See the complete scientific and technical scope in
[docs/zebrafish_drug_detection_scope.md](docs/zebrafish_drug_detection_scope.md).

The behavior-labeling pipeline (`fishbehavior` package) is being built one step at a
time, one pull request per step. The project setup (settings loading, command-line
entry point, tests), the workbook catalog (cleaning + video matching), and scene
setup (background, waterline, fish region per video) are in place; the analysis steps listed under
[Project Structure](#project-structure) as *planned* are added in later PRs. The
classifier, backend, and frontend are not yet implemented.

## Project Structure

```text
fish-detection/
├── .env.example              # template for your local .env (data paths); copy it, never commit .env
├── .gitignore                # keeps data, outputs, and .env out of Git
├── pyproject.toml            # package definition and dependencies
├── requirements.txt          # one-line install: the package + developer tools
├── docs/                     # project scope and contribution workflow
├── src/fishbehavior/         # the pipeline package
│   ├── __main__.py           # enables `python -m fishbehavior <command>`
│   ├── catalog.py            # cleans the trial workbook and matches each trial to its video(s)
│   ├── cli.py                # command-line entry point; one sub-command per pipeline step
│   ├── config.py             # reads paths from .env / environment and parameters from YAML
│   ├── default_config.yaml   # default, data-independent parameters
│   ├── review.py             # scene-review: local browser page to check/correct waterlines and ROIs
│   ├── review_page.html      # the review page template (self-contained, no external files)
│   ├── roi.py                # per video: empty-beaker background, waterline, fish region (ROI), QA image
│   └── video.py              # video timing (fps, frames, duration) and frame reading
└── tests/                    # automated tests (synthetic data only, never real trials)
    └── synthetic.py          # draws synthetic beaker/fish videos for the tests
```

Pipeline steps planned for later PRs, in order:

| Step | Module | What it does |
|---|---|---|
| Fish tracking | `tracking.py` | locates the fish in every frame (position, size, tilt) |
| Features | `features.py` | speed, turning, smoothness, height, and posture per frame and per time bin |
| Behavior labeling | `labeling.py` | assigns one of the five ethogram states to each time bin and merges them into segments |
| Slide digitizer | `reference.py` | turns the reference ethogram figures into approximate timelines for tuning |
| Priors and calibration | `priors.py`, `calibrate.py` | tunes thresholds against the reference and sanity checks |
| Dataset export | `export.py` | writes segments, per-subject summaries, and training datasets |
| Plots | `plots.py` | ethogram charts, summary bar charts, and review videos |
| Colab and docs | `notebooks/`, `docs/` | Google Colab quick start and full usage guide |

Everything the pipeline generates is written to the output folder (`outputs/` by
default), which is ignored by Git.

## Data Sources and Access

### Trial database

The planned tabular source is `00_NTT_DataBase.xlsx`, which contains 720 rows:

- 353 rows contain real trial data; the remaining rows are blank templates;
- metrics include distance moved, velocity, and time-in-zone measurements;
- the empty `Body Tissue` column is excluded; and
- blank compound rows are excluded from processing.

### Video dataset

There are 353 videos, one for each real trial. Filenames identify sex and subject
number, such as `F_0068.mp4`, and match the corresponding database row.

The reference recordings are approximately 320x240 pixels, 30 fps, and 20 minutes
long, with a single fish in a side-view tank. The complete collection is
approximately 2 GB.

The database and videos are restricted research data and must be obtained through
the appropriate project channels. Do not commit them to this repository.

## Prerequisites

- Python 3.10 or newer (developed on 3.11) in a virtual environment (`venv`);
- Windows, macOS, Linux, or Google Colab.

The pipeline's own dependencies are listed in `pyproject.toml` and grow as steps are
added. The planned stack beyond the labeling pipeline includes:

- TensorFlow for learned components;
- OpenCV for computer-vision tracking;
- a mid-range GPU for the classifier and tracking experiments; and
- either a hosted language-model API or a self-hosted model for reporting.

## Quick Start

The intended Phase 1 workflow is:

```text
data acquisition
	-> dataset validation and database/video matching
	-> database cleaning
	-> video tracking
	-> behavior-state extraction
	-> classifier training and evaluation
	-> anomaly detection
	-> report generation
```

### Environment Setup

From the repository folder:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest                      # runs the automated tests (no real data needed)
```

### Data Placement

Keep the approved workbook and videos outside Git (for example in a `data/` folder,
which is ignored) and tell the pipeline where they are through a local `.env` file:

```bash
cp .env.example .env        # Windows: copy .env.example .env
# then edit .env and set FISH_VIDEO_DIR, FISH_DB_PATH, FISH_REFERENCE_PDF, FISH_OUTPUT_DIR
python -m fishbehavior check-config
```

`check-config` prints every path the pipeline will use and marks it `ok`, `not set`,
or `MISSING`. Environment variables with the same names override the `.env` file,
which is how Google Colab is configured. Parameters can be changed without editing
the code by pointing `FISH_CONFIG` to a YAML file with only the values to change.

### Run the Phase 1 Pipeline

Each pipeline step adds its own command (`python -m fishbehavior <command>`) as it
is merged; `python -m fishbehavior --help` lists the commands available so far.

**1. Validate the workbook and videos**

```bash
python -m fishbehavior validate            # add --strict to exit with code 1 if any issue is found
```

This reads the workbook (`FISH_DB_PATH`), applies the data-preparation rules below,
matches every subject to its video file(s) in `FISH_VIDEO_DIR`, and writes to
`<FISH_OUTPUT_DIR>/catalog/`:

- `trials.csv`: one row per subject with the cleaned values, `ntt_tracked`,
  `video_status` (`matched`, `missing`, `part_mismatch`,
  `duplicate_videos`, or `not_checked`), and `video_paths`;
- `videos.csv`: every video file found and what was read from its name;
- `validation_report.md`: everything that needs attention, plus reference lists.

Video names must contain the subject number, such as `F_0042.mp4`; a letter prefix
is allowed and ignored (sex comes from the workbook). A
recording split into several files uses a part letter (`M_0012a.mp4`,
`M_0012_b.mp4`, or `M_0012-b.mp4`); a subject number that appears on several
workbook rows is treated as one fish recorded in parts and joined into one subject.
Without `FISH_VIDEO_DIR`, only the workbook is checked.

**2. Scene setup: background, waterline, and fish region**

```bash
python -m fishbehavior scene                     # every subject with video_status = matched
python -m fishbehavior scene --subjects 42 F_0043  # only these ("42", "0042", "F_0042" all work)
python -m fishbehavior scene --force             # redo subjects that already have results
```

Needs `trials.csv` from `validate`. For every video it samples frames evenly
(`scene.n_background_frames`, default 100) and computes:

- the **background**: the per-pixel median of the samples. The fish keeps moving, so it
  disappears from the median and what remains is the empty beaker;
- the **activity map**: how often each pixel differs clearly from the background. By
  default (`scene.activity_method: both`) a pixel must change in brightness **and** in
  color. Light flicker, camera exposure changes, and glare on the water change only
  brightness, and the faint reflection in the glass changes little in brightness, so
  none of them count; a colored fish does. Each frame's overall brightness change is
  removed first, and small separate specks are dropped. For fish without color, set
  `activity_method: gray` in your FISH_CONFIG file;
- the **ROI** (region where the fish can be): the box around the activity, padded, with
  its top raised just above the waterline so a surface breach stays inside;
- the **waterline**: the row with the strongest horizontal edge in the background near the
  top of the activity box, with a confidence score (edge strength / typical row).

Videos are processed in `FISH_WORKERS` parallel processes. Results go to
`<FISH_OUTPUT_DIR>/scene/`:

- `<video>.json`: video info (fps, frames, duration, size), `waterline_y`,
  `waterline_confidence`, `roi` as `[x0, y0, x1, y1]` in original pixels (x1/y1
  exclusive), `method` (`auto` or `override`), and flags;
- `<video>_background.png`: the empty-beaker background;
- `<video>_activity.png`: where the fish moved (white = often), used by the review page;
- `<video>_frames.jpg`: a few real color frames from across the video, stacked, for the review page;
- `<video>_qa.png`: the check image (below);
- `video_check.csv`: one row per video (`subject_id, file, fps, frames, duration_s,
  width, height, flags`). Flags: `duration` (the subject's parts together differ from
  `scene.expected_duration_s` by more than `scene.duration_tolerance`), `fps_mismatch`
  (parts of one subject have different frame rates), `low_waterline_confidence`,
  `no_activity` (nothing moved; the ROI falls back to the whole frame), `error`; plus a
  `checked` column that is `True` once a person confirmed or corrected the video.

The command prints how many videos were done or taken from the cache, and lists
the low-confidence waterlines and the duration/fps flags. Existing results are reused
unless `--force` is given, that video's entry in `overrides.yaml` changed, or a scene
setting that affects the result was changed (then it is redone automatically).

**Checking the QA image.** `<video>_qa.png` shows the background (enlarged for small
videos) with a text header giving the file, method, flags, waterline row, confidence,
and ROI. On the image:

- **blue line** = detected waterline; it should lie on the water surface;
- **green box** = ROI; it should enclose all the water the fish can reach, from just
  above the surface to the beaker bottom, and should **not** include the reflection
  below the beaker;
- **faint red** = where the fish was seen moving. Red outside the water usually means
  a reflection or lighting change; no red at all means the fish hardly moved.

If the background still shows a fish-shaped ghost, the fish stayed in one place for
more than half of the video. Look at the QA images of every flagged video first.

**3. Review and correct scenes in the browser (recommended)**

```bash
python -m fishbehavior scene-review                 # all matched subjects
python -m fishbehavior scene-review --subjects 347  # only some
```

Beakers, zoom, cropping and camera height differ between recordings, so every automatic
result should be looked at once. `scene-review` first brings the scene results up to
date, then opens a review page in your browser. The page runs only on this computer
(127.0.0.1) and uses no internet. It shows one video at a time:

- the empty-beaker background, or **real color frames** from across the video (slider),
  with a **red tint where the fish swam**, which marks the water it can reach;
- the waterline (blue) and ROI (green), dashed while automatic and solid once set by hand;
- the cursor position in **original video pixels** plus a magnifier, so you can read
  the exact row of the surface without any zoom arithmetic;
- hints (listed first) for results that look wrong: a weak waterline edge, a waterline
  at the very top of the frame, an ROI reaching the top edge, or a fish that hardly moved.

| Action | How |
|---|---|
| Set the waterline | click on the water surface; `↑`/`↓` nudge 1 px (`Shift`: 5 px) |
| Set the ROI | drag a box around the water the fish can reach (not the reflection below the beaker) |
| Accept the result | `Enter` (or "Looks right"), which also jumps to the next unchecked video |
| Undo your changes for a video | `Z` (back to the automatic values) |
| Real frames / background | `V` (press again for the next frame, or use the slider), `B` |
| Fix a wrong heatmap | opacity and "hide weak" sliders; `E` eraser to paint away red on glare, light or reflections |
| ROI from the heatmap | `F` fits the ROI to the red that is left (padded, top above the waterline) |
| Browse / hide the red tint | `←`/`→`, `M` |

Erasing only changes the heatmap on the page, to guide "Fit ROI"; what is saved is
the resulting ROI. Every change is saved immediately to `overrides.yaml`. When you press **Finish** (or
Ctrl+C in the terminal), the corrected videos are updated and `video_check.csv` shows
which videos are `checked`. The page can be reopened any time; it continues where you
left off. On Google Colab, or wherever a local page cannot be opened, use
`scene-review --no-serve`: it writes `scene/review.html`, whose **Download
overrides.yaml** button gives the file to copy into the scene folder before running
`scene` again.

**Correcting a result by hand (`overrides.yaml`).** Create `<FISH_OUTPUT_DIR>/scene/overrides.yaml`
(it stays in the git-ignored output folder) with an entry per video *file name*.
Values are pixels of the original video (the QA header shows the current values; divide
positions measured on an enlarged QA image by the zoom factor):

```yaml
F_0042.mp4:
  waterline_y: 118            # row of the water surface
  roi: [30, 95, 290, 215]     # x0, y0, x1, y1 (x1/y1 exclusive)
M_0012a.mp4:
  waterline_y: 121            # only the waterline; the ROI top is re-derived from it
M_0012b.mp4:
  checked: true               # looked at, the automatic result is fine
```

Then run `scene` again. Videos whose override changed are redone automatically, and
their JSON records `method: override`. A `roi` override is used as-is. A
`waterline_y` override alone still raises the automatic ROI top to just above it.

The
full pipeline will clean and validate the trial database, match each trial to its
video, extract kinematic features, discover behavior states, train and evaluate
the classifier, and run anomaly detection.

### Generate a Report

The future reporting step will combine the behavior log, classifier probabilities,
feature evidence, and anomaly result into an offline ethogram-style report. The
report schema and language-model provider are still open decisions.

## Expected Outputs

The planned Phase 1 outputs are:

- cleaned tabular data and a database/video validation report;
- per-frame and per-session tracking features;
- a timestamped behavior log and behavior-state summaries;
- classifier probabilities over compound-and-dose labels;
- an anomaly or novelty flag;
- feature-based evidence and a natural-language reasoning summary; and
- an offline, shareable report with researcher-query support.

Live narration during an active session is a Phase 2 requirement. Phase 1 may
produce timestamped narration from recorded sessions after the reporting design is
implemented.

## Development Phases

### Phase 1: Offline analysis of recorded video

1. Clean the database, apply missing-data rules, and pool small classes.
2. Build computer-vision tracking across the 353 videos.
3. Discover and validate behavior states with the Pharmacy Department.
4. Train and evaluate the supervised compound classifier, starting with tabular
	 features and later adding video-derived features.
5. Build anomaly and novelty detection.
6. Build natural-language reporting for offline reports and queries.
7. Add the minimal backend and frontend needed to integrate these modules.

### Phase 2: Real-time operation

8. Integrate a live camera and streaming inference.
9. Add real-time natural-language narration.

Phase 1 is the realistic scope for the current project timeline. Phase 2 is a
stretch goal or subsequent milestone.

## Known Decisions and Limitations

### Data preparation

- Rows with a blank compound field are excluded.
- A `-` in a top-half or bottom-half field is interpreted as zero time in that
	zone.
- Trials missing all eight movement columns are treated as untracked and masked
	out of training rather than imputed. The scope identifies 104 such trials.
- `Body Tissue` is excluded because it is empty for all real trials.
- `N/A` and other text in numeric columns is treated as missing; the raw UV exposure
  text (for example `10*`) is kept in `uv_min_raw`.
- Workbook rows that share a subject number are one recording split into parts and
  are joined into one subject; conflicting values between parts are reported.

### Modeling

- No frame-level behavior annotations exist, so Phase 1 behavior-state extraction
	uses tracking, rules, and unsupervised methods rather than a supervised video
	classifier.
- Compound and dose are separate label dimensions; each compound-and-dose pair is
	treated as a classification class.
- Four very small classes are pooled into adjacent doses of the same compound,
	producing a 31-class working label set pending the decision for HS-1-51 @ 0.03.
- Cross-validation is planned instead of a single held-out test split because
	several classes remain small.

### Open decisions

- HS-1-51 @ 0.03 has one trial and cannot be pooled with an adjacent dose. It may
	be excluded, retained as a singleton, or handled through another approved rule.
- The meaning of probability examples such as `60% MDMA, 30% DOB, 10% Veh` must
	be confirmed as mutually exclusive class probabilities or a mixture/multi-label
	result.
- Observation windows, confidence thresholds, calibration, stopping rules, report
	schema, and video/database mismatch behavior remain to be defined.
- The reporting layer may use a hosted API or a self-hosted language model. This
	choice depends on privacy requirements, network access, maintenance, and GPU
	availability.

## Contribution Workflow

Use a feature branch for every change and submit work through a pull request. Do
not commit or push directly to `master`.

The complete workflow, including branch creation, review, commits, pushes, PRs,
branch synchronization, and cleanup, is documented in
[docs/how-to-start.md](docs/how-to-start.md).

## Data Handling and Privacy

Never commit the trial workbook, video recordings, generated reports, model
artifacts, credentials, API keys, or other restricted research data. Keep local
data paths and secrets in ignored configuration or environment variables.

A hosted language model may simplify deployment but sends report inputs outside
the local infrastructure and requires approval under the project's data-handling
requirements. A self-hosted model keeps processing local but adds GPU, setup, and
maintenance costs. This decision must be resolved before production reporting is
enabled.