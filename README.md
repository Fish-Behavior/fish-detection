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

The behavior-labeling pipeline (`fishbehavior` package) is complete: workbook catalog,
scene setup, tracking, movement features, behavior labeling, reference digitizing, sanity
checks and calibration, dataset export and plots, plus one `all` command that runs everything
and a Google Colab notebook. **Start with the user guide:
[docs/behavior-labeling.md](docs/behavior-labeling.md)** (setup on every platform, every
command, how the states are decided, calibration and its limits, troubleshooting). The
classifier, backend, and frontend are not yet implemented.

## Project Structure

```text
fish-detection/
├── .env.example              # template for your local .env (data paths); copy it, never commit .env
├── .gitignore                # keeps data, outputs, and .env out of Git
├── pyproject.toml            # package definition and dependencies
├── requirements.txt          # one-line install: the package + developer tools
├── docs/
│   ├── behavior-labeling.md  # user guide: setup, every command, states, calibration, troubleshooting
│   ├── how-to-start.md       # contribution workflow
│   └── zebrafish_drug_detection_scope.md  # project scope
├── notebooks/
│   └── colab_quickstart.ipynb  # Google Colab: mount Drive, install, run `all` (saved without outputs)
├── src/fishbehavior/         # the pipeline package
│   ├── __main__.py           # enables `python -m fishbehavior <command>`
│   ├── calibrate.py          # tunes the labeling thresholds against the reference timelines (random search, K-fold)
│   ├── catalog.py            # cleans the trial workbook and matches each trial to its video(s)
│   ├── cli.py                # command-line entry point; one sub-command per pipeline step
│   ├── config.py             # reads paths from .env / environment and parameters from YAML
│   ├── default_config.yaml   # default, data-independent parameters
│   ├── export.py             # final datasets: segments, per-second labels, one row per subject, training windows, clips
│   ├── features.py           # per subject: speed, turning, depth, posture per frame and per time bin; endpoints
│   ├── labeling.py           # per subject: behavior state of every time bin (rules + pooled swim model), segments
│   ├── parallel.py           # runs per-video / per-subject jobs in FISH_WORKERS processes, with a progress bar
│   ├── plots.py              # ethograms per group, bar charts per state, overlay review videos
│   ├── priors.py             # sanity checks: video labels vs workbook (NTT hints) and reference group means
│   ├── reference.py          # digitizes the reference ethogram figures into approximate per-subject timelines
│   ├── review.py             # scene-review: local browser page to check/correct waterlines and ROIs
│   ├── review_page.html      # the review page template (self-contained, no external files)
│   ├── roi.py                # per video: empty-beaker background, waterline, fish region (ROI), QA image
│   ├── tracking.py           # per subject: fish position, outline size and tilt in every frame (parts joined)
│   └── video.py              # video timing (fps, frames, duration) and frame reading
└── tests/                    # automated tests (synthetic data only, never real trials)
    ├── synthetic.py          # draws synthetic beaker/fish videos for the tests
    └── test_all.py           # `all` end to end on a tiny synthetic project (workbook, 20 s video, reference PDF)

Everything the pipeline generates is written to the output folder (`outputs/` by
default), which is ignored by Git.

## Data Sources and Access

### Trial database

The planned tabular source is the trial workbook (`FISH_DB_PATH`), which contains 720 rows:

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

**Everything at once:**

```bash
python -m fishbehavior all                        # every step, every subject
python -m fishbehavior all --subjects 42          # try one subject first
python -m fishbehavior all --skip-reference       # without reference / priors / calibration
```

`all` runs validate → scene → track → features → label → (reference → priors → calibrate →
label again, when `FISH_REFERENCE_PDF` is set and `mapping.yaml` is filled in) → export →
plot. It skips finished work (a second run only redoes what changed), stops with a clear
message at the first failing step, and prints the seconds per step. On Google Colab use
[notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb). The steps below can also
be run one at a time (`python -m fishbehavior --help` lists them all).

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

- the **background**: for each pixel, a high brightness percentile of the samples
  (`scene.background_percentile`, default 90). The fish is darker than the water, so
  each pixel's brighter samples are the ones without the fish, and what remains is the
  empty beaker. This works even where a large fish in shallow water sits in many of the
  samples; a median (50) would keep a faint "ghost" fish there. For a fish lighter than
  its background, set it to 50;
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
| Set the ROI (the crop) | drag a box around the water the fish can reach (not the reflection below the beaker) |
| Adjust the crop | `C` crop mode: drag edges/corners to resize, inside to move; outside is darkened. Or type exact x0, y0, x1, y1 (and the waterline) under "Current values" |
| Accept the result | `Enter` (or "Looks right"), which also jumps to the next unchecked video |
| Undo your changes for a video | `Z` (back to the automatic values) |
| Real frames / background | `V` (press again for the next frame, or use the slider), `B` |
| Fix a wrong heatmap | opacity and "hide weak" sliders; `E` eraser to remove red on glare, light or reflections; `P` paint (orange) to add water the fish can reach but that has no red |
| ROI from the heatmap | `F` fits the ROI to the red that is left (padded, top above the waterline) |
| Browse / hide the red tint | `←`/`→`, `M` |

Painting and erasing only change the heatmap on the page, to guide "Fit ROI"; what
is saved is the resulting ROI. Every change is saved immediately to `overrides.yaml`. When you press **Finish** (or
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

**4. Fish tracking**

```bash
python -m fishbehavior track                     # every subject with video_status = matched
python -m fishbehavior track --subjects 42 F_0043  # only these
python -m fishbehavior track --force             # redo subjects that already have results
```

Needs `trials.csv` from `validate`. A video without a scene result gets one first
(the same as running `scene`), so check the scene QA images or run `scene-review`
before tracking many subjects. For every frame (`tracking.frame_stride`, default 1 =
every frame) the fish is found by:

1. grayscale, shrunk by `tracking.scale` (default 1.0; 0.5 = half size, faster), and
   cut to the scene ROI (nothing outside it can be the fish, e.g. the reflection below
   the beaker);
2. a Gaussian blur, then the difference from the (equally blurred) empty-beaker
   background, after removing the frame's overall brightness change (flicker, auto
   exposure);
3. pixels **darker** than the background by more than `tracking.diff_threshold`
   (default 18), cleaned by a morphological open/close, grouped into blobs. Lighter
   pixels (glare, highlights) are ignored; set `tracking.polarity: both` to count them
   too;
4. the fish = the blob that **overlaps last frame's fish**. From one frame to the next
   the fish always overlaps itself, while its mirror images in the glass (under the
   surface, on the beaker bottom), ripples and shadows lie next to it. So the track
   stays on a fish that briefly looks small, for example when it turns toward the
   camera or its see-through body splits into pieces, even if a mirror image is bigger
   at that moment. A blob more than `tracking.switch_area_ratio` (default 1.5) times
   bigger does take over, so the track cannot stay stuck on a mirror image. When nothing
   overlaps (after a gap or a very fast move), the fish is the largest blob within
   `tracking.max_jump_bl` body lengths (default 2) of its last position, or else simply
   the largest blob. Blobs smaller than `tracking.min_area_fraction` of the frame are
   ignored;
5. the **head** = the centre of the darkest part of the fish, which is the eye (much
   darker than the see-through body). It tells which end of the body is the front, so
   the tilt becomes a **pitch**: nose up (+) or down (−), whichever way the fish faces.
   Useful for Listing and Surface Breach. No head is recorded when no part of the fish
   is at least `tracking.head_min_contrast` gray levels darker than the rest. The pitch
   is left empty when the head is too close to the body centre to tell front from back
   (`tracking.head_min_offset_fraction`), e.g. when the fish faces the camera.

Kernel sizes and the minimum blob area are fractions of the frame size, so the same
settings work at any resolution; all times are `frame / fps` from the file.

Missing stretches up to `tracking.max_gap_s` (default 0.5 s) between two detections
are filled by linear interpolation (`interpolated = True`). Longer ones stay empty and
count as untracked seconds. Parts of a split recording are joined into one timeline:
part b starts at part a's duration.

Subjects run in `FISH_WORKERS` parallel processes, with a progress bar. Results go to
`<FISH_OUTPUT_DIR>/tracks/`:

- `<subject>.csv.gz`: one row per analysed frame (columns below);
- `<subject>_track_qa.png`: the track drawn over the background, colored by time
  (dark purple = start, yellow = end), with the ROI in green and the waterline in blue.
  Breaks in the line are untracked stretches. A split recording is drawn for part 1 only,
  on part 1's background;
- `<subject>_track.json`: the summary, which file each `part` number is and its start
  time on the joined timeline, and the settings used;
- `summary.csv`: one row per tracked subject: `subject_id, frames, detected_pct,
  interpolated_pct, untracked_s, median_body_length_px, mean_n_blobs, head_pct,
  pitch_pct` (the last two: share of detected frames with a head, and with a pitch).

A subject is skipped when its track exists, unless `--force` is given, a `tracking`
setting changed, or its scene ROI changed (for example after `scene-review`). Then it
is redone automatically.

Columns of `<subject>.csv.gz` (positions and sizes in **original video pixels**, whatever
`tracking.scale` is; y grows downward):

| Column | Meaning |
|---|---|
| `subject_id` | subject, zero-padded text (`0042`) |
| `part` | which file of the subject: 1 = first in `video_paths` (part a), 2 = part b, ... |
| `part_frame` | frame number inside that file (for cutting clips from the right file) |
| `frame` | frame number on the joined timeline (continuous across parts) |
| `time_s` | seconds on the joined timeline (part b starts at part a's duration) |
| `detected` | the fish was found in this frame |
| `x`, `y` | centroid of the detected fish outline (mid-body when the whole fish is seen; it moves toward the head when only the dark front is detected) |
| `top_y`, `bottom_y` | highest and lowest row of the fish outline (for surface breaches) |
| `area` | outline area, px² |
| `major_axis`, `minor_axis` | length and height of an ellipse fitted to the outline (major ≈ body length) |
| `angle_deg` | tilt of the long axis, −90 to 90: 0 = horizontal, positive = right end higher on screen (does not say which end is the head) |
| `head_x`, `head_y` | the head: centre of the darkest part of the fish (the eye); empty when there is no clear eye |
| `pitch_deg` | nose direction, −90 to 90: positive = nose up, whichever way the fish faces; empty when the front end is unclear |
| `n_blobs` | how many large-enough blobs were in the ROI (more than 1 = something else moved too) |
| `interpolated` | filled in over a short gap (`detected` is False) |

Frames that are neither detected nor interpolated have empty measurements.

**5. Movement features**

```bash
python -m fishbehavior features                     # every matched subject with a track
python -m fishbehavior features --subjects 42 F_0043  # only these
python -m fishbehavior features --force             # redo subjects that already have results
```

Needs the tracks from `track` and the scene results (waterline, ROI). Everything is in
**body lengths (BL)** and **seconds**: BL = the subject's median fitted `major_axis` over
detected frames, and the time between rows comes from the track's `time_s`, so the same
settings work at any frame rate, resolution or `tracking.frame_stride`. x and y are
smoothed with a Savitzky-Golay filter over `features.smooth_s` seconds (default 0.2)
before speeds are computed; untracked frames stay empty. All `features` settings are
starting values, to be calibrated in a later step.

Results go to `<FISH_OUTPUT_DIR>/features/`:

- `<subject>_frames.csv.gz`: one row per track row, at the full frame rate (brief
  events stay visible): `subject_id, part, frame, time_s, tracked` and

  | Column | Meaning |
  |---|---|
  | `vx`, `vy`, `speed_bl_s` | velocity and speed, BL/s (y grows downward) |
  | `accel_bl_s2`, `jerk_bl_s3` | change of speed per second, and of that per second |
  | `heading_deg` | movement direction, −180 to 180 |
  | `turn_deg` | heading change since the previous row, −180 to 180; empty unless both rows move faster than `features.min_speed_for_heading_bl_s` |
  | `angular_velocity_deg_s` | `turn_deg` per second |
  | `meander_deg_per_bl` | \|`turn_deg`\| / BL moved since the previous row |
  | `depth_bl` | (y − waterline) / BL: 0 = at the waterline, positive = below it |
  | `at_surface` | top of the fish outline within `features.surface_margin_bl` BL of the waterline (or above it) |
  | `pitch_deg` | head–tail line from the track, −90 to 90: positive = nose up; empty when the front end is unclear |
  | `nose_up_at_surface` | the head (eye) is within `features.surface_margin_bl` BL of the waterline and `pitch_deg` ≥ `features.nose_up_threshold_deg` (25) |
  | `tilt_deg` | body axis angle from horizontal, 0–90 |
  | `side_on` | the eye is at least `features.side_on_min_head_offset_bl` (0.3) BL from the body centre, so the fish is seen from the side. A fish facing the camera has its eyes mid-blob and a tall outline whose angle is not a tilt |
  | `aspect` | `major_axis` / `minor_axis` (low = fish seen end-on or bent) |
  | `area_ratio` | outline area / the subject's median area |

- `<subject>_bins.csv.gz`: one row per `features.bin_s` bin (default 1 s) from t = 0,
  `ceil(duration / bin_s)` rows (a bin without rows has `n_frames` 0):
  `bin, t_start_s, n_frames, tracked_fraction`, speed `mean / median / max / cv`,
  mean \|accel\|, mean \|jerk\|, `turn_rate_var_deg2_s2` (variance of the angular velocity: turning variability per second, the same at any frame rate), mean \|angular velocity\|,
  `meander_deg_per_bl` (total \|turn\| / total distance), `distance_bl`,
  `surface_fraction`, `nose_up_surface_fraction` (share of rows with `nose_up_at_surface`),
  `depth_bl_min`, `tilt_median_deg`, `tilt_fraction` (share of rows that are `side_on` and
  tilted more than `features.tilt_threshold_deg`), `aspect_median`, `area_ratio_median`;
- `endpoints.csv`: one row per subject over the whole exposure video: `body_length_px,
  duration_s, tracked_s, distance_bl, mean_velocity_bl_s, highly_mobile_s` (faster than
  `features.high_mobility_bl_s`), `immobile_s` (slower than `features.immobile_bl_s`),
  `top_half_pct` (share of tracked time in the top half: from the waterline to halfway
  down to the ROI bottom), `latency_top_half_s` (first time there; empty = never),
  `meander_deg_per_bl, mean_angular_velocity_deg_s`;
- `<subject>_features.json`: the endpoint row and the settings used.

A subject is skipped when its results exist, unless `--force` is given, a `features`
setting changed, or its track is newer than the results.

**Which features can be trusted?** There are no hand labels, so `features-qa` checks
whether each bin feature measures the fish or tracking noise:

```bash
python -m fishbehavior features-qa                     # every matched subject with a track
python -m fishbehavior features-qa --subjects 42 F_0043
```

For each subject it recomputes the bins from the track twice more (it writes no
features): with every second frame (half the frame rate) and with double
`features.smooth_s`. A real movement keeps its value; jitter of the tracked body centre
does not. A feature counts as `stable` when its per-bin values at half the frame rate
correlate with the full-rate ones by at least `features.qa_min_r` (0.8) and its mean
moves by at most `features.qa_max_change` (25%) in both checks. It prints, per feature,
how many subjects pass and the median numbers, and writes `features/qa.csv` (the
subjects of this run): `subject_id, feature, r_half_rate, change_half_rate,
change_smooth_2x, stable, jitter_bl` (RMS distance between the raw and smoothed body
centre, BL) and `body_length_cv` (how much the fitted length varies over the video).
An empty `r_half_rate` means the feature did not vary (e.g. never at the surface), so
it cannot fail.

**6. Behavior labels**

```bash
python -m fishbehavior label                     # every matched subject with features
python -m fishbehavior label --subjects 42 F_0043  # only these (reuses the saved swim model)
python -m fishbehavior label --force             # redo all subjects and refit the swim model
```

Needs `features` and `track`. Every time bin (`features.bin_s`, default 1 s) gets one of
the five behavior states, or `untracked`. The rules are checked in this order and the
first one that fits wins (all numbers are `labeling:` settings, starting values until
calibration):

| # | Label | In plain words | Rule on the bin |
|---|---|---|---|
| 0 | `untracked` | not a behavior: the fish was not seen well enough to judge | seen in less than `min_tracked_fraction` (0.5) of the frames |
| 1 | `surface_breach` | the fish pushes its head up to the surface, body angled nose-up | `nose_up_at_surface` (head at the waterline, head–tail line ≥ 25° nose-up) in at least `surface_breach_fraction` (0.3) of the frames |
| 2 | `lorr` | listing / loss of righting: hangs steeply (e.g. head-down) and barely moves | seen side-on and tilted more than `features.tilt_threshold_deg` in at least `lorr_tilt_fraction` (0.5) of the frames, median speed below `lorr_max_speed_bl_s` (0.5 BL/s), for at least `lorr_min_s` (3 s) in a row |
| 3 | `freeze_drift` | motionless, or only drifting | median speed below `freeze_speed_bl_s` (0.1 BL/s) for at least `freeze_min_s` (2 s) in a row |
| 4a | `controlled_swim` | normal swimming: steady speed, smooth path, or slow cruising | any other bin with mean speed below `swim_min_speed_bl_s` (0.5 BL/s), or one the swim model calls smooth |
| 4b | `erratic` | darting and zig-zagging: bursts of speed, sharp and frequent turns | any other bin the swim model calls erratic |

Why the surface and tilt rules look at the head: in shallow dishes the water is often less
than one body length deep, so the top of the fish is near the waterline almost all the
time. A breach is therefore the head itself at the surface with the body angled nose-up.
A resting fish facing the camera looks like a tall blob whose fitted angle is steep, so
tilt only counts when the eye is near one end of the body (seen side-on).

Bins slower than `swim_min_speed_bl_s` have no turning measure, so they would form a
"hovering" group of their own instead of the smooth-versus-erratic split. They are
`controlled_swim` and stay out of the model.

The swim model sorts the remaining faster bins into two groups using four features: speed
variation (`speed_cv`), jerkiness (mean \|jerk\|), turning variability (turn-rate
variance) and `meander` (the last three log-compressed, all standardized). It is fitted
on **all subjects together**, so "erratic" means the same for every fish, and the group
with the higher values is `erratic`. `labeling.swim_split: gmm` (default) is a Gaussian
mixture that judges each bin on its own; `hmm` is a hidden Markov model that also favors
staying in the same state, giving smoother timelines. The model is saved as
`labels/swim_model.json` and reused when labeling single subjects; it is refitted when
it is missing, when any `labeling` setting changes, when any subject's features are newer
than it, or with `--force` (without `--subjects`).

After labeling, a bout shorter than `labeling.min_bout_s` (per label, default 1 s) is
merged into its longer neighbor. `untracked` is never merged away, so tracking gaps stay
visible. Each bin has a `confidence` from 0 to 1. For `untracked`, `surface_breach` and
`lorr` it is the share of frames that met the rule. For `freeze_drift` it is how far
the median speed is below the threshold (1 = not moving), and for slow `controlled_swim`
how far the mean speed is below `swim_min_speed_bl_s`. For the swim states it is the
model's probability of the chosen state. Bins merged by the cleanup get 0.

If `<FISH_OUTPUT_DIR>/calibration/calibrated.yaml` exists, its `labeling:` values replace
the configured ones (the command prints that calibrated values are in use).

Results go to `<FISH_OUTPUT_DIR>/labels/`:

- `<subject>_bins.csv`: the feature bins plus `label` and `confidence`;
- `<subject>_segments.csv` and `segments.csv` (all subjects): runs of the same label,
  `subject_id, label, start_s, end_s, duration_s, start_frame, end_frame, part,
  mean_confidence`. Times are seconds on the joined timeline, and the segments tile the
  whole video without gaps. `start_frame` / `end_frame` (inclusive) are frame numbers
  inside that part's own video file. A segment that crosses from one part file to the
  next is split there;
- `summary.csv`: one row per subject: `duration_s`, then `<label>_s` and `<label>_pct`
  (seconds and % of the video) for every label;
- `swim_model.json`: the swim model and the settings it was made with.

A subject is skipped when its results exist, unless `--force` is given, its features
are newer, or the swim model was refitted.

**7. Reference ethograms (for tuning)**

```bash
python -m fishbehavior reference            # extract, digitize, compare
python -m fishbehavior reference --force    # re-extract the figures and redo everything
```

There are no hand labels, so the labeling is tuned against the published reference
figures in `FISH_REFERENCE_PDF`. This step turns them into approximate second-by-second
timelines. It needs only the PDF (not the videos). Everything it writes stays in the
git-ignored `<FISH_OUTPUT_DIR>/reference/`.

The ethogram pages (`reference.ethogram_pages`, default 16 and 17) each hold one picture
with five stacked panels, one per group, and one thin row per subject. The x axis runs
0 to `reference.axis_seconds` (1200). The step:

1. saves the largest image of each ethogram page as `page16.png` / `page17.png`, and the
   bar-chart page (`reference.barchart_page`, 18) rendered as `page18.png`;
2. finds the plot: the columns and rows where the legend colors dominate. Each block of
   colored rows is a panel, and panels are numbered from the top. All panels of a page share
   one time axis, so a row whose data stops early (white) is still placed right;
3. splits each panel into as many equal rows as its `subject_ids` in `mapping.yaml`, reads
   the middle line of each row pixel by pixel, and keeps the most common color per second.

**First run: fill in `mapping.yaml`.** The subject labels on the left of the figures
overlap and cannot be read, so the first run writes a template and stops (exit code 1):

```yaml
panels:
- page: 16
  panel: 1
  rows: 12          # estimated from the image; count the rows in page16.png and correct it if needed
  group: ''         # fill in, e.g. VEH
  subject_ids: []   # fill in, TOP to BOTTOM, exactly `rows` of them
  skip: false       # true for a panel that repeats another one
```

For each panel:

- `group`: the panel title (e.g. `VEH`, `COMPOUND_A 30uM`). It names the group in the outputs.
- `subject_ids`: the subjects of that group, **top row first**. Take the group's subjects
  from the workbook (or `catalog/trials.csv`: same compound and dose). The figures were
  made with R/ggplot, which puts a group's first subject at the **bottom**. So the list from
  top to bottom is usually the subject numbers in **descending** order, e.g.
  `[F_0048, F_0045, F_0042, ...]`. `42`, `0042` and `F_0042` all work. Check the order
  against the top and bottom labels, which are usually readable because nothing overlaps them.
- `rows`: the row count estimated from where the colors change between rows. The number
  of `subject_ids` must equal it. If they differ, the step stops and names the panel, for
  example `page 16 panel 2: 9 subject_ids but rows: 10`. Then either a subject is missing from
  your list or the estimate is off; count the rows in the zoomed `page<N>.png` and fix
  whichever is wrong.
- `skip: true` for a repeated panel, e.g. the control group drawn on both pages. It needs no
  group or subject_ids and is not digitized. Listing a subject in two panels is an error, so it
  cannot be counted twice.

Run `reference` again after editing. Digitizing is redone automatically when
`mapping.yaml` is newer than the results. Use `--force` after changing a `reference:`
setting or the PDF.

**Outputs** (`<FISH_OUTPUT_DIR>/reference/`):

- `timelines.csv`: `subject_id, group, second, label`, one row per subject and second.
  `label` is one of the five states, `no_data` (white: the subject has no data there), or
  `unknown` (no pixel of that second matched a legend color);
- `group_means.csv`: per group, `n_subjects` and the mean seconds of every label;
- `digitize_check.png`: **look at this first.** Every digitized panel is drawn as the
  original figure crop (left) next to the redrawn ethogram (right), in the style of the
  reference: subject IDs down the side, time 0–1200 s across, colored by state, one titled
  panel per group, with the legend at the bottom (black = unknown). Rows line up across both
  sides, so a row shifted by one subject or a wrong row count is obvious;
- `slide18_values.yaml` (optional): a template of group × state, all `null`. Fill in the
  mean seconds you read **by eye** from the bar charts in `page18.png` (they are not stored
  as numbers in the PDF). The next run compares them with `group_means.csv` and writes
  `slide18_check.csv` (`group, state, slide18_s, digitized_s, difference_s`), and prints the
  median and largest difference. This checks the digitizer; it is not an input to it.

How colors are read (all in `reference:`): each pixel gets the nearest `palette` color
by CIELAB distance, with lightness differences weighted by `lightness_weight` (3).
JPEG keeps brightness sharp but smears color over about 2 px, so a thin gray row under a
pink one turns pinkish while its brightness stays gray. A pixel farther than
`max_color_distance` from its nearest color is unknown. Pink is nearly a red + white mix,
so the blurred edge of every red bout looks pinkish. A pink stretch therefore counts as
LORR only if at least one of its pixels is within `seed_color_distance.lorr` (15) of the
palette pink; otherwise it takes its next-nearest color.

**8. Sanity checks and calibration**

```bash
python -m fishbehavior priors      # video labels vs workbook and reference group means
python -m fishbehavior calibrate   # tune the labeling thresholds, writes calibrated.yaml
python -m fishbehavior label       # relabel with the calibrated values
```

`priors` needs `label`, `features` and (for the group checks) `reference`. It writes
`<FISH_OUTPUT_DIR>/calibration/priors_report.md` with:

1. **Video vs workbook.** Spearman correlation of freeze_drift seconds, distance (BL) and
   surface_breach seconds with the workbook's `tdm_full`, `velocity_full` and `time_top_s`,
   over the NTT-tracked subjects. The workbook is the Novel Tank Test, a **different**
   10-minute session, so this is a hint only: a loose relation is expected. Subjects far off
   the rank trend of a matching pair (|z| > `priors.flag_z`, 2) are listed. Look at their
   track QA image first: it is often a tracking failure.
2. **Group means.** Video-derived mean seconds per state next to `reference/group_means.csv`.
3. **Direction checks** from `<FISH_OUTPUT_DIR>/reference/expectations.yaml`, a local file
   (group names stay out of Git). The first run writes a template:

   ```yaml
   groups:            # reference group -> its workbook subjects (trials.csv columns)
     VEH: {compound: '', concentration: ''}           # fill in by hand
     COMPOUND_A 30uM: {compound: COMPOUND_A, concentration: '0.03'}  # filled from mapping.yaml
   expectations:
   - {group: COMPOUND_A 30uM, state: lorr, relation: greater_than, than: VEH}
   ```

   A group is filled in automatically when `mapping.yaml` lists some of its subjects. The
   others (e.g. a group whose rows are all placeholders) need `compound` and `concentration`
   exactly as in `catalog/trials.csv`. `relation` is `greater_than` or `less_than`. Each
   expectation is tested on the reference means and on the video means.

`calibrate` needs `label` (for `labels/swim_model.json`), `features` and `reference`. It uses
the subjects that have both feature bins and a reference timeline. It tries
`calibration.n_trials` (200) random threshold sets from `calibration.search`, with a fixed
`seed`. The first set is always the current settings. Each set relabels the bins with the
pure labeling functions (no video work, about a minute for 50 subjects) and is scored:

    score = macro-F1 per second vs the reference + group_weight (0.5) x group agreement

Group agreement is 1 minus the total-variation distance between our time budget and the
reference one per group (0 to 1). It forgives the small timing offsets of the digitized
rows, which the per-second F1 counts as errors. The search space is a list of values to
try, or `{low, high}` for a uniform range. `min_bout_s` is one value used for every state.
Set a key to `null` in your `FISH_CONFIG` file to leave it untuned.

To show overfitting, the choice is repeated with `calibration.folds` (5) **K-fold by
subject**. Values are chosen on four folds and scored on the fifth. Held-out F1 far below
the in-sample F1 means the search fits noise. Held-out tuned at or below held-out current
means calibration does not help.

Outputs (`<FISH_OUTPUT_DIR>/calibration/`, git-ignored):

- `calibrated.yaml`: only the tuned `labeling:` keys. `label` merges it over the settings
  (run `label` again afterwards; delete the file to go back to the defaults);
- `calibration_report.md`: before/after score, the tuned values, F1 per state, the
  confusion matrix (seconds, reference x ours), the held-out check and the number of
  subjects used, with a warning below `calibration.min_subjects` (30).

Limits: the reference is digitized from figures, so it is approximate. Calibration can only
move thresholds; it cannot create a state that the features do not see (check the per-state
F1 and the confusion matrix). With few subjects, or subjects from only a few groups, the
values fit those groups.

**9. Export the datasets**

```bash
python -m fishbehavior export                          # all tables, every subject
python -m fishbehavior export --clips --subjects 42    # + fish-centered crops for these subjects
python -m fishbehavior export --clips                  # crops for everyone (slow: reads every video)
```

Needs `label` (run it again after `calibrate`), `features` and `track`. The tables always
cover every subject. `--subjects` and `--force` only apply to `--clips`, which skips
subjects whose clip file already exists. Everything goes to `<FISH_OUTPUT_DIR>/datasets/`:

- `segments.csv`: every labeled segment (see step 6) plus `sex, compound,
  concentration_raw, concentration_mm` from `trials.csv`.
- `per_second_labels.csv`: `subject_id, second, label, confidence` plus the key bin
  features `tracked_fraction, speed_mean_bl_s, speed_median_bl_s, speed_cv,
  jerk_abs_mean_bl_s3, turn_rate_var_deg2_s2, meander_deg_per_bl, depth_bl_min,
  nose_up_surface_fraction, tilt_fraction` of the bin that covers that second.
- `behavior_dataset.csv` and `behavior_dataset.xlsx`: **one row per subject**. The xlsx has a
  second sheet, `column_guide`, that explains every column in plain words. Columns, in order:
  - all cleaned workbook columns of `trials.csv` (NTT values, compound, dose, video status...);
  - per state (`controlled_swim, erratic, freeze_drift, lorr, surface_breach`):
    `<state>_s` (seconds), `<state>_pct` (% of the recording), `<state>_bouts`,
    `<state>_mean_bout_s`, `<state>_latency_s` (seconds until the first bout; empty if never).
    A bout cut in two by a part boundary counts once;
  - `<from>_to_<to>`: how often a bout of one label is directly followed by another, for
    every pair of the five states and `untracked`;
  - `untracked_s`;
  - the video endpoints from `features/endpoints.csv` (`body_length_px, duration_s,
    tracked_s, distance_bl, mean_velocity_bl_s, highly_mobile_s, immobile_s, top_half_pct,
    latency_top_half_s, meander_deg_per_bl, mean_angular_velocity_deg_s`).

  Subjects without a usable video keep their workbook values; their behavior and endpoint
  columns are empty. The state seconds plus `untracked_s` add up to the recording length.
- `behavior_windows.csv`: the training manifest for a behavior classifier, one row per
  `export.window_s` (1 s) window: `subject_id, video_path, part, part_start_frame,
  part_end_frame, start_s, end_s, label, confidence, fold, use_for_training`. `video_path`
  is the **part file** the window lies in, and the frame numbers count inside that file
  (inclusive), so a clip can be cut straight from it. A window that crosses the end of a part
  is split in two. `label` is the most common label over the window's frames. `fold` (0 to
  `export.folds` - 1) is assigned by subject, so one fish is never in both train and test.
  `use_for_training` is False for untracked windows and for confidence below
  `export.min_confidence` (0.5; bins merged by the bout cleanup have confidence 0).
- `clips/<subject_id>.npz` (with `--clips`): `frames` (windows x `export.clip_frames` (8)
  x `crop_size` x `crop_size`, gray uint8), `window` (row number among that subject's
  windows) and `label`. Each crop is a square of `export.crop_bl` (1.5) body lengths
  centered on the tracked fish, resized to `export.crop_size` (64) px. Frames where the fish
  was not found stay black. The frames are spread evenly over each window; all frames of a
  20-minute video would be about 150 MB per subject. Even with 8 frames per window, expect
  roughly 15 MB per subject.

**10. Plots**

```bash
python -m fishbehavior plot                                     # ethograms for every group + bar charts
python -m fishbehavior plot --subjects 42                       # only the ethogram of subject 42's group
python -m fishbehavior plot --subjects 42 --overlay             # + review video of the whole recording
python -m fishbehavior plot --subjects 42 --overlay --start 600 --end 720   # a 2-minute stretch
```

Needs `label` (and `track` + `scene` for `--overlay`). Everything goes to
`<FISH_OUTPUT_DIR>/plots/`. The colors are the reference legend colors (`reference.palette`),
so our figures and the slides read the same way.

- `ethogram_<group>.png`: one per workbook group (compound + concentration), in the style of
  the reference ethograms. One row per subject, highest subject number on top; x = 0 to
  `reference.axis_seconds` (1200); white = untracked. When any subject of the group has a
  digitized reference row (`reference`), a second panel shows the reference rows lined up
  with ours ("(no ref)" marks subjects without one).
- `bars_<state>.png`: one per state, like the summary bar charts. Seconds in that state per
  group: mean ± SEM, with every subject as a dot.
- `overlay/<subject>_<start>-<end>s.mp4` (with `--overlay`): the original video with the ROI
  (green), waterline (blue), fitted body ellipse (yellow), track point (red), and the current
  second, label and confidence at the top. A timeline bar of the whole recording, with a moving
  cursor, sits under the picture. `--start`/`--end` pick a stretch in seconds on the joined
  timeline, so a split recording plays on from part a into part b.
  `plots.overlay_speed` (4) sets the playback speed; above 1, every n-th frame is kept, so the
  file stays small. Existing videos are kept unless `--force`.

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

- **LORR (listing / loss of righting) is not detected yet**: it is about 0% of the labeled
	time, before and after calibration. In the seconds the reference marks as LORR, the fish lies
	rolled or curled, low in the dish, with its body axis nearly horizontal (median tilt about 9°,
	the same as a freezing fish). The tilt rule never fires, so these seconds are labeled
	`freeze_drift`. A tested roll / belly-up cue (brightness and color of the body above vs
	below its axis, eye position, outline fill) did not separate LORR from freezing consistently
	across fish. Only a few reference fish show LORR. LORR is left to the behavior classifier
	trained on `behavior_windows.csv` / clips with the reference LORR seconds, and
	`lorr_*` calibration values carry no meaning until then.
- No frame-level behavior annotations exist, so Phase 1 behavior-state extraction
	uses tracking, rules, and unsupervised methods rather than a supervised video
	classifier.
- Compound and dose are separate label dimensions; each compound-and-dose pair is
	treated as a classification class.
- Four very small classes are pooled into adjacent doses of the same compound,
	producing a 31-class working label set pending the decision for COMPOUND_G @ 0.03.
- Cross-validation is planned instead of a single held-out test split because
	several classes remain small.

### Open decisions

- COMPOUND_G @ 0.03 has one trial and cannot be pooled with an adjacent dose. It may
	be excluded, retained as a singleton, or handled through another approved rule.
- The meaning of probability examples such as `60% COMPOUND_A, 30% COMPOUND_B, 10% Veh` must
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