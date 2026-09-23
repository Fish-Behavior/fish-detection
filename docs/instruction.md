# Behavior labeling: user guide

This guide takes you from raw videos to a second-by-second behavior timeline for every fish,
training datasets and figures. It covers setup on every platform, each command with its
inputs and outputs, how the states are decided, calibration and its limits, the three files
you edit by hand, and troubleshooting. The README has the same steps in more detail
(section "Run the Phase 1 Pipeline").

The pipeline turns each ~20-minute side-view video of one zebrafish into one of five
**ethogram states** per second:

| State | In plain words |
|---|---|
| `controlled_swim` | normal swimming: steady speed, smooth path, or slow cruising |
| `erratic` | darting and zig-zagging: bursts of speed, sharp and frequent turns |
| `freeze_drift` | motionless, or only drifting |
| `lorr` | listing / loss of righting (see [Limits](#7-limits-and-known-issues): not detected yet) |
| `surface_breach` | head pushed up at the water surface, body angled nose-up |

plus `untracked` (not a behavior: the fish was not seen well enough to judge).

There are no hand annotations. Labels come from tracking, rules and a pooled clustering model,
tuned against digitized reference ethogram figures. The workbook's NTT values (Novel Tank Test, a
**different** session) are used as hints only, never as per-second truth.

---

## 1. Setup

You need Python 3.10 or newer. All commands below run from the repository folder.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest                       # optional: runs the synthetic tests (no real data needed)
```

### Windows (PowerShell or Command Prompt)

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
pytest
```

If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt.

### Google Colab

Open `notebooks/colab_quickstart.ipynb` in Colab (*File > Upload notebook*, or from GitHub). It
mounts Google Drive, clones and installs the repository, sets the paths with `os.environ`
instead of a `.env` file, and runs the pipeline. **No GPU is needed**: everything here is
classic computer vision and small statistical models on the CPU. Keep `FISH_OUTPUT_DIR` on
Drive, so results survive a disconnect, and rerun `all` to continue.

### The `.env` file (laptop / workstation)

Copy the template and fill in the paths on your machine. `.env` is git-ignored.

```bash
cp .env.example .env         # Windows: copy .env.example .env
```

| Variable | What |
|---|---|
| `FISH_VIDEO_DIR` | folder with the trial videos (`F_0042.mp4`, `M_0012a.mp4`, ...) |
| `FISH_DB_PATH` | the trial workbook (`.xlsx`) |
| `FISH_REFERENCE_PDF` | optional: the presentation with the reference ethograms (calibration only) |
| `FISH_OUTPUT_DIR` | where everything is written (default `outputs`, git-ignored) |
| `FISH_WORKERS` | videos / subjects processed in parallel (default 1; try the number of CPU cores) |
| `FISH_CONFIG` | optional: a YAML file with only the settings you want to change |

Environment variables with the same names override `.env`. Check the setup with:

```bash
python -m fishbehavior check-config      # every path: ok / not set / MISSING
```

**Settings.** Every tunable number lives in `src/fishbehavior/default_config.yaml`, with a
comment on what it does. Do not edit that file for your own runs. Put the values you want to
change in your own YAML file (e.g. `my_settings.yaml` with `plots: {overlay_speed: 8}`) and point
`FISH_CONFIG` (or `--config`) at it.

---

## 2. One command: `all`

```bash
python -m fishbehavior all                        # every subject
python -m fishbehavior all --subjects 42 F_0043   # only these ("42", "0042", "F_0042" all work)
python -m fishbehavior all --force                # redo finished subjects too
python -m fishbehavior all --skip-reference       # no reference / priors / calibration
```

It runs, in order:

```text
validate -> scene -> track -> features -> label
         -> [reference -> priors -> calibrate -> label again]    (when FISH_REFERENCE_PDF is set
         -> export -> plot                                          and mapping.yaml is filled in)
```

- **Finished work is skipped.** Scene, track, features and label keep one result per video or
  subject and redo it only when it is missing, older than its inputs, made with other
  settings, or with `--force`. `calibrate` is skipped while `calibrated.yaml` is newer than the
  reference timelines and every feature file. `export` and `plot` always rewrite their tables and
  figures (about 2 minutes for ~300 subjects).
- **Stops at the first failing step** and says which one. Fix the problem, then run `all` again.
- **Timing summary** at the end: seconds per step.
- The first run with a reference PDF writes `reference/mapping.yaml` and skips the tuning steps.
  Fill it in ([section 5](#5-files-you-fill-in-by-hand)); the next `all` runs them.

## 3. The steps one by one

Every step is also its own command: `python -m fishbehavior <command> --help`. Per-subject
steps accept `--subjects` and `--force`. All outputs are under `FISH_OUTPUT_DIR`.

| Command | Needs | Writes | What it does |
|---|---|---|---|
| `validate` | workbook, videos | `catalog/trials.csv`, `videos.csv`, `validation_report.md` | cleans the workbook, matches each subject to its video file(s), joins split recordings |
| `scene` | `validate` | `scene/<video>.json`, `_background.png`, `_qa.png`, `video_check.csv` | per video: empty-beaker background, waterline, region the fish can reach (ROI) |
| `scene-review` | `validate` | `scene/overrides.yaml` | browser page to check and correct waterlines and ROIs |
| `track` | `scene` | `tracks/<subject>.csv.gz`, `_track_qa.png`, `summary.csv` | fish position, outline, tilt and head in every frame; parts joined into one timeline |
| `features` | `track` | `features/<subject>_frames.csv.gz`, `_bins.csv.gz`, `endpoints.csv` | speed, turning, depth and posture per frame and per 1 s bin, in body lengths (BL) |
| `features-qa` | `features` | `features/qa.csv` | which bin features stay stable at half the frame rate / double smoothing |
| `label` | `features` | `labels/<subject>_bins.csv`, `segments.csv`, `summary.csv`, `swim_model.json` | one state per bin, merged into segments |
| `reference` | reference PDF | `reference/timelines.csv`, `group_means.csv`, `digitize_check.png`, `mapping.yaml` | digitizes the reference ethogram figures |
| `priors` | `label`, `reference` | `calibration/priors_report.md` | sanity checks vs the workbook (hints) and the reference group means |
| `calibrate` | `label`, `reference` | `calibration/calibrated.yaml`, `calibration_report.md` | tunes the labeling thresholds against the reference |
| `export` | `label` | `datasets/*` | final datasets (below) |
| `plot` | `label` | `plots/*` | ethograms per group, bar charts per state, `--overlay` review videos |

**Datasets** (`datasets/`, column list in the README, step 9):

- `behavior_dataset.csv` / `.xlsx`: one row per subject = the cleaned workbook columns + seconds,
  %, bouts, mean bout length and latency per state, transition counts, untracked time and the
  video endpoints. The xlsx has a `column_guide` sheet explaining every column;
- `per_second_labels.csv`: subject, second, label, confidence and key features;
- `segments.csv`: every run of one label, with sex and group;
- `behavior_windows.csv`: training manifest, one row per 1 s window, with the part file and the
  frame numbers inside that file, a fold by subject, and `use_for_training`;
- `clips/<subject>.npz` (`export --clips`): fish-centered crops per window.

`export --labels <folder>` exports a copy of a labels folder (e.g. labels saved before
calibration) to `datasets_<folder name>/` so two labelings can be compared.

**Live view:** `python -m fishbehavior live` opens a local page that processes one video
while you watch. It shows the scene (correctable), every tracked frame with its outline, the
labels as they are decided, with the rule checks behind each second, the ethogram next to the
reference row, and the measurements the rules read. It ends with exactly the batch labels. Labels
shown while it plays are final except for the hatched last seconds when `features` has already run
for that subject; for a new video the whole timeline stays provisional until the end. About
10 s per 20-minute video at `Max` pace; choose `1×`–`30×` for demos. See the README, step 11.

**Review video:** `plot --subjects 42 --overlay --start 600 --end 720` shows the original
frames with ROI, waterline, fitted body ellipse, track point, the current label and a timeline
bar with a moving cursor. Check it for a few subjects of every group before trusting the labels.

---

## 4. How the states are decided

Each 1 s bin (`features.bin_s`) gets the first rule that matches, in this order. All numbers are
`labeling:` settings (starting values; calibration may replace some):

| # | Label | Rule on the bin |
|---|---|---|
| 0 | `untracked` | fish seen in less than `min_tracked_fraction` (0.5) of the frames |
| 1 | `surface_breach` | seen side-on, head at the waterline and body ≥ 25° nose-up in at least `surface_breach_fraction` (0.3) of the frames |
| 2 | `lorr` | seen side-on and tilted > 45° in ≥ `lorr_tilt_fraction` (0.5) of the frames, median speed < `lorr_max_speed_bl_s` (0.5 BL/s), for ≥ `lorr_min_s` (3 s) in a row |
| 3 | `freeze_drift` | median speed < `freeze_speed_bl_s` (0.1 BL/s) for ≥ `freeze_min_s` (2 s) in a row |
| 4a | `controlled_swim` | any other bin slower than `swim_min_speed_bl_s` (0.5 BL/s), or one the swim model calls smooth |
| 4b | `erratic` | any other bin the swim model calls erratic |

The **swim model** splits the faster swimming bins in two using speed variation, jerkiness,
turn-rate variance and meander. It is fitted on **all subjects together**, so "erratic" means the
same for every fish (`labeling.swim_split`: `gmm`, or `hmm` for smoother timelines). Afterwards,
bouts shorter than `min_bout_s` merge into their longer neighbour (untracked gaps are never
hidden). Every bin has a `confidence` from 0 to 1 (how strongly the rule held, or the model's
probability; 0 for bins changed by the cleanup).

## 5. Files you fill in by hand

All three live in the git-ignored output folder, so group names and subject numbers never
reach Git.

### `scene/overrides.yaml`: correct a waterline or region

Easiest with `scene-review` (it writes the file for you). By hand: one entry per video **file
name**, in original video pixels:

```yaml
F_0042.mp4:
  waterline_y: 118            # row of the water surface
  roi: [30, 95, 290, 215]     # x0, y0, x1, y1
M_0012b.mp4:
  checked: true               # looked at, the automatic result is fine
```

Then run `scene` (or `all`) again; only changed videos are redone.

### `reference/mapping.yaml`: which subject is each reference row

The subject labels in the reference figures overlap and cannot be read, so the first `reference`
run writes a template with one entry per panel and an estimated row count:

```yaml
panels:
- page: 16
  panel: 1
  rows: 12
  group: ''          # the panel title, e.g. VEH or COMPOUND_A 30uM
  subject_ids: []    # TOP to BOTTOM, exactly `rows` of them
  skip: false        # true for a panel that repeats another one
```

Take each group's subjects from `catalog/trials.csv` (same compound and dose). The figures put a
group's first subject at the bottom, so top to bottom is usually the subject numbers in
**descending** order. Check against the readable top and bottom labels, then look at
`digitize_check.png`: the original crop and the redrawn rows side by side. If a row count is
wrong, the step says which panel. A placeholder id (e.g. `9001`) keeps a row you cannot match: it
counts in the group means but matches no video.

### `reference/expectations.yaml`: group links and direction checks

Written by the first `priors` run:

```yaml
groups:              # reference group -> workbook compound / concentration (trials.csv)
  VEH: {compound: '', concentration: ''}                       # fill in by hand
  COMPOUND_A 30uM: {compound: COMPOUND_A, concentration: '0.03'}  # filled from mapping.yaml
expectations:
- {group: COMPOUND_A 30uM, state: freeze_drift, relation: greater_than, than: VEH}
```

`groups` links each reference group to its workbook subjects, so all of that group's videos are
compared (not only the mapped ones). `expectations` are relations the labels should reproduce
(`greater_than` / `less_than`). Each is tested on the reference means and on our video means.

## 6. Calibration

`calibrate` uses the subjects that have both a video and a digitized reference row. It tries
`calibration.n_trials` (200) random threshold sets from `calibration.search` (the first is always
the current settings), relabels the bins with each one (no video work, about a minute for 50
subjects), and scores it:

```text
score = macro-F1 per second vs the reference + group_weight (0.5) x group agreement
```

Group agreement is 1 minus the total-variation distance between our and the reference time
budget per group. It forgives the small timing offsets of the digitized rows. The best set goes
to `calibration/calibrated.yaml` (only the tuned keys). `label` merges it over the settings and
says so. Delete the file to go back to the defaults.

**Read the report before trusting it** (`calibration_report.md`):

- **Held-out check.** K-fold by subject: the values are chosen on four folds and scored on the
  fifth. Held-out F1 far below in-sample F1 means the search fits noise. Held-out "tuned" at or
  below "current" means calibration does not help.
- **Number of subjects.** Below `calibration.min_subjects` (30) the report warns. Even above it,
  values fitted on only a few groups fit those groups.
- **Per-state F1 and the confusion matrix.** Calibration only moves thresholds. It cannot create
  a state the features do not see (see LORR below).
- The reference is itself approximate (digitized from figures).

To compare labelings later, copy `labels/` before relabeling, e.g. to `labels_before_calibration/`,
and export both (`export --labels outputs/labels_before_calibration`).

## 7. Limits and known issues

- **LORR is not detected yet** (about 0% of the time). In the seconds the reference marks as LORR,
  the fish lies rolled or curled, low in the dish, with its body axis nearly horizontal, so it
  looks like a freezing fish to the tilt rule and is labeled `freeze_drift`. A roll / belly-up cue
  (brightness and color above vs below the body axis, eye position, outline fill) did not separate
  LORR from freezing consistently across fish. LORR is left to a behavior classifier trained on
  `behavior_windows.csv` / clips with the reference LORR seconds. Until then, the `lorr_*`
  calibration values carry no meaning.
- **NTT values are a different session.** They correlate only loosely with the video (see
  `priors_report.md`). Never use them as labels.
- **Surface breach** needs the head (eye) to be found. In very small or blurry videos it is missed.
- Recordings much longer or shorter than 1200 s are flagged by `scene` (`video_check.csv`). Their
  seconds still count in the datasets.

## 8. Higher frame rates and larger videos

Nothing assumes 30 fps or 320×240. Times come from `frame_index / fps` read from each file, and
distances are in body lengths, so the same settings work for 60 fps or 1080p footage. For large or
fast videos:

- `tracking.scale: 0.5` analyses frames at half size (coordinates are converted back);
- `tracking.frame_stride: 2` analyses every second frame (times stay exact);
- `features.smooth_s` is in seconds, so smoothing adapts to the frame rate. `features-qa`
  shows which features stay stable at half the frame rate;
- `FISH_WORKERS` processes several videos at once (each needs its own CPU core and memory);
- `export --clips` and `plot --overlay` read every frame: use them on a few subjects first.

## 9. Privacy rules

- Never commit data, outputs, `.env`, local paths, compound names, real subject IDs or numbers
  fitted on real data. `.gitignore` keeps `data/`, `outputs/`, `.env` and data file types out.
- Values fitted on real data (e.g. `calibrated.yaml`) stay under `FISH_OUTPUT_DIR`, never in
  `default_config.yaml`.
- Code, docs and tests use made-up IDs (`F_0042`, `M_0012a`) and names (`COMPOUND_A`).
- Before committing, run `pytest`, then check `git status` and `git diff --cached` for anything
  from `data/`, `outputs/`, `.env`, real IDs or compound names.
- The notebook is saved without outputs. Clear outputs before committing it again.
- CI checks every pull request for tracked data files, `.env` and notebook outputs
  ([docs/how-to-start.md](how-to-start.md), section 6). It cannot recognize real IDs, compound
  names or fitted numbers, so checking for those is still up to you.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `pip: command not found` (macOS) | use `python3 -m pip install -r requirements.txt` (or activate the venv first: then `python -m pip`) |
| `python: command not found` (macOS/Linux) | use `python3`; inside an activated venv `python` works |
| `'python' is not recognized` (Windows) | use `py`, or reinstall Python with "Add to PATH" |
| `Configuration error: FISH_... is not set` | add it to `.env` (or `os.environ` on Colab); `check-config` shows every path |
| `... not found: run python -m fishbehavior <step> first` | run that step (or simply `all`) |
| a video `failed` in `scene` / `track` | the file cannot be opened (codec) or has no frames. Re-encode it to mp4 (H.264) and run again |
| waterline or ROI wrong in a `_qa.png` | correct it in `scene-review` (or `overrides.yaml`), then run `all` again |
| fish lost for long stretches (`untracked`) | look at `_track_qa.png`. Usually the ROI is too small, or `tracking.diff_threshold` too high for a pale fish |
| `reference` says `not filled in yet` | fill in `reference/mapping.yaml` ([section 5](#5-files-you-fill-in-by-hand)) |
| `reference`: `9 subject_ids but rows: 10` | count the rows in the zoomed `page<N>.png` and fix `rows` or the list |
| warnings about worker processes, runs one by one | the environment forbids process pools. Results are the same, only slower; set `FISH_WORKERS=1` to silence it |
| Colab disconnected | rerun the cells from the top; `all` continues where it stopped |
| `Matplotlib is building the font cache` | first plot only, harmless |
