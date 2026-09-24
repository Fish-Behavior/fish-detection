# Strain / ROI Tracking Spot-Check Notes (Phase 3, T029)

Informal validation log required by PRD §12 T029: "Spot-check background/waterline
detection against 1 real video per strain (Casper, Wild-type AB, ABSL) - log
findings/screenshots... feeding Phase 4 risk mitigation." Findings below are from
real, local-only research data (never committed - see root `.gitignore` /
`preprocessing_dataset_system/.gitignore`); this file contains no restricted data
itself, only numeric/statistical findings and file-path references.

## 1. Initial 3-video, 1-per-strain check

Picked the first `MATCHED` trial per `strain` from a real `catalog` run
(`PDS_VIDEO_DIR=videos/Phase_1`, `PDS_DB_PATH=sample_data/00_NTT_DataBase.xlsx`):

| Strain | Subject | Compound | Resolution | Duration | `detect_waterline()` row |
|---|---|---|---|---|---|
| Casper (roya9; mitfaw2) | 0001 | Veh | 320x240 | 1202.7s | 165 (69% down) |
| Wild-type (AB) | 0057 | Veh | 320x240 | 1202.1s | 144 (60% down) |
| ABSL (ednrb1a[b140]:mitifa[b692]) | 0307 | Veh | 192x240 | 1200.7s | 5 (2% down, degenerate) |

Visual inspection of the background images (`estimate_background()` output) for all
three showed a **side-view of a round beaker**, water filling most/all of the visible
frame, with the strongest horizontal edge in each case landing on the **beaker's
curved bottom rim / table junction**, not a visible air-water interface - the region
above that edge is just more (fainter) beaker glass and out-of-focus background, not
open air. A raw mid-video frame from the Casper video (subject 0001) confirms a real
fish is visible resting near that same bottom-rim region; the top ~1/3 of the frame
(cropped and pixel-inspected) is flat and featureless, with no detectable secondary
edge - i.e., the true water surface is most likely **above the visible frame
entirely** in this specific video, not just faint.

**First-pass conclusion (later revised, see §2): treat the detected edge as
"beaker-bottom reference," not a true waterline, and consider `depth_from_surface`
relative to `y=0` (top of frame) as a fallback proxy.** This was checked against the
advisor before committing to any design decision, which correctly flagged that 3
videos - all from the *same* compound folder (`Veh`) - is too narrow a sample to
generalize from, and separately flagged that the two resolutions already seen
(320x240 vs 192x240) is itself an important, separable finding.

## 2. Broader survey: 48 videos across all 16 matched compounds

Sampled first/middle/last (by `Date of EXP:`) `MATCHED` trial per compound from the
same real catalog run - 48 videos, spanning every compound family with a local
video match. For each: `probe()` resolution, then `estimate_background(stride=60)`
+ row-wise grayscale gradient, recording the peak row, its fraction of frame height,
its raw gradient magnitude, and its SNR relative to the frame's own median row-to-row
gradient (`peak_grad / median_grad`).

### Resolution is not uniform (confirms it must never be assumed a constant)

| Resolution | Count |
|---|---|
| 320x240 | 47 |
| 192x240 | 1 |

Also worth noting: the standalone demo video in `sample_data/` (UUID-named, not part
of the 353-trial set) measures **304x240** - a *third* distinct resolution, and not
representative of the main `Phase_1` mirror's dominant 320x240. Any pixel-unit
threshold (contour area, jump distance, depth thresholds in
`config/default_thresholds.yaml`) must be computed as a fraction of frame
width/height, or explicitly scaled per-video's own measured resolution - never a
fixed pixel count. **Action for Phase 4:** `tracking.py`/`features.py` must read and
carry each video's own resolution (already available from `probe()`) through to
every pixel-unit computation.

### The detected edge is strong (not a noise/confidence problem)...

SNR (peak gradient vs. that frame's own median row-gradient) ranged **9.6x-110.5x**
(median 37.7x) across all 48 videos. `detect_waterline()`'s underlying row-wise
gradient-peak method reliably finds *a* real, strong structural edge in every video
sampled - this is not a "signal too weak to trust" problem.

### ...but it is bimodal in *where* that edge lands, i.e. *which* edge it is

| Peak row as fraction of frame height | Interpretation |
|---|---|
| ~0.01-0.09 (≈12 of 48 videos) | Very near top of frame - possibly the true water surface, or a bright rim highlight |
| ~0.42-0.90 (≈36 of 48 videos) | Lower half to bottom of frame - beaker-bottom rim, confirmed by direct visual inspection in §1 |

This is the important, unresolved finding: **`detect_waterline()`'s return value does
not consistently correspond to the same physical feature across videos.** It is
possible that some recording sessions/camera setups actually do show the true
water-air surface near the top of frame, while others (like all 3 in §1, which
happened to all be `Veh`) crop lower and never show it. The single 192x240-resolution
video (subject 329, `FD-2-67`) also has its peak very near the top (row 15/240,
6%) - worth checking whether resolution and edge-position correlate with a shared
root cause (different camera/zoom setup for a subset of recording sessions), which
was not conclusively determined from this survey alone.

Raw survey data: `subject_id, compound, resolution, peak_row, frame_h,
peak_row_frac, peak_grad, median_grad, snr` for all 48 videos was generated by a
one-off script during this session (not committed - transient analysis, not a
pipeline artifact); regenerate by sampling `MATCHED` rows from a real `catalog` run
and calling `estimate_background()` + a row-gradient scan, as done here.

## 2b. Pigmented-fish contrast check (T030)

Visual check of raw mid-video frames (not background-subtracted diffs - that's
Phase 4/T035's more rigorous check) for the same 3 strain-representative videos from
§1:

- **Casper (translucent)**, subject 0001: fish clearly visible, pink/orange body
  faintly outlined against the pale beaker background.
- **Wild-type (AB)**, subject 0057: fish clearly visible with a strong, high-contrast
  striped pattern - if anything *more* visually distinct than the translucent Casper
  fish, not less.
- **ABSL**, subject 0307: fish visible (pink/orange coloring) though smaller in frame
  in this narrower 192x240 video; still discernible against the background by eye.

**T030 conclusion: no `roi.py` threshold change made based on this spot-check** - all
three strains show a visually distinguishable fish against the background at the raw
pixel level. This is a lightweight, "does contrast look obviously broken" check, not
a rigorous validation of the actual foreground-extraction algorithm (which doesn't
exist yet - that's Phase 4's `tracking.py`). The real test is Phase 4's own T035
task (debug-overlay video export, reviewed visually, across 10-15 real videos
spanning strains/compounds) - if background-subtraction-based foreground extraction
turns out to struggle specifically on pigmented strains once actually implemented,
that is the point to revisit thresholds here, not now on inconclusive raw-frame
visual inspection alone.

## 3. Design implication for Phase 4/6/7 (not resolved here - deferred deliberately)

Per the "don't hand-pick reasonable-looking thresholds" principle (PRD §9.5.5) and
the advisor's explicit recommendation: **do not commit to a single "waterline"
semantic now.** `roi.py::detect_waterline()`'s docstring has been corrected to no
longer claim its output is validated as *the* water surface (it previously
overclaimed this before the 48-video survey existed). Phase 4 (`tracking.py`) should
record **two** candidate depth signals per frame rather than committing to one:

1. `depth_from_surface_topframe = fish_y` (distance from `y=0`, i.e. top of frame) -
   always available, resolution-normalized.
2. `depth_from_surface_edge = fish_y - detect_waterline(background)` - the
   edge-relative signal, carrying whatever that edge actually is per-video.

Phase 6 calibration (which has ground truth only at the *group* level, from the
reference images) is unlikely to be able to discriminate between these two
candidates on its own. The real resolution point is **T035** (Phase 4's own task:
"visually spot-check with an overlay-annotated debug video export... before scaling
to all 353") - overlaying detected fish position and the detected edge on actual
video frames, across a larger and strain/compound-balanced sample, is what will
actually show whether the edge is the real surface, the beaker bottom, or something
that varies by recording session. This file's job is to make sure that spot-check
happens with this ambiguity already known, not rediscovered from scratch.
