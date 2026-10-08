# Request to `prepds`: record tracker, resolution and camera setup per video

**From:** the classifier (`dcs`, [classifier_PRD.md](classifier_PRD.md)) · **Plan task:** T5.6 · **Status:** draft for the owner to file
**Privacy:** placeholders only; no data, names, dates or paths.

## What we ask

For every video, write three more fields into `manifest.json`, and carry them into `accepted_index.parquet` (and,
where it applies, `trials_catalog.parquet`):

| Field | Example | Written by |
|---|---|---|
| `tracker` | `classical`, or `model:<run id>` | `prepds run` (it knows which tracker it used) |
| `video_width_px`, `video_height_px` | `304`, `240` | `prepds run` (the video probe already reads them) |
| `camera_setup` (optional) | `setup-2` | Whoever records a camera or zoom change (a short table of first dates per setup), or calibration |

Old manifests without the fields stay valid; `dcs` keeps its current fallbacks when a field is missing.

## Why

| Today in `dcs` | Problem | With the fields |
|---|---|---|
| The tracker is guessed from the calibration profile's name (`-model` in it, D-003), cross-checked against `detections.parquet` only where the processed folder is present (D-034) | A renamed profile, or an accepted folder copied without the processed folder, cannot be checked (PRD change C1, EC-29) | `dcs featurize` reads the tracker directly and the audit stops guessing |
| Resolution is not recorded anywhere; the audit prints "not recorded upstream" (C2) | Pixel speeds and depth depend on resolution and zoom (EC-26, D-015) | The audit's EC-26 check covers resolution as well as frame rate |
| Camera framing is inferred from where the fish swim (audit EC-31: per-date depth percentiles grouped into setups, D-050) | On the real set this found 18 setups in 3 camera epochs; the epochs line up with compounds, so a model can recognize the camera instead of the drug. Vehicle normalization by date leaks the date; by camera epoch it helps (D-061), but the epoch start dates must be typed in by hand (`training.camera_epochs`) | `dcs` takes the reference groups for vehicle normalization from `camera_setup` directly, with no inference |

## Acceptance

- A new `prepds run` writes `tracker`, `video_width_px`, `video_height_px` into every manifest it creates.
- `prepds export-index` copies them (and `camera_setup` when present) into `accepted_index.parquet`.
- Existing outputs still load in `prepds review` and in `dcs featurize`.
