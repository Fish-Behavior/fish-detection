"""Final datasets: segments, per-second labels, one row per subject, and a training manifest.

Inputs: catalog/trials.csv, labels/ (segments + labeled bins), features/endpoints.csv and
tracks/ (which part file and frame every moment is). Outputs in ``<FISH_OUTPUT_DIR>/datasets/``:

    segments.csv                every labeled segment + sex and group (compound, concentration)
    per_second_labels.csv       subject_id, second, label, confidence + key bin features
    behavior_dataset.csv/.xlsx  ONE row per subject: the cleaned workbook columns, then time,
                                bouts, mean bout and latency per state, transition counts,
                                untracked time and the video endpoints. Subjects without a
                                usable video keep their workbook values with empty behavior
                                columns. The .xlsx has a second sheet "column_guide".
    behavior_windows.csv        training manifest: one row per `export.window_s` window, with
                                the PART file and part-local frames to cut it from, the label,
                                a fold (by subject) and use_for_training
    clips/<subject_id>.npz      optional (--clips): fish-centered gray crops per window
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from fishbehavior.calibrate import fold_split, per_second
from fishbehavior.catalog import STATUS_MATCHED
from fishbehavior.config import ConfigError, Settings
from fishbehavior.features import features_dir
from fishbehavior.labeling import LABELS, SEGMENTS_FILE, STATES, UNTRACKED, labeled_bins_path, labels_dir
from fishbehavior.parallel import run_parallel
from fishbehavior.priors import ENDPOINTS_FILE
from fishbehavior.tracking import track_path, tracks_dir
from fishbehavior.video import iter_frames

log = logging.getLogger(__name__)

GROUP_COLUMNS = ["sex", "compound", "concentration_raw", "concentration_mm"]
# Bin features copied next to each second's label (the ones the rules and the swim model use).
KEY_FEATURES = [
    "tracked_fraction", "speed_mean_bl_s", "speed_median_bl_s", "speed_cv", "jerk_abs_mean_bl_s3",
    "turn_rate_var_deg2_s2", "meander_deg_per_bl", "depth_bl_min", "nose_up_surface_fraction", "tilt_fraction",
]
WINDOW_COLUMNS = [
    "subject_id", "video_path", "part", "part_start_frame", "part_end_frame", "start_s", "end_s",
    "label", "confidence", "fold", "use_for_training",
]
DATASET_FILE, WINDOWS_FILE, PER_SECOND_FILE = "behavior_dataset", "behavior_windows.csv", "per_second_labels.csv"

# Plain-words meaning of every column of behavior_dataset (the "column_guide" sheet).
WORKBOOK_GUIDE = {
    "exp_date": "date of the experiment (workbook)",
    "subject_id": "subject number, zero-padded text",
    "strain": "fish strain (workbook)",
    "sex": "M or F (workbook)",
    "age": "age (workbook, approximate)",
    "compound": "compound the fish was exposed to; Veh = vehicle control (workbook)",
    "concentration_raw": "concentration as written in the workbook (mM)",
    "concentration_mm": "concentration as a number in mM; empty for mixtures and vehicle",
    "exposure_min": "exposure time in minutes (workbook)",
    "ntt_min": "Novel Tank Test length in minutes (workbook; a different session from the video)",
    "uv_min": "UV exposure time in minutes (workbook)",
    "uv_min_raw": "UV exposure time as written in the workbook (e.g. with a footnote mark)",
    "tdm_full": "NTT total distance moved, whole tank (workbook)",
    "tdm_top": "NTT distance moved in the top half (workbook)",
    "tdm_bottom": "NTT distance moved in the bottom half (workbook)",
    "velocity_full": "NTT mean velocity, whole tank (workbook)",
    "velocity_top": "NTT mean velocity in the top half (workbook)",
    "velocity_bottom": "NTT mean velocity in the bottom half (workbook)",
    "time_top_s": "NTT seconds spent in the top half (workbook)",
    "time_bottom_s": "NTT seconds spent in the bottom half (workbook)",
    "h2o_before": "water measurement before (workbook)",
    "h2o_after": "water measurement after (workbook)",
    "brain_tissue": "brain tissue note (workbook)",
    "subject_num": "subject number as an integer",
    "ntt_tracked": "True when the workbook has NTT movement values for this subject",
    "n_workbook_rows": "workbook rows joined into this subject (> 1 = split recording)",
    "excel_rows": "row numbers in the workbook",
    "video_status": "matched = every video part found; otherwise why no behavior columns are filled",
    "video_paths": "the video file(s), ';'-joined in part order",
    "n_videos_found": "video files found for this subject",
}
ENDPOINT_GUIDE = {
    "body_length_px": "video: median fish length in pixels (the body length, BL, used for all distances)",
    "duration_s": "video: length of the whole recording (parts joined), seconds",
    "tracked_s": "video: seconds in which the fish was found",
    "distance_bl": "video: total distance swum, in body lengths",
    "mean_velocity_bl_s": "video: mean speed, body lengths per second",
    "highly_mobile_s": "video: seconds faster than features.high_mobility_bl_s",
    "immobile_s": "video: seconds slower than features.immobile_bl_s",
    "top_half_pct": "video: % of time in the top half of the water",
    "latency_top_half_s": "video: seconds until the fish first entered the top half",
    "meander_deg_per_bl": "video: mean turning per body length swum (degrees/BL)",
    "mean_angular_velocity_deg_s": "video: mean absolute turning speed (degrees/s)",
}


def datasets_dir(settings: Settings, labels: Path | None = None) -> Path:
    """The step's output folder: datasets/, or datasets_<folder name> for another labels folder."""
    return settings.paths.output_dir / ("datasets" if labels is None else f"datasets_{labels.name}")


# ---------------------------------------------------------------------------
# 1. Per-subject behavior columns (pure)
# ---------------------------------------------------------------------------


def bouts(segments: pd.DataFrame) -> pd.DataFrame:
    """One subject's segments with consecutive equal labels joined (a bout cut by a part boundary is one bout)."""
    segments = segments.sort_values("start_s")
    new_bout = segments["label"].ne(segments["label"].shift()).cumsum()
    return segments.groupby(new_bout).agg(label=("label", "first"), start_s=("start_s", "first"),
                                          duration_s=("duration_s", "sum")).reset_index(drop=True)


def behavior_columns(segments: pd.DataFrame) -> dict[str, float]:
    """Time, share, bouts, mean bout and latency per state; transitions; untracked time."""
    runs = bouts(segments)
    total = float(runs["duration_s"].sum())
    row: dict[str, float] = {}
    for state in STATES:
        mine = runs[runs["label"] == state]
        seconds = float(mine["duration_s"].sum())
        row[f"{state}_s"] = round(seconds, 2)
        row[f"{state}_pct"] = round(100 * seconds / total, 2) if total else np.nan
        row[f"{state}_bouts"] = len(mine)
        row[f"{state}_mean_bout_s"] = round(seconds / len(mine), 2) if len(mine) else np.nan
        row[f"{state}_latency_s"] = round(float(mine["start_s"].iloc[0]), 2) if len(mine) else np.nan  # never shown
    pairs = list(zip(runs["label"], runs["label"].iloc[1:]))
    for a in LABELS:
        for b in LABELS:
            if a != b:
                row[f"{a}_to_{b}"] = sum(1 for p in pairs if p == (a, b))
    row["untracked_s"] = round(float(runs.loc[runs["label"] == UNTRACKED, "duration_s"].sum()), 2)
    return row


def behavior_column_names() -> list[str]:
    """behavior_columns() keys, in order (also for subjects without video)."""
    names = [f"{s}_{k}" for s in STATES for k in ("s", "pct", "bouts", "mean_bout_s", "latency_s")]
    names += [f"{a}_to_{b}" for a in LABELS for b in LABELS if a != b]
    return [*names, "untracked_s"]


def column_guide(columns: list[str]) -> pd.DataFrame:
    """The "column_guide" sheet: every behavior_dataset column in plain words."""
    words = {"s": "seconds in {state}", "pct": "% of the recording in {state}", "bouts": "number of {state} bouts",
             "mean_bout_s": "mean length of a {state} bout, seconds",
             "latency_s": "seconds from the start until the first {state} bout (empty: never)"}
    guide = {**WORKBOOK_GUIDE, **ENDPOINT_GUIDE, "untracked_s": "video: seconds labeled untracked (fish not seen)"}
    for state in STATES:
        for key, text in words.items():
            guide[f"{state}_{key}"] = "video: " + text.format(state=state)
    for a in LABELS:
        for b in LABELS:
            guide[f"{a}_to_{b}"] = f"video: times a {a} bout was directly followed by a {b} bout"
    return pd.DataFrame({"column": columns, "meaning": [guide.get(c, "workbook column") for c in columns]})


# ---------------------------------------------------------------------------
# 2. Training windows (pure)
# ---------------------------------------------------------------------------


def make_windows(labeled: pd.DataFrame, timeline: pd.DataFrame, video_paths: list[str], window_s: float,
                 bin_s: float, min_confidence: float) -> pd.DataFrame:
    """One row per window of `window_s` seconds on the joined timeline, split where a part ends.

    Each frame takes its bin's label; the window's label is the most common one over its frames
    and its confidence their mean confidence for that label. Frames are part-local, so a clip is
    cut from `video_path` (the part file) at part_start_frame..part_end_frame (inclusive).
    """
    timeline = timeline.sort_values("time_s").reset_index(drop=True)
    time_s = timeline["time_s"].to_numpy(float)
    index = np.clip(np.floor(time_s / bin_s + 1e-9).astype(int), 0, len(labeled) - 1)  # same bins as labeling
    frames = pd.DataFrame({
        "part": timeline["part"].to_numpy(int), "part_frame": timeline["part_frame"].to_numpy(int), "time_s": time_s,
        "window": np.floor(time_s / window_s + 1e-9).astype(int),
        "label": labeled["label"].to_numpy(object)[index], "confidence": labeled["confidence"].to_numpy(float)[index],
    })
    gap = float(np.median(np.diff(time_s))) if len(time_s) > 1 else window_s
    rows = []
    for (part, _), group in frames.groupby(["part", "window"], sort=False):
        label = group["label"].value_counts().idxmax()
        confidence = float(group.loc[group["label"] == label, "confidence"].mean())
        rows.append({
            "subject_id": str(labeled["subject_id"].iloc[0]), "video_path": video_paths[part - 1], "part": part,
            "part_start_frame": int(group["part_frame"].iloc[0]), "part_end_frame": int(group["part_frame"].iloc[-1]),
            "start_s": round(float(group["time_s"].iloc[0]), 4), "end_s": round(float(group["time_s"].iloc[-1]) + gap, 4),
            "label": label, "confidence": round(confidence, 4),
            "use_for_training": bool(label != UNTRACKED and np.isfinite(confidence) and confidence >= min_confidence),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Clips (optional, one worker per subject)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipJob:
    """Everything a worker needs to cut one subject's clips (plain values, picklable)."""

    subject_id: str
    windows: pd.DataFrame  # this subject's rows of behavior_windows.csv
    track_file: Path
    body_length_px: float
    crop_bl: float
    crop_size: int
    clip_frames: int
    out: Path
    force: bool


def crop(frame: np.ndarray, x: float, y: float, side: int, size: int) -> np.ndarray:
    """A side x side square centered on (x, y), edge pixels repeated outside the frame, resized to size."""
    half = side // 2
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_REPLICATE)
    x0, y0 = int(round(x)), int(round(y))  # the +half padding offset cancels the -half of the corner
    return cv2.resize(padded[y0:y0 + 2 * half, x0:x0 + 2 * half], (size, size), interpolation=cv2.INTER_AREA)


def clip_frame_numbers(windows: pd.DataFrame, n: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """(part, part_frame) -> (window row, slot): n frames spread evenly over each window."""
    wanted = {}
    for w, row in enumerate(windows.itertuples(index=False)):
        for slot, frame in enumerate(np.linspace(row.part_start_frame, row.part_end_frame, n).round().astype(int)):
            wanted.setdefault((row.part, int(frame)), []).append((w, slot))
    return wanted


def cut_clips(job: ClipJob) -> dict[str, Any]:
    """Read each part file once, in order, and keep the wanted frames as fish-centered crops."""
    path = job.out / f"{job.subject_id}.npz"
    if path.is_file() and not job.force:
        return {"subject_id": job.subject_id, "cached": True}
    try:
        track = pd.read_csv(job.track_file, usecols=["part", "part_frame", "x", "y"]).set_index(["part", "part_frame"])
        clips = np.zeros((len(job.windows), job.clip_frames, job.crop_size, job.crop_size), np.uint8)  # 0 = fish not seen
        wanted = clip_frame_numbers(job.windows, job.clip_frames)
        side = max(2, int(round(job.crop_bl * job.body_length_px)))
        for part, video_path in job.windows.groupby("part")["video_path"].first().items():
            last = max(f for p, f in wanted if p == part)
            for frame_index, _, frame in iter_frames(video_path, gray=True):
                if frame_index > last:
                    break  # nothing more to keep from this file
                if (part, frame_index) not in wanted or (part, frame_index) not in track.index:
                    continue
                x, y = track.loc[(part, frame_index)]
                if np.isfinite(x) and np.isfinite(y):
                    image = crop(frame, x, y, side, job.crop_size)
                    for w, slot in wanted[(part, frame_index)]:
                        clips[w, slot] = image
    except Exception as error:  # noqa: BLE001 - reported per subject in the CLI summary
        log.error("%s: %s", job.subject_id, error)
        return {"subject_id": job.subject_id, "error": str(error)}
    job.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=clips, window=np.arange(len(job.windows)),
                        label=job.windows["label"].to_numpy(str))
    return {"subject_id": job.subject_id, "cached": False}


# ---------------------------------------------------------------------------
# 4. Whole run (called by the CLI)
# ---------------------------------------------------------------------------


@dataclass
class ExportResult:
    """What `run_export` wrote, for the CLI summary."""

    dataset: pd.DataFrame
    windows: pd.DataFrame
    n_labeled: int
    clip_records: list[dict[str, Any]]
    out: Path


def run_export(settings: Settings, trials: pd.DataFrame, clip_subjects: pd.DataFrame | None = None,
               force: bool = False, labels: Path | None = None) -> ExportResult:
    """Write every dataset; with `clip_subjects`, also cut their clips.

    `labels`: another labels folder (e.g. a backup copy of labels/) to export instead,
    written to datasets_<its name>/ so both versions can be compared.
    """
    if labels is not None and not (labels / SEGMENTS_FILE).is_file():
        raise ConfigError(f"{labels} has no {SEGMENTS_FILE}: give a folder made by `label` (or a copy of one)")
    params = settings.params["export"]
    bin_s = float(settings.params["features"]["bin_s"])
    labels, out = labels or labels_dir(settings), datasets_dir(settings, labels)
    segments_file, endpoints_file = labels / SEGMENTS_FILE, features_dir(settings) / ENDPOINTS_FILE
    for path, step in ((segments_file, "label"), (endpoints_file, "features")):
        if not path.is_file():
            raise ConfigError(f"{path} not found: run `python -m fishbehavior {step}` first")
    out.mkdir(parents=True, exist_ok=True)
    segments = pd.read_csv(segments_file, dtype={"subject_id": str})
    endpoints = pd.read_csv(endpoints_file, dtype={"subject_id": str})
    matched = trials[trials["video_status"] == STATUS_MATCHED]
    labeled_ids = [s for s in matched["subject_id"] if labeled_bins_path(labels, s).is_file()
                   and track_path(tracks_dir(settings), s).is_file()]

    # segments.csv: + sex and group columns.
    segments.merge(trials[["subject_id", *GROUP_COLUMNS]], on="subject_id", how="left").to_csv(
        out / SEGMENTS_FILE, index=False)

    # behavior_dataset: workbook row + behavior columns (empty without video) + video endpoints.
    behavior = pd.DataFrame([{"subject_id": s, **behavior_columns(rows)} for s, rows in segments.groupby("subject_id")],
                            columns=["subject_id", *behavior_column_names()])
    dataset = trials.merge(behavior, on="subject_id", how="left").merge(endpoints, on="subject_id", how="left")
    dataset.to_csv(out / f"{DATASET_FILE}.csv", index=False)
    with pd.ExcelWriter(out / f"{DATASET_FILE}.xlsx") as writer:
        dataset.to_excel(writer, sheet_name="behavior_dataset", index=False)
        column_guide(list(dataset.columns)).to_excel(writer, sheet_name="column_guide", index=False)

    # per_second_labels.csv and behavior_windows.csv, one labeled subject at a time.
    folds = fold_split(len(labeled_ids), int(params["folds"]), int(params["seed"]))
    fold_of = {labeled_ids[i]: k for k, members in enumerate(folds) for i in members}
    paths = matched.set_index("subject_id")["video_paths"]
    seconds, windows = [], []
    for subject_id in labeled_ids:
        labeled = pd.read_csv(labeled_bins_path(labels, subject_id), dtype={"subject_id": str})
        n_seconds = int(np.ceil(len(labeled) * bin_s - 1e-9))
        index = np.minimum(np.floor((np.arange(n_seconds) + 0.5) / bin_s).astype(int), len(labeled) - 1)
        seconds.append(pd.DataFrame({"subject_id": subject_id, "second": np.arange(n_seconds),
                                     "label": per_second(labeled, bin_s, n_seconds),
                                     "confidence": labeled["confidence"].to_numpy(float)[index],
                                     **{f: labeled[f].to_numpy(float)[index] for f in KEY_FEATURES}}))
        timeline = pd.read_csv(track_path(tracks_dir(settings), subject_id), usecols=["part", "part_frame", "time_s"])
        windows.append(make_windows(labeled, timeline, paths[subject_id].split(";"), float(params["window_s"]),
                                    bin_s, float(params["min_confidence"])).assign(fold=fold_of[subject_id]))
    pd.concat(seconds, ignore_index=True).to_csv(out / PER_SECOND_FILE, index=False, float_format="%.5g")
    all_windows = pd.concat(windows, ignore_index=True)[WINDOW_COLUMNS] if windows else pd.DataFrame(columns=WINDOW_COLUMNS)
    all_windows.to_csv(out / WINDOWS_FILE, index=False)

    records: list[dict[str, Any]] = []
    if clip_subjects is not None:
        body = endpoints.set_index("subject_id")["body_length_px"]
        jobs = [ClipJob(s, all_windows[all_windows["subject_id"] == s].reset_index(drop=True),
                        track_path(tracks_dir(settings), s), float(body[s]), float(params["crop_bl"]),
                        int(params["crop_size"]), int(params["clip_frames"]), out / "clips", force)
                for s in clip_subjects["subject_id"] if s in labeled_ids]
        records = run_parallel(cut_clips, jobs, settings.workers, desc="clips")
    return ExportResult(dataset, all_windows, len(labeled_ids), records, out)
