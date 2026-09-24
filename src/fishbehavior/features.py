"""Movement features per frame and per time bin, plus per-subject endpoints.

Input: the per-frame track of each subject (tracking.py) and the waterline / ROI of its
first video part (roi.py). Units are body lengths (BL) and seconds:

* BL = median fitted `major_axis` over detected frames, so every speed and depth
  compares across fish, zoom levels and resolutions.
* Frame spacing comes from the track's own `time_s` (median gap), never an assumed fps,
  so `tracking.frame_stride`, 60 fps originals and joined parts all work unchanged.

x and y are smoothed with a Savitzky-Golay filter (`features.smooth_s` seconds) before
differentiating; untracked frames stay empty rather than being invented by the filter.

Outputs (in ``<FISH_OUTPUT_DIR>/features/``, git-ignored):
    <subject_id>_frames.csv.gz     full frame rate, one row per track row (FRAME_COLUMNS)
    <subject_id>_bins.csv.gz       one row per `features.bin_s` bin from t = 0 (BIN_COLUMNS)
    <subject_id>_features.json     body length, endpoint row and settings used (cache key)
    endpoints.csv                  one row per subject (ENDPOINT_COLUMNS)
    qa.csv                         `features-qa`: per subject and bin feature, is it stable? (QA_COLUMNS)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from fishbehavior.catalog import STATUS_MATCHED
from fishbehavior.config import Settings
from fishbehavior.parallel import run_parallel
from fishbehavior.roi import json_path as scene_json_path
from fishbehavior.roi import scene_dir as get_scene_dir
from fishbehavior.tracking import track_path
from fishbehavior.tracking import tracks_dir as get_tracks_dir

log = logging.getLogger(__name__)

ENDPOINTS_FILE = "endpoints.csv"
FRAME_COLUMNS = [
    "subject_id", "part", "frame", "time_s", "tracked", "vx", "vy", "speed_bl_s", "accel_bl_s2", "jerk_bl_s3",
    "heading_deg", "turn_deg", "angular_velocity_deg_s", "meander_deg_per_bl", "depth_bl", "at_surface",
    "pitch_deg", "nose_up_at_surface", "tilt_deg", "side_on", "aspect", "area_ratio",
]
BIN_COLUMNS = [
    "subject_id", "bin", "t_start_s", "n_frames", "tracked_fraction", "speed_mean_bl_s", "speed_median_bl_s",
    "speed_max_bl_s", "speed_cv", "accel_abs_mean_bl_s2", "jerk_abs_mean_bl_s3", "turn_rate_var_deg2_s2",
    "angular_velocity_abs_mean_deg_s", "meander_deg_per_bl", "distance_bl", "surface_fraction",
    "nose_up_surface_fraction", "depth_bl_min", "tilt_median_deg", "tilt_fraction", "aspect_median", "area_ratio_median",
]
ENDPOINT_COLUMNS = [
    "subject_id", "body_length_px", "duration_s", "tracked_s", "distance_bl", "mean_velocity_bl_s",
    "highly_mobile_s", "immobile_s", "top_half_pct", "latency_top_half_s", "meander_deg_per_bl",
    "mean_angular_velocity_deg_s",
]
# Settings that change the outputs; cached results made with other values are redone.
FEATURE_PARAMS = (
    "smooth_s", "smooth_polyorder", "min_speed_for_heading_bl_s", "surface_margin_bl", "bin_s",
    "tilt_threshold_deg", "high_mobility_bl_s", "immobile_bl_s", "nose_up_threshold_deg", "side_on_min_head_offset_bl",
)
# Raised when a code change alters the outputs, so results cached by older code are redone too.
# 2: nose_up_at_surface counts only frames seen side-on.
FEATURES_VERSION = 2


# ---------------------------------------------------------------------------
# 1. Small numeric helpers
# ---------------------------------------------------------------------------


def body_length_px(track: pd.DataFrame) -> float:
    """BL in pixels: median fitted length over detected (not interpolated) frames."""
    return float(track.loc[track["detected"].astype(bool), "major_axis"].median())


def frame_gap_s(time_s: np.ndarray) -> float:
    """Typical seconds between rows (median positive gap); NaN with fewer than 2 rows."""
    gaps = np.diff(time_s)
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    return float(np.median(gaps)) if len(gaps) else float("nan")


def odd_window(seconds: float, gap_s: float, polyorder: int) -> int:
    """Smoothing window in rows: `seconds` at this frame spacing, odd.

    At least polyorder + 3 rows: with fewer, the polynomial passes through every point
    and the filter silently stops smoothing (e.g. 3 rows at order 2, as at 15 fps).
    """
    minimum = polyorder + 3
    size = max(minimum, round(seconds / gap_s)) if np.isfinite(gap_s) else minimum
    return size if size % 2 else size + 1


def smooth(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing that leaves untracked (NaN) rows NaN.

    The filter cannot take NaN, so gaps are bridged by interpolation just for filtering,
    then put back to NaN: smoothing never creates a position where the fish was not seen.
    """
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return values.copy()  # too little to smooth
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)  # odd and <= length
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    out = savgol_filter(filled, window, min(polyorder, window - 1))
    out[~finite] = np.nan
    return out


def wrap180(degrees: np.ndarray) -> np.ndarray:
    """Angle difference folded into [-180, 180), so 179 -> -179 is a 2 degree turn."""
    return (degrees + 180.0) % 360.0 - 180.0


def row_seconds(time_s: np.ndarray) -> np.ndarray:
    """Seconds each row stands for: the gap to the previous row (first row: the next gap)."""
    gaps = np.diff(time_s)
    return np.r_[gaps[:1], gaps] if len(gaps) else np.full(len(time_s), np.nan)


def step_distance_bl(frames: pd.DataFrame) -> np.ndarray:
    """BL moved since the previous row (speed x gap), on the later row; first row NaN."""
    return np.r_[np.nan, frames["speed_bl_s"].to_numpy(float)[1:] * np.diff(frames["time_s"].to_numpy(float))]


# ---------------------------------------------------------------------------
# 2. Per frame
# ---------------------------------------------------------------------------


def frame_features(track: pd.DataFrame, bl_px: float, waterline_y: float, params: dict[str, Any]) -> pd.DataFrame:
    """Full-rate feature table (FRAME_COLUMNS) from one subject's track table."""
    track = track.sort_values("frame").reset_index(drop=True)
    if track.empty:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    time_s = track["time_s"].to_numpy(float)
    polyorder = int(params["smooth_polyorder"])
    window = odd_window(float(params["smooth_s"]), frame_gap_s(time_s), polyorder)
    x = smooth(track["x"].to_numpy(float), window, polyorder)
    y = smooth(track["y"].to_numpy(float), window, polyorder)

    gaps = np.diff(time_s)
    gaps = np.where(gaps > 0, gaps, np.nan)  # a repeated timestamp gives no rate

    def rate(values: np.ndarray) -> np.ndarray:
        """Change per second since the previous row (backward difference); first row NaN."""
        return np.r_[np.nan, np.diff(values) / gaps]

    # Kinematics in BL: velocity, then speed and its derivatives (tangential, so turning at
    # constant speed shows in turn/angular velocity, not in accel).
    vx, vy = rate(x / bl_px), rate(y / bl_px)
    speed = np.hypot(vx, vy)
    accel = rate(speed)
    jerk = rate(accel)

    # Heading of the movement; its change only counts when both rows move fast enough,
    # because a near-still fish's heading is just position noise.
    heading = np.degrees(np.arctan2(vy, vx))
    moving = speed > float(params["min_speed_for_heading_bl_s"])
    turn = np.r_[np.nan, np.where(moving[1:] & moving[:-1], wrap180(np.diff(heading)), np.nan)]
    angular_velocity = np.r_[np.nan, turn[1:] / gaps]
    step = np.r_[np.nan, speed[1:] * gaps]  # BL moved since the previous row
    with np.errstate(divide="ignore", invalid="ignore"):
        meander = np.where(step > 0, np.abs(turn) / step, np.nan)

    # Posture and position relative to the waterline (y grows downward: depth > 0 = below).
    top_limit = waterline_y + float(params["surface_margin_bl"]) * bl_px
    median_area = track.loc[track["detected"].astype(bool), "area"].median()
    # Seen side-on: the eye is near one end of the body, far from its centre. A fish facing
    # the camera has its eyes mid-blob and a tall outline whose fitted angle is not a tilt.
    side_on = np.hypot(track["head_x"] - track["x"], track["head_y"] - track["y"]).to_numpy(float) \
        >= float(params["side_on_min_head_offset_bl"]) * bl_px  # no head (NaN) -> False
    # Surface breach: the head (eye) itself is at the waterline and the head-tail line points
    # nose-up. Only the top of the body touching the surface is not enough: in shallow dishes
    # the fish is that close to the surface most of the time. Side-on only, as for the tilt: a
    # fish facing the camera is a near-round blob whose fitted axis (so its pitch) is noise.
    pitch = track["pitch_deg"].to_numpy(float)
    head_at_surface = track["head_y"].to_numpy(float) <= top_limit  # no head (NaN) -> False
    nose_up = head_at_surface & (pitch >= float(params["nose_up_threshold_deg"])) & side_on
    frames = pd.DataFrame({
        "subject_id": track["subject_id"],
        "part": track["part"],
        "frame": track["frame"],
        "time_s": track["time_s"],
        "tracked": track["detected"].astype(bool) | track["interpolated"].astype(bool),
        "vx": vx, "vy": vy, "speed_bl_s": speed, "accel_bl_s2": accel, "jerk_bl_s3": jerk,
        "heading_deg": heading, "turn_deg": turn, "angular_velocity_deg_s": angular_velocity,
        "meander_deg_per_bl": meander,
        "depth_bl": (y - waterline_y) / bl_px,
        "at_surface": track["top_y"].to_numpy(float) <= top_limit,  # NaN (untracked) -> False
        "pitch_deg": pitch,
        "nose_up_at_surface": nose_up,
        "tilt_deg": track["angle_deg"].abs(),  # -90..90 from horizontal folded to 0..90
        "side_on": side_on,
        "aspect": track["major_axis"] / track["minor_axis"],
        "area_ratio": track["area"] / median_area,
    })
    return frames[FRAME_COLUMNS]


# ---------------------------------------------------------------------------
# 3. Per time bin
# ---------------------------------------------------------------------------


def meander_of(turn_deg: pd.Series, distance_bl: pd.Series) -> float:
    """Total |turn| / total distance (deg per BL); NaN when the fish did not move."""
    distance = float(distance_bl.sum())
    return float(turn_deg.abs().sum()) / distance if distance > 0 else float("nan")


def bin_features(frames: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """One row per `features.bin_s` bin from t = 0; bins without rows are kept (n_frames 0)."""
    bin_s = float(params["bin_s"])
    tilt_threshold = float(params["tilt_threshold_deg"])
    time_s = frames["time_s"].to_numpy(float)
    duration = float(time_s.max() + frame_gap_s(time_s)) if len(time_s) > 1 else float(len(time_s)) * bin_s
    n_bins = int(np.ceil(duration / bin_s - 1e-9))
    work = frames.assign(bin=np.floor(time_s / bin_s + 1e-9).astype(int), distance_bl=step_distance_bl(frames),
                         tilted=(frames["tilt_deg"] > tilt_threshold) & frames["side_on"].astype(bool))
    work["accel_abs"] = work["accel_bl_s2"].abs()
    work["jerk_abs"] = work["jerk_bl_s3"].abs()
    work["angular_abs"] = work["angular_velocity_deg_s"].abs()
    grouped = work.groupby("bin")
    speed = grouped["speed_bl_s"]
    out = pd.DataFrame({
        "n_frames": grouped.size(),
        "tracked_fraction": grouped["tracked"].mean(),
        "speed_mean_bl_s": speed.mean(),
        "speed_median_bl_s": speed.median(),
        "speed_max_bl_s": speed.max(),
        "speed_cv": speed.std() / speed.mean(),
        "accel_abs_mean_bl_s2": grouped["accel_abs"].mean(),
        "jerk_abs_mean_bl_s3": grouped["jerk_abs"].mean(),
        "turn_rate_var_deg2_s2": grouped["angular_velocity_deg_s"].var(),  # per second: same at any fps
        "angular_velocity_abs_mean_deg_s": grouped["angular_abs"].mean(),
        "meander_deg_per_bl": grouped[["turn_deg", "distance_bl"]].apply(
            lambda g: meander_of(g["turn_deg"], g["distance_bl"])),
        "distance_bl": grouped["distance_bl"].sum(),
        "surface_fraction": grouped["at_surface"].mean(),
        "nose_up_surface_fraction": grouped["nose_up_at_surface"].mean(),
        "depth_bl_min": grouped["depth_bl"].min(),
        "tilt_median_deg": grouped["tilt_deg"].median(),
        "tilt_fraction": grouped["tilted"].mean(),
        "aspect_median": grouped["aspect"].median(),
        "area_ratio_median": grouped["area_ratio"].median(),
    }).reindex(range(n_bins))  # empty bins -> NaN row
    out["n_frames"] = out["n_frames"].fillna(0).astype(int)
    out["tracked_fraction"] = out["tracked_fraction"].fillna(0.0)
    out.index.name = "bin"
    out = out.reset_index()
    out.insert(0, "subject_id", frames["subject_id"].iloc[0] if len(frames) else "")
    out.insert(2, "t_start_s", out["bin"] * bin_s)
    return out[BIN_COLUMNS]


# ---------------------------------------------------------------------------
# 4. Per subject endpoints (slide-14 style, whole exposure video)
# ---------------------------------------------------------------------------


def endpoint_row(frames: pd.DataFrame, bl_px: float, waterline_y: float, roi_bottom_y: float,
                 params: dict[str, Any]) -> dict[str, Any]:
    """One summary row. Top half = from the waterline to halfway down to the ROI bottom."""
    time_s = frames["time_s"].to_numpy(float)
    seconds = row_seconds(time_s)
    tracked = frames["tracked"].to_numpy(bool)
    speed = frames["speed_bl_s"].to_numpy(float)
    in_top = tracked & (frames["depth_bl"].to_numpy(float) <= (roi_bottom_y - waterline_y) / 2 / bl_px)
    tracked_s = float(seconds[tracked].sum())
    distance = step_distance_bl(frames)
    angular = frames["angular_velocity_deg_s"].abs()
    return {
        "subject_id": str(frames["subject_id"].iloc[0]) if len(frames) else "",
        "body_length_px": round(bl_px, 2),
        "duration_s": round(float(seconds.sum()), 2),
        "tracked_s": round(tracked_s, 2),
        "distance_bl": round(float(np.nansum(distance)), 2),
        "mean_velocity_bl_s": round(float(np.nanmean(speed)), 4) if np.isfinite(speed).any() else None,
        "highly_mobile_s": round(float(seconds[speed > float(params["high_mobility_bl_s"])].sum()), 2),
        "immobile_s": round(float(seconds[speed < float(params["immobile_bl_s"])].sum()), 2),
        # Share of tracked time: untracked time cannot be placed in either half.
        "top_half_pct": round(100 * float(seconds[in_top].sum()) / tracked_s, 2) if tracked_s else None,
        "latency_top_half_s": round(float(time_s[in_top][0]), 3) if in_top.any() else None,  # None = never
        "meander_deg_per_bl": round(meander_of(frames["turn_deg"], pd.Series(distance)), 3),
        "mean_angular_velocity_deg_s": round(float(angular.mean()), 3) if angular.notna().any() else None,
    }


# ---------------------------------------------------------------------------
# 5. One subject (runs in a worker process) with caching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureJob:
    """Everything a worker needs for one subject (plain values, so it can be sent to a process)."""

    subject_id: str
    first_video: Path  # part 1: its scene result gives the waterline and ROI
    tracks_dir: Path
    scene_dir: Path
    features_dir: Path
    params: dict[str, Any]  # features section
    force: bool


def frames_path(features_dir: Path, subject_id: str) -> Path:
    """Per-frame features of one subject."""
    return features_dir / f"{subject_id}_frames.csv.gz"


def bins_path(features_dir: Path, subject_id: str) -> Path:
    """Per-bin features of one subject."""
    return features_dir / f"{subject_id}_bins.csv.gz"


def meta_path(features_dir: Path, subject_id: str) -> Path:
    """Small JSON next to the tables: endpoint row and settings (the cache key)."""
    return features_dir / f"{subject_id}_features.json"


def feature_params(params: dict[str, Any]) -> dict[str, Any]:
    """The settings the outputs depend on, JSON-compatible (stored in the meta file)."""
    return json.loads(json.dumps({"version": FEATURES_VERSION, **{key: params[key] for key in FEATURE_PARAMS}}))


def _load_cached(job: FeatureJob) -> dict[str, Any] | None:
    """The saved meta if all outputs exist, are newer than the track and used the same settings."""
    meta_file = meta_path(job.features_dir, job.subject_id)
    outputs = [meta_file, frames_path(job.features_dir, job.subject_id), bins_path(job.features_dir, job.subject_id)]
    track_file = track_path(job.tracks_dir, job.subject_id)
    if job.force or not all(p.is_file() for p in outputs):
        return None
    if track_file.is_file() and track_file.stat().st_mtime > meta_file.stat().st_mtime:
        return None  # re-tracked since: redo
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # damaged: redo
    return meta if meta.get("params") == feature_params(job.params) else None


def _scene_geometry(job: FeatureJob) -> tuple[float, float]:
    """Waterline row and ROI bottom row of part 1 (original pixels, like the track)."""
    path = scene_json_path(job.scene_dir, job.first_video)
    if not path.is_file():
        raise RuntimeError(f"no scene result for {job.first_video.name}; run `track` first")
    scene = json.loads(path.read_text(encoding="utf-8"))
    return float(scene["waterline_y"]), float(scene["roi"][3])


def _load_inputs(job: FeatureJob) -> tuple[pd.DataFrame, float, float, float]:
    """The subject's track, BL (px), waterline row and ROI bottom row; raises if unusable."""
    track_file = track_path(job.tracks_dir, job.subject_id)
    if not track_file.is_file():
        raise RuntimeError(f"no track ({track_file.name}); run `track` first")
    track = pd.read_csv(track_file, dtype={"subject_id": str})
    bl_px = body_length_px(track)
    if not bl_px > 0:
        raise RuntimeError("no detected frame with a body length; check the track QA image")
    return (track, bl_px, *_scene_geometry(job))


def process_subject(job: FeatureJob) -> dict[str, Any]:
    """Compute and write one subject's features; errors are returned, not raised."""
    cached = _load_cached(job)
    if cached is not None:
        return {**cached, "cached": True}
    try:
        track, bl_px, waterline_y, roi_bottom_y = _load_inputs(job)
        frames = frame_features(track, bl_px, waterline_y, job.params)
        bins = bin_features(frames, job.params)
        endpoint = endpoint_row(frames, bl_px, waterline_y, roi_bottom_y, job.params)
    except Exception as error:  # noqa: BLE001 - reported per subject in the CLI summary
        log.error("%s: %s", job.subject_id, error)
        return {"subject_id": job.subject_id, "error": str(error)}

    job.features_dir.mkdir(parents=True, exist_ok=True)
    frames.to_csv(frames_path(job.features_dir, job.subject_id), index=False, float_format="%.5g")
    bins.to_csv(bins_path(job.features_dir, job.subject_id), index=False, float_format="%.5g")
    meta = {"endpoint": endpoint, "params": feature_params(job.params)}
    meta_path(job.features_dir, job.subject_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("%s: BL %.1f px, %.0f BL moved, %d bins", job.subject_id, bl_px, endpoint["distance_bl"], len(bins))
    return {**meta, "cached": False}


# ---------------------------------------------------------------------------
# 6. Whole run (called by the CLI)
# ---------------------------------------------------------------------------


@dataclass
class FeatureRun:
    """What `run_features` did, for the CLI summary."""

    records: list[dict[str, Any]]  # this run's subjects, in order (meta or error records)
    endpoints: pd.DataFrame  # endpoints.csv contents (every matched subject with features)
    skipped: list[str]  # subjects not processed (video_status is not matched)
    features_dir: Path


def features_dir(settings: Settings) -> Path:
    """The step's output folder."""
    return settings.paths.output_dir / "features"


def _jobs(settings: Settings, selected: pd.DataFrame, force: bool) -> tuple[list[FeatureJob], list[str]]:
    """One job per selected matched subject, and the skipped (unmatched) subjects."""
    out = features_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    matched = selected["video_status"] == STATUS_MATCHED
    skipped = [f"{row.subject_id} ({row.video_status})" for row in selected[~matched].itertuples()]
    jobs = [
        FeatureJob(row.subject_id, Path(row.video_paths.split(";")[0]), get_tracks_dir(settings),
                   get_scene_dir(settings), out, settings.params["features"], force)
        for row in selected[matched].itertuples()
    ]
    return jobs, skipped


def run_features(settings: Settings, trials: pd.DataFrame, selected: pd.DataFrame,
                 force: bool = False) -> FeatureRun:
    """Compute features for every `selected` matched subject, then rewrite endpoints.csv."""
    out = features_dir(settings)
    jobs, skipped = _jobs(settings, selected, force)
    records = run_parallel(process_subject, jobs, settings.workers, desc="features")

    # endpoints.csv covers every matched subject with features (this run or earlier ones).
    rows = []
    for row in trials[trials["video_status"] == STATUS_MATCHED].itertuples():
        path = meta_path(out, row.subject_id)
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8"))["endpoint"])
    endpoints = pd.DataFrame(rows, columns=ENDPOINT_COLUMNS)
    endpoints.to_csv(out / ENDPOINTS_FILE, index=False)
    return FeatureRun(records=records, endpoints=endpoints, skipped=skipped, features_dir=out)


# ---------------------------------------------------------------------------
# 7. QA: is each bin feature measuring the fish, or tracking noise?
# ---------------------------------------------------------------------------

QA_FILE = "qa.csv"
# Bin features checked; a real movement measure keeps its value when half the frames
# are dropped and when the smoothing is doubled, while noise does not.
QA_FEATURES = (
    "speed_mean_bl_s", "distance_bl", "accel_abs_mean_bl_s2", "jerk_abs_mean_bl_s3", "turn_rate_var_deg2_s2",
    "angular_velocity_abs_mean_deg_s", "meander_deg_per_bl", "surface_fraction", "nose_up_surface_fraction", "depth_bl_min",
    "tilt_median_deg", "aspect_median", "area_ratio_median",
)
QA_COLUMNS = [
    "subject_id", "feature", "r_half_rate", "change_half_rate", "change_smooth_2x", "stable",
    "jitter_bl", "body_length_cv",
]


def compare(base: pd.Series, other: pd.Series) -> tuple[float, float]:
    """Correlation over the bins both have, and the relative change of the mean (other vs base).

    NaN when it cannot be judged (fewer than 3 shared bins, a constant series, zero mean).
    """
    both = pd.concat([base, other], axis=1, join="inner").dropna()
    a, b = both.iloc[:, 0], both.iloc[:, 1]
    varies = len(both) > 2 and a.std() > 0 and b.std() > 0
    r = float(np.corrcoef(a, b)[0, 1]) if varies else float("nan")
    change = float(b.mean() / a.mean() - 1) if len(both) and a.mean() != 0 else float("nan")
    return r, change


def qa_subject(job: FeatureJob) -> dict[str, Any]:
    """QA rows (QA_COLUMNS) of one subject, recomputed from its track; errors are returned."""
    try:
        track, bl_px, waterline_y, _ = _load_inputs(job)
        params = job.params

        def bins(rows: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
            """Bin features of these track rows, indexed by bin number."""
            return bin_features(frame_features(rows, bl_px, waterline_y, settings), settings).set_index("bin")

        base = bins(track, params)
        half = bins(track.iloc[::2].reset_index(drop=True), params)  # every 2nd frame: half the frame rate
        smoother = bins(track, {**params, "smooth_s": 2 * float(params["smooth_s"])})

        # Jitter: RMS distance between the raw and the smoothed centroid, in BL.
        polyorder = int(params["smooth_polyorder"])
        window = odd_window(float(params["smooth_s"]), frame_gap_s(track["time_s"].to_numpy(float)), polyorder)
        residual = [track[c].to_numpy(float) - smooth(track[c].to_numpy(float), window, polyorder) for c in "xy"]
        jitter = float(np.sqrt(np.nanmean(residual[0] ** 2 + residual[1] ** 2))) / bl_px
        lengths = track.loc[track["detected"].astype(bool), "major_axis"]
    except Exception as error:  # noqa: BLE001 - reported per subject in the CLI summary
        log.error("%s: %s", job.subject_id, error)
        return {"subject_id": job.subject_id, "error": str(error)}

    min_r, max_change = float(params["qa_min_r"]), float(params["qa_max_change"])
    rows = []
    for feature in QA_FEATURES:
        r, change_half = compare(base[feature], half[feature])
        _, change_smooth = compare(base[feature], smoother[feature])
        # NaN (cannot be judged) does not fail a check: NaN comparisons are False.
        failed = r < min_r or abs(change_half) > max_change or abs(change_smooth) > max_change
        rows.append({
            "subject_id": job.subject_id, "feature": feature, "r_half_rate": round(r, 3),
            "change_half_rate": round(change_half, 3), "change_smooth_2x": round(change_smooth, 3),
            "stable": not failed, "jitter_bl": round(jitter, 4), "body_length_cv": round(float(lengths.std() / bl_px), 3),
        })
    return {"subject_id": job.subject_id, "rows": rows}


@dataclass
class QARun:
    """What `run_features_qa` did, for the CLI summary."""

    records: list[dict[str, Any]]  # per subject: {"rows": [...]} or {"error": ...}
    table: pd.DataFrame  # qa.csv contents (the subjects of this run)
    skipped: list[str]
    path: Path


def run_features_qa(settings: Settings, selected: pd.DataFrame) -> QARun:
    """Check every `selected` matched subject and write qa.csv (this run's subjects only)."""
    jobs, skipped = _jobs(settings, selected, force=False)
    records = run_parallel(qa_subject, jobs, settings.workers, desc="features-qa")
    table = pd.DataFrame([row for record in records for row in record.get("rows", [])], columns=QA_COLUMNS)
    path = features_dir(settings) / QA_FILE
    table.to_csv(path, index=False)
    return QARun(records=records, table=table, skipped=skipped, path=path)
