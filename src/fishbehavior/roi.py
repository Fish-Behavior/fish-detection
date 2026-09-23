"""Scene setup: per video, find the empty-beaker background, the waterline and the fish ROI.

For every matched video (see catalog.py):

1. **Background** = per-pixel median of `scene.n_background_frames` frames sampled
   evenly over the video. The fish keeps moving, so it disappears from the median.
2. **Activity map** = fraction of those frames in which a pixel clearly differs from
   the background (`scene.activity_method`): by default it must change in brightness
   AND in color, so light flicker, exposure changes and glare on the water (brightness
   only) and the faint reflection in the glass (little brightness change) do not count.
   Small separate specks are dropped; the main patch is where the fish swims.
3. **ROI** = bounding box of the activity map, padded; its top is raised to just
   above the waterline so a surface breach stays inside.
4. **Waterline** = the row with the strongest horizontal edge in the background
   (vertical brightness gradient over the central columns), searched near the top
   of the activity box.

A person can check or correct each result: `<video>_qa.png` draws everything on the
background, and `overrides.yaml` replaces automatic values.

Outputs (in ``<FISH_OUTPUT_DIR>/scene/``, git-ignored):
    <video_stem>.json             video info, waterline_y, roi, confidence, method
    <video_stem>_background.png   the empty-beaker background (grayscale)
    <video_stem>_activity.png     activity map (white = fish seen there often), for review
    <video_stem>_frames.jpg       a few real color frames stacked vertically, for review
    <video_stem>_qa.png           QA image: waterline blue, ROI green, activity faint red
    video_check.csv               one row per video with duration / fps / waterline flags
    overrides.yaml                manual corrections and "checked" marks (written by a
                                  person, or by the `scene-review` page, see review.py)

Coordinates are pixels of the original video; ``roi = [x0, y0, x1, y1]`` with x1/y1
exclusive, so ``frame[y0:y1, x0:x1]`` is the ROI.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import yaml

from fishbehavior.catalog import STATUS_MATCHED
from fishbehavior.config import ConfigError, Settings
from fishbehavior.video import VideoInfo, iter_sampled_frames, probe

log = logging.getLogger(__name__)

OVERRIDES_FILE = "overrides.yaml"
VIDEO_CHECK_FILE = "video_check.csv"
GEOMETRY_KEYS = ("waterline_y", "roi")  # values that replace automatic results
OVERRIDE_KEYS = {*GEOMETRY_KEYS, "checked"}  # checked: a person confirmed this video

# Flag names used in the JSON files and video_check.csv (';'-joined there).
FLAG_LOW_CONFIDENCE = "low_waterline_confidence"
FLAG_NO_ACTIVITY = "no_activity"  # nothing moved: ROI falls back to the whole frame
FLAG_DURATION = "duration"  # subject total differs from scene.expected_duration_s
FLAG_FPS = "fps_mismatch"  # parts of one subject have different frame rates
FLAG_ERROR = "error"  # the video could not be processed (see the summary / log)

ACTIVITY_METHODS = ("both", "color", "gray")
# Settings that change a video's result or images: a cached result made with other
# values is redone automatically (the other scene settings only affect video_check.csv).
DETECTION_PARAMS = (
    "n_background_frames", "activity_method", "diff_threshold", "color_threshold",
    "analysis_max_width", "activity_min_fraction", "activity_min_component_fraction",
    "roi_padding_fraction", "waterline_search", "waterline_central_fraction",
    "waterline_margin_fraction", "min_waterline_confidence", "n_review_frames",
)

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (x1/y1 exclusive)


# ---------------------------------------------------------------------------
# 1. Image analysis (pure functions on numpy arrays)
# ---------------------------------------------------------------------------


def compute_background(frames: np.ndarray) -> np.ndarray:
    """Per-pixel median of a (n, h, w) uint8 stack -> (h, w) uint8 empty-scene image."""
    return np.median(frames, axis=0).round().astype(np.uint8)


def activity_map(frames: np.ndarray, method: str = "both", diff_threshold: float = 18,
                 color_threshold: float = 8) -> np.ndarray:
    """Fraction (0-1) of frames in which each pixel clearly differs from the background.

    `frames` is an (n, h, w, 3) BGR uint8 stack. Work is done in CIELAB, which splits
    a pixel into lightness (L) and color (a*, b*):

    * lightness change: |L - background L| > diff_threshold, after removing each frame's
      overall brightness change (its median difference), so a light flicker or camera
      exposure change does not light up the whole frame;
    * color change: distance in the a*/b* plane > color_threshold. Brightness-only
      changes (glare, shadows, exposure) barely move a pixel here; a colored fish does.

    method "both" requires both changes, "color" / "gray" only one of them.
    """
    if method not in ACTIVITY_METHODS:
        raise ValueError(f"activity_method must be one of {', '.join(ACTIVITY_METHODS)}, got {method!r}")
    # int16 avoids uint8 wrap-around when subtracting.
    lab = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) for frame in frames]).astype(np.int16)
    lab -= np.median(lab, axis=0).astype(np.int16)  # difference from the (Lab) background
    light = lab[..., 0] - np.median(lab[..., 0].reshape(len(lab), -1), axis=1)[:, None, None]
    changed_light = np.abs(light) > diff_threshold
    changed_color = np.hypot(lab[..., 1], lab[..., 2]) > color_threshold
    changed = {"both": changed_light & changed_color, "color": changed_color, "gray": changed_light}[method]
    return changed.mean(axis=0).astype(np.float32)


def activity_mask(activity: np.ndarray, min_fraction: float, min_component_fraction: float = 0.0) -> np.ndarray:
    """Pixels where the fish was: active in > `min_fraction` of frames, main patches only.

    A 3x3 morphological opening removes isolated pixels (compression noise). Then every
    separate patch smaller than `min_component_fraction` of the largest patch is dropped,
    so a speck far from the fish (a brief light change at the rim) cannot stretch the box.
    """
    mask = (activity > min_fraction).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:  # label 0 is the background
        return np.zeros(mask.shape, bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.nonzero(areas >= min_component_fraction * areas.max())[0]
    return np.isin(labels, keep)


def mask_box(mask: np.ndarray) -> Box | None:
    """Bounding box (x1/y1 exclusive) of the True pixels, or None if there are none."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1



def pad_box(box: Box, width: int, height: int, fraction: float) -> Box:
    """Grow a box by `fraction` of the frame width/height on each side, clipped to the frame."""
    dx, dy = round(fraction * width), round(fraction * height)
    x0, y0, x1, y1 = box
    return max(0, x0 - dx), max(0, y0 - dy), min(width, x1 + dx), min(height, y1 + dy)


def find_waterline(
    background: np.ndarray, box: Box, search: Iterable[float], central_fraction: float
) -> tuple[int, float]:
    """Return (waterline row, confidence) from the background image.

    The water surface is a horizontal edge, i.e. a strong change of brightness from
    one row to the next. The absolute vertical gradient is averaged over the central
    columns of `box` (the curved beaker walls are left out) and the strongest row in
    the search band is taken. Confidence = that peak / the band's median, so a clear
    lone edge scores high and a flat or busy profile scores low.
    """
    height = background.shape[0]
    x0, y0, x1, y1 = box
    box_h, box_w = y1 - y0, x1 - x0
    # Central columns of the activity box.
    half = max(1, round(box_w * central_fraction / 2))
    cx = (x0 + x1) // 2
    c0, c1 = max(0, cx - half), min(background.shape[1], cx + half)
    # Sobel with dy=1 measures the change between rows (it also smooths across columns).
    gradient = np.abs(cv2.Sobel(background.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
    profile = gradient[:, c0:c1].mean(axis=1)
    # Search band relative to the activity box top (negative = above the box).
    start_frac, end_frac = search
    r0 = int(np.clip(round(y0 + start_frac * box_h), 1, height - 2))  # skip the border rows
    r1 = int(np.clip(round(y0 + end_frac * box_h), r0 + 1, height - 1))
    band = profile[r0:r1]
    row = r0 + int(np.argmax(band))
    # Floor of 1 gray level keeps the ratio finite on a perfectly flat (synthetic) image.
    confidence = float(band.max() / max(float(np.median(band)), 1.0))
    return row, confidence


# ---------------------------------------------------------------------------
# 2. One video: detect, apply overrides, draw the QA image
# ---------------------------------------------------------------------------


@dataclass
class SceneDetection:
    """Result for one video before it is written to disk."""

    info: VideoInfo
    background: np.ndarray
    activity: np.ndarray
    activity_box: Box | None
    waterline_y: int
    waterline_confidence: float
    roi: Box
    method: str  # "auto" or "override"
    overrides: dict[str, Any]
    base_roi: Box  # padded activity box before the waterline raises its top (review preview)
    review_frames: list[np.ndarray]  # a few real color frames, for the review page
    auto: dict[str, Any]  # automatic waterline_y / roi, kept even when overridden (review "reset")
    flags: list[str] = field(default_factory=list)


def _check_override(override: dict[str, Any], width: int, height: int) -> None:
    """Refuse override values that do not fit this video's frame."""
    if "waterline_y" in override and not 0 <= override["waterline_y"] < height:
        raise ValueError(f"override waterline_y={override['waterline_y']} is outside the frame height {height}")
    if "roi" in override:
        x0, y0, x1, y1 = override["roi"]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"override roi={override['roi']} is outside the {width}x{height} frame")


def detect_scene(video_path: Path, params: dict[str, Any], override: dict[str, Any] | None = None) -> SceneDetection:
    """Run the background / activity / waterline / ROI analysis for one video file."""
    override = override or {}
    info = probe(video_path)
    _check_override(override, info.width, info.height)
    grays, smalls, review_frames = sample_scene_frames(video_path, info, params)
    background = compute_background(np.stack(grays))
    del grays  # the largest array; free it before the activity work
    small_activity = activity_map(np.stack(smalls), str(params["activity_method"]),
                                  float(params["diff_threshold"]), float(params["color_threshold"]))
    # Back to full size so boxes and images are in original video pixels.
    activity = cv2.resize(small_activity, (info.width, info.height), interpolation=cv2.INTER_LINEAR)
    mask = activity_mask(activity, float(params["activity_min_fraction"]),
                         float(params["activity_min_component_fraction"]))
    activity = np.where(mask, activity, 0).astype(np.float32)  # keep only what the ROI is based on
    box = mask_box(mask)

    flags = []
    if box is None:
        # Nothing moved (or the fish never left one spot): search the whole frame instead.
        flags.append(FLAG_NO_ACTIVITY)
        search_box: Box = (0, 0, info.width, info.height)
    else:
        search_box = box
    waterline, confidence = find_waterline(
        background, search_box, params["waterline_search"], float(params["waterline_central_fraction"])
    )
    base_roi = pad_box(search_box, info.width, info.height, float(params["roi_padding_fraction"]))
    auto = {"waterline_y": int(waterline), "roi": list(raise_roi_top(base_roi, waterline, info.height, params))}
    if "waterline_y" in override:
        waterline = int(override["waterline_y"])  # a person checked it; confidence is kept for the record
    elif confidence < float(params["min_waterline_confidence"]):
        flags.append(FLAG_LOW_CONFIDENCE)

    if "roi" in override:
        roi = tuple(int(v) for v in override["roi"])  # taken as-is: the person drew the final box
    else:
        roi = raise_roi_top(base_roi, waterline, info.height, params)

    return SceneDetection(
        info=info,
        background=background,
        activity=activity,
        activity_box=box,
        waterline_y=int(waterline),
        waterline_confidence=round(confidence, 2),
        roi=roi,  # type: ignore[arg-type]
        method="override" if override else "auto",
        overrides=dict(override),
        base_roi=base_roi,
        review_frames=review_frames,
        auto=auto,
        flags=flags,
    )


def sample_scene_frames(video_path: Path, info: VideoInfo,
                        params: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Read the sampled frames once and keep what each part of the analysis needs.

    Returns full-size grayscale frames (background and waterline need full detail),
    color frames shrunk to `analysis_max_width` (the activity map is coarse, and this
    keeps memory low for large videos), and `n_review_frames` full-size color frames.
    """
    n = int(params["n_background_frames"])
    scale = min(1.0, float(params["analysis_max_width"]) / info.width)
    # A short video has fewer frames than requested; pick review frames among those sampled.
    sampled = min(n, info.frame_count)
    keep = set(np.linspace(0, sampled - 1, int(params["n_review_frames"])).round().astype(int).tolist())
    grays, smalls, review = [], [], []
    for i, frame in enumerate(iter_sampled_frames(video_path, n, gray=False)):
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        smalls.append(frame if scale == 1.0 else
                      cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        if i in keep:
            review.append(frame)
    return grays, smalls, review


def raise_roi_top(box: Box, waterline: int, height: int, params: dict[str, Any]) -> Box:
    """Raise the box top to just above the waterline so a fish breaking the surface is inside."""
    margin = round(float(params["waterline_margin_fraction"]) * height)
    x0, y0, x1, y1 = box
    return x0, max(0, min(y0, waterline - margin)), x1, y1


def draw_qa(detection: SceneDetection) -> np.ndarray:
    """QA image: background, activity as a faint red tint, waterline blue, ROI green.

    Small videos are enlarged to at least 640 px wide before drawing so lines and
    text stay readable; everything is drawn after enlarging, so lines stay crisp.
    """
    background, activity = detection.background, detection.activity
    k = max(1, int(np.ceil(640 / background.shape[1])))  # integer zoom keeps pixels square
    image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR).astype(np.float32)
    # Red tint proportional to activity, at most 55% so the background stays visible.
    peak = float(activity.max()) or 1.0
    alpha = (0.55 * activity / peak)[..., None]
    red = np.array([0, 0, 255], np.float32)
    image = (image * (1 - alpha) + red * alpha).astype(np.uint8)
    image = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)

    x0, y0, x1, y1 = detection.roi
    cv2.rectangle(image, (x0 * k, y0 * k), (x1 * k - 1, y1 * k - 1), (0, 200, 0), 2)
    y = detection.waterline_y * k + k // 2
    cv2.line(image, (0, y), (image.shape[1] - 1, y), (255, 0, 0), 2)

    # Two text lines on a black strip above the image, so the label never hides the scene.
    lines = [
        f"{detection.info.path.name}  [{detection.method}]  " + ", ".join(detection.flags),
        f"waterline y={detection.waterline_y} (conf {detection.waterline_confidence:.1f})  "
        f"roi={list(detection.roi)}",
    ]
    header = np.zeros((44, image.shape[1], 3), np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(header, text, (6, 17 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, image])


# ---------------------------------------------------------------------------
# 3. Overrides file
# ---------------------------------------------------------------------------


def validate_overrides(data: Any, source: str) -> dict[str, dict[str, Any]]:
    """Check an overrides mapping ({video file name: entry}); raise ConfigError naming `source`.

    An entry holds waterline_y (int), roi ([x0, y0, x1, y1] ints) and/or checked (true/false).
    Shared by load_overrides and the review page's save, so both accept exactly the same file.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"{source} must map video file names to settings, e.g. 'F_0042.mp4: {{waterline_y: 120}}'")
    overrides: dict[str, dict[str, Any]] = {}
    for name, entry in data.items():
        if not isinstance(entry, dict) or not entry or set(entry) - OVERRIDE_KEYS:
            raise ConfigError(f"{source}: entry {name!r} must contain only waterline_y, roi and/or checked")
        # bool is a subclass of int in Python, so exclude it explicitly.
        if "waterline_y" in entry and (not isinstance(entry["waterline_y"], int) or isinstance(entry["waterline_y"], bool)):
            raise ConfigError(f"{source}: {name}: waterline_y must be a whole number of pixels")
        roi = entry.get("roi")
        if roi is not None and not (isinstance(roi, list) and len(roi) == 4
                                    and all(isinstance(v, int) and not isinstance(v, bool) for v in roi)):
            raise ConfigError(f"{source}: {name}: roi must be [x0, y0, x1, y1] in whole pixels")
        if "checked" in entry and not isinstance(entry["checked"], bool):
            raise ConfigError(f"{source}: {name}: checked must be true or false")
        overrides[str(name)] = entry
    return overrides


def load_overrides(scene_dir: Path) -> dict[str, dict[str, Any]]:
    """Read `<scene_dir>/overrides.yaml` ({video file name: entry}); {} if absent."""
    path = scene_dir / OVERRIDES_FILE
    if not path.is_file():
        return {}
    return validate_overrides(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, str(path))


def geometry_of(entry: dict[str, Any]) -> dict[str, Any]:
    """The part of an override entry that changes results (everything except `checked`)."""
    return {key: entry[key] for key in GEOMETRY_KEYS if key in entry}


def override_for(overrides: dict[str, dict[str, Any]], video_path: Path) -> dict[str, Any]:
    """The override entry for a video, looked up by file name (or by name without extension)."""
    return overrides.get(video_path.name) or overrides.get(video_path.stem) or {}


# ---------------------------------------------------------------------------
# 4. Per-video job (runs in a worker process) with caching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneJob:
    """Everything a worker needs for one video (plain values, so it can be sent to a process)."""

    subject_id: str
    video_path: Path
    scene_dir: Path
    params: dict[str, Any]
    override: dict[str, Any]
    force: bool


def json_path(scene_dir: Path, video_path: Path) -> Path:
    """Where the scene result of one video is stored."""
    return scene_dir / f"{video_path.stem}.json"


def _load_cached(job: SceneJob) -> dict[str, Any] | None:
    """The saved result if it can be reused: exists, readable, made with the same overrides."""
    path = json_path(job.scene_dir, job.video_path)
    if job.force or not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # damaged file: redo it
    if record.get("params") != detection_params(job.params):
        return None  # made with other settings (or by an older version): redo it
    # An edited waterline/roi must take effect without --force.
    return record if record.get("overrides", {}) == geometry_of(job.override) else None


def detection_params(params: dict[str, Any]) -> dict[str, Any]:
    """The settings a result depends on, as stored in its JSON (JSON-compatible values)."""
    return json.loads(json.dumps({key: params[key] for key in DETECTION_PARAMS}))


def _update_checked(record: dict[str, Any], job: SceneJob) -> dict[str, Any]:
    """Store a changed `checked` mark in a cached JSON (no need to redo the analysis for it)."""
    checked = bool(job.override.get("checked", False))
    if record.get("checked", False) != checked:
        record = {**record, "checked": checked}
        json_path(job.scene_dir, job.video_path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def process_video(job: SceneJob) -> dict[str, Any]:
    """Detect, save the JSON + images, and return the record (or an error record).

    Errors are returned, not raised, so one bad file does not stop a run over hundreds.
    """
    cached = _load_cached(job)
    if cached is not None:
        return {**_update_checked(cached, job), "cached": True}
    try:
        detection = detect_scene(job.video_path, job.params, geometry_of(job.override))
    except Exception as error:  # noqa: BLE001 - reported per video in the summary and video_check.csv
        log.error("%s: %s", job.video_path.name, error)
        return {"subject_id": job.subject_id, "file": job.video_path.name, "error": str(error)}

    stem = job.video_path.stem
    background_name, qa_name = f"{stem}_background.png", f"{stem}_qa.png"
    activity_name, frames_name = f"{stem}_activity.png", f"{stem}_frames.jpg"
    cv2.imwrite(str(job.scene_dir / background_name), detection.background)
    # Activity fraction scaled so the most active pixel is white (shown as a tint when reviewing).
    peak = float(detection.activity.max()) or 1.0
    cv2.imwrite(str(job.scene_dir / activity_name), (detection.activity / peak * 255).astype(np.uint8))
    # Real frames stacked top to bottom; the review page shows one slice at a time.
    cv2.imwrite(str(job.scene_dir / frames_name), np.vstack(detection.review_frames),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(job.scene_dir / qa_name), draw_qa(detection))
    record = {
        "subject_id": job.subject_id,
        "file": job.video_path.name,
        "video": detection.info.to_dict(),
        "waterline_y": detection.waterline_y,
        "waterline_confidence": detection.waterline_confidence,
        "roi": list(detection.roi),
        "activity_box": list(detection.activity_box) if detection.activity_box else None,
        "base_roi": list(detection.base_roi),
        "auto": detection.auto,
        "method": detection.method,
        "overrides": detection.overrides,
        "checked": bool(job.override.get("checked", False)),
        "flags": detection.flags,
        "background_image": background_name,
        "activity_image": activity_name,
        "frames_image": frames_name,
        "n_frames_image": len(detection.review_frames),
        "params": detection_params(job.params),
        "qa_image": qa_name,
    }
    json_path(job.scene_dir, job.video_path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.info("%s: waterline y=%d (conf %.1f), roi %s", job.video_path.name,
             detection.waterline_y, detection.waterline_confidence, list(detection.roi))
    return {**record, "cached": False}


def _init_worker() -> None:
    """One OpenCV thread per worker process: the parallelism comes from the processes."""
    cv2.setNumThreads(1)


def run_jobs(jobs: list[SceneJob], workers: int) -> list[dict[str, Any]]:
    """Run jobs in `workers` processes (in this process when 1), preserving job order."""
    if workers <= 1 or len(jobs) <= 1:
        return [process_video(job) for job in jobs]
    try:
        pool = ProcessPoolExecutor(max_workers=min(workers, len(jobs)), initializer=_init_worker)
    except (OSError, NotImplementedError) as error:
        # Some locked-down environments forbid the semaphores a process pool needs.
        log.warning("cannot start %d worker processes (%s); processing videos one by one", workers, error)
        return [process_video(job) for job in jobs]
    results: dict[int, dict[str, Any]] = {}
    with pool:
        futures = {pool.submit(process_video, job): i for i, job in enumerate(jobs)}
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            log.debug("scene: %d/%d videos finished", done, len(jobs))
    return [results[i] for i in range(len(jobs))]


# ---------------------------------------------------------------------------
# 5. Video check table (per subject: duration + fps; per video: waterline)
# ---------------------------------------------------------------------------


def video_check(records: list[dict[str, Any]], params: dict[str, Any]) -> pd.DataFrame:
    """One row per video with its timing and flags; subject-level flags go on every part."""
    rows = []
    for record in records:
        video = record.get("video", {})
        flags = list(record.get("flags", []))
        if "error" in record:
            flags.append(FLAG_ERROR)
        rows.append({
            "subject_id": record["subject_id"],
            "file": record["file"],
            "fps": round(video["fps"], 3) if video else None,
            "frames": video.get("frame_count"),
            "duration_s": round(video["duration_s"], 1) if video else None,
            "width": video.get("width"),
            "height": video.get("height"),
            "flags": flags,
            # True once a person confirmed or corrected this video (scene-review / overrides.yaml).
            "checked": bool(record.get("checked", False)),
        })
    columns = ["subject_id", "file", "fps", "frames", "duration_s", "width", "height", "flags", "checked"]
    table = pd.DataFrame(rows, columns=columns)

    expected = float(params["expected_duration_s"])
    tolerance = float(params["duration_tolerance"])
    fps_tolerance = float(params["fps_tolerance"])
    for _, group in table.groupby("subject_id", sort=False):
        if group["duration_s"].isna().any():
            continue  # a part failed; its error flag already says so
        subject_flags = []
        if abs(group["duration_s"].sum() - expected) > tolerance * expected:
            subject_flags.append(FLAG_DURATION)
        if group["fps"].max() - group["fps"].min() > fps_tolerance * group["fps"].min():
            subject_flags.append(FLAG_FPS)
        for index in group.index:
            table.at[index, "flags"] = table.at[index, "flags"] + subject_flags
    table["flags"] = table["flags"].map(";".join)
    return table


# ---------------------------------------------------------------------------
# 6. Whole run (called by the CLI)
# ---------------------------------------------------------------------------


@dataclass
class SceneRun:
    """What `run_scene` did, for the CLI summary."""

    records: list[dict[str, Any]]  # this run's videos, in order
    check: pd.DataFrame  # video_check.csv contents (all processed subjects)
    skipped: list[str]  # subjects not processed (video_status is not matched)
    scene_dir: Path


def scene_dir(settings: Settings) -> Path:
    """The step's output folder."""
    return settings.paths.output_dir / "scene"


def run_scene(settings: Settings, trials: pd.DataFrame, selected: pd.DataFrame, force: bool = False) -> SceneRun:
    """Process every video of the `selected` matched subjects, then rewrite video_check.csv.

    video_check.csv covers every matched subject in `trials` that has a result so far
    (from this or earlier runs), so running a few subjects never shrinks it.
    """
    params = settings.params["scene"]
    out = scene_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(out)

    matched = selected["video_status"] == STATUS_MATCHED
    skipped = [f"{row.subject_id} ({row.video_status})" for row in selected[~matched].itertuples()]
    jobs = [
        SceneJob(row.subject_id, Path(path), out, params, override_for(overrides, Path(path)), force)
        for row in selected[matched].itertuples()
        for path in row.video_paths.split(";") if path
    ]
    records = run_jobs(jobs, settings.workers)

    # Combine with earlier results for the other matched subjects.
    this_run = {record["file"]: record for record in records}
    all_records = []
    for row in trials[trials["video_status"] == STATUS_MATCHED].itertuples():
        for path in (Path(p) for p in row.video_paths.split(";") if p):
            if path.name in this_run:
                all_records.append(this_run[path.name])
            elif json_path(out, path).is_file():
                all_records.append(json.loads(json_path(out, path).read_text(encoding="utf-8")))
    check = video_check(all_records, params)
    check.to_csv(out / VIDEO_CHECK_FILE, index=False)
    return SceneRun(records=records, check=check, skipped=skipped, scene_dir=out)
