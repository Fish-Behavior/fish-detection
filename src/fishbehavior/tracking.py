"""Per-frame fish tracking: position, outline size and tilt in every frame of every subject.

For each matched subject (see catalog.py) every part file is tracked against its
scene result (roi.py: background, ROI), and the parts are joined into one timeline
where part b starts at part a's duration. Per frame:

1. grayscale (and shrink by `tracking.scale`), keep only the ROI;
2. Gaussian blur, difference from the (equally blurred) empty-beaker background, with
   the frame's overall brightness change removed (lights / auto exposure); by default
   only pixels darker than the background count (`tracking.polarity`);
3. threshold, morphological open (drops specks) + close (fills the body), connected
   components;
4. choose the fish blob (`choose_blob`: the one overlapping last frame's fish, else
   the largest near its last position, else the largest) and measure it: centroid, top/bottom row,
   area, and an ellipse fitted to its outline (length, width, tilt);
5. find the head (`find_head`): the darkest spot of the blob, i.e. the eye, which tells
   which end of the body is the front, so the tilt becomes a nose-up / nose-down pitch.

Short gaps (<= `tracking.max_gap_s`) are filled by linear interpolation; longer ones
stay empty and are counted as untracked seconds.

Outputs (in ``<FISH_OUTPUT_DIR>/tracks/``, git-ignored):
    <subject_id>.csv.gz          one row per frame (columns: TRACK_COLUMNS)
    <subject_id>_track.json      per-subject summary, part files and the settings used (cache key)
    <subject_id>_track_qa.png    the track drawn over the background, colored by time
    summary.csv                  one row per tracked subject (SUMMARY_COLUMNS)

Coordinates are pixels of the original video, whatever `tracking.scale` is.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from fishbehavior.catalog import STATUS_MATCHED
from fishbehavior.config import Settings
from fishbehavior.parallel import run_parallel
from fishbehavior.roi import SceneJob, load_overrides, override_for, process_video
from fishbehavior.roi import scene_dir as get_scene_dir
from fishbehavior.video import iter_frames

log = logging.getLogger(__name__)

SUMMARY_FILE = "summary.csv"
TRACK_COLUMNS = [
    "subject_id", "part", "part_frame", "frame", "time_s", "detected", "x", "y", "top_y", "bottom_y",
    "area", "major_axis", "minor_axis", "angle_deg", "head_x", "head_y", "pitch_deg", "n_blobs", "interpolated",
]
# Measurements filled in by interpolation over short gaps (angle is handled separately).
MEASURED = ["x", "y", "top_y", "bottom_y", "area", "major_axis", "minor_axis", "head_x", "head_y", "pitch_deg"]
SUMMARY_COLUMNS = [
    "subject_id", "frames", "detected_pct", "interpolated_pct", "untracked_s",
    "median_body_length_px", "mean_n_blobs", "head_pct", "pitch_pct",
]
# Decimals kept in the track file: sub-pixel positions, 0.1 ms times.
ROUNDING = {"time_s": 4, "x": 2, "y": 2, "top_y": 2, "bottom_y": 2, "area": 1,
            "major_axis": 2, "minor_axis": 2, "angle_deg": 1, "head_x": 2, "head_y": 2, "pitch_deg": 1}
# Settings that change the track; a cached track made with other values is redone.
TRACK_PARAMS = (
    "frame_stride", "scale", "blur_kernel_fraction", "diff_threshold", "polarity", "remove_brightness_change",
    "morph_kernel_fraction", "min_area_fraction", "max_jump_bl", "switch_area_ratio", "body_length_window",
    "max_gap_s", "head_min_contrast", "head_min_offset_fraction",
)

POLARITIES = ("darker", "both")

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (x1/y1 exclusive)


# ---------------------------------------------------------------------------
# 1. Image analysis for one frame (pure functions on numpy arrays)
# ---------------------------------------------------------------------------


def odd_kernel(fraction: float, shape: tuple[int, ...], minimum: int = 3) -> int:
    """Odd kernel size = `fraction` of the smaller frame side, so it scales with resolution."""
    size = max(minimum, round(fraction * min(shape[:2])))
    return size if size % 2 else size + 1  # Gaussian blur needs an odd size


@dataclass(frozen=True)
class Detector:
    """Everything fixed for one part file: blurred background crop, kernels, thresholds.

    All values are in the (possibly scaled) analysis frame; `roi` is where the crop
    starts, to shift blob positions back to whole-frame coordinates.
    """

    background: np.ndarray  # blurred ROI crop of the background, float32
    roi: Box
    blur: int
    morph: np.ndarray
    diff_threshold: float
    darker_only: bool  # tracking.polarity == "darker"
    remove_brightness_change: bool
    min_area: float  # pixels
    head_min_contrast: float
    head_min_offset_fraction: float


def make_detector(background: np.ndarray, roi: Box, params: dict[str, Any]) -> Detector:
    """Prepare a Detector from a full-frame background (already at the analysis scale)."""
    if params["polarity"] not in POLARITIES:
        raise ValueError(f"tracking.polarity must be one of {', '.join(POLARITIES)}, got {params['polarity']!r}")
    blur = odd_kernel(float(params["blur_kernel_fraction"]), background.shape)
    morph = odd_kernel(float(params["morph_kernel_fraction"]), background.shape)
    x0, y0, x1, y1 = roi
    crop = background[y0:y1, x0:x1]
    return Detector(
        background=cv2.GaussianBlur(crop, (blur, blur), 0).astype(np.float32),
        roi=roi,
        blur=blur,
        morph=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph)),
        diff_threshold=float(params["diff_threshold"]),
        darker_only=params["polarity"] == "darker",
        remove_brightness_change=bool(params["remove_brightness_change"]),
        # Relative to the whole frame, so the same setting works at any resolution.
        min_area=float(params["min_area_fraction"]) * background.shape[0] * background.shape[1],
        head_min_contrast=float(params["head_min_contrast"]),
        head_min_offset_fraction=float(params["head_min_offset_fraction"]),
    )


def foreground(gray: np.ndarray, detector: Detector) -> tuple[np.ndarray, np.ndarray]:
    """uint8 mask (ROI crop) of pixels that clearly differ from the background, and the
    signed difference image itself (negative = darker than the background)."""
    x0, y0, x1, y1 = detector.roi
    # Blur both images the same way: compression noise and single-pixel flicker average out.
    crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (detector.blur, detector.blur), 0)
    diff = crop.astype(np.float32) - detector.background
    if detector.remove_brightness_change:
        # The fish covers a small part of the ROI, so the median change is the lighting's.
        diff -= float(np.median(diff))
    # Darker only: the fish is darker than the water, so lighter pixels (glare, and any
    # trace of the fish left in the background) cannot be the fish.
    change = -diff if detector.darker_only else np.abs(diff)
    mask = (change > detector.diff_threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, detector.morph)  # drop specks
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, detector.morph), diff  # close: fill small holes in the body


def choose_blob(areas: np.ndarray, centroids: np.ndarray, min_area: float,
                previous: tuple[float, float] | None, max_jump_px: float,
                overlaps: np.ndarray | None = None, switch_area_ratio: float = 1.5) -> tuple[int | None, int]:
    """Pick the fish among the blobs; return (index into areas, number of large-enough blobs).

    Blobs smaller than `min_area` are ignored. Then, in order:

    1. **Overlap**: the blob covering most of last frame's fish outline (`overlaps` =
       pixels of each blob inside it). At video frame rates the fish always overlaps
       itself from one frame to the next, while a mirror image of the fish in the glass,
       a ripple or a shadow lies next to it, not on it. So a fish that briefly looks
       small (turned toward the camera, or its see-through body split into pieces) is
       kept even when a mirror image is bigger at that moment. Exception: a blob more
       than `switch_area_ratio` times larger wins, so the track cannot stay locked on a
       small mirror image once it has jumped there.
    2. **Distance** (no overlap, e.g. after a gap or a very fast move): the largest blob
       within `max_jump_px` of the previous position, so a ripple or shadow that is
       briefly bigger than the fish somewhere else does not steal the track.
    3. Otherwise (first frame, or nothing near) simply the largest, so the track
       recovers after the fish was lost or truly moved far.
    """
    candidates = np.nonzero(areas >= min_area)[0]
    if len(candidates) == 0:
        return None, 0
    largest = int(candidates[np.argmax(areas[candidates])])
    if overlaps is not None and overlaps[candidates].max() > 0:
        kept = int(candidates[np.argmax(overlaps[candidates])])
        return (largest if areas[largest] > switch_area_ratio * areas[kept] else kept), len(candidates)
    if previous is not None:
        distance = np.hypot(centroids[candidates, 0] - previous[0], centroids[candidates, 1] - previous[1])
        near = candidates[distance <= max_jump_px]
        if len(near):
            return int(near[np.argmax(areas[near])]), len(candidates)
    return largest, len(candidates)


def measure_blob(labels: np.ndarray, label: int, stats: np.ndarray, centroid: np.ndarray) -> dict[str, float]:
    """Centroid, top/bottom row, area and fitted ellipse of one blob (crop coordinates)."""
    left, top, width, height, area = stats[label]
    # Outline of this blob only, from its bounding box (fast: no whole-crop copy).
    patch = (labels[top:top + height, left:left + width] == label).astype(np.uint8)
    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)
    major = minor = angle = np.nan
    if len(contour) >= 5:  # fitEllipse needs at least 5 points
        _, (axis_a, axis_b), rotation = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
        angle = tilt_from_fit(axis_a, axis_b, rotation)
    return {"x": float(centroid[0]), "y": float(centroid[1]), "top_y": float(top),
            "bottom_y": float(top + height - 1), "area": float(area),
            "major_axis": float(major), "minor_axis": float(minor), "angle_deg": float(angle)}


def find_head(diff: np.ndarray, labels: np.ndarray, label: int, stats: np.ndarray, blob: dict[str, float],
              min_contrast: float, min_offset_fraction: float) -> dict[str, float]:
    """Head position and nose-up pitch of one blob (crop coordinates); NaN when unclear.

    The eye is the darkest spot of the fish, far darker than its see-through body, so
    the head = the centre of the blob's darkest part (relative to the background). It must be at
    least `min_contrast` gray levels darker than the blob's typical pixel, or the fish
    shows no clear eye (too faint, or only a body piece was detected) and there is no head.

    The head tells which end of the fitted body axis is the front, which turns the tilt
    into a pitch: degrees the nose points above (+) or below (-) horizontal, whichever
    way the fish faces. When the head lies within `min_offset_fraction` of the body length
    of the centre along the axis (e.g. the fish faces the camera), the front end cannot
    be told and the pitch is NaN; the head position is still given.
    """
    left, top, width, height, _ = stats[label]
    inside = labels[top:top + height, left:left + width] == label
    values = diff[top:top + height, left:left + width]
    darkest, typical = float(values[inside].min()), float(np.median(values[inside]))
    nan = {"head_x": np.nan, "head_y": np.nan, "pitch_deg": np.nan}
    if typical - darkest < min_contrast:
        return nan
    # Centre of the darkest part (within a quarter of the contrast of the minimum), not the
    # single darkest pixel: the eye is a small dark disk, and its centre is stable.
    rows, cols = np.nonzero(inside & (values <= darkest + 0.25 * (typical - darkest)))
    head_x, head_y = float(left + cols.mean()), float(top + rows.mean())
    out = {**nan, "head_x": head_x, "head_y": head_y}
    angle, length = blob["angle_deg"], blob["major_axis"]
    if not np.isfinite(angle):
        return out  # no fitted body axis
    # Unit vector along the body axis, toward the right end (image y points down).
    axis = np.array([np.cos(np.radians(angle)), -np.sin(np.radians(angle))])
    along = float(np.dot([head_x - blob["x"], head_y - blob["y"]], axis))
    if abs(along) < min_offset_fraction * length:
        return out  # head near the middle: front end unclear
    front = axis if along > 0 else -axis  # points from tail to nose
    out["pitch_deg"] = float(np.degrees(np.arctan2(-front[1], abs(front[0]))))  # up = +, either facing
    return out


def tilt_from_fit(axis_a: float, axis_b: float, rotation: float) -> float:
    """Body tilt in degrees from cv2.fitEllipse output, in [-90, 90).

    0 = horizontal body; positive = the right end is higher on screen (counter-clockwise
    as seen in the video). OpenCV gives the rotation of its first axis, clockwise on
    screen (image y points down), so the long axis direction is flipped to our sign.
    """
    long_axis = rotation if axis_a >= axis_b else rotation + 90.0
    return float((-long_axis + 90.0) % 180.0 - 90.0)


def detect(gray: np.ndarray, detector: Detector, previous: tuple[float, float] | None, max_jump_px: float,
           previous_mask: np.ndarray | None = None,
           switch_area_ratio: float = 1.5) -> tuple[dict[str, float] | None, int, np.ndarray | None]:
    """Find the fish in one grayscale frame; positions in (scaled) whole-frame pixels.

    `previous_mask` is last frame's fish outline (ROI crop, bool); the new outline is
    returned as the third value, for the next frame.
    """
    mask, diff = foreground(gray, detector)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    x0, y0 = detector.roi[:2]
    # Label 0 is the background; move blob centroids to whole-frame coordinates for the jump test.
    shifted = centroids[1:] + (x0, y0)
    # Pixels of each blob that lie on last frame's fish (one pass over those pixels).
    overlaps = None if previous_mask is None else np.bincount(labels[previous_mask], minlength=count)[1:]
    index, n_blobs = choose_blob(stats[1:, cv2.CC_STAT_AREA].astype(float), shifted,
                                 detector.min_area, previous, max_jump_px, overlaps, switch_area_ratio)
    if index is None:
        return None, n_blobs, None
    blob = measure_blob(labels, index + 1, stats, centroids[index + 1])
    blob.update(find_head(diff, labels, index + 1, stats, blob, detector.head_min_contrast,
                          detector.head_min_offset_fraction))
    for key in ("x", "head_x"):
        blob[key] += x0
    for key in ("y", "top_y", "bottom_y", "head_y"):
        blob[key] += y0
    return blob, n_blobs, labels == index + 1


# ---------------------------------------------------------------------------
# 2. One part file -> per-frame rows
# ---------------------------------------------------------------------------


def scaled_roi(roi: list[int], scale: float, shape: tuple[int, int]) -> Box:
    """ROI in analysis pixels: rounded outward so no original ROI pixel is lost."""
    x0, y0, x1, y1 = roi
    height, width = shape
    return (max(0, int(np.floor(x0 * scale))), max(0, int(np.floor(y0 * scale))),
            min(width, int(np.ceil(x1 * scale))), min(height, int(np.ceil(y1 * scale))))


def to_original(blob: dict[str, float], scale: float) -> dict[str, float]:
    """Undo `tracking.scale`: positions (pixel centers), lengths and area back to original pixels."""
    if scale == 1.0:
        return blob
    out = dict(blob)
    for key in ("x", "y", "top_y", "bottom_y", "head_x", "head_y"):
        out[key] = (blob[key] + 0.5) / scale - 0.5
    out["major_axis"], out["minor_axis"] = blob["major_axis"] / scale, blob["minor_axis"] / scale
    out["area"] = blob["area"] / scale ** 2
    return out  # tilt and pitch do not change with uniform scaling


def track_frames(frames: Iterable[tuple[int, float, np.ndarray]], background: np.ndarray, roi: list[int],
                 params: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Track the fish frame by frame: one row per (index, time_s, gray frame at `tracking.scale`).

    `background` is the full-size grayscale scene background; `roi` is in original pixels.
    A generator, so the live view can show every row as it comes; `track_part` collects them.
    """
    scale = float(params["scale"])
    if scale != 1.0:
        # Same call as iter_frames uses, so the background has exactly the frames' size.
        background = cv2.resize(background, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    detector = make_detector(background, scaled_roi(roi, scale, background.shape), params)
    window = deque(maxlen=int(params["body_length_window"]))  # recent body lengths (analysis px)
    max_jump_bl = float(params["max_jump_bl"])
    switch_area_ratio = float(params["switch_area_ratio"])
    previous: tuple[float, float] | None = None
    previous_mask: np.ndarray | None = None  # last frame's fish outline; None after a missed frame
    for index, time_s, gray in frames:
        # The jump limit is in body lengths: a running median of the recent fitted lengths.
        max_jump = max_jump_bl * float(np.median(window)) if window else np.inf
        blob, n_blobs, previous_mask = detect(gray, detector, previous, max_jump, previous_mask, switch_area_ratio)
        row: dict[str, Any] = {"part_frame": index, "part_time_s": time_s, "n_blobs": n_blobs}
        if blob is not None:
            previous = (blob["x"], blob["y"])
            if np.isfinite(blob["major_axis"]):
                window.append(blob["major_axis"])
            row.update(to_original(blob, scale))
        yield row


def track_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Rows from `track_frames` as one part's table (every column present, `detected` added)."""
    table = pd.DataFrame(rows, columns=["part_frame", "part_time_s", "n_blobs", *MEASURED, "angle_deg"])
    table["detected"] = table["x"].notna()
    return table


def track_part(video_path: Path, background: np.ndarray, roi: list[int], params: dict[str, Any],
               frame_count: int | None = None, progress: bool = False) -> pd.DataFrame:
    """Track the fish in one video file; one row per analysed frame (part_frame, time in the file)."""
    stride, scale = int(params["frame_stride"]), float(params["scale"])
    total = None if frame_count is None else -(-frame_count // stride)
    frames = tqdm(iter_frames(video_path, stride=stride, scale=scale), total=total, desc=video_path.name,
                  unit="frame", leave=False, disable=None if progress else True)
    return track_table(list(track_frames(frames, background, roi, params)))


# ---------------------------------------------------------------------------
# 3. Joining parts, gap filling, summary
# ---------------------------------------------------------------------------


def join_parts(parts: list[pd.DataFrame], infos: list[dict[str, Any]], subject_id: str) -> pd.DataFrame:
    """One continuous timeline: part k starts at the summed frames / duration of parts before it."""
    frame_offset = 0
    joined = []
    for number, (table, info, time_offset) in enumerate(zip(parts, infos, part_offsets(infos)), start=1):
        table = table.copy()
        table["part"] = number  # 1 = first file in trials.csv video_paths (part a), 2 = part b, ...
        table["frame"] = table["part_frame"] + frame_offset
        table["time_s"] = table["part_time_s"] + time_offset
        # Seconds each row stands for (stride frames), used for gap lengths and untracked time.
        table["row_s"] = info["stride"] / info["fps"]
        joined.append(table)
        frame_offset += int(info["frame_count"])
    track = pd.concat(joined, ignore_index=True)
    track.insert(0, "subject_id", subject_id)
    return track


def part_offsets(infos: list[dict[str, Any]]) -> list[float]:
    """Start time (s) of each part on the joined timeline: the summed durations before it."""
    return list(np.cumsum([0.0] + [float(info["duration_s"]) for info in infos[:-1]]))


def fill_gaps(track: pd.DataFrame, max_gap_s: float) -> pd.DataFrame:
    """Linearly interpolate missing runs lasting <= `max_gap_s` between two detections.

    Longer runs, and runs at the very start or end (nothing to interpolate from),
    stay NaN. The tilt is interpolated on the doubled angle (cos 2a, sin 2a) because
    -89 deg and +89 deg are almost the same body direction.
    """
    track = track.copy()
    detected = track["detected"].to_numpy()
    run_id = np.cumsum(np.r_[True, detected[1:] != detected[:-1]])  # consecutive equal values share an id
    fill = np.zeros(len(track), bool)
    for _, rows in track.groupby(run_id).groups.items():
        first, last = rows[0], rows[-1]
        is_gap = not detected[first] and first > 0 and last < len(track) - 1  # detections on both sides
        if is_gap and track["row_s"].iloc[first:last + 1].sum() <= max_gap_s + 1e-9:
            fill[first:last + 1] = True

    time = track["time_s"].to_numpy()
    known = detected
    for column in MEASURED:
        values = track[column].to_numpy(float)
        ok = known & np.isfinite(values)
        if ok.sum() >= 2:
            values = values.copy()
            values[fill] = np.interp(time[fill], time[ok], values[ok])
            track[column] = values
    angle = np.radians(track["angle_deg"].to_numpy(float) * 2)
    ok = known & np.isfinite(angle)
    if ok.sum() >= 2:
        cos = np.interp(time[fill], time[ok], np.cos(angle[ok]))
        sin = np.interp(time[fill], time[ok], np.sin(angle[ok]))
        filled = track["angle_deg"].to_numpy(float).copy()
        filled[fill] = (np.degrees(np.arctan2(sin, cos)) / 2 + 90) % 180 - 90
        track["angle_deg"] = filled
    track["interpolated"] = fill
    return track


def summarize(track: pd.DataFrame) -> dict[str, Any]:
    """One summary row: how much of the video is tracked, and the typical body length."""
    frames = len(track)
    untracked = ~(track["detected"] | track["interpolated"])
    body = track.loc[track["detected"], "major_axis"]
    seen = track.loc[track["detected"]]
    return {
        "subject_id": str(track["subject_id"].iloc[0]) if frames else "",
        "frames": frames,
        "detected_pct": round(100 * track["detected"].mean(), 2) if frames else 0.0,
        "interpolated_pct": round(100 * track["interpolated"].mean(), 2) if frames else 0.0,
        "untracked_s": round(float(track.loc[untracked, "row_s"].sum()), 2),
        "median_body_length_px": round(float(body.median()), 2) if body.notna().any() else None,
        "mean_n_blobs": round(float(track["n_blobs"].mean()), 3) if frames else 0.0,
        # Of the detected frames: head (eye) found, and nose direction known too.
        "head_pct": round(100 * seen["head_x"].notna().mean(), 2) if len(seen) else 0.0,
        "pitch_pct": round(100 * seen["pitch_deg"].notna().mean(), 2) if len(seen) else 0.0,
    }


def finalize(track: pd.DataFrame) -> pd.DataFrame:
    """Output columns in order, with sensible rounding (keeps the .csv.gz small)."""
    out = track[TRACK_COLUMNS].copy()
    for column, decimals in ROUNDING.items():
        out[column] = out[column].round(decimals)
    return out


# ---------------------------------------------------------------------------
# 4. QA image
# ---------------------------------------------------------------------------


def draw_track_qa(background: np.ndarray, track: pd.DataFrame, roi: list[int], waterline_y: int,
                  title: str) -> np.ndarray:
    """The track over the background, colored by time (dark purple = start, yellow = end).

    Enlarged to >= 640 px wide like the scene QA image. Lines are not drawn across
    untracked gaps, so a jump over a gap is visible as a break. ROI green, waterline blue.
    """
    k = max(1, int(np.ceil(640 / background.shape[1])))  # integer zoom keeps pixels square
    image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    x0, y0, x1, y1 = roi
    cv2.rectangle(image, (x0 * k, y0 * k), (x1 * k - 1, y1 * k - 1), (0, 200, 0), 1)
    wy = waterline_y * k + k // 2
    cv2.line(image, (0, wy), (image.shape[1] - 1, wy), (255, 0, 0), 1)

    # Viridis color per row, by position in time.
    ramp = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_VIRIDIS)[:, 0]
    xs = ((track["x"].to_numpy(float) + 0.5) * k).round()
    ys = ((track["y"].to_numpy(float) + 0.5) * k).round()
    time = track["time_s"].to_numpy(float)
    span = max(float(time[-1] - time[0]), 1e-9) if len(time) else 1.0
    shade = ((time - (time[0] if len(time) else 0)) / span * 255).astype(int) if len(time) else []
    ok = np.isfinite(xs) & np.isfinite(ys)
    for i in range(1, len(track)):
        if ok[i] and ok[i - 1]:
            color = tuple(int(c) for c in ramp[shade[i]])
            cv2.line(image, (int(xs[i - 1]), int(ys[i - 1])), (int(xs[i]), int(ys[i])), color, 1, cv2.LINE_AA)

    # Header with the numbers, and a strip showing the time colors.
    header = np.zeros((44, image.shape[1], 3), np.uint8)
    cv2.putText(header, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    bar_x0, bar_x1 = 60, image.shape[1] - 70
    bar = cv2.resize(ramp[None], (bar_x1 - bar_x0, 10), interpolation=cv2.INTER_LINEAR)
    header[27:37, bar_x0:bar_x1] = bar
    end = f"{int(span // 60)}:{int(span % 60):02d}"
    cv2.putText(header, "start", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, f"end {end}", (bar_x1 + 6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([header, image])


# ---------------------------------------------------------------------------
# 5. One subject (runs in a worker process) with caching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackJob:
    """Everything a worker needs for one subject (plain values, so it can be sent to a process)."""

    subject_id: str
    video_paths: tuple[Path, ...]  # part order
    tracks_dir: Path
    scene_dir: Path
    params: dict[str, Any]  # tracking section
    scene_params: dict[str, Any]  # scene section (to run scene setup for a part if needed)
    overrides: dict[str, dict[str, Any]]  # scene overrides.yaml
    force: bool
    progress: bool  # per-frame progress bar (only when running in the main process)


def track_path(tracks_dir: Path, subject_id: str) -> Path:
    """Where the per-frame track of one subject is stored."""
    return tracks_dir / f"{subject_id}.csv.gz"


def meta_path(tracks_dir: Path, subject_id: str) -> Path:
    """Small JSON next to the track: summary, part files and settings (the cache key)."""
    return tracks_dir / f"{subject_id}_track.json"


def track_params(params: dict[str, Any]) -> dict[str, Any]:
    """The settings a track depends on, JSON-compatible (stored in the meta file)."""
    return json.loads(json.dumps({key: params[key] for key in TRACK_PARAMS}))


def scene_key(scene: dict[str, Any]) -> dict[str, Any]:
    """The parts of a scene result that change the track (ROI and how the background was made)."""
    return {"file": scene["file"], "roi": scene["roi"], "params": scene.get("params")}


def _load_cached(job: TrackJob, scenes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The saved meta if the track can be reused: both files exist, same settings and scenes."""
    path = meta_path(job.tracks_dir, job.subject_id)
    if job.force or not path.is_file() or not track_path(job.tracks_dir, job.subject_id).is_file():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # damaged: redo
    if meta.get("params") != track_params(job.params):
        return None  # other tracking settings
    return meta if meta.get("scenes") == [scene_key(s) for s in scenes] else None  # ROI changed


def _scene_for(job: TrackJob, video_path: Path) -> dict[str, Any]:
    """The scene result of one part; scene setup runs now if it is missing or out of date."""
    record = process_video(SceneJob(job.subject_id, video_path, job.scene_dir, job.scene_params,
                                    override_for(job.overrides, video_path), force=False))
    if "error" in record:
        raise RuntimeError(f"scene setup failed for {video_path.name}: {record['error']}")
    return record


def process_subject(job: TrackJob) -> dict[str, Any]:
    """Track all parts of one subject, write the outputs and return its meta record.

    Errors are returned, not raised, so one bad subject does not stop a long run.
    """
    try:
        scenes = [_scene_for(job, path) for path in job.video_paths]
        cached = _load_cached(job, scenes)
        if cached is not None:
            return {**cached, "cached": True}
        parts, infos = [], []
        for path, scene in zip(job.video_paths, scenes):
            background = cv2.imread(str(job.scene_dir / scene["background_image"]), cv2.IMREAD_GRAYSCALE)
            if background is None:
                raise RuntimeError(f"cannot read {scene['background_image']}; run `scene --force`")
            video = scene["video"]
            parts.append(track_part(path, background, scene["roi"], job.params,
                                    frame_count=video["frame_count"], progress=job.progress))
            infos.append({**video, "stride": int(job.params["frame_stride"])})
        track = fill_gaps(join_parts(parts, infos, job.subject_id), float(job.params["max_gap_s"]))
    except Exception as error:  # noqa: BLE001 - reported per subject in the CLI summary
        log.error("%s: %s", job.subject_id, error)
        return {"subject_id": job.subject_id, "error": str(error)}

    job.tracks_dir.mkdir(parents=True, exist_ok=True)
    finalize(track).to_csv(track_path(job.tracks_dir, job.subject_id), index=False)
    summary = summarize(track)
    first = scenes[0]
    background = cv2.imread(str(job.scene_dir / first["background_image"]), cv2.IMREAD_GRAYSCALE)
    title = (f"{job.subject_id}  detected {summary['detected_pct']:.1f}%  untracked {summary['untracked_s']:.1f}s  "
             f"body {summary['median_body_length_px']} px" + ("  (background: part 1)" if len(scenes) > 1 else ""))
    # Only the first part's rows are drawn on the first part's background when parts differ.
    qa = draw_track_qa(background, track[track["part"] == 1] if len(scenes) > 1 else track,
                       first["roi"], first["waterline_y"], title)
    qa_name = f"{job.subject_id}_track_qa.png"
    cv2.imwrite(str(job.tracks_dir / qa_name), qa)
    meta = {
        "summary": summary,
        # Which file each `part` number is, and where it starts on the joined timeline.
        "parts": [{"part": i, "file": s["file"], "fps": s["video"]["fps"],
                   "frame_count": s["video"]["frame_count"], "offset_s": round(offset, 4)}
                  for i, (s, offset) in enumerate(zip(scenes, part_offsets(infos)), start=1)],
        "qa_image": qa_name,
        "params": track_params(job.params),
        "scenes": [scene_key(s) for s in scenes],
    }
    meta_path(job.tracks_dir, job.subject_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("%s: detected %.1f%%, untracked %.1f s, body length %s px", job.subject_id,
             summary["detected_pct"], summary["untracked_s"], summary["median_body_length_px"])
    return {**meta, "cached": False}


# ---------------------------------------------------------------------------
# 6. Whole run (called by the CLI)
# ---------------------------------------------------------------------------


@dataclass
class TrackRun:
    """What `run_tracking` did, for the CLI summary."""

    records: list[dict[str, Any]]  # this run's subjects, in order (meta or error records)
    summary: pd.DataFrame  # summary.csv contents (every matched subject with a track)
    skipped: list[str]  # subjects not processed (video_status is not matched)
    tracks_dir: Path


def tracks_dir(settings: Settings) -> Path:
    """The step's output folder."""
    return settings.paths.output_dir / "tracks"


def run_tracking(settings: Settings, trials: pd.DataFrame, selected: pd.DataFrame, force: bool = False) -> TrackRun:
    """Track every `selected` matched subject, then rewrite summary.csv for all tracked subjects."""
    out = tracks_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    scene_out = get_scene_dir(settings)
    scene_out.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(scene_out)

    matched = selected["video_status"] == STATUS_MATCHED
    skipped = [f"{row.subject_id} ({row.video_status})" for row in selected[~matched].itertuples()]
    workers = settings.workers
    jobs = [
        TrackJob(row.subject_id, tuple(Path(p) for p in row.video_paths.split(";") if p), out, scene_out,
                 settings.params["tracking"], settings.params["scene"], overrides, force,
                 progress=workers <= 1)
        for row in selected[matched].itertuples()
    ]
    records = run_parallel(process_subject, jobs, workers, desc="track")

    # summary.csv covers every matched subject tracked so far (this run or earlier ones).
    rows = []
    for row in trials[trials["video_status"] == STATUS_MATCHED].itertuples():
        path = meta_path(out, row.subject_id)
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8"))["summary"])
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary.to_csv(out / SUMMARY_FILE, index=False)
    return TrackRun(records=records, summary=summary, skipped=skipped, tracks_dir=out)
