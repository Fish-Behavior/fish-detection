"""Per-frame fish localization via foreground extraction (PRD §9.5.3) - FR-005.

Foreground extraction uses `cv2.createBackgroundSubtractorKNN`, not a static
per-video median-diff. `detected=False` (never a guessed position) wherever
no plausible-sized, plausible-position contour is found - occlusion/glare/
bubble frames must never silently produce a fabricated coordinate (§3.3).

**Design history (important, found during T035's real-video spot-check):**
the first implementation used `cv2.absdiff` against `roi.py`'s static
per-video median background. That failed badly whenever a fish rested in
roughly the same spot across enough of the sparse background-sampling
frames (every 30th frame) - the median absorbed the fish's own silhouette
into the "background," so later absdiff saw almost nothing. Measured on a
real video (Subject 0057, `Veh`, Wild-type): only 44.9% of frames detected.
Switched to `BackgroundSubtractorKNN` (the alternative PRD §9.5.3 point 3
explicitly names for "videos with subtler background drift" - a semi-
stationary foreground object turns out to be the same underlying failure
mode). **This is a real, measured improvement but only a partial fix, not
the resolution it first looked like:** re-measured on the full Subject 0057
video (35865 frames, not just spot-check frames), detection is **56.4%**
- better than 44.9%, but far short of an earlier figure recorded in this
docstring that turned out to be wrong (not reproducible, likely a
transcription error from an interrupted session - retracted; see
docs/progress.md Phase 4 for the correction and the ongoing investigation).
Visual inspection of an undetected frame mid-video (fish plainly visible,
resting still near the tank bottom, no occlusion) confirms this is a real
detection-pipeline gap, not unrecoverable footage - consecutive-undetected
run lengths on this video go up to 2214 frames, consistent with KNN's
adaptive background model slowly re-absorbing a fish that stays still
longer than the `history` window, the same fundamental failure mode as the
original bug just requiring sustained stillness instead of sparse-sample
alignment. This matters beyond raw detection rate: sustained stillness is
exactly what Freezing/Drift, Listing/LORR, and Dead need to measure
(FR-006/FR-007) - if the tracker's own background model swallows a
genuinely still fish, that's not a benign gap, it corrupts the states this
pipeline most needs to get right. Under investigation - see docs/progress.md
Phase 4 before treating this module as validated.

One consequence of the switch, confirmed on both the synthetic fixture and
Subject 0057: the KNN model has no prior history on its first few `apply()`
calls, so **the first 4 frames of every video are structurally never
detected** (not an occlusion/glare edge case - a fixed warmup property of
the algorithm, independent of video length or `history=500`, verified on a
35865-frame real video). `tests/test_tracking.py` asserts this leading
window explicitly rather than assuming a single frame 0.

Per T029 (docs/strain_tracking_notes.md §2), the real-video sample is
~98% one resolution (320x240) with one outlier at 192x240 - the
pixel-unit constants below (`DEFAULT_MAX_JUMP_PX`,
`DEFAULT_{MIN,MAX}_CONTOUR_AREA_PX2`) are NOT resolution-normalized.
Phase 6's calibration search (T045) must account for this explicitly -
either search per-resolution or normalize by frame size - rather than
silently fitting only the dominant resolution.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from prepds.models import Track
from prepds.roi import detect_waterline, estimate_background
from prepds.video_io import frames, probe

# Provisional defaults (PRD §9.5.5/§9.5.6: real calibration is a later,
# explicit search - config/default_thresholds.yaml's `tracking` section is
# still `null`-placeholder as of Phase 4). These are a starting point for
# T035's real-video spot-check to validate/adjust, not a finished
# calibration - grounded in the synthetic fixture's known dot (radius 6px,
# area ~113px²) and a generous upper/jump bound, refined against real
# footage in T035 (see module docstring), not picked from real fish
# measurements alone.
DEFAULT_KNN_HISTORY = 500
DEFAULT_KNN_DIST2_THRESHOLD = 400.0
DEFAULT_MIN_CONTOUR_AREA_PX2 = 15.0
DEFAULT_MAX_CONTOUR_AREA_PX2 = 5000.0
DEFAULT_MAX_JUMP_PX = 60.0
# How many consecutive missed frames (no contour, or a contour rejected as an
# implausible jump) before the jump-rejection anchor is dropped and the next
# found contour is trusted unconditionally. Without this, a real occlusion/
# stillness gap (observed up to 2214 frames on real footage - see module
# docstring) leaves `last_detected_xy` stuck on a stale position, so every
# later real detection keeps getting rejected as "too far" once the fish has
# actually moved - a re-acquisition cascade failure caught by code review.
# NOT fps-normalized (a frame count, not a duration) - same class of gap as
# the resolution-normalization risk above; flagged for Phase 6/T045 alongside
# it rather than fixed here, since both are provisional pending calibration.
DEFAULT_MAX_CONSECUTIVE_MISSES_BEFORE_REACQUIRE = 30
# `cv2.createBackgroundSubtractorKNN`'s internal sample-replacement history
# update is stochastic (OpenCV's global RNG, not exposed per-subtractor) -
# found via a real-video determinism check: two `track_video()` calls on the
# identical file, no seeding, differed in `detected` on ~4% of frames. This
# violates NFR-002 ("deterministic tracking... no unseeded randomness").
# Fix: `cv2.setRNGSeed()` DOES pin this RNG (verified: identical output
# across repeated calls, on both the synthetic fixture and a real video,
# once seeded) - global process state, not the OpenCV object itself, but
# reseeding to the same fixed value at the start of every `track_video()`
# call reproduces the guarantee NFR-002 asks for.
DEFAULT_TRACKING_RANDOM_SEED = 20260101


class TrackResult(NamedTuple):
    """Per-video context alongside a `track_video()` call, for callers that also want it.

    `waterline_y` is the detected-edge row from `roi.py::detect_waterline()`
    - kept separate from `Track.y_from_frame_top` (which is raw
    top-of-frame-relative pixel position) per the two-candidate-signal
    design in docs/strain_tracking_notes.md §3, since T029 found this edge
    is not reliably the true water surface across all videos.
    """

    tracks: list[Track]
    waterline_y: int


def track_video(
    video_path: Path,
    *,
    min_area_px2: float = DEFAULT_MIN_CONTOUR_AREA_PX2,
    max_area_px2: float = DEFAULT_MAX_CONTOUR_AREA_PX2,
    max_jump_px: float = DEFAULT_MAX_JUMP_PX,
    knn_history: int = DEFAULT_KNN_HISTORY,
    knn_dist2_threshold: float = DEFAULT_KNN_DIST2_THRESHOLD,
    max_consecutive_misses_before_reacquire: int = DEFAULT_MAX_CONSECUTIVE_MISSES_BEFORE_REACQUIRE,
    random_seed: int = DEFAULT_TRACKING_RANDOM_SEED,
) -> list[Track]:
    """Track the fish across every frame of `video_path` at its native fps.

    Returns one `Track` per frame, in frame order. Raises ValueError (not a
    crash) for a missing/unreadable video, consistent with `probe()`/
    `frames()` - one bad video must not halt a batch run.

    `random_seed` reseeds OpenCV's global RNG before building the KNN
    subtractor - NFR-002 requires deterministic output given the same video,
    and `BackgroundSubtractorKNN`'s history update is otherwise stochastic
    (see `DEFAULT_TRACKING_RANDOM_SEED`'s comment). This makes `track_video`
    NOT thread-safe with respect to other concurrent OpenCV RNG use in the
    same process - acceptable for Phase 12's per-video sequential batch
    model, but a real constraint if that ever changes.
    """
    asset = probe(video_path)  # raises ValueError early for a bad file
    cv2.setRNGSeed(random_seed)
    subtractor = cv2.createBackgroundSubtractorKNN(
        history=knn_history, dist2Threshold=knn_dist2_threshold, detectShadows=False
    )

    tracks: list[Track] = []
    last_detected_xy: tuple[float, float] | None = None
    consecutive_misses = 0
    for frame_idx, frame in frames(video_path, stride=1):
        foreground_mask = subtractor.apply(frame)
        centroid, orientation_deg = _detect_fish(
            foreground_mask, min_area_px2=min_area_px2, max_area_px2=max_area_px2
        )
        detected = False
        x = y = 0.0
        if centroid is not None:
            cx, cy = centroid
            jump_ok = last_detected_xy is None or math.hypot(
                cx - last_detected_xy[0], cy - last_detected_xy[1]
            ) <= max_jump_px
            if jump_ok:
                x, y = cx, cy
                detected = True
                last_detected_xy = (cx, cy)
                consecutive_misses = 0
            # An implausible jump is treated as a bad detection (glare/
            # occlusion artifact), not real motion - orientation_deg is
            # discarded too in that case, since it came from the same
            # rejected contour.
            else:
                orientation_deg = None

        if not detected:
            consecutive_misses += 1
            if consecutive_misses >= max_consecutive_misses_before_reacquire:
                # A long-enough gap means the anchor is no longer meaningful
                # - drop it so the next found contour is trusted outright,
                # rather than rejecting every real detection after a real
                # occlusion/stillness gap because it's "too far" from a
                # stale position.
                last_detected_xy = None

        tracks.append(
            Track(
                frame_idx=frame_idx,
                t_sec=frame_idx / asset.fps,
                x=x,
                y=y,
                orientation_deg=orientation_deg if detected else None,
                y_from_frame_top=y if detected else None,  # see TrackResult docstring
                detected=detected,
            )
        )
    return tracks


def track_video_with_context(
    video_path: Path,
    *,
    min_area_px2: float = DEFAULT_MIN_CONTOUR_AREA_PX2,
    max_area_px2: float = DEFAULT_MAX_CONTOUR_AREA_PX2,
    max_jump_px: float = DEFAULT_MAX_JUMP_PX,
    knn_history: int = DEFAULT_KNN_HISTORY,
    knn_dist2_threshold: float = DEFAULT_KNN_DIST2_THRESHOLD,
    max_consecutive_misses_before_reacquire: int = DEFAULT_MAX_CONSECUTIVE_MISSES_BEFORE_REACQUIRE,
    random_seed: int = DEFAULT_TRACKING_RANDOM_SEED,
) -> TrackResult:
    """`track_video()` plus the per-video detected waterline edge - see `TrackResult`.

    Mirrors `track_video`'s keyword-only parameters explicitly (rather than
    `**kwargs`) so a misspelled parameter fails at call time with a clear
    `TypeError`, not silently inside `track_video`'s own call.
    """
    tracks = track_video(
        video_path,
        min_area_px2=min_area_px2,
        max_area_px2=max_area_px2,
        max_jump_px=max_jump_px,
        knn_history=knn_history,
        knn_dist2_threshold=knn_dist2_threshold,
        max_consecutive_misses_before_reacquire=max_consecutive_misses_before_reacquire,
        random_seed=random_seed,
    )
    waterline_y = detect_waterline(estimate_background(video_path))
    return TrackResult(tracks=tracks, waterline_y=waterline_y)


def render_debug_overlay(video_path: Path, output_path: Path, tracks: list[Track]) -> Path:
    """Write a copy of `video_path` with detected position/orientation drawn per frame.

    Pure visual QA tool (T035: "visually spot-check ... with an overlay-
    annotated debug video export ... before scaling to all 353") - never
    read back by the pipeline itself. A green dot + orientation line marks a
    `detected=True` frame; a red "UNDETECTED" label marks the rest, so a
    human reviewer can immediately see where tracking is failing.
    """
    asset = probe(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, asset.fps, asset.resolution)
    if not writer.isOpened():
        raise ValueError(f"Could not open VideoWriter for {output_path}")
    try:
        tracks_by_idx = {t.frame_idx: t for t in tracks}
        for frame_idx, frame in frames(video_path, stride=1):
            annotated = frame.copy()
            track = tracks_by_idx.get(frame_idx)
            if track is not None and track.detected:
                center = (int(round(track.x)), int(round(track.y)))
                cv2.drawMarker(annotated, center, (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
                if track.orientation_deg is not None:
                    length = 20
                    angle_rad = math.radians(track.orientation_deg)
                    dx, dy = length * math.cos(angle_rad), length * math.sin(angle_rad)
                    tip = (int(round(track.x + dx)), int(round(track.y + dy)))
                    cv2.line(annotated, center, tip, (0, 255, 0), 2)
            else:
                cv2.putText(
                    annotated, "UNDETECTED", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )
            writer.write(annotated)
    finally:
        writer.release()
    return output_path


def _detect_fish(
    foreground_mask: np.ndarray,
    *,
    min_area_px2: float,
    max_area_px2: float,
) -> tuple[tuple[float, float] | None, float | None]:
    """Return ((cx, cy), orientation_deg) for the largest plausible contour, or (None, None)."""
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if not (min_area_px2 <= area <= max_area_px2):
        return None, None  # too small (noise) or too large (lighting/glare artifact)

    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None, None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    orientation_deg = None
    if len(largest) >= 5:  # cv2.fitEllipse requires at least 5 contour points
        (_center, _axes, angle) = cv2.fitEllipse(largest)
        orientation_deg = float(angle)

    return (cx, cy), orientation_deg
