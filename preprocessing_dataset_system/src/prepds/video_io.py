"""Video file inspection: reads real container metadata (PRD FR-005, §3.3).

Never assumes a nominal 30fps/1200s: 3 of the 353 real trials are ~600s, and
measured fps is ~29.83, not exactly 30 (confirmed against the real sample
video - docs/progress.md §0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from prepds.models import VideoAsset


def probe(video_path: Path) -> VideoAsset:
    """Open `video_path` and read its real fps/frame_count/duration/resolution.

    Raises ValueError (not a crash) for a missing, zero-byte, or otherwise
    unreadable file, so callers (catalog.py's batch matching) can report it
    per-video instead of halting the whole run (FR-004).
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"Video file has invalid or corrupt metadata "
                f"(fps={fps}, frame_count={frame_count}, size={width}x{height}): {video_path}"
            )
    finally:
        capture.release()
    return VideoAsset(
        path=video_path,
        duration_s=frame_count / fps,
        fps=fps,
        frame_count=frame_count,
        resolution=(width, height),
    )


def frames(video_path: Path, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(frame_idx, frame_bgr)` for every `stride`-th frame, in order.

    `frame_idx` is always the true index into the original video (0, stride,
    2*stride, ...), not a re-numbered count of yielded frames - callers that
    need real timestamps (t_sec = frame_idx / fps) depend on this. Used by
    roi.py's sparse background sampling (every-Nth-frame) and, from Phase 4,
    tracking.py's full per-frame pass (stride=1).
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        capture.release()
