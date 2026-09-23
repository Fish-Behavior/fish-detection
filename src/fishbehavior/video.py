"""Video reading: file timing/size (`probe`), sequential frame iteration, and even sampling.

Everything downstream measures time in seconds computed as ``frame_index / fps`` with
the fps read from the file itself, so the pipeline works for any frame rate and
resolution. `CAP_PROP_POS_MSEC` is not used: some containers report it unreliably.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoError(RuntimeError):
    """A video file cannot be used (cannot be opened, or has no usable frame rate)."""


@dataclass(frozen=True)
class VideoInfo:
    """Timing and size of one video file, as reported by the file."""

    path: Path
    fps: float
    frame_count: int
    duration_s: float  # frame_count / fps
    width: int
    height: int

    def to_dict(self) -> dict:
        """JSON-friendly version (path as text)."""
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def _open(path: Path) -> cv2.VideoCapture:
    """Open a video or raise VideoError (OpenCV itself fails silently on a bad file)."""
    if not Path(path).is_file():
        raise VideoError(f"{path}: file not found")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError(f"{path}: OpenCV cannot open this file (unsupported codec or damaged file)")
    return capture


def _read_fps(capture: cv2.VideoCapture, path: Path) -> float:
    """Frame rate from the file; 0 or NaN means the file cannot be timed, so fail loudly."""
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        raise VideoError(f"{path}: invalid frame rate {fps!r} in the file header")
    return fps


def probe(path: str | Path) -> VideoInfo:
    """Read fps, frame count and size of a video without decoding it.

    If the header has no frame count (some containers report 0), the frames are
    counted by reading through the file once.
    """
    path = Path(path)
    capture = _open(path)
    try:
        fps = _read_fps(capture, path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0:
            log.warning("%s: no frame count in header; counting frames", path.name)
            frame_count = 0
            while capture.grab():  # grab() skips decoding, so counting is fast
                frame_count += 1
    finally:
        capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise VideoError(f"{path}: no readable frames")
    return VideoInfo(path, fps, frame_count, frame_count / fps, width, height)


def _prepare(frame: np.ndarray, scale: float, gray: bool) -> np.ndarray:
    """Convert a decoded BGR frame to grayscale and/or shrink it, as requested."""
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        # INTER_AREA averages pixels when shrinking, which avoids aliasing noise.
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return frame


def iter_frames(
    path: str | Path, stride: int = 1, scale: float = 1.0, gray: bool = True
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, time_s, frame)`` for every `stride`-th frame, reading sequentially.

    ``time_s = frame_index / fps``. Frames in between are skipped with `grab()`
    (no decoding) instead of seeking, because seeking per frame is slow and on
    some codecs lands on the wrong frame. `scale` < 1 shrinks frames for speed.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    path = Path(path)
    capture = _open(path)
    try:
        fps = _read_fps(capture, path)
        index = 0
        while capture.grab():
            if index % stride == 0:
                ok, frame = capture.retrieve()  # decode only the frames we keep
                if not ok:
                    break
                yield index, index / fps, _prepare(frame, scale, gray)
            index += 1
    finally:
        capture.release()


def sample_frames(path: str | Path, n: int, gray: bool = True) -> np.ndarray:
    """Return up to `n` frames spread evenly from the first to the last frame, stacked.

    Seeking is fine here because only a few frames are needed. A frame that cannot
    be read (the header frame count sometimes overshoots by a frame or two) is
    skipped, so the result can be slightly shorter than `n`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    info = probe(path)
    indices = np.unique(np.linspace(0, info.frame_count - 1, n).round().astype(int))
    capture = _open(info.path)
    frames = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(_prepare(frame, 1.0, gray))
    finally:
        capture.release()
    if not frames:
        raise VideoError(f"{info.path}: none of the {len(indices)} sampled frames could be read")
    if len(frames) < len(indices):
        log.debug("%s: %d of %d sampled frames unreadable", info.path.name, len(indices) - len(frames), len(indices))
    return np.stack(frames)
