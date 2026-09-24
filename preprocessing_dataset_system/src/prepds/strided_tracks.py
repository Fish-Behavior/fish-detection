"""Turn detections made on every Nth frame into the per-frame `Track` list the rest of the pipeline expects.

A learned detector is too slow to run on all ~11.7M frames of the dataset, but everything downstream needs only
~1 s-lag displacements and 1 s bins. Detections are therefore made every `stride` frames and the frames between
two detected samples are linearly interpolated (and marked detected); frames next to a missed sample, and any
trailing frames, stay undetected - a gap is never bridged by guesswork.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from prepds.models import Track


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    orientation_deg: float | None


def expand_strided_detections(
    samples: Mapping[int, Detection | None], *, n_frames: int, stride: int, fps: float
) -> list[Track]:
    """`samples` maps sampled frame indices (0, stride, 2*stride, ...) to a detection or None (a miss)."""
    if n_frames < 1 or stride < 1 or fps <= 0:
        raise ValueError("n_frames and stride must be >= 1 and fps > 0")
    tracks: list[Track] = []
    for frame in range(n_frames):
        lo = (frame // stride) * stride
        hi = lo + stride
        a = samples.get(lo)
        detection = None
        if frame == lo:
            detection = a
        elif a is not None and (b := samples.get(hi)) is not None:
            w = (frame - lo) / stride
            nearest = a if w < 0.5 else b
            detection = Detection(a.x + w * (b.x - a.x), a.y + w * (b.y - a.y), nearest.orientation_deg)
        t = frame / fps
        if detection is None:
            tracks.append(Track(frame, t, 0.0, 0.0, None, None, False))
        else:
            tracks.append(Track(frame, t, detection.x, detection.y, detection.orientation_deg, detection.y, True))
    return tracks
