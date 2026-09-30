"""Pick the frames to annotate for the tracker fine-tuning set (Phase 15).

Frames are chosen where a better tracker matters most, not uniformly: the short/medium gaps where the classical
tracker lost the fish (the most recoverable Undetermined time), long gaps, the still-to-the-end flagged videos, and
late-session slow frames of Fentanyl videos (candidate LORR, a rare class that would otherwise not be trainable),
plus ordinary detected frames so the model also sees the normal case. Selection is stratified across compounds,
spaced within a video, capped per video, and deterministic for a seed. Only videos in the requested split are used,
so the frozen train/held-out split (`split.py`) is never crossed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# Bucket -> share of the requested frames, in selection priority order; whatever a bucket cannot supply is
# filled from the final "detected" bucket.
DEFAULT_MIX: tuple[tuple[str, float], ...] = (
    ("gap_1_10s", 0.35),
    ("gap_long", 0.10),
    ("flagged_tail", 0.10),
    ("lorr_candidate", 0.15),
)
FALLBACK_BUCKET = "detected"
SHORT_GAP_S, LONG_GAP_S = 1.0, 10.0
LORR_MIN_T_S = 300.0
LORR_STATES = ("Freezing/Drift", "Undetermined")
STRIDE_S = {"flagged_tail": 20.0, "lorr_candidate": 15.0, "detected": 5.0}


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    compound: str
    fps: float
    detected: np.ndarray  # bool per frame
    states: np.ndarray  # label per frame
    flagged_from_s: float | None = None


@dataclass(frozen=True)
class FrameRef:
    video_id: str
    frame_idx: int
    reason: str


def sample_frames(
    videos: Sequence[VideoInfo],
    split: Mapping[str, str],
    *,
    n_frames: int,
    seed: int,
    split_name: str = "train",
    min_gap_s: float = 2.0,
    max_per_video: int = 6,
    mix: Sequence[tuple[str, float]] = DEFAULT_MIX,
) -> list[FrameRef]:
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    missing = [v.video_id for v in videos if v.video_id not in split]
    if missing:
        raise ValueError(f"videos missing from the split: {missing[:3]}")
    pool = sorted((v for v in videos if split[v.video_id] == split_name), key=lambda v: v.video_id)

    chosen: list[FrameRef] = []
    per_video: dict[str, list[int]] = defaultdict(list)
    taken: set[tuple[str, int]] = set()
    plan = [(name, round(share * n_frames)) for name, share in mix]
    for name, quota in [*plan, (FALLBACK_BUCKET, None)]:
        target = n_frames - len(chosen) if quota is None else min(quota, n_frames - len(chosen))
        if target <= 0:
            continue
        candidates = {v.video_id: _candidates(v, name) for v in pool}
        by_compound: dict[str, list[VideoInfo]] = defaultdict(list)
        for v in pool:
            if candidates[v.video_id]:
                by_compound[v.compound].append(v)
        rng = random.Random(f"{seed}:{name}")
        for members in by_compound.values():
            rng.shuffle(members)
        picked = 0
        progressed = True
        while picked < target and progressed:
            progressed = False
            for compound in sorted(by_compound):
                for v in by_compound[compound]:
                    if picked >= target:
                        break
                    frame = _take(v, candidates[v.video_id], per_video[v.video_id], taken, rng, min_gap_s, max_per_video)
                    if frame is None:
                        continue
                    chosen.append(FrameRef(v.video_id, frame, name))
                    per_video[v.video_id].append(frame)
                    taken.add((v.video_id, frame))
                    picked += 1
                    progressed = True
    return chosen


def _take(v, options, already, taken, rng, min_gap_s, max_per_video):
    if len(already) >= max_per_video:
        return None
    spacing = min_gap_s * v.fps
    ok = [f for f in options if (v.video_id, f) not in taken and all(abs(f - a) >= spacing for a in already)]
    if not ok:
        return None
    frame = ok[rng.randrange(len(ok))]
    options.remove(frame)
    return frame


def _candidates(v: VideoInfo, bucket: str) -> list[int]:
    n = len(v.detected)
    if bucket in ("gap_1_10s", "gap_long"):
        frames: list[int] = []
        for start, length in _undetected_runs(v.detected):
            seconds = length / v.fps
            if bucket == "gap_1_10s" and SHORT_GAP_S <= seconds < LONG_GAP_S:
                frames.append(start + length // 2)
            elif bucket == "gap_long" and seconds >= LONG_GAP_S:
                frames.extend(start + int(length * q) for q in (0.25, 0.5, 0.75))
        return frames
    stride = max(1, int(STRIDE_S[bucket] * v.fps))
    if bucket == "flagged_tail":
        if v.flagged_from_s is None:
            return []
        return list(range(int(v.flagged_from_s * v.fps), n, stride))
    if bucket == "lorr_candidate":
        if v.compound.strip().lower() != "fentanyl":
            return []
        slow = np.isin(v.states, LORR_STATES)
        return [f for f in range(int(LORR_MIN_T_S * v.fps), n, stride) if slow[f]]
    return [f for f in range(0, n, stride) if v.detected[f]]


def _undetected_runs(detected: np.ndarray) -> list[tuple[int, int]]:
    missing = ~detected
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]
