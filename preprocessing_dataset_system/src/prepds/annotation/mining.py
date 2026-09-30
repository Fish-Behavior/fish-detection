"""Pick frames that are likely Listing/LORR (inverted fish) for the next labeling batch (Phase 15).

Random sampling finds too few positives to train or evaluate on. A model scans frames across all videos, and the
ones with a large predicted dorsal->ventral tilt are put in front of the annotator to confirm or reject. This is
active learning: the annotator's answer, not the model's tilt, is the label. "Ambiguous" mid-tilt frames are
included so the labeled set also contains the hard negatives and borderline cases. The train/held-out split of
each video is respected: candidates are chosen per split.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Scanned:
    video_id: str
    frame_idx: int
    fps: float
    compound: str
    split: str
    box_score: float
    tilt: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Candidate:
    video_id: str
    frame_idx: int
    split: str
    reason: str  # "listing_candidate" | "listing_ambiguous"
    tilt: float
    box_score: float
    box: tuple[float, float, float, float]


def select_candidates(
    scanned: Sequence[Scanned],
    existing: Mapping[str, Sequence[int]],
    *,
    quotas: Mapping[str, tuple[int, int]],
    seed: int,
    min_box_score: float = 0.5,
    candidate_tilt: float = 100.0,
    ambiguous_tilt: tuple[float, float] = (60.0, 100.0),
    min_gap_s: float = 20.0,
    exclude_within_s: float = 2.0,
    max_per_video: int = 4,
) -> list[Candidate]:
    """`quotas[split] = (n likely-inverted, n ambiguous)`. Highest tilt first, interleaved across compounds."""
    usable = sorted(
        (s for s in scanned if s.box_score >= min_box_score and not math.isnan(s.tilt)
         and not any(abs(s.frame_idx - f) <= exclude_within_s * s.fps for f in existing.get(s.video_id, ()))),
        key=lambda s: (s.video_id, s.frame_idx),
    )
    chosen: list[Candidate] = []
    per_video: dict[str, list[int]] = defaultdict(list)
    for split in sorted(quotas):
        n_candidates, n_ambiguous = quotas[split]
        rng = random.Random(f"{seed}:{split}")
        pool = [s for s in usable if s.split == split]
        stages = (
            ("listing_candidate", n_candidates, sorted((s for s in pool if s.tilt >= candidate_tilt), key=lambda s: -s.tilt)),
            ("listing_ambiguous", n_ambiguous, [s for s in pool if ambiguous_tilt[0] <= s.tilt < ambiguous_tilt[1]]),
        )
        for reason, quota, ranked in stages:
            if reason == "listing_ambiguous":
                rng.shuffle(ranked)
            by_compound: dict[str, list[Scanned]] = defaultdict(list)
            for s in ranked:
                by_compound[s.compound].append(s)
            picked = 0
            progressed = True
            while picked < quota and progressed:
                progressed = False
                for compound in sorted(by_compound):
                    queue = by_compound[compound]
                    while queue and picked < quota:
                        s = queue.pop(0)
                        if _allowed(s, per_video[s.video_id], min_gap_s, max_per_video):
                            per_video[s.video_id].append(s.frame_idx)
                            chosen.append(Candidate(s.video_id, s.frame_idx, split, reason, s.tilt, s.box_score, s.box))
                            picked += 1
                            progressed = True
                            break
    return chosen


def _allowed(s: Scanned, taken: Sequence[int], min_gap_s: float, max_per_video: int) -> bool:
    return len(taken) < max_per_video and all(abs(s.frame_idx - f) >= min_gap_s * s.fps for f in taken)
