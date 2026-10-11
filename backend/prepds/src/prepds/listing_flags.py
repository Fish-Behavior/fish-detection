"""Advisory 'possible Listing/LORR' flags for the reviewer (Phase 15).

A crop classifier scores sampled frames; consecutive high scores become time ranges stored in
`listing_flags.json` beside the video's artifacts. They are hints for the human reviewer only: they never change
a state, a manifest, or whether a video can be accepted. Precision is low by design (high recall), so the
reviewer confirms each one.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SIDECAR = "listing_flags.json"


@dataclass(frozen=True)
class ListingFlag:
    start_s: float
    end_s: float
    max_score: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {"start_s": self.start_s, "end_s": self.end_s, "max_score": self.max_score, "n_samples": self.n_samples}


def merge_flags(
    samples: Sequence[tuple[float, float]], *, threshold: float, step_s: float, min_samples: int = 1
) -> list[ListingFlag]:
    """`samples` = (t_sec, score) taken every `step_s`. Neighbouring above-threshold samples (at most 1.5 steps
    apart) form one range, padded by half a step on each side (start clamped at 0). Ranges with fewer than
    `min_samples` samples are dropped (isolated hits are mostly noise)."""
    hits = sorted((t, s) for t, s in samples if s >= threshold)
    flags: list[ListingFlag] = []
    group: list[tuple[float, float]] = []
    for hit in hits:
        if group and hit[0] - group[-1][0] > 1.5 * step_s:
            flags.append(_flag(group, step_s))
            group = []
        group.append(hit)
    if group:
        flags.append(_flag(group, step_s))
    return [f for f in flags if f.n_samples >= min_samples]


def _flag(group: Sequence[tuple[float, float]], step_s: float) -> ListingFlag:
    return ListingFlag(max(0.0, group[0][0] - step_s / 2), group[-1][0] + step_s / 2, max(s for _, s in group), len(group))


def write_listing_flags(video_dir: Path, flags: Sequence[ListingFlag], *, meta: Mapping[str, Any]) -> None:
    target = Path(video_dir) / SIDECAR
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"meta": dict(meta), "flags": [f.to_dict() for f in flags]}, indent=1), encoding="utf-8")
    os.replace(temporary, target)


def read_listing_flags(video_dir: Path) -> list[ListingFlag]:
    """[] when the video was never scanned; ValueError when the sidecar exists but cannot be read."""
    path = Path(video_dir) / SIDECAR
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        flags = [ListingFlag(float(f["start_s"]), float(f["end_s"]), float(f["max_score"]), int(f["n_samples"]))
                 for f in payload["flags"]]
        if not all(math.isfinite(v) for f in flags for v in (f.start_s, f.end_s, f.max_score)):
            raise ValueError("non-finite value")
        return flags
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is unreadable: {error}") from error
