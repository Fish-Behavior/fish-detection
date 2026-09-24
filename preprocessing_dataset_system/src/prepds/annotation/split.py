"""Subject-level train/held-out split for the tracker fine-tuning set (Phase 15).

Frozen before any frame is sampled: frames from one subject must never appear on both sides, and the held-out
side is the only honest evaluation of the fine-tuned model. Stratified by compound group so every family is
represented in the evaluation; groups with fewer than `MIN_GROUP_SIZE_FOR_HOLDOUT` videos stay in train.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

TRAIN, HELDOUT = "train", "heldout"
MIN_GROUP_SIZE_FOR_HOLDOUT = 5


def make_split(video_groups: Mapping[str, str], *, heldout_fraction: float, seed: int) -> dict[str, str]:
    """`video_id -> "train" | "heldout"`; deterministic and independent of the mapping's order."""
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError(f"heldout_fraction must be strictly between 0 and 1, got {heldout_fraction}")
    by_group: dict[str, list[str]] = defaultdict(list)
    for video_id, group in video_groups.items():
        by_group[group].append(video_id)
    split: dict[str, str] = {}
    for group, members in by_group.items():
        ranked = sorted(members, key=lambda v: hashlib.sha256(f"{seed}:{group}:{v}".encode()).hexdigest())
        n_heldout = 0 if len(members) < MIN_GROUP_SIZE_FOR_HOLDOUT else max(1, round(heldout_fraction * len(members)))
        for index, video_id in enumerate(ranked):
            split[video_id] = HELDOUT if index < n_heldout else TRAIN
    return dict(sorted(split.items()))


def write_split(path: Path, split: Mapping[str, str], *, seed: int, heldout_fraction: float) -> None:
    """Write once; an existing split is never overwritten (annotations depend on it)."""
    payload = json.dumps(
        {"seed": seed, "heldout_fraction": heldout_fraction, "assignments": dict(split)}, indent=2, sort_keys=True
    )
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(payload)


def load_split(path: Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assignments = data["assignments"]
    bad = {v for v in assignments.values() if v not in (TRAIN, HELDOUT)}
    if bad:
        raise ValueError(f"split has unknown labels {sorted(bad)}")
    return dict(assignments)
