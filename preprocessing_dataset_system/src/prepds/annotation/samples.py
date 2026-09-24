"""The frames to label: the original `sample.json` plus later labeling batches `batch_*.json` (Phase 15).

A batch adds frames without touching the frozen originals. Every frame appears once, and a video keeps the same
train/held-out assignment everywhere: a batch can never move a video across the split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_samples(work_dir: Path) -> list[dict[str, Any]]:
    work_dir = Path(work_dir)
    files = [work_dir / "sample.json", *sorted(work_dir.glob("batch_*.json"))]
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    video_split: dict[str, str] = {}
    for path in files:
        batch = "sample" if path.name == "sample.json" else path.stem
        for sample in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            if sample["image"] in seen:
                raise ValueError(f"duplicate frame {sample['image']} (also in {path.name})")
            seen.add(sample["image"])
            if video_split.setdefault(sample["video_id"], sample["split"]) != sample["split"]:
                raise ValueError(f"video {sample['video_id']} changes split in {path.name}")
            samples.append({**sample, "batch": batch})
    return samples
