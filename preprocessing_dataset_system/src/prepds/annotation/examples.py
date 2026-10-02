"""Training/evaluation examples built from the annotations (Phase 15). Pure Python (no torch)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from PIL import Image

from prepds.annotation import store
from prepds.annotation.samples import load_samples
from prepds.annotation.split import HELDOUT, TRAIN


@dataclass(frozen=True)
class Example:
    frame_id: str
    image_path: Path
    width: int
    height: int
    box: tuple[float, float, float, float] | None  # None = "no fish visible" (a negative example)
    keypoints: Mapping[str, tuple[float, float, int]]
    listing: str | None


def load_examples(work_dir: Path, split_name: str) -> list[Example]:
    """Labeled frames of one split. Held-out frames are only ever returned when asked for by name, and an
    unreadable annotation is an error (a silently smaller ground-truth set would skew every metric)."""
    if split_name not in (TRAIN, HELDOUT):
        raise ValueError(f"unknown split {split_name!r}")
    work_dir = Path(work_dir)
    samples = load_samples(work_dir)
    examples = []
    for sample in samples:
        if sample["split"] != split_name:
            continue
        frame_id = sample["image"].rsplit(".", 1)[0]
        annotation = store.load_annotation(work_dir / "annotations", frame_id)  # ValueError if corrupt
        if annotation is None:
            continue
        path = work_dir / "frames" / sample["image"]
        with Image.open(path) as image:
            width, height = image.size
        examples.append(Example(frame_id, path, width, height, annotation.box, dict(annotation.keypoints), annotation.listing))
    return examples


def hflip(example: Example) -> Example:
    """Mirror left-right. Never flip vertically: that would swap dorsal and ventral, i.e. turn a normal fish
    into a Listing one."""
    w = example.width
    box = None if example.box is None else (w - example.box[2], example.box[1], w - example.box[0], example.box[3])
    keypoints = {name: (w - x, y, v) for name, (x, y, v) in example.keypoints.items()}
    return replace(example, box=box, keypoints=keypoints)
