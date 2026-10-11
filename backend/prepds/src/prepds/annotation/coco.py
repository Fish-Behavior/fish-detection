"""COCO-keypoints export of the annotated frames, one file per split (Phase 15)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from prepds.annotation.store import KEYPOINT_NAMES, Annotation


def build_coco(
    samples: Sequence[Mapping[str, Any]],
    annotations: Mapping[str, Annotation],
    *,
    split_name: str,
    size_of: Callable[[str], tuple[int, int]],
) -> dict[str, Any]:
    """Annotated frames of `split_name`; "no fish visible" frames are images with no annotation (negatives)."""
    images: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for sample in samples:
        stem = str(sample["image"]).rsplit(".", 1)[0]
        annotation = annotations.get(stem)
        if sample["split"] != split_name or annotation is None:
            continue
        image_id = len(images) + 1
        width, height = size_of(sample["image"])
        images.append({"id": image_id, "file_name": sample["image"], "width": width, "height": height,
                       "video_id": sample["video_id"], "frame_idx": sample["frame_idx"], "compound": sample["compound"]})
        if not annotation.fish_visible or annotation.box is None:
            continue
        x0, y0, x1, y1 = annotation.box
        flat: list[float | int] = []
        for name in KEYPOINT_NAMES:
            x, y, v = annotation.keypoints.get(name, (0, 0, 0))
            flat.extend([x, y, v])
        entries.append({
            "id": len(entries) + 1, "image_id": image_id, "category_id": 1, "iscrowd": 0,
            "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
            "keypoints": flat, "num_keypoints": len(annotation.keypoints), "listing": annotation.listing,
        })
    return {
        "images": images,
        "annotations": entries,
        "categories": [{"id": 1, "name": "fish", "supercategory": "animal", "keypoints": list(KEYPOINT_NAMES), "skeleton": []}],
    }
