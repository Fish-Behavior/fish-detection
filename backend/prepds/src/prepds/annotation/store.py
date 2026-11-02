"""Annotation records for the tracker fine-tuning set (Phase 15): one atomic JSON file per sampled frame.

A frame is either "no fish visible" (no box, keypoints or listing tag) or has a box, a `listing` tag
(yes/no/unsure) and any subset of the five keypoints. The keypoint schema includes a dorsal and a ventral point
because a head-tail axis only measures pitch: roll or inversion (Listing/LORR) needs both.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

KEYPOINT_NAMES = ("snout", "dorsal_fin_base", "ventral", "tail_base", "tail_tip")
LISTING_TAGS = ("yes", "no", "unsure")
VISIBILITY = (0, 1, 2)  # 0 unlabeled, 1 occluded, 2 visible (COCO convention)
MIN_BOX_SIDE_PX = 2.0
_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class InvalidAnnotation(ValueError):
    """The submitted annotation breaks a schema rule."""


@dataclass(frozen=True)
class Annotation:
    fish_visible: bool
    box: tuple[float, float, float, float] | None
    keypoints: Mapping[str, tuple[float, float, int]]
    listing: str | None
    annotator: str
    annotated_at: dt.datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "fish_visible": self.fish_visible,
            "box": list(self.box) if self.box else None,
            "keypoints": {name: list(value) for name, value in self.keypoints.items()},
            "listing": self.listing,
            "annotator": self.annotator,
            "annotated_at": self.annotated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Annotation":
        return cls(
            fish_visible=bool(data["fish_visible"]),
            box=tuple(float(v) for v in data["box"]) if data["box"] else None,
            keypoints={n: (float(v[0]), float(v[1]), int(v[2])) for n, v in data["keypoints"].items()},
            listing=data["listing"],
            annotator=data["annotator"],
            annotated_at=dt.datetime.fromisoformat(data["annotated_at"]),
        )


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidAnnotation(f"not a finite number: {value!r}")
    return float(value)


def parse_annotation(payload: Mapping[str, Any], *, width: int, height: int, now: dt.datetime) -> Annotation:
    annotator = str(payload.get("annotator", "")).strip()
    if not annotator or len(annotator) > 100 or any(ord(c) < 32 or ord(c) == 127 for c in annotator):
        raise InvalidAnnotation("annotator must be a non-blank name without control characters")
    visible = payload.get("fish_visible")
    if not isinstance(visible, bool):
        raise InvalidAnnotation("fish_visible must be true or false")
    box, keypoints, listing = payload.get("box"), payload.get("keypoints") or {}, payload.get("listing")
    if not visible:
        if box or keypoints or listing:
            raise InvalidAnnotation("a frame with no fish visible has no box, keypoints or listing tag")
        return Annotation(False, None, {}, None, annotator, now)

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise InvalidAnnotation("box must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (_finite(v) for v in box)
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise InvalidAnnotation("box must lie inside the image")
    if x1 - x0 < MIN_BOX_SIDE_PX or y1 - y0 < MIN_BOX_SIDE_PX:
        raise InvalidAnnotation("box is too small")
    if listing not in LISTING_TAGS:
        raise InvalidAnnotation(f"listing must be one of {LISTING_TAGS}")

    parsed: dict[str, tuple[float, float, int]] = {}
    for name, value in keypoints.items():
        if name not in KEYPOINT_NAMES:
            raise InvalidAnnotation(f"unknown keypoint {name!r}")
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise InvalidAnnotation(f"keypoint {name} must be [x, y, visibility]")
        x, y, v = _finite(value[0]), _finite(value[1]), value[2]
        if v not in VISIBILITY or isinstance(v, bool):
            raise InvalidAnnotation(f"keypoint {name} visibility must be one of {VISIBILITY}")
        if v == 0:
            continue  # unlabeled: nothing to keep
        if not (0 <= x <= width and 0 <= y <= height):
            raise InvalidAnnotation(f"keypoint {name} lies outside the image")
        parsed[name] = (x, y, int(v))
    return Annotation(True, (x0, y0, x1, y1), parsed, listing, annotator, now)


def _path(directory: Path, stem: str) -> Path:
    if not _STEM.fullmatch(stem):
        raise ValueError(f"unusable frame id {stem!r}")
    return Path(directory) / f"{stem}.json"


def save_annotation(directory: Path, stem: str, annotation: Annotation) -> None:
    target = _path(directory, stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")  # unique: two tabs never share it
    try:
        temporary.write_text(json.dumps(annotation.to_dict(), indent=1), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_annotation(directory: Path, stem: str) -> Annotation | None:
    path = _path(directory, stem)
    if not path.is_file():
        return None
    try:
        return Annotation.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"annotation {stem} is unreadable: {error}") from error
