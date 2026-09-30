"""Metrics for the fine-tuned tracker on labeled frames (Phase 15 success metrics 1, 2 and 5-support).

`predictions[i]` is None (nothing detected) or {"box": (x0,y0,x1,y1), "score": float, "keypoints": {name: (x, y)}}
for `examples[i]`. Pure numpy/Python so it is testable without torch.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

from prepds.annotation.examples import Example
from prepds.annotation.store import KEYPOINT_NAMES

IOU_THRESHOLD = 0.5


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def tilt_deg(dorsal: tuple[float, float], ventral: tuple[float, float]) -> float:
    """Angle (0-180) between the dorsal->ventral vector and image-down: 0 upright, 90 on its side, 180 inverted."""
    dx, dy = ventral[0] - dorsal[0], ventral[1] - dorsal[1]
    return math.degrees(math.atan2(abs(dx), dy)) if dy >= 0 else 180.0 - math.degrees(math.atan2(abs(dx), -dy))


def _summary(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean": mean(values) if values else None, "median": median(values) if values else None}


def evaluate(examples: Sequence[Example], predictions: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("one prediction (or None) per example is required")
    ious: list[float] = []
    errors: dict[str, list[float]] = defaultdict(list)
    tilts: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"gt": [], "pred": []})
    detected = false_pos = n_neg = 0
    for example, pred in zip(examples, predictions):
        if example.box is None:
            n_neg += 1
            false_pos += pred is not None
            continue
        ious.append(box_iou(pred["box"], example.box) if pred else 0.0)
        detected += pred is not None
        if not pred:
            continue
        for name, (x, y, v) in example.keypoints.items():
            if v > 0 and name in pred["keypoints"]:
                px, py = pred["keypoints"][name]
                errors[name].append(math.hypot(px - x, py - y))
        gt = example.keypoints
        if example.listing and "dorsal_fin_base" in gt and "ventral" in gt:
            tilts[example.listing]["gt"].append(tilt_deg(gt["dorsal_fin_base"][:2], gt["ventral"][:2]))
            pk = pred["keypoints"]
            if "dorsal_fin_base" in pk and "ventral" in pk:
                tilts[example.listing]["pred"].append(tilt_deg(pk["dorsal_fin_base"], pk["ventral"]))
    n_pos = len(ious)
    return {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "detection_rate": detected / n_pos if n_pos else None,
        "recall_at_iou_0.5": sum(i >= IOU_THRESHOLD for i in ious) / n_pos if n_pos else None,
        "mean_iou": mean(ious) if ious else None,
        "false_positives_on_negatives": false_pos,
        "keypoint_error_px": {name: _summary(errors[name]) for name in KEYPOINT_NAMES},
        "tilt_deg_by_listing": {
            tag: {"n": len(v["gt"]), "gt_mean": mean(v["gt"]) if v["gt"] else None,
                  "pred_mean": mean(v["pred"]) if v["pred"] else None}
            for tag, v in sorted(tilts.items())
        },
    }
