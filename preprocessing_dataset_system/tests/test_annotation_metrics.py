"""Evaluation metrics for the fine-tuned tracker (Phase 15): the fixed success metrics on labeled frames."""

from __future__ import annotations

from pathlib import Path

import pytest

from prepds.annotation.examples import Example
from prepds.annotation.metrics import box_iou, evaluate, tilt_deg


def _ex(box=(50.0, 50.0, 150.0, 150.0), kps=None, listing="no", fid="a"):
    kps = {"snout": (60.0, 100.0, 2), "dorsal_fin_base": (100.0, 60.0, 2), "ventral": (100.0, 140.0, 2)} if kps is None else kps
    return Example(fid, Path(f"{fid}.png"), 320, 240, box, kps, listing if box else None)


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_tilt_is_zero_for_an_upright_fish_and_180_for_an_inverted_one() -> None:
    assert tilt_deg((100.0, 60.0), (100.0, 140.0)) == pytest.approx(0.0)  # dorsal above ventral (image y grows down)
    assert tilt_deg((100.0, 140.0), (100.0, 60.0)) == pytest.approx(180.0)
    assert tilt_deg((60.0, 100.0), (140.0, 100.0)) == pytest.approx(90.0)  # rolled onto its side


def test_perfect_predictions_score_perfectly() -> None:
    ex = [_ex(fid="a"), _ex(fid="b")]
    preds = [{"box": e.box, "score": 0.9, "keypoints": {n: (x, y) for n, (x, y, v) in e.keypoints.items()}} for e in ex]
    m = evaluate(ex, preds)
    assert m["n_positive"] == 2 and m["recall_at_iou_0.5"] == 1.0 and m["mean_iou"] == 1.0
    assert m["keypoint_error_px"]["snout"]["mean"] == 0.0 and m["false_positives_on_negatives"] == 0


def test_misses_and_errors_are_counted() -> None:
    ex = [_ex(fid="a"), _ex(fid="b")]
    preds = [None, {"box": (50.0, 50.0, 150.0, 150.0), "score": 0.9, "keypoints": {"snout": (63.0, 104.0)}}]
    m = evaluate(ex, preds)
    assert m["recall_at_iou_0.5"] == 0.5 and m["detection_rate"] == 0.5
    assert m["keypoint_error_px"]["snout"]["mean"] == pytest.approx(5.0)
    assert m["keypoint_error_px"]["ventral"]["n"] == 0  # predicted nothing there


def test_negative_frames_count_false_positives() -> None:
    neg = Example("n", Path("n.png"), 320, 240, None, {}, None)
    preds = [{"box": (10.0, 10.0, 50.0, 50.0), "score": 0.8, "keypoints": {}}]
    m = evaluate([neg], preds)
    assert m["n_negative"] == 1 and m["false_positives_on_negatives"] == 1 and m["n_positive"] == 0


def test_listing_tilt_stats_by_tag_from_both_ground_truth_and_predictions() -> None:
    upright = _ex(fid="u", listing="no")
    rolled = _ex(fid="r", listing="yes", kps={"snout": (60.0, 100.0, 2), "dorsal_fin_base": (60.0, 100.0, 2), "ventral": (140.0, 100.0, 2)})
    preds = [{"box": e.box, "score": 0.9, "keypoints": {n: (x, y) for n, (x, y, v) in e.keypoints.items()}} for e in (upright, rolled)]
    m = evaluate([upright, rolled], preds)
    assert m["tilt_deg_by_listing"]["no"]["gt_mean"] == pytest.approx(0.0)
    assert m["tilt_deg_by_listing"]["yes"]["gt_mean"] == pytest.approx(90.0)
    assert m["tilt_deg_by_listing"]["yes"]["pred_mean"] == pytest.approx(90.0)


def test_length_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError):
        evaluate([_ex()], [])
