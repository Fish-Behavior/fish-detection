"""Ablations and vehicle normalization (plan U14, T3.3; PRD §6.8, FR-9; D-008, D-061)."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dcs.ablations import FR9_NORMALIZED, FR9_RAW, epoch_of, keep_rule, run_ablations, vehicle_normalize, without_class
from dcs.config import load_settings
from dcs.evaluate import evaluate
from dcs.folds import make_folds
from dcs.report import StageResult
from dcs.trainset import build_trainset, load_table
from tests.dcs.eval_helpers import as_trainset

FAST = ["majority", "date_only", "logreg"]


@pytest.fixture(scope="module")
def training() -> dict[str, Any]:
    return {**load_settings().training, "folds": 3, "repeats": 2}


def hand_set() -> Any:
    """Two dates with 2 vehicle fish, one date with 1, one with none; `bouts` is a count (log1p), `speed` is not."""
    rows = [
        ("VEHICLE", "d1", 1.0, 10.0), ("VEHICLE", "d1", 3.0, 20.0), ("COMPOUND_A", "d1", 7.0, 50.0),
        ("VEHICLE", "d2", 0.0, 30.0), ("VEHICLE", "d2", 0.0, 40.0), ("COMPOUND_A", "d2", 7.0, 60.0),
        ("VEHICLE", "d3", 5.0, 70.0), ("COMPOUND_B", "d3", 7.0, 80.0),
        ("COMPOUND_B", "d4", 7.0, 90.0),
    ]  # fmt: skip
    X = pd.DataFrame({"bouts": [r[2] for r in rows], "speed": [r[3] for r in rows]})
    ts = as_trainset(X, [r[0] for r in rows], [r[1] for r in rows])
    return dataclasses.replace(ts, kinds={"bouts": "count", "speed": "continuous"})


def test_vehicle_fish_are_removed_and_kinds_become_continuous() -> None:
    normalized, levels = vehicle_normalize(hand_set(), "VEHICLE", {})
    assert "VEHICLE" not in set(normalized.y) and set(normalized.classes) == {"COMPOUND_A", "COMPOUND_B"}
    assert len(normalized.y) == 4
    assert normalized.kinds == {"bouts": "continuous", "speed": "continuous"}  # already logged; not logged twice


def test_a_date_with_two_vehicle_fish_is_its_own_reference() -> None:
    normalized, levels = vehicle_normalize(hand_set(), "VEHICLE", {})
    row = normalized.X.loc[2]  # COMPOUND_A on d1
    assert row["speed"] == pytest.approx(50 - 15)
    assert row["bouts"] == pytest.approx(np.log1p(7) - np.median(np.log1p([1, 3])))
    assert levels["date"] == 2


def test_fallback_goes_to_the_framing_setup_then_to_all_vehicle_fish() -> None:
    """d3 has one vehicle fish: its setup (shared with d1) is the reference; d4 has no setup: all vehicle fish."""
    normalized, levels = vehicle_normalize(hand_set(), "VEHICLE", {"d1": 1, "d3": 1, "d2": 2})
    assert normalized.X.loc[7, "speed"] == pytest.approx(80 - np.median([10, 20, 70]))
    assert normalized.X.loc[8, "speed"] == pytest.approx(90 - np.median([10, 20, 30, 40, 70]))
    assert levels == {"date": 2, "setup": 1, "all vehicle fish": 1}


def test_without_class_keeps_the_other_rows_and_relabels_the_classes() -> None:
    raw = without_class(hand_set(), "VEHICLE")
    assert len(raw.y) == 4 and set(raw.classes) == {"COMPOUND_A", "COMPOUND_B"}
    pd.testing.assert_frame_equal(raw.X, hand_set().X.loc[raw.X.index])


def summary(model: str, mean: float, std: float) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples([(model, "A")], names=["model", "scheme"])
    columns = pd.MultiIndex.from_tuples([("balanced_accuracy", "mean"), ("balanced_accuracy", "std")])
    return pd.DataFrame([[mean, std]], index=index, columns=columns)


@pytest.mark.parametrize(
    ("with_it", "without", "keep"),
    [((0.60, 0.02), (0.50, 0.03), True), ((0.52, 0.02), (0.50, 0.03), False), ((0.40, 0.01), (0.50, 0.01), False)],
)
def test_ntt_is_kept_only_when_scheme_a_gains_more_than_the_spread(
    with_it: tuple[float, float], without: tuple[float, float], keep: bool
) -> None:
    kept, reason = keep_rule(summary("logreg", *with_it), summary("logreg", *without), "logreg")
    assert kept is keep and reason


def test_keep_rule_without_scheme_a_is_undecided() -> None:
    kept, reason = keep_rule(summary("logreg", 0.5, 0.0).iloc[0:0], summary("logreg", 0.5, 0.0), "logreg")
    assert kept is None and "scheme A" in reason


@pytest.fixture(scope="module")
def main_and_ablations(tiny_table: Any, training: dict[str, Any]) -> tuple[StageResult, list[Any]]:
    table, described = load_table(tiny_table)
    ts = build_trainset(table, described, training, "compound")
    folds = make_folds(ts.y, ts.groups, ts.ids, training)
    main = StageResult(ts, folds, evaluate(ts, folds, FAST, training))
    return main, run_ablations(table, described, training, main, FAST, "cpu", {}, lambda message: None)


def test_every_ablation_is_reported(main_and_ablations: tuple[StageResult, list[Any]]) -> None:
    _, ablations = main_and_ablations
    assert [a.name for a in ablations] == [
        "use_ntt=true", "use_demographics=true", "use_depth=true", FR9_RAW, FR9_NORMALIZED,
    ]  # fmt: skip
    assert all(a.result is not None or a.note for a in ablations)


def test_switch_ablations_share_the_main_folds(main_and_ablations: tuple[StageResult, list[Any]]) -> None:
    main, ablations = main_and_ablations
    for ablation in ablations[:3]:
        if ablation.result is not None:
            assert ablation.result.folds is main.folds
    depth = next(a for a in ablations if a.name == "use_depth=true").result
    assert any(name.startswith("depth_") for name in depth.trainset.features)
    assert not any(name.startswith("depth_") for name in main.trainset.features)


def test_fr9_pair_shares_classes_and_folds(main_and_ablations: tuple[StageResult, list[Any]]) -> None:
    """D-008: raw and normalized run on the same non-vehicle fish and the main folds without the vehicle rows."""
    main, ablations = main_and_ablations
    raw, normalized = (next(a for a in ablations if a.name == name).result for name in (FR9_RAW, FR9_NORMALIZED))
    assert raw.trainset.classes == normalized.trainset.classes and "VEHICLE" not in raw.trainset.classes
    pd.testing.assert_frame_equal(raw.folds.table, normalized.folds.table)
    expected = main.folds.table[main.folds.table["label"] != "VEHICLE"]
    pd.testing.assert_frame_equal(raw.folds.table, expected)


def test_fr9_is_a_note_when_the_vehicle_is_not_a_kept_class(tiny_table: Any, training: dict[str, Any]) -> None:
    table, described = load_table(tiny_table)
    other = {**training, "vehicle_compound": "NOT_HERE"}
    ts = build_trainset(table, described, other, "compound")
    folds = make_folds(ts.y, ts.groups, ts.ids, other)
    main = StageResult(ts, folds, evaluate(ts, folds, ["majority"], other))
    fr9 = [a for a in run_ablations(table, described, other, main, ["majority"], "cpu", {}, print) if a.name in (FR9_RAW, FR9_NORMALIZED)]
    assert all(a.result is None and "vehicle_compound" in a.note for a in fr9)


def test_epoch_of_counts_the_start_dates_passed() -> None:
    starts = ["2000-03-01", "2000-06-01"]
    assert [epoch_of(d, starts) for d in ("2000-01-15", "2000-03-01", "2000-05-31", "2000-07-01")] == [
        "epoch 1", "epoch 2", "epoch 2", "epoch 3",
    ]  # fmt: skip


def test_epoch_reference_skips_the_date_level() -> None:
    """D-061: with camera epochs, every fish is referenced to its epoch's vehicle fish, never its own date's."""
    epochs = {"d1": "epoch 1", "d2": "epoch 1", "d3": "epoch 2", "d4": "epoch 2"}
    normalized, levels = vehicle_normalize(hand_set(), "VEHICLE", epochs, by_date=False)
    assert normalized.X.loc[2, "speed"] == pytest.approx(50 - np.median([10, 20, 30, 40]))
    assert normalized.X.loc[8, "speed"] == pytest.approx(90 - np.median([10, 20, 30, 40, 70]))  # epoch 2 has 1 vehicle fish
    assert levels == {"date": 0, "camera epoch": 2, "all vehicle fish": 2}


def test_camera_epochs_setting_switches_the_fr9_reference(tiny_table: Any, training: dict[str, Any]) -> None:
    table, described = load_table(tiny_table)
    epochs = {**training, "camera_epochs": ("2000-01-03",)}
    ts = build_trainset(table, described, epochs, "compound")
    folds = make_folds(ts.y, ts.groups, ts.ids, epochs)
    main = StageResult(ts, folds, evaluate(ts, folds, ["majority"], epochs))
    fr9 = run_ablations(table, described, epochs, main, ["majority"], "cpu", {}, lambda message: None)[-1]
    assert fr9.name == FR9_NORMALIZED and "camera epoch for" in fr9.what


def test_a_switch_that_changes_the_fish_gets_new_folds_and_says_so(
    tiny_table: Any, training: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    table, described = load_table(tiny_table)
    ts = build_trainset(table, described, training, "compound")
    folds = make_folds(ts.y, ts.groups, ts.ids, training)
    main = StageResult(ts, folds, evaluate(ts, folds, FAST, training))
    real = build_trainset
    monkeypatch.setattr(  # demographics "drops" one fish, as a missing sex/strain/age value would
        "dcs.ablations.build_trainset",
        lambda *args: real(*args).take(np.arange(len(ts.y)) != 0) if args[2]["use_demographics"] != training["use_demographics"] else real(*args),
    )
    by_name = {a.name: a for a in run_ablations(table, described, training, main, FAST, "cpu", {}, lambda message: None)}
    assert "new folds" in by_name["use_demographics=true"].what and by_name["use_demographics=true"].result.folds is not main.folds
    assert "new folds" not in by_name["use_depth=true"].what and by_name["use_depth=true"].result.folds is main.folds
