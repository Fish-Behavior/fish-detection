"""Evaluation (plan U11, T2.5; PRD §6.4-6.7): shared folds, EC-9, pooled metrics (D-017), permutation, decision rule."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, log_loss

from dcs import evaluate as evaluate_module
from dcs.config import load_settings
from dcs.evaluate import (
    BASELINE,
    METRICS,
    NOT_USEFUL,
    PERMUTED,
    UNDECIDED,
    USEFUL,
    evaluate,
    permute_within_date,
)
from dcs.folds import SCHEME_A, SCHEME_B, make_folds
from dcs.trainset import TrainSet

CLASSES = ("COMPOUND_A", "COMPOUND_B", "VEHICLE")
FAST = ["majority", "date_only", "logreg"]


@pytest.fixture(scope="module")
def training() -> dict[str, Any]:
    return {**load_settings().training, "folds": 3, "repeats": 2}


def as_trainset(X: pd.DataFrame, y: list[str], dates: list[str]) -> TrainSet:
    labels = pd.Series(y, index=X.index, name="label")
    return TrainSet(
        stage="compound",
        X=X,
        y=labels,
        groups=pd.Series(dates, index=X.index),
        ids=pd.Series([f"F_{i:04d}" for i in range(len(X))], index=X.index),
        features=tuple(X.columns),
        kinds={name: "continuous" for name in X.columns},
        classes={str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        very_small=(),
        dropped_classes=(),
        dropped_states=(),
        dropped_features=(),
    )


def signal_set(signal: float = 3.0, n_dates: int = 6, per_cell: int = 2, seed: int = 0) -> TrainSet:
    """Every date holds every class; feature f0 carries the class, f2 a date offset."""
    rng = np.random.default_rng(seed)
    rows = [(c, d) for d in range(n_dates) for c in range(len(CLASSES)) for _ in range(per_cell)]
    offsets = rng.normal(scale=2, size=n_dates)
    X = pd.DataFrame(
        {
            "f0": [c * signal + rng.normal() for c, _ in rows],
            "f1": rng.normal(size=len(rows)),
            "f2": [offsets[d] + rng.normal(scale=0.1) for _, d in rows],
        }
    )
    return as_trainset(X, [CLASSES[c] for c, _ in rows], [f"2000-01-{d + 1:02d}" for _, d in rows])


def date_label_set(n_dates: int = 9, per_date: int = 4, seed: int = 0) -> TrainSet:
    """The leakage canary: each date holds one class and the features know only the date. With more features
    than dates, a linear model can memorize which date (hence which class) a fish came from."""
    rng = np.random.default_rng(seed)
    dates = [d for d in range(n_dates) for _ in range(per_date)]
    width = n_dates + 3
    offsets = rng.normal(scale=10, size=(n_dates, width))
    X = pd.DataFrame(offsets[dates] + rng.normal(scale=0.1, size=(len(dates), width)), columns=[f"f{i}" for i in range(width)])
    return as_trainset(X, [CLASSES[d % 3] for d in dates], [f"2000-01-{d + 1:02d}" for d in dates])


def run(ts: TrainSet, training: dict[str, Any], models: list[str] = FAST) -> Any:
    folds = make_folds(ts.y, ts.groups, ts.ids, training)
    return folds, evaluate(ts, folds, models, training)


def summary(result: Any, model: str, scheme: str, metric: str = "balanced_accuracy") -> float:
    return float(result.summary.loc[(model, scheme), (metric, "mean")])


def verdict(result: Any, model: str) -> str:
    return result.decision.set_index("model").loc[model, "verdict"]


def test_every_model_sees_identical_folds(training: dict[str, Any]) -> None:
    folds, result = run(signal_set(), training)
    keys = ["scheme", "repeat", "fold", "video_id"]
    expected = folds.table.query("fold >= 0")[keys].sort_values(keys).reset_index(drop=True)
    real = result.predictions[result.predictions["scheme"] != PERMUTED]
    for model in FAST:
        got = real[real["model"] == model][keys].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_baselines_always_run_beside_the_chosen_models(training: dict[str, Any]) -> None:
    _, result = run(signal_set(), training, ["logreg"])
    assert list(result.decision["model"]) == ["majority", "date_only", "logreg"]


def test_preprocessing_is_fitted_on_the_training_rows_of_each_fold(
    training: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """EC-9: the rows left out of every fit are exactly one test fold."""
    ts = signal_set()
    fitted_on: list[set[int]] = []
    real_fit = evaluate_module.fit_preprocess

    def spy(X: pd.DataFrame, kinds: Any) -> Any:
        fitted_on.append(set(X.index))
        return real_fit(X, kinds)

    monkeypatch.setattr(evaluate_module, "fit_preprocess", spy)
    folds, _ = run(ts, training)
    position = dict(zip(ts.ids, ts.X.index))
    test_sets = [
        {position[v] for v in group["video_id"]} for _, group in folds.table.query("fold >= 0").groupby(["scheme", "repeat", "fold"])
    ]
    assert fitted_on
    for rows in fitted_on:
        left_out = set(ts.X.index) - rows
        assert left_out in test_sets


def test_metrics_come_from_the_pooled_out_of_fold_predictions_of_each_repeat(training: dict[str, Any]) -> None:
    """D-017: one score per repeat over all its folds' test fish; no plain accuracy anywhere (PRD §6.7)."""
    _, result = run(signal_set(signal=1.0), training)
    rows = result.predictions.query("model == 'logreg' and scheme == @SCHEME_B and repeat == 1")
    row = result.metrics.query("model == 'logreg' and scheme == @SCHEME_B and repeat == 1").iloc[0]
    assert row["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(rows["label"], rows["predicted"]))
    proba = rows[[f"p:{c}" for c in CLASSES]].to_numpy()
    assert row["log_loss"] == pytest.approx(log_loss(rows["label"], proba, labels=list(CLASSES)))
    assert row["fish"] == len(rows) == 36
    assert set(METRICS) == {"balanced_accuracy", "macro_f1", "log_loss", "top3_accuracy"}
    assert "accuracy" not in result.metrics.columns


def test_per_class_scores_pool_every_repeat(training: dict[str, Any]) -> None:
    _, result = run(signal_set(), training)
    table = result.per_class.query("model == 'logreg' and scheme == @SCHEME_B")
    assert set(table["label"]) == set(CLASSES)
    assert table["support"].sum() == 36 * training["repeats"]
    assert {"precision", "recall", "f1"} <= set(table.columns)


def test_spread_is_the_standard_deviation_across_repeats(training: dict[str, Any]) -> None:
    _, result = run(signal_set(signal=1.0), training)
    scores = result.metrics.query("model == 'logreg' and scheme == @SCHEME_A")["balanced_accuracy"]
    assert result.summary.loc[("logreg", SCHEME_A), ("balanced_accuracy", "std")] == pytest.approx(scores.std())


def test_permutation_shuffles_labels_only_among_fish_of_one_date() -> None:
    y = pd.Series(list("AABBCC" * 4))
    dates = pd.Series([f"d{i % 4}" for i in range(len(y))])
    shuffled = permute_within_date(y, dates, np.random.default_rng(0))
    for date in dates.unique():
        assert sorted(shuffled[dates == date]) == sorted(y[dates == date])
    assert not shuffled.equals(y)


def test_a_planted_compound_signal_is_judged_useful(training: dict[str, Any]) -> None:
    _, result = run(signal_set(signal=4.0), training)
    assert summary(result, "logreg", SCHEME_A) > 0.9
    assert summary(result, "logreg", PERMUTED) < 0.7
    assert verdict(result, "logreg") == USEFUL
    assert verdict(result, "majority") == verdict(result, "date_only") == BASELINE


def test_leakage_canary_label_that_depends_only_on_the_date(training: dict[str, Any]) -> None:
    """Plan U11: scheme B looks great, scheme A is near chance, permuting within dates changes nothing,
    and the decision rule refuses the model."""
    _, result = run(date_label_set(), training)
    assert summary(result, "logreg", SCHEME_B) > 0.9
    assert summary(result, "logreg", SCHEME_A) < 0.6  # chance is 1/3
    assert abs(summary(result, "logreg", PERMUTED) - summary(result, "logreg", SCHEME_B)) < 0.1
    assert summary(result, "date_only", SCHEME_B) == 1.0
    assert verdict(result, "logreg") == NOT_USEFUL


def test_pinned_fish_are_never_scored_in_scheme_a(training: dict[str, Any]) -> None:
    ts = signal_set()
    one_date = ts.groups == "2000-01-01"
    y = ts.y.where(~(one_date & (ts.y == "VEHICLE")), "COMPOUND_C")  # a class seen on one date only
    ts = as_trainset(ts.X, list(y), list(ts.groups))
    folds, result = run(ts, training)
    pinned = set(ts.ids[y == "COMPOUND_C"])
    assert folds.date_confounded == ("COMPOUND_C",)
    assert not pinned & set(result.predictions.query("scheme == @SCHEME_A")["video_id"])
    assert pinned <= set(result.predictions.query("scheme == @SCHEME_B")["video_id"])


def test_without_scheme_a_the_decision_is_undecided(training: dict[str, Any]) -> None:
    ts = signal_set(n_dates=3)
    dates = [f"2000-01-0{CLASSES.index(label) + 1}" for label in ts.y]  # every class on its own single date
    folds, result = run(as_trainset(ts.X, list(ts.y), dates), training)
    assert folds.k[SCHEME_A] == 0
    assert SCHEME_A not in set(result.metrics["scheme"])
    assert verdict(result, "logreg") == UNDECIDED


def test_vehicle_versus_drug_is_scored_in_scheme_a(training: dict[str, Any]) -> None:
    _, result = run(signal_set(signal=4.0), {**training, "vehicle_compound": " vehicle "})
    row = result.vehicle.set_index("model").loc["logreg"]
    assert row["auroc_mean"] > 0.9 and 0 <= row["balanced_accuracy_mean"] <= 1
    _, without = run(signal_set(), {**training, "vehicle_compound": "OTHER"})
    assert without.vehicle.empty


def test_same_seed_gives_identical_metrics(training: dict[str, Any]) -> None:
    """EC-14 (CPU)."""
    ts = signal_set(signal=1.0)
    first = run(ts, training)[1].metrics
    pd.testing.assert_frame_equal(first, run(ts, training)[1].metrics)
