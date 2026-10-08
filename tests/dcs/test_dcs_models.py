"""Baseline models (plan U10, T2.3; PRD §6.2): one interface, date-only rule (D-011), class weights, seeds."""

from __future__ import annotations

import numpy as np
import pytest

from dcs.models import BASELINES, make_model

DATES = np.array(["2000-01-01", "2000-01-01", "2000-01-10", "2000-01-10", "2000-01-10", "2000-01-20"])
LABELS = np.array(["A", "B", "B", "B", "C", "C"])


def separable(n: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three classes whose first feature tells them apart."""
    rng = np.random.default_rng(seed)
    y = np.array(["A", "B", "C"] * (n // 3))
    X = rng.normal(size=(n, 4))
    X[:, 0] += 5 * (y == "B") + 10 * (y == "C")
    dates = np.array([f"2000-01-{1 + i % 5:02d}" for i in range(n)])
    return X, y, dates


@pytest.mark.parametrize("name", BASELINES)
def test_every_baseline_shares_one_interface(name: str) -> None:
    X, y, dates = separable()
    model = make_model(name, seed=0).fit(X, y, dates)
    proba = model.predict_proba(X[:7], dates[:7])
    assert list(model.classes_) == ["A", "B", "C"]
    assert proba.shape == (7, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_baselines_are_the_prd_list_without_the_mlp() -> None:
    assert BASELINES == ("majority", "date_only", "logreg", "random_forest", "hist_gb")


def test_majority_predicts_the_class_shares_of_the_training_rows() -> None:
    model = make_model("majority", seed=0).fit(np.zeros((6, 1)), LABELS, DATES)
    proba = model.predict_proba(np.zeros((2, 1)), DATES[:2])
    np.testing.assert_allclose(proba, [[1 / 6, 3 / 6, 2 / 6]] * 2)


def test_date_only_uses_the_label_mix_of_the_same_date() -> None:
    """Scheme B: the test fish's own date was seen in training."""
    model = make_model("date_only", seed=0).fit(np.zeros((6, 1)), LABELS, DATES)
    proba = model.predict_proba(np.zeros((2, 1)), np.array(["2000-01-10", "2000-01-20"]))
    np.testing.assert_allclose(proba, [[0, 2 / 3, 1 / 3], [0, 0, 1]])


def test_date_only_uses_the_nearest_training_date_for_an_unseen_date() -> None:
    """Scheme A (D-011): the held-out date is unseen, so the nearest training date stands in."""
    model = make_model("date_only", seed=0).fit(np.zeros((6, 1)), LABELS, DATES)
    proba = model.predict_proba(np.zeros((2, 1)), np.array(["2000-01-02", "2000-02-01"]))
    np.testing.assert_allclose(proba, [[0.5, 0.5, 0], [0, 0, 1]])


def test_date_only_breaks_a_tie_towards_the_earlier_date() -> None:
    model = make_model("date_only", seed=0).fit(np.zeros((6, 1)), LABELS, DATES)
    proba = model.predict_proba(np.zeros((1, 1)), np.array(["2000-01-15"]))  # 5 days from 01-10 and 01-20
    np.testing.assert_allclose(proba, [[0, 2 / 3, 1 / 3]])


def test_date_only_falls_back_to_the_class_shares_when_a_date_is_not_a_date() -> None:
    model = make_model("date_only", seed=0).fit(np.zeros((6, 1)), LABELS, DATES)
    proba = model.predict_proba(np.zeros((1, 1)), np.array(["<date>"]))
    np.testing.assert_allclose(proba, [[1 / 6, 3 / 6, 2 / 6]])


@pytest.mark.parametrize("name", ["logreg", "random_forest", "hist_gb"])
def test_learning_baselines_find_a_planted_signal(name: str) -> None:
    X, y, dates = separable(60)
    test_X, test_y, test_dates = separable(30, seed=1)
    model = make_model(name, seed=0).fit(X, y, dates)
    predicted = model.classes_[model.predict_proba(test_X, test_dates).argmax(axis=1)]
    assert (predicted == test_y).mean() > 0.9


@pytest.mark.parametrize("name", ["logreg", "random_forest", "hist_gb"])
def test_learning_baselines_weight_classes(name: str) -> None:
    assert make_model(name, seed=0).estimator.class_weight == "balanced"


@pytest.mark.parametrize("name", BASELINES)
def test_same_seed_gives_identical_probabilities(name: str) -> None:
    X, y, dates = separable(seed=2)
    X[:, 0] += np.random.default_rng(3).normal(scale=6, size=len(y))  # overlap, so the forests have choices to make
    runs = [make_model(name, seed=7).fit(X, y, dates).predict_proba(X, dates) for _ in range(2)]
    np.testing.assert_array_equal(runs[0], runs[1])


def test_unknown_or_unbuilt_model_is_an_error() -> None:
    with pytest.raises(ValueError, match="U13"):
        make_model("mlp", seed=0)
    with pytest.raises(ValueError, match="bogus"):
        make_model("bogus", seed=0)
