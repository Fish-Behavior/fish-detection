"""Fold-fitted preprocessing (plan U9, T2.1): EC-9, EC-2, log1p on count/duration kinds, JSON round trip."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dcs.preprocess import Preprocess, fit_preprocess

KINDS = {"bouts": "count", "latency_s": "duration", "share": "fraction", "speed": "continuous", "ntt_tdm": "continuous"}


def frame(rows: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "bouts": rng.integers(0, 20, rows).astype(float),
            "latency_s": rng.uniform(0, 600, rows),
            "share": rng.uniform(0, 1, rows),
            "speed": rng.uniform(10, 100, rows),
            "ntt_tdm": rng.uniform(0, 300, rows),
        }
    )


def test_training_rows_come_out_standardized() -> None:
    train = frame()
    out = fit_preprocess(train, KINDS).transform(train)
    assert out.shape == train.shape
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=0), 1)


def test_extreme_test_fold_does_not_move_the_fitted_statistics() -> None:
    """EC-9: statistics come from the rows passed to fit; transforming other rows changes nothing."""
    train = frame()
    fitted = fit_preprocess(train, KINDS)
    before = json.dumps(fitted.as_json())
    extreme = frame(rows=3, seed=1) * 1e6
    out = fitted.transform(extreme)
    assert json.dumps(fitted.as_json()) == before
    assert (np.abs(out[:, list(KINDS).index("speed")]) > 1e3).all()  # scaled by the training spread, not its own
    np.testing.assert_allclose(fitted.transform(train), fit_preprocess(train, KINDS).transform(train))


def test_missing_values_get_the_training_fold_median() -> None:
    """EC-2: a fish without NTT gets the median of the training fold, never one computed from the test rows."""
    train = frame()
    train.loc[0, "ntt_tdm"] = np.nan
    fitted = fit_preprocess(train, KINDS)
    median = train["ntt_tdm"].median()  # pandas skips the gap
    assert fitted.fill["ntt_tdm"] == pytest.approx(median)
    test = frame(rows=2, seed=3)
    test["ntt_tdm"] = [np.nan, 1e9]
    out = fitted.transform(test)
    column = list(KINDS).index("ntt_tdm")
    assert out[0, column] == pytest.approx((median - fitted.mean["ntt_tdm"]) / fitted.std["ntt_tdm"])
    assert not np.isnan(out).any()


def test_log1p_only_on_count_and_duration_kinds() -> None:
    train = frame()
    fitted = fit_preprocess(train, KINDS)
    assert set(fitted.log1p) == {"bouts", "latency_s"}
    assert fitted.mean["bouts"] == pytest.approx(np.log1p(train["bouts"]).mean())
    assert fitted.mean["share"] == pytest.approx(train["share"].mean())
    assert fitted.fill["latency_s"] == pytest.approx(np.log1p(train["latency_s"]).median())


def test_json_round_trip_gives_identical_transforms() -> None:
    train = frame()
    train.loc[2, "speed"] = np.nan
    fitted = fit_preprocess(train, KINDS)
    reloaded = Preprocess.from_json(json.loads(json.dumps(fitted.as_json())))
    assert reloaded == fitted
    test = frame(rows=5, seed=9)
    np.testing.assert_array_equal(reloaded.transform(test), fitted.transform(test))


def test_constant_and_empty_columns_in_a_fold_cause_no_division_by_zero() -> None:
    """A column can be constant or empty inside one training fold even when it varies over the whole set."""
    train = frame()
    train["share"] = 0.5
    train["ntt_tdm"] = np.nan
    fitted = fit_preprocess(train, KINDS)
    out = fitted.transform(pd.concat([train, frame(rows=2, seed=4)]))
    assert np.isfinite(out).all()
    assert fitted.std["share"] == 1.0 and fitted.fill["ntt_tdm"] == 0.0


def test_columns_follow_the_fitted_feature_order() -> None:
    train = frame()
    fitted = fit_preprocess(train, KINDS)
    shuffled = train[list(reversed(train.columns))]
    np.testing.assert_array_equal(fitted.transform(shuffled), fitted.transform(train))
    assert fitted.features == tuple(train.columns)


def test_category_features_become_one_hot_columns_fitted_on_the_fold() -> None:
    """Demographics (U14): one 0/1 column per category seen in training; an unseen or missing category is all 0."""
    train = frame().assign(sex=["F", "M"] * 4)
    fitted = fit_preprocess(train, {**KINDS, "sex": "category"})
    assert fitted.categories == {"sex": ["F", "M"]}
    assert list(fitted.mean)[-2:] == ["sex=F", "sex=M"]
    test = frame(rows=3, seed=5).assign(sex=["M", "X", None])
    out = fitted.transform(test)
    assert out.shape == (3, len(KINDS) + 2)
    raw = (out[:, -2:] * np.array([fitted.std["sex=F"], fitted.std["sex=M"]])) + np.array([fitted.mean["sex=F"], fitted.mean["sex=M"]])
    np.testing.assert_allclose(raw, [[0, 1], [0, 0], [0, 0]], atol=1e-12)
    reloaded = Preprocess.from_json(json.loads(json.dumps(fitted.as_json())))
    np.testing.assert_array_equal(reloaded.transform(test), out)
