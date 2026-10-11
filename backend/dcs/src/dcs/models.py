"""Baseline models behind one interface (plan U10, T2.4; PRD §6.2; D-011, D-053).

Every model has `fit(X, y, dates)`, `predict_proba(X, dates)` and `classes_` (sorted labels, the column order of
the probabilities). `X` is the preprocessed matrix of one fold; only the date-only baseline reads `dates`. The MLP
(`dcs.mlp`, U13) has the same interface; torch is imported only when it is built, never here.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

BASELINES = ("majority", "date_only", "logreg", "random_forest", "hist_gb")  # PRD §6.2, in report order


class SklearnModel:
    """A scikit-learn classifier that ignores the dates."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray) -> SklearnModel:
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X: np.ndarray, dates: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)


class DateOnly:
    """D-011: the label mix of the test fish's date in training, else of the nearest training date (ties: the
    earlier one), else of all training fish (a date that is not a date). It sees no feature: it measures how much
    the day alone explains, the leakage ceiling of PRD §6.2."""

    def fit(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray) -> DateOnly:
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        counts = pd.crosstab(np.asarray(dates), y).reindex(columns=self.classes_, fill_value=0)
        self._mix = counts.div(counts.sum(axis=1), axis=0)  # rows sorted by date text; ISO text sorts by day
        self._days = _days(self._mix.index)
        self._prior = pd.Series(y).value_counts(normalize=True).reindex(self.classes_).to_numpy()
        return self

    def predict_proba(self, X: np.ndarray, dates: np.ndarray) -> np.ndarray:
        rows = []
        for date, day in zip(dates, _days(dates)):
            if date in self._mix.index:
                rows.append(self._mix.loc[date].to_numpy())
            elif np.isnan(day) or np.isnan(self._days).all():
                rows.append(self._prior)
            else:
                rows.append(self._mix.iloc[int(np.nanargmin(np.abs(self._days - day)))].to_numpy())
        return np.vstack(rows).astype(float)


def torch_available() -> bool:
    """True when torch can be imported, found without importing it."""
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # a hidden or broken torch entry in sys.modules
        return False


def make_model(name: str, seed: int, mlp: Mapping[str, Any] | None = None, device: str = "cpu") -> Any:
    """A fresh, unfitted model. Baseline settings are the PRD's (L2 logreg, balanced classes) plus D-053's sizes;
    the MLP needs the `training.mlp` settings and a resolved device (D-059)."""
    if name == "mlp":
        if mlp is None:
            raise ValueError("The MLP needs the training.mlp settings")
        from dcs.mlp import MLP  # torch is imported only here

        return MLP(mlp, seed, device)
    if name == "date_only":
        return DateOnly()
    estimators = {
        "majority": lambda: DummyClassifier(strategy="prior"),
        "logreg": lambda: LogisticRegression(class_weight="balanced", max_iter=5000),  # L2 is the default
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "hist_gb": lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
    }
    if name not in estimators:
        raise ValueError(f"Unknown model {name!r}; expected one of {', '.join(BASELINES)} or mlp")
    return SklearnModel(estimators[name]())


def _days(dates: Any) -> np.ndarray:
    """Days since 1970 per ISO date text (as prepds catalog writes it); NaN where the text is not one."""
    stamps = pd.to_datetime(pd.Series(np.asarray(dates), dtype=object), errors="coerce", format="ISO8601")
    return ((stamps - pd.Timestamp(0)) / pd.Timedelta(days=1)).to_numpy(dtype=float)
