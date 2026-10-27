"""Hand-built training sets shared by test_dcs_evaluate.py and test_dcs_report.py; not a test module itself."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dcs.trainset import TrainSet

CLASSES = ("COMPOUND_A", "COMPOUND_B", "VEHICLE")


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
