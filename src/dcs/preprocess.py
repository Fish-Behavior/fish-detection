"""Feature matrix -> model input, fitted on one training fold (plan U9, T2.2; PRD §5.4, §5.5; EC-2, EC-9).

`fit_preprocess` sees only the training rows of a fold: `log1p` on count and duration features, then the
median of each feature fills its gaps (missing NTT, EC-2; a `-` half's velocity, D-051), then each feature
is standardized with that fold's mean and spread. The test rows are transformed with those numbers and never
fitted on (EC-9). `as_json` / `from_json` carry the fitted numbers into the model folder (`preprocess.json`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from dcs.config import ConfigError

LOG_KINDS = ("count", "duration")  # heavy-tailed, never negative (PRD §5.5)


@dataclass(frozen=True)
class Preprocess:
    features: tuple[str, ...]  # input column order of the model
    log1p: tuple[str, ...]
    fill: dict[str, float]  # training-fold median, after log1p
    mean: dict[str, float]
    std: dict[str, float]  # 1.0 where the fold is constant, so nothing divides by zero

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Rows of `X` (any column order, extra columns ignored) as a float matrix in `features` order."""
        values = _logged(X[list(self.features)], self.log1p).fillna(self.fill)
        return ((values - pd.Series(self.mean)) / pd.Series(self.std)).to_numpy(dtype=float)

    def as_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Preprocess:
        return cls(
            features=tuple(data["features"]),
            log1p=tuple(data["log1p"]),
            fill=dict(data["fill"]),
            mean=dict(data["mean"]),
            std=dict(data["std"]),
        )


def fit_preprocess(X: pd.DataFrame, kinds: Mapping[str, str]) -> Preprocess:
    """Fit on the training rows `X`; `kinds` is the TrainSet's feature -> kind map."""
    categorical = [name for name in X.columns if kinds[name] == "category"]
    if categorical:
        raise ConfigError(
            f"Feature(s) {', '.join(categorical)} are categories, which are not encoded yet (U14). "
            "Set training.use_demographics: false."
        )
    log = tuple(name for name in X.columns if kinds[name] in LOG_KINDS)
    values = _logged(X, log)
    fill = values.median().fillna(0.0)  # a column empty in this fold becomes 0, then constant
    filled = values.fillna(fill)
    std = filled.std(ddof=0).replace(0.0, 1.0)
    return Preprocess(
        features=tuple(X.columns),
        log1p=log,
        fill={name: float(v) for name, v in fill.items()},
        mean={name: float(v) for name, v in filled.mean().items()},
        std={name: float(v) for name, v in std.items()},
    )


def _logged(X: pd.DataFrame, log: tuple[str, ...]) -> pd.DataFrame:
    values = X.astype(float)
    values[list(log)] = np.log1p(values[list(log)])
    return values
