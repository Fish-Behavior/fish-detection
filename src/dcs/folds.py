"""Training set -> the folds every model shares (plan U7, T1.15; PRD §6.4).

Scheme A holds out whole dates (stratified group K-fold, group = date); scheme B is stratified K-fold, the
leakage twin. Both run `training.repeats` times on seeds drawn from `training.seed` (EC-14). A class seen on
one date cannot be held out: in scheme A its fish are pinned to training and flagged date-confounded (EC-5).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SCHEME_A = "A"  # date held out (primary)
SCHEME_B = "B"  # random stratified (reference)
PINNED = -1  # scheme-A fold of a date-confounded fish: always on the training side, never scored


@dataclass(frozen=True)
class Folds:
    table: pd.DataFrame  # video_id, label, date, scheme, repeat, fold: one row per fish, scheme and repeat
    k: dict[str, int]  # folds used per scheme; 0 = scheme skipped
    date_confounded: tuple[str, ...]  # classes on one date: pinned in scheme A, left out of its score
    notes: tuple[str, ...]  # for the audit and the report


def make_folds(y: pd.Series, groups: pd.Series, ids: pd.Series, training: Mapping[str, Any]) -> Folds:
    """Scheme A and B folds for a TrainSet's `y`, `groups` (date) and `ids`."""
    if not (y.index.equals(groups.index) and y.index.equals(ids.index)):
        raise ValueError("y, groups and ids must share one index (pass the fields of one TrainSet)")
    dates_per_class = groups.groupby(y).nunique()
    confounded = tuple(sorted(str(label) for label in dates_per_class.index[dates_per_class == 1]))
    held = ~y.isin(confounded)
    k = training["folds"]
    notes = []
    if confounded:
        notes.append(
            f"scheme A: {', '.join(confounded)} seen on one date only; pinned to training and left out of the "
            "scheme-A score (date-confounded)"
        )
    n_dates = groups[held].nunique()
    k_a = min(k, n_dates) if n_dates >= 2 else 0
    if not k_a:
        notes.append(f"scheme A skipped: {n_dates} date(s) hold fish of a class seen on two or more dates, 2 needed")
    elif k_a < k:
        notes.append(f"scheme A: only {n_dates} dates can be held out, so K is {k_a}, not {k}")
    k_b = min(k, int(y.value_counts().max()))  # sklearn refuses K above every class size
    if k_b < k:
        notes.append(f"scheme B: the largest class has {k_b} fish, so K is {k_b}, not {k}")

    tables = []
    for repeat in range(training["repeats"]):
        state = int(np.random.SeedSequence([training["seed"], repeat]).generate_state(1)[0])
        if k_a:
            fold = pd.Series(PINNED, index=y.index)
            splitter = StratifiedGroupKFold(k_a, shuffle=True, random_state=state)
            for number, (_, test) in enumerate(splitter.split(y[held], y[held], groups[held])):
                fold[y[held].index[test]] = number
            tables.append(_rows(SCHEME_A, repeat, fold, y, groups, ids))
            notes += _one_fold_notes(repeat, fold[held], y[held], groups[held])
        fold = pd.Series(PINNED, index=y.index)
        for number, (_, test) in enumerate(StratifiedKFold(k_b, shuffle=True, random_state=state).split(y, y)):
            fold.iloc[test] = number
        tables.append(_rows(SCHEME_B, repeat, fold, y, groups, ids))
    return Folds(pd.concat(tables, ignore_index=True), {SCHEME_A: k_a, SCHEME_B: k_b}, confounded, tuple(notes))


def _one_fold_notes(repeat: int, fold: pd.Series, y: pd.Series, groups: pd.Series) -> list[str]:
    """EC-6: a class on two or more dates whose dates all landed in one fold leaves that fold without it."""
    by_class = pd.DataFrame({"fold": fold, "date": groups}).groupby(y).nunique()
    stuck = by_class.index[(by_class["date"] >= 2) & (by_class["fold"] == 1)]
    return [
        f"scheme A, repeat {repeat}: the dates of {label} all fell in one fold, which trains without it"
        for label in stuck
    ]


def _rows(scheme: str, repeat: int, fold: pd.Series, y: pd.Series, groups: pd.Series, ids: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": ids.to_numpy(),
            "label": y.to_numpy(),
            "date": groups.to_numpy(),
            "scheme": scheme,
            "repeat": repeat,
            "fold": fold.to_numpy(),
        }
    )
