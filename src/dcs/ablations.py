"""Ablations of the compound stage (plan U14, T3.4; PRD §6.8, FR-9; D-008, D-061).

Each switch ablation (NTT, demographics, depth) flips one feature group and re-runs every model on the main run's
folds (new folds, and the row says so, when the switch changes which fish are kept). The FR-9 pair runs the same
models on the same non-vehicle fish and folds twice: raw, and vehicle-normalized (each feature minus the median of
reference vehicle fish, D-008). The keep rule says whether NTT earns its place.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dcs.config import ConfigError
from dcs.audit import markdown_table
from dcs.evaluate import REFERENCE, evaluate
from dcs.folds import SCHEME_A, SCHEME_B, Folds, make_folds
from dcs.gold_rules import compound_label
from dcs.preprocess import CATEGORY, LOG_KINDS
from dcs.report import StageResult, score_cell
from dcs.trainset import TrainSet, build_trainset

SWITCHES = (
    ("use_ntt", "NTT features"),
    ("use_demographics", "sex, strain and age"),
    ("use_depth", "depth features (camera framing, D-015)"),
)
FR9_RAW = "non-vehicle, raw"
FR9_NORMALIZED = "non-vehicle, vehicle-normalized"
MIN_REFERENCE = 2  # vehicle fish needed for a date (or setup) to be its own reference (PRD FR-9)


@dataclass(frozen=True)
class Ablation:
    name: str
    what: str
    result: StageResult | None  # None when it could not run; `note` says why
    note: str = ""


def run_ablations(
    table: pd.DataFrame,
    described: Mapping[str, Any],
    training: Mapping[str, Any],
    main: StageResult,
    models: Sequence[str],
    device: str,
    setups: Mapping[str, Any],
    log: Callable[[str], None],
) -> list[Ablation]:
    """Every ablation of `main` (a compound-stage result); `setups` maps date -> framing setup (audit, EC-31)."""
    out = []
    for key, what in SWITCHES:
        flipped = not training[key]
        name, what = f"{key}={str(flipped).lower()}", f"{what} {'on' if flipped else 'off'}"
        try:
            ts = build_trainset(table, described, {**training, key: flipped}, main.trainset.stage)
        except ConfigError as error:
            out.append(Ablation(name, what, None, str(error)))
            continue
        if ts.ids.equals(main.trainset.ids):
            folds = main.folds
        else:  # the switch changed which fish or classes are kept, so the main folds no longer fit
            folds, what = make_folds(ts.y, ts.groups, ts.ids, training), f"{what} (different fish: new folds)"
        log(f"ablation {name}")
        out.append(Ablation(name, what, StageResult(ts, folds, evaluate(ts, folds, models, training, log, device))))

    vehicle = compound_label(training["vehicle_compound"])
    if vehicle not in main.trainset.classes:
        note = f"not run: the vehicle {vehicle} is not a kept class (set training.vehicle_compound)"
        return out + [Ablation(FR9_RAW, "vehicle fish left out", None, note), Ablation(FR9_NORMALIZED, "FR-9", None, note)]
    raw = without_class(main.trainset, vehicle)
    if len(raw.classes) < 2:
        note = "not run: fewer than 2 classes besides the vehicle"
        return out + [Ablation(FR9_RAW, "vehicle fish left out", None, note), Ablation(FR9_NORMALIZED, "FR-9", None, note)]
    normalized, levels = vehicle_normalize(main.trainset, vehicle, *reference_groups(training, main.trainset.groups, setups))
    keep = main.folds.table["label"] != vehicle
    folds = Folds(main.folds.table[keep], main.folds.k, tuple(c for c in main.folds.date_confounded if c != vehicle), main.folds.notes)
    reference = ", ".join(f"{level} for {n} date(s)" for level, n in levels.items() if n)
    for name, what, ts in (
        (FR9_RAW, "vehicle fish left out, features as they are", raw),
        (FR9_NORMALIZED, f"vehicle fish left out, features minus the reference vehicle median ({reference})", normalized),
    ):
        log(f"ablation {name}")
        out.append(Ablation(name, what, StageResult(ts, folds, evaluate(ts, folds, models, training, log, device))))
    return out


def without_class(ts: TrainSet, label: str) -> TrainSet:
    """The training set without the fish of one class (the vehicle in the FR-9 pair)."""
    return ts.take(ts.y != label)


def reference_groups(training: Mapping[str, Any], dates: pd.Series, setups: Mapping[str, Any]) -> tuple[dict[Any, Any], bool]:
    """The FR-9 reference groups and whether the date level comes first: camera epochs when `training.camera_epochs`
    is set (no date level), else the audit's framing setups after the date (D-061)."""
    epochs = training["camera_epochs"]
    if epochs is None:
        return dict(setups), True
    return {date: epoch_of(date, epochs) for date in dates.unique()}, False


def epoch_of(date: Any, starts: Sequence[Any]) -> str:
    """`epoch 1` before the first start date, `epoch 2` from it, and so on (ISO dates compare as text)."""
    return f"epoch {1 + sum(str(date) >= str(start) for start in starts)}"


def vehicle_normalize(
    ts: TrainSet, vehicle: str, setups: Mapping[str, Any], by_date: bool = True
) -> tuple[TrainSet, dict[str, int]]:
    """FR-9: every non-category feature (after log1p for counts and durations) minus the median of the reference
    vehicle fish: the fish's own date when it has 2+ vehicle fish (only with `by_date`), else its group in `setups`
    (framing setup, EC-31, or camera epoch) when that has 2+, else all vehicle fish (D-061). Vehicle fish are the
    reference only and leave the set (D-008). Also returns how many dates used each reference."""
    numeric = [f for f in ts.features if ts.kinds[f] != CATEGORY]
    values = ts.X[numeric].astype(float)
    logged = [f for f in numeric if ts.kinds[f] in LOG_KINDS]
    values[logged] = np.log1p(values[logged])
    is_vehicle = ts.y == vehicle
    reference = values[is_vehicle]
    setup = ts.groups.map(setups)
    per_date = reference.groupby(ts.groups[is_vehicle])
    per_setup = reference.groupby(setup[is_vehicle])
    group = "setup" if by_date else "camera epoch"
    levels = {"date": 0, group: 0, "all vehicle fish": 0}
    for date in ts.groups[~is_vehicle].unique():
        rows = (ts.groups == date) & ~is_vehicle
        date_setup = setups.get(date)
        if by_date and (ts.groups[is_vehicle] == date).sum() >= MIN_REFERENCE:
            level, median = "date", per_date.get_group(date).median()
        elif date_setup is not None and (setup[is_vehicle] == date_setup).sum() >= MIN_REFERENCE:
            level, median = group, per_setup.get_group(date_setup).median()
        else:
            level, median = "all vehicle fish", reference.median()
        values.loc[rows] = values.loc[rows] - median
        levels[level] += 1
    rest = without_class(ts, vehicle)
    X = rest.X.copy()
    X[numeric] = values.loc[rest.X.index]
    kinds = {f: ("continuous" if f in numeric else kind) for f, kind in ts.kinds.items()}  # logged once, here
    return dataclasses.replace(rest, X=X, kinds=kinds), levels


def keep_rule(with_group: pd.DataFrame, without_group: pd.DataFrame, model: str) -> tuple[bool | None, str]:
    """PRD §6.8: keep a feature group only if scheme-A balanced accuracy with it beats the run without it by more
    than the larger of the two spreads. `None` when scheme A is missing."""
    if (model, SCHEME_A) not in with_group.index or (model, SCHEME_A) not in without_group.index:
        return None, "undecided: no scheme A"
    a = with_group.loc[(model, SCHEME_A), "balanced_accuracy"]
    b = without_group.loc[(model, SCHEME_A), "balanced_accuracy"]
    gain, spread = float(a["mean"] - b["mean"]), float(max(a["std"], b["std"]))
    keep = gain > spread
    return keep, f"scheme A {float(a['mean']):.3f} with, {float(b['mean']):.3f} without: gain {gain:+.3f} vs spread {spread:.3f}"


def render_ablations(ablations: Sequence[Ablation], main: StageResult, training: Mapping[str, Any]) -> list[str]:
    """The report section: one row per ablation and model, the change in scheme-A balanced accuracy against the main
    run (the normalized FR-9 run against the raw one), and the NTT keep rule."""
    by_name = {a.name: a for a in ablations}
    raw = by_name.get(FR9_RAW)
    rows = []
    for a in ablations:
        if a.result is None:
            rows.append({"ablation": a.name, "change": a.what, "model": "-", "scheme A": a.note, "delta A": "-", "scheme B": "-"})
            continue
        base = raw.result.evaluation.summary if a.name == FR9_NORMALIZED and raw and raw.result else main.evaluation.summary
        summary = a.result.evaluation.summary
        for model in a.result.evaluation.decision["model"]:
            if model in REFERENCE:
                continue
            rows.append(
                {
                    "ablation": a.name,
                    "change": a.what,
                    "model": model,
                    "scheme A": score_cell(summary, model, SCHEME_A),
                    "delta A": _delta(summary, base, model),
                    "scheme B": score_cell(summary, model, SCHEME_B),
                }
            )
    lines = [
        "",
        "## Ablations (compound stage, the main run's folds unless the row says new folds; PRD §6.8)",
        "",
        "One change per row. `delta A` = scheme-A balanced accuracy minus the main run's; for the vehicle-normalized run, "
        "minus the raw non-vehicle run (same classes and folds, D-008). A change matters only when `delta A` is larger "
        "than the spreads (`±`).",
        "",
        markdown_table(pd.DataFrame(rows)),
    ]
    flipped = by_name.get(f"use_ntt={str(not training['use_ntt']).lower()}")
    if flipped is not None and flipped.result is not None:
        with_ntt, without_ntt = (
            (main.evaluation.summary, flipped.result.evaluation.summary)
            if training["use_ntt"]
            else (flipped.result.evaluation.summary, main.evaluation.summary)
        )
        lines += ["", "**NTT keep rule** (keep NTT only if scheme A improves by more than the spread):", ""]
        for model in main.evaluation.decision["model"]:
            if model not in REFERENCE:
                keep, reason = keep_rule(with_ntt, without_ntt, model)
                verdict = "undecided" if keep is None else "keep NTT" if keep else "NTT does not help"
                lines.append(f"- {model}: {verdict} ({reason})")
    return lines



def _delta(summary: pd.DataFrame, base: pd.DataFrame, model: str) -> str:
    if (model, SCHEME_A) not in summary.index or (model, SCHEME_A) not in base.index:
        return "-"
    change = summary.loc[(model, SCHEME_A), ("balanced_accuracy", "mean")] - base.loc[(model, SCHEME_A), ("balanced_accuracy", "mean")]
    return f"{float(change):+.3f}"
