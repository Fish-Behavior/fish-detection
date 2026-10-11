"""Stage 2: dose within compound (plan U15, T4.2; PRD §6.5, FR-7; D-064).

One model per compound with at least two eligible doses (the dose training set's class filter already removed small
doses and compounds left with one dose, vehicle included). Each runs on its own folds; scheme A is usually skipped
because doses of one compound are rarely recorded on the same date, so the scores are scheme B next to date-only and
the within-date permutation. The FR-9 variant (dose fish minus the reference vehicle median) runs on the same folds:
the PRD's only informative test for dose. Every result is exploratory and date-confounded.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from dcs.ablations import reference_groups, vehicle_normalize
from dcs.audit import markdown_table
from dcs.config import ConfigError
from dcs.evaluate import PERMUTED, evaluate
from dcs.folds import SCHEME_A, SCHEME_B, make_folds
from dcs.gold_rules import compound_label
from dcs.report import StageResult, score_cell
from dcs.trainset import DOSE_SEPARATOR, TrainSet, build_trainset


@dataclass(frozen=True)
class DoseModel:
    compound: str
    raw: StageResult
    normalized: StageResult | None  # FR-9 variant on the same folds; None when there is no vehicle reference
    note: str = ""


def stage_key(compound: str) -> str:
    """How a compound's dose model is named in the run folder (stage column, report section)."""
    return f"dose {compound}"


def run_stage2(
    table: pd.DataFrame,
    described: Mapping[str, Any],
    training: Mapping[str, Any],
    models: Sequence[str],
    device: str,
    setups: Mapping[str, Any],
    log: Callable[[str], None],
) -> tuple[list[DoseModel], list[str]]:
    """The dose models, and notes (a dose stage that cannot be built is a note, never a stop)."""
    try:
        ts = build_trainset(table, described, training, "dose")
    except ConfigError as error:
        return [], [f"dose stage cannot be built: {error}"]
    compound_of = ts.y.str.split(DOSE_SEPARATOR, n=1).str[0]
    vehicle = compound_label(training["vehicle_compound"])
    vehicle_rows = table[table["compound"].map(compound_label) == vehicle]
    out = []
    for compound in sorted(compound_of.unique()):
        sub = ts.take(compound_of == compound)
        folds = make_folds(sub.y, sub.groups, sub.ids, training)
        log(f"{stage_key(compound)}: {len(sub.y)} fish, {len(sub.classes)} doses")
        raw = StageResult(sub, folds, evaluate(sub, folds, models, training, log, device))
        if vehicle_rows.empty:
            note = f"FR-9 not run: the vehicle {vehicle} is not in the table (set training.vehicle_compound)"
            out.append(DoseModel(compound, raw, None, note))
            continue
        groups, by_date = reference_groups(training, table["date"], setups)
        normalized, _ = vehicle_normalize(_with_vehicle(sub, vehicle_rows, vehicle), vehicle, groups, by_date)
        log(f"{stage_key(compound)}: FR-9 variant")
        out.append(DoseModel(compound, raw, StageResult(normalized, folds, evaluate(normalized, folds, models, training, log, device))))
    return out, []


def render_stage2(dose_models: Sequence[DoseModel]) -> list[str]:
    """Report summary: every compound's dose models with scheme B raw, permuted and vehicle-normalized side by side."""
    rows = []
    for m in dose_models:
        raw = m.raw.evaluation.summary
        normalized = None if m.normalized is None else m.normalized.evaluation.summary
        for model in m.raw.evaluation.decision["model"]:
            rows.append(
                {
                    "compound": m.compound,
                    "doses": len(m.raw.trainset.classes),
                    "fish": len(m.raw.trainset.y),
                    "model": model,
                    "scheme A": score_cell(raw, model, SCHEME_A),
                    "scheme B": score_cell(raw, model, SCHEME_B),
                    "B permuted": score_cell(raw, model, PERMUTED),
                    "B vehicle-normalized": m.note or score_cell(normalized, model, SCHEME_B),
                }
            )
    return [
        "",
        "## Stage 2 summary: dose within compound (exploratory, PRD §6.5)",
        "",
        "Balanced accuracy, mean ± spread. Doses of one compound are rarely recorded on the same date, so a dose model "
        "can score by recognizing the date: compare every model with `date_only` and with `B permuted`. The vehicle-"
        "normalized column (FR-9) is the only informative one: if a model still separates doses there, the day's "
        "vehicle level does not explain it. No row here is a claim of dose recognition.",
        "",
        markdown_table(pd.DataFrame(rows)) if rows else "No compound has two eligible doses.",
    ]


def _with_vehicle(sub: TrainSet, rows: pd.DataFrame, vehicle: str) -> TrainSet:
    """A compound's dose fish plus every vehicle fish (same feature columns), so vehicle_normalize has a reference."""
    return dataclasses.replace(
        sub,
        X=pd.concat([sub.X, rows[list(sub.features)]]),
        y=pd.concat([sub.y, pd.Series(vehicle, index=rows.index, name=sub.y.name)]),
        groups=pd.concat([sub.groups, rows["date"]]),
        ids=pd.concat([sub.ids, rows["video_id"]]),
        classes={**sub.classes, vehicle: len(rows)},
    )
