"""Evaluation -> report.md and confusion PNGs (plan U12, T2.8; PRD FR-5, §6.4-6.7, AC-5, US-2).

The report puts the majority, date-only and within-date permuted scores on the same row as every model, gives the
date effect (scheme B - scheme A), the decision rule's verdict and reason, vehicle against drug, and per-class scores
of the best model. It says plainly when the data is unreviewed (temporarily accepted, D-033) and what is not yet
corrected for (camera framing, FR-9).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from dcs.audit import Audit, markdown_table
from dcs.config_rules import SOURCE_PROCESSED
from dcs.evaluate import PERMUTED, REFERENCE, Evaluation
from dcs.folds import SCHEME_A, SCHEME_B, Folds
from dcs.trainset import TrainSet

SKIPPED = "skipped"
READING = (
    "Scheme A holds whole dates out: it is the main score, the only one that says whether the model works on a day it "
    "never saw. Scheme B splits fish at random; same-date fish sit on both sides, so B can be won by recognizing the day.",
    "Date effect = B - A. A large positive gap means much of scheme B comes from the date.",
    "`majority` always predicts the class shares; `date_only` predicts from the experiment date alone (the leakage "
    "ceiling, D-011). B permuted = scheme B with labels shuffled among fish of the same date: a model that scores as "
    "well there learned the date, not the compound.",
    "A model is **useful** only when its scheme-A balanced accuracy beats majority and date_only by more than the "
    "spread (`±`) and beats its own permuted score (PRD §6.6).",
)
DOSE_CAVEAT = (
    "> **Exploratory (PRD §6.5).** Doses of one compound are rarely recorded on the same date, so scheme B can be won "
    "by recognizing the date and scheme A often cannot hold a dose out. Draw no claim about dose recognition from "
    "scheme-B results; compare every model with `date_only`."
)


@dataclass(frozen=True)
class StageResult:
    trainset: TrainSet
    folds: Folds
    evaluation: Evaluation


def scores_table(ev: Evaluation) -> pd.DataFrame:
    """One row per model, baselines first: balanced accuracy per scheme, the date effect, other scheme-A metrics."""
    summary = ev.summary

    def cell(model: str, scheme: str, metric: str = "balanced_accuracy") -> str:
        if (model, scheme) not in summary.index:
            return SKIPPED
        row = summary.loc[(model, scheme), metric]
        return f"{row['mean']:.3f} ± {row['std']:.3f}"

    def mean(model: str, scheme: str) -> float:
        return float(summary.loc[(model, scheme), ("balanced_accuracy", "mean")])

    verdicts = ev.decision.set_index("model")["verdict"]
    rows = []
    for model in ev.decision["model"]:
        held_out = (model, SCHEME_A) in summary.index
        rows.append(
            {
                "model": model,
                "scheme A": cell(model, SCHEME_A),
                "scheme B": cell(model, SCHEME_B),
                "B permuted": cell(model, PERMUTED),
                "date effect (B - A)": f"{mean(model, SCHEME_B) - mean(model, SCHEME_A):+.3f}" if held_out else "-",
                "A macro-F1": cell(model, SCHEME_A, "macro_f1"),
                "A log-loss": cell(model, SCHEME_A, "log_loss"),
                "A top-3": cell(model, SCHEME_A, "top3_accuracy"),
                "verdict": verdicts[model],
            }
        )
    return pd.DataFrame(rows)


def best_model(ev: Evaluation) -> tuple[str, str]:
    """The model (references only when nothing else ran) with the highest scheme-A balanced accuracy; scheme B
    when scheme A was skipped."""
    models = [m for m in ev.decision["model"] if m not in REFERENCE] or list(ev.decision["model"])
    scheme = SCHEME_A if any((m, SCHEME_A) in ev.summary.index for m in models) else SCHEME_B
    scores = {m: float(ev.summary.loc[(m, scheme), ("balanced_accuracy", "mean")]) for m in models}
    return max(scores, key=scores.__getitem__), scheme


def render_report(info: Mapping[str, Any], results: Mapping[str, StageResult], audit: Audit, notes: Sequence[str]) -> str:
    """report.md for one run: source, settings, caveats, how to read it, then one section per stage."""
    if info["gold_source"] == SOURCE_PROCESSED:
        source = (
            "**Source: UNREVIEWED pipeline output, treated as temporarily accepted (Q17, D-033).** No video has been "
            "reviewed, so labels and states are not validated and no result here is a claim about drug identity. "
            "Re-run on Accepted videos (`training.gold_source: accepted`) before drawing conclusions."
        )
    else:
        source = "**Source: Accepted gold folder.**"
    skipped = ", ".join(info["skipped_models"])
    lines = [
        f"# Training report {info['run_id']}",
        "",
        source,
        "",
        f"- Command: `{info['command']}`",
        f"- Git commit: {info['git_commit']}" + (" (with uncommitted changes)" if info["git_dirty"] else ""),
        f"- Stages: {', '.join(results)}",
        f"- Models: {', '.join(info['models'])}" + (f"; skipped: {skipped} (not built yet)" if skipped else ""),
        f"- Seed {info['seed']}, {info['folds']} folds, {info['repeats']} repeats. Scores come from the pooled "
        "out-of-fold predictions of each repeat; `±` is the standard deviation across repeats, the spread (D-017)",
        "- The data audit of the same table is `audit.md` in this folder",
    ]
    caveats = []
    if audit.facts.get("framing differs between dates"):
        caveats.append(
            "Camera framing differs between dates (EC-31, see `audit.md`): pixel speeds, state shares and NTT "
            "coverage follow the camera setup, so a model can score by recognizing the camera instead of the compound"
        )
    caveats.append("Features are not vehicle-normalized yet (FR-9 comes with the ablations, U14)")
    lines += ["", "## Caveats", "", *(f"- {text}" for text in [*caveats, *notes])]
    lines += ["", "## How to read this", "", *(f"- {text}" for text in READING)]
    for stage, result in results.items():
        lines += _stage_section(stage, result, audit)
    return "\n".join(lines) + "\n"


def save_confusion(result: StageResult, path: Path) -> None:
    """Row-normalized confusion matrix of the best model, pooled over repeats."""
    ev = result.evaluation
    model, scheme = best_model(ev)
    rows = ev.predictions[(ev.predictions["model"] == model) & (ev.predictions["scheme"] == scheme)]
    counts = confusion_matrix(rows["label"], rows["predicted"], labels=list(ev.classes))
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    n = len(ev.classes)
    figure = Figure(figsize=(max(4.0, 0.45 * n + 2), max(4.0, 0.45 * n + 2)))
    axes = figure.subplots()
    image = axes.imshow(share, vmin=0, vmax=1, cmap="Blues")
    axes.set_xticks(range(n), ev.classes, rotation=90)
    axes.set_yticks(range(n), ev.classes)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_title(f"{result.trainset.stage}: {model}, scheme {scheme}\n(share of each true class, all repeats)")
    figure.colorbar(image, ax=axes)
    figure.tight_layout()
    figure.savefig(path, dpi=120)


def _stage_section(stage: str, result: StageResult, audit: Audit) -> list[str]:
    ts, folds, ev = result.trainset, result.folds, result.evaluation
    lines = ["", f"## Stage: {stage}", ""]
    if stage == "dose":
        lines += [DOSE_CAVEAT, ""]
    dropped = ", ".join(f"{d['label']} ({d['fish']} fish, {d['reason']})" for d in ts.dropped_classes)
    lines += [
        f"- {len(ts.y)} fish in {len(ts.classes)} classes: {', '.join(f'{c} ({n})' for c, n in ts.classes.items())}",
        f"- Very small classes (exactly `min_class_size` fish): {', '.join(ts.very_small) or 'none'}",
        f"- Dropped classes: {dropped or 'none'}",
        f"- States whose features were dropped (shown by too few fish): {', '.join(ts.dropped_states) or 'none'}",
        f"- {len(ts.features)} features",
        f"- Folds: scheme A K = {folds.k[SCHEME_A]}" + (" (skipped)" if not folds.k[SCHEME_A] else "")
        + f", scheme B K = {folds.k[SCHEME_B]}; date-confounded classes (pinned to training in scheme A, not in its "
        f"score): {', '.join(folds.date_confounded) or 'none'}",
    ]
    share = audit.facts.get(f"{stage}: share of fish on dates with 2+ classes")
    if share is not None:
        lines.append(f"- Share of fish on dates holding 2+ classes (all the within-date permutation can use, D-016): {share:.0%}")
    lines += [f"- Fold note: {note}" for note in folds.notes]
    lines += ["", "### Balanced accuracy (mean ± spread across repeats) and scheme-A metrics", "", markdown_table(scores_table(ev))]
    lines += ["", "### Decision (PRD §6.6)", "", *(f"- **{m}**: {v}. {r}" for m, v, r in ev.decision.itertuples(index=False))]
    if stage == "compound":
        lines += ["", "### Vehicle against any drug (scheme A)", ""]
        lines.append(
            markdown_table(ev.vehicle)
            if not ev.vehicle.empty
            else "Not available: `training.vehicle_compound` is not among the kept classes, or scheme A was skipped."
        )
    model, scheme = best_model(ev)
    per_class = ev.per_class[(ev.per_class["model"] == model) & (ev.per_class["scheme"] == scheme)]
    lines += [
        "",
        f"### Per class: {model}, scheme {scheme} (pooled over repeats)",
        "",
        markdown_table(per_class[["label", "support", "precision", "recall", "f1"]].reset_index(drop=True)),
        "",
        f"![confusion matrix](confusion_{stage}.png)",
    ]
    return lines
