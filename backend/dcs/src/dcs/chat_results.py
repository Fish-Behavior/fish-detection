"""Chat tools that read a `dcs train` run folder: model scores, per-class scores, ablations, the audit (plan U19; D-071).

Same rules as `dcs.chat_tools`: deterministic, read-only, every number comes from the run's files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from dcs.audit import build_audit
from dcs.chat_core import ResearchData, ToolError, num, tool
from dcs.evaluate import date_effects
from dcs.folds import SCHEME_A, SCHEME_B
from dcs.gold_rules import compound_label


@tool(
    "Results of the latest training run for one stage ('compound' or 'dose <compound>'): every model's scores with "
    "the date held out (scheme A, the honest one), random split (B), labels shuffled within dates, the date effect, "
    "its verdict and why, and the saved model.",
    stage={"type": "string", "description": "default compound"},
)
def model_results(data: ResearchData, stage: str = "compound") -> dict[str, Any]:
    run = _run(data)
    metrics = _metrics(run, stage)
    path = run / "decisions.csv"  # absent in runs from before U19
    decisions = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=["stage", "model", "verdict", "reason"])
    decisions = decisions[decisions["stage"] == stage].set_index("model")
    predictions = pd.read_csv(run / "predictions.csv", low_memory=False)
    effect = date_effects(predictions[predictions["stage"] == stage])
    rows = []
    for model, part in metrics.groupby("model", sort=False):
        score = part.groupby("scheme")["balanced_accuracy"].agg(["mean", "std"]).fillna(0.0)
        row = {"model": model, **{f"scheme_{s}": _score(score, s) for s in (SCHEME_A, SCHEME_B, "B_permuted")}}
        if model in effect.index:
            row["date_effect_B_minus_A"] = float(effect[model])
        if model in decisions.index:
            row.update(verdict=decisions.loc[model, "verdict"], reason=decisions.loc[model, "reason"])
        rows.append(row)
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    return {
        "run_id": run.name,
        "stage": stage,
        "metric": "balanced accuracy (mean ± standard deviation across repeats); chance = 1 / number of classes",
        "models": rows,
        "saved_model": info.get("saved_model"),
        "source": info.get("gold_source"),
        "stages_in_run": info.get("stages"),
        "rule": "useful = scheme A beats majority and date_only by more than the spread and beats its own permuted score",
    }


@tool(
    "Per-class precision, recall and F1 of one model in one stage of the latest run (scheme A when it ran), and "
    "which classes it confuses most.",
    stage={"type": "string"},
    model={"type": "string", "description": "default: the saved model, else the best one"},
)
def class_scores(data: ResearchData, stage: str = "compound", model: str | None = None) -> dict[str, Any]:
    run = _run(data)
    predictions = pd.read_csv(run / "predictions.csv", low_memory=False)
    predictions = predictions[predictions["stage"] == stage]
    if predictions.empty:
        raise ToolError(f"stage {stage!r} is not in run {run.name}; stages: {sorted(_read_metrics(run)['stage'].unique())}")
    model = model or _chosen_model(run, stage)
    scheme = SCHEME_A if (predictions["scheme"] == SCHEME_A).any() else SCHEME_B
    rows = predictions[(predictions["model"] == model) & (predictions["scheme"] == scheme)]
    if rows.empty:
        raise ToolError(f"model {model!r} is not in stage {stage!r}; models: {sorted(predictions['model'].unique())}")
    labels = sorted(set(rows["label"]) | set(rows["predicted"]))
    p, r, f, n = precision_recall_fscore_support(rows["label"], rows["predicted"], labels=labels, zero_division=0)
    matrix = confusion_matrix(rows["label"], rows["predicted"], labels=labels)
    confusions = [{"true": labels[i], "predicted_as": labels[j], "count": int(matrix[i, j])}
                  for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i, j]]  # fmt: skip
    return {
        "stage": stage, "model": model, "scheme": scheme, "pooled_repeats": int(rows["repeat"].nunique()),
        "classes": [{"label": l, "precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)}
                    for l, a, b, c, d in zip(labels, p, r, f, n) if d],
        "top_confusions": sorted(confusions, key=lambda c: -c["count"])[:10],
    }  # fmt: skip


@tool("Ablations of the latest run: each model's scores with one change (NTT off, demographics on, depth on, vehicle-normalized) next to the main run.")
def ablation_results(data: ResearchData) -> dict[str, Any]:
    metrics = _read_metrics(_run(data))
    rows = []
    for (ablation, stage, model), part in metrics.groupby(["ablation", "stage", "model"], sort=False):
        score = part.groupby("scheme")["balanced_accuracy"].agg(["mean", "std"]).fillna(0.0)
        if ablation != "main" or stage == "compound":
            rows.append({"ablation": ablation, "stage": stage, "model": model, "scheme_A": _score(score, SCHEME_A), "scheme_B": _score(score, SCHEME_B)})
    note = "" if any(r["ablation"] != "main" for r in rows) else "this run has no ablations (it ran with --no-ablations)"
    return {"ablations": rows, "note": note}


@tool("Data audit facts: fish, dates, compounds, review status, frame rates, camera framing, the G1 go/wait rule, and the notes to keep in mind.")
def audit_facts(data: ResearchData) -> dict[str, Any]:
    audit = build_audit(data.table, data.described, data.training)
    return {"facts": audit.facts, "notes": list(audit.notes)}


def _run(data: ResearchData) -> Path:
    if data.run is None or not (data.run / "metrics.csv").is_file():
        raise ToolError("no training run found: run `python -m dcs train` first (or pass the run folder)")
    return data.run


def _read_metrics(run: Path) -> pd.DataFrame:
    """metrics.csv; a run from before the ablations (U14) has no `ablation` column: all of it is the main run."""
    metrics = pd.read_csv(run / "metrics.csv")
    return metrics if "ablation" in metrics.columns else metrics.assign(ablation="main")


def _metrics(run: Path, stage: str) -> pd.DataFrame:
    metrics = _read_metrics(run)
    main = metrics[(metrics["ablation"] == "main") & (metrics["stage"] == stage)]
    if main.empty:
        raise ToolError(f"stage {stage!r} is not in run {run.name}; stages: {', '.join(sorted(metrics['stage'].unique()))}")
    return main


def _chosen_model(run: Path, stage: str) -> str:
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8")).get("saved_model") or {}
    if info.get("stage") == stage:
        return info["model"]
    score = _metrics(run, stage).query("scheme == @SCHEME_A or scheme == @SCHEME_B").groupby("model")["balanced_accuracy"].mean()
    return str(score.idxmax())


def fish_predictions(data: ResearchData, video_id: str) -> dict[str, Any] | None:
    if data.run is None or not (data.run / "predictions.csv").is_file():
        return None
    model = _chosen_model(data.run, "compound")
    predictions = pd.read_csv(data.run / "predictions.csv", low_memory=False)
    rows = predictions[(predictions["stage"] == "compound") & (predictions["model"] == model) & (predictions["video_id"] == video_id)]
    scheme = SCHEME_A if (rows["scheme"] == SCHEME_A).any() else SCHEME_B
    rows = rows[rows["scheme"] == scheme]
    if rows.empty:
        return {"model": model, "repeats": 0, "note": "never scored out of fold (its class was pinned or not kept)"}
    true = f"p:{compound_label(rows['label'].iloc[0])}"
    return {"model": model, "scheme": scheme, "repeats": len(rows), "predicted_counts": rows["predicted"].value_counts().to_dict(),
            "mean_probability_of_true_class": num(rows[true].mean()) if true in rows else None}  # fmt: skip


def _score(score: pd.DataFrame, scheme: str) -> str | None:
    return None if scheme not in score.index else f"{score.loc[scheme, 'mean']:.3f} ± {score.loc[scheme, 'std']:.3f}"
