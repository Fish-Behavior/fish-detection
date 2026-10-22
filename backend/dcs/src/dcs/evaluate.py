"""Training set + folds -> out-of-fold predictions, metrics and verdicts (plan U11, T2.6; PRD §6.4-6.7; FR-4).

Every model is trained on identical folds (`Folds.table`). Inside each fold the preprocessing is fitted on the
training fish only (EC-9) and shared by all models of that fold. Pinned fish (scheme A fold -1) train in every fold
and are never scored. Metrics are computed per repeat on the pooled out-of-fold predictions (D-017); the spread is
their standard deviation across repeats. The within-date permutation re-runs scheme B with labels shuffled among
fish of the same date (PRD §6.6). The decision rule and the vehicle-vs-drug scores use scheme A.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

from dcs.folds import SCHEME_A, SCHEME_B, Folds, repeat_seed
from dcs.gold_rules import compound_label
from dcs.models import make_model
from dcs.preprocess import fit_preprocess
from dcs.trainset import TrainSet

METRICS = ("balanced_accuracy", "macro_f1", "log_loss", "top3_accuracy")  # PRD §6.7; never plain accuracy
PERMUTED = "B_permuted"  # scheme B re-run with labels shuffled within each date
REFERENCE = ("majority", "date_only")  # always run: the decision rule compares every model with them
USEFUL, NOT_USEFUL, UNDECIDED, BASELINE = "useful", "not useful", "undecided", "baseline"
TOP_K = 3
PERMUTATION_STREAM = 1  # keeps the permutation's random numbers apart from the folds' and models'


@dataclass(frozen=True)
class Evaluation:
    classes: tuple[str, ...]  # probability column order (`p:<class>`)
    predictions: pd.DataFrame  # model, scheme, repeat, fold, video_id, date, label, predicted, p:<class>...
    metrics: pd.DataFrame  # model, scheme, repeat, fish, METRICS: one row per repeat
    summary: pd.DataFrame  # index (model, scheme); columns (metric, mean|std) across repeats
    per_class: pd.DataFrame  # model, scheme, label, precision, recall, f1, support (pooled over repeats)
    decision: pd.DataFrame  # model, verdict, reason (PRD §6.6 rule 4)
    vehicle: pd.DataFrame  # model, auroc_mean/std, balanced_accuracy_mean/std (scheme A); empty when not applicable


def evaluate(
    ts: TrainSet,
    folds: Folds,
    models: Sequence[str],
    training: Mapping[str, Any],
    log: Callable[[str], None] = lambda message: None,
    device: str = "cpu",
) -> Evaluation:
    """Cross-validate `models` (majority and date-only are always added first) on `folds`; `device` is the
    resolved device for the MLP."""
    names = list(dict.fromkeys([*REFERENCE, *models]))
    classes = tuple(sorted(ts.classes))
    dates = ts.groups.to_numpy(dtype=object)
    frames = []
    for (scheme, repeat), block in folds.table.groupby(["scheme", "repeat"]):
        started = time.monotonic()
        fold = block.set_index("video_id")["fold"].loc[ts.ids].to_numpy()
        runs = [(scheme, ts.y.to_numpy(dtype=object))]
        if scheme == SCHEME_B:
            rng = np.random.default_rng([training["seed"], repeat, PERMUTATION_STREAM])
            runs.append((PERMUTED, permute_within_date(ts.y, ts.groups, rng).to_numpy(dtype=object)))
        for label_scheme, labels in runs:
            seed = repeat_seed(training["seed"], repeat)
            frames.append(_cross_validate(ts, fold, labels, dates, names, classes, seed, label_scheme, repeat, training, device))
        log(f"{ts.stage}: scheme {scheme} repeat {repeat + 1}/{training['repeats']} ({time.monotonic() - started:.1f} s)")

    predictions = pd.concat(frames, ignore_index=True)
    metrics = _metrics(predictions, classes)
    summary = metrics.groupby(["model", "scheme"], sort=False)[list(METRICS)].agg(["mean", "std"]).fillna(0.0)
    return Evaluation(
        classes=classes,
        predictions=predictions,
        metrics=metrics,
        summary=summary,
        per_class=_per_class(predictions),
        decision=_decide(summary, names),
        vehicle=_vehicle(predictions, ts.stage, classes, compound_label(training["vehicle_compound"])),
    )


def permute_within_date(y: pd.Series, dates: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Labels shuffled among the fish of each date; a date holding one class stays as it is (D-016)."""
    shuffled = y.copy()
    for rows in y.groupby(dates).indices.values():
        shuffled.iloc[rows] = y.iloc[rng.permutation(rows)].to_numpy()
    return shuffled


# --- parts -------------------------------------------------------------------------------------


def _cross_validate(
    ts: TrainSet,
    fold: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    names: list[str],
    classes: tuple[str, ...],
    seed: int,
    scheme: str,
    repeat: int,
    training: Mapping[str, Any],
    device: str,
) -> pd.DataFrame:
    """Out-of-fold probabilities of every model for one scheme and repeat; scored fish only."""
    column = {label: i for i, label in enumerate(classes)}
    proba = {name: np.zeros((len(fold), len(classes))) for name in names}
    for number in np.unique(fold[fold >= 0]):
        train, test = fold != number, fold == number  # pinned fish (fold -1) are always on the training side
        fitted = fit_preprocess(ts.X[train], ts.kinds)
        X_train, X_test = fitted.transform(ts.X[train]), fitted.transform(ts.X[test])
        for name in names:
            model = make_model(name, seed, training["mlp"], device).fit(X_train, labels[train], dates[train])
            # a class missing from this training fold gets probability 0
            proba[name][np.ix_(test, [column[c] for c in model.classes_])] = model.predict_proba(X_test, dates[test])
    scored = fold >= 0
    out = []
    for name in names:
        p = proba[name][scored]
        frame = pd.DataFrame(
            {
                "model": name,
                "scheme": scheme,
                "repeat": repeat,
                "fold": fold[scored],
                "video_id": ts.ids.to_numpy()[scored],
                "date": dates[scored],
                "label": labels[scored],
                "predicted": np.asarray(classes, dtype=object)[p.argmax(axis=1)],
            }
        )
        out.append(pd.concat([frame, pd.DataFrame(p, columns=[f"p:{c}" for c in classes])], axis=1))
    return pd.concat(out, ignore_index=True)


def _metrics(predictions: pd.DataFrame, classes: tuple[str, ...]) -> pd.DataFrame:
    """One row per model, scheme and repeat; class averages over the classes present among the scored fish."""
    rows = []
    for (model, scheme, repeat), group in predictions.groupby(["model", "scheme", "repeat"], sort=False):
        present = sorted(set(group["label"]))
        proba = group[[f"p:{c}" for c in classes]].to_numpy()
        truth, guess = group["label"], group["predicted"]
        rows.append(
            {
                "model": model,
                "scheme": scheme,
                "repeat": repeat,
                "fish": len(group),
                "balanced_accuracy": recall_score(truth, guess, labels=present, average="macro", zero_division=0),
                "macro_f1": f1_score(truth, guess, labels=present, average="macro", zero_division=0),
                "log_loss": log_loss(truth, proba, labels=list(classes)),
                "top3_accuracy": _top_k(truth.to_numpy(), proba, classes),
            }
        )
    return pd.DataFrame(rows)


def date_effects(predictions: pd.DataFrame) -> pd.Series:
    """Per model: scheme-B balanced accuracy minus scheme-A's, with B limited to the fish scheme A scores. A
    date-confounded class is pinned (never scored) in A but scored in B, so comparing the two summaries would mix
    different classes. Empty when scheme A did not run."""
    a = predictions[predictions["scheme"] == SCHEME_A]
    b = predictions[(predictions["scheme"] == SCHEME_B) & predictions["video_id"].isin(a["video_id"])]

    def score(part: pd.DataFrame) -> pd.Series:
        by_repeat = part.groupby(["model", "repeat"], sort=False).apply(
            lambda g: recall_score(g["label"], g["predicted"], labels=sorted(set(g["label"])), average="macro", zero_division=0),
            include_groups=False,
        )
        return by_repeat.groupby("model", sort=False).mean()

    return (score(b) - score(a)).dropna() if len(a) else pd.Series(dtype=float)


def _top_k(truth: np.ndarray, proba: np.ndarray, classes: tuple[str, ...]) -> float:
    column = {label: i for i, label in enumerate(classes)}
    top = np.argsort(-proba, axis=1, kind="stable")[:, : min(TOP_K, len(classes))]
    return float((top == np.array([column[t] for t in truth])[:, None]).any(axis=1).mean())


def _per_class(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, scheme), group in predictions.groupby(["model", "scheme"], sort=False):
        present = sorted(set(group["label"]))
        scores = precision_recall_fscore_support(group["label"], group["predicted"], labels=present, zero_division=0)
        for label, precision, recall, f1, support in zip(present, *scores):
            rows.append(
                {"model": model, "scheme": scheme, "label": label, "precision": precision, "recall": recall, "f1": f1,
                 "support": int(support)}
            )  # fmt: skip
    return pd.DataFrame(rows)


def _decide(summary: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """PRD §6.6 rule 4 on scheme-A balanced accuracy. "More than the spread" uses the larger spread of the two
    scores compared (D-055)."""

    def score(model: str, scheme: str) -> tuple[float, float]:
        row = summary.loc[(model, scheme), "balanced_accuracy"]
        return float(row["mean"]), float(row["std"])

    rows = []
    for name in names:
        if name in REFERENCE:
            rows.append((name, BASELINE, "reference for the decision rule"))
            continue
        if (name, SCHEME_A) not in summary.index:
            rows.append((name, UNDECIDED, "scheme A was skipped: no class is seen on two or more dates"))
            continue
        mean, spread = score(name, SCHEME_A)
        fails = []
        for reference in REFERENCE:
            ref_mean, ref_spread = score(reference, SCHEME_A)
            margin = max(spread, ref_spread)
            if mean <= ref_mean + margin:
                fails.append(f"{mean:.3f} is not above {reference} {ref_mean:.3f} by more than the spread {margin:.3f}")
        permuted = score(name, PERMUTED)[0]
        if mean <= permuted:
            fails.append(f"{mean:.3f} is not above its within-date permuted score {permuted:.3f}")
        reason = "; ".join(fails) or f"scheme A {mean:.3f} beats majority and date-only by more than the spread, and the permuted labels"
        rows.append((name, NOT_USEFUL if fails else USEFUL, reason))
    return pd.DataFrame(rows, columns=["model", "verdict", "reason"])


def _vehicle(predictions: pd.DataFrame, stage: str, classes: tuple[str, ...], vehicle: str) -> pd.DataFrame:
    """PRD §6.6 rule 5: vehicle vs any drug, read off the compound model's scheme-A probabilities."""
    columns = ["model", "auroc_mean", "auroc_std", "balanced_accuracy_mean", "balanced_accuracy_std"]
    if stage != "compound" or vehicle not in classes:
        return pd.DataFrame(columns=columns)
    rows = []
    scheme_a = predictions[predictions["scheme"] == SCHEME_A]
    for (model, repeat), group in scheme_a.groupby(["model", "repeat"], sort=False):
        drug = (group["label"] != vehicle).to_numpy()
        if drug.all() or not drug.any():
            continue
        rows.append(
            {
                "model": model,
                "auroc": roc_auc_score(drug, 1 - group[f"p:{vehicle}"]),
                "balanced_accuracy": balanced_accuracy_score(drug, group["predicted"] != vehicle),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(rows).groupby("model", sort=False).agg(["mean", "std"]).fillna(0.0)
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    return table.reset_index()[columns]
