"""Tune the labeling thresholds against the digitized reference timelines (random search).

Subjects used: those with feature bins AND a reference timeline (reference/timelines.csv).
For every candidate threshold set (`calibration.search`, `calibration.n_trials`, fixed
seed; candidate 0 = the current settings) each subject's bins are relabeled with the pure
labeling functions (no video work) and compared with its reference, second by second:

    score = macro-F1 over the five states
            + calibration.group_weight x group agreement

Group agreement = 1 - total-variation distance between our and the reference time budget
per group (reference/group_means.csv), averaged over groups. It forgives small timing
shifts of the digitized rows that the per-second F1 punishes.

K-fold by SUBJECT (`calibration.folds`) repeats the choice on the training folds and scores
the held-out fold, so an overfitted choice shows up as held-out F1 far below the in-sample F1.

Outputs (``<FISH_OUTPUT_DIR>/calibration/``, git-ignored):
    calibrated.yaml         only the tuned `labeling:` keys; `label` merges it over the config
    calibration_report.md   before/after F1, per-state F1, confusion matrix, held-out score
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fishbehavior.config import ConfigError, Settings, deep_merge
from fishbehavior.features import bins_path, features_dir
from fishbehavior.labeling import CALIBRATED_FILE, LABELS, MODEL_FILE, STATES, UNTRACKED, check_params, label_bins
from fishbehavior.labeling import labels_dir, read_bins
from fishbehavior.priors import GROUP_MEANS_FILE, TIMELINES_FILE, calibration_dir, markdown_table
from fishbehavior.reference import reference_dir

REPORT_FILE = "calibration_report.md"
BOUT_KEY = "min_bout_s"  # one searched value, applied to every state except untracked


# ---------------------------------------------------------------------------
# 1. Candidates
# ---------------------------------------------------------------------------


def sample_candidates(search: dict[str, Any], n: int, seed: int) -> list[dict[str, Any]]:
    """n candidates: {} (the current settings) first, then random draws.

    A list in `search` = pick one of its values; {low, high} = uniform in that range.
    """
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = [{}]
    for _ in range(max(0, n - 1)):
        draw = {}
        for key, space in search.items():
            if isinstance(space, dict):
                draw[key] = round(float(rng.uniform(float(space["low"]), float(space["high"]))), 4)
            else:
                draw[key] = space[int(rng.integers(len(space)))]  # keeps the list's own type (int / float)
        candidates.append(draw)
    return candidates


def check_search(search: dict[str, Any], base: dict[str, Any]) -> None:
    """Refuse unknown keys or malformed ranges before a long search."""
    if not search:
        raise ConfigError("calibration.search is empty: list the labeling keys to tune")
    for key, space in search.items():
        if key not in base:
            raise ConfigError(f"calibration.search.{key} is not a labeling setting")
        ok = (isinstance(space, dict) and {"low", "high"} <= set(space) and float(space["low"]) <= float(space["high"])) \
            or (isinstance(space, list) and len(space) > 0)
        if not ok:
            raise ConfigError(f"calibration.search.{key} must be a list of values or {{low: a, high: b}} with a <= b")


def candidate_params(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """The labeling settings with one candidate applied (min_bout_s: same value for every state)."""
    override = dict(candidate)
    if BOUT_KEY in override and not isinstance(override[BOUT_KEY], dict):
        override[BOUT_KEY] = {state: override[BOUT_KEY] for state in STATES}  # untracked keeps its own value
    return deep_merge(base, override)


# ---------------------------------------------------------------------------
# 2. Scores (pure)
# ---------------------------------------------------------------------------


def per_second(labeled: pd.DataFrame, bin_s: float, n_seconds: int) -> np.ndarray:
    """Our label at the middle of every whole second 0..n_seconds-1 ('' after the video ends)."""
    index = np.floor((np.arange(n_seconds) + 0.5) / bin_s).astype(int)
    labels = labeled["label"].to_numpy(object)
    return np.where(index < len(labels), labels[np.minimum(index, len(labels) - 1)], "")


def confusion(reference: np.ndarray, ours: np.ndarray) -> np.ndarray:
    """Seconds per (reference state, our label): rows STATES, columns LABELS (5 states + untracked).

    Seconds where the reference is no_data / unknown, or our video has ended, are left out.
    """
    rows, cols = {s: i for i, s in enumerate(STATES)}, {s: i for i, s in enumerate(LABELS)}
    r = np.array([rows.get(x, -1) for x in reference])
    c = np.array([cols.get(x, -1) for x in ours])
    keep = (r >= 0) & (c >= 0)
    out = np.zeros((len(STATES), len(LABELS)))
    np.add.at(out, (r[keep], c[keep]), 1)
    return out


def f1_scores(matrix: np.ndarray) -> np.ndarray:
    """F1 per state from a confusion matrix; NaN for a state neither side ever shows."""
    n = len(STATES)
    tp = np.diag(matrix[:, :n])
    fp, fn = matrix[:, :n].sum(axis=0) - tp, matrix.sum(axis=1) - tp
    denominator = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, 2 * tp / denominator, np.nan)


def macro_f1(matrix: np.ndarray) -> float:
    """Mean F1 over the states that occur (in the reference or in our labels)."""
    scores = f1_scores(matrix)
    return float(np.nanmean(scores)) if np.isfinite(scores).any() else 0.0


def group_agreement(seconds: np.ndarray, groups: np.ndarray, ref_means: pd.DataFrame) -> float:
    """1 - total-variation distance between our and the reference time budget, mean over groups.

    `seconds`: (subjects, LABELS) our seconds per label; `groups`: each subject's group.
    Our untracked time counts as disagreement (the reference has no such state).
    """
    ref = ref_means.set_index("group")
    values = []
    for group in np.unique(groups):
        if group not in ref.index:
            continue
        ours = seconds[groups == group].mean(axis=0)
        theirs = np.r_[ref.loc[group, list(STATES)].to_numpy(float), 0.0]  # no untracked in the reference
        if ours.sum() <= 0 or theirs.sum() <= 0:
            continue
        values.append(1.0 - 0.5 * np.abs(ours / ours.sum() - theirs / theirs.sum()).sum())
    return float(np.mean(values)) if values else 0.0


def fold_split(n_subjects: int, folds: int, seed: int) -> list[np.ndarray]:
    """Subject indices of each test fold: shuffled once, split into near-equal folds (one subject never in two)."""
    order = np.random.default_rng(seed).permutation(n_subjects)
    return [np.sort(f) for f in np.array_split(order, max(2, min(folds, n_subjects)))]


# ---------------------------------------------------------------------------
# 3. The search
# ---------------------------------------------------------------------------


@dataclass
class Evaluated:
    """Per candidate and subject: confusion matrix and our seconds per label."""

    confusion: np.ndarray  # (candidates, subjects, STATES, LABELS)
    seconds: np.ndarray  # (candidates, subjects, LABELS)


def evaluate(candidates: list[dict[str, Any]], base: dict[str, Any], bins: list[pd.DataFrame],
             references: list[np.ndarray], model: dict[str, Any], bin_s: float) -> Evaluated:
    """Relabel every subject with every candidate."""
    conf = np.zeros((len(candidates), len(bins), len(STATES), len(LABELS)))
    seconds = np.zeros((len(candidates), len(bins), len(LABELS)))
    for c, candidate in enumerate(candidates):
        params = candidate_params(base, candidate)
        for s, (table, reference) in enumerate(zip(bins, references)):
            labeled = label_bins(table, params, model, bin_s)
            conf[c, s] = confusion(reference, per_second(labeled, bin_s, len(reference)))
            counts = labeled["label"].value_counts()
            seconds[c, s] = [counts.get(name, 0) * bin_s for name in LABELS]
    return Evaluated(conf, seconds)


def scores(ev: Evaluated, subjects: np.ndarray, groups: np.ndarray, ref_means: pd.DataFrame,
           weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(score, macro-F1, group agreement) of every candidate on these subject indices."""
    f1 = np.array([macro_f1(ev.confusion[c, subjects].sum(axis=0)) for c in range(len(ev.confusion))])
    agree = np.array([group_agreement(ev.seconds[c, subjects], groups[subjects], ref_means)
                      for c in range(len(ev.confusion))])
    return f1 + weight * agree, f1, agree


def held_out(ev: Evaluated, groups: np.ndarray, ref_means: pd.DataFrame, weight: float,
             folds: list[np.ndarray]) -> pd.DataFrame:
    """Per fold: pick the best candidate on the other folds, report F1 on this fold (tuned and current)."""
    rows, everyone = [], np.arange(ev.confusion.shape[1])
    for k, test in enumerate(folds):
        train = np.setdiff1d(everyone, test)
        train_score, train_f1, _ = scores(ev, train, groups, ref_means, weight)
        best = int(np.argmax(train_score))
        _, test_f1, _ = scores(ev, test, groups, ref_means, weight)
        rows.append({"fold": k + 1, "test_subjects": len(test), "chosen_candidate": best,
                     "train_f1": train_f1[best],
                     "test_f1_tuned": test_f1[best], "test_f1_current": test_f1[0]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Whole run (called by the CLI)
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """What `run_calibration` did, for the CLI summary and the report."""

    subjects: list[str]
    tuned: dict[str, Any]  # the winning candidate (written to calibrated.yaml)
    before: dict[str, float]  # score / macro_f1 / agreement of the current settings
    after: dict[str, float]
    state_f1: pd.DataFrame  # state, before, after
    confusion_after: np.ndarray
    folds: pd.DataFrame
    min_subjects: int
    calibrated_path: Path
    report_path: Path


def load_inputs(settings: Settings) -> tuple[list[str], list[pd.DataFrame], list[np.ndarray], np.ndarray, pd.DataFrame]:
    """Subjects with bins and a reference timeline: ids, bins, reference labels per second, groups; + group means."""
    ref_dir = reference_dir(settings)
    timelines_path, means_path = ref_dir / TIMELINES_FILE, ref_dir / GROUP_MEANS_FILE
    if not timelines_path.is_file():
        raise ConfigError(f"{timelines_path} not found: run `python -m fishbehavior reference` first")
    timelines = pd.read_csv(timelines_path, dtype={"subject_id": str}).sort_values(["subject_id", "second"])
    ids, bins, references, groups = [], [], [], []
    for subject_id, rows in timelines.groupby("subject_id", sort=True):
        if not bins_path(features_dir(settings), subject_id).is_file():
            continue  # a placeholder row or a subject without video
        ids.append(subject_id)
        bins.append(read_bins(features_dir(settings), subject_id))
        references.append(rows["label"].to_numpy(object))
        groups.append(rows["group"].iloc[0])
    if not ids:
        raise ConfigError("no subject has both feature bins and a reference timeline; "
                          "check the subject_ids in reference/mapping.yaml")
    return ids, bins, references, np.array(groups, dtype=object), pd.read_csv(means_path)


def run_calibration(settings: Settings) -> CalibrationResult:
    """Search, cross-validate, and write calibrated.yaml + calibration_report.md."""
    params = settings.params["calibration"]
    search = {k: v for k, v in (params["search"] or {}).items() if v is not None}  # null = not tuned
    base = settings.params["labeling"]  # the configured values, NOT an earlier calibrated.yaml: "before" = defaults
    check_search(search, base)
    model_path = labels_dir(settings) / MODEL_FILE
    if not model_path.is_file():
        raise ConfigError(f"{model_path} not found: run `python -m fishbehavior label` first")
    # ponytail: the swim model is reused as fitted by `label`; the GMM posterior of a bin does not
    # depend on which other bins are swimming, so only the next `label` run refits it once.
    model = json.loads(model_path.read_text(encoding="utf-8"))
    ids, bins, references, groups, ref_means = load_inputs(settings)
    bin_s = float(settings.params["features"]["bin_s"])
    weight, seed = float(params["group_weight"]), int(params["seed"])

    candidates = sample_candidates(search, int(params["n_trials"]), seed)
    for candidate in candidates:
        check_params(candidate_params(base, candidate))
    ev = evaluate(candidates, base, bins, references, model, bin_s)
    everyone = np.arange(len(ids))
    score, f1, agree = scores(ev, everyone, groups, ref_means, weight)
    best = int(np.argmax(score))
    folds = held_out(ev, groups, ref_means, weight, fold_split(len(ids), int(params["folds"]), seed))

    tuned = {key: candidates[best].get(key, base[key]) for key in search}
    if BOUT_KEY in tuned:  # labeling needs one value per state; untracked keeps its configured one
        tuned[BOUT_KEY] = {k: v for k, v in candidate_params(base, {BOUT_KEY: tuned[BOUT_KEY]})[BOUT_KEY].items()
                           if k != UNTRACKED}
    out = calibration_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    result = CalibrationResult(
        subjects=ids, tuned=tuned,
        before={"score": score[0], "macro_f1": f1[0], "agreement": agree[0]},
        after={"score": score[best], "macro_f1": f1[best], "agreement": agree[best]},
        state_f1=pd.DataFrame({"state": STATES, "before": f1_scores(ev.confusion[0].sum(axis=0)),
                               "after": f1_scores(ev.confusion[best].sum(axis=0))}),
        confusion_after=ev.confusion[best].sum(axis=0), folds=folds, min_subjects=int(params["min_subjects"]),
        calibrated_path=settings.paths.output_dir / CALIBRATED_FILE, report_path=out / REPORT_FILE,
    )
    header = (f"# Written by `calibrate` on {dt.date.today()} from {len(ids)} subjects (see {REPORT_FILE}).\n"
              f"# Local only (git-ignored). `label` merges these over the labeling settings; delete to go back.\n")
    result.calibrated_path.write_text(header + yaml.safe_dump({"labeling": tuned}, sort_keys=False), encoding="utf-8")
    result.report_path.write_text(report_text(result, base, weight), encoding="utf-8")
    return result


def report_text(result: CalibrationResult, base: dict[str, Any], weight: float) -> str:
    """calibration_report.md."""
    n = len(result.subjects)
    parts = ["# Calibration report", ""]
    if n < result.min_subjects:
        parts += [f"> **Warning:** only {n} subjects (fewer than calibration.min_subjects = {result.min_subjects}). "
                  "Do not trust these values yet.", ""]
    parts += [f"Subjects used (bins + reference timeline): **{n}**. Score = macro-F1 + {weight:g} x group agreement.",
              "The reference is digitized from figures (approximate), and per-second F1 also counts small",
              "timing offsets as errors, so read the numbers as relative, not absolute.", "",
              "## Before / after", "",
              markdown_table(pd.DataFrame([{"settings": "current", **result.before},
                                           {"settings": "calibrated", **result.after}]), 3), "",
              "## Tuned values", "",
              markdown_table(pd.DataFrame([{"key": k, "current": str(base[k]), "calibrated": str(v)}
                                           for k, v in result.tuned.items()])), "",
              "## F1 per state", "", markdown_table(result.state_f1, 3), "",
              "## Confusion matrix, calibrated (seconds; rows = reference, columns = ours)", "",
              markdown_table(pd.DataFrame(result.confusion_after, columns=list(LABELS)).assign(
                  reference=list(STATES))[["reference", *LABELS]], 0), "",
              "## Held-out check (K-fold by subject)", "",
              "Each fold's values are chosen on the other folds only. Held-out F1 far below the in-sample",
              "F1 means the search fits noise; held-out tuned <= held-out current means calibration does not help.", "",
              markdown_table(result.folds, 3), "",
              f"Mean held-out F1: tuned **{result.folds['test_f1_tuned'].mean():.3f}**, "
              f"current {result.folds['test_f1_current'].mean():.3f} "
              f"(in-sample calibrated {result.after['macro_f1']:.3f})."]
    return "\n".join(parts) + "\n"
