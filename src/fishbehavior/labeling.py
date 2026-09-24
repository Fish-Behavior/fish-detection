"""Behavior state of every time bin, and the bins merged into segments.

Input: the per-bin features of each subject (features.py) and its track timeline
(tracking.py: which part file and frame each moment is). Each bin gets one of the five
ethogram states, or the QA label ``untracked`` (not a behavior: the fish was not seen).

Rules, in priority order (the first that matches wins; thresholds in `labeling:`):

0. untracked       tracked_fraction < min_tracked_fraction
1. surface_breach  nose_up_surface_fraction >= surface_breach_fraction (seen side-on, head at the waterline, nose up)
2. lorr            tilt_fraction >= lorr_tilt_fraction and speed_median < lorr_max_speed_bl_s,
                   for at least lorr_min_s seconds in a row
3. freeze_drift    speed_median < freeze_speed_bl_s, for at least freeze_min_s seconds in a row
4. every other bin is swimming. Slow swimming (mean speed < swim_min_speed_bl_s: too slow
   for turning to be measured) is controlled_swim. Faster bins are split into
   controlled_swim / erratic by a 2-component Gaussian mixture (or a 2-state Gaussian HMM,
   `swim_split: hmm`) fitted on these bins of ALL subjects together, so "erratic" means
   the same for every fish.

Then bouts shorter than `min_bout_s[state]` are merged into their longer neighbour. Last,
human relabels from ``labels/overrides.csv`` (written by the live page, or by hand) replace the
automatic label of every bin they cover. Equal consecutive labels become segments (split where
one video part ends).

The label functions are pure (bins table in, labels out), so calibration can relabel
quickly. A `<FISH_OUTPUT_DIR>/calibration/calibrated.yaml` with a `labeling:` section is
merged over the config (written by the calibration step).

Outputs (in ``<FISH_OUTPUT_DIR>/labels/``, git-ignored):
    swim_model.json            the pooled swim model and the settings it was fitted with
    <subject_id>_bins.csv      the feature bins plus `label`, `confidence`, `auto_label`, `label_source`
    <subject_id>_segments.csv  that subject's segments (SEGMENT_COLUMNS)
    segments.csv               every labeled subject's segments
    summary.csv                seconds and % of the video per label, per subject
    overrides.csv              INPUT, kept across runs: human relabels (OVERRIDE_COLUMNS)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from fishbehavior.catalog import STATUS_MATCHED
from fishbehavior.config import ConfigError, Settings, deep_merge
from fishbehavior.features import bins_path, frame_gap_s
from fishbehavior.features import features_dir as get_features_dir
from fishbehavior.parallel import run_parallel
from fishbehavior.tracking import track_path
from fishbehavior.tracking import tracks_dir as get_tracks_dir

log = logging.getLogger(__name__)

UNTRACKED = "untracked"
SURFACE, LORR, FREEZE, CALM, ERRATIC = "surface_breach", "lorr", "freeze_drift", "controlled_swim", "erratic"
STATES = (CALM, ERRATIC, FREEZE, LORR, SURFACE)  # the five ethogram states
LABELS = (*STATES, UNTRACKED)
SWIM = "swim"  # placeholder for bins no rule claimed, before the swim split
SWIM_SPLITS = ("gmm", "hmm")
# Swim-split inputs. Every one grows with erratic swimming; the last three are heavy-tailed
# (turn variance reaches 1e7), so they are log1p-compressed before standardizing.
SWIM_FEATURES = ("speed_cv", "jerk_abs_mean_bl_s3", "turn_rate_var_deg2_s2", "meander_deg_per_bl")
LOG_FEATURES = ("jerk_abs_mean_bl_s3", "turn_rate_var_deg2_s2", "meander_deg_per_bl")
SEGMENT_COLUMNS = [
    "subject_id", "label", "start_s", "end_s", "duration_s", "start_frame", "end_frame", "part", "mean_confidence",
]
MODEL_FILE, SEGMENTS_FILE, SUMMARY_FILE = "swim_model.json", "segments.csv", "summary.csv"
# Human relabels: seconds [start_s, end_s) on the joined timeline of one subject (or live video name).
OVERRIDES_FILE, OVERRIDE_COLUMNS = "overrides.csv", ["subject_id", "start_s", "end_s", "label"]
HUMAN, AUTO = "human", "auto"  # label_source values
CALIBRATED_FILE = Path("calibration") / "calibrated.yaml"
SEED = 0  # fixed, so the same data always gives the same swim model


# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------


def labeling_params(settings: Settings) -> tuple[dict[str, Any], Path | None]:
    """The `labeling:` settings with calibrated.yaml merged over them, and that file if used."""
    params = settings.params["labeling"]
    path = settings.paths.output_dir / CALIBRATED_FILE
    calibrated = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("labeling") if path.is_file() else None
    if calibrated:
        params = deep_merge(params, calibrated)
    check_params(params)
    return params, path if calibrated else None


def check_params(params: dict[str, Any]) -> None:
    """Refuse settings that would silently do the wrong thing (unknown split or state names)."""
    if params["swim_split"] not in SWIM_SPLITS:
        raise ConfigError(f"labeling.swim_split must be one of {', '.join(SWIM_SPLITS)}, got {params['swim_split']!r}")
    unknown = set(params["min_bout_s"]) - set(LABELS)
    if unknown:
        raise ConfigError(f"labeling.min_bout_s has unknown states {sorted(unknown)}; use {', '.join(LABELS)}")


# ---------------------------------------------------------------------------
# 2. Rules (pure functions on the bins table)
# ---------------------------------------------------------------------------


def run_bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and stop (exclusive) index of each run of equal consecutive values."""
    if len(values) == 0:
        return np.array([], int), np.array([], int)
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    return np.r_[0, change], np.r_[change, len(values)]


def sustained(mask: np.ndarray, min_bins: int) -> np.ndarray:
    """Keep only the runs of True that are at least `min_bins` long."""
    out = np.zeros(len(mask), bool)
    for start, stop in zip(*run_bounds(mask)):
        if mask[start] and stop - start >= min_bins:
            out[start:stop] = True
    return out


def n_bins(seconds: float, bin_s: float) -> int:
    """Bins needed to cover `seconds` (at least 1)."""
    return max(1, int(np.ceil(seconds / bin_s - 1e-9)))


def rule_labels(bins: pd.DataFrame, params: dict[str, Any], bin_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Labels from rules 0-3 (other bins: SWIM) and confidence = how strongly the rule held."""
    label = np.full(len(bins), SWIM, dtype=object)
    confidence = np.full(len(bins), np.nan)
    free = np.ones(len(bins), bool)  # not claimed by a higher-priority rule yet

    def claim(mask: np.ndarray, name: str, strength: np.ndarray) -> None:
        """Give the still-free bins of `mask` this label; the rule's strength is the confidence."""
        mask = mask & free
        label[mask], confidence[mask] = name, np.clip(strength[mask], 0.0, 1.0)
        free[mask] = False

    tracked = bins["tracked_fraction"].fillna(0.0).to_numpy(float)
    surface = bins["nose_up_surface_fraction"].fillna(0.0).to_numpy(float)
    tilt = bins["tilt_fraction"].fillna(0.0).to_numpy(float)
    speed = bins["speed_median_bl_s"].to_numpy(float)  # NaN compares False: never "slow"
    freeze_speed = float(params["freeze_speed_bl_s"])

    claim(tracked < float(params["min_tracked_fraction"]), UNTRACKED, 1.0 - tracked)  # share of frames not seen
    claim(surface >= float(params["surface_breach_fraction"]), SURFACE, surface)  # share of frames nose-up there
    # Sustained rules: the run length is counted over bins no earlier rule took.
    lorr = (tilt >= float(params["lorr_tilt_fraction"])) & (speed < float(params["lorr_max_speed_bl_s"]))
    claim(sustained(lorr & free, n_bins(float(params["lorr_min_s"]), bin_s)), LORR, tilt)  # share of frames tilted
    still = speed < freeze_speed
    # Bins only keep the median speed, so strength = how far below the threshold it is (1 = not moving).
    claim(sustained(still & free, n_bins(float(params["freeze_min_s"]), bin_s)), FREEZE, 1.0 - speed / freeze_speed)
    return label, confidence


# ---------------------------------------------------------------------------
# 3. Swim split: pooled 2-component model, stored as plain JSON
# ---------------------------------------------------------------------------


def moving(bins: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    """Bins fast enough for the swim model: slower ones have no turning measure, and would
    form their own "hovering" cluster instead of the smooth-vs-erratic split."""
    return bins["speed_mean_bl_s"].to_numpy(float) >= float(params["swim_min_speed_bl_s"])  # NaN -> False


def swim_matrix(bins: pd.DataFrame) -> np.ndarray:
    """Raw swim features (SWIM_FEATURES order), heavy-tailed ones log1p-compressed; NaN kept."""
    columns = [np.log1p(bins[f].clip(lower=0)) if f in LOG_FEATURES else bins[f] for f in SWIM_FEATURES]
    return np.column_stack([c.to_numpy(float) for c in columns])


def standardize(raw: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Fill missing values with the pooled median, then z-score with the pooled mean / SD."""
    filled = np.where(np.isfinite(raw), raw, np.asarray(model["median"]))
    return (filled - np.asarray(model["mean"])) / np.asarray(model["scale"])


def swim_lengths(swim: np.ndarray) -> list[int]:
    """Lengths of the consecutive swim stretches: separate sequences for the HMM."""
    return [int(stop - start) for start, stop in zip(*run_bounds(swim)) if swim[start]]


def fit_swim_model(tables: list[tuple[pd.DataFrame, np.ndarray]], params: dict[str, Any]) -> dict[str, Any]:
    """Fit the swim split on the swim bins of every (bins, swim mask) pair, pooled."""
    raw = np.vstack([swim_matrix(bins[swim]) for bins, swim in tables])
    if len(raw) < 10:
        raise RuntimeError(f"only {len(raw)} swim bins in all subjects; too few to fit the swim model")
    median = np.nanmedian(raw, axis=0)
    filled = np.where(np.isfinite(raw), raw, median)
    scale = filled.std(axis=0)
    model: dict[str, Any] = {
        "method": params["swim_split"], "features": list(SWIM_FEATURES), "log1p": list(LOG_FEATURES),
        "median": median.tolist(), "mean": filled.mean(axis=0).tolist(),
        "scale": np.where(scale > 0, scale, 1.0).tolist(),  # a constant feature must not divide by 0
        "n_bins": len(raw), "n_subjects": len(tables),
    }
    x = standardize(raw, model)
    if params["swim_split"] == "hmm":
        from hmmlearn.hmm import GaussianHMM  # only needed for this option

        lengths = [n for _, swim in tables for n in swim_lengths(swim)]
        hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=SEED).fit(x, lengths)
        model.update(startprob=hmm.startprob_.tolist(), transmat=hmm.transmat_.tolist(),
                     means=hmm.means_.tolist(), covariances=hmm.covars_.tolist())
    else:
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=SEED).fit(x)
        model.update(weights=gmm.weights_.tolist(), means=gmm.means_.tolist(), covariances=gmm.covariances_.tolist())
    # Erratic score of a component = sum of its standardized means (all features rise with erratic swimming).
    model["erratic_component"] = int(np.argmax(np.asarray(model["means"]).sum(axis=1)))
    return model


def erratic_probability(bins: pd.DataFrame, swim: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """P(erratic) for each swim bin (in order), from the saved model."""
    x = standardize(swim_matrix(bins[swim]), model)
    if len(x) == 0:
        return np.array([])
    if model["method"] == "hmm":
        from hmmlearn.hmm import GaussianHMM

        hmm = GaussianHMM(n_components=2, covariance_type="full")
        hmm.n_features = x.shape[1]
        hmm.startprob_, hmm.transmat_ = np.asarray(model["startprob"]), np.asarray(model["transmat"])
        hmm.means_, hmm.covars_ = np.asarray(model["means"]), np.asarray(model["covariances"])
        posterior = hmm.predict_proba(x, swim_lengths(swim))  # smoothed over each swim stretch
    else:
        log_p = np.column_stack([
            np.log(w) + multivariate_normal.logpdf(x, mean, cov, allow_singular=True)
            for w, mean, cov in zip(model["weights"], model["means"], model["covariances"])
        ])
        posterior = np.exp(log_p - logsumexp(log_p, axis=1, keepdims=True))
    return posterior[:, model["erratic_component"]]


# ---------------------------------------------------------------------------
# 4. Cleanup, whole labeling, segments, summary (pure)
# ---------------------------------------------------------------------------


def clean_bouts(label: np.ndarray, confidence: np.ndarray, bin_s: float,
                min_bout_s: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Merge bouts shorter than their state's minimum into the longer neighbour, shortest first.

    Merged bins get confidence 0: no measurement supports their new label directly.
    """
    label, confidence = label.copy(), confidence.copy()
    while True:
        starts, stops = run_bounds(label)
        seconds = (stops - starts) * bin_s
        short = [i for i in range(len(starts))
                 if seconds[i] < float(min_bout_s.get(label[starts[i]], 0)) - 1e-9 and len(starts) > 1]
        if not short:
            return label, confidence
        i = min(short, key=lambda k: seconds[k])
        # Longer neighbour wins; on a tie (or at the end) the previous one.
        before = seconds[i - 1] if i > 0 else -1.0
        after = seconds[i + 1] if i + 1 < len(starts) else -1.0
        into = starts[i - 1] if before >= after else starts[i + 1]
        label[starts[i]:stops[i]], confidence[starts[i]:stops[i]] = label[into], 0.0


def label_bins(bins: pd.DataFrame, params: dict[str, Any], model: dict[str, Any] | None,
               bin_s: float) -> pd.DataFrame:
    """The bins table plus `label` and `confidence` (rules, swim split, bout cleanup)."""
    label, confidence = rule_labels(bins, params, bin_s)
    slow = (label == SWIM) & ~moving(bins, params)
    # Slow, not frozen swimming is controlled; strength = how far below the speed gate (1 = still).
    speed = bins["speed_mean_bl_s"].fillna(0.0).to_numpy(float)
    label[slow], confidence[slow] = CALM, np.clip(1.0 - speed[slow] / float(params["swim_min_speed_bl_s"]), 0, 1)
    swim = label == SWIM
    if swim.any():
        if model is None:
            raise RuntimeError("swim bins but no swim model")
        p_erratic = erratic_probability(bins, swim, model)
        erratic = p_erratic >= 0.5
        label[swim] = np.where(erratic, ERRATIC, CALM)
        confidence[swim] = np.where(erratic, p_erratic, 1.0 - p_erratic)  # posterior of the chosen state
    label, confidence = clean_bouts(label, confidence, bin_s, params["min_bout_s"])
    return bins.assign(label=label, confidence=np.round(confidence, 4))


def check_label_overrides(table: pd.DataFrame, source: str) -> pd.DataFrame:
    """Human relabels with numeric times, a known label and start < end; ConfigError naming `source`."""
    if set(OVERRIDE_COLUMNS) - set(table.columns):
        raise ConfigError(f"{source}: needs the columns {', '.join(OVERRIDE_COLUMNS)}")
    table = table.assign(subject_id=table["subject_id"].astype(str),  # "0042" stays text
                         start_s=pd.to_numeric(table["start_s"], errors="coerce"),
                         end_s=pd.to_numeric(table["end_s"], errors="coerce"))[OVERRIDE_COLUMNS]
    bad = ~table["label"].isin(LABELS) | ~(table["start_s"] < table["end_s"])  # NaN times fail the `<`
    if bad.any():
        row = table[bad].iloc[0].to_dict()
        raise ConfigError(f"{source}: bad relabel {row}: label must be one of {', '.join(LABELS)} "
                          f"and start_s < end_s (seconds)")
    return table.reset_index(drop=True)


def load_label_overrides(labels_dir: Path) -> pd.DataFrame:
    """Every human relabel in `labels_dir/overrides.csv`, in file order; empty if there is none."""
    path = labels_dir / OVERRIDES_FILE
    if not path.is_file():
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)
    return check_label_overrides(pd.read_csv(path, dtype={"subject_id": str}), str(path))


def apply_label_overrides(labeled: pd.DataFrame, overrides: pd.DataFrame, bin_s: float) -> pd.DataFrame:
    """One subject's human relabels over its automatic labels.

    A bin whose middle lies in [start_s, end_s) takes the relabel with confidence 1 (a person
    decided it); later rows win where ranges overlap. `auto_label` keeps the rules/model label,
    `label_source` says which one `label` is. Applied after the bout cleanup, so a human bout
    is never merged away.
    """
    label = labeled["label"].to_numpy(object).copy()
    confidence = labeled["confidence"].to_numpy(float).copy()
    human = np.zeros(len(labeled), bool)
    middle = labeled["t_start_s"].to_numpy(float) + bin_s / 2
    for row in overrides.itertuples(index=False):
        inside = (middle >= row.start_s) & (middle < row.end_s)
        label[inside], confidence[inside], human[inside] = row.label, 1.0, True
    return labeled.assign(label=label, confidence=confidence, auto_label=labeled["label"],
                          label_source=np.where(human, HUMAN, AUTO))


def make_segments(labeled: pd.DataFrame, timeline: pd.DataFrame, bin_s: float) -> pd.DataFrame:
    """Runs of equal labels on the frame timeline (part, part_frame, time_s), split at part changes.

    A segment runs from its first frame's time to the next segment's first frame (the last
    one to the end of the video), so the segments tile the whole video.
    """
    timeline = timeline.sort_values("time_s").reset_index(drop=True)
    time_s = timeline["time_s"].to_numpy(float)
    index = np.clip(np.floor(time_s / bin_s + 1e-9).astype(int), 0, len(labeled) - 1)  # same bins as features
    label = labeled["label"].to_numpy(object)[index]
    confidence = labeled["confidence"].to_numpy(float)[index]
    part = timeline["part"].to_numpy(int)
    key = np.array([f"{p}|{name}" for p, name in zip(part, label)], dtype=object)
    starts, stops = run_bounds(key)
    gap = frame_gap_s(time_s)
    end_of_video = time_s[-1] + (gap if np.isfinite(gap) else bin_s)
    ends = np.r_[time_s[starts[1:]], end_of_video]
    part_frame = timeline["part_frame"].to_numpy(int)
    segments = pd.DataFrame({
        "subject_id": str(labeled["subject_id"].iloc[0]),
        "label": label[starts],
        "start_s": time_s[starts].round(4),
        "end_s": ends.round(4),
        "duration_s": (ends - time_s[starts]).round(4),
        "start_frame": part_frame[starts],  # frame numbers inside that part's video file
        "end_frame": part_frame[stops - 1],  # inclusive
        "part": part[starts],
        "mean_confidence": [round(float(np.nanmean(confidence[a:b])), 4) if np.isfinite(confidence[a:b]).any()
                            else np.nan for a, b in zip(starts, stops)],
    })
    return segments[SEGMENT_COLUMNS]


def summarize(segments: pd.DataFrame) -> pd.DataFrame:
    """Per subject: total seconds, and seconds + % of the video for every label."""
    seconds = segments.pivot_table(index="subject_id", columns="label", values="duration_s", aggfunc="sum",
                                   fill_value=0.0).reindex(columns=list(LABELS), fill_value=0.0)
    total = seconds.sum(axis=1)
    out = pd.DataFrame({"duration_s": total.round(2)})
    for name in LABELS:
        out[f"{name}_s"] = seconds[name].round(2)
        out[f"{name}_pct"] = (100 * seconds[name] / total).round(2)
    return out.reset_index()


# ---------------------------------------------------------------------------
# 5. One subject (runs in a worker process) with caching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelJob:
    """Everything a worker needs for one subject (plain values, so it can be sent to a process)."""

    subject_id: str
    features_dir: Path
    tracks_dir: Path
    labels_dir: Path
    params: dict[str, Any]  # labeling section (calibrated values merged in)
    bin_s: float  # features.bin_s
    model: dict[str, Any]
    force: bool
    overrides: pd.DataFrame  # this subject's human relabels (rows of overrides.csv)


def labeled_bins_path(labels_dir: Path, subject_id: str) -> Path:
    """Per-bin labels of one subject."""
    return labels_dir / f"{subject_id}_bins.csv"


def segments_path(labels_dir: Path, subject_id: str) -> Path:
    """Segments of one subject (segments.csv is these, for all subjects)."""
    return labels_dir / f"{subject_id}_segments.csv"


def _is_cached(job: LabelJob) -> bool:
    """Outputs exist and are newer than the features, swim model and human relabels they were made from."""
    outputs = [labeled_bins_path(job.labels_dir, job.subject_id), segments_path(job.labels_dir, job.subject_id)]
    if job.force or not all(p.is_file() for p in outputs):
        return False
    oldest = min(p.stat().st_mtime for p in outputs)
    inputs = [bins_path(job.features_dir, job.subject_id), job.labels_dir / MODEL_FILE, job.labels_dir / OVERRIDES_FILE]
    return all(not p.is_file() or p.stat().st_mtime <= oldest for p in inputs)


def read_bins(features_dir: Path, subject_id: str) -> pd.DataFrame:
    """The subject's feature bins; raises if `features` has not run."""
    path = bins_path(features_dir, subject_id)
    if not path.is_file():
        raise RuntimeError(f"no features ({path.name}); run `features` first")
    return pd.read_csv(path, dtype={"subject_id": str})


def process_subject(job: LabelJob) -> dict[str, Any]:
    """Label one subject and write its bins and segments; errors are returned, not raised."""
    if _is_cached(job):
        return {"subject_id": job.subject_id, "cached": True}
    try:
        bins = read_bins(job.features_dir, job.subject_id)
        labeled = apply_label_overrides(label_bins(bins, job.params, job.model, job.bin_s), job.overrides, job.bin_s)
        track_file = track_path(job.tracks_dir, job.subject_id)
        if not track_file.is_file():
            raise RuntimeError(f"no track ({track_file.name}); run `track` first")
        timeline = pd.read_csv(track_file, usecols=["part", "part_frame", "time_s"])
        segments = make_segments(labeled, timeline, job.bin_s)
    except Exception as error:  # noqa: BLE001 - reported per subject in the CLI summary
        log.error("%s: %s", job.subject_id, error)
        return {"subject_id": job.subject_id, "error": str(error)}
    job.labels_dir.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(labeled_bins_path(job.labels_dir, job.subject_id), index=False, float_format="%.5g")
    segments.to_csv(segments_path(job.labels_dir, job.subject_id), index=False)
    log.info("%s: %d segments", job.subject_id, len(segments))
    return {"subject_id": job.subject_id, "cached": False}


# ---------------------------------------------------------------------------
# 6. Whole run (called by the CLI)
# ---------------------------------------------------------------------------


@dataclass
class LabelRun:
    """What `run_labeling` did, for the CLI summary."""

    records: list[dict[str, Any]]  # this run's subjects: {"cached"} or {"error"}
    summary: pd.DataFrame  # summary.csv contents (every labeled subject)
    skipped: list[str]  # subjects not processed (video_status is not matched)
    model: dict[str, Any]
    model_refit: bool
    calibrated: Path | None  # calibrated.yaml, when its values were used
    labels_dir: Path


def labels_dir(settings: Settings) -> Path:
    """The step's output folder."""
    return settings.paths.output_dir / "labels"


def model_params(params: dict[str, Any]) -> dict[str, Any]:
    """The settings stored with the model (JSON-compatible): a change means refit and relabel all."""
    return json.loads(json.dumps(params))


def load_or_fit_model(settings: Settings, subjects: list[str], params: dict[str, Any],
                      refit: bool) -> tuple[dict[str, Any], bool]:
    """The saved swim model if it was made with these settings from the current features,
    else one fitted on `subjects`."""
    path = labels_dir(settings) / MODEL_FILE
    if not refit and path.is_file():
        model = json.loads(path.read_text(encoding="utf-8"))
        made = path.stat().st_mtime
        fresh = all(bins_path(get_features_dir(settings), s).stat().st_mtime <= made for s in subjects)
        if model.get("params") == model_params(params) and fresh:
            return model, False
    bin_s = float(settings.params["features"]["bin_s"])
    tables = []
    for subject_id in subjects:
        bins = read_bins(get_features_dir(settings), subject_id)
        tables.append((bins, (rule_labels(bins, params, bin_s)[0] == SWIM) & moving(bins, params)))
    model = {**fit_swim_model(tables, params), "params": model_params(params)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    log.info("swim model (%s) fitted on %d bins of %d subjects", params["swim_split"], model["n_bins"], len(tables))
    return model, True


def run_labeling(settings: Settings, trials: pd.DataFrame, selected: pd.DataFrame, force: bool = False) -> LabelRun:
    """Label every `selected` matched subject, then rewrite segments.csv and summary.csv.

    The swim model is fitted on every matched subject with features (not only `selected`)
    when it is missing, the labeling settings changed, any features are newer than it, or
    `force` is given for all subjects.
    """
    params, calibrated = labeling_params(settings)
    if calibrated:
        log.info("using calibrated labeling values from %s", calibrated)
    out, features = labels_dir(settings), get_features_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    matched = trials[trials["video_status"] == STATUS_MATCHED]
    with_features = [s for s in matched["subject_id"] if bins_path(features, s).is_file()]
    model, refit = load_or_fit_model(settings, with_features, params, refit=force and len(selected) == len(trials))

    is_matched = selected["video_status"] == STATUS_MATCHED
    skipped = [f"{row.subject_id} ({row.video_status})" for row in selected[~is_matched].itertuples()]
    bin_s = float(settings.params["features"]["bin_s"])
    overrides = load_label_overrides(out)  # human relabels (live page / by hand), read once for all subjects
    jobs = [LabelJob(s, features, get_tracks_dir(settings), out, params, bin_s, model, force,
                     overrides[overrides["subject_id"] == s])
            for s in selected.loc[is_matched, "subject_id"]]
    records = run_parallel(process_subject, jobs, settings.workers, desc="label")

    # segments.csv / summary.csv cover every matched subject labeled so far (this run or earlier).
    parts = [pd.read_csv(segments_path(out, s), dtype={"subject_id": str}) for s in matched["subject_id"]
             if segments_path(out, s).is_file()]
    segments = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SEGMENT_COLUMNS)
    segments.to_csv(out / SEGMENTS_FILE, index=False)
    summary = summarize(segments) if len(segments) else pd.DataFrame(columns=["subject_id", "duration_s"])
    summary.to_csv(out / SUMMARY_FILE, index=False)
    return LabelRun(records=records, summary=summary, skipped=skipped, model=model, model_refit=refit,
                    calibrated=calibrated, labels_dir=out)
