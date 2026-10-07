"""Training table -> what one stage trains on (plan U6, T1.13; PRD §5.3, §5.6).

`build_trainset` cleans the labels (EC-7, EC-8), drops classes with too few fish (EC-4), keeps the
feature groups switched on (D-015), refuses forbidden columns (EC-12), drops the features of states
too few fish show (EC-23) and features with nothing to learn from (EC-20), and lists everything it
dropped for the audit. It fills no missing value and fits no transform: that is per fold (U9).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import pandas as pd

from dcs import config_rules
from dcs.config import ConfigError
from dcs.featurize import BEHAVIOR_STATES, schema_path, slug
from dcs.gold_rules import compound_label

STAGES = tuple(stage for stage in config_rules.STAGES if stage != "both")  # `both` builds one, then the other
DOSE_SEPARATOR = " @ "  # dose-stage label: `COMPOUND_A @ 0.1`
# Group switches; a group not listed here (states, transitions, kinematics) is always on.
GROUP_SWITCHES = {"depth": "use_depth", "ntt": "use_ntt", "demographics": "use_demographics"}
# PRD §5.3 exclusions plus the plan's additions (U6) and the meta fields featurize writes.
FORBIDDEN = (
    "compound", "concentration_mM", "subject_id", "date", "video_id", "*_path",
    "reviewer", "reviewed_at", "reviewed", "review_status", "review_flags", "review_flag_count",
    "edited", "edit_count", "manual_share", "provenance", "processed_at",
    "calibration_profile_version", "pipeline_version",
    "agent_exposure_min", "ntt_min", "uv_min", "h2o_*", "brain_tissue", "body_tissue",
    "video_duration_s", "video_fps", "resolution", "recording_s",
)  # fmt: skip
DEMOGRAPHICS = ("age", "sex", "strain")  # allowed only with use_demographics (an ablation)

DROP_FEW_FISH = "fewer_than_min_class_size"
DROP_SINGLE_DOSE = "single_dose"
DROP_STATE_SUPPORT = "state_shown_by_too_few_fish"
DROP_CONSTANT = "constant"
DROP_ALL_MISSING = "all_missing"


@dataclass(frozen=True)
class TrainSet:
    """Rows keep the training table's index, so X, y, groups and ids line up."""

    stage: str
    X: pd.DataFrame  # feature columns only
    y: pd.Series  # cleaned label
    groups: pd.Series  # date, for the folds
    ids: pd.Series  # video_id
    features: tuple[str, ...]
    kinds: dict[str, str]  # feature -> kind; log1p on count and duration (U9)
    classes: dict[str, int]  # kept label -> fish
    very_small: tuple[str, ...]  # kept classes of exactly min_class_size
    dropped_classes: tuple[dict[str, Any], ...]  # {label, fish, reason}
    dropped_states: tuple[str, ...]
    dropped_features: tuple[dict[str, str], ...]  # {name, reason}


def dose_label(value: Any) -> str:
    """The written dose without spaces; kept as text because combination doses are not numbers (EC-7)."""
    return "".join(str(value).split())


def is_forbidden(name: str, use_demographics: bool) -> bool:
    """True for a column that must never be a feature (EC-12)."""
    patterns = FORBIDDEN if use_demographics else (*FORBIDDEN, *DEMOGRAPHICS)
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def load_table(table_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The training table and its schema JSON, as `dcs featurize` wrote them."""
    json_path = schema_path(Path(table_path))
    if not json_path.is_file():
        raise ConfigError(f"No schema file {json_path} beside the table. Run `python -m dcs featurize`, or copy both files.")
    try:
        described = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ConfigError(f"The schema file {json_path} is damaged ({error}). Run `python -m dcs featurize` again.") from None
    return pd.read_parquet(table_path), described


def build_trainset(table: pd.DataFrame, described: Mapping[str, Any], training: Mapping[str, Any], stage: str) -> TrainSet:
    """Labels, class filter, feature selection and drop lists for `stage` (compound or dose)."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}, got {stage!r}")
    if table.empty:
        raise ConfigError("The training table has no fish. Run `python -m dcs featurize` and check its dropped list.")

    labels = stage_labels(table, stage)
    keep, dropped_classes = filter_classes(labels, table["compound"].map(compound_label), stage, training["min_class_size"])
    labels_before, labels, rows = labels, labels[keep], table[keep]
    classes = labels.value_counts().sort_index()
    if len(classes) < 2:
        before = ", ".join(f"{label} {fish}" for label, fish in labels_before.value_counts().sort_index().items())
        raise ConfigError(
            f"{len(classes)} class(es) left for the {stage} stage after the class filter; at least 2 are needed "
            f"(fish per class: {before}). Add fish, or lower training.min_class_size."
            + (" A compound needs 2 kept doses to enter the dose stage." if stage == "dose" else "")
        )

    columns = [c for c in described["columns"] if c["role"] == "feature" and _group_on(c.get("feature_group"), training)]
    leaked = [c["name"] for c in columns if is_forbidden(c["name"], training["use_demographics"])]
    if leaked:
        raise ConfigError(
            f"The schema marks forbidden column(s) {', '.join(leaked)} as features. Run `python -m dcs featurize` again."
        )
    missing = [c["name"] for c in columns if c["name"] not in rows.columns]
    if missing:
        raise ConfigError(
            f"Feature(s) {', '.join(missing)} are in the schema but not in the table. "
            "Copy the two files from one featurize run."
        )
    dropped_states, dropped_features = _drop_features(columns, rows, training["min_state_fish"])
    gone = {drop["name"] for drop in dropped_features}
    features = tuple(c["name"] for c in columns if c["name"] not in gone)
    if not features:
        raise ConfigError(
            "No feature left: every feature in the switched-on groups is constant, empty or of a rare state. "
            "Check the training table, or switch on another group (training.use_*)."
        )

    return TrainSet(
        stage=stage,
        X=rows[list(features)],
        y=labels.rename("label"),
        groups=rows["date"],
        ids=rows["video_id"],
        features=features,
        kinds={c["name"]: c.get("kind") for c in columns if c["name"] in features},
        classes={str(label): int(fish) for label, fish in classes.items()},
        very_small=tuple(str(label) for label, fish in classes.items() if fish == training["min_class_size"]),
        dropped_classes=tuple(dropped_classes),
        dropped_states=dropped_states,
        dropped_features=tuple(dropped_features),
    )


def stage_labels(table: pd.DataFrame, stage: str) -> pd.Series:
    """The cleaned label of every row: compound, or `<compound> @ <dose>` in the dose stage."""
    compound = table["compound"].map(compound_label)
    return compound if stage == "compound" else compound + DOSE_SEPARATOR + table["concentration_mM"].map(dose_label)


def _group_on(group: str | None, training: Mapping[str, Any]) -> bool:
    switch = GROUP_SWITCHES.get(group or "")
    return switch is None or bool(training[switch])


def _drop_features(
    columns: list[Mapping[str, Any]], rows: pd.DataFrame, min_state_fish: int
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """States shown by fewer than `min_state_fish` fish (EC-23), and the features dropped with a reason (EC-20)."""
    shown = {state: int((rows[f"state_{slug(state)}_bouts"] > 0).sum()) for state in BEHAVIOR_STATES}
    rare = tuple(state for state in BEHAVIOR_STATES if shown[state] < min_state_fish)
    dropped = []
    for column in columns:
        values = rows[column["name"]]
        if set(column.get("states", ())) & set(rare):  # a transition goes when either of its states does
            reason = DROP_STATE_SUPPORT
        elif values.isna().all():
            reason = DROP_ALL_MISSING
        elif values.nunique(dropna=True) == 1:
            reason = DROP_CONSTANT
        else:
            continue
        dropped.append({"name": column["name"], "reason": reason})
    return rare, dropped


def filter_classes(
    labels: pd.Series, compound: pd.Series, stage: str, min_size: int
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Rows to keep, and the dropped classes: small ones first, then (dose stage) compounds left with one dose."""
    counts = labels.value_counts()
    small = labels.isin(counts[counts < min_size].index)
    dropped = _listed(labels[small], DROP_FEW_FISH)
    keep = ~small
    if stage == "dose":
        doses = labels[keep].groupby(compound[keep]).nunique()
        single = compound.isin(doses[doses < 2].index) & keep
        dropped += _listed(labels[single], DROP_SINGLE_DOSE)
        keep &= ~single
    return keep, dropped


def _listed(labels: pd.Series, reason: str) -> list[dict[str, Any]]:
    counts = labels.value_counts().sort_index()
    return [{"label": str(label), "fish": int(fish), "reason": reason} for label, fish in counts.items()]
