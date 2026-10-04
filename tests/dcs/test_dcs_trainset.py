"""Training set from the training table (plan U6, T1.12): EC-4, EC-7, EC-8, EC-12, EC-17, EC-20, EC-23, D-015."""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dcs import schema
from dcs.config import ConfigError, load_settings
from dcs.featurize import COLUMNS, featurize, schema_path, slug, write_outputs
from dcs.gold import read_accepted
from dcs.synthetic import make_gold_dataset
from dcs.trainset import (
    DROP_ALL_MISSING,
    DROP_CONSTANT,
    DROP_FEW_FISH,
    DROP_SINGLE_DOSE,
    DROP_STATE_SUPPORT,
    TrainSet,
    build_trainset,
    compound_label,
    dose_label,
    is_forbidden,
    load_table,
)
from tests.dcs.synth_helpers import SMALL

LORR = schema.LISTING_LORR
SCHEMA = {"columns": [column.as_json() for column in COLUMNS]}
FEATURES = [column.name for column in COLUMNS if column.role == "feature"]
GROUP = {column.name: column.feature_group for column in COLUMNS}


@pytest.fixture(scope="module")
def defaults() -> dict[str, Any]:
    return dict(load_settings().training)


def table(*classes: tuple[str, str, int], seed: int = 0) -> pd.DataFrame:
    """(compound, dose, fish) groups -> a training table whose features all vary and every fish shows every state."""
    labels = [(compound, dose) for compound, dose, fish in classes for _ in range(fish)]
    rng = np.random.default_rng(seed)
    n = len(labels)
    data: dict[str, Any] = {column.name: np.ones(n) for column in COLUMNS}  # meta columns: any value
    data.update({name: rng.uniform(1, 2, n) for name in FEATURES})
    data.update(
        video_id=[f"F_{i:04d}" for i in range(n)],
        subject_id=[f"{i:04d}" for i in range(n)],
        compound=[compound for compound, _ in labels],
        concentration_mM=[dose for _, dose in labels],
        date=[f"<date {i % 4}>" for i in range(n)],
        sex=rng.choice(["F", "M"], n),
        strain=rng.choice(["S1", "S2"], n),
        has_ntt=rng.integers(0, 2, n),
    )
    return pd.DataFrame(data, columns=[column.name for column in COLUMNS])


def build(frame: pd.DataFrame, training: dict[str, Any], stage: str = "compound", **changes: Any) -> TrainSet:
    return build_trainset(frame, SCHEMA, {**training, **changes}, stage)


def dropped_names(result: TrainSet, reason: str) -> set[str]:
    return {drop["name"] for drop in result.dropped_features if drop["reason"] == reason}


# --- labels (EC-7, EC-8) ---------------------------------------------------------------------


@pytest.mark.parametrize("written", ["COMPOUND_A", "compound_a", "  Compound_A\t", "COMPOUND_A "])
def test_compound_label_ignores_case_and_outer_spaces(written: str) -> None:
    assert compound_label(written) == "COMPOUND_A"


def test_compound_label_collapses_inner_spaces() -> None:
    assert compound_label("compound_a  +\tCompound_B") == compound_label("COMPOUND_A + COMPOUND_B") == "COMPOUND_A + COMPOUND_B"


@pytest.mark.parametrize("written", ["0.03 + 0.01", "0.03+0.01", " 0.03 +0.01 "])
def test_dose_label_removes_spaces_so_spellings_match(written: str) -> None:
    assert dose_label(written) == "0.03+0.01"


def test_dose_label_keeps_the_written_string() -> None:
    assert (dose_label("0.1"), dose_label(0.1), dose_label("1")) == ("0.1", "0.1", "1")


def test_case_and_space_variants_are_one_class(defaults: dict[str, Any]) -> None:
    frame = table(("compound_a", "0.1", 3), ("COMPOUND_A ", "0.1", 3), ("COMPOUND_B", "0.1", 6))
    result = build(frame, defaults)
    assert result.classes == {"COMPOUND_A": 6, "COMPOUND_B": 6}


def test_dose_spellings_are_one_dose_class(defaults: dict[str, Any]) -> None:
    frame = table(("A", "0.03 + 0.01", 3), ("a", "0.03+0.01", 3), ("A", "1", 6))
    result = build(frame, defaults, stage="dose")
    assert result.classes == {"A @ 0.03+0.01": 6, "A @ 1": 6}


# --- class filter (EC-4, very small) ---------------------------------------------------------


def test_small_classes_are_dropped_and_listed(defaults: dict[str, Any]) -> None:
    result = build(table(("A", "0.1", 7), ("B", "0.1", 6), ("C", "0.1", 5)), defaults)
    assert result.classes == {"A": 7, "B": 6}
    assert result.dropped_classes == ({"label": "C", "fish": 5, "reason": DROP_FEW_FISH},)
    assert set(result.y) == {"A", "B"} and len(result.X) == 13


def test_classes_of_exactly_min_class_size_are_flagged_very_small(defaults: dict[str, Any]) -> None:
    result = build(table(("A", "0.1", 7), ("B", "0.1", 6), ("C", "0.1", 9)), defaults, min_class_size=7)
    assert result.very_small == ("A",)
    assert "B" in {drop["label"] for drop in result.dropped_classes}


def test_dose_stage_drops_small_doses_then_compounds_left_with_one_dose(defaults: dict[str, Any]) -> None:
    frame = table(
        ("A", "0.1", 6), ("A", "0.3", 7), ("A", "1", 3), ("VEHICLE", "0", 8), ("B", "0.1", 6), ("B", "0.3", 2)
    )
    result = build(frame, defaults, stage="dose")
    assert result.classes == {"A @ 0.1": 6, "A @ 0.3": 7}
    assert result.very_small == ("A @ 0.1",)
    assert result.dropped_classes == (
        {"label": "A @ 1", "fish": 3, "reason": DROP_FEW_FISH},
        {"label": "B @ 0.3", "fish": 2, "reason": DROP_FEW_FISH},
        {"label": "B @ 0.1", "fish": 6, "reason": DROP_SINGLE_DOSE},
        {"label": "VEHICLE @ 0", "fish": 8, "reason": DROP_SINGLE_DOSE},
    )


def test_rows_line_up_across_matrix_labels_groups_and_ids(defaults: dict[str, Any]) -> None:
    frame = table(("A", "0.1", 6), ("B", "0.1", 6), ("C", "0.1", 2))
    result = build(frame, defaults)
    kept = frame[frame["compound"] != "C"]
    for part in (result.X, result.y, result.groups, result.ids):
        assert list(part.index) == list(kept.index)
    assert list(result.ids) == list(kept["video_id"])
    assert list(result.groups) == list(kept["date"])


@pytest.mark.parametrize("stage", ["both", "Compound", ""])
def test_unknown_stage_is_an_error(defaults: dict[str, Any], stage: str) -> None:
    with pytest.raises(ValueError, match="stage"):
        build(table(("A", "0.1", 6), ("B", "0.1", 6)), defaults, stage=stage)


# --- too little to train on (EC-17) ----------------------------------------------------------


def test_one_class_left_is_a_config_error_with_a_hint(defaults: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="min_class_size"):
        build(table(("A", "0.1", 8), ("B", "0.1", 5)), defaults)


def test_no_class_left_is_a_config_error(defaults: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="1 class|0 class"):
        build(table(("A", "0.1", 2), ("B", "0.1", 3)), defaults)


def test_empty_table_is_a_config_error(defaults: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="featurize"):
        build(table(), defaults)


def test_no_feature_left_is_a_config_error(defaults: dict[str, Any]) -> None:
    frame = table(("A", "0.1", 6), ("B", "0.1", 6))
    frame[FEATURES] = 1
    with pytest.raises(ConfigError, match="No feature left"):
        build(frame, defaults)


def test_schema_feature_missing_from_the_table_is_a_config_error(defaults: dict[str, Any]) -> None:
    frame = table(("A", "0.1", 6), ("B", "0.1", 6)).drop(columns="velocity_mean")
    with pytest.raises(ConfigError, match="velocity_mean"):
        build(frame, defaults)


# --- forbidden columns (EC-12) ---------------------------------------------------------------

PRD_FORBIDDEN = (
    "compound", "concentration_mM", "subject_id", "date", "video_path", "frames_path", "segments_path",
    "strip_path", "manifest_path", "reviewer", "reviewed_at", "edit_count", "manual_share",
    "calibration_profile_version", "pipeline_version", "agent_exposure_min", "ntt_min", "uv_min", "h2o_before",
    "h2o_after", "brain_tissue", "video_duration_s", "video_fps", "resolution",
    "video_id", "provenance", "edited", "review_flags", "processed_at",
)  # fmt: skip


@pytest.mark.parametrize("name", PRD_FORBIDDEN)
def test_prd_excluded_columns_are_forbidden(name: str) -> None:
    assert is_forbidden(name, use_demographics=False)
    assert is_forbidden(name, use_demographics=True)


@pytest.mark.parametrize("name", ["age", "sex", "strain"])
def test_demographics_are_forbidden_unless_switched_on(name: str) -> None:
    assert is_forbidden(name, use_demographics=False)
    assert not is_forbidden(name, use_demographics=True)


@pytest.mark.parametrize("name", FEATURES)
def test_no_featurize_feature_is_forbidden_when_its_group_is_on(name: str) -> None:
    assert not is_forbidden(name, use_demographics=True)


@pytest.mark.parametrize("switches", [{}, {"use_depth": True, "use_demographics": True}, {"use_ntt": False}])
def test_feature_matrix_holds_only_allowed_feature_columns(defaults: dict[str, Any], switches: dict[str, bool]) -> None:
    training = {**defaults, **switches}
    result = build(table(("A", "0.1", 6), ("B", "0.1", 6)), training)
    assert result.features and list(result.X.columns) == list(result.features)
    assert not [name for name in result.X.columns if is_forbidden(name, training["use_demographics"])]
    assert set(result.X.columns) <= set(FEATURES)


def test_schema_marking_a_forbidden_column_as_feature_stops_the_run(defaults: dict[str, Any]) -> None:
    promoted = {"role": "feature", "feature_group": "kinematics"}
    tampered = {"columns": [{**c, **promoted} if c["name"] == "manual_share" else c for c in SCHEMA["columns"]]}
    with pytest.raises(ConfigError, match="manual_share"):
        build_trainset(table(("A", "0.1", 6), ("B", "0.1", 6)), tampered, defaults, "compound")


def test_meta_columns_never_reach_the_matrix(defaults: dict[str, Any]) -> None:
    result = build(table(("A", "0.1", 6), ("B", "0.1", 6)), defaults, use_depth=True, use_demographics=True)
    meta = {column.name for column in COLUMNS if column.role != "feature"}
    assert not meta & set(result.X.columns)


# --- feature groups (D-015) ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "switch", "default_on"),
    [("depth", "use_depth", False), ("demographics", "use_demographics", False), ("ntt", "use_ntt", True)],
)
def test_feature_groups_follow_their_switch(defaults: dict[str, Any], group: str, switch: str, default_on: bool) -> None:
    frame = table(("A", "0.1", 6), ("B", "0.1", 6))
    in_group = {name for name in FEATURES if GROUP[name] == group}
    assert defaults[switch] is default_on
    on, off = build(frame, defaults, **{switch: True}), build(frame, defaults, **{switch: False})
    assert in_group <= set(on.X.columns)
    assert not in_group & set(off.X.columns)


def test_kinds_name_each_feature_kind_for_the_transforms(defaults: dict[str, Any]) -> None:
    result = build(table(("A", "0.1", 6), ("B", "0.1", 6)), defaults)
    assert set(result.kinds) == set(result.features)
    assert result.kinds["state_controlled_swim_bouts"] == "count"
    assert result.kinds["velocity_mean"] == "continuous"


# --- rare states (EC-23) ---------------------------------------------------------------------


def lorr_columns() -> set[str]:
    return {column.name for column in COLUMNS if column.role == "feature" and LORR in column.states}


def with_lorr_fish(frame: pd.DataFrame, fish: list[int]) -> pd.DataFrame:
    """Only the fish at these positions show Listing/LORR."""
    frame = frame.copy()
    hidden = ~frame.index.isin(fish)
    s = slug(LORR)
    frame.loc[hidden, [f"state_{s}_share", f"state_{s}_bouts", f"state_{s}_mean_bout_s"]] = 0.0
    return frame


def test_state_shown_by_too_few_fish_loses_all_its_features(defaults: dict[str, Any]) -> None:
    frame = with_lorr_fish(table(("A", "0.1", 8), ("B", "0.1", 8)), list(range(9)))
    result = build(frame, defaults)
    assert result.dropped_states == (LORR,)
    assert lorr_columns() == dropped_names(result, DROP_STATE_SUPPORT)
    assert not lorr_columns() & set(result.X.columns)
    assert "trans_controlled_swim_to_freezing_drift" in result.X.columns  # other transitions stay


def test_state_shown_by_min_state_fish_is_kept(defaults: dict[str, Any]) -> None:
    frame = with_lorr_fish(table(("A", "0.1", 8), ("B", "0.1", 8)), list(range(10)))
    result = build(frame, defaults)
    assert result.dropped_states == ()
    assert lorr_columns() <= set(result.X.columns)


def test_state_support_counts_only_fish_of_kept_classes(defaults: dict[str, Any]) -> None:
    frame = table(("A", "0.1", 8), ("B", "0.1", 8), ("C", "0.1", 2))
    frame = with_lorr_fish(frame, [0, 1, 2, 3, 4, 5, 6, 7, 16, 17])  # 10 fish, 2 of them in the dropped class C
    assert build(frame, defaults).dropped_states == (LORR,)


# --- constant and empty features (EC-20) -----------------------------------------------------


def test_constant_and_all_missing_features_are_dropped_without_warnings(defaults: dict[str, Any]) -> None:
    frame = table(("A", "0.1", 6), ("B", "0.1", 6))
    frame["meander_mean"] = 0.0
    frame["velocity_cv"] = np.nan
    frame.loc[0, "abs_acceleration_mean"] = np.nan  # some missing, otherwise varying: kept
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = build(frame, defaults)
    assert dropped_names(result, DROP_CONSTANT) == {"meander_mean"}
    assert dropped_names(result, DROP_ALL_MISSING) == {"velocity_cv"}
    assert "abs_acceleration_mean" in result.X.columns


def test_constancy_is_judged_on_kept_fish_only(defaults: dict[str, Any]) -> None:
    frame = table(("A", "0.1", 6), ("B", "0.1", 6), ("C", "0.1", 2))
    frame.loc[frame["compound"] != "C", "meander_mean"] = 0.5
    assert "meander_mean" in dropped_names(build(frame, defaults), DROP_CONSTANT)


# --- reading the two files, and a synthetic set end to end -----------------------------------


@pytest.fixture(scope="module")
def synthetic_table(tmp_path_factory: pytest.TempPathFactory, defaults: dict[str, Any]) -> Path:
    config = dataclasses.replace(SMALL, messy_labels=True, tiny_class=True, rare_state_fish=3, constant_meander=True)
    synth = make_gold_dataset(tmp_path_factory.mktemp("trainset"), config)
    result = featurize(read_accepted(synth.accepted_dir), synth.workbook_path, defaults)
    table_path, _ = write_outputs(result, tmp_path_factory.mktemp("out") / "training_table.parquet")
    return table_path


def test_load_table_reads_the_table_and_its_schema(synthetic_table: Path) -> None:
    frame, described = load_table(synthetic_table)
    assert list(frame.columns) == [column["name"] for column in described["columns"]]


def test_load_table_without_schema_says_to_run_featurize(synthetic_table: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copied.parquet"
    copy.write_bytes(synthetic_table.read_bytes())
    with pytest.raises(ConfigError, match="featurize"):
        load_table(copy)


def test_load_table_with_a_damaged_schema_is_a_config_error(synthetic_table: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copied.parquet"
    copy.write_bytes(synthetic_table.read_bytes())
    schema_path(copy).write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="schema"):
        load_table(copy)


def test_synthetic_set_end_to_end(synthetic_table: Path, defaults: dict[str, Any]) -> None:
    frame, described = load_table(synthetic_table)
    result = build_trainset(frame, described, {**defaults, "min_class_size": 4, "min_state_fish": 4}, "compound")
    # messy spellings joined their canonical class; the 5-fish tiny class passes at min_class_size 4
    assert result.classes == {"COMPOUND_A": 8, "COMPOUND_B": 8, "COMPOUND_A + COMPOUND_B": 4, "COMPOUND_T": 5, "VEHICLE": 4}
    assert set(result.very_small) == {"COMPOUND_A + COMPOUND_B", "VEHICLE"}
    assert LORR in result.dropped_states  # 3 fish < 4 (synthetic fish never show Dead either)
    assert "meander_mean" in dropped_names(result, DROP_CONSTANT)
    assert not [name for name in result.X.columns if is_forbidden(name, use_demographics=False)]
    assert not result.X.isna().all().any()
    dose = build_trainset(frame, described, {**defaults, "min_class_size": 4, "min_state_fish": 4}, "dose")
    assert "COMPOUND_A + COMPOUND_B @ 0.03+0.01" not in dose.classes  # one dose only: single_dose
    assert {"COMPOUND_A @ 0.1", "COMPOUND_A @ 0.3", "COMPOUND_B @ 0.1", "COMPOUND_B @ 0.3"} == set(dose.classes)
    json.dumps([*result.dropped_classes, *result.dropped_features])  # audit-ready: plain JSON values
