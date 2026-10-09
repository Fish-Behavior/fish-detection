"""Training table, schema JSON and `dcs featurize` on synthetic gold data (plan U5, T1.10): FR-1, EC-11, D-015."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from dcs import schema
from dcs.cli import main
from dcs.config import ENV_ACCEPTED_DIR, ENV_DB_PATH, ENV_OUTPUT_DIR, ENV_PROCESSED_DIR, load_settings
from dcs.featurize import (
    DROP_NO_DETECTED,
    KINDS,
    ROLES,
    Featurized,
    featurize,
    kinematic_features,
    schema_path,
    segment_features,
    write_outputs,
)
from dcs.gold import DROP_MISSING_FILE, read_accepted, read_processed
from dcs.synthetic import SynthResult, make_gold_dataset
from dcs.synthetic_config import KNOB_LOW_DETECTED_FISH, KNOB_MISSING_FILE, KNOB_UNDETERMINED
from dcs.workbook import HAS_NTT, NTT_COLUMNS, read_ntt
from tests.dcs.gold_helpers import only, rewrite_table, to_processed
from tests.dcs.synth_helpers import SMALL

LABELS = ["compound", "concentration_mM"]


@pytest.fixture(scope="module")
def training() -> dict[str, Any]:
    return dict(load_settings().training)


@pytest.fixture(scope="module")
def knobbed(tmp_path_factory: pytest.TempPathFactory) -> SynthResult:
    config = dataclasses.replace(SMALL, low_detected_fish=2, missing_file="frames")
    return make_gold_dataset(tmp_path_factory.mktemp("knobbed"), config)


@pytest.fixture(scope="module")
def result(knobbed: SynthResult, training: dict[str, Any]) -> Featurized:
    return featurize(read_accepted(knobbed.accepted_dir), knobbed.workbook_path, training)


def columns_by(result: Featurized, key: str, value: Any) -> list[str]:
    return [column["name"] for column in result.schema["columns"] if column.get(key) == value]


# --- the table -------------------------------------------------------------------------------


def test_one_row_per_kept_fish_in_gold_order(knobbed: SynthResult, result: Featurized) -> None:
    gold = read_accepted(knobbed.accepted_dir)
    assert list(result.table["video_id"]) == list(gold.video_ids)
    assert result.table["video_id"].is_unique


def test_table_columns_are_the_schema_columns_in_order(result: Featurized) -> None:
    assert list(result.table.columns) == [column["name"] for column in result.schema["columns"]]


def test_features_match_the_feature_functions_on_the_fish_files(knobbed: SynthResult, result: Featurized) -> None:
    row = result.table.iloc[0]
    folder = knobbed.accepted_dir / row["video_id"]
    expected = {
        **segment_features(pd.read_csv(folder / schema.SEGMENTS_FILE)),
        **kinematic_features(pd.read_parquet(folder / schema.FRAMES_FILE)),
    }
    for name, value in expected.items():
        assert row[name] == pytest.approx(value), name


def test_labels_and_date_come_from_the_gold_set(knobbed: SynthResult, result: Featurized) -> None:
    videos = read_accepted(knobbed.accepted_dir).videos
    for name in [*LABELS, "date", "sex", "subject_id"]:
        assert list(result.table[name]) == list(videos[name]), name


def test_no_missing_feature_except_ntt_of_fish_without_ntt(result: Featurized) -> None:
    features = [name for name in columns_by(result, "role", "feature") if name not in NTT_COLUMNS]
    assert not result.table[features].isna().any().any()
    ntt_missing = result.table[list(NTT_COLUMNS)].isna().any(axis=1)
    assert (result.table.loc[ntt_missing, HAS_NTT] == 0).all()


def test_ntt_columns_come_from_the_workbook(knobbed: SynthResult, result: Featurized) -> None:
    expected = read_ntt(knobbed.workbook_path, read_accepted(knobbed.accepted_dir).videos)
    pd.testing.assert_frame_equal(
        result.table[["video_id", *NTT_COLUMNS, HAS_NTT]].reset_index(drop=True), expected, check_dtype=False
    )
    assert result.schema["ntt_available"] is True


def test_without_a_workbook_ntt_is_absent_for_every_fish(knobbed: SynthResult, training: dict[str, Any]) -> None:
    result = featurize(read_accepted(knobbed.accepted_dir), None, training)
    assert (result.table[HAS_NTT] == 0).all()
    assert result.table[list(NTT_COLUMNS)].isna().all().all()
    assert result.schema["ntt_available"] is False


def test_manual_share_is_the_share_of_manual_frames(knobbed: SynthResult, result: Featurized) -> None:
    for record in result.table.to_dict("records"):
        frames = pd.read_parquet(knobbed.accepted_dir / record["video_id"] / schema.FRAMES_FILE)
        assert record["manual_share"] == pytest.approx((frames["source"] == schema.SOURCE_MANUAL).mean())
    assert (result.table["manual_share"] > 0).any()  # SMALL edits some fish


# --- EC-11 flags: kept, never dropped ----------------------------------------------------------


def test_ec11_low_detected_share_is_flagged_and_kept(knobbed: SynthResult, result: Featurized) -> None:
    flagged = set(result.table.loc[result.table["flag_low_detected"] == 1, "video_id"])
    assert flagged == set(knobbed.targets[KNOB_LOW_DETECTED_FISH])
    assert (result.table["detected_share"] < 0.8).sum() == len(flagged)


def test_ec11_duration_flag_follows_the_configured_range(knobbed: SynthResult, training: dict[str, Any]) -> None:
    gold = read_accepted(knobbed.accepted_dir)
    unset = featurize(gold, None, training)
    assert (unset.table["flag_odd_duration"] == 0).all()  # null range: no check (D-021)
    outside = featurize(gold, None, {**training, "duration_range_s": [30.0, 60.0]})
    assert (outside.table["flag_odd_duration"] == 1).all()  # SMALL records 20 s
    inside = featurize(gold, None, {**training, "duration_range_s": [20.0, 60.0]})
    assert (inside.table["flag_odd_duration"] == 0).all()
    assert len(outside.table) == len(unset.table)


# --- drops are counted -------------------------------------------------------------------------


def test_fish_without_any_detected_frame_is_dropped_and_counted(
    knobbed: SynthResult, training: dict[str, Any], tmp_path: Path
) -> None:
    accepted = tmp_path / "accepted"
    shutil.copytree(knobbed.accepted_dir, accepted)
    gold = read_accepted(accepted)
    victim = gold.video_ids[0]
    rewrite_table(accepted / victim / schema.FRAMES_FILE, lambda frames: frames.assign(detected=False))
    result = featurize(gold, None, training)
    assert victim not in set(result.table["video_id"])
    assert len(result.table) == len(gold.video_ids) - 1
    reasons = {drop["video_id"]: drop["reason"] for drop in result.schema["dropped"]}
    assert reasons[victim] == DROP_NO_DETECTED


def test_gold_drops_are_carried_into_the_schema(knobbed: SynthResult, result: Featurized) -> None:
    reasons = {drop["video_id"]: drop["reason"] for drop in result.schema["dropped"]}
    assert reasons[only(knobbed, KNOB_MISSING_FILE)] == DROP_MISSING_FILE


# --- unreviewed source -------------------------------------------------------------------------


def test_unreviewed_rows_keep_undetermined_as_meta_only(training: dict[str, Any], tmp_path: Path) -> None:
    synth = make_gold_dataset(tmp_path / "synth", dataclasses.replace(SMALL, undetermined=True))
    gold = read_processed(to_processed(synth, tmp_path / "pds"))
    result = featurize(gold, None, training)
    table = result.table.set_index("video_id")
    victim = only(synth, KNOB_UNDETERMINED)
    assert 0 < table.loc[victim, "undetermined_share"] < 1
    assert not table["reviewed"].any()
    assert result.schema["gold_source"] == "processed"
    assert not [name for name in table.columns if "undetermined" in name and name != "undetermined_share"]
    shares = table[[name for name in table.columns if name.startswith("state_") and name.endswith("_share")]]
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)  # shares of the known time


# --- the schema JSON ---------------------------------------------------------------------------


def test_every_column_has_a_known_role_source_and_kind(result: Featurized) -> None:
    for column in result.schema["columns"]:
        assert column["role"] in ROLES, column
        assert column["source"], column
        if column["role"] == "feature":
            assert column["kind"] in KINDS, column
            assert column["feature_group"], column


def test_roles_of_identifiers_labels_and_group(result: Featurized) -> None:
    assert columns_by(result, "role", "id") == ["video_id", "subject_id"]
    assert columns_by(result, "role", "label") == LABELS
    assert columns_by(result, "role", "group") == ["date"]


def test_d015_depth_features_form_their_own_group(result: Featurized) -> None:
    depth = columns_by(result, "feature_group", "depth")
    assert depth and all(name.startswith("depth_") for name in depth)
    assert not [name for name in result.table.columns if name.startswith("depth_") and name not in depth]


def test_state_features_name_their_states(result: Featurized) -> None:
    for column in result.schema["columns"]:
        if column["name"].startswith("state_"):
            assert len(column["states"]) == 1, column
        elif column["name"].startswith("trans_"):
            assert len(column["states"]) == 2, column
        else:
            assert "states" not in column, column


def test_counts_and_durations_are_marked_for_log1p(result: Featurized) -> None:
    kinds = {column["name"]: column.get("kind") for column in result.schema["columns"]}
    assert kinds["state_controlled_swim_bouts"] == "count"
    assert kinds["trans_controlled_swim_to_erratic_movement"] == "count"
    assert kinds["state_controlled_swim_mean_bout_s"] == "duration"
    assert kinds["state_controlled_swim_share"] == "fraction"


def test_write_outputs_round_trips(result: Featurized, tmp_path: Path) -> None:
    table_path, json_path = write_outputs(result, tmp_path / "out" / "training_table.parquet")
    assert json_path == schema_path(table_path) == tmp_path / "out" / "training_table_schema.json"
    pd.testing.assert_frame_equal(pd.read_parquet(table_path), result.table)
    assert json.loads(json_path.read_text(encoding="utf-8")) == result.schema


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_featurize_writes_table_and_schema(
    knobbed: SynthResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    override = tmp_path / "accepted.yaml"
    override.write_text(yaml.safe_dump({"training": {"gold_source": "accepted"}}), encoding="utf-8")
    monkeypatch.setenv(ENV_ACCEPTED_DIR, str(knobbed.accepted_dir))
    monkeypatch.setenv(ENV_DB_PATH, str(knobbed.workbook_path))
    monkeypatch.setenv(ENV_OUTPUT_DIR, str(tmp_path / "out"))
    assert main(["--config", str(override), "featurize"]) == 0
    table = pd.read_parquet(tmp_path / "out" / "training_table.parquet")
    assert len(table) == len(read_accepted(knobbed.accepted_dir).video_ids)
    assert (tmp_path / "out" / "training_table_schema.json").is_file()
    out = capsys.readouterr().out
    assert f"{len(table)} fish" in out and "1 dropped" in out


def test_cli_featurize_without_data_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_PROCESSED_DIR, str(tmp_path / "nowhere"))
    assert main(["featurize"]) == 2
    assert "Configuration error" in capsys.readouterr().out
