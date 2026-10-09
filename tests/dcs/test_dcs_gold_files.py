"""Gold reader, per-file checks (plan U3): EC-1 (drop and list), EC-22 (Undetermined), EC-25 (file contract),
plus `load_video` and the small value helpers. Accepted source unless a test says otherwise."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from dcs import schema
from dcs.config import ConfigError
from dcs.gold import DROP_MISSING_FILE, DROP_UNREADABLE, GoldDataError, load_video, read_accepted, read_processed
from dcs.gold_checks import ARROW_TYPE_CHECKS
from dcs.gold_rules import same_label
from dcs.synthetic import SynthResult
from dcs.synthetic_config import (
    KNOB_MISSING_FILE,
    KNOB_MISSING_FRAME_COLUMN,
    KNOB_UNDETERMINED,
    KNOB_WRONG_DTYPE_COLUMN,
)
from tests.dcs.gold_helpers import (
    damage_column,
    dropped_reasons,
    only,
    rewrite_segments,
    rewrite_table,
    set_manifest,
    to_processed,
)
from tests.dcs.synth_helpers import index, make

# --- EC-1: missing or unreadable files are dropped, counted and listed ------------------


@pytest.mark.parametrize("missing", ["frames", "segments"])
def test_missing_file_drops_the_fish(tmp_path: Path, missing: str) -> None:
    result = make(tmp_path, missing_file=missing)
    victim = only(result, KNOB_MISSING_FILE)
    gold = read_accepted(result.accepted_dir)
    assert dropped_reasons(gold) == {victim: DROP_MISSING_FILE}
    assert victim not in gold.video_ids
    assert len(gold.videos) == len(result.truth) - 1
    detail = gold.dropped["detail"].iloc[0]
    assert (schema.FRAMES_FILE if missing == "frames" else schema.SEGMENTS_FILE) in detail


def test_missing_manifest_or_folder_drops_the_fish(tmp_path: Path) -> None:
    result = make(tmp_path)
    first, second = index(result)["video_id"].iloc[:2]
    (result.accepted_dir / first / schema.MANIFEST_FILE).unlink()
    shutil.rmtree(result.accepted_dir / second)
    assert dropped_reasons(read_accepted(result.accepted_dir)) == {first: DROP_MISSING_FILE, second: DROP_MISSING_FILE}


@pytest.mark.parametrize(
    ("name", "content"),
    [
        (schema.FRAMES_FILE, b"not a parquet file"),
        (schema.SEGMENTS_FILE, b""),
        (schema.MANIFEST_FILE, b"{broken"),
        (schema.MANIFEST_FILE, b"[1, 2]"),
    ],
)
def test_unreadable_file_drops_the_fish(tmp_path: Path, name: str, content: bytes) -> None:
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[3]
    (result.accepted_dir / victim / name).write_bytes(content)
    gold = read_accepted(result.accepted_dir)
    assert dropped_reasons(gold) == {victim: DROP_UNREADABLE}
    assert name in gold.dropped["detail"].iloc[0]


def test_segments_without_rows_drops_the_fish(tmp_path: Path) -> None:
    """A header-only segments.csv reads as text columns; it is a damaged file, not a contract violation."""
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[2]
    rewrite_segments(result.accepted_dir / victim, lambda table: table.iloc[0:0])
    gold = read_accepted(result.accepted_dir)
    assert dropped_reasons(gold) == {victim: DROP_UNREADABLE}
    assert "no rows" in gold.dropped["detail"].iloc[0]


@pytest.mark.parametrize("source", ["accepted", "processed"])
def test_damaged_frames_data_with_a_valid_footer_drops_the_fish(tmp_path: Path, source: str) -> None:
    """The schema reads, the state column does not: dropped at read time, never later in load_video."""
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[4]
    if source == "accepted":
        damage_column(result.accepted_dir / victim / schema.FRAMES_FILE, "state")
        gold = read_accepted(result.accepted_dir)
    else:
        out = to_processed(result, tmp_path / "outputs")
        damage_column(out / schema.PROCESSED_DIR_NAME / victim / schema.FRAMES_FILE, "state")
        gold = read_processed(out)
    assert dropped_reasons(gold) == {victim: DROP_UNREADABLE}


@pytest.mark.parametrize("name", [schema.INDEX_FILE, schema.CATALOG_FILE])
def test_unreadable_index_or_catalog_names_the_file(tmp_path: Path, name: str) -> None:
    result = make(tmp_path)
    if name == schema.INDEX_FILE:
        result.index_path.write_bytes(b"garbage")
        with pytest.raises(GoldDataError, match=rf"Cannot read .*{name}"):
            read_accepted(result.accepted_dir)
    else:
        out = to_processed(result, tmp_path / "outputs")
        (out / name).write_bytes(b"garbage")
        with pytest.raises(GoldDataError, match=rf"Cannot read .*{name}"):
            read_processed(out)


# --- EC-22: states ---------------------------------------------------------------------------


def test_undetermined_in_accepted_video_is_an_error(tmp_path: Path) -> None:
    result = make(tmp_path, undetermined=True)
    with pytest.raises(GoldDataError) as caught:
        read_accepted(result.accepted_dir)
    message = str(caught.value)
    assert schema.UNDETERMINED in message and only(result, KNOB_UNDETERMINED) in message and "EC-22" in message


def test_undetermined_only_in_frames_is_an_error(tmp_path: Path) -> None:
    """The segments may have been regenerated; the frames are checked on their own."""
    result = make(tmp_path, undetermined=True)
    victim = only(result, KNOB_UNDETERMINED)
    rewrite_segments(
        result.accepted_dir / victim,
        lambda table: table.assign(state=table["state"].replace(schema.UNDETERMINED, schema.CONTROLLED_SWIM)),
    )
    with pytest.raises(GoldDataError, match=rf"{victim}/{schema.FRAMES_FILE}.*{schema.UNDETERMINED}"):
        read_accepted(result.accepted_dir)


@pytest.mark.parametrize("state", ["Moonwalk", None])
def test_unknown_or_empty_segment_state_is_an_error(tmp_path: Path, state: str | None) -> None:
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[0]
    rewrite_segments(result.accepted_dir / victim, lambda table: table.assign(state=state))
    with pytest.raises(GoldDataError, match=rf"{victim}/{schema.SEGMENTS_FILE}.*state"):
        read_accepted(result.accepted_dir)


def test_unknown_frames_state_is_an_error_in_the_unreviewed_source_too(tmp_path: Path) -> None:
    result = make(tmp_path)
    out = to_processed(result, tmp_path / "outputs")
    victim = index(result)["video_id"].iloc[0]
    path = out / schema.PROCESSED_DIR_NAME / victim / schema.FRAMES_FILE
    frames = pd.read_parquet(path)
    frames["state"] = frames["state"].cat.add_categories(["Moonwalk"])
    frames.loc[0, "state"] = "Moonwalk"
    frames.to_parquet(path, index=False)
    with pytest.raises(GoldDataError, match=rf"{victim}/{schema.FRAMES_FILE}.*Moonwalk"):
        read_processed(out)


# --- EC-25: the file contract -------------------------------------------------------------


def test_missing_frames_column_names_file_and_column(tmp_path: Path) -> None:
    result = make(tmp_path, missing_frame_column="meander")
    victim = only(result, KNOB_MISSING_FRAME_COLUMN)
    with pytest.raises(GoldDataError, match=rf"{victim}/{schema.FRAMES_FILE}: column\(s\) meander missing"):
        read_accepted(result.accepted_dir)


def test_wrong_frames_dtype_names_file_and_column(tmp_path: Path) -> None:
    result = make(tmp_path, wrong_dtype_column="velocity")
    victim = only(result, KNOB_WRONG_DTYPE_COLUMN)
    expected = rf"{victim}/{schema.FRAMES_FILE}: column velocity has type .*expected float32"
    with pytest.raises(GoldDataError, match=expected):
        read_accepted(result.accepted_dir)


def test_every_frames_dtype_has_a_parquet_check() -> None:
    assert set(schema.FRAME_DTYPES.values()) <= set(ARROW_TYPE_CHECKS)


def test_missing_segments_column_names_file_and_column(tmp_path: Path) -> None:
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[0]
    rewrite_segments(result.accepted_dir / victim, lambda table: table.drop(columns=["duration_s"]))
    with pytest.raises(GoldDataError, match=rf"{victim}/{schema.SEGMENTS_FILE}: column\(s\) duration_s missing"):
        read_accepted(result.accepted_dir)


def test_text_segment_times_are_an_error(tmp_path: Path) -> None:
    """Text durations would make every Undetermined share 0; they must fail, naming the column."""
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[0]
    rewrite_segments(result.accepted_dir / victim, lambda table: table.assign(duration_s="long"))
    with pytest.raises(GoldDataError, match=rf"{victim}/{schema.SEGMENTS_FILE}: column duration_s"):
        read_accepted(result.accepted_dir)


@pytest.mark.parametrize("key", ["calibration_profile_version", "video_fps", "review_status"])
def test_missing_manifest_key_names_file_and_key(tmp_path: Path, key: str) -> None:
    result = make(tmp_path)
    out = to_processed(result, tmp_path / "outputs")
    victim = index(result)["video_id"].iloc[2]
    path = out / schema.PROCESSED_DIR_NAME / victim / schema.MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest[key]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(GoldDataError, match=rf"{victim}/{schema.MANIFEST_FILE}: key\(s\) {key} missing"):
        read_processed(out)


def test_missing_index_column_names_file_and_column(tmp_path: Path) -> None:
    result = make(tmp_path)
    rewrite_table(result.index_path, lambda table: table.drop(columns=["calibration_profile_version"]))
    with pytest.raises(GoldDataError, match=rf"{schema.INDEX_FILE}: column\(s\) calibration_profile_version missing"):
        read_accepted(result.accepted_dir)


def test_missing_index_is_a_config_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError, match="prepds export-index"):
        read_accepted(tmp_path / "empty")


# --- load_video ----------------------------------------------------------------------------


def test_load_video_returns_typed_frames_and_segments(small: SynthResult) -> None:
    gold = read_accepted(small.accepted_dir)
    video = load_video(gold, gold.video_ids[0])
    assert {name: str(dtype) for name, dtype in video.frames.dtypes.items()} == schema.FRAME_DTYPES
    assert tuple(video.segments.columns) == schema.SEGMENT_COLUMNS
    with pytest.raises(KeyError, match="F_9999"):
        load_video(gold, "F_9999")


def test_load_video_rechecks_files_changed_after_the_scan(tmp_path: Path) -> None:
    result = make(tmp_path)
    gold = read_accepted(result.accepted_dir)
    first, second, third = gold.video_ids[:3]
    frames = pd.read_parquet(result.accepted_dir / first / schema.FRAMES_FILE)
    retyped = frames.assign(velocity=frames["velocity"].astype("float64"))
    retyped.to_parquet(result.accepted_dir / first / schema.FRAMES_FILE)
    rewrite_segments(result.accepted_dir / second, lambda table: table.drop(columns=["source"]))
    (result.accepted_dir / third / schema.FRAMES_FILE).write_bytes(b"gone bad")
    with pytest.raises(GoldDataError, match=rf"{first}/{schema.FRAMES_FILE}: column velocity"):
        load_video(gold, first)
    with pytest.raises(GoldDataError, match=rf"{second}/{schema.SEGMENTS_FILE}: column\(s\) source missing"):
        load_video(gold, second)
    with pytest.raises(GoldDataError, match=rf"Cannot read .*{third}/{schema.FRAMES_FILE}"):
        load_video(gold, third)


# --- value helpers ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        ("1", 1.0, True),
        (1, "1.0", True),
        ("0.1", 0.1, True),
        (" 0.3 ", "0.3", True),
        ("1+3", "1+3", True),
        (None, float("nan"), True),
        (None, "", True),
        ("COMPOUND_A", "COMPOUND_A", True),
        ("1", "1.5", False),
        ("1+3", "1+30", False),
        ("COMPOUND_A", "compound_a", False),  # case variants are the class filter's job (EC-8), not equal here
        (None, "0", False),
    ],
)
def test_same_label_compares_numbers_by_value(left: object, right: object, same: bool) -> None:
    assert same_label(left, right) is same


def test_non_numeric_edit_count_is_an_error(tmp_path: Path) -> None:
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[1]
    set_manifest(result.accepted_dir / victim, edit_count="many")
    with pytest.raises(GoldDataError, match=rf"{victim}/{schema.MANIFEST_FILE}.*edit_count"):
        read_accepted(result.accepted_dir)
