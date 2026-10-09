"""Gold reader, accepted source, set-level rules (plan U3): EC-10, EC-21, EC-26 (fps), EC-27, EC-29.

Per-file checks (EC-1, EC-22, EC-25) are in test_dcs_gold_files.py; the unreviewed prepds source
and `read_gold(settings)` in test_dcs_gold_processed.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Collection, Iterable

import pandas as pd
import pytest
import yaml

from dcs import schema
from dcs.config import DEFAULT_CONFIG_FILE, ConfigError
from dcs.gold import (
    DEFAULT_FPS_TOLERANCE,
    DEFAULT_MARKER,
    DROP_MISSING_FILE,
    DROP_OTHER_PROFILE,
    DROP_UNREADABLE,
    SOURCE_ACCEPTED,
    TRACKER_CLASSICAL,
    TRACKER_MODEL,
    VIDEO_COLUMNS,
    GoldDataError,
    ReadOptions,
    read_accepted,
)
from dcs.synthetic import DEFAULT_PROFILE, MODEL_PROFILE, SynthResult
from dcs.synthetic_config import KNOB_MIXED_PROFILES, KNOB_NULL_DATE
from tests.dcs.gold_helpers import damage_column, dropped_reasons, only, rewrite_table, set_manifest
from tests.dcs.synth_helpers import index, make


# --- a clean set ------------------------------------------------------------------------


def test_clean_set_reads_every_fish(small: SynthResult) -> None:
    gold = read_accepted(small.accepted_dir)
    truth = small.truth.set_index("video_id")
    videos = gold.videos.set_index("video_id")
    assert tuple(gold.videos.columns) == VIDEO_COLUMNS
    assert sorted(videos.index) == sorted(truth.index)
    assert sorted(gold.video_ids) == sorted(truth.index)
    videos = videos.loc[truth.index]
    assert gold.dropped.empty
    assert gold.source == SOURCE_ACCEPTED
    assert (gold.profile, gold.tracker) == (DEFAULT_PROFILE, TRACKER_CLASSICAL)
    assert gold.fps_uniform
    assert videos["reviewed"].all()
    assert (videos["review_status"] == schema.REVIEW_ACCEPTED).all()
    assert (videos["undetermined_share"] == 0).all()
    assert (videos["date"] == truth["date"]).all()
    assert (videos["compound"] == truth["compound"]).all()  # SMALL has no messy labels


def test_tables_are_copies(small: SynthResult) -> None:
    gold = read_accepted(small.accepted_dir)
    videos, dropped = gold.videos, gold.dropped
    videos.drop(videos.index, inplace=True)
    dropped.loc[0] = ["x", "y", "z"]
    assert len(gold.videos) == len(small.truth)
    assert gold.dropped.empty


def test_strip_and_provenance_are_not_required(tmp_path: Path) -> None:
    result = make(tmp_path)
    first = index(result)["video_id"].iloc[0]
    (result.accepted_dir / first / schema.PROVENANCE_FILE).unlink()
    assert not any(result.accepted_dir.glob(f"*/{schema.STRIP_FILE}"))
    assert len(read_accepted(result.accepted_dir).videos) == len(result.truth)


def test_stored_paths_are_ignored_and_the_folder_can_move(tmp_path: Path) -> None:
    """D-012: per-video files are found at <accepted>/<video_id>/ wherever the folder now is."""
    result = make(tmp_path)
    rewrite_table(result.index_path, lambda table: table.assign(frames_path="/nowhere/frames.parquet"))
    moved = tmp_path / "moved"
    shutil.copytree(result.accepted_dir, moved)
    shutil.rmtree(result.accepted_dir)
    gold = read_accepted(moved)
    assert len(gold.videos) == len(result.truth)
    assert gold.folder_of(gold.video_ids[0]).parent == moved


def test_no_usable_fish_is_an_error_with_the_reasons(tmp_path: Path) -> None:
    result = make(tmp_path)
    for frames in result.accepted_dir.glob(f"*/{schema.FRAMES_FILE}"):
        frames.unlink()
    with pytest.raises(GoldDataError, match=rf"No usable fish.*{DROP_MISSING_FILE}"):
        read_accepted(result.accepted_dir)


def test_defaults_match_the_packaged_settings() -> None:
    training = yaml.safe_load(DEFAULT_CONFIG_FILE.read_text(encoding="utf-8"))["training"]
    assert DEFAULT_MARKER == training["model_profile_marker"]
    assert DEFAULT_FPS_TOLERANCE == training["fps_tolerance"]


@pytest.mark.parametrize(
    "options", [{"marker": ""}, {"marker": "  "}, {"fps_tolerance": -0.1}, {"fps_tolerance": float("nan")}]
)
def test_invalid_read_options_are_a_config_error(options: dict[str, object]) -> None:
    with pytest.raises(ConfigError, match="marker|fps_tolerance"):
        ReadOptions(**options)  # type: ignore[arg-type]


# --- EC-10: one row per fish --------------------------------------------------------------


def test_repeated_video_id_is_an_error(tmp_path: Path) -> None:
    result = make(tmp_path)
    repeated = index(result)["video_id"].iloc[2]
    rewrite_table(result.index_path, lambda table: pd.concat([table, table.iloc[[2]]]))
    with pytest.raises(GoldDataError) as caught:
        read_accepted(result.accepted_dir)
    assert repeated in str(caught.value) and "more than once" in str(caught.value)


def test_repeated_subject_under_another_video_id_is_an_error(tmp_path: Path) -> None:
    result = make(tmp_path)
    table = index(result)
    clone = table.iloc[[0]].assign(video_id="X" + table["video_id"].iloc[0])
    rewrite_table(result.index_path, lambda current: pd.concat([current, clone]))
    with pytest.raises(GoldDataError, match="subject .*more than once"):
        read_accepted(result.accepted_dir)


@pytest.mark.parametrize("bad_id", ["../escape", "/abs/path", "a/b"])
def test_video_id_must_be_a_plain_folder_name(tmp_path: Path, bad_id: str) -> None:
    result = make(tmp_path)
    rewrite_table(result.index_path, lambda table: table.assign(video_id=table["video_id"].where(table.index != 0, bad_id)))
    with pytest.raises(GoldDataError, match="folder name"):
        read_accepted(result.accepted_dir)


def test_manifest_naming_another_fish_is_an_error(tmp_path: Path) -> None:
    result = make(tmp_path)
    first, second = index(result)["video_id"].iloc[:2]
    shutil.copy(result.accepted_dir / second / schema.MANIFEST_FILE, result.accepted_dir / first / schema.MANIFEST_FILE)
    with pytest.raises(GoldDataError) as caught:
        read_accepted(result.accepted_dir)
    assert f"describes fish {second}" in str(caught.value) and first in str(caught.value)


def test_index_and_manifest_labels_must_agree(tmp_path: Path) -> None:
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[1]
    set_manifest(result.accepted_dir / victim, compound="COMPOUND_Z")
    with pytest.raises(GoldDataError) as caught:
        read_accepted(result.accepted_dir)
    message = str(caught.value)
    assert victim in message and "compound" in message and "COMPOUND_Z" in message


def test_numeric_dose_in_the_manifest_matches_its_text_in_the_index(tmp_path: Path) -> None:
    """prepds may write a plain dose as a JSON number; the index holds text."""
    result = make(tmp_path)
    table = index(result)
    victim = table.loc[table["concentration_mM"] == "0.3", "video_id"].iloc[0]
    set_manifest(result.accepted_dir / victim, concentration_mM=0.3)
    assert victim in read_accepted(result.accepted_dir).video_ids


def test_manifest_not_accepted_is_an_error(tmp_path: Path) -> None:
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[1]
    set_manifest(result.accepted_dir / victim, review_status=schema.REVIEW_PROCESSED_AUTO)
    with pytest.raises(GoldDataError, match=rf"{victim}.*review status {schema.REVIEW_PROCESSED_AUTO}"):
        read_accepted(result.accepted_dir)


# --- EC-21 / D-003: one tracker and one calibration profile -------------------------------


def test_mixed_profiles_are_an_error_naming_the_versions(tmp_path: Path) -> None:
    result = make(tmp_path, mixed_profiles=True)
    n_model = len(result.targets[KNOB_MIXED_PROFILES])
    with pytest.raises(GoldDataError) as caught:
        read_accepted(result.accepted_dir)
    message = str(caught.value)
    assert f"{MODEL_PROFILE} ({n_model} fish)" in message
    assert f"{DEFAULT_PROFILE} ({len(result.truth) - n_model} fish)" in message


@pytest.mark.parametrize(("profile", "tracker"), [(DEFAULT_PROFILE, TRACKER_CLASSICAL), (MODEL_PROFILE, TRACKER_MODEL)])
def test_selected_profile_keeps_its_fish_and_lists_the_rest(tmp_path: Path, profile: str, tracker: str) -> None:
    result = make(tmp_path, mixed_profiles=True)
    model_fish = set(result.targets[KNOB_MIXED_PROFILES])
    gold = read_accepted(result.accepted_dir, ReadOptions(profile=profile))
    kept = set(gold.video_ids)
    assert (gold.profile, gold.tracker) == (profile, tracker)
    assert kept == (model_fish if profile == MODEL_PROFILE else set(result.truth["video_id"]) - model_fish)
    assert set(dropped_reasons(gold).values()) == {DROP_OTHER_PROFILE}
    assert len(kept) + len(gold.dropped) == len(result.truth)


def test_fish_of_another_profile_never_stops_the_run(tmp_path: Path) -> None:
    """Profile drops come before the data checks."""
    result = make(tmp_path, mixed_profiles=True)
    model_fish = result.targets[KNOB_MIXED_PROFILES][0]
    (result.accepted_dir / model_fish / schema.SEGMENTS_FILE).write_text("start_s\n1\n", encoding="utf-8")
    gold = read_accepted(result.accepted_dir, ReadOptions(profile=DEFAULT_PROFILE))
    assert dropped_reasons(gold)[model_fish] == DROP_OTHER_PROFILE


def test_profile_mix_counts_only_fish_that_are_kept(tmp_path: Path) -> None:
    """A second profile whose every fish is dropped for its data does not make the set 'mixed'."""
    result = make(tmp_path, mixed_profiles=True)
    model_fish = set(result.targets[KNOB_MIXED_PROFILES])
    for video_id in model_fish:
        damage_column(result.accepted_dir / video_id / schema.FRAMES_FILE, "state")
    gold = read_accepted(result.accepted_dir)
    assert gold.profile == DEFAULT_PROFILE
    assert dropped_reasons(gold) == {video_id: DROP_UNREADABLE for video_id in model_fish}


def test_unknown_selected_profile_lists_those_found(small: SynthResult) -> None:
    with pytest.raises(GoldDataError, match=rf"cal-nonexistent.*not found.*{DEFAULT_PROFILE} \("):
        read_accepted(small.accepted_dir, ReadOptions(profile="cal-nonexistent"))


def test_model_marker_is_configurable(small: SynthResult) -> None:
    assert read_accepted(small.accepted_dir, ReadOptions(marker="-synthetic")).tracker == TRACKER_MODEL


@pytest.mark.parametrize("blank", ["", None])
def test_blank_profile_is_an_error(tmp_path: Path, blank: str | None) -> None:
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[2]
    rewrite_table(
        result.index_path,
        lambda table: table.assign(
            calibration_profile_version=table["calibration_profile_version"].where(table["video_id"] != victim, blank)
        ),
    )
    set_manifest(result.accepted_dir / victim, calibration_profile_version=blank)
    with pytest.raises(GoldDataError, match=rf"calibration_profile_version.*{victim}"):
        read_accepted(result.accepted_dir)


# --- EC-26: frame rate uniformity is a flag ----------------------------------------------


def test_mixed_fps_is_flagged_not_an_error(tmp_path: Path) -> None:
    result = make(tmp_path, mixed_fps=True)
    gold = read_accepted(result.accepted_dir)
    assert not gold.fps_uniform
    assert gold.fps_range == (10.0, 15.0)
    assert len(gold.videos) == len(result.truth)
    assert read_accepted(result.accepted_dir, ReadOptions(fps_tolerance=0.5)).fps_uniform  # exactly at the boundary
    assert not read_accepted(result.accepted_dir, ReadOptions(fps_tolerance=0.49)).fps_uniform


def test_tiny_fps_jitter_counts_as_uniform(tmp_path: Path) -> None:
    """prepds measures fps per video (e.g. 29.835 vs 29.821); a relative spread under the tolerance is uniform."""
    result = make(tmp_path)
    jitter = lambda table: table.assign(video_fps=table["video_fps"] * (1 + 0.001 * (table.index % 2)))  # noqa: E731
    rewrite_table(result.index_path, jitter)
    assert read_accepted(result.accepted_dir).fps_uniform


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan")])
def test_fps_must_be_a_positive_number(tmp_path: Path, bad: float) -> None:
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[3]
    rewrite_table(
        result.index_path,
        lambda table: table.assign(video_fps=table["video_fps"].where(table["video_id"] != victim, bad)),
    )
    with pytest.raises(GoldDataError, match=rf"video_fps.*{victim}"):
        read_accepted(result.accepted_dir)


# --- EC-27 / D-005: a kept fish needs a date ------------------------------------------------


def test_null_date_stops_with_the_fix(tmp_path: Path) -> None:
    result = make(tmp_path, null_date=True)
    with pytest.raises(ConfigError, match="prepds export-index") as caught:
        read_accepted(result.accepted_dir)
    assert only(result, KNOB_NULL_DATE) in str(caught.value)


def test_blank_date_stops_with_the_fix(tmp_path: Path) -> None:
    result = make(tmp_path)
    victim = index(result)["video_id"].iloc[0]
    rewrite_table(
        result.index_path,
        lambda table: table.assign(date=table["date"].where(table["video_id"] != victim, "  ")),
    )
    with pytest.raises(ConfigError, match=rf"no date \({victim}\)"):
        read_accepted(result.accepted_dir)


def test_null_date_on_a_dropped_fish_does_not_stop(tmp_path: Path) -> None:
    result = make(tmp_path, null_date=True)
    victim = only(result, KNOB_NULL_DATE)
    (result.accepted_dir / victim / schema.FRAMES_FILE).unlink()
    assert dropped_reasons(read_accepted(result.accepted_dir)) == {victim: DROP_MISSING_FILE}


# --- EC-29: tracker evidence from the prepds output folder ---------------------------------


def _evidence(tmp_path: Path, video_ids: Iterable[str], with_detections: Collection[str] = ()) -> Path:
    """A prepds output folder holding processed/<video_id>/ (and detections.parquet for some)."""
    out = tmp_path / "prepds_out"
    for video_id in video_ids:
        folder = out / schema.PROCESSED_DIR_NAME / video_id
        folder.mkdir(parents=True)
        if video_id in with_detections:
            pd.DataFrame({"score": [0.9]}).to_parquet(folder / schema.DETECTIONS_FILE)
    return out


def test_agreeing_tracker_evidence_is_counted(small: SynthResult, tmp_path: Path) -> None:
    ids = list(small.truth["video_id"])
    gold = read_accepted(small.accepted_dir, processed_dir=_evidence(tmp_path, ids[:5]))
    assert gold.tracker_checked == 5
    assert read_accepted(small.accepted_dir).tracker_checked == 0


def test_model_profile_with_detections_passes(tmp_path: Path) -> None:
    result = make(tmp_path, mixed_profiles=True)
    model_fish = list(result.targets[KNOB_MIXED_PROFILES])
    evidence = _evidence(tmp_path, model_fish, with_detections=model_fish)
    gold = read_accepted(result.accepted_dir, ReadOptions(profile=MODEL_PROFILE), processed_dir=evidence)
    assert (gold.tracker, gold.tracker_checked) == (TRACKER_MODEL, len(model_fish))


def test_detections_beside_a_classical_profile_is_an_error(small: SynthResult, tmp_path: Path) -> None:
    ids = list(small.truth["video_id"])
    with pytest.raises(GoldDataError, match=rf"Tracker evidence.*{ids[2]}"):
        read_accepted(small.accepted_dir, processed_dir=_evidence(tmp_path, ids, with_detections={ids[2]}))


def test_model_profile_without_detections_is_an_error(tmp_path: Path) -> None:
    result = make(tmp_path, mixed_profiles=True)
    model_fish = list(result.targets[KNOB_MIXED_PROFILES])
    with pytest.raises(GoldDataError, match=rf"Tracker evidence.*{model_fish[0]}"):
        read_accepted(
            result.accepted_dir, ReadOptions(profile=MODEL_PROFILE), processed_dir=_evidence(tmp_path, model_fish[:1])
        )
