"""Gold reader, unreviewed prepds source (plan U3, D-033) and `read_gold(settings)`.

Owner decision 2026-10-03: until videos are Accepted, train on the unreviewed pipeline output
(`<prepds output>/processed/<video_id>/` + `trials_catalog.parquet`); every row is marked unreviewed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from dcs import schema
from dcs.config import ConfigError, Settings, load_settings
from dcs.gold import (
    DROP_MISSING_FILE,
    DROP_STATUS,
    SOURCE_ACCEPTED,
    SOURCE_PROCESSED,
    TRACKER_MODEL,
    VIDEO_COLUMNS,
    GoldDataError,
    load_video,
    read_gold,
    read_processed,
)
from dcs.synthetic import SynthResult
from dcs.synthetic_config import KNOB_UNDETERMINED
from tests.dcs.gold_helpers import dropped_reasons, rewrite_table, set_manifest, to_processed
from tests.dcs.synth_helpers import make


@pytest.fixture(scope="module")
def pds(small: SynthResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The SMALL set in the unreviewed prepds layout; tests only read it (mutating tests use `_fresh`)."""
    return to_processed(small, tmp_path_factory.mktemp("pds") / "outputs")


def _fresh(tmp_path: Path, **changes: Any) -> tuple[SynthResult, Path]:
    result = make(tmp_path, **changes)
    return result, to_processed(result, tmp_path / "outputs")


def _folder(out: Path, video_id: str) -> Path:
    return out / schema.PROCESSED_DIR_NAME / video_id


# --- a clean unreviewed set ------------------------------------------------------------------


def test_unreviewed_set_reads_every_fish(small: SynthResult, pds: Path) -> None:
    gold = read_processed(pds)
    truth = small.truth.set_index("video_id")
    videos = gold.videos.set_index("video_id")
    assert tuple(gold.videos.columns) == VIDEO_COLUMNS
    assert gold.source == SOURCE_PROCESSED
    assert sorted(videos.index) == sorted(truth.index)
    videos = videos.loc[truth.index]  # folder order differs from the design order
    assert not videos["reviewed"].any()
    assert (videos["review_status"] == schema.REVIEW_PROCESSED_AUTO).all()
    assert (videos["date"] == truth["date"]).all()  # date comes from the catalog
    assert (videos["compound"] == truth["compound"]).all()
    assert gold.dropped.empty
    assert gold.folder_of(truth.index[0]) == _folder(pds, truth.index[0])


def test_catalog_rows_without_a_video_and_stray_files_are_ignored(tmp_path: Path) -> None:
    result, out = _fresh(tmp_path)
    extra = pd.read_parquet(out / schema.CATALOG_FILE).iloc[[0]].assign(subject_id="9998", match_status="no_video")
    rewrite_table(out / schema.CATALOG_FILE, lambda table: pd.concat([table, extra]))
    (out / schema.PROCESSED_DIR_NAME / "notes.txt").write_text("not a video", encoding="utf-8")
    gold = read_processed(out)
    assert len(gold.videos) == len(result.truth) and gold.dropped.empty


def test_matched_trial_without_a_processed_folder_is_listed(tmp_path: Path) -> None:
    """EC-1: a trial matched to a video but not run (or deleted since) must not vanish."""
    result, out = _fresh(tmp_path)
    victim = result.truth["video_id"].iloc[6]
    shutil.rmtree(_folder(out, victim))
    gold = read_processed(out)
    assert dropped_reasons(gold) == {victim: DROP_MISSING_FILE}
    assert "not run" in gold.dropped["detail"].iloc[0]


# --- EC-22 relaxed: Undetermined is expected before review -----------------------------------


def test_undetermined_is_allowed_and_measured(tmp_path: Path) -> None:
    result, out = _fresh(tmp_path, undetermined=True)
    (victim,) = result.targets[KNOB_UNDETERMINED]
    gold = read_processed(out)
    shares = gold.videos.set_index("video_id")["undetermined_share"]
    assert 0 < shares[victim] < 1
    assert (shares.drop(victim) == 0).all()
    assert (load_video(gold, victim).frames["state"] == schema.UNDETERMINED).any()


# --- review status ---------------------------------------------------------------------------


def test_review_status_decides_kept_dropped_and_reviewed(tmp_path: Path) -> None:
    result, out = _fresh(tmp_path)
    ids = list(result.truth["video_id"])
    statuses = {
        ids[0]: schema.REVIEW_REJECTED,
        ids[1]: schema.REVIEW_NOT_PROCESSED,
        ids[2]: schema.REVIEW_ACCEPTED,
        ids[3]: schema.REVIEW_EDITED,
    }
    for video_id, status in statuses.items():
        set_manifest(_folder(out, video_id), review_status=status)
    gold = read_processed(out)
    assert dropped_reasons(gold) == {ids[0]: DROP_STATUS, ids[1]: DROP_STATUS}
    reviewed = gold.videos.set_index("video_id")["reviewed"]
    assert reviewed[ids[2]] and not reviewed[ids[3]]
    assert reviewed.sum() == 1


def test_rejected_fish_with_broken_files_is_dropped_not_fatal(tmp_path: Path) -> None:
    """Status drops come before the data checks."""
    result, out = _fresh(tmp_path)
    victim = result.truth["video_id"].iloc[0]
    set_manifest(_folder(out, victim), review_status=schema.REVIEW_REJECTED)
    (_folder(out, victim) / schema.SEGMENTS_FILE).write_text("start_s\n1\n", encoding="utf-8")
    assert dropped_reasons(read_processed(out)) == {victim: DROP_STATUS}


@pytest.mark.parametrize("status", [schema.REVIEW_REJECTED, schema.REVIEW_NOT_PROCESSED])
def test_sparse_folder_of_a_dropped_status_is_dropped_by_status(tmp_path: Path, status: str) -> None:
    """Only identity and status are needed to drop a fish; its other keys and data files are not required."""
    result, out = _fresh(tmp_path)
    victim = result.truth.iloc[1]
    folder = _folder(out, victim["video_id"])
    shutil.rmtree(folder)
    folder.mkdir()
    sparse = {"sex": victim["sex"], "subject_id": victim["subject_id"], "review_status": status}
    (folder / schema.MANIFEST_FILE).write_text(json.dumps(sparse), encoding="utf-8")
    assert dropped_reasons(read_processed(out)) == {victim["video_id"]: DROP_STATUS}


def test_unknown_review_status_is_an_error(tmp_path: Path) -> None:
    result, out = _fresh(tmp_path)
    victim = result.truth["video_id"].iloc[0]
    set_manifest(_folder(out, victim), review_status="MAYBE")
    with pytest.raises(GoldDataError, match=rf"{victim}/{schema.MANIFEST_FILE}: unknown review status 'MAYBE'"):
        read_processed(out)


# --- catalog join -----------------------------------------------------------------------------


def test_video_without_catalog_row_stops_with_the_fix(tmp_path: Path) -> None:
    result, out = _fresh(tmp_path)
    victim = result.truth.iloc[4]
    rewrite_table(out / schema.CATALOG_FILE, lambda table: table[table["subject_id"] != victim["subject_id"]])
    with pytest.raises(ConfigError, match="prepds catalog") as caught:
        read_processed(out)
    assert victim["video_id"] in str(caught.value)


def test_repeated_catalog_subject_is_an_error(tmp_path: Path) -> None:
    _, out = _fresh(tmp_path)
    rewrite_table(out / schema.CATALOG_FILE, lambda table: pd.concat([table, table.iloc[[1]]]))
    with pytest.raises(GoldDataError, match=rf"{schema.CATALOG_FILE}: subject .*more than once"):
        read_processed(out)


@pytest.mark.parametrize("column", ["sex", "subject_id"])
def test_blank_catalog_key_is_an_error(tmp_path: Path, column: str) -> None:
    _, out = _fresh(tmp_path)
    rewrite_table(
        out / schema.CATALOG_FILE,
        lambda table: table.assign(**{column: table[column].where(table.index != 2, None)}),
    )
    with pytest.raises(GoldDataError, match=rf"{schema.CATALOG_FILE}: blank {column}"):
        read_processed(out)


def test_manifest_and_catalog_labels_must_agree(tmp_path: Path) -> None:
    result, out = _fresh(tmp_path)
    victim = result.truth["video_id"].iloc[2]
    set_manifest(_folder(out, victim), concentration_mM="999")
    with pytest.raises(GoldDataError, match=rf"{victim}: concentration_mM is '999' in {schema.MANIFEST_FILE}"):
        read_processed(out)


def test_folder_named_after_another_fish_is_an_error(tmp_path: Path) -> None:
    result, out = _fresh(tmp_path)
    victim = result.truth["video_id"].iloc[0]
    shutil.move(_folder(out, victim), _folder(out, "M_9997"))
    with pytest.raises(GoldDataError, match=rf"M_9997/{schema.MANIFEST_FILE} describes fish {victim}"):
        read_processed(out)


def test_missing_frames_drops_the_fish(tmp_path: Path) -> None:
    result, out = _fresh(tmp_path)
    victim = result.truth["video_id"].iloc[5]
    (_folder(out, victim) / schema.FRAMES_FILE).unlink()
    assert dropped_reasons(read_processed(out)) == {victim: DROP_MISSING_FILE}


@pytest.mark.parametrize("remove", ["catalog", "processed"])
def test_missing_catalog_or_processed_folder_is_a_config_error(tmp_path: Path, remove: str) -> None:
    _, out = _fresh(tmp_path)
    if remove == "catalog":
        (out / schema.CATALOG_FILE).unlink()
    else:
        shutil.rmtree(out / schema.PROCESSED_DIR_NAME)
    with pytest.raises(ConfigError, match="prepds catalog" if remove == "catalog" else "prepds run"):
        read_processed(out)


def test_missing_catalog_column_names_file_and_column(tmp_path: Path) -> None:
    _, out = _fresh(tmp_path)
    rewrite_table(out / schema.CATALOG_FILE, lambda table: table.drop(columns=["date"]))
    with pytest.raises(GoldDataError, match=rf"{schema.CATALOG_FILE}: column\(s\) date missing"):
        read_processed(out)


# --- EC-29 in the processed folder itself ----------------------------------------------------


def test_detections_beside_a_classical_profile_is_an_error(tmp_path: Path) -> None:
    result, out = _fresh(tmp_path)
    victim = result.truth["video_id"].iloc[1]
    pd.DataFrame({"score": [0.5]}).to_parquet(_folder(out, victim) / schema.DETECTIONS_FILE)
    with pytest.raises(GoldDataError, match=rf"Tracker evidence.*{victim}"):
        read_processed(out)


def test_every_processed_fish_has_tracker_evidence(pds: Path, small: SynthResult) -> None:
    assert read_processed(pds).tracker_checked == len(small.truth)


# --- read_gold(settings) picks the source ------------------------------------------------------


def _settings(tmp_path: Path, environ: dict[str, str], **training: Any) -> Settings:
    config = tmp_path / "override.yaml"
    config.write_text(yaml.safe_dump({"training": training}), encoding="utf-8")
    return load_settings(environ=environ, config_file=config)


def test_read_gold_uses_the_configured_source(small: SynthResult, pds: Path, tmp_path: Path) -> None:
    accepted = _settings(tmp_path, {"DCS_ACCEPTED_DIR": str(small.accepted_dir)}, gold_source="accepted")
    processed = _settings(tmp_path, {"DCS_PROCESSED_DIR": str(pds)}, gold_source="processed")
    assert read_gold(accepted).source == SOURCE_ACCEPTED
    assert read_gold(processed).source == SOURCE_PROCESSED


def test_read_gold_passes_marker_and_tolerance(tmp_path: Path) -> None:
    result = make(tmp_path, mixed_fps=True)
    environ = {"DCS_ACCEPTED_DIR": str(result.accepted_dir)}
    strict = _settings(tmp_path, environ, gold_source="accepted", model_profile_marker="-synthetic")
    loose = _settings(tmp_path, environ, gold_source="accepted", fps_tolerance=0.5)
    assert read_gold(strict).tracker == TRACKER_MODEL and not read_gold(strict).fps_uniform
    assert read_gold(loose).fps_uniform


def test_read_gold_passes_the_selected_profile(small: SynthResult, tmp_path: Path) -> None:
    settings = _settings(tmp_path, {"DCS_ACCEPTED_DIR": str(small.accepted_dir)}, gold_source="accepted")
    with pytest.raises(GoldDataError, match="cal-other.*not found"):
        read_gold(settings, profile="cal-other")


def test_processed_source_needs_dcs_processed_dir(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="DCS_PROCESSED_DIR"):
        read_gold(_settings(tmp_path, {}, gold_source="processed"))


def test_default_source_is_the_unreviewed_output() -> None:
    """D-033: owner, 2026-10-03; switch to `accepted` once videos are Accepted."""
    assert load_settings(environ={}).training["gold_source"] == SOURCE_PROCESSED
