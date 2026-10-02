"""FR-001..004: workbook cleaning + video matching + exceptions report (T015-T019).

Uses tests/fixtures/synth_db.xlsx (T012) for workbook cases and builds a
throwaway video-folder tree per test for matching cases, mirroring the real
OneDrive layout confirmed in docs/progress.md §0: `{compound}(status
suffix)/{sex}_{subject}.mp4`, with genuinely inconsistent case (F_/f_) and
zero-padding (3 vs 4 digit) - not assumed uniform.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from prepds.catalog import (
    build_exceptions_report,
    dedupe_rows,
    find_duration_mismatches,
    load_workbook,
    match_videos,
    to_trials,
)
from prepds.models import MatchStatus

FIXTURES = Path(__file__).parent / "fixtures"
SYNTH_DB = FIXTURES / "synth_db.xlsx"
SYNTH_VIDEO = FIXTURES / "synth_tiny.mp4"  # 5.0s real playable video, for duration tests


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")  # matching is filename-based only; content unused here


# --- T015: blank rows excluded -----------------------------------------------


def test_excludes_blank_rows() -> None:
    df = load_workbook(SYNTH_DB)
    # synth_db.xlsx has 10 rows, 2 with a blank Compund: (subjects 100, 88).
    assert len(df) == 8
    assert df["Subject #:"].tolist() == [22, 45, 320, 320, 55, 66, 77, 99]


# --- T016: Body Tissue column excluded ---------------------------------------


def test_excludes_body_tissue_column() -> None:
    df = load_workbook(SYNTH_DB)
    assert "Body Tissue:" not in df.columns


# --- T017: normalized matching (case / zero-pad / suffix / '+' variants) ----


def test_normalized_matching(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    # Real-world-shaped folder names: parenthetical status suffix (C12),
    # inconsistent case and zero-padding (C11), '+' combo-treatment spacing.
    _touch(video_dir / "GT-1-42(DONE Compressed Only)" / "F_0022.mp4")
    _touch(video_dir / "FD-2-66 + methylone(DONE Compressed Only)" / "m_045.mp4")  # subject 45 is sex M
    _touch(video_dir / "Veh(DONE Compressed Only, Missing)" / "F_055.mp4")

    df = load_workbook(SYNTH_DB)
    deduped, _dup_groups = dedupe_rows(df)
    trials = to_trials(deduped)
    matched, _unmatched_videos, _ambiguous = match_videos(trials, video_dir)

    by_subject = {t.subject_id: t for t in matched}
    assert by_subject["0022"].match_status == MatchStatus.MATCHED
    assert by_subject["0022"].video_path == video_dir / "GT-1-42(DONE Compressed Only)" / "F_0022.mp4"
    # combo-treatment name with '+' and lowercase f_/3-digit pad
    assert by_subject["0045"].match_status == MatchStatus.MATCHED
    # "Missing" is part of the status suffix, not the compound - must still match
    assert by_subject["0055"].match_status == MatchStatus.MATCHED


# --- T018: duplicate row detected and flagged --------------------------------


def test_duplicate_row_detected_and_flagged() -> None:
    df = load_workbook(SYNTH_DB)
    deduped, duplicate_groups = dedupe_rows(df)
    # Subject 320 appears twice, fully identical -> collapses to one logical row.
    assert (deduped["Subject #:"] == 320).sum() == 1
    assert len(deduped) == 7  # 8 real rows - 1 collapsed duplicate
    assert len(duplicate_groups) == 1
    assert duplicate_groups[0]["subject_id"] == "0320"
    assert len(duplicate_groups[0]["row_indices"]) == 2


# --- T019: unmatched trial and unmatched video reported ----------------------


def test_unmatched_trial_and_unmatched_video_reported(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    _touch(video_dir / "GT-1-42(DONE Compressed Only)" / "F_0022.mp4")
    # An extra video with no corresponding trial row.
    _touch(video_dir / "GT-1-42(DONE Compressed Only)" / "F_9999.mp4")

    df = load_workbook(SYNTH_DB)
    deduped, duplicate_groups = dedupe_rows(df)
    trials = to_trials(deduped)
    matched, unmatched_videos, _ambiguous = match_videos(trials, video_dir)

    report = build_exceptions_report(
        trials=matched,
        duplicate_groups=duplicate_groups,
        unmatched_videos=unmatched_videos,
        duration_mismatches=[],
        corrupt_videos=[],
    )
    # Subject 320 (deduped) and every other subject besides 0022 have no video.
    unmatched_subject_ids = {row["subject_id"] for row in report.unmatched_trials}
    assert "0320" in unmatched_subject_ids
    assert "0045" in unmatched_subject_ids
    assert len(report.unmatched_videos) == 1
    assert report.unmatched_videos[0]["path"].endswith("F_9999.mp4")


# --- extra coverage: FR-004 duration mismatch + corrupt-file handling --------


def test_find_duration_mismatches_flags_deviation_beyond_tolerance(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    matched_path = video_dir / "GT-1-42(DONE Compressed Only)" / "F_0022.mp4"
    matched_path.parent.mkdir(parents=True)
    shutil.copy(SYNTH_VIDEO, matched_path)  # real 5.0s video

    df = load_workbook(SYNTH_DB)
    deduped, _ = dedupe_rows(df)
    trials = to_trials(deduped)
    matched, _, _ = match_videos(trials, video_dir)
    trial_0022 = next(t for t in matched if t.subject_id == "0022")
    assert trial_0022.match_status == MatchStatus.MATCHED
    assert trial_0022.agent_exposure_min == 20  # expects 1200s, video is 5.0s

    mismatches, corrupt = find_duration_mismatches([trial_0022], tolerance_s=30.0)
    assert corrupt == []
    assert len(mismatches) == 1
    assert mismatches[0]["subject_id"] == "0022"
    assert mismatches[0]["expected_s"] == pytest.approx(1200.0)
    assert mismatches[0]["measured_s"] == pytest.approx(5.0, abs=0.2)


def test_find_duration_mismatches_corrupt_file_reported_not_raised(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    corrupt_path = video_dir / "GT-1-42(DONE Compressed Only)" / "F_0022.mp4"
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"")  # zero-byte

    df = load_workbook(SYNTH_DB)
    deduped, _ = dedupe_rows(df)
    trials = to_trials(deduped)
    matched, _, _ = match_videos(trials, video_dir)
    trial_0022 = next(t for t in matched if t.subject_id == "0022")

    mismatches, corrupt = find_duration_mismatches([trial_0022], tolerance_s=30.0)
    assert mismatches == []
    assert len(corrupt) == 1
    assert corrupt[0]["subject_id"] == "0022"


# --- T022: `prepds catalog` CLI subcommand -----------------------------------


def test_catalog_cli_writes_trials_catalog_and_exceptions_report(tmp_path: Path, monkeypatch) -> None:
    from prepds.cli import main

    video_dir = tmp_path / "videos"
    _touch(video_dir / "GT-1-42(DONE Compressed Only)" / "F_0022.mp4")
    _touch(video_dir / "GT-1-42(DONE Compressed Only)" / "F_9999.mp4")  # unmatched video
    output_dir = tmp_path / "outputs"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PDS_VIDEO_DIR", str(video_dir))
    monkeypatch.setenv("PDS_DB_PATH", str(SYNTH_DB))
    monkeypatch.setenv("PDS_OUTPUT_DIR", str(output_dir))

    exit_code = main(["catalog"])
    assert exit_code == 0

    catalog_path = output_dir / "trials_catalog.parquet"
    exceptions_path = output_dir / "exceptions_report.json"
    assert catalog_path.is_file()
    assert exceptions_path.is_file()

    catalog_df = pd.read_parquet(catalog_path)
    assert len(catalog_df) == 7  # 8 real rows - 1 deduped Subject #320
    assert (catalog_df["subject_id"] == "0022").sum() == 1
    matched_row = catalog_df[catalog_df["subject_id"] == "0022"].iloc[0]
    assert matched_row["match_status"] == "matched"

    report = json.loads(exceptions_path.read_text())
    assert any(v["path"].endswith("F_9999.mp4") for v in report["unmatched_videos"])
    assert any(d["subject_id"] == "0320" for d in report["duplicate_trials"])


# --- code-review follow-ups: case-insensitive extension, key collisions, NaN --


def test_match_videos_finds_uppercase_mp4_extension(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    # glob("*.mp4") alone is case-sensitive on Linux and would miss this even
    # though VIDEO_FILENAME_RE already tolerates letter-case via IGNORECASE.
    _touch(video_dir / "GT-1-42(DONE Compressed Only)" / "F_0022.MP4")

    df = load_workbook(SYNTH_DB)
    deduped, _ = dedupe_rows(df)
    trials = to_trials(deduped)
    matched, unmatched_videos, _ambiguous = match_videos(trials, video_dir)

    by_subject = {t.subject_id: t for t in matched}
    assert by_subject["0022"].match_status == MatchStatus.MATCHED
    assert unmatched_videos == []


def test_match_videos_reports_duplicate_video_files_for_same_key(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    # Two distinct files that both normalize to the same (compound, subject)
    # key - must not silently pick one and hide the other.
    _touch(video_dir / "GT-1-42(DONE Compressed Only)" / "F_0022.mp4")
    _touch(video_dir / "GT-1-42(DONE Compressed Only)" / "F_022.mp4")

    df = load_workbook(SYNTH_DB)
    deduped, _ = dedupe_rows(df)
    trials = to_trials(deduped)
    _matched, _unmatched, ambiguous = match_videos(trials, video_dir)

    assert len(ambiguous) == 1
    assert ambiguous[0]["kind"] == "duplicate_video_files"
    assert len(ambiguous[0]["video_paths"]) == 2


def test_match_videos_reports_duplicate_trial_key_and_does_not_double_assign(tmp_path: Path) -> None:
    """Two non-identical trial rows sharing a (compound, subject) key.

    dedupe_rows() only collapses EXACT full-row duplicates (C13); a future
    workbook update could introduce two genuinely different rows (e.g. a
    re-test entered with a different date) that still share a match key.
    Constructed directly here since to_trials()/dedupe_rows() would never
    produce this from today's real data (docs/progress.md §2 notes this is a
    defensive check, not a currently-observed case).
    """
    import datetime as dt

    from prepds.models import MatchStatus as _MatchStatus
    from prepds.models import Trial

    video_dir = tmp_path / "videos"
    _touch(video_dir / "GT-1-42(DONE Compressed Only)" / "F_0022.mp4")

    trial_a = Trial(
        subject_id="0022",
        sex="F",
        strain="Casper (roya9; mitfaw2)",
        age=4.0,
        compound="GT-1-42",
        concentration_mM="0.03",
        date=dt.date(2026, 1, 1),
        agent_exposure_min=20.0,
        video_path=None,
        match_status=MatchStatus.NO_VIDEO,
    )
    # Same subject/compound key, but a genuinely different row (different
    # date/age) - not an exact duplicate, so dedupe_rows() would not merge it.
    trial_b = dataclasses.replace(trial_a, date=dt.date(2026, 3, 1), age=5.0)

    matched, _unmatched, ambiguous = match_videos([trial_a, trial_b], video_dir)

    assert len(ambiguous) == 1
    assert ambiguous[0]["kind"] == "duplicate_trial_key"
    assert set(ambiguous[0]["subject_ids"]) == {"0022"}
    # Exactly one of the two claims the video; the other is left NO_VIDEO -
    # never both silently marked MATCHED to the same file.
    statuses = sorted(t.match_status.value for t in matched)
    assert statuses == ["matched", "no_video"]


def test_find_duration_mismatches_reports_nan_agent_exposure_instead_of_silently_passing(
    tmp_path: Path,
) -> None:
    from prepds.models import MatchStatus as _MatchStatus
    from prepds.models import Trial

    video_dir = tmp_path / "videos"
    video_path = video_dir / "GT-1-42(DONE Compressed Only)" / "F_0022.mp4"
    video_path.parent.mkdir(parents=True)
    shutil.copy(SYNTH_VIDEO, video_path)

    trial_with_nan_exposure = Trial(
        subject_id="0022",
        sex="F",
        strain="Casper (roya9; mitfaw2)",
        age=None,
        compound="GT-1-42",
        concentration_mM="0.03",
        date=__import__("datetime").date(2026, 1, 1),
        agent_exposure_min=float("nan"),
        video_path=video_path,
        match_status=_MatchStatus.MATCHED,
    )

    mismatches, corrupt = find_duration_mismatches([trial_with_nan_exposure], tolerance_s=30.0)
    assert corrupt == []
    # Must be reported, not silently treated as "within tolerance" (NaN
    # comparisons are always False in Python).
    assert len(mismatches) == 1
    assert mismatches[0]["subject_id"] == "0022"
    assert "error" in mismatches[0]


def test_match_videos_finds_compound_folders_nested_under_phase_folders(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    _touch(video_dir / "Phase_1" / "GT-1-42(DONE Compressed Only)" / "F_0022.mp4")

    df = load_workbook(SYNTH_DB)
    deduped, _ = dedupe_rows(df)
    matched, unmatched_videos, _ambiguous = match_videos(to_trials(deduped), video_dir)

    assert {t.subject_id: t for t in matched}["0022"].match_status == MatchStatus.MATCHED
    assert unmatched_videos == []
