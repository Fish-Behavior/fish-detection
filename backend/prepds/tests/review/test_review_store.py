"""Review state machine, edits, accept/reject and the accepted index (T061-T065) - FR-012..FR-016.

FR-015 and FR-015a are deliberately ASYMMETRIC: a strip with any Undetermined segment can never be
accepted (hard block, `force` is ignored), while a strip with Dead is accepted only after an explicit
confirmation (`force=True`) - Dead is a real label, just a higher-stakes one.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from prepds.export import export_video
from prepds.features import derive_features
from prepds.models import (
    BehaviorState as B,
    FrameSource,
    ManifestRecord,
    MatchStatus,
    ReviewStatus as S,
    StateFrame,
    Track,
    Trial,
    VideoAsset,
)
from prepds.review_store import (
    AcceptBlocked,
    ConfirmationRequired,
    InvalidTransition,
    accept,
    load_manifest,
    reject,
    save_edit,
    should_process,
    transition_allowed,
)
from prepds.segments import DeadMonotonicityError

FPS = 30.0
N = 300  # 10 s
NOW = dt.datetime(2026, 9, 23, 14, 0, tzinfo=dt.timezone.utc)


def _trial(subject="0332") -> Trial:
    return Trial(subject, "F", "Casper", None, "Fentanyl", "0.03", dt.date(2026, 3, 1), 20.0,
                 Path(f"videos/F_{subject}.mp4"), MatchStatus.MATCHED)


def _make(root: Path, plan: list[B] | None = None, subject="0332") -> Path:
    tracks = [Track(i, i / FPS, i * 2.0, 100.0, 90.0, 100.0, True) for i in range(N)]
    plan = plan or ([B.CONTROLLED_SWIM] * 100 + [B.FREEZING_DRIFT] * 100 + [B.ERRATIC_MOVEMENT] * 100)
    states = [StateFrame(i, i / FPS, plan[i], FrameSource.AUTO, None) for i in range(N)]
    asset = VideoAsset(Path(f"videos/F_{subject}.mp4"), N / FPS, FPS, N, (192, 240))
    out = root / "processed" / f"F_{subject}"
    export_video(out, trial=_trial(subject), asset=asset, tracks=tracks, features=derive_features(tracks),
                 states=states, pipeline_version="0.1.0", calibration_profile_version="cal-2026-09-23", processed_at=NOW)
    return out


def _states(video_dir: Path) -> list[str]:
    return list(pd.read_parquet(video_dir / "frames.parquet")["state"].astype(str))


# --- T061: status transitions ------------------------------------------------------


@pytest.mark.parametrize(
    "current,target,allowed",
    [
        (S.NOT_PROCESSED, S.PROCESSED_AUTO, True),
        (S.PROCESSED_AUTO, S.EDITED, True),
        (S.PROCESSED_AUTO, S.ACCEPTED, True),
        (S.EDITED, S.EDITED, True),
        (S.EDITED, S.ACCEPTED, True),
        (S.PROCESSED_AUTO, S.REJECTED, True),
        (S.EDITED, S.REJECTED, True),
        (S.REJECTED, S.PROCESSED_AUTO, True),
        (S.NOT_PROCESSED, S.ACCEPTED, False),
        (S.NOT_PROCESSED, S.EDITED, False),
        (S.ACCEPTED, S.EDITED, False),
        (S.ACCEPTED, S.REJECTED, False),
        (S.REJECTED, S.ACCEPTED, False),
        (S.REJECTED, S.EDITED, False),
    ],
)
def test_status_transitions_valid_only(current: S, target: S, allowed: bool) -> None:
    assert transition_allowed(current, target) is allowed


def test_editing_an_accepted_video_is_an_invalid_transition(tmp_path: Path) -> None:
    video = _make(tmp_path)
    accept(video, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW)
    with pytest.raises(InvalidTransition):
        save_edit(video, [(0.0, 1.0, B.DEAD)], reviewer="lk", now=NOW)
    with pytest.raises(InvalidTransition):
        accept(video, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW)


# --- FR-012: frame-accurate edits, saved without accepting --------------------------


def test_save_edit_relabels_the_range_frame_accurately_and_marks_it_manual(tmp_path: Path) -> None:
    video = _make(tmp_path)
    save_edit(video, [(4.0, 6.0, B.LISTING_LORR)], reviewer="lk", now=NOW)

    frames = pd.read_parquet(video / "frames.parquet")
    edited = frames[frames["state"].astype(str) == "Listing/LORR"]
    assert list(edited["frame_idx"]) == list(range(120, 180))  # [4 s, 6 s) at 30 fps, end exclusive
    assert set(edited["source"].astype(str)) == {"manual"}
    assert set(frames.drop(edited.index)["source"].astype(str)) == {"auto"}

    manifest = load_manifest(video)
    assert manifest.review_status == S.EDITED and manifest.edited is True and manifest.edit_count == 1
    segments = pd.read_csv(video / "segments.csv")
    assert "Listing/LORR" in set(segments["state"])  # segments regenerated from the edited frames


def test_each_save_counts_as_one_edit(tmp_path: Path) -> None:
    video = _make(tmp_path)
    save_edit(video, [(0.0, 1.0, B.SURFACE_BREACH)], reviewer="lk", now=NOW)
    save_edit(video, [(1.0, 2.0, B.SURFACE_BREACH)], reviewer="lk", now=NOW)
    assert load_manifest(video).edit_count == 2


def test_an_edit_that_would_break_dead_monotonicity_is_rejected_and_writes_nothing(tmp_path: Path) -> None:
    video = _make(tmp_path)
    before = _states(video)
    with pytest.raises(DeadMonotonicityError):
        save_edit(video, [(2.0, 3.0, B.DEAD)], reviewer="lk", now=NOW)  # Dead followed by non-Dead
    assert _states(video) == before
    assert load_manifest(video).review_status == S.PROCESSED_AUTO and load_manifest(video).edit_count == 0


def test_an_edit_outside_the_video_is_rejected(tmp_path: Path) -> None:
    video = _make(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        save_edit(video, [(5.0, 99.0, B.SURFACE_BREACH)], reviewer="lk", now=NOW)
    with pytest.raises(ValueError, match="start"):
        save_edit(video, [(5.0, 5.0, B.SURFACE_BREACH)], reviewer="lk", now=NOW)


def test_undetermined_is_an_assignable_label_in_the_editor(tmp_path: Path) -> None:
    video = _make(tmp_path)
    save_edit(video, [(1.0, 2.0, B.UNDETERMINED)], reviewer="lk", now=NOW)
    assert "Undetermined" in set(_states(video))


# --- T063 / FR-015: Undetermined is a hard block --------------------------------------


def test_accept_blocks_on_undetermined_without_force(tmp_path: Path) -> None:
    plan = [B.UNDETERMINED] * 30 + [B.CONTROLLED_SWIM] * (N - 30)
    video = _make(tmp_path, plan)
    with pytest.raises(AcceptBlocked):
        accept(video, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW)
    assert load_manifest(video).review_status == S.PROCESSED_AUTO
    assert not (tmp_path / "accepted").exists()


def test_force_never_overrides_an_undetermined_block(tmp_path: Path) -> None:
    plan = [B.UNDETERMINED] * 30 + [B.CONTROLLED_SWIM] * (N - 30)
    video = _make(tmp_path, plan)
    with pytest.raises(AcceptBlocked):
        accept(video, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW, force=True)


# --- FR-015a: Dead is a soft, overridable warning ----------------------------------------


def test_accept_with_dead_needs_confirmation_and_force_allows_it(tmp_path: Path) -> None:
    plan = [B.CONTROLLED_SWIM] * 200 + [B.DEAD] * 100
    video = _make(tmp_path, plan)
    with pytest.raises(ConfirmationRequired):
        accept(video, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW)
    assert load_manifest(video).review_status == S.PROCESSED_AUTO

    accept(video, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW, force=True)
    assert load_manifest(video).review_status == S.ACCEPTED


def test_undetermined_plus_dead_is_still_a_hard_block_even_with_force(tmp_path: Path) -> None:
    plan = [B.UNDETERMINED] * 10 + [B.CONTROLLED_SWIM] * 190 + [B.DEAD] * 100
    video = _make(tmp_path, plan)
    with pytest.raises(AcceptBlocked):
        accept(video, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW, force=True)


# --- FR-014: accept writes the gold copy, provenance and the accepted index ------------------


def test_accept_writes_artifacts_provenance_and_index_row(tmp_path: Path) -> None:
    video = _make(tmp_path)
    gold = tmp_path / "accepted"
    accept(video, accepted_dir=gold, reviewer="lk", now=NOW, trial=_trial())

    copy = gold / "F_0332"
    assert {p.name for p in copy.iterdir()} >= {"frames.parquet", "segments.csv", "strip.png", "manifest.json", "provenance.json"}
    provenance = json.loads((copy / "provenance.json").read_text())
    assert provenance["reviewer"] == "lk" and provenance["source"] == "auto"
    assert provenance["calibration_profile_version"] == "cal-2026-09-23"

    manifest = ManifestRecord.from_dict(json.loads((copy / "manifest.json").read_text()))
    assert manifest.review_status == S.ACCEPTED and manifest.reviewer == "lk" and manifest.reviewed_at == NOW

    index = pd.read_parquet(gold / "accepted_index.parquet")
    assert len(index) == 1
    row = index.iloc[0]
    assert row["subject_id"] == "0332" and row["compound"] == "Fentanyl" and row["strain"] == "Casper"
    assert row["provenance"] == "auto" and row["strip_path"].endswith("strip.png")


def test_edited_then_accepted_is_recorded_as_manual(tmp_path: Path) -> None:
    video = _make(tmp_path)
    save_edit(video, [(1.0, 2.0, B.SURFACE_BREACH)], reviewer="lk", now=NOW)
    accept(video, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW)
    provenance = json.loads((tmp_path / "accepted" / "F_0332" / "provenance.json").read_text())
    assert provenance["source"] == "manual"


def test_accepted_index_has_one_row_per_video_and_grows(tmp_path: Path) -> None:
    gold = tmp_path / "accepted"
    accept(_make(tmp_path, subject="0332"), accepted_dir=gold, reviewer="lk", now=NOW)
    accept(_make(tmp_path, subject="0333"), accepted_dir=gold, reviewer="lk", now=NOW)
    index = pd.read_parquet(gold / "accepted_index.parquet")
    assert sorted(index["subject_id"]) == ["0332", "0333"]


# --- T062: reject clears edits and re-queues ----------------------------------------------


def test_reject_clears_edits_and_requeues(tmp_path: Path) -> None:
    video = _make(tmp_path)
    save_edit(video, [(1.0, 2.0, B.SURFACE_BREACH)], reviewer="lk", now=NOW)
    reject(video, reviewer="lk", now=NOW)

    manifest = load_manifest(video)
    assert manifest.review_status == S.REJECTED
    assert manifest.edited is False and manifest.edit_count == 0
    assert not (video / "frames.parquet").exists() and not (video / "segments.csv").exists()
    assert should_process(manifest.review_status, force=False) is True


# --- T065 / FR-016: batch re-run is idempotent and resumable ---------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [(S.NOT_PROCESSED, True), (S.REJECTED, True), (S.PROCESSED_AUTO, False), (S.EDITED, False), (S.ACCEPTED, False)],
)
def test_batch_rerun_skips_accepted_videos(status: S, expected: bool) -> None:
    assert should_process(status, force=False) is expected


def test_force_reprocesses_anything_but_the_caller_must_ask_for_it() -> None:
    assert should_process(S.ACCEPTED, force=True) is True
    assert should_process(S.EDITED, force=True) is True


# --- review findings: atomicity, boundaries, input validation ------------------------------


def test_a_failed_index_write_leaves_the_video_acceptable_so_a_retry_works(tmp_path: Path, monkeypatch) -> None:
    import prepds.review_store as store

    video = _make(tmp_path)
    monkeypatch.setattr(store, "_update_index", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        accept(video, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW)
    assert load_manifest(video).review_status == S.PROCESSED_AUTO  # not stranded as ACCEPTED

    monkeypatch.undo()
    accept(video, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW)
    assert load_manifest(video).review_status == S.ACCEPTED


def test_a_failure_while_writing_an_edit_leaves_every_artifact_untouched(tmp_path: Path, monkeypatch) -> None:
    import prepds.review_store as store

    video = _make(tmp_path)
    before = {n: (video / n).read_bytes() for n in ("frames.parquet", "segments.csv", "strip.png", "manifest.json")}
    monkeypatch.setattr(store, "render_strip", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        save_edit(video, [(1.0, 2.0, B.SURFACE_BREACH)], reviewer="lk", now=NOW)
    assert {n: (video / n).read_bytes() for n in before} == before


def test_accept_copies_segments_derived_from_the_frames_not_a_stale_file(tmp_path: Path) -> None:
    video = _make(tmp_path)
    (video / "segments.csv").write_text("start_s,end_s,duration_s,state,source\n0,10,10,Erratic Movement,auto\n")
    accept(video, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW)
    gold = pd.read_csv(tmp_path / "accepted" / "F_0332" / "segments.csv")
    assert list(gold["state"]) == ["Controlled Swim", "Freezing/Drift", "Erratic Movement"]


def test_edit_boundaries_follow_frame_index_time_not_float32_rounding(tmp_path: Path) -> None:
    video = _make(tmp_path)
    save_edit(video, [(1.0 / 3, 2.0 / 3, B.SURFACE_BREACH)], reviewer="lk", now=NOW)  # frames 10..19 at 30 fps
    frames = pd.read_parquet(video / "frames.parquet")
    assert list(frames[frames["state"].astype(str) == "Surface Breach"]["frame_idx"]) == list(range(10, 20))


@pytest.mark.parametrize("edits", [[], [(float("nan"), 2.0, B.DEAD)], [(2.0, float("nan"), B.DEAD)], [(0.001, 0.002, B.SURFACE_BREACH)]])
def test_degenerate_edits_are_rejected_and_do_not_count(tmp_path: Path, edits) -> None:
    video = _make(tmp_path)
    with pytest.raises(ValueError):
        save_edit(video, edits, reviewer="lk", now=NOW)
    assert load_manifest(video).edit_count == 0 and load_manifest(video).review_status == S.PROCESSED_AUTO


@pytest.mark.parametrize("bad_name", [".."])
def test_accept_refuses_a_video_directory_whose_name_could_escape_the_gold_dir(tmp_path: Path, bad_name: str) -> None:
    video = _make(tmp_path)
    with pytest.raises(ValueError, match="directory name"):
        accept(video / bad_name, accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW)


def test_the_manifest_write_is_atomic_a_failed_replace_leaves_the_old_manifest_intact(tmp_path: Path, monkeypatch) -> None:
    import os

    import prepds.review_store as store

    video = _make(tmp_path)
    before = (video / "manifest.json").read_text()
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        store._write_manifest(video, load_manifest(video))
    assert (video / "manifest.json").read_text() == before
    assert not list(video.glob("*.tmp"))


def test_invalid_edits_raise_invalid_edit_which_is_a_value_error(tmp_path: Path) -> None:
    from prepds.review_store import InvalidEdit

    video = _make(tmp_path)
    with pytest.raises(InvalidEdit):
        save_edit(video, [], reviewer="lk", now=NOW)
    assert issubclass(InvalidEdit, ValueError)


def test_edit_boundaries_copied_from_float32_segments_csv_select_exactly_that_frame(tmp_path: Path) -> None:
    import numpy as np

    for k in range(1, N - 1):
        video = _make(tmp_path / f"k{k}")
        lo, hi = float(np.float32(k / FPS)), float(np.float32((k + 1) / FPS))
        save_edit(video, [(lo, hi, B.SURFACE_BREACH)], reviewer="lk", now=NOW)
        frames = pd.read_parquet(video / "frames.parquet")
        assert list(frames[frames["state"].astype(str) == "Surface Breach"]["frame_idx"]) == [k]


# --- T081: rebuild the accepted index -------------------------------------------------


def test_rebuild_index_recovers_a_deleted_index_from_accepted_manifests(tmp_path: Path) -> None:
    from prepds.review_store import INDEX_FILE, rebuild_index

    gold = tmp_path / "accepted"
    for subject in ("0332", "0333"):
        video = _make(tmp_path, [B.CONTROLLED_SWIM] * 300, subject=subject)
        accept(video, accepted_dir=gold, reviewer="lk", now=NOW)
    (gold / INDEX_FILE).unlink()
    count = rebuild_index(gold)
    index = pd.read_parquet(gold / INDEX_FILE)
    assert count == 2 and sorted(index["video_id"]) == ["F_0332", "F_0333"]
    assert set(index["reviewer"]) == {"lk"} and set(index["provenance"]) == {"auto"}


def test_rebuild_index_ignores_non_accepted_and_broken_directories_and_drops_stale_rows(tmp_path: Path) -> None:
    from prepds.review_store import INDEX_FILE, rebuild_index

    gold = tmp_path / "accepted"
    accept(_make(tmp_path, [B.CONTROLLED_SWIM] * 300), accepted_dir=gold, reviewer="lk", now=NOW)
    (gold / "F_junk").mkdir()
    (gold / "F_bad").mkdir()
    (gold / "F_bad" / "manifest.json").write_text("{not json")
    stale = pd.read_parquet(gold / INDEX_FILE)
    stale.assign(video_id="F_gone").pipe(lambda d: pd.concat([stale, d])).to_parquet(gold / INDEX_FILE)
    assert rebuild_index(gold) == 1
    assert list(pd.read_parquet(gold / INDEX_FILE)["video_id"]) == ["F_0332"]


def test_rebuild_index_fills_trial_fields_from_a_catalog(tmp_path: Path) -> None:
    from prepds.review_store import INDEX_FILE, rebuild_index

    gold = tmp_path / "accepted"
    accept(_make(tmp_path, [B.CONTROLLED_SWIM] * 300), accepted_dir=gold, reviewer="lk", now=NOW)
    rebuild_index(gold, trials=[_trial("0332")])
    row = pd.read_parquet(gold / INDEX_FILE).iloc[0]
    assert row["strain"] == _trial("0332").strain


def test_rebuild_index_on_an_empty_or_missing_dir_writes_an_empty_index(tmp_path: Path) -> None:
    from prepds.review_store import INDEX_FILE, rebuild_index

    assert rebuild_index(tmp_path / "nothing") == 0
    assert list(pd.read_parquet(tmp_path / "nothing" / INDEX_FILE)["video_id"]) == []


def test_index_columns_match_the_row_written_on_accept(tmp_path: Path) -> None:
    from prepds.review_store import INDEX_COLUMNS, INDEX_FILE

    accept(_make(tmp_path, [B.CONTROLLED_SWIM] * 300), accepted_dir=tmp_path / "accepted", reviewer="lk", now=NOW)
    assert tuple(pd.read_parquet(tmp_path / "accepted" / INDEX_FILE).columns) == INDEX_COLUMNS
