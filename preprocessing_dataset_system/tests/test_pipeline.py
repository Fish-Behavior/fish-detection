"""Batch orchestration (T079-T082): per-video pipeline, resumability, failure isolation, worker pool."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from prepds.models import MatchStatus, ReviewStatus, Track, Trial
from prepds.pipeline import Outcome, RunConfig, process_trial, run_batch, video_id
from prepds.review_store import load_manifest, save_edit
from prepds.models import BehaviorState as B

FIXTURE_VIDEO = Path(__file__).parent / "fixtures" / "synth_tiny.mp4"
NOW = dt.datetime(2026, 9, 23, tzinfo=dt.timezone.utc)
THRESHOLDS = {
    "freeze_speed_floor_px_per_s": 22.67,
    "min_freeze_bout_s": 0.03,
    "erratic_speed_threshold_px_per_s": 41.2,
    "surface_breach_depth_threshold_px": 20.0,
    "min_dead_bout_s": 60.0,
    "speed_lag_s": 1.0,
    "listing_orientation_deviation_deg": None,
}


def _trial(subject="0001", sex="F", video: Path | None = FIXTURE_VIDEO, status=MatchStatus.MATCHED) -> Trial:
    return Trial(subject, sex, "Casper", None, "Veh", "1% DMSO", dt.date(2026, 3, 1), 20.0, video, status)


def _config(tmp_path: Path, force: bool = False) -> RunConfig:
    return RunConfig(out_dir=tmp_path / "processed", thresholds=THRESHOLDS, pipeline_version="0.1.0",
                     calibration_profile_version="cal-test", processed_at=NOW, force=force)


def _fake_tracker(path: Path) -> list[Track]:
    return [Track(i, i / 30.0, 50.0 + (i % 7), 100.0, 90.0, 100.0, True) for i in range(120)]


def test_video_id_is_sex_and_subject() -> None:
    assert video_id(_trial("0332", "F")) == "F_0332"


def test_process_trial_writes_all_artifacts_and_status(tmp_path: Path) -> None:
    result = process_trial(_trial(), _config(tmp_path), tracker=_fake_tracker)
    assert result.outcome is Outcome.PROCESSED
    out = tmp_path / "processed" / "F_0001"
    assert {p.name for p in out.iterdir()} >= {"frames.parquet", "segments.csv", "strip.png", "manifest.json"}
    assert load_manifest(out).review_status is ReviewStatus.PROCESSED_AUTO
    assert load_manifest(out).calibration_profile_version == "cal-test"


def test_an_already_processed_video_is_skipped_so_runs_are_resumable(tmp_path: Path) -> None:
    process_trial(_trial(), _config(tmp_path), tracker=_fake_tracker)

    def boom(path):
        raise AssertionError("must not re-track")

    result = process_trial(_trial(), _config(tmp_path), tracker=boom)
    assert result.outcome is Outcome.SKIPPED


def test_reviewed_work_survives_a_normal_run_and_is_replaced_only_with_force(tmp_path: Path) -> None:
    process_trial(_trial(), _config(tmp_path), tracker=_fake_tracker)
    out = tmp_path / "processed" / "F_0001"
    save_edit(out, [(0.5, 1.0, B.SURFACE_BREACH)], reviewer="lk", now=NOW)
    assert process_trial(_trial(), _config(tmp_path), tracker=_fake_tracker).outcome is Outcome.SKIPPED
    assert load_manifest(out).review_status is ReviewStatus.EDITED
    assert process_trial(_trial(), _config(tmp_path, force=True), tracker=_fake_tracker).outcome is Outcome.PROCESSED
    assert load_manifest(out).review_status is ReviewStatus.PROCESSED_AUTO


def test_rejected_video_is_regenerated(tmp_path: Path) -> None:
    from prepds.review_store import reject

    process_trial(_trial(), _config(tmp_path), tracker=_fake_tracker)
    reject(tmp_path / "processed" / "F_0001", reviewer="lk", now=NOW)
    assert process_trial(_trial(), _config(tmp_path), tracker=_fake_tracker).outcome is Outcome.PROCESSED


def test_a_failing_video_is_reported_not_raised(tmp_path: Path) -> None:
    def broken(path):
        raise ValueError("cannot decode")

    result = process_trial(_trial(), _config(tmp_path), tracker=broken)
    assert result.outcome is Outcome.FAILED and "cannot decode" in result.message
    assert not (tmp_path / "processed" / "F_0001" / "manifest.json").exists()


def test_a_trial_without_a_video_fails_cleanly(tmp_path: Path) -> None:
    result = process_trial(_trial(video=None, status=MatchStatus.NO_VIDEO), _config(tmp_path), tracker=_fake_tracker)
    assert result.outcome is Outcome.FAILED


def test_batch_continues_past_a_failure_and_reports_every_video(tmp_path: Path) -> None:
    def flaky(path):
        if "bad" in path.name:
            raise ValueError("corrupt")
        return _fake_tracker(path)

    good = _trial("0001")
    bad = _trial("0002", video=Path("bad.mp4"))
    results = run_batch([good, bad], _config(tmp_path), workers=1, tracker=flaky)
    assert {r.video_id: r.outcome for r in results} == {"F_0001": Outcome.PROCESSED, "F_0002": Outcome.FAILED}


def test_duplicate_video_ids_are_failed_not_overwritten(tmp_path: Path) -> None:
    results = run_batch([_trial("0001"), _trial("0001")], _config(tmp_path), workers=1, tracker=_fake_tracker)
    assert sorted(r.outcome.value for r in results) == ["failed", "processed"]
    assert any("duplicate" in r.message for r in results)


def test_worker_pool_processes_real_video_end_to_end(tmp_path: Path) -> None:
    trials = [_trial("0001"), _trial("0002")]
    results = run_batch(trials, _config(tmp_path), workers=2)
    assert [r.outcome for r in results] == [Outcome.PROCESSED, Outcome.PROCESSED]
    frames = pd.read_parquet(tmp_path / "processed" / "F_0002" / "frames.parquet")
    assert len(frames) > 0


def _crash_task(args):
    import os

    trial, _config = args
    if trial.subject_id == "0002":
        os._exit(1)  # native crash: no Python exception can catch this
    from prepds.pipeline import _pool_task

    return _pool_task(args)


def test_a_crashing_worker_fails_only_its_video_and_the_batch_finishes(tmp_path: Path) -> None:
    trials = [_trial("0001"), _trial("0002"), _trial("0003")]
    results = run_batch(trials, _config(tmp_path), workers=2, _task=_crash_task)
    assert {r.video_id: r.outcome for r in results} == {
        "F_0001": Outcome.PROCESSED, "F_0002": Outcome.FAILED, "F_0003": Outcome.PROCESSED}
    assert "crash" in next(r for r in results if r.video_id == "F_0002").message.lower()


def test_an_exception_in_the_progress_callback_does_not_abort_the_batch(tmp_path: Path) -> None:
    def bad_callback(result):
        raise RuntimeError("printing failed")

    results = run_batch([_trial("0001"), _trial("0002")], _config(tmp_path), workers=1, tracker=_fake_tracker,
                        on_result=bad_callback)
    assert [r.outcome for r in results] == [Outcome.PROCESSED, Outcome.PROCESSED]


def test_dedupe_trials_keeps_the_first_and_reports_the_rest() -> None:
    from prepds.pipeline import dedupe_trials

    a, b, c = _trial("0001"), _trial("0001"), _trial("0002")
    unique, duplicates = dedupe_trials([a, b, c])
    assert unique == [a, c] and duplicates == [b]


def test_a_tracker_without_detections_writes_no_sidecar(tmp_path: Path) -> None:
    process_trial(_trial(), _config(tmp_path), tracker=_fake_tracker)
    assert not (tmp_path / "processed" / "F_0001" / "detections.parquet").exists()


def test_a_tracker_with_last_detections_gets_a_sidecar_next_to_the_artifacts(tmp_path: Path) -> None:
    class Tracker:
        last_detections = pd.DataFrame({"frame_idx": [0, 5], "score": [0.9, 0.2]})

        def __call__(self, path):
            return _fake_tracker(path)

    process_trial(_trial(), _config(tmp_path), tracker=Tracker())
    sidecar = pd.read_parquet(tmp_path / "processed" / "F_0001" / "detections.parquet")
    assert list(sidecar["frame_idx"]) == [0, 5]


def test_a_failed_sidecar_write_is_reported_and_leaves_no_partial_file(tmp_path: Path, monkeypatch) -> None:
    import os

    class Tracker:
        last_detections = pd.DataFrame({"frame_idx": [0], "score": [0.9]})

        def __call__(self, path):
            return _fake_tracker(path)

    real = os.replace

    def failing(src, dst):
        if str(dst).endswith("detections.parquet"):
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", failing)
    result = process_trial(_trial(), _config(tmp_path), tracker=Tracker())
    assert result.outcome is Outcome.FAILED and "disk full" in result.message
    assert not list((tmp_path / "processed" / "F_0001").glob("detections*"))


def test_a_failed_forced_export_does_not_leave_the_old_sidecar_beside_new_artifacts(tmp_path: Path, monkeypatch) -> None:
    import dataclasses

    import prepds.pipeline as pipeline

    class Tracker:
        last_detections = pd.DataFrame({"frame_idx": [0], "score": [0.9]})

        def __call__(self, path):
            return _fake_tracker(path)

    process_trial(_trial(), _config(tmp_path), tracker=Tracker())
    sidecar = tmp_path / "processed" / "F_0001" / "detections.parquet"
    assert sidecar.exists()

    def failing_export(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "export_video", failing_export)
    result = process_trial(_trial(), dataclasses.replace(_config(tmp_path), force=True), tracker=Tracker())
    assert result.outcome is Outcome.FAILED
    assert not sidecar.exists()
