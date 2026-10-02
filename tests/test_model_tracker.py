"""ModelTracker plumbing (Phase 15): contract with the pipeline, not model quality. CPU, untrained tiny model."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from prepds.annotation.examples import Example  # noqa: E402
from prepds.annotation.train import train  # noqa: E402
from prepds.model_tracker import ModelTracker  # noqa: E402
from prepds.video_io import probe  # noqa: E402
from PIL import Image  # noqa: E402

FIXTURE_VIDEO = Path(__file__).parent / "fixtures" / "synth_tiny.mp4"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("model")
    image = tmp / "f.png"
    Image.new("RGB", (128, 96), (100, 120, 90)).save(image)
    example = Example("f", image, 128, 96, (10.0, 20.0, 100.0, 80.0), {"snout": (20.0, 50.0, 2)}, "no")
    train([example], tmp / "run", epochs=1, pretrained=False, device="cpu", min_size=128, max_size=170, log=lambda *_: None)
    return tmp / "run"


def test_returns_one_track_per_frame_with_contiguous_indices(run_dir: Path) -> None:
    tracker = ModelTracker(run_dir, device="cpu", stride=3, min_score=0.0, min_size=128, batch_size=4)
    tracks = tracker(FIXTURE_VIDEO)
    assert len(tracks) == probe(FIXTURE_VIDEO).frame_count
    assert [t.frame_idx for t in tracks] == list(range(len(tracks)))


def test_high_score_threshold_means_nothing_detected(run_dir: Path) -> None:
    tracker = ModelTracker(run_dir, device="cpu", stride=3, min_score=1.01, min_size=128)
    assert not any(t.detected for t in tracker(FIXTURE_VIDEO))


def test_name_records_the_run_and_stride(run_dir: Path) -> None:
    assert ModelTracker(run_dir, device="cpu", stride=5, min_size=128).name == "model:run@stride5"


def test_invalid_arguments(run_dir: Path) -> None:
    with pytest.raises(ValueError):
        ModelTracker(run_dir, device="cpu", stride=0)


def test_plugs_into_process_trial(run_dir: Path, tmp_path: Path) -> None:
    import datetime as dt

    from prepds.models import MatchStatus, Trial
    from prepds.pipeline import Outcome, RunConfig, process_trial

    thresholds = {"freeze_speed_floor_px_per_s": 22.67, "min_freeze_bout_s": 0.03, "erratic_speed_threshold_px_per_s": 41.2,
                  "surface_breach_depth_threshold_px": 20.0, "min_dead_bout_s": 60.0, "speed_lag_s": 1.0,
                  "listing_orientation_deviation_deg": None}
    trial = Trial("0001", "F", "Casper", None, "Veh", "1% DMSO", dt.date(2026, 3, 1), 20.0, FIXTURE_VIDEO, MatchStatus.MATCHED)
    config = RunConfig(tmp_path / "processed", thresholds, "0.1.0", "cal-test", dt.datetime(2026, 9, 23, tzinfo=dt.timezone.utc))
    tracker = ModelTracker(run_dir, device="cpu", stride=3, min_score=0.0, min_size=128)
    assert process_trial(trial, config, tracker=tracker).outcome is Outcome.PROCESSED


def test_raw_detections_are_recorded_for_every_sampled_frame(run_dir: Path) -> None:
    tracker = ModelTracker(run_dir, device="cpu", stride=3, min_score=0.0, min_size=128, batch_size=4)
    tracks = tracker(FIXTURE_VIDEO)
    raw = tracker.last_detections
    assert list(raw["frame_idx"]) == list(range(0, len(tracks), 3))
    expected = {"frame_idx", "score", "x0", "y0", "x1", "y1"} | {
        f"{name}_{axis}" for name in ("snout", "dorsal_fin_base", "ventral", "tail_base", "tail_tip") for axis in ("x", "y", "score")}
    assert expected <= set(raw.columns)


def test_a_detection_below_the_threshold_is_still_recorded_but_not_used_for_the_track(run_dir: Path) -> None:
    tracker = ModelTracker(run_dir, device="cpu", stride=3, min_score=1.01, min_size=128)
    tracks = tracker(FIXTURE_VIDEO)
    assert not any(t.detected for t in tracks)
    assert tracker.last_detections["score"].notna().any()  # the model's best guess is kept for later re-thresholding


def test_last_detections_belong_to_the_latest_video_only(run_dir: Path) -> None:
    tracker = ModelTracker(run_dir, device="cpu", stride=3, min_score=0.0, min_size=128)
    tracker(FIXTURE_VIDEO)
    first = tracker.last_detections
    tracker(FIXTURE_VIDEO)
    assert tracker.last_detections is not first and len(tracker.last_detections) == len(first)


def test_process_trial_writes_a_detections_sidecar_for_a_model_tracker(run_dir: Path, tmp_path: Path) -> None:
    import datetime as dt

    import pandas as pd

    from prepds.models import MatchStatus, Trial
    from prepds.pipeline import RunConfig, process_trial

    thresholds = {"freeze_speed_floor_px_per_s": 22.67, "min_freeze_bout_s": 0.03, "erratic_speed_threshold_px_per_s": 41.2,
                  "surface_breach_depth_threshold_px": 20.0, "min_dead_bout_s": 60.0, "speed_lag_s": 1.0,
                  "listing_orientation_deviation_deg": None}
    trial = Trial("0001", "F", "Casper", None, "Veh", "1% DMSO", dt.date(2026, 3, 1), 20.0, FIXTURE_VIDEO, MatchStatus.MATCHED)
    config = RunConfig(tmp_path / "processed", thresholds, "0.1.0", "cal-test", dt.datetime(2026, 9, 23, tzinfo=dt.timezone.utc))
    tracker = ModelTracker(run_dir, device="cpu", stride=3, min_score=0.0, min_size=128)
    assert process_trial(trial, config, tracker=tracker).outcome.value == "processed"
    sidecar = tmp_path / "processed" / "F_0001" / "detections.parquet"
    assert sidecar.is_file() and len(pd.read_parquet(sidecar)) == len(tracker.last_detections)
    assert not list((tmp_path / "processed" / "F_0001").glob("*.tmp"))
