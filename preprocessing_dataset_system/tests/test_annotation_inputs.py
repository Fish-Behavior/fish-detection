"""Loading processed videos for sampling and extracting the sampled frames (Phase 15)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from prepds.annotation.inputs import extract_frame, load_video_infos
from prepds.models import MatchStatus, Trial
from prepds.pipeline import RunConfig, process_trial

FIXTURE_VIDEO = Path(__file__).parent / "fixtures" / "synth_tiny.mp4"
THRESHOLDS = {"freeze_speed_floor_px_per_s": 22.67, "min_freeze_bout_s": 0.03, "erratic_speed_threshold_px_per_s": 41.2,
              "surface_breach_depth_threshold_px": 20.0, "min_dead_bout_s": 60.0, "speed_lag_s": 1.0,
              "listing_orientation_deviation_deg": None}


@pytest.fixture
def processed(tmp_path: Path) -> Path:
    trial = Trial("0001", "F", "Casper", None, "Fentanyl", "0.03", dt.date(2026, 3, 1), 20.0, FIXTURE_VIDEO, MatchStatus.MATCHED)
    config = RunConfig(tmp_path / "processed", THRESHOLDS, "0.1.0", "cal-test", dt.datetime(2026, 9, 23, tzinfo=dt.timezone.utc))
    assert process_trial(trial, config).outcome.value == "processed"
    return tmp_path / "processed"


def test_load_video_infos_reads_frames_and_manifest(processed: Path) -> None:
    (info,) = load_video_infos(processed)
    assert info.video_id == "F_0001" and info.compound == "Fentanyl" and info.fps > 0
    assert info.detected.dtype == bool and len(info.detected) == len(info.states) > 0


def test_flag_start_is_read_from_the_manifest(processed: Path) -> None:
    manifest = processed / "F_0001" / "manifest.json"
    data = json.loads(manifest.read_text())
    data["review_flags"] = [{"kind": "terminal_no_action", "start_s": 2.5, "end_s": 5.0, "message": "x"}]
    manifest.write_text(json.dumps(data))
    assert load_video_infos(processed)[0].flagged_from_s == 2.5


def test_unreadable_video_directories_are_skipped(processed: Path) -> None:
    (processed / "F_bad").mkdir()
    (processed / "F_bad" / "manifest.json").write_text("{nope")
    assert [v.video_id for v in load_video_infos(processed)] == ["F_0001"]


def test_extract_frame_returns_the_requested_frame() -> None:
    frame = extract_frame(FIXTURE_VIDEO, 3)
    assert isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3


def test_extract_frame_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        extract_frame(FIXTURE_VIDEO, 10_000)


def test_non_contiguous_frame_indices_are_refused(processed: Path) -> None:
    import pandas as pd

    path = processed / "F_0001" / "frames.parquet"
    frames = pd.read_parquet(path)
    frames.loc[frames.index[5:], "frame_idx"] += 2  # a gap: row position no longer equals video frame
    frames.to_parquet(path)
    assert load_video_infos(processed) == []  # skipped, never mis-indexed
