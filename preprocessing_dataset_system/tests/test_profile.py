"""Calibration profiles (T046): immutable frozen thresholds and their loading into the labeler."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from prepds.calibration.profile import labeling_thresholds, write_calibration_profile
from prepds.config import ConfigError, load_settings
from prepds.labeling import classify_video

THRESHOLDS = {
    "freeze_speed_floor_px_per_s": 17.0,
    "min_freeze_bout_s": 1.2,
    "erratic_speed_threshold_px_per_s": 35.0,
    "surface_breach_depth_threshold_px": 22.0,
    "min_dead_bout_s": 120.0,
}


def test_write_profile_records_thresholds_metrics_and_provenance(tmp_path: Path) -> None:
    path = write_calibration_profile(
        tmp_path / "cal-2026-09-23.yaml",
        version="cal-2026-09-23",
        created_at="2026-09-23",
        thresholds=THRESHOLDS,
        metrics={"mean_tv": 0.18, "heldout_mean_tv": 0.24},
        notes="Listing/LORR excluded (manual).",
    )
    data = yaml.safe_load(path.read_text())
    assert data["calibration_profile"]["version"] == "cal-2026-09-23"
    assert data["labeling"]["freeze_speed_floor_px_per_s"] == 17.0
    assert data["labeling"]["listing_orientation_deviation_deg"] is None  # Listing stays manual
    assert data["calibration_profile"]["metrics"]["heldout_mean_tv"] == 0.24


def test_a_frozen_profile_is_immutable(tmp_path: Path) -> None:
    target = tmp_path / "cal.yaml"
    write_calibration_profile(target, version="v1", created_at="2026-09-23", thresholds=THRESHOLDS, metrics={})
    with pytest.raises(FileExistsError):
        write_calibration_profile(target, version="v1", created_at="2026-09-23", thresholds=THRESHOLDS, metrics={})


def test_a_profile_missing_a_threshold_is_rejected_at_write_time(tmp_path: Path) -> None:
    incomplete = {k: v for k, v in THRESHOLDS.items() if k != "min_dead_bout_s"}
    with pytest.raises(ValueError, match="min_dead_bout_s"):
        write_calibration_profile(tmp_path / "x.yaml", version="v", created_at="d", thresholds=incomplete, metrics={})


def test_labeling_thresholds_from_a_loaded_profile_drive_classify_video(tmp_path: Path) -> None:
    path = write_calibration_profile(
        tmp_path / "cal.yaml", version="v", created_at="2026-09-23", thresholds=THRESHOLDS, metrics={}
    )
    settings = load_settings(config_file=path, environ={})
    kwargs = labeling_thresholds(settings.params)

    assert kwargs["freeze_speed_floor_px_per_s"] == 17.0
    assert kwargs["listing_orientation_deviation_deg"] is None
    assert classify_video([], **kwargs) == []  # every key is a real classify_video parameter


def test_packaged_defaults_are_the_frozen_profile_and_classify_video_accepts_them() -> None:
    kwargs = labeling_thresholds(load_settings(environ={}).params)
    assert kwargs["freeze_speed_floor_px_per_s"] < kwargs["erratic_speed_threshold_px_per_s"]
    assert kwargs["listing_orientation_deviation_deg"] is None  # Listing/LORR is manual
    assert classify_video([], **kwargs) == []

    frozen = yaml.safe_load((Path(__file__).parent.parent / "config/calibration_profiles/cal-2026-09-23-r2.yaml").read_text())
    assert frozen["labeling"]["freeze_speed_floor_px_per_s"] == kwargs["freeze_speed_floor_px_per_s"]


def test_null_thresholds_refuse_to_provide_thresholds(tmp_path: Path) -> None:
    override = tmp_path / "null.yaml"
    override.write_text("labeling:\n  freeze_speed_floor_px_per_s: null\n")
    with pytest.raises(ConfigError, match="uncalibrated"):
        labeling_thresholds(load_settings(config_file=override, environ={}).params)


def test_an_unserializable_metric_does_not_leave_a_partial_file_or_burn_the_version(tmp_path: Path) -> None:
    target = tmp_path / "cal.yaml"
    with pytest.raises(Exception):
        write_calibration_profile(
            target, version="v", created_at="d", thresholds=THRESHOLDS, metrics={"bad": object()}
        )
    assert not target.exists()
    write_calibration_profile(target, version="v", created_at="d", thresholds=THRESHOLDS, metrics={})  # still writable


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0])
def test_non_finite_or_non_positive_thresholds_are_rejected(tmp_path: Path, bad: float) -> None:
    with pytest.raises(ValueError, match="freeze_speed_floor_px_per_s"):
        write_calibration_profile(
            tmp_path / "x.yaml",
            version="v",
            created_at="d",
            thresholds={**THRESHOLDS, "freeze_speed_floor_px_per_s": bad},
            metrics={},
        )


def test_the_speed_lag_is_recorded_and_returned_so_thresholds_stay_tied_to_it(tmp_path: Path) -> None:
    path = write_calibration_profile(
        tmp_path / "cal.yaml", version="v", created_at="d", thresholds=THRESHOLDS, metrics={}, speed_lag_s=1.0
    )
    kwargs = labeling_thresholds(load_settings(config_file=path, environ={}).params)
    assert kwargs["speed_lag_s"] == 1.0
    assert classify_video([], **kwargs) == []


def test_profile_rejects_an_erratic_threshold_at_or_below_the_freeze_floor(tmp_path: Path) -> None:
    bad = {**THRESHOLDS, "freeze_speed_floor_px_per_s": 40.0, "erratic_speed_threshold_px_per_s": 30.0}
    with pytest.raises(ValueError, match="erratic_speed_threshold_px_per_s"):
        write_calibration_profile(tmp_path / "x.yaml", version="v", created_at="d", thresholds=bad, metrics={})
