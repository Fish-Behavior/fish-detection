"""Review API: fuller time-per-state numbers and the per-second 'why this label' endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prepds.calibration.profile import write_calibration_profile
from prepds.webapp.app import create_app
from tests.test_webapp_api import _make

VERSION = "cal-2026-09-23-r2"
THRESHOLDS = {"freeze_speed_floor_px_per_s": 6.0, "min_freeze_bout_s": 1.0, "erratic_speed_threshold_px_per_s": 40.0,
              "surface_breach_depth_threshold_px": 20.0, "min_dead_bout_s": 60.0}


@pytest.fixture
def env(tmp_path: Path):
    processed, sources, profiles = tmp_path / "processed", tmp_path / "videos", tmp_path / "profiles"
    processed.mkdir(), sources.mkdir(), profiles.mkdir()
    video_id = _make(processed, sources)
    app = create_app(processed, tmp_path / "accepted", video_dir=sources, profile_dir=profiles, allowed_hosts=("testserver",))
    return TestClient(app), video_id, profiles


def _profile(profiles: Path) -> None:
    write_calibration_profile(profiles / f"{VERSION}.yaml", version=VERSION, created_at="2026-09-23", thresholds=THRESHOLDS, metrics={})


def test_summary_has_total_and_bouts_per_state(env) -> None:
    client, video_id, _ = env
    summary = client.get(f"/videos/{video_id}").json()["frame_summary"]
    assert summary["total_s"] == pytest.approx(sum(summary["seconds_by_state"].values()))
    assert summary["bouts_by_state"] == {"Controlled Swim": 1, "Freezing/Drift": 1, "Erratic Movement": 1}


def test_explain_returns_rules_and_numbers_when_the_profile_is_found(env) -> None:
    client, video_id, profiles = env
    _profile(profiles)
    body = client.get(f"/videos/{video_id}/explain", params={"second": 1}).json()
    assert body["second"] == 1 and body["thresholds_available"] is True and body["thresholds_error"] is None
    assert [r["state"] for r in body["rules"]][0] == "Undetermined" and len(body["rules"]) == 7
    assert body["stored_state"] == "Controlled Swim" and body["n_frames"] == 30


def test_explain_without_the_profile_still_gives_measurements(env) -> None:
    client, video_id, _ = env
    body = client.get(f"/videos/{video_id}/explain", params={"second": 4}).json()
    assert body["thresholds_available"] is False and body["rules"] == [] and body["thresholds_error"]
    assert body["speed_median_px_per_s"] is not None


def test_explain_validates_video_and_second(env) -> None:
    client, video_id, _ = env
    assert client.get("/videos/nope/explain", params={"second": 1}).status_code == 404
    assert client.get(f"/videos/{video_id}/explain", params={"second": 5000}).status_code == 404
    assert client.get(f"/videos/{video_id}/explain", params={"second": -1}).status_code == 422
    assert client.get(f"/videos/{video_id}/explain", params={"second": "x"}).status_code == 422
    assert client.get(f"/videos/{video_id}/explain").status_code == 422


def test_explain_reads_the_profile_by_name_only(env, tmp_path: Path) -> None:
    client, video_id, profiles = env
    outside = tmp_path / "evil.yaml"
    outside.write_text("labeling: {}\n")
    manifest = tmp_path / "processed" / video_id / "manifest.json"
    manifest.write_text(manifest.read_text().replace(VERSION, "../evil"))
    body = client.get(f"/videos/{video_id}/explain", params={"second": 4}).json()
    assert body["thresholds_available"] is False
