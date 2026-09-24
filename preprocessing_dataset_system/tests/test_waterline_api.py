"""Review API: the manual waterline of a video, and the flag that says whether the run has a detector box."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prepds.webapp.app import create_app
from tests.test_overlay_api import _detections
from tests.test_webapp_api import _make


@pytest.fixture
def env(tmp_path: Path):
    processed, sources = tmp_path / "processed", tmp_path / "videos"
    processed.mkdir(), sources.mkdir()
    video_id = _make(processed, sources)
    shutil.copy(Path(__file__).parent / "fixtures" / "synth_tiny.mp4", sources / "synth_tiny.mp4")
    app = create_app(processed, tmp_path / "accepted", video_dir=sources, allowed_hosts=("testserver",))
    return TestClient(app), processed / video_id, video_id


def test_no_waterline_is_reported_as_none_not_as_zero(env) -> None:
    client, _, video_id = env
    body = client.get(f"/videos/{video_id}/waterline").json()
    assert body == {"y_px": None, "source": None}


def test_a_saved_waterline_is_returned_and_stored_beside_the_artifacts(env) -> None:
    client, directory, video_id = env
    saved = client.put(f"/videos/{video_id}/waterline", json={"y_px": 42.5})
    assert saved.status_code == 200 and saved.json() == {"y_px": 42.5, "source": "manual"}
    assert client.get(f"/videos/{video_id}/waterline").json()["y_px"] == 42.5
    assert json.loads((directory / "waterline.json").read_text())["y_px"] == 42.5


@pytest.mark.parametrize("bad", [-1, 1e9, "top", None])
def test_an_out_of_frame_or_non_numeric_waterline_is_refused(env, bad) -> None:
    client, directory, video_id = env
    response = client.put(f"/videos/{video_id}/waterline", json={"y_px": bad})
    assert response.status_code == 422 and not (directory / "waterline.json").exists()


def test_a_corrupt_waterline_file_reads_as_none(env) -> None:
    client, directory, video_id = env
    (directory / "waterline.json").write_text("{not json")
    assert client.get(f"/videos/{video_id}/waterline").json()["y_px"] is None


def test_the_waterline_can_be_cleared(env) -> None:
    client, directory, video_id = env
    client.put(f"/videos/{video_id}/waterline", json={"y_px": 10})
    assert client.delete(f"/videos/{video_id}/waterline").status_code == 200
    assert client.get(f"/videos/{video_id}/waterline").json()["y_px"] is None and not (directory / "waterline.json").exists()


def test_unknown_video_is_404(env) -> None:
    client, _, _ = env
    assert client.get("/videos/nope/waterline").status_code == 404


def test_a_foreign_origin_cannot_set_the_waterline(env) -> None:
    client, _, video_id = env
    response = client.put(f"/videos/{video_id}/waterline", json={"y_px": 10}, headers={"Origin": "http://evil.example"})
    assert response.status_code == 403


def test_overlay_says_whether_the_run_has_a_detector(env) -> None:
    client, directory, video_id = env
    params = {"start_s": 0.0, "end_s": 1.0}
    assert client.get(f"/videos/{video_id}/overlay", params=params).json()["has_detector"] is False
    _detections(directory)
    assert client.get(f"/videos/{video_id}/overlay", params=params).json()["has_detector"] is True


def test_a_nan_waterline_is_refused(env) -> None:
    client, directory, video_id = env
    response = client.put(f"/videos/{video_id}/waterline", content='{"y_px": NaN}', headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and not (directory / "waterline.json").exists()
