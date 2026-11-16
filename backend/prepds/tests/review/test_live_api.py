"""Live sessions through the review API: a camera is one more source. It ends in the same finish step as a file
and then is an ordinary session; a file with no workbook row goes through the same pipeline."""

from __future__ import annotations

import datetime as dt
import itertools
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prepds import review_store
from prepds.config import load_settings
from prepds.live import LiveSession
from prepds.pipeline import UNLABELED, unlabeled_trial
from prepds.webapp import workspace
from prepds.webapp.app import create_app

SYNTH = Path("backend/shared/tests/fixtures/synth_tiny.mp4")


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Review app with pipeline settings; the 'camera' is the synthetic video played at real speed."""
    processed, sources = tmp_path / "processed", tmp_path / "videos"
    processed.mkdir(); sources.mkdir()
    env = tmp_path / "test.env"; env.write_text("")
    ticks = itertools.count()
    clock = lambda: dt.datetime(2026, 11, 16, 12, 0, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=next(ticks))
    opened = []

    def fake_session(camera, recording, **kwargs):
        opened.append(camera)
        return LiveSession(SYNTH, recording, realtime=True, **kwargs)

    monkeypatch.setattr(workspace, "LiveSession", fake_session)
    monkeypatch.setattr(workspace, "list_cameras", lambda: [{"index": 0, "name": "Camera 0", "width": 304, "height": 240}])

    def client(settings=True):
        return TestClient(create_app(processed, tmp_path / "accepted", video_dir=sources, clock=clock, allowed_hosts=("testserver",),
                                     settings=load_settings(env_file=env, environ={}) if settings else None))
    return client, processed, sources, opened


def _wait(client, condition, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        status = client.get("/api/live").json()
        if condition(status):
            return status
        time.sleep(0.05)
    raise AssertionError(f"timed out; last status {status}")


def test_live_session_runs_ends_and_becomes_an_ordinary_session(app):
    make, processed, _, opened = app
    client = make()
    assert client.get("/api/live").json() == {"active": False}
    assert client.get("/api/live/cameras").json()["cameras"][0]["name"] == "Camera 0"

    started = client.post("/api/live", json={"camera": 0})
    assert started.status_code == 201 and opened == [0]
    vid = started.json()["video_id"]
    assert vid.startswith("live_20261116T1200")
    running = _wait(client, lambda s: s["frame_count"] > 15)
    assert running["phase"] == "running" and running["active"] and running["state"]
    preview = client.get("/api/live/preview")
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/jpeg" and preview.content[:2] == b"\xff\xd8"
    assert client.post("/api/live", json={"camera": 0}).status_code == 409  # one session at a time
    assert client.get("/api/live/cameras").status_code == 409  # the camera is in use
    assert all(row["video_id"] != vid for row in client.get("/api/sessions").json())  # not a session until it ends

    ended = client.post("/api/live/stop")
    assert ended.status_code == 202
    done = _wait(client, lambda s: s["phase"] in ("finished", "failed"))
    assert done["phase"] == "finished" and not done["active"], done["message"]
    assert client.post("/api/live/stop").status_code == 409

    row = next(row for row in client.get("/api/sessions").json() if row["video_id"] == vid)
    assert row["name"] == vid and row["processing_status"] == "processed" and row["processable"]
    session = client.get(f"/api/sessions/{vid}").json()
    assert session["has_tracking"] and session["segments"] and session["name"] == vid
    assert client.get(session["video_url"]).status_code == 200
    manifest = review_store.load_manifest(processed / vid)
    assert (manifest.compound, manifest.concentration_mM, manifest.sex) == (UNLABELED, UNLABELED, "live")
    assert manifest.video_path.parent.name == vid and manifest.video_path.stem == "recording"


def test_unmatched_file_is_processed_without_a_workbook_row(app):
    make, processed, sources, _ = app
    shutil.copy(SYNTH, sources / "new_fish.mp4")
    client = make()
    row = next(row for row in client.get("/api/sessions").json() if row["name"] == "new_fish.mp4")
    assert row["processable"] and row["processing_status"] == "unprocessed"
    assert client.post(f"/api/sessions/{row['video_id']}/process", json={"reviewer": "R"}).status_code == 202
    assert client.get(f"/api/sessions/{row['video_id']}/processing").json()["status"] == "processed"
    manifest = review_store.load_manifest(processed / row["video_id"])
    assert manifest.compound == UNLABELED and f"{manifest.sex}_{manifest.subject_id}" == row["video_id"]


def test_live_needs_settings_a_valid_camera_and_a_frame(app):
    make, _, _, _ = app
    assert make(settings=False).post("/api/live", json={"camera": 0}).status_code == 503
    client = make()
    assert client.post("/api/live", json={"camera": 99}).status_code == 422
    assert client.post("/api/live", json={"camera": "0"}).status_code == 422
    assert client.get("/api/live/preview").status_code == 404
    assert client.post("/api/live/stop").status_code == 409


def test_a_live_session_that_recorded_nothing_fails_cleanly_and_frees_the_slot(app, monkeypatch, tmp_path):
    make, processed, _, _ = app
    monkeypatch.setattr(workspace, "LiveSession", lambda camera, recording, **kw: LiveSession(tmp_path / "missing.mp4", recording, **kw))
    client = make()
    vid = client.post("/api/live", json={"camera": 1}).json()["video_id"]
    errored = _wait(client, lambda s: s["phase"] == "error")
    assert "Could not open video" in errored["message"]
    assert client.post("/api/live/stop").status_code == 202
    assert _wait(client, lambda s: s["phase"] in ("finished", "failed"))["phase"] == "failed"
    assert not (processed / vid / "manifest.json").exists()
    assert client.post("/api/live", json={"camera": 1}).status_code == 201  # the slot is free again


def test_unlabeled_trial_round_trips_the_video_id():
    trial = unlabeled_trial("raw_ab12", Path("x.mp4"), dt.date(2026, 11, 16))
    assert (trial.sex, trial.subject_id, trial.compound) == ("raw", "ab12", UNLABELED)
    with pytest.raises(ValueError):
        unlabeled_trial("nounderscore", Path("x.mp4"), dt.date(2026, 11, 16))
