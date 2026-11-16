"""Live sessions through the review API: a camera is one more source. It ends in the same finish step as a file
and then is an ordinary session; a file with no workbook row goes through the same pipeline."""

from __future__ import annotations

import datetime as dt
import itertools
import re
import shutil
import threading
import time
from pathlib import Path

import cv2
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
    assert client.get("/api/live").json() == {"active": False, "suggested_name": "Video1"}
    assert client.get("/api/live/cameras").json()["cameras"][0]["name"] == "Camera 0"

    started = client.post("/api/live", json={"camera": 0, "reviewer": "Ann"})
    assert started.status_code == 201 and opened == [0]
    vid = started.json()["video_id"]
    assert re.fullmatch(r"Video1_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d", vid)  # default name, local date and time
    running = _wait(client, lambda s: s["frame_count"] > 15)
    assert running["phase"] == "running" and running["active"] and running["state"]
    preview = client.get("/api/live/preview")
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/jpeg" and preview.content[:2] == b"\xff\xd8"
    second = client.post("/api/live", json={"camera": 0})
    assert second.status_code == 409 and "already open" in second.text  # one session at a time
    assert client.get("/api/live/cameras").status_code == 409  # the camera is in use
    assert all(row["video_id"] != vid for row in client.get("/api/sessions").json())  # not a session until it ends

    ended = client.post("/api/live/stop")
    assert ended.status_code == 202
    done = _wait(client, lambda s: s["phase"] in ("finished", "failed"))
    assert done["phase"] == "finished" and not done["active"], done["message"]
    assert client.post("/api/live/stop").status_code == 409

    row = next(row for row in client.get("/api/sessions").json() if row["video_id"] == vid)
    assert row["name"] == "Live camera recording" and row["processing_status"] == "processed" and row["processable"]
    session = client.get(f"/api/sessions/{vid}").json()
    assert session["has_tracking"] and session["segments"] and session["name"] == "Live camera recording"
    assert session["inference_available"] is False and session["model_verdict"] is None  # no classifier configured here
    assert client.get(session["video_url"]).status_code == 200
    manifest = review_store.load_manifest(processed / vid)
    assert (manifest.compound, manifest.concentration_mM, manifest.sex) == (UNLABELED, UNLABELED, "Video1")
    assert session["live_recording"] is True
    assert session["processing"]["status"] == "processed" and session["processing"]["reviewer"] == "Ann"  # drives the auto prediction
    assert client.get("/api/live").json()["suggested_name"] == "Video2"  # numbered after the camera sessions so far
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


class _UnpluggedCamera:
    """A few real frames from the synthetic video, then the read fails as an unplugged camera does."""

    def __init__(self, frames: int = 5) -> None:
        self.capture, self.frames = cv2.VideoCapture(str(SYNTH)), frames

    def isOpened(self) -> bool:
        return self.capture.isOpened()

    def read(self):
        if self.frames <= 0:
            raise RuntimeError("camera unplugged")
        self.frames -= 1
        return self.capture.read()

    def get(self, prop) -> float:
        return self.capture.get(prop)

    def release(self) -> None:
        self.capture.release()


def test_a_capture_error_fails_the_session_with_its_cause_and_saves_nothing(app, monkeypatch):
    make, processed, _, _ = app
    monkeypatch.setattr(workspace, "LiveSession", lambda camera, recording, **kw: LiveSession(
        SYNTH, recording, opener=lambda source: _UnpluggedCamera(), **kw))
    client = make()
    vid = client.post("/api/live", json={"camera": 0}).json()["video_id"]
    errored = _wait(client, lambda s: s["phase"] == "error")
    assert errored["frame_count"] == 5  # frames were recorded before the camera failed
    assert client.post("/api/live/stop").status_code == 202
    done = _wait(client, lambda s: s["phase"] in ("finished", "failed"))
    assert done["phase"] == "failed" and "camera unplugged" in done["message"]
    assert not (processed / vid / "manifest.json").exists()


def test_an_errored_session_cannot_be_replaced_while_it_is_finishing(app, monkeypatch):
    make, _, _, _ = app
    stopping, release = threading.Event(), threading.Event()

    class HeldSession(LiveSession):
        def join(self, timeout=None):  # the stop request waits here, with the finish still pending
            stopping.set()
            release.wait(10)
            super().join(timeout)

    monkeypatch.setattr(workspace, "LiveSession", lambda camera, recording, **kw: HeldSession(
        SYNTH, recording, opener=lambda source: _UnpluggedCamera(), **kw))
    client = make()
    client.post("/api/live", json={"camera": 0})
    _wait(client, lambda s: s["phase"] == "error")
    stop = threading.Thread(target=client.post, args=("/api/live/stop",))
    stop.start()
    try:
        assert stopping.wait(10) and client.get("/api/live").json()["phase"] == "finishing"
        assert client.post("/api/live", json={"camera": 0}).status_code == 409
    finally:
        release.set()
        stop.join(timeout=30)
    assert _wait(client, lambda s: s["phase"] in ("finished", "failed"))["phase"] == "failed"
    assert client.post("/api/live", json={"camera": 0}).status_code == 201  # free once the finish is done


class _HungCamera:
    """Opens, then its read hangs (as a stalled camera driver does) until `release` is set; it never returns a frame."""

    def __init__(self, release: threading.Event) -> None:
        self._release = release

    def isOpened(self) -> bool:
        return True

    def read(self):
        self._release.wait(10)
        return False, None

    def get(self, prop) -> float:
        return 30.0

    def release(self) -> None:
        pass


def test_a_session_whose_capture_thread_is_still_alive_cannot_be_replaced(app, monkeypatch):
    make, _, _, _ = app
    release, sessions = threading.Event(), []

    class HungSession(LiveSession):
        def join(self, timeout=None):  # the stop request's 30 s wait, cut short; a join without a timeout still waits
            super().join(timeout and 0.1)

    def hung_session(camera, recording, **kw):
        sessions.append(HungSession(SYNTH, recording, opener=lambda source: _HungCamera(release), **kw))
        return sessions[-1]

    monkeypatch.setattr(workspace, "LiveSession", hung_session)
    client = make()
    try:
        assert client.post("/api/live", json={"camera": 0}).status_code == 201
        _wait(client, lambda s: s["phase"] == "running")
        assert client.post("/api/live/stop").status_code == 202
        assert _wait(client, lambda s: s["phase"] in ("finished", "failed"))["phase"] == "failed"  # it did not stop
        assert sessions[0].is_alive()
        refused = client.post("/api/live", json={"camera": 0})  # the old thread still holds the camera and the file
        assert refused.status_code == 409 and "still releasing" in refused.json()["detail"]["message"]
        assert len(sessions) == 1
    finally:
        release.set()
    sessions[0].join()
    assert client.post("/api/live", json={"camera": 0}).status_code == 201  # free once the thread has ended


def test_unlabeled_trial_round_trips_the_video_id():
    trial = unlabeled_trial("raw_ab12", Path("x.mp4"), dt.date(2026, 11, 16))
    assert (trial.sex, trial.subject_id, trial.compound) == ("raw", "ab12", UNLABELED)
    with pytest.raises(ValueError):
        unlabeled_trial("nounderscore", Path("x.mp4"), dt.date(2026, 11, 16))


def test_session_reports_the_configured_models_own_verdict(tmp_path):
    processed, sources = tmp_path / "processed", tmp_path / "videos"
    processed.mkdir(); sources.mkdir()
    shutil.copy(SYNTH, sources / "fish.mp4")
    root, run = tmp_path / "dcs", tmp_path / "run"
    (root / "src/dcs").mkdir(parents=True); (run / "model").mkdir(parents=True)
    (run / "model/model_info.json").write_text('{"verdict": "not useful"}')
    client = TestClient(create_app(processed, tmp_path / "accepted", video_dir=sources, classifier_root=root, model_run=run,
                                   allowed_hosts=("testserver",)))
    vid = client.get("/api/sessions").json()[0]["video_id"]
    session = client.get(f"/api/sessions/{vid}").json()
    assert session["inference_available"] is True and session["model_verdict"] == "not useful"
    (run / "model/model_info.json").write_text("{broken")
    broken = TestClient(create_app(processed, tmp_path / "accepted", video_dir=sources, classifier_root=root, model_run=run,
                                   allowed_hosts=("testserver",)))
    assert broken.get(f"/api/sessions/{vid}").json()["model_verdict"] is None  # an unreadable run never blocks startup


def test_the_user_names_a_live_session_and_bad_names_are_refused(app):
    make, _, _, _ = app
    client = make()
    for bad in ("Tank_A", "Tank A", "-tank", "x" * 41, ""):
        assert client.post("/api/live", json={"camera": 0, "name": bad}).status_code == 422, bad
    vid = client.post("/api/live", json={"camera": 0, "name": "TankA-dose1"}).json()["video_id"]
    assert vid.startswith("TankA-dose1_")
    _wait(client, lambda s: s["frame_count"] > 0)
    client.post("/api/live/stop")
    assert _wait(client, lambda s: s["phase"] in ("finished", "failed"))["phase"] == "finished"
    session = client.get(f"/api/sessions/{vid}").json()
    assert session["live_recording"] is True and session["name"] == "Live camera recording"


def test_looking_for_cameras_and_starting_a_recording_never_overlap(app, monkeypatch):
    """Some camera drivers allow one user at a time: a lookup holds the cameras until it is done."""
    make, _, _, _ = app
    probing, release = threading.Event(), threading.Event()

    def slow_lookup():
        probing.set()
        release.wait(10)
        return []

    monkeypatch.setattr(workspace, "list_cameras", slow_lookup)
    client = make()
    lookup = threading.Thread(target=client.get, args=("/api/live/cameras",))
    lookup.start()
    try:
        assert probing.wait(10)
        start = client.post("/api/live", json={"camera": 0})
        assert start.status_code == 409 and "Looking for cameras" in start.text
        again = client.get("/api/live/cameras")
        assert again.status_code == 409 and "Already looking" in again.text
    finally:
        release.set()
        lookup.join(timeout=10)
    assert client.post("/api/live", json={"camera": 0}).status_code == 201  # free once the lookup is done
