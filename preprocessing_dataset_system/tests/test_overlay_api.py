"""Review API: the fish overlay data for a time window (track point, path, detector box and keypoints)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from prepds.annotation.store import KEYPOINT_NAMES
from prepds.webapp.app import create_app
from tests.test_webapp_api import _make


@pytest.fixture
def env(tmp_path: Path):
    processed, sources = tmp_path / "processed", tmp_path / "videos"
    processed.mkdir(), sources.mkdir()
    video_id = _make(processed, sources)
    app = create_app(processed, tmp_path / "accepted", video_dir=sources, allowed_hosts=("testserver",))
    return TestClient(app), processed / video_id, video_id


def _detections(directory: Path) -> None:
    rows = []
    for frame in (0, 5, 10, 150):
        row = {"frame_idx": frame, "score": 0.9 if frame != 10 else 0.1, "x0": 10.0, "y0": 20.0, "x1": 110.0, "y1": 80.0}
        for k, name in enumerate(KEYPOINT_NAMES):
            row.update({f"{name}_x": 20.0 + k, f"{name}_y": 30.0 + k, f"{name}_score": 0.8})
        rows.append(row)
    pd.DataFrame(rows).to_parquet(directory / "detections.parquet", index=False)


def test_window_returns_the_track_points_in_video_pixels(env) -> None:
    client, _, video_id = env
    body = client.get(f"/videos/{video_id}/overlay", params={"start_s": 1.0, "end_s": 2.0}).json()
    assert len(body["t"]) == len(body["x"]) == len(body["y"]) == len(body["detected"]) == 30
    assert body["t"][0] == pytest.approx(1.0, abs=1e-3) and body["x"][0] == pytest.approx(60.0, abs=0.1) and body["y"][0] == pytest.approx(100.0, abs=0.1)
    assert body["detections"] == [] and body["fps"] > 0


def test_detector_samples_are_included_only_above_the_score_floor(env) -> None:
    client, directory, video_id = env
    _detections(directory)
    body = client.get(f"/videos/{video_id}/overlay", params={"start_s": 0.0, "end_s": 1.0}).json()
    assert [d["frame_idx"] for d in body["detections"]] == [0, 5]  # frame 10 is below 0.3; 150 is outside the window
    first = body["detections"][0]
    assert first["box"] == [10.0, 20.0, 110.0, 80.0] and first["keypoints"]["snout"] == [20.0, 30.0, 0.8]
    assert first["t"] == pytest.approx(0.0, abs=1e-3)


def test_undetected_frames_are_flagged_not_drawn_at_zero(env) -> None:
    client, directory, video_id = env
    frames = pd.read_parquet(directory / "frames.parquet")
    frames.loc[frames.frame_idx.between(30, 59), "detected"] = False
    frames.to_parquet(directory / "frames.parquet", index=False)
    body = client.get(f"/videos/{video_id}/overlay", params={"start_s": 1.0, "end_s": 2.0}).json()
    assert set(body["detected"]) == {False} and all(x is None for x in body["x"]) and all(y is None for y in body["y"])


def test_window_is_validated(env) -> None:
    client, _, video_id = env
    get = lambda **p: client.get(f"/videos/{video_id}/overlay", params=p).status_code  # noqa: E731
    assert get(start_s=2, end_s=1) == 422 and get(start_s=-1, end_s=1) == 422 and get(start_s=0, end_s=100) == 422
    assert get(start_s="a", end_s=1) == 422 and get() == 422
    assert client.get("/videos/nope/overlay", params={"start_s": 0, "end_s": 1}).status_code == 404


def test_a_corrupt_detections_file_does_not_break_the_track_overlay(env) -> None:
    client, directory, video_id = env
    (directory / "detections.parquet").write_bytes(b"not parquet")
    response = client.get(f"/videos/{video_id}/overlay", params={"start_s": 0, "end_s": 1})
    assert response.status_code == 200 and response.json()["detections"] == [] and response.json()["detections_error"]


def test_the_coded_video_size_is_returned_so_the_page_can_scale_each_axis(env, tmp_path: Path) -> None:
    import shutil

    import cv2

    client, _, video_id = env
    source = tmp_path / "videos" / f"{video_id}.mp4"
    shutil.copy(Path(__file__).parent / "fixtures" / "synth_tiny.mp4", source)
    capture = cv2.VideoCapture(str(source))
    expected = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    body = client.get(f"/videos/{video_id}/overlay", params={"start_s": 0, "end_s": 1}).json()
    assert (body["width"], body["height"]) == expected


def test_size_is_null_when_the_source_video_cannot_be_read(env) -> None:
    client, _, video_id = env  # the fixture's source file holds arbitrary bytes, not a video
    body = client.get(f"/videos/{video_id}/overlay", params={"start_s": 0, "end_s": 1}).json()
    assert body["width"] is None and body["height"] is None
