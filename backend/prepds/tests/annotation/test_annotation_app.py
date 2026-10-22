"""Labeling web app backend (Phase 15)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from prepds.annotation.app import create_annotation_app

NOW = dt.datetime(2026, 9, 23, tzinfo=dt.timezone.utc)
SAMPLES = [
    {"video_id": "F_0001", "frame_idx": 10, "t_sec": 0.3, "reason": "gap_1_10s", "split": "train", "compound": "Veh",
     "image": "F_0001_000010.png", "classical": {"detected": False, "x": 0.0, "y": 0.0}},
    {"video_id": "F_0002", "frame_idx": 20, "t_sec": 0.6, "reason": "detected", "split": "train", "compound": "MDMA",
     "image": "F_0002_000020.png", "classical": {"detected": True, "x": 10.0, "y": 20.0}},
    {"video_id": "F_0003", "frame_idx": 30, "t_sec": 1.0, "reason": "lorr_candidate", "split": "heldout", "compound": "Fentanyl",
     "image": "F_0003_000030.png", "classical": {"detected": False, "x": 0.0, "y": 0.0}},
]
PREFILL = {"F_0001_000010.png": {"score": 0.7, "box": [40.0, 50.0, 200.0, 140.0]}, "F_0002_000020.png": None}
GOOD = {"fish_visible": True, "box": [40, 50, 200, 140], "keypoints": {"snout": [45, 90, 2]}, "listing": "no", "annotator": "lk"}


@pytest.fixture
def env(tmp_path: Path):
    work = tmp_path / "phase15"
    (work / "frames").mkdir(parents=True)
    for s in SAMPLES:
        Image.new("RGB", (320, 240), (120, 130, 100)).save(work / "frames" / s["image"])
    (work / "sample.json").write_text(json.dumps({"seed": 15, "samples": SAMPLES}))
    (work / "prefill.json").write_text(json.dumps(PREFILL))
    client = TestClient(create_annotation_app(work, clock=lambda: NOW, allowed_hosts=("testserver",)))
    return client, work


def test_list_shows_status_and_supports_filters(env) -> None:
    client, _ = env
    rows = client.get("/api/frames").json()
    assert [r["id"] for r in rows] == ["F_0001_000010", "F_0002_000020", "F_0003_000030"]
    assert {r["status"] for r in rows} == {"todo"}
    assert [r["id"] for r in client.get("/api/frames?split=heldout").json()] == ["F_0003_000030"]
    client.put("/api/frames/F_0001_000010", json=GOOD)
    assert [r["id"] for r in client.get("/api/frames?status=done").json()] == ["F_0001_000010"]


def test_detail_carries_prefill_dimensions_and_schema(env) -> None:
    client, _ = env
    d = client.get("/api/frames/F_0001_000010").json()
    assert d["width"] == 320 and d["height"] == 240 and d["annotation"] is None
    assert d["prefill"]["box"] == [40.0, 50.0, 200.0, 140.0] and d["reason"] == "gap_1_10s"
    assert d["keypoint_names"] == ["snout", "dorsal_fin_base", "ventral", "tail_base", "tail_tip"]
    assert client.get("/api/frames/F_0002_000020").json()["prefill"] is None


def test_image_is_served_as_png(env) -> None:
    client, _ = env
    r = client.get("/api/frames/F_0001_000010/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"


def test_saving_stores_the_annotation_with_server_time_and_survives_reload(env) -> None:
    client, work = env
    r = client.put("/api/frames/F_0001_000010", json=GOOD)
    assert r.status_code == 200 and r.json()["annotation"]["annotated_at"] == NOW.isoformat()
    assert (work / "annotations" / "F_0001_000010.json").is_file()
    assert client.get("/api/frames/F_0001_000010").json()["annotation"]["keypoints"]["snout"] == [45.0, 90.0, 2]


def test_invalid_annotations_are_422_and_store_nothing(env) -> None:
    client, work = env
    r = client.put("/api/frames/F_0001_000010", json={**GOOD, "box": [40, 50, 900, 140]})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_annotation"
    assert not (work / "annotations" / "F_0001_000010.json").exists()
    assert client.put("/api/frames/F_0001_000010", json={"fish_visible": True}).status_code == 422


def test_no_fish_visible_is_a_done_status(env) -> None:
    client, _ = env
    client.put("/api/frames/F_0002_000020", json={"fish_visible": False, "annotator": "lk"})
    row = next(r for r in client.get("/api/frames").json() if r["id"] == "F_0002_000020")
    assert row["status"] == "no_fish"


@pytest.mark.parametrize("bad", ["../etc/passwd", "F_9999_000001", "%2e%2e", ".hidden"])
def test_unknown_or_unsafe_ids_are_404(env, bad: str) -> None:
    client, _ = env
    assert client.get(f"/api/frames/{bad}").status_code == 404
    assert client.get(f"/api/frames/{bad}/image").status_code == 404
    assert client.put(f"/api/frames/{bad}", json=GOOD).status_code == 404


def test_progress_counts_by_split_status_and_listing_tag(env) -> None:
    client, _ = env
    client.put("/api/frames/F_0001_000010", json={**GOOD, "listing": "yes"})
    client.put("/api/frames/F_0002_000020", json={"fish_visible": False, "annotator": "lk"})
    p = client.get("/api/progress").json()
    assert p["total"] == 3 and p["done"] == 2 and p["by_split"]["train"] == {"total": 2, "done": 2}
    assert p["listing"] == {"yes": 1, "no": 0, "unsure": 0}


def test_coco_export_per_split(env) -> None:
    client, _ = env
    client.put("/api/frames/F_0001_000010", json=GOOD)
    coco = client.get("/api/export/coco/train").json()
    assert len(coco["images"]) == 1 and coco["annotations"][0]["bbox"] == [40.0, 50.0, 160.0, 90.0]
    assert client.get("/api/export/coco/validation").status_code == 404


def test_foreign_host_and_origin_are_refused(env) -> None:
    client, _ = env
    assert client.get("/api/frames", headers={"host": "evil.example"}).status_code == 400
    assert client.put("/api/frames/F_0001_000010", json=GOOD, headers={"origin": "http://evil.example"}).status_code == 403


def test_index_page_is_served(env) -> None:
    client, _ = env
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


def test_corrupt_annotation_file_gives_a_500_without_paths(env) -> None:
    client, work = env
    (work / "annotations").mkdir()
    (work / "annotations" / "F_0001_000010.json").write_text("{nope")
    r = client.get("/api/frames/F_0001_000010")
    assert r.status_code == 500 and str(work) not in r.text
    assert len(client.get("/api/frames").json()) == 3  # the list survives one bad file


def test_export_refuses_when_an_annotation_is_unreadable_instead_of_dropping_it(env) -> None:
    client, work = env
    client.put("/api/frames/F_0001_000010", json=GOOD)
    (work / "annotations" / "F_0002_000020.json").write_text("{nope")
    r = client.get("/api/export/coco/train")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "corrupt_annotations"
    assert "F_0002_000020" in r.json()["detail"]["ids"] and str(work) not in r.text


def test_corrupt_frames_can_be_found_and_relabeled(env) -> None:
    client, work = env
    (work / "annotations").mkdir()
    (work / "annotations" / "F_0002_000020.json").write_text("{nope")
    assert [r["id"] for r in client.get("/api/frames?status=corrupt").json()] == ["F_0002_000020"]
    assert client.put("/api/frames/F_0002_000020", json=GOOD).status_code == 200  # overwriting repairs it


def test_export_uses_each_frames_own_size(env) -> None:
    client, work = env
    Image.new("RGB", (192, 240)).save(work / "frames" / "F_0002_000020.png")
    client.put("/api/frames/F_0001_000010", json=GOOD)
    client.put("/api/frames/F_0002_000020", json={**GOOD, "box": [10, 10, 100, 100], "keypoints": {}})
    sizes = [(i["width"], i["height"]) for i in client.get("/api/export/coco/train").json()["images"]]
    assert sizes == [(320, 240), (192, 240)]


def _add_batch(work: Path):
    sample = {"video_id": "F_0009", "frame_idx": 90, "t_sec": 3.0, "reason": "listing_candidate", "split": "heldout",
              "compound": "DOB", "image": "F_0009_000090.png", "classical": {"detected": False, "x": 0.0, "y": 0.0},
              "prefill": {"score": 0.8, "box": [30.0, 40.0, 210.0, 150.0]}}
    Image.new("RGB", (320, 240), (120, 130, 100)).save(work / "frames" / sample["image"])
    (work / "batch_002.json").write_text(json.dumps({"seed": 1, "samples": [sample]}))


def test_batch_frames_are_listed_with_their_batch_prefill_and_reason(env) -> None:
    _, work = env
    _add_batch(work)
    client = TestClient(create_annotation_app(work, clock=lambda: NOW, allowed_hosts=("testserver",)))
    rows = client.get("/api/frames").json()
    assert len(rows) == 4 and rows[-1]["id"] == "F_0009_000090" and rows[-1]["batch"] == "batch_002"
    assert rows[-1]["has_prefill"] is True
    detail = client.get("/api/frames/F_0009_000090").json()
    assert detail["prefill"]["box"] == [30.0, 40.0, 210.0, 150.0] and detail["batch"] == "batch_002"


def test_frames_can_be_filtered_by_batch_and_reason(env) -> None:
    _, work = env
    _add_batch(work)
    client = TestClient(create_annotation_app(work, clock=lambda: NOW, allowed_hosts=("testserver",)))
    assert [r["id"] for r in client.get("/api/frames?batch=batch_002").json()] == ["F_0009_000090"]
    assert [r["id"] for r in client.get("/api/frames?reason=listing_candidate").json()] == ["F_0009_000090"]
    assert len(client.get("/api/frames?batch=sample").json()) == 3


def test_progress_reports_each_batch_and_exports_include_batch_frames(env) -> None:
    _, work = env
    _add_batch(work)
    client = TestClient(create_annotation_app(work, clock=lambda: NOW, allowed_hosts=("testserver",)))
    client.put("/api/frames/F_0009_000090", json=GOOD)
    p = client.get("/api/progress").json()
    assert p["by_batch"]["batch_002"] == {"total": 1, "done": 1} and p["by_batch"]["sample"]["done"] == 0
    assert [i["file_name"] for i in client.get("/api/export/coco/heldout").json()["images"]] == ["F_0009_000090.png"]
