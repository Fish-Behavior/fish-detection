"""Review web app backend (T066-T072) - FR-011..FR-016, PRD §9.5.8.

The API is a thin layer over `review_store`: it must not re-implement its rules, only map them to HTTP.
Accept semantics (T070): Undetermined present -> 409 always (`force` never overrides it); Dead present
without `force` -> 409 asking for confirmation; Dead with `force=true` -> accepted.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prepds.export import export_video
from prepds.features import derive_features
from prepds.models import BehaviorState as B, FrameSource, MatchStatus, ReviewFlag, StateFrame, Track, Trial, VideoAsset
from prepds.webapp.app import create_app

FPS = 30.0
N = 300
NOW = dt.datetime(2026, 9, 23, 14, 0, tzinfo=dt.timezone.utc)
VIDEO_BYTES = bytes(range(256)) * 8  # 2048 bytes: stands in for the source mp4


def _make(processed: Path, source_dir: Path, plan: list[B] | None = None, subject="0332", flags=()) -> str:
    source = source_dir / f"F_{subject}.mp4"
    source.write_bytes(VIDEO_BYTES)
    tracks = [Track(i, i / FPS, i * 2.0, 100.0, 90.0, 100.0, True) for i in range(N)]
    plan = plan or ([B.CONTROLLED_SWIM] * 100 + [B.FREEZING_DRIFT] * 100 + [B.ERRATIC_MOVEMENT] * 100)
    states = [StateFrame(i, i / FPS, plan[i], FrameSource.AUTO, None) for i in range(N)]
    trial = Trial(subject, "F", "Casper", None, "Fentanyl", "0.03", dt.date(2026, 3, 1), 20.0, source, MatchStatus.MATCHED)
    asset = VideoAsset(source, N / FPS, FPS, N, (192, 240))
    export_video(processed / f"F_{subject}", trial=trial, asset=asset, tracks=tracks, features=derive_features(tracks),
                 states=states, pipeline_version="0.1.0", calibration_profile_version="cal-2026-09-23-r2",
                 processed_at=NOW, review_flags=flags)
    return f"F_{subject}"


@pytest.fixture
def env(tmp_path: Path):
    processed, sources, gold = tmp_path / "processed", tmp_path / "videos", tmp_path / "accepted"
    processed.mkdir(), sources.mkdir()
    client = TestClient(create_app(processed, gold, clock=lambda: NOW, allowed_hosts=("testserver",)))
    return client, processed, sources, gold


def _post(client, path, **body):
    return client.post(path, json=body)


# --- listing and detail (T067) ------------------------------------------------------------


def test_lists_videos_with_status_and_filters_by_status(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources, subject="0332")
    _make(processed, sources, subject="0333")
    _post(client, "/videos/F_0333/reject", reviewer="lk")

    rows = client.get("/videos").json()
    assert [r["video_id"] for r in rows] == ["F_0332", "F_0333"]
    assert {r["video_id"]: r["review_status"] for r in rows} == {"F_0332": "PROCESSED_AUTO", "F_0333": "REJECTED"}
    assert [r["video_id"] for r in client.get("/videos", params={"status": "REJECTED"}).json()] == ["F_0333"]
    assert rows[0]["subject_id"] == "0332" and rows[0]["compound"] == "Fentanyl"


def test_unknown_status_filter_is_a_422(env) -> None:
    assert env[0].get("/videos", params={"status": "NOPE"}).status_code == 422


def test_detail_has_manifest_segments_summary_flags_and_urls(env) -> None:
    client, processed, sources, _ = env
    flag = ReviewFlag("terminal_no_action", 8.0, 10.0, "check")
    _make(processed, sources, flags=(flag,))
    body = client.get("/videos/F_0332").json()

    assert body["manifest"]["calibration_profile_version"] == "cal-2026-09-23-r2"
    assert [s["state"] for s in body["segments"]] == ["Controlled Swim", "Freezing/Drift", "Erratic Movement"]
    assert body["frame_summary"]["seconds_by_state"]["Freezing/Drift"] == pytest.approx(100 / FPS, abs=0.05)
    assert body["frame_summary"]["has_undetermined"] is False and body["frame_summary"]["has_dead"] is False
    assert body["manifest"]["review_flags"][0]["kind"] == "terminal_no_action"
    assert body["video_url"] == "/videos/F_0332/video" and body["strip_url"] == "/videos/F_0332/strip"


@pytest.mark.parametrize("bad", ["F_9999", "%2e%2e", "a%2Fb", ".hidden"])
def test_unknown_or_unsafe_ids_are_404(env, bad: str) -> None:
    assert env[0].get(f"/videos/{bad}").status_code == 404


# --- range-enabled video stream (T068) -----------------------------------------------------


def test_video_stream_supports_range_requests_for_scrubbing(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    full = client.get("/videos/F_0332/video")
    assert full.status_code == 200 and full.content == VIDEO_BYTES
    assert full.headers["accept-ranges"] == "bytes"

    part = client.get("/videos/F_0332/video", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.content == VIDEO_BYTES[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(VIDEO_BYTES)}"

    tail = client.get("/videos/F_0332/video", headers={"Range": "bytes=-16"})
    assert tail.status_code == 206 and tail.content == VIDEO_BYTES[-16:]


def test_missing_source_video_is_404_not_500(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    (sources / "F_0332.mp4").unlink()
    assert client.get("/videos/F_0332/video").status_code == 404


def test_strip_png_is_served(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    r = client.get("/videos/F_0332/strip")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:8] == b"\x89PNG\r\n\x1a\n"


# --- edits (T069) ------------------------------------------------------------------------------


def test_edit_relabels_the_range_and_returns_the_updated_detail(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    r = _post(client, "/videos/F_0332/edits", reviewer="lk",
              edits=[{"start_s": 4.0, "end_s": 6.0, "new_state": "Listing/LORR"}])
    assert r.status_code == 200
    body = r.json()
    assert body["manifest"]["review_status"] == "EDITED" and body["manifest"]["edit_count"] == 1
    listing = [s for s in body["segments"] if s["state"] == "Listing/LORR"]
    assert listing and listing[0]["source"] == "manual"
    assert listing[0]["start_s"] == pytest.approx(4.0, abs=0.05) and listing[0]["end_s"] == pytest.approx(6.0, abs=0.05)


@pytest.mark.parametrize(
    "edits",
    [
        [{"start_s": 5.0, "end_s": 5.0, "new_state": "Dead"}],  # empty range
        [{"start_s": 1.0, "end_s": 99.0, "new_state": "Surface Breach"}],  # outside the video
        [{"start_s": 1.0, "end_s": 2.0, "new_state": "Sleeping"}],  # not a label
        [{"start_s": 2.0, "end_s": 3.0, "new_state": "Dead"}],  # Dead then non-Dead
        [],
    ],
)
def test_invalid_edits_are_422_and_change_nothing(env, edits) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    assert _post(client, "/videos/F_0332/edits", reviewer="lk", edits=edits).status_code == 422
    assert client.get("/videos/F_0332").json()["manifest"]["edit_count"] == 0


def test_edit_requires_a_reviewer(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    r = _post(client, "/videos/F_0332/edits", edits=[{"start_s": 1.0, "end_s": 2.0, "new_state": "Surface Breach"}])
    assert r.status_code == 422


# --- accept (T070) ----------------------------------------------------------------------------------


def test_accept_clean_video_writes_gold_and_returns_accepted(env) -> None:
    client, processed, sources, gold = env
    _make(processed, sources)
    r = _post(client, "/videos/F_0332/accept", reviewer="lk")
    assert r.status_code == 200 and r.json()["manifest"]["review_status"] == "ACCEPTED"
    assert (gold / "F_0332" / "provenance.json").is_file() and (gold / "accepted_index.parquet").is_file()


@pytest.mark.parametrize("force", [False, True])
def test_undetermined_blocks_accept_and_force_never_overrides_it(env, force: bool) -> None:
    client, processed, sources, gold = env
    _make(processed, sources, [B.UNDETERMINED] * 30 + [B.CONTROLLED_SWIM] * (N - 30))
    r = _post(client, "/videos/F_0332/accept", reviewer="lk", force=force)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "undetermined_present"
    assert not gold.exists()


def test_dead_needs_confirmation_and_force_allows_it(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources, [B.CONTROLLED_SWIM] * 200 + [B.DEAD] * 100)
    blocked = _post(client, "/videos/F_0332/accept", reviewer="lk")
    assert blocked.status_code == 409 and blocked.json()["detail"]["code"] == "dead_confirmation_required"
    ok = _post(client, "/videos/F_0332/accept", reviewer="lk", force=True)
    assert ok.status_code == 200 and ok.json()["manifest"]["review_status"] == "ACCEPTED"


def test_accepting_twice_or_editing_after_accept_is_409(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    _post(client, "/videos/F_0332/accept", reviewer="lk")
    assert _post(client, "/videos/F_0332/accept", reviewer="lk").status_code == 409
    edit = _post(client, "/videos/F_0332/edits", reviewer="lk", edits=[{"start_s": 1.0, "end_s": 2.0, "new_state": "Surface Breach"}])
    assert edit.status_code == 409


# --- reject (T071) --------------------------------------------------------------------------------------


def test_reject_clears_edits_and_requeues(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    _post(client, "/videos/F_0332/edits", reviewer="lk", edits=[{"start_s": 1.0, "end_s": 2.0, "new_state": "Surface Breach"}])
    r = _post(client, "/videos/F_0332/reject", reviewer="lk")
    assert r.status_code == 200
    body = r.json()
    assert body["manifest"]["review_status"] == "REJECTED" and body["manifest"]["edit_count"] == 0
    assert body["segments"] == [] and body["strip_url"] is None


def test_rejecting_an_accepted_video_is_409(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    _post(client, "/videos/F_0332/accept", reviewer="lk")
    assert _post(client, "/videos/F_0332/reject", reviewer="lk").status_code == 409


def test_an_empty_or_missing_processed_directory_lists_no_videos(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "does-not-exist", tmp_path / "accepted", allowed_hosts=("testserver",)))
    assert client.get("/videos").json() == []


def test_concurrent_edits_are_serialised_and_all_counted(env) -> None:
    from concurrent.futures import ThreadPoolExecutor

    client, processed, sources, _ = env
    _make(processed, sources)

    def edit(i: int) -> int:
        return _post(client, "/videos/F_0332/edits", reviewer="lk",
                     edits=[{"start_s": float(i), "end_s": float(i) + 1.0, "new_state": "Surface Breach"}]).status_code

    with ThreadPoolExecutor(6) as pool:
        assert list(pool.map(edit, range(6))) == [200] * 6
    assert client.get("/videos/F_0332").json()["manifest"]["edit_count"] == 6


# --- hardening from code review ---------------------------------------------------------------------------


def test_requests_with_a_foreign_host_header_are_refused_dns_rebinding(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    assert client.get("/videos", headers={"host": "evil.example"}).status_code == 400


def test_state_changing_requests_from_a_foreign_origin_are_refused(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    r = client.post("/videos/F_0332/reject", json={"reviewer": "lk"}, headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    assert client.get("/videos/F_0332").json()["manifest"]["review_status"] == "PROCESSED_AUTO"
    same = client.post("/videos/F_0332/reject", json={"reviewer": "lk"}, headers={"origin": "http://testserver"})
    assert same.status_code == 200


def test_a_corrupt_manifest_does_not_break_the_list_and_never_leaks_a_path(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources, subject="0332")
    _make(processed, sources, subject="0333")
    (processed / "F_0333" / "manifest.json").write_text("{not json")
    assert [r["video_id"] for r in client.get("/videos").json()] == ["F_0332"]
    r = client.get("/videos/F_0333")
    assert r.status_code == 500 and str(processed) not in r.text


@pytest.mark.parametrize("artifact", ["frames.parquet", "strip.png"])
def test_missing_artifacts_on_edit_or_accept_are_a_409_not_a_500(env, artifact: str) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    (processed / "F_0332" / artifact).unlink()
    accept_r = _post(client, "/videos/F_0332/accept", reviewer="lk")
    assert accept_r.status_code == 409 and accept_r.json()["detail"]["code"] == "missing_artifacts"
    if artifact == "frames.parquet":
        edit_r = _post(client, "/videos/F_0332/edits", reviewer="lk", edits=[{"start_s": 1.0, "end_s": 2.0, "new_state": "Surface Breach"}])
        assert edit_r.status_code == 409 and str(processed) not in edit_r.text


def test_an_oversized_edit_list_is_rejected(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    edits = [{"start_s": 1.0, "end_s": 2.0, "new_state": "Surface Breach"}] * 1001
    assert _post(client, "/videos/F_0332/edits", reviewer="lk", edits=edits).status_code == 422


@pytest.mark.parametrize("reviewer", ["   ", "lk\nadmin", ""])
def test_blank_or_control_character_reviewer_names_are_rejected(env, reviewer: str) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    assert _post(client, "/videos/F_0332/reject", reviewer=reviewer).status_code == 422


def test_the_source_video_must_live_under_the_configured_video_root(tmp_path: Path) -> None:
    processed, sources, outside = tmp_path / "processed", tmp_path / "videos", tmp_path / "secrets"
    processed.mkdir(), sources.mkdir(), outside.mkdir()
    _make(processed, sources)
    (outside / "passwords.mp4").write_bytes(b"secret")
    manifest = processed / "F_0332" / "manifest.json"
    import json as _json
    data = _json.loads(manifest.read_text())
    client = TestClient(create_app(processed, tmp_path / "accepted", video_dir=sources, allowed_hosts=("testserver",)))

    assert client.get("/videos/F_0332/video").status_code == 200  # inside the root
    for evil in (str(outside / "passwords.mp4"), "../secrets/passwords.mp4"):
        data["video_path"] = evil
        manifest.write_text(_json.dumps(data))
        assert client.get("/videos/F_0332/video").status_code == 404


def test_a_source_that_is_not_a_video_file_is_not_served(env) -> None:
    client, processed, sources, _ = env
    _make(processed, sources)
    import json as _json
    manifest = processed / "F_0332" / "manifest.json"
    data = _json.loads(manifest.read_text())
    (sources / "notes.txt").write_text("x")
    data["video_path"] = str(sources / "notes.txt")
    manifest.write_text(_json.dumps(data))
    assert client.get("/videos/F_0332/video").status_code == 404


def test_ids_with_double_dots_inside_a_name_are_still_valid(env) -> None:
    client, processed, sources, _ = env
    assert client.get("/videos/a..b").status_code == 404  # valid shape, simply not there


def test_index_page_and_static_assets_are_served(env):
    client = env[0]
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "/static/app.js" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_static_mount_does_not_expose_paths_outside_static(env):
    client = env[0]
    assert client.get("/static/../app.py").status_code in (404, 400)
    assert client.get("/static/%2e%2e/app.py").status_code == 404
