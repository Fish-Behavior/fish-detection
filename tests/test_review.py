"""Tests for the scene review page (fishbehavior.review) and `scene-review`, on synthetic videos."""

import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest
import yaml

from fishbehavior.cli import main
from fishbehavior.config import ConfigError, load_settings
from fishbehavior.review import (
    collect_items,
    make_server,
    render_page,
    review_hints,
    save_page_overrides,
)
from fishbehavior.roi import SceneJob, load_overrides, process_video, video_check
from synthetic import make_video

PARAMS = load_settings(environ={}).params["scene"]
SIZES = {"F_0042.avi": (320, 240), "F_0043.avi": (320, 240)}


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Two synthetic videos, a trials.csv as `validate` writes it, and scene results for both."""
    for name in ("FISH_VIDEO_DIR", "FISH_DB_PATH", "FISH_REFERENCE_PDF", "FISH_OUTPUT_DIR", "FISH_WORKERS", "FISH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in SIZES:
        make_video(videos / name, duration_s=3.0)
    catalog = tmp_path / "outputs" / "catalog"
    catalog.mkdir(parents=True)
    pd.DataFrame({
        "subject_id": ["0042", "0043"],
        "video_status": ["matched", "matched"],
        "video_paths": [str(videos / name) for name in SIZES],
    }).to_csv(catalog / "trials.csv", index=False)
    (tmp_path / ".env").write_text("FISH_OUTPUT_DIR=outputs\n")
    assert main(["scene"]) == 0
    return tmp_path


def scene_dir(project):
    return project / "outputs" / "scene"


def trials(project):
    return pd.read_csv(project / "outputs" / "catalog" / "trials.csv", dtype={"subject_id": str})


# --- page data ---------------------------------------------------------------------


def test_review_hints_point_at_suspicious_results():
    record = {"video": {"height": 240}, "flags": ["low_waterline_confidence"],
              "auto": {"waterline_y": 12, "roi": [83, 0, 237, 240]}}
    hints = review_hints(record)
    assert "waterline edge is weak" in hints
    assert "waterline at the very top of the frame" in hints
    assert "ROI reaches the top edge" in hints
    assert review_hints({"video": {"height": 240}, "flags": [], "auto": {"waterline_y": 90, "roi": [0, 70, 9, 200]}}) == []


def test_collect_items_lists_images_and_automatic_values(project):
    items, missing = collect_items(scene_dir(project), trials(project))

    assert missing == []
    assert [item["file"] for item in items] == ["F_0042.avi", "F_0043.avi"]
    item = items[0]
    assert (item["background"], item["activity"], item["frames"]) == (
        "F_0042_background.png", "F_0042_activity.png", "F_0042_frames.jpg")
    assert item["n_frames"] == PARAMS["n_review_frames"]
    assert {"waterline_y", "roi"} <= set(item["auto"]) and len(item["base_roi"]) == 4


def page_data(page):
    return json.loads(page.split('<script id="data" type="application/json">')[1].split("</script>")[0])


def test_served_page_links_images_and_file_page_embeds_them(project):
    items, _ = collect_items(scene_dir(project), trials(project))

    served = page_data(render_page(items, {}, PARAMS, serve=True))["items"][0]
    embedded = page_data(render_page(items, {}, PARAMS, serve=False, scene_dir=scene_dir(project)))["items"][0]

    assert served["frames"] == "/files/F_0042_frames.jpg"  # fetched from the local server
    assert embedded["background"].startswith("data:image/jpeg;base64,")  # works as a plain file
    assert embedded["activity"].startswith("data:image/png;base64,")
    assert embedded["frames"].startswith("data:image/jpeg;base64,")


def test_page_is_self_contained(project):
    items, _ = collect_items(scene_dir(project), trials(project))
    page = render_page(items, {"F_0042.avi": {"checked": True}, "M_0001.mp4": {"waterline_y": 5}}, PARAMS, serve=True)

    assert "__REVIEW_DATA__" not in page
    assert "http://" not in page and "https://" not in page  # no external resources
    data = page_data(page)
    assert data["overrides"] == {"F_0042.avi": {"checked": True}}
    assert data["other_overrides"] == {"M_0001.mp4": {"waterline_y": 5}}  # kept for the download button


# --- saving -------------------------------------------------------------------------


def test_save_replaces_page_entries_and_keeps_the_others(tmp_path):
    (tmp_path / "overrides.yaml").write_text("M_0001.mp4:\n  waterline_y: 50\nF_0043.avi:\n  checked: true\n")

    save_page_overrides(tmp_path, {"F_0042.avi": {"waterline_y": 180, "roi": [80, 170, 240, 224], "checked": True}}, SIZES)

    saved = load_overrides(tmp_path)
    assert saved["M_0001.mp4"] == {"waterline_y": 50}  # not on the page: untouched
    assert "F_0043.avi" not in saved  # on the page without an entry: reset to automatic
    assert saved["F_0042.avi"] == {"waterline_y": 180, "roi": [80, 170, 240, 224], "checked": True}
    assert (tmp_path / "overrides.yaml").read_text().startswith("# Manual scene corrections")


@pytest.mark.parametrize("entries", [
    {"F_0042.avi": {"waterline_y": 240}},  # below the last row
    {"F_0042.avi": {"roi": [0, 0, 321, 10]}},  # wider than the frame
    {"F_0042.avi": {"waterline_y": 10.5}},  # not whole pixels
    {"F_0042.avi": {"checked": "yes"}},
    {"F_9999.avi": {"checked": True}},  # not on the page
    ["not", "a", "mapping"],
])
def test_save_rejects_bad_values(tmp_path, entries):
    with pytest.raises(ConfigError):
        save_page_overrides(tmp_path, entries, SIZES)
    assert not (tmp_path / "overrides.yaml").exists()


def test_server_saves_and_stops(tmp_path):
    try:
        (tmp_path / "F_0042_frames.jpg").write_bytes(b"jpeg bytes")
        (tmp_path / "secret.txt").write_text("not for the page")
        server = make_server(tmp_path, "<html>page</html>", SIZES, port=0,  # port 0 = any free port
                             images={"F_0042_frames.jpg"})
    except PermissionError:
        pytest.skip("this environment does not allow opening a local port")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, data):
        request = urllib.request.Request(base + path, data=json.dumps(data).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            assert response.read() == b"<html>page</html>"
        with urllib.request.urlopen(base + "/files/F_0042_frames.jpg", timeout=5) as response:
            assert response.read() == b"jpeg bytes" and response.headers["Content-Type"] == "image/jpeg"
        for other in ("/files/secret.txt", "/files/..%2Fsecret.txt"):  # only the page's own images
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(base + other, timeout=5)
        assert post("/save", {"F_0042.avi": {"waterline_y": 100, "checked": True}}) == (200, {"ok": True, "entries": 1})
        status, reply = post("/save", {"F_0042.avi": {"waterline_y": -1}})
        assert status == 400 and "outside" in reply["error"]
        assert load_overrides(tmp_path)["F_0042.avi"]["waterline_y"] == 100  # bad save changed nothing
        assert post("/done", {})[0] == 200
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()


# --- "checked" marks ------------------------------------------------------------------


def test_checked_mark_is_recorded_without_recomputing(project):
    video = project / "videos" / "F_0042.avi"
    job = SceneJob("0042", video, scene_dir(project), PARAMS, {"checked": True}, force=False)

    record = process_video(job)

    assert record["cached"] is True and record["checked"] is True
    assert json.loads((scene_dir(project) / "F_0042.json").read_text())["checked"] is True
    assert video_check([record], PARAMS)["checked"].tolist() == [True]


# --- the command --------------------------------------------------------------------------


def test_cli_scene_review_no_serve_writes_the_page(project, capsys):
    assert main(["scene-review", "--no-serve"]) == 0
    page = (scene_dir(project) / "review.html").read_text()
    assert "F_0042.avi" in page and "F_0043.avi" in page
    assert "downloaded overrides.yaml" in capsys.readouterr().out


def test_cli_scene_review_applies_what_the_page_saved(project, monkeypatch, capsys):
    def fake_review(server, open_browser=True):
        # Stand-in for a person using the page: correct 0042, confirm 0043.
        save_page_overrides(scene_dir(project), {
            "F_0042.avi": {"waterline_y": 70, "roi": [40, 60, 280, 170], "checked": True},
            "F_0043.avi": {"checked": True},
        }, SIZES)

    class FakeServer:
        def server_close(self):
            pass

    # No real port (some sandboxes forbid it); the server itself is tested above.
    monkeypatch.setattr("fishbehavior.cli.make_server", lambda *args: FakeServer())
    monkeypatch.setattr("fishbehavior.cli.serve_review", fake_review)

    assert main(["scene-review", "--port", "0", "--no-browser"]) == 0

    assert "2 of 2 videos checked, 1 with corrections" in capsys.readouterr().out
    result = json.loads((scene_dir(project) / "F_0042.json").read_text())
    assert (result["waterline_y"], result["roi"], result["method"]) == (70, [40, 60, 280, 170], "override")
    assert result["auto"]["waterline_y"] != 70  # the automatic value is kept for "reset"
    check = pd.read_csv(scene_dir(project) / "video_check.csv")
    assert check["checked"].tolist() == [True, True]
    assert yaml.safe_load((scene_dir(project) / "overrides.yaml").read_text())["F_0043.avi"] == {"checked": True}
