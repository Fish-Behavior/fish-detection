"""Tests for fishbehavior.live: processing a synthetic video step by step, and the local server."""

import http.client
import json
import threading
import time
import urllib.error
import urllib.request

import pandas as pd
import pytest
import yaml

from fishbehavior.cli import main
from fishbehavior.config import load_settings
from fishbehavior.live import (
    LABELS, OUTPUT_FILES, Session, Source, make_server, open_source, resolve_source, safe_name, start_processing,
)
from synthetic import make_video

DURATION_S = 20.0


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """One 20 s synthetic video, run through the batch steps (track, features, label) once."""
    root = tmp_path_factory.mktemp("live")
    make_video(root / "F_0042.avi", duration_s=DURATION_S)
    catalog = root / "outputs" / "catalog"
    catalog.mkdir(parents=True)
    pd.DataFrame({"subject_id": ["0042"], "video_status": ["matched"], "compound": ["CPD_A"],
                  "concentration_raw": ["0.03"], "video_paths": [str(root / "F_0042.avi")]}).to_csv(
        catalog / "trials.csv", index=False)
    (root / ".env").write_text(f"FISH_OUTPUT_DIR={root / 'outputs'}\nFISH_WORKERS=1\n")  # absolute: no chdir needed
    # Frequent relabeling and a paced run, so the test sees several live updates.
    (root / "config.yaml").write_text(yaml.safe_dump({"live": {"update_s": 0.2, "preview_fps": 10}}))
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        for name in ("FISH_VIDEO_DIR", "FISH_DB_PATH", "FISH_REFERENCE_PDF", "FISH_OUTPUT_DIR", "FISH_CONFIG"):
            patch.delenv(name, raising=False)
        env = ["--env-file", str(root / ".env")]
        assert main([*env, "track"]) == 0 and main([*env, "features"]) == 0 and main([*env, "label"]) == 0
    return root, load_settings(env_file=root / ".env", config_file=root / "config.yaml")


def wait(session, timeout=60.0):
    """Collect state snapshots while the processing thread runs; returns them."""
    seen, end = [], time.time() + timeout
    while session.busy() and time.time() < end:
        seen.append(session.snapshot())
        time.sleep(0.05)
    assert not session.busy(), "processing did not finish"
    return seen + [session.snapshot()]


def test_live_labels_grow_while_reading_and_end_equal_to_the_batch_labels(project):
    root, settings = project
    session = Session()
    open_source(settings, session, resolve_source(settings, subject_id="42"))
    assert session.state["stage"] == "scene" and session.state["parts"][0]["file"] == "F_0042.avi"

    start_processing(settings, session, None, None, pace=10.0)  # 20 s of video in about 2 s
    states = wait(session)

    lengths = [len(s["labels"]) for s in states if s.get("labels")]
    assert len(set(lengths)) >= 3 and lengths == sorted(lengths)  # labels appeared second by second
    assert any(s.get("labeling", {}).get("provisional_from_s") is not None for s in states if s["stage"] == "labels")
    final = states[-1]
    assert final["stage"] == "saved" and final["labeling"]["provisional_from_s"] is None
    # While playing: the BL of the earlier `features` run; the final pass measures it on the track.
    batch_bl = json.loads((root / "outputs" / "features" / "0042_features.json").read_text())["endpoint"]["body_length_px"]
    running = [s["features"] for s in states if s["stage"] == "labels"]
    assert running and all(f["body_length_px"] == batch_bl and "earlier" in f["body_length_from"] for f in running)
    assert final["features"]["body_length_px"] == pytest.approx(batch_bl, abs=0.01)
    assert "whole track" in final["features"]["body_length_from"]
    batch = pd.read_csv(root / "outputs" / "labels" / "0042_bins.csv")
    assert [LABELS[i] for i in final["labels"]] == batch["label"].tolist()  # identical to `all` for this video
    assert final["confidence"] == pytest.approx(batch["confidence"].tolist(), abs=1e-3)
    out = root / "outputs" / "live" / "0042"
    assert all((out / name).is_file() for name in OUTPUT_FILES)
    live_segments = pd.read_csv(out / "segments.csv")
    batch_segments = pd.read_csv(root / "outputs" / "labels" / "0042_segments.csv")
    assert live_segments[["label", "start_s", "end_s"]].equals(batch_segments[["label", "start_s", "end_s"]])
    assert sum(v["s"] for v in final["summary"].values()) == pytest.approx(DURATION_S, abs=0.1)
    assert session.frame is not None and session.frame["jpeg"]  # preview frames were sent


def test_page_corrections_to_the_scene_are_used(project):
    _, settings = project
    session = Session()
    open_source(settings, session, resolve_source(settings, subject_id="0042"))
    auto = session.state["parts"][0]
    start_processing(settings, session, auto["waterline_y"] + 3, auto["roi"], pace=0)
    final = wait(session)[-1]
    assert final["parts"][0]["waterline_y"] == auto["waterline_y"] + 3 and final["parts"][0]["method"] == "override"
    assert any("corrected on the page" in line for line in final["log"])


def test_a_new_video_is_provisional_throughout_until_the_final_pass(project):
    root, settings = project
    session = Session()
    open_source(settings, session, Source("new", [root / "F_0042.avi"]))  # not in the catalog: no known BL
    start_processing(settings, session, None, None, pace=10.0)
    states = wait(session)
    running = [s for s in states if s["stage"] == "labels"]
    assert running and all(s["labeling"]["provisional_from_s"] == 0.0 for s in running)
    assert all("so far" in s["features"]["body_length_from"] for s in running)
    assert states[-1]["stage"] == "saved" and states[-1]["labeling"]["provisional_from_s"] is None


def test_upload_names_cannot_leave_the_upload_folder():
    assert safe_name("../../etc/passwd.mp4") == "passwd.mp4"
    assert safe_name("my video (1).mp4") == "my_video_1_.mp4"
    assert safe_name("..") == "video"


@pytest.fixture
def server(project):
    """The live server on a free port, in a background thread."""
    _, settings = project
    try:
        httpd = make_server(settings, 0)
    except PermissionError:  # same as test_review: some sandboxes forbid local ports
        pytest.skip("this environment does not allow opening a local port")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def request(url, data=None, method=None):
    """(status, body bytes) of one request; JSON bodies for dicts."""
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    req = urllib.request.Request(url, data=body, method=method or ("POST" if data is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_server_upload_open_process_review_and_download(server, project):
    httpd, base = server
    root, _ = project
    status, page = request(base + "/")
    assert status == 200 and b"Ethogram" in page and b"/*CONFIG*/null" not in page  # configuration filled in
    assert json.loads(request(base + "/api/sources")[1])["catalog"][0]["subject_id"] == "0042"

    video = (root / "F_0042.avi").read_bytes()
    assert request(base + "/api/upload?name=notes.txt", video)[0] == 400  # not a video
    status, body = request(base + "/api/upload?name=..%2FF_0042.avi", video)
    assert status == 200 and json.loads(body)["upload"] == "F_0042.avi"
    assert (root / "outputs" / "live" / "uploads" / "F_0042.avi").stat().st_size == len(video)

    assert request(base + "/api/open", {"upload": "F_0042.avi"})[0] == 200
    state = json.loads(request(base + "/api/state")[1])
    assert state["source"]["subject_id"] == "0042" and state["source"]["details"]["compound"] == "CPD_A"  # matched by name

    # The event stream sends the current state right away.
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
    connection.request("GET", "/api/events")
    first = connection.getresponse().read1(65536).decode()
    connection.close()
    assert first.startswith("event: state") and '"stage": "scene"' in first

    assert request(base + "/api/start", {"pace": 0})[0] == 200
    wait(httpd.session)
    assert json.loads(request(base + "/api/state")[1])["stage"] == "saved"
    frame = json.loads(request(base + "/api/frame?t=5")[1])
    assert abs(frame["t"] - 5) < 0.1 and frame["jpeg"] and frame["part"] == 1
    status, body = request(base + "/files/segments.csv")
    assert status == 200 and body.startswith(b"subject_id,label")
    assert request(base + "/files/..%2F..%2Fcatalog%2Ftrials.csv")[0] == 404  # only the result files
    assert request(base + "/api/start", {"pace": "fast"})[0] == 400  # bad input: an error, not a crash
