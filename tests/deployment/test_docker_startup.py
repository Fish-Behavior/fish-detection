"""The Docker gate must finish preparation before it advertises a working app."""
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts import docker_app


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("videos", "outputs", "accepted"):
        (tmp_path / name).mkdir()
    (tmp_path / "videos/GT-1-42").mkdir()
    shutil.copy(Path(__file__).parents[1] / "fixtures/synth_tiny.mp4", tmp_path / "videos/GT-1-42/F_0022.mp4")
    shutil.copy(Path(__file__).parents[1] / "fixtures/synth_db.xlsx", tmp_path / "database.xlsx")
    monkeypatch.setenv("PDS_VIDEO_DIR", "videos")
    monkeypatch.setenv("PDS_DB_PATH", "database.xlsx")
    monkeypatch.setenv("PDS_OUTPUT_DIR", "outputs")
    monkeypatch.setenv("PDS_ACCEPTED_DIR", "accepted")
    monkeypatch.setenv("PDS_WORKERS", "1")
    monkeypatch.setenv("FISHLAB_DCS_ENABLED", "0")
    monkeypatch.setenv("FISHLAB_HOST", "127.0.0.1")
    monkeypatch.setenv("FISHLAB_PORT", "8000")
    for key in ("PDS_REFERENCE_DIR", "PDS_CONFIG", "FISHLAB_CHAT_URL"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def test_real_prepds_prepare_resumes_and_preserves_review(project):
    assert docker_app.prepare() == 0
    manifest = project / "outputs/processed/F_0022/manifest.json"
    before = manifest.read_bytes()
    assert docker_app.prepare() == 0
    assert manifest.read_bytes() == before
    state = json.loads((project / "outputs/docker/startup.json").read_text())
    assert state["ok"] is True
    assert "prepds run" in state["completed"]
    assert "DCS" in state["skipped"]


def test_startup_does_not_require_read_or_rebuild_accepted_index(project):
    index = project / "accepted/accepted_index.parquet"
    index.write_bytes(b"existing index is outside the startup checks")
    before = index.read_bytes()
    assert docker_app.prepare() == 0
    assert index.read_bytes() == before
    state = json.loads((project / "outputs/docker/startup.json").read_text())
    assert "prepds export-index" not in state["completed"]


def test_failed_step_blocks_app_and_records_unfinished_steps(project, monkeypatch, capsys):
    original = docker_app.command

    def fail_run(package, *args):
        if args == ("run",):
            raise RuntimeError("F_0001: unreadable video")
        return original(package, *args)

    monkeypatch.setattr(docker_app, "command", fail_run)
    assert docker_app.prepare() == 1
    state = json.loads((project / "outputs/docker/startup.json").read_text())
    assert state["ok"] is False
    assert state["failed"] == "prepds run"
    assert "prepds outputs" in state["pending"]
    assert "F_0001" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="preparation"):
        docker_app.review_app()


def test_missing_required_config_and_dcs_source_fail_before_processing(project, monkeypatch):
    monkeypatch.setenv("PDS_DB_PATH", "missing.xlsx")
    assert docker_app.prepare() == 1
    state_path = project / "outputs/docker/startup.json"
    assert json.loads(state_path.read_text())["failed"] == "configuration"
    monkeypatch.setenv("PDS_DB_PATH", "database.xlsx")
    monkeypatch.setenv("FISHLAB_DCS_ENABLED", "1")
    monkeypatch.setattr(docker_app, "CLASSIFIER_ROOT", project / "absent-classifier")
    assert docker_app.prepare() == 1
    assert json.loads(state_path.read_text())["failed"] == "configuration"
    assert not (project / "outputs/trials_catalog.parquet").exists()


def test_dcs_reuses_model_or_trains_when_missing_and_never_borrows_stale_predictions(project, monkeypatch):
    root = project / "classifier"
    (root / "src/dcs").mkdir(parents=True)
    (root / "src/dcs/__main__.py").touch()
    monkeypatch.setattr(docker_app, "CLASSIFIER_ROOT", root)
    monkeypatch.setenv("FISHLAB_DCS_ENABLED", "1")
    calls = []
    paths = type("Paths", (), {"output_dir": project / "outputs/dcs", "table": project / "outputs/dcs/table.parquet"})()
    monkeypatch.setattr(docker_app, "dcs_settings", lambda: type("Settings", (), {"paths": paths})())

    def command(package, *args):
        calls.append((package, *args))
        if args == ("train",):
            model = paths.output_dir / "training/new-run/model"
            model.mkdir(parents=True)
            (model / "model_info.json").write_text('{}')
        if args and args[0] == "predict":
            Path(args[args.index("--out") + 1]).write_text('video_id,predicted,p:Veh\nF_0001,Veh,1\n')

    monkeypatch.setattr(docker_app, "command", command)
    monkeypatch.setattr(docker_app, "validate_outputs", lambda settings: None)
    assert docker_app.prepare() == 0
    assert ("dcs", "train") in calls
    calls.clear()
    assert docker_app.prepare() == 0
    assert ("dcs", "train") not in calls
    assert ("dcs", "featurize") in calls
    assert any(call[:2] == ("dcs", "predict") for call in calls)


def test_runtime_keeps_host_and_origin_guards_and_serves_built_frontend(project, monkeypatch):
    from fastapi.testclient import TestClient

    frontend = project / "frontend"
    (frontend / "assets").mkdir(parents=True)
    (frontend / "index.html").write_text('<html>FishLab Docker</html>')
    monkeypatch.setattr(docker_app, "FRONTEND_DIR", frontend)
    assert docker_app.prepare() == 0
    client = TestClient(docker_app.review_app(), base_url="http://127.0.0.1")
    assert "FishLab Docker" in client.get("/").text
    assert client.get("/api/health").json()["ok"]
    assert client.get("/api/sessions", headers={"Host": "evil.example"}).status_code == 400
    assert client.post("/api/ask", headers={"Origin": "https://evil.example"}).status_code == 403


def test_partial_reviewed_output_is_reported_without_overwriting_it(project):
    assert docker_app.prepare() == 0
    directory = project / "outputs/processed/F_0022"
    before = (directory / "manifest.json").read_bytes()
    (directory / "frames.parquet").unlink()
    assert docker_app.prepare() == 1
    assert (directory / "manifest.json").read_bytes() == before
    state = json.loads((project / "outputs/docker/startup.json").read_text())
    assert state["failed"] == "configuration"
    assert "frames.parquet" in state["error"]


def test_model_bridge_forwards_only_to_the_docker_host(monkeypatch):
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from urllib.request import ProxyHandler, build_opener

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"local-test-model"}]}')

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original = socket.create_connection
    requested = []

    def connect(address, *args, **kwargs):
        if address[0] == docker_app.HOST_GATEWAY:
            requested.append(address)
            address = ("127.0.0.1", server.server_port)
        return original(address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", connect)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}/v1"
    try:
        with docker_app.model_bridge(url):
            with build_opener(ProxyHandler({})).open(url + "/models") as response:
                assert json.load(response)["data"][0]["id"] == "local-test-model"
        assert requested == [(docker_app.HOST_GATEWAY, port)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_backup_contains_saved_work_and_config_without_source_videos(project, monkeypatch):
    (project / ".env").write_text("PDS_OUTPUT_DIR=outputs\n")
    (project / "config").mkdir()
    (project / "config/custom.yaml").write_text("training: {}\n")
    (project / "outputs/processed/F_0022").mkdir(parents=True)
    (project / "outputs/processed/F_0022/review.json").write_text('{"revision": 2}')
    (project / "accepted/accepted_index.parquet").write_bytes(b"saved gold")
    (project / "dcs-results").mkdir()
    (project / "dcs-results/model.joblib").write_bytes(b"saved model")
    monkeypatch.setenv("DCS_OUTPUT_DIR", "dcs-results")

    archive = docker_app.backup()
    assert archive.is_file() and archive.stat().st_mode & 0o777 == 0o600
    with tarfile.open(archive) as saved:
        assert saved.extractfile("outputs/processed/F_0022/review.json").read() == b'{"revision": 2}'
        assert saved.extractfile("accepted/accepted_index.parquet").read() == b"saved gold"
        assert saved.extractfile("dcs-results/model.joblib").read() == b"saved model"
        assert saved.extractfile(".env").read() == b"PDS_OUTPUT_DIR=outputs\n"
        assert saved.extractfile("config/custom.yaml").read() == b"training: {}\n"
        assert not any(name.startswith("videos/") for name in saved.getnames())


def test_backup_failure_leaves_no_partial_archive(project, monkeypatch):
    (project / ".env").touch()
    (project / "config").mkdir()
    monkeypatch.setenv("DCS_OUTPUT_DIR", str(project.parent / "outside"))
    with pytest.raises(ValueError, match="outside"):
        docker_app.backup()
    assert not list((project / "backups").glob("*.tar"))


@pytest.mark.parametrize("failure", ["", "prepare", "app"])
def test_shell_prints_link_only_after_success_and_releases_its_lock(tmp_path, failure):
    import os

    root = Path(__file__).parents[2]
    shutil.copy(root / "start.sh", tmp_path / "start.sh")
    (tmp_path / "videos").mkdir()
    (tmp_path / "db.xlsx").touch()
    (tmp_path / ".env").write_text("PDS_VIDEO_DIR=videos\nPDS_DB_PATH=db.xlsx\nFISHLAB_HOST=127.0.0.1\nFISHLAB_PORT=8003\n")
    tools = tmp_path / "bin"
    tools.mkdir()
    docker = tools / "docker"
    docker.write_text('''#!/bin/sh
echo "$*" >> "$DOCKER_CALLS"
case "$*" in
  "compose up --no-build"*) [ "$FAIL_STAGE" != prepare ] || exit 7 ;;
  "compose up -d"*) [ "$FAIL_STAGE" != app ] || exit 9 ;;
esac
exit 0
''')
    docker.chmod(0o755)
    log = tmp_path / "calls.txt"
    result = subprocess.run(["sh", str(tmp_path / "start.sh")], text=True, capture_output=True,
                            env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "DOCKER_CALLS": str(log), "FAIL_STAGE": failure})
    calls = log.read_text()
    assert not (tmp_path / ".fishlab-launch.lock").exists()
    if failure:
        assert result.returncode != 0
        assert "FAILED:" in result.stderr
        assert "FishLab ready:" not in result.stdout
        if failure == "prepare":
            assert "compose up -d" not in calls
        else:
            assert calls.count("compose stop app") == 2
    else:
        assert result.returncode == 0
        assert "FishLab ready: http://127.0.0.1:8003" in result.stdout
        assert calls.index("--exit-code-from prepare") < calls.index("--wait --wait-timeout")


def test_start_lists_missing_setup_without_starting_anything(tmp_path):
    import os

    root = Path(__file__).parents[2]
    shutil.copy(root / "start.sh", tmp_path / "start.sh")
    (tmp_path / ".env").write_text("PDS_OUTPUT_DIR=outputs\n")
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "docker").write_text('#!/bin/sh\necho "$*" >> "$DOCKER_CALLS"\n')
    (tools / "docker").chmod(0o755)
    log = tmp_path / "calls.txt"
    result = subprocess.run(["sh", str(tmp_path / "start.sh")], text=True, capture_output=True,
                            env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "DOCKER_CALLS": str(log)})
    assert result.returncode != 0
    assert "not configured yet" in result.stderr and "PDS_VIDEO_DIR" in result.stderr and "PDS_DB_PATH" in result.stderr
    assert "compose build" not in log.read_text()


@pytest.mark.parametrize("backup_fails", [False, True])
def test_stop_backs_up_before_compose_down(tmp_path, backup_fails):
    import os

    root = Path(__file__).parents[2]
    shutil.copy(root / "stop.sh", tmp_path / "stop.sh")
    (tmp_path / ".env").touch()
    tools = tmp_path / "bin"
    tools.mkdir()
    docker = tools / "docker"
    docker.write_text('''#!/bin/sh
echo "$*" >> "$DOCKER_CALLS"
case "$*" in
  "compose run"*) [ "$BACKUP_FAILS" != 1 ] || exit 7 ;;
esac
exit 0
''')
    docker.chmod(0o755)
    log = tmp_path / "calls.txt"
    result = subprocess.run(["sh", str(tmp_path / "stop.sh")], text=True, capture_output=True,
                            env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}",
                                 "DOCKER_CALLS": str(log), "BACKUP_FAILS": "1" if backup_fails else "0"})
    calls = log.read_text()
    assert "compose stop -t 300 app" in calls
    assert not (tmp_path / ".fishlab-launch.lock").exists()
    if backup_fails:
        assert result.returncode != 0
        assert "compose down" not in calls
        assert "Backup failed" in result.stderr
    else:
        assert result.returncode == 0
        assert calls.index("compose stop -t 300 app") < calls.index(" backup") < calls.index("compose down")
