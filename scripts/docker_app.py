"""Prepare the native pipelines, then serve the built frontend and API together."""
from __future__ import annotations

import json
import os
import hashlib
import select
import socket
import socketserver
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from prepds.cli import _load_catalog_trials
from prepds.config import load_settings
from prepds.models import MatchStatus, ReviewStatus
from prepds.review_store import load_manifest

CLASSIFIER_ROOT = Path(os.environ.get("FISHLAB_CLASSIFIER_ROOT", "/classifier"))
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend/dist"
ARTIFACTS = ("manifest.json", "frames.parquet", "segments.csv", "strip.png")
HOST_GATEWAY = "host.docker.internal"


def service(name: str) -> str:
    """Host/port/URL settings come only from .env (passed in by Compose); no built-in defaults."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in .env.")
    return value


def app_port() -> int:
    text = service("FISHLAB_PORT")
    if not (text.isdigit() and 1 <= int(text) <= 65535):
        raise RuntimeError(f"FISHLAB_PORT must be a port number, got {text!r}.")
    return int(text)


@contextmanager
def model_bridge(base_url):
    """Keep DCS's loopback-only policy while reaching a model on the Docker host."""
    parsed = urlparse(base_url)
    if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        yield
        return
    if parsed.hostname == "::1":
        raise RuntimeError("Use 127.0.0.1 rather than ::1 for DCS_CHAT_BASE_URL in Docker.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    class Forward(socketserver.BaseRequestHandler):
        def handle(self):
            with socket.create_connection((HOST_GATEWAY, port), timeout=5) as upstream:
                while True:
                    ready, _, _ = select.select([self.request, upstream], [], [], 1)
                    for source in ready:
                        chunk = source.recv(65536)
                        if not chunk:
                            return
                        (upstream if source is self.request else self.request).sendall(chunk)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with Server((parsed.hostname, port), Forward) as server:
        server.daemon_threads = True
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield
        finally:
            server.shutdown()
            thread.join()


def check_chat(settings):
    from dcs.chat import check_engine
    from dcs.chat_tools import ResearchData

    check_engine(settings.chat)
    ResearchData.from_settings(settings, run=model_run(settings))
    request = Request(settings.chat["base_url"].rstrip("/") + "/models")
    with model_bridge(settings.chat["base_url"]):
        with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
            models = json.load(response)
    if settings.chat["model"] not in [model.get("id") for model in models.get("data", [])]:
        raise RuntimeError("chat.model is not available on DCS_CHAT_BASE_URL; start/pull the configured local model first.")


def command(package: str, *args: str) -> None:
    env = os.environ.copy()
    if package == "dcs":
        env["PYTHONPATH"] = str(CLASSIFIER_ROOT / "src")
    result = subprocess.run([sys.executable, "-u", "-m", package, *args], env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"{package} {' '.join(args)} exited {result.returncode}; see the error above")


def dcs_settings():
    from dcs.config import load_settings as load_dcs

    return load_dcs()


def validate_existing(settings) -> None:
    for path in (settings.paths.output_dir / "processed").glob("*/manifest.json"):
        manifest = load_manifest(path.parent)
        if manifest.review_status not in (ReviewStatus.NOT_PROCESSED, ReviewStatus.REJECTED):
            missing = [name for name in ARTIFACTS if not (path.parent / name).is_file()]
            if missing:
                raise RuntimeError(f"{path.parent.name}: missing {', '.join(missing)}. Restore the artifacts before starting; reviewed work is preserved.")


def validate_outputs(settings) -> None:
    trials = _load_catalog_trials(settings) or []
    matched = [t for t in trials if t.match_status is MatchStatus.MATCHED]
    if not matched:
        raise RuntimeError("No matched videos; check exceptions_report.json and the workbook/video paths.")
    validate_existing(settings)
    for trial in matched:
        directory = settings.paths.output_dir / "processed" / f"{trial.sex}_{trial.subject_id}"
        missing = [name for name in ARTIFACTS if not (directory / name).is_file()]
        if missing:
            raise RuntimeError(f"{directory.name}: preprocessing did not finish ({', '.join(missing)} missing)")


def model_run(settings) -> Path | None:
    explicit = os.environ.get("FISHLAB_MODEL_RUN", "").strip()
    if explicit:
        run = Path(explicit).expanduser()
        if not (run / "model/model_info.json").is_file():
            raise RuntimeError("FISHLAB_MODEL_RUN must point to a trained DCS run containing model/model_info.json.")
        return run
    candidates = sorted((settings.paths.output_dir / "training").glob("*/model/model_info.json"))
    return candidates[-1].parents[1] if candidates else None


def state_path() -> Path:
    return Path(os.environ.get("PDS_OUTPUT_DIR") or "outputs").expanduser() / "docker/startup.json"


def save_state(state: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare() -> int:
    state = {"ok": False, "started_at": datetime.now(timezone.utc).isoformat(), "completed": [], "skipped": [],
             "pending": ["configuration", "prepds check-config", "prepds catalog", "prepds run", "prepds outputs"],
             "model_run": None, "predictions": None, "chat_url": None}
    step = "configuration"

    def run(label, action):
        nonlocal step
        step = label
        print(f"\n[RUN] {label}", flush=True)
        action()
        state["completed"].append(label)
        state["pending"].remove(label)
        save_state(state)
        print(f"[OK] {label}", flush=True)

    try:
        save_state(state)  # invalidate the previous successful preparation before any work
        settings = load_settings()
        if service("FISHLAB_HOST") != "127.0.0.1":
            raise RuntimeError("FISHLAB_HOST must be 127.0.0.1 for Docker (loopback IP; the app has no authentication).")
        port = app_port()
        if not settings.require("video_dir").is_dir():
            raise RuntimeError("PDS_VIDEO_DIR must be a directory.")
        if not settings.require("db_path").is_file():
            raise RuntimeError("PDS_DB_PATH must be a workbook file.")
        mode = os.environ.get("FISHLAB_DCS_ENABLED", "").strip() or "auto"  # blank = auto
        if mode not in ("0", "1", "auto"):
            raise RuntimeError("FISHLAB_DCS_ENABLED must be auto, 1 or 0.")
        source = (CLASSIFIER_ROOT / "src/dcs/__main__.py").is_file()
        enabled = mode == "1" or (mode == "auto" and source)
        if mode == "auto" and not source and any(os.environ.get(k, "").strip() for k in ("DCS_CONFIG", "DCS_TABLE", "DCS_PROCESSED_DIR", "FISHLAB_MODEL_RUN")):
            raise RuntimeError("DCS is configured but its source is missing. Set FISHLAB_DCS_ROOT to its checkout, or FISHLAB_DCS_ENABLED=0 for review only.")
        if enabled and not source:
            raise RuntimeError("DCS source is missing. Set FISHLAB_DCS_ROOT to the checkout containing src/dcs/__main__.py.")
        dcs = None
        if enabled:
            sys.path.append(str(CLASSIFIER_ROOT / "src"))
            dcs = dcs_settings()  # validate YAML before processing; generated inputs are checked below
            model_run(dcs)  # reject an invalid explicitly requested model before processing
            state["pending"] += ["dcs featurize", "dcs check-config", "dcs audit", "dcs model", "dcs predict"]
        else:
            state["skipped"].append("DCS")
            print("[SKIP] DCS: disabled or no checkout configured; review only.", flush=True)
        chat_mode = os.environ.get("FISHLAB_DCS_CHAT", "").strip() or "0"  # blank = off
        if chat_mode not in ("0", "1"):
            raise RuntimeError("FISHLAB_DCS_CHAT must be 1 or 0.")
        chat_url = settings.service.get("FISHLAB_CHAT_URL")
        if chat_mode == "1" or chat_url:
            if dcs is None:
                raise RuntimeError("Research chat requires the DCS checkout; enable DCS first.")
            if not chat_url:
                raise RuntimeError("FISHLAB_DCS_CHAT=1 needs FISHLAB_CHAT_URL in .env.")
            parsed = urlparse(chat_url)
            if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1") or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
                raise RuntimeError("FISHLAB_CHAT_URL must be a plain loopback HTTP URL (http://<loopback host>:<port>).")
            if (parsed.port or 80) == port:
                raise RuntimeError("FISHLAB_CHAT_URL cannot use the same port as FISHLAB_PORT.")
            state["chat_url"] = chat_url.rstrip("/")
            state["pending"].append("dcs chat")
        else:
            state["skipped"].append("DCS chat")
            print("[SKIP] DCS chat: not enabled.", flush=True)
        run("configuration", lambda: validate_existing(settings))
        run("prepds check-config", lambda: command("prepds", "check-config"))
        run("prepds catalog", lambda: command("prepds", "catalog"))
        run("prepds run", lambda: command("prepds", "run"))
        run("prepds outputs", lambda: validate_outputs(settings))
        if dcs is not None:
            run("dcs featurize", lambda: command("dcs", "featurize"))
            run("dcs check-config", lambda: command("dcs", "check-config"))
            run("dcs audit", lambda: command("dcs", "audit"))

            def ensure_model():
                selected = model_run(dcs)
                if selected is None:
                    print("No saved DCS model; running dcs train with your configured training settings.", flush=True)
                    command("dcs", "train")
                    selected = model_run(dcs)
                if selected is None:
                    raise RuntimeError("DCS training did not produce a compound model.")
                state["model_run"] = str(selected.resolve())
                print(f"Using DCS model: {selected}", flush=True)

            run("dcs model", ensure_model)

            def predict():
                target = dcs.paths.output_dir / "compound_predictions.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name("compound_predictions.pending.csv")
                try:
                    command("dcs", "predict", "--model", state["model_run"], "--input", str(dcs.paths.table), "--out", str(temporary))
                    if not temporary.is_file():
                        raise RuntimeError("DCS prediction output was not written.")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                state["predictions"] = str(target.resolve())

            run("dcs predict", predict)
            if state["chat_url"]:
                run("dcs chat", lambda: check_chat(dcs))
        state["ok"] = True
        save_state(state)
        print("\n[OK] All required preparation steps finished. Frontend + backend may now start.", flush=True)
        return 0
    except Exception as error:
        state.update(failed=step, error=str(error))
        try:
            save_state(state)
        except OSError as write_error:
            print(f"Could not save startup status: {write_error}", flush=True)
        print(f"\n[FAILED] {step}: {error}\nCompleted: {', '.join(state['completed']) or 'none'}\nUnfinished: {', '.join(state['pending'])}", flush=True)
        return 1


def review_app():
    from prepds.config import DEFAULT_CONFIG_FILE
    from prepds.webapp.app import create_app

    state = json.loads(state_path().read_text(encoding="utf-8"))
    if not state.get("ok"):
        raise RuntimeError("Pipeline preparation failed; run ./start.sh and fix the named step first.")
    if not (FRONTEND_DIR / "index.html").is_file():
        raise RuntimeError("Built frontend is missing; rebuild the Docker image.")
    settings = load_settings()
    return create_app(settings.paths.output_dir / "processed", settings.paths.accepted_dir,
                      video_dir=settings.paths.video_dir, profile_dir=DEFAULT_CONFIG_FILE.parent / "calibration_profiles",
                      settings=settings, frontend_dir=FRONTEND_DIR,
                      chat_url=state.get("chat_url"),
                      classifier_root=CLASSIFIER_ROOT if state.get("model_run") else None,
                      model_run=Path(state["model_run"]) if state.get("model_run") else None,
                      predictions_path=Path(state["predictions"]) if state.get("predictions") else None)


def serve():
    import uvicorn

    app = review_app()
    state = json.loads(state_path().read_text())
    child = None
    # All container interfaces so Compose can publish it; the host side is FISHLAB_HOST (loopback) from .env.
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=app_port(), log_level="info"))
    failed = []
    sys.path.append(str(CLASSIFIER_ROOT / "src"))
    chat = dcs_settings().chat if state.get("chat_url") else None
    with model_bridge(chat["base_url"]) if chat else nullcontext():
        try:
            if chat:
                child = subprocess.Popen([sys.executable, "-u", "-m", "dcs", "serve-chat", "--port", str(urlparse(state["chat_url"]).port), "--run", state["model_run"]],
                                         env={**os.environ, "PYTHONPATH": str(CLASSIFIER_ROOT / "src")})
                deadline = time.monotonic() + 30
                while True:
                    if child.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("[FAILED] DCS chat startup; check its log above.")
                    try:
                        with build_opener(ProxyHandler({})).open(state["chat_url"] + "/api/health", timeout=1) as response:
                            if json.load(response).get("ok"):
                                break
                    except OSError:
                        time.sleep(0.2)

                def watch_chat():
                    code = child.wait()
                    if not server.should_exit:
                        failed.append(code)
                        print(f"[FAILED] DCS chat stopped (exit {code}); stopping the app.", flush=True)
                        server.should_exit = True

                Thread(target=watch_chat, daemon=True).start()
            server.run()
        finally:
            server.should_exit = True
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    if failed:
        raise RuntimeError("DCS chat exited unexpectedly.")


def health() -> None:
    opener = build_opener(ProxyHandler({}))
    base = f"http://{service('FISHLAB_HOST')}:{app_port()}"
    with opener.open(base + "/api/health", timeout=2) as response:
        if not json.load(response).get("ok"):
            raise RuntimeError("Backend is not healthy.")
    with opener.open(base + "/", timeout=2) as response:
        if b"<html" not in response.read().lower():
            raise RuntimeError("Frontend is not available.")


def backup() -> Path:
    root = Path.cwd().resolve()
    sources = [Path(".env"), Path("config"), Path(os.environ.get("PDS_OUTPUT_DIR") or "outputs"),
               Path(os.environ.get("PDS_ACCEPTED_DIR") or "accepted"),
               Path(os.environ.get("DCS_OUTPUT_DIR") or "outputs/dcs")]
    sources += [Path(os.environ[key]) for key in ("PDS_CONFIG", "DCS_CONFIG") if os.environ.get(key)]
    resolved = []
    for source in sources:
        path = source.expanduser().resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError(f"Backup source is outside the project: {source}")
        if source in (Path(".env"), Path("config")) and not path.exists():
            raise FileNotFoundError(f"Required backup source is missing: {source}")
        if path.exists() and not any(path == prior or path.is_relative_to(prior) for prior in resolved):
            resolved.append(path)

    files = sorted(path for source in resolved
                   for path in (source.rglob("*") if source.is_dir() else [source]) if path.is_file() or path.is_symlink())
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Backup source links outside the project or is a symlink: {path}")

    directory = root / "backups"
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Backup directory must not be a symlink")
    os.chmod(directory, 0o700)
    destination = directory / f"fishlab-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.tar"
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".fishlab-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        os.chmod(temporary, 0o600)
        checksums = {}
        with tarfile.open(temporary, "w") as archive:
            for path in files:
                name = path.relative_to(root).as_posix()
                with path.open("rb") as source:
                    checksums[name] = hashlib.file_digest(source, "sha256").digest()
                archive.add(path, arcname=name, recursive=False)
        with tarfile.open(temporary, "r") as archive:
            members = archive.getmembers()
            if set(member.name for member in members) != set(checksums) or any(not member.isfile() for member in members):
                raise RuntimeError("Backup archive contents did not match the saved files")
            for member in members:
                with archive.extractfile(member) as saved:
                    if hashlib.file_digest(saved, "sha256").digest() != checksums[member.name]:
                        raise RuntimeError(f"Backup verification failed for {member.name}")
        with temporary.open("rb") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, destination)
        print(f"[OK] Verified backup: {destination} ({len(files)} files)", flush=True)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    if mode == "prepare":
        sys.exit(prepare())
    elif mode == "serve":
        serve()
    elif mode == "health":
        health()
    elif mode == "backup":
        backup()
    else:
        sys.exit("Expected prepare, serve, health or backup.")
