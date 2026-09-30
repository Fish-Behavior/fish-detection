"""FastAPI review backend (T066-T071) - a thin HTTP layer over `review_store`.

It maps the store's rules to HTTP and adds none of its own:
- `POST .../accept`: Undetermined present -> 409 `undetermined_present` always (`force` is ignored,
  FR-015); Dead present without `force` -> 409 `dead_confirmation_required` (FR-015a).
- invalid status transitions -> 409; invalid edits -> 422.
Video ids are directory names under `processed_dir` and are validated before touching the filesystem.
Local single-user tool: bind it to 127.0.0.1.
"""

from __future__ import annotations

import datetime as dt
import json
import mimetypes
import re
import threading
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import cv2
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from prepds import review_store
from prepds.config import ConfigError
from prepds.explain import VideoExplainer
from prepds.calibration.profile import labeling_thresholds
from prepds.listing_flags import read_listing_flags
from prepds.models import BehaviorState, ManifestRecord, ReviewStatus
from prepds.segments import DeadMonotonicityError
from prepds.webapp.security import DEFAULT_ALLOWED_HOSTS, install_local_guards
from prepds.webapp.schemas import (
    AcceptRequest,
    EditsRequest,
    FrameSummary,
    DetectionOut,
    WaterlineIn,
    WaterlineOut,
    ListingFlagOut,
    ManifestOut,
    OverlayWindow,
    RejectRequest,
    SecondExplanation,
    SegmentOut,
    VideoDetail,
    VideoSummary,
)

STATIC_DIR = Path(__file__).parent / "static"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MAX_OVERLAY_WINDOW_S = 30.0
MIN_DETECTION_SCORE = 0.3  # the model tracker's own score floor: weaker top-1 detections are not treated as the fish
_KEYPOINTS = ("snout", "dorsal_fin_base", "ventral", "tail_base", "tail_tip")
_VIDEO_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".avi", ".mkv"})


def create_app(
    processed_dir: Path,
    accepted_dir: Path,
    *,
    video_dir: Path | None = None,
    profile_dir: Path | None = None,
    clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS,
) -> FastAPI:
    """`processed_dir`: one subdirectory per video (export output).

    `video_dir`: when given, source videos must resolve to a file under it (and relative manifest paths are
    resolved against it). `allowed_hosts`: Host headers accepted (DNS-rebinding defence for a localhost
    tool); state-changing requests carrying a foreign `Origin` are refused too.
    """
    processed_dir, accepted_dir = Path(processed_dir), Path(accepted_dir)
    profile_dir = None if profile_dir is None else Path(profile_dir)
    sizes: dict[str, tuple[int, int] | None] = {}  # coded frame size per video: some files display with a non-square pixel aspect
    explainers: dict[tuple[str, int, int, str], tuple[VideoExplainer, str | None]] = {}  # small cache: replaying a track takes a moment
    app = FastAPI(title="prepds review")
    install_local_guards(app, allowed_hosts)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, error: RequestValidationError) -> JSONResponse:
        # the default body echoes the rejected input, which cannot be serialized when it is NaN
        detail = [{"loc": list(e.get("loc", ())), "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))} for e in error.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    write_lock = threading.Lock()  # one reviewer, but two browser tabs must not interleave read-modify-write

    def video_path_of(video_id: str) -> Path:
        if not _ID.fullmatch(video_id):
            raise HTTPException(404, "unknown video")
        path = processed_dir / video_id
        if not (path / "manifest.json").is_file():
            raise HTTPException(404, "unknown video")
        return path

    def manifest_of(directory: Path) -> ManifestRecord:
        try:
            return review_store.load_manifest(directory)
        except (OSError, ValueError, KeyError, TypeError) as error:  # bad JSON, missing keys, bad enum/date values
            raise HTTPException(500, {"code": "corrupt_manifest", "message": f"the manifest of {directory.name} is unreadable"}) from error

    def detail(video_id: str) -> VideoDetail:
        directory = video_path_of(video_id)
        manifest = manifest_of(directory)
        segments = _read_segments(directory)
        seconds: dict[str, float] = defaultdict(float)
        bouts: dict[str, int] = defaultdict(int)
        for segment in segments:
            seconds[segment.state.value] += segment.duration_s
            bouts[segment.state.value] += 1
        try:
            listing_flags, listing_error = [ListingFlagOut(**f.to_dict()) for f in read_listing_flags(directory)], None
        except ValueError as error:  # advisory data must never make the video unreviewable
            listing_flags, listing_error = [], str(error)
        return VideoDetail(
            video_id=video_id,
            manifest=_manifest_out(manifest),
            segments=segments,
            frame_summary=FrameSummary(
                seconds_by_state=dict(seconds),
                bouts_by_state=dict(bouts),
                total_s=float(sum(seconds.values())),
                has_undetermined=BehaviorState.UNDETERMINED.value in seconds,
                has_dead=BehaviorState.DEAD.value in seconds,
            ),
            video_url=f"/videos/{video_id}/video",
            strip_url=f"/videos/{video_id}/strip" if (directory / "strip.png").is_file() else None,
            listing_flags=listing_flags,
            listing_flags_error=listing_error,
        )

    @app.get("/videos", response_model=list[VideoSummary])
    def list_videos(status: ReviewStatus | None = None) -> list[VideoSummary]:
        rows = []
        if not processed_dir.is_dir():
            return rows
        for directory in sorted(p for p in processed_dir.iterdir() if _ID.fullmatch(p.name) and (p / "manifest.json").is_file()):
            try:
                manifest = review_store.load_manifest(directory)
            except (OSError, ValueError, KeyError, TypeError):
                continue  # one unreadable manifest must not hide every other video
            if status is not None and manifest.review_status != status:
                continue
            rows.append(
                VideoSummary(
                    video_id=directory.name,
                    subject_id=manifest.subject_id,
                    sex=manifest.sex,
                    compound=manifest.compound,
                    concentration_mM=manifest.concentration_mM,
                    review_status=manifest.review_status,
                    edited=manifest.edited,
                    edit_count=manifest.edit_count,
                    flag_count=len(manifest.review_flags),
                )
            )
        return rows

    def thresholds_for(version: str) -> tuple[dict | None, str | None]:
        if profile_dir is None:
            return None, "the review app was started without a calibration profile folder"
        if not _ID.fullmatch(version):
            return None, f"unusable calibration profile name {version!r}"
        path = profile_dir / f"{version}.yaml"
        if not path.is_file():
            return None, f"calibration profile {version} was not found in {profile_dir.name}/"
        try:
            return labeling_thresholds(yaml.safe_load(path.read_text(encoding="utf-8")) or {}), None
        except (OSError, yaml.YAMLError, ConfigError, ValueError, TypeError) as error:
            return None, f"calibration profile {version} could not be read: {error}"

    @app.get("/videos/{video_id}/overlay", response_model=OverlayWindow)
    def overlay_window(video_id: str, start_s: float = Query(ge=0), end_s: float = Query(gt=0)) -> OverlayWindow:
        if not 0 < end_s - start_s <= MAX_OVERLAY_WINDOW_S:
            raise HTTPException(422, f"the window must be longer than 0 s and at most {MAX_OVERLAY_WINDOW_S:g} s")
        directory = video_path_of(video_id)
        manifest = manifest_of(directory)
        frames_path = directory / "frames.parquet"
        if not frames_path.is_file():
            raise HTTPException(409, {"code": "missing_artifacts", "message": f"{video_id} has no frames.parquet"})
        frames = pd.read_parquet(frames_path, columns=["frame_idx", "t_sec", "x", "y", "detected"],
                                 filters=[("t_sec", ">=", start_s), ("t_sec", "<", end_s)])
        detected = frames["detected"].to_numpy(bool)
        # undetected frames store x = y = 0 as a sentinel: send null so the page can never draw a fish at the corner
        xs = [round(float(v), 1) if d else None for v, d in zip(frames["x"], detected)]
        ys = [round(float(v), 1) if d else None for v, d in zip(frames["y"], detected)]
        detections, detections_error = [], None
        det_path = directory / "detections.parquet"
        if det_path.is_file() and len(frames):
            try:
                detections = _detections_in(det_path, frames)
            except (OSError, ValueError, KeyError) as error:  # an unreadable sidecar must not hide the track
                detections_error = f"detections.parquet could not be read: {error}"
        if video_id not in sizes:
            source = _resolve_source(manifest.video_path, video_dir)
            sizes[video_id] = _coded_size(source) if source is not None else None
        width, height = sizes[video_id] or (None, None)
        return OverlayWindow(fps=manifest.video_fps, width=width, height=height, t=[round(float(v), 4) for v in frames["t_sec"]], x=xs, y=ys,
                             detected=[bool(d) for d in detected], detections=detections, detections_error=detections_error,
                             has_detector=det_path.is_file())

    def waterline_of(directory: Path) -> WaterlineOut:
        try:
            y = float(json.loads((directory / "waterline.json").read_text(encoding="utf-8"))["y_px"])
        except (OSError, ValueError, KeyError, TypeError):  # absent or unreadable: no waterline, never a made-up one
            return WaterlineOut()
        return WaterlineOut(y_px=y, source="manual") if y == y and y >= 0 else WaterlineOut()

    @app.get("/videos/{video_id}/waterline", response_model=WaterlineOut)
    def get_waterline(video_id: str) -> WaterlineOut:
        return waterline_of(video_path_of(video_id))

    @app.put("/videos/{video_id}/waterline", response_model=WaterlineOut)
    def put_waterline(video_id: str, body: WaterlineIn) -> WaterlineOut:
        directory = video_path_of(video_id)
        size = sizes.get(video_id)
        if size is not None and body.y_px > size[1]:
            raise HTTPException(422, f"the waterline must lie inside the frame (height {size[1]} px)")
        with write_lock:
            temp = directory / "waterline.json.tmp"
            temp.write_text(json.dumps({"y_px": body.y_px}), encoding="utf-8")
            temp.replace(directory / "waterline.json")
        return WaterlineOut(y_px=body.y_px, source="manual")

    @app.delete("/videos/{video_id}/waterline", response_model=WaterlineOut)
    def delete_waterline(video_id: str) -> WaterlineOut:
        directory = video_path_of(video_id)
        with write_lock:
            (directory / "waterline.json").unlink(missing_ok=True)
        return WaterlineOut()

    @app.get("/videos/{video_id}/explain", response_model=SecondExplanation)
    def explain_second(video_id: str, second: int = Query(ge=0)) -> SecondExplanation:
        directory = video_path_of(video_id)
        manifest = manifest_of(directory)
        frames_path = directory / "frames.parquet"
        if not frames_path.is_file():
            raise HTTPException(409, {"code": "missing_artifacts", "message": f"{video_id} has no frames.parquet"})
        key = (video_id, frames_path.stat().st_mtime_ns, frames_path.stat().st_size, manifest.calibration_profile_version)
        if key not in explainers:
            thresholds, thresholds_error = thresholds_for(manifest.calibration_profile_version)
            if len(explainers) >= 4:
                explainers.pop(next(iter(explainers)))
            explainers[key] = (VideoExplainer(pd.read_parquet(frames_path), thresholds), thresholds_error)
        explainer, thresholds_error = explainers[key]
        try:
            payload = explainer.explain(second)
        except KeyError:
            raise HTTPException(404, "no such second in this video") from None
        return SecondExplanation(**payload, thresholds_error=thresholds_error)

    @app.get("/videos/{video_id}", response_model=VideoDetail)
    def get_video(video_id: str) -> VideoDetail:
        return detail(video_id)

    @app.get("/videos/{video_id}/video")
    def stream_video(video_id: str) -> FileResponse:
        source = _resolve_source(manifest_of(video_path_of(video_id)).video_path, video_dir)
        if source is None:
            raise HTTPException(404, "source video not found")
        return FileResponse(source, media_type=mimetypes.guess_type(source.name)[0] or "video/mp4")  # serves Range requests

    @app.get("/videos/{video_id}/strip")
    def get_strip(video_id: str) -> FileResponse:
        strip = video_path_of(video_id) / "strip.png"
        if not strip.is_file():
            raise HTTPException(404, "no strip (video was rejected)")
        return FileResponse(strip, media_type="image/png")

    @app.post("/videos/{video_id}/edits", response_model=VideoDetail)
    def post_edits(video_id: str, body: EditsRequest) -> VideoDetail:
        directory = video_path_of(video_id)
        try:
            with write_lock:
                review_store.save_edit(
                    directory,
                    [(e.start_s, e.end_s, e.new_state) for e in body.edits],
                    reviewer=body.reviewer,
                    now=clock(),
                )
        except review_store.InvalidTransition as error:
            raise _conflict("invalid_transition", error) from error
        except (review_store.InvalidEdit, DeadMonotonicityError) as error:
            raise HTTPException(422, {"code": "invalid_edit", "message": str(error)}) from error
        except review_store.MissingArtifacts as error:
            raise _conflict("missing_artifacts", error) from error
        return detail(video_id)

    @app.post("/videos/{video_id}/accept", response_model=VideoDetail)
    def post_accept(video_id: str, body: AcceptRequest) -> VideoDetail:
        directory = video_path_of(video_id)
        try:
            with write_lock:
                review_store.accept(
                    directory, accepted_dir=accepted_dir, reviewer=body.reviewer, now=clock(), force=body.force
                )
        except review_store.AcceptBlocked as error:
            raise _conflict("undetermined_present", error) from error
        except review_store.ConfirmationRequired as error:
            raise _conflict("dead_confirmation_required", error) from error
        except review_store.MissingArtifacts as error:
            raise _conflict("missing_artifacts", error) from error
        except review_store.InvalidTransition as error:
            raise _conflict("invalid_transition", error) from error
        return detail(video_id)

    @app.post("/videos/{video_id}/reject", response_model=VideoDetail)
    def post_reject(video_id: str, body: RejectRequest) -> VideoDetail:
        directory = video_path_of(video_id)
        try:
            with write_lock:
                review_store.reject(directory, reviewer=body.reviewer, now=clock())
        except review_store.InvalidTransition as error:
            raise _conflict("invalid_transition", error) from error
        return detail(video_id)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    return app


def _conflict(code: str, error: Exception) -> HTTPException:
    return HTTPException(409, {"code": code, "message": str(error)})


def _resolve_source(path: Path, video_dir: Path | None) -> Path | None:
    """The source video to serve, or None: it must be a video file, and under `video_dir` when one is set."""
    if not path.is_absolute():
        if video_dir is None:
            return None
        path = video_dir / path
    resolved = path.resolve()
    if video_dir is not None and not resolved.is_relative_to(video_dir.resolve()):
        return None
    if resolved.suffix.lower() not in _VIDEO_SUFFIXES or not resolved.is_file():
        return None
    return resolved


def _coded_size(source: Path) -> tuple[int, int] | None:
    capture = cv2.VideoCapture(str(source))
    try:
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return (width, height) if width > 0 and height > 0 else None


def _detections_in(path: Path, frames: pd.DataFrame) -> list[DetectionOut]:
    """Detector samples (box + keypoints) whose frame lies in the window, above the tracker's score floor."""
    low, high = int(frames["frame_idx"].min()), int(frames["frame_idx"].max())
    time_of = dict(zip(frames["frame_idx"].astype(int), frames["t_sec"].astype(float)))
    raw = pd.read_parquet(path)
    raw = raw[raw["frame_idx"].between(low, high) & (raw["score"] >= MIN_DETECTION_SCORE)]
    out = []
    for row in raw.itertuples(index=False):
        d = row._asdict()
        out.append(DetectionOut(
            frame_idx=int(d["frame_idx"]), t=round(time_of.get(int(d["frame_idx"]), 0.0), 4), score=round(float(d["score"]), 3),
            box=[round(float(d[k]), 1) for k in ("x0", "y0", "x1", "y1")],
            keypoints={name: [round(float(d[f"{name}_x"]), 1), round(float(d[f"{name}_y"]), 1), round(float(d[f"{name}_score"]), 2)]
                       for name in _KEYPOINTS}))
    return out


def _read_segments(directory: Path) -> list[SegmentOut]:
    path = directory / "segments.csv"
    if not path.is_file():
        return []
    return [SegmentOut(**row) for row in pd.read_csv(path).to_dict("records")]


def _manifest_out(manifest: ManifestRecord) -> ManifestOut:
    return ManifestOut(**manifest.to_dict())
