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

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from prepds import review_store
from prepds.listing_flags import read_listing_flags
from prepds.models import BehaviorState, ManifestRecord, ReviewStatus
from prepds.segments import DeadMonotonicityError
from prepds.webapp.security import DEFAULT_ALLOWED_HOSTS, install_local_guards
from prepds.webapp.schemas import (
    AcceptRequest,
    EditsRequest,
    FrameSummary,
    ListingFlagOut,
    ManifestOut,
    RejectRequest,
    SegmentOut,
    VideoDetail,
    VideoSummary,
)

STATIC_DIR = Path(__file__).parent / "static"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_VIDEO_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".avi", ".mkv"})


def create_app(
    processed_dir: Path,
    accepted_dir: Path,
    *,
    video_dir: Path | None = None,
    clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS,
) -> FastAPI:
    """`processed_dir`: one subdirectory per video (export output).

    `video_dir`: when given, source videos must resolve to a file under it (and relative manifest paths are
    resolved against it). `allowed_hosts`: Host headers accepted (DNS-rebinding defence for a localhost
    tool); state-changing requests carrying a foreign `Origin` are refused too.
    """
    processed_dir, accepted_dir = Path(processed_dir), Path(accepted_dir)
    app = FastAPI(title="prepds review")
    install_local_guards(app, allowed_hosts)

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
        for segment in segments:
            seconds[segment.state.value] += segment.duration_s
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


def _read_segments(directory: Path) -> list[SegmentOut]:
    path = directory / "segments.csv"
    if not path.is_file():
        return []
    return [SegmentOut(**row) for row in pd.read_csv(path).to_dict("records")]


def _manifest_out(manifest: ManifestRecord) -> ManifestOut:
    return ManifestOut(**manifest.to_dict())
