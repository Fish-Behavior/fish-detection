"""Labeling web app backend (Phase 15): a thin HTTP layer over `annotation.store`.

Local single-user tool (loopback only). Frame ids come from the frozen `sample.json`, never from the filesystem,
so a request can only ever name a sampled frame.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from prepds.annotation import store
from prepds.annotation.coco import build_coco
from prepds.annotation.samples import load_samples
from prepds.annotation.split import HELDOUT, TRAIN
from prepds.webapp.security import DEFAULT_ALLOWED_HOSTS, install_local_guards

STATIC_DIR = Path(__file__).parent / "static"


class AnnotationRequest(BaseModel):
    fish_visible: bool
    box: list[float] | None = None
    keypoints: dict[str, list[float]] = {}
    listing: str | None = None
    annotator: str


def create_annotation_app(
    work_dir: Path,
    *,
    clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS,
) -> FastAPI:
    """`work_dir` holds sample.json, frames/, optionally prefill.json; annotations/ is created on first save."""
    work_dir = Path(work_dir)
    annotations_dir = work_dir / "annotations"
    samples = load_samples(work_dir)
    by_id = {s["image"].rsplit(".", 1)[0]: s for s in samples}
    prefill_path = work_dir / "prefill.json"
    prefill: dict[str, Any] = json.loads(prefill_path.read_text(encoding="utf-8")) if prefill_path.is_file() else {}

    app = FastAPI(title="prepds annotation")
    install_local_guards(app, allowed_hosts)

    def sample_of(frame_id: str) -> dict[str, Any]:
        sample = by_id.get(frame_id)
        if sample is None:
            raise HTTPException(404, "unknown frame")
        return sample

    def annotation_of(frame_id: str) -> store.Annotation | None:
        try:
            return store.load_annotation(annotations_dir, frame_id)
        except ValueError as error:
            raise HTTPException(500, {"code": "corrupt_annotation", "message": f"the annotation of {frame_id} is unreadable"}) from error

    def prefill_of(sample: dict[str, Any]) -> Any:
        return sample["prefill"] if "prefill" in sample else prefill.get(sample["image"])

    def status_of(frame_id: str) -> str:
        try:
            annotation = store.load_annotation(annotations_dir, frame_id)
        except ValueError:
            return "corrupt"
        if annotation is None:
            return "todo"
        return "done" if annotation.fish_visible else "no_fish"

    @app.get("/api/frames")
    def list_frames(status: str | None = None, split: str | None = None, batch: str | None = None,
                    reason: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for frame_id, sample in by_id.items():
            row_status = status_of(frame_id)
            if (split and sample["split"] != split) or (batch and sample["batch"] != batch) or (reason and sample["reason"] != reason):
                continue
            if status and not (row_status == status or (status == "done" and row_status == "no_fish")):
                continue
            rows.append({"id": frame_id, "video_id": sample["video_id"], "frame_idx": sample["frame_idx"],
                         "reason": sample["reason"], "split": sample["split"], "compound": sample["compound"],
                         "batch": sample["batch"], "status": row_status, "has_prefill": prefill_of(sample) is not None})
        return rows

    @app.get("/api/frames/{frame_id}")
    def get_frame(frame_id: str) -> dict[str, Any]:
        sample = sample_of(frame_id)
        width, height = _image_size(work_dir / "frames" / sample["image"])
        annotation = annotation_of(frame_id)
        return {
            "id": frame_id, "video_id": sample["video_id"], "frame_idx": sample["frame_idx"], "t_sec": sample["t_sec"],
            "reason": sample["reason"], "split": sample["split"], "compound": sample["compound"],
            "width": width, "height": height, "image_url": f"/api/frames/{frame_id}/image",
            "batch": sample["batch"], "prefill": prefill_of(sample), "keypoint_names": list(store.KEYPOINT_NAMES),
            "annotation": annotation.to_dict() if annotation else None,
        }

    @app.get("/api/frames/{frame_id}/image")
    def get_image(frame_id: str) -> FileResponse:
        path = work_dir / "frames" / sample_of(frame_id)["image"]
        if not path.is_file():
            raise HTTPException(404, "image missing")
        return FileResponse(path, media_type="image/png")

    @app.put("/api/frames/{frame_id}")
    def put_annotation(frame_id: str, body: AnnotationRequest) -> dict[str, Any]:
        sample = sample_of(frame_id)
        width, height = _image_size(work_dir / "frames" / sample["image"])
        try:
            annotation = store.parse_annotation(body.model_dump(), width=width, height=height, now=clock())
        except store.InvalidAnnotation as error:
            raise HTTPException(422, {"code": "invalid_annotation", "message": str(error)}) from error
        store.save_annotation(annotations_dir, frame_id, annotation)
        return get_frame(frame_id)

    @app.get("/api/progress")
    def progress() -> dict[str, Any]:
        by_split: dict[str, dict[str, int]] = {}
        by_batch: dict[str, dict[str, int]] = {}
        listing: Counter[str] = Counter()
        done = 0
        for frame_id, sample in by_id.items():
            entry = by_split.setdefault(sample["split"], {"total": 0, "done": 0})
            batch_entry = by_batch.setdefault(sample["batch"], {"total": 0, "done": 0})
            entry["total"] += 1
            batch_entry["total"] += 1
            if status_of(frame_id) in ("done", "no_fish"):
                entry["done"] += 1
                batch_entry["done"] += 1
                done += 1
                annotation = store.load_annotation(annotations_dir, frame_id)
                if annotation and annotation.listing:
                    listing[annotation.listing] += 1
        return {"total": len(by_id), "done": done, "by_split": by_split, "by_batch": by_batch,
                "listing": {tag: listing[tag] for tag in store.LISTING_TAGS}}

    @app.get("/api/export/coco/{split_name}")
    def export_coco(split_name: str) -> dict[str, Any]:
        if split_name not in (TRAIN, HELDOUT):
            raise HTTPException(404, "unknown split")
        annotations = {}
        corrupt = []
        for frame_id in by_id:
            try:
                annotation = store.load_annotation(annotations_dir, frame_id)
            except ValueError:
                corrupt.append(frame_id)
                continue
            if annotation is not None:
                annotations[frame_id] = annotation
        if corrupt:  # never export a ground-truth set with silently missing frames
            raise HTTPException(409, {"code": "corrupt_annotations", "ids": corrupt,
                                      "message": f"{len(corrupt)} unreadable annotation(s): relabel them (status=corrupt) first"})
        return build_coco(samples, annotations, split_name=split_name,
                          size_of=lambda image: _image_size(work_dir / "frames" / image))

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    return app


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
