"""Tracker backed by the fine-tuned fish box+keypoint model (Phase 15). Needs the `ml` extra and, for speed, CUDA.

Same contract as `tracking.track_video` (path -> one `Track` per frame), so it plugs into `pipeline.process_trial`.
The model runs on every `stride`-th frame and `strided_tracks` fills the rest. It is not picklable across worker
processes: run it in a single process (`run_batch(..., workers=1)`), never one copy per CPU worker.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from prepds.annotation.store import KEYPOINT_NAMES
from prepds.annotation.train import load_model
from prepds.models import Track
from prepds.strided_tracks import Detection, expand_strided_detections
from prepds.video_io import probe

DEFAULT_STRIDE = 5
DEFAULT_MIN_SCORE = 0.3
DEFAULT_MIN_SIZE = 400  # r1 held-out: recall@0.5 0.93 at 400 vs 0.93 at 600, ~1.7x faster


class ModelTracker:
    def __init__(
        self,
        run_dir: Path,
        *,
        device: str = "cuda",
        stride: int = DEFAULT_STRIDE,
        min_score: float = DEFAULT_MIN_SCORE,
        min_size: int = DEFAULT_MIN_SIZE,
        batch_size: int = 16,
    ) -> None:
        if stride < 1 or batch_size < 1:
            raise ValueError("stride and batch_size must be >= 1")
        self.run_dir, self.device, self.stride = Path(run_dir), device, stride
        self.min_score, self.batch_size = min_score, batch_size
        self.last_detections: pd.DataFrame | None = None  # raw top-1 detection per sampled frame of the latest video
        self._raw: list[dict] = []
        self.model = load_model(self.run_dir, device=device)
        self.model.transform.min_size = (min_size,)
        self.model.transform.max_size = round(min_size * 4 / 3)

    @property
    def name(self) -> str:
        return f"model:{self.run_dir.name}@stride{self.stride}"

    def __call__(self, video_path: Path) -> list[Track]:
        asset = probe(Path(video_path))
        self.last_detections, self._raw = None, []
        samples: dict[int, Detection | None] = {}
        pending: list[tuple[int, torch.Tensor]] = []
        capture = cv2.VideoCapture(str(video_path))
        try:
            for index in range(asset.frame_count):
                if index % self.stride:
                    if not capture.grab():
                        break
                    continue
                ok, frame = capture.read()
                if not ok:
                    break
                pending.append((index, torch.from_numpy(np.ascontiguousarray(frame[:, :, ::-1])).permute(2, 0, 1)))
                if len(pending) == self.batch_size:
                    self._flush(pending, samples)
        finally:
            capture.release()
        self._flush(pending, samples)
        self.last_detections = _raw_frame(self._raw)
        return expand_strided_detections(samples, n_frames=asset.frame_count, stride=self.stride, fps=asset.fps)

    @torch.no_grad()
    def _flush(self, pending: list[tuple[int, torch.Tensor]], samples: dict[int, Detection | None]) -> None:
        if not pending:
            return
        images = [t.to(self.device).float() / 255.0 for _, t in pending]
        with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
            outputs = self.model(images)
        for (index, _), out in zip(pending, outputs):
            samples[index] = self._detection(out)
            self._raw.append(_raw_record(index, out))
        pending.clear()

    def _detection(self, out) -> Detection | None:
        if len(out["scores"]) == 0 or float(out["scores"][0]) < self.min_score:
            return None
        x0, y0, x1, y1 = (float(v) for v in out["boxes"][0])
        keypoints = out["keypoints"][0].float().cpu().numpy()  # order: snout, dorsal, ventral, tail_base, tail_tip
        dx, dy = keypoints[3][0] - keypoints[0][0], keypoints[3][1] - keypoints[0][1]
        orientation = math.degrees(math.atan2(dy, dx)) % 180.0 if math.hypot(dx, dy) > 1.0 else None
        return Detection((x0 + x1) / 2, (y0 + y1) / 2, orientation)


_RAW_COLUMNS = ["frame_idx", "score", "x0", "y0", "x1", "y1"] + [
    f"{name}_{axis}" for name in KEYPOINT_NAMES for axis in ("x", "y", "score")
]


def _raw_record(index: int, out) -> dict:
    """The model's best detection for one frame whatever its score (None-like NaNs when it found nothing)."""
    record = {column: float("nan") for column in _RAW_COLUMNS}
    record["frame_idx"] = index
    if len(out["scores"]) == 0:
        return record
    record["score"] = float(out["scores"][0])
    record["x0"], record["y0"], record["x1"], record["y1"] = (float(v) for v in out["boxes"][0])
    keypoints = out["keypoints"][0].float().cpu().numpy()
    scores = out["keypoints_scores"][0].float().cpu().numpy() if "keypoints_scores" in out else [float("nan")] * len(KEYPOINT_NAMES)
    for i, name in enumerate(KEYPOINT_NAMES):
        record[f"{name}_x"], record[f"{name}_y"], record[f"{name}_score"] = float(keypoints[i][0]), float(keypoints[i][1]), float(scores[i])
    return record


def _raw_frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=_RAW_COLUMNS).sort_values("frame_idx", ignore_index=True)
    frame["frame_idx"] = frame["frame_idx"].astype("int32")
    return frame.astype({c: "float32" for c in _RAW_COLUMNS if c != "frame_idx"})
