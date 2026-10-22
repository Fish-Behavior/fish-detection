"""Inputs for the annotation workflow: per-video summaries from `processed/` and single-frame extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from prepds.annotation.sampler import VideoInfo
from prepds.review_store import load_manifest

TERMINAL_NO_ACTION = "terminal_no_action"


def load_video_infos(processed_dir: Path) -> list[VideoInfo]:
    """One `VideoInfo` per readable processed video directory; unreadable ones are skipped."""
    infos = []
    for directory in sorted(p for p in Path(processed_dir).iterdir() if (p / "manifest.json").is_file()):
        try:
            manifest = load_manifest(directory)
            frames = pd.read_parquet(directory / "frames.parquet", columns=["frame_idx", "detected", "state"])
            if not np.array_equal(frames["frame_idx"].to_numpy(), np.arange(len(frames))):
                raise ValueError("frame_idx is not 0..n-1: row position would not match the video frame")
        except (OSError, ValueError, KeyError, TypeError):
            continue
        flag = next((f for f in manifest.review_flags if f.kind == TERMINAL_NO_ACTION), None)
        infos.append(
            VideoInfo(
                video_id=directory.name,
                compound=manifest.compound,
                fps=manifest.video_fps,
                detected=frames["detected"].to_numpy(dtype=bool),
                states=frames["state"].astype(str).to_numpy(dtype=object),
                flagged_from_s=flag.start_s if flag else None,
            )
        )
    return infos


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """BGR frame `frame_idx` of the video; ValueError if it cannot be read."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"cannot read frame {frame_idx} of {video_path}")
    return frame
