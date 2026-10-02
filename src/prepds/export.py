"""Per-video export (PRD §5.4, FR-010, FR-017): frames.parquet, segments.csv, strip.png, manifest.json.

The three artifacts are mutually consistent by construction: `segments.csv` is `frames_to_segments` of exactly
the states written to `frames.parquet` (which also re-validates Dead monotonicity), and the strip is drawn
from those segments over the video's real duration. The manifest records provenance and reviewer flags.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from prepds.models import (
    BehaviorState,
    FeatureFrame,
    FrameRow,
    FrameSource,
    ManifestRecord,
    ReviewFlag,
    ReviewStatus,
    StateFrame,
    Track,
    Trial,
    VideoAsset,
)
from prepds.rendering import DEFAULT_UNDETERMINED_COLOR_HEX, render_strip
from prepds.segments import frames_to_segments

_STATE_CATEGORIES = [state.value for state in BehaviorState]
_SOURCE_CATEGORIES = [source.value for source in FrameSource]
_FLOAT32_COLUMNS = ("t_sec", "x", "y", "orientation_deg", "depth_from_surface", "velocity", "acceleration",
                    "angular_velocity", "meander", "confidence")


class ReviewedWorkExists(RuntimeError):
    """The target directory holds an EDITED or ACCEPTED video; re-exporting would discard reviewer work."""


@dataclass(frozen=True)
class ExportPaths:
    frames: Path
    segments: Path
    strip: Path
    manifest: Path


def export_video(
    out_dir: Path,
    *,
    trial: Trial,
    asset: VideoAsset,
    tracks: Sequence[Track],
    features: Sequence[FeatureFrame],
    states: Sequence[StateFrame],
    pipeline_version: str,
    calibration_profile_version: str,
    processed_at: dt.datetime,
    review_flags: Sequence[ReviewFlag] = (),
    undetermined_color_hex: str = DEFAULT_UNDETERMINED_COLOR_HEX,
    overwrite: bool = False,
) -> ExportPaths:
    """Write the video's three artifacts and manifest into `out_dir` (created if needed).

    Existing PROCESSED_AUTO/REJECTED output is regenerated; existing EDITED/ACCEPTED output is never
    overwritten (`ReviewedWorkExists`) unless `overwrite=True` (FR-016).
    """
    if not tracks:
        raise ValueError("cannot export a video with no frames")
    if not (len(tracks) == len(features) == len(states)):
        raise ValueError(
            f"tracks, features and states must have the same length, got {len(tracks)}, {len(features)}, {len(states)}"
        )
    if any(t.frame_idx != s.frame_idx or t.frame_idx != f.frame_idx for t, f, s in zip(tracks, features, states)):
        raise ValueError("tracks, features and states are not aligned frame by frame")

    out_dir = Path(out_dir)
    existing = out_dir / "manifest.json"
    if existing.is_file() and not overwrite:
        status = ManifestRecord.from_dict(json.loads(existing.read_text(encoding="utf-8"))).review_status
        if status in (ReviewStatus.EDITED, ReviewStatus.ACCEPTED):
            raise ReviewedWorkExists(f"{out_dir} holds {status.value} work; pass overwrite=True to discard it")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Artifacts are rewritten in place and the manifest is written last, so drop the old manifest first: a
    # failure midway must leave "no manifest" (regenerated next run), never new frames under an old status.
    existing.unlink(missing_ok=True)
    paths = ExportPaths(
        frames=out_dir / "frames.parquet",
        segments=out_dir / "segments.csv",
        strip=out_dir / "strip.png",
        manifest=out_dir / "manifest.json",
    )

    rows = [_frame_row(t, f, s) for t, f, s in zip(tracks, features, states)]
    _frames_dataframe(rows).to_parquet(paths.frames, index=False)

    # parquet stores float32 t_sec: derive the segments from the same rounded values so segments.csv is the
    # exact run-length encoding of the frames on disk (and of any later edit re-derived from them)
    segments = frames_to_segments([replace(s, t_sec=float(np.float32(s.t_sec))) for s in states])
    pd.DataFrame([s.to_dict() for s in segments]).to_csv(paths.segments, index=False)

    render_strip(
        segments,
        duration_s=max(asset.duration_s, segments[-1].end_s),
        subject_id=f"{trial.sex}_{trial.subject_id}",
        output_path=paths.strip,
        undetermined_color_hex=undetermined_color_hex,
    )

    manifest = ManifestRecord(
        subject_id=trial.subject_id,
        sex=trial.sex,
        compound=trial.compound,
        concentration_mM=trial.concentration_mM,
        video_path=asset.path,
        video_duration_s=asset.duration_s,
        video_fps=asset.fps,
        pipeline_version=pipeline_version,
        calibration_profile_version=calibration_profile_version,
        processed_at=processed_at,
        review_status=ReviewStatus.PROCESSED_AUTO,
        reviewer=None,
        reviewed_at=None,
        edited=False,
        edit_count=0,
        review_flags=tuple(review_flags),
    )
    temporary = paths.manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    os.replace(temporary, paths.manifest)
    return paths


def _frame_row(track: Track, feature: FeatureFrame, state: StateFrame) -> FrameRow:
    return FrameRow(
        frame_idx=track.frame_idx,
        t_sec=track.t_sec,
        x=track.x,
        y=track.y,
        orientation_deg=track.orientation_deg,
        depth_from_surface=track.y_from_frame_top,
        detected=track.detected,
        velocity=feature.velocity,
        acceleration=feature.acceleration,
        angular_velocity=feature.angular_velocity,
        meander=feature.meander,
        is_immobile=feature.is_immobile,
        state=state.state,
        source=state.source,
        confidence=state.confidence,
    )


def _frames_dataframe(rows: list[FrameRow]) -> pd.DataFrame:
    frame = pd.DataFrame([row.to_dict() for row in rows])
    frame["frame_idx"] = frame["frame_idx"].astype("int32")
    for column in _FLOAT32_COLUMNS:
        frame[column] = frame[column].astype("float32")
    frame["state"] = pd.Categorical(frame["state"], categories=_STATE_CATEGORIES)
    frame["source"] = pd.Categorical(frame["source"], categories=_SOURCE_CATEGORIES)
    return frame
