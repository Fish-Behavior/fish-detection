"""Review state machine, edits, accept/reject and the accepted index (FR-012..FR-017).

A video's state lives in its directory: `manifest.json` (status + provenance), `frames.parquet`,
`segments.csv`, `strip.png`. Every operation validates first and writes only afterwards, so a rejected
operation (invalid transition, Dead-monotonicity violation, blocked Accept) leaves the files untouched.

FR-015 and FR-015a are deliberately asymmetric:
- any `Undetermined` frame  -> `AcceptBlocked`, always; `force` is ignored;
- any `Dead` frame          -> `ConfirmationRequired` unless `force=True` (a real label, a higher-stakes claim).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd

from prepds.models import (
    BehaviorState,
    FrameSource,
    ManifestRecord,
    ReviewStatus,
    StateFrame,
    Trial,
)
from prepds.rendering import DEFAULT_UNDETERMINED_COLOR_HEX, render_strip
from prepds.segments import frames_to_segments

_TRANSITIONS: dict[ReviewStatus, frozenset[ReviewStatus]] = {
    ReviewStatus.NOT_PROCESSED: frozenset({ReviewStatus.PROCESSED_AUTO}),
    ReviewStatus.PROCESSED_AUTO: frozenset({ReviewStatus.EDITED, ReviewStatus.ACCEPTED, ReviewStatus.REJECTED}),
    ReviewStatus.EDITED: frozenset({ReviewStatus.EDITED, ReviewStatus.ACCEPTED, ReviewStatus.REJECTED}),
    ReviewStatus.ACCEPTED: frozenset(),
    ReviewStatus.REJECTED: frozenset({ReviewStatus.PROCESSED_AUTO}),
}
_BOUNDARY_EPS_S = 5e-4  # segments.csv stores float32 times (error up to ~6e-5 s at 20 min); far below half a frame (~17 ms)
_ARTIFACTS = ("frames.parquet", "segments.csv", "strip.png", "manifest.json")
INDEX_FILE = "accepted_index.parquet"
_TRIAL_INDEX_FIELDS = ("strain", "age", "date", "agent_exposure_min")
INDEX_COLUMNS = (
    "video_id", "subject_id", "sex", "compound", "concentration_mM", "strain", "age", "date", "agent_exposure_min",
    "video_path", "video_duration_s", "video_fps", "pipeline_version", "calibration_profile_version", "reviewer",
    "reviewed_at", "provenance", "edit_count", "frames_path", "segments_path", "strip_path", "manifest_path",
)

Edit = tuple[float, float, BehaviorState]


class InvalidTransition(ValueError):
    """The requested review-status change is not allowed from the current status."""


class InvalidEdit(ValueError):
    """An edit request is malformed (empty, non-finite, reversed, outside the video, or covers no frame)."""


class MissingArtifacts(RuntimeError):
    """A video directory lacks files it needs (e.g. after an interrupted reject); regenerate it."""


class AcceptBlocked(RuntimeError):
    """FR-015: the strip still contains Undetermined; Accept is impossible, whatever `force` says."""


class ConfirmationRequired(RuntimeError):
    """FR-015a: the strip contains Dead; Accept needs one explicit confirmation (`force=True`)."""


def transition_allowed(current: ReviewStatus, target: ReviewStatus) -> bool:
    return target in _TRANSITIONS[current]


def should_process(status: ReviewStatus, *, force: bool) -> bool:
    """FR-016: a batch run only (re)generates videos that have nothing to lose, unless forced."""
    return force or status in (ReviewStatus.NOT_PROCESSED, ReviewStatus.REJECTED)


def load_manifest(video_dir: Path) -> ManifestRecord:
    return ManifestRecord.from_dict(json.loads((Path(video_dir) / "manifest.json").read_text(encoding="utf-8")))


def save_edit(video_dir: Path, edits: Sequence[Edit], *, reviewer: str, now: dt.datetime) -> ManifestRecord:
    """Relabel `[start_s, end_s)` ranges (any of the seven labels), frame-accurately, as manual edits.

    Counts as one edit per call; status becomes EDITED. Everything is validated and built in a temporary
    directory first, then moved into place with the manifest last, so a failure leaves the video untouched.
    """
    video_dir = Path(video_dir)
    manifest = _require_transition(video_dir, ReviewStatus.EDITED)
    if not edits:
        raise InvalidEdit("no edits given")
    for start_s, end_s, _ in edits:
        if not (math.isfinite(start_s) and math.isfinite(end_s)):
            raise InvalidEdit(f"edit boundaries must be finite, got [{start_s}, {end_s}]")
        if end_s <= start_s:
            raise InvalidEdit(f"edit start ({start_s}) must be before its end ({end_s})")
        if start_s < 0 or end_s > manifest.video_duration_s + _BOUNDARY_EPS_S:
            raise InvalidEdit(f"edit [{start_s}, {end_s}] is outside the video (0 to {manifest.video_duration_s} s)")

    _require_files(video_dir, "frames.parquet")
    frames = pd.read_parquet(video_dir / "frames.parquet")
    # frame time from the exact frame_idx / fps (float64), not the float32 t_sec stored in the parquet
    frame_time = frames["frame_idx"].to_numpy(dtype="float64") / manifest.video_fps
    states = frames["state"].astype(str).to_numpy(copy=True)
    sources = frames["source"].astype(str).to_numpy(copy=True)
    for start_s, end_s, state in edits:
        mask = (frame_time >= start_s - _BOUNDARY_EPS_S) & (frame_time < end_s - _BOUNDARY_EPS_S)
        if not mask.any():
            raise InvalidEdit(f"edit [{start_s}, {end_s}] covers no frame")
        states[mask] = state.value
        sources[mask] = FrameSource.MANUAL.value

    updated = frames.copy()
    updated["state"] = pd.Categorical(states, categories=list(frames["state"].cat.categories))
    updated["source"] = pd.Categorical(sources, categories=list(frames["source"].cat.categories))
    segments = frames_to_segments(_state_frames(updated))  # raises DeadMonotonicityError before anything is written
    result = replace(
        manifest,
        review_status=ReviewStatus.EDITED,
        edited=True,
        edit_count=manifest.edit_count + 1,
        reviewer=reviewer,
        reviewed_at=now,
    )

    with tempfile.TemporaryDirectory(dir=video_dir, prefix=".edit-") as tmp:
        staging = Path(tmp)
        updated.to_parquet(staging / "frames.parquet", index=False)
        pd.DataFrame([s.to_dict() for s in segments]).to_csv(staging / "segments.csv", index=False)
        render_strip(
            segments,
            duration_s=max(manifest.video_duration_s, segments[-1].end_s),
            subject_id=f"{manifest.sex}_{manifest.subject_id}",
            output_path=staging / "strip.png",
            undetermined_color_hex=DEFAULT_UNDETERMINED_COLOR_HEX,
        )
        _write_manifest(staging, result)
        for name in _ARTIFACTS:  # manifest.json is last in _ARTIFACTS
            os.replace(staging / name, video_dir / name)
    return result


def accept(
    video_dir: Path,
    *,
    accepted_dir: Path,
    reviewer: str,
    now: dt.datetime,
    force: bool = False,
    trial: Trial | None = None,
) -> ManifestRecord:
    """Commit the video to the gold dataset: artifacts + provenance + accepted-index row (FR-014)."""
    video_dir = Path(video_dir)
    if video_dir.name in ("", ".", ".."):
        raise ValueError(f"unusable video directory name {video_dir.name!r}")
    manifest = _require_transition(video_dir, ReviewStatus.ACCEPTED)
    _require_files(video_dir, "frames.parquet", "strip.png")
    frames = pd.read_parquet(video_dir / "frames.parquet")
    present = set(frames["state"].astype(str))
    if BehaviorState.UNDETERMINED.value in present:
        raise AcceptBlocked("the strip still contains Undetermined segments; resolve them before accepting")
    if BehaviorState.DEAD.value in present and not force:
        raise ConfirmationRequired("the strip contains Dead; confirm to accept (Dead is hard to tell from prolonged LORR)")

    accepted = replace(manifest, review_status=ReviewStatus.ACCEPTED, reviewer=reviewer, reviewed_at=now)
    source = FrameSource.MANUAL.value if manifest.edited else FrameSource.AUTO.value
    gold = Path(accepted_dir) / video_dir.name
    gold.mkdir(parents=True, exist_ok=True)
    # Gold copy first (segments re-derived from the frames on disk, never copied from a possibly stale
    # file), the accepted index next, and the source manifest flipped to ACCEPTED last: a failure at any
    # step leaves the source video acceptable, so a retry works.
    for name in ("frames.parquet", "strip.png"):
        shutil.copy2(video_dir / name, gold / name)
    pd.DataFrame([s.to_dict() for s in frames_to_segments(_state_frames(frames))]).to_csv(
        gold / "segments.csv", index=False
    )
    _write_manifest(gold, accepted)
    (gold / "provenance.json").write_text(
        json.dumps(
            {
                "reviewer": reviewer,
                "reviewed_at": now.isoformat(),
                "source": source,
                "calibration_profile_version": manifest.calibration_profile_version,
                "pipeline_version": manifest.pipeline_version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _update_index(Path(accepted_dir), gold, accepted, source, trial)
    _write_manifest(video_dir, accepted)
    return accepted


def reject(video_dir: Path, *, reviewer: str, now: dt.datetime) -> ManifestRecord:
    """Discard the current auto/manual state and re-queue the video for regeneration from scratch."""
    video_dir = Path(video_dir)
    manifest = _require_transition(video_dir, ReviewStatus.REJECTED)
    for name in _ARTIFACTS[:3]:
        (video_dir / name).unlink(missing_ok=True)
    result = replace(
        manifest, review_status=ReviewStatus.REJECTED, edited=False, edit_count=0, reviewer=reviewer, reviewed_at=now
    )
    _write_manifest(video_dir, result)
    return result


def _require_files(video_dir: Path, *names: str) -> None:
    missing = [name for name in names if not (video_dir / name).is_file()]
    if missing:
        raise MissingArtifacts(f"{video_dir.name} is missing {', '.join(missing)}; regenerate the video")


def _require_transition(video_dir: Path, target: ReviewStatus) -> ManifestRecord:
    manifest = load_manifest(video_dir)
    if not transition_allowed(manifest.review_status, target):
        raise InvalidTransition(f"cannot go from {manifest.review_status.value} to {target.value}")
    return manifest


def _write_manifest(directory: Path, manifest: ManifestRecord) -> None:
    """Atomic: readers see the old or the new manifest, never a truncated one."""
    target = directory / "manifest.json"
    temporary = directory / "manifest.json.tmp"
    try:
        temporary.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _state_frames(frames: pd.DataFrame) -> list[StateFrame]:
    return [
        StateFrame(int(r.frame_idx), float(r.t_sec), BehaviorState(str(r.state)), FrameSource(str(r.source)), None)
        for r in frames.itertuples()
    ]


def _index_row(gold: Path, manifest: ManifestRecord, source: str, trial: Trial | None) -> dict:
    return {
        "video_id": gold.name,
        "subject_id": manifest.subject_id,
        "sex": manifest.sex,
        "compound": manifest.compound,
        "concentration_mM": manifest.concentration_mM,
        "strain": trial.strain if trial else None,
        "age": trial.age if trial else None,
        "date": trial.date.isoformat() if trial else None,
        "agent_exposure_min": trial.agent_exposure_min if trial else None,
        "video_path": str(manifest.video_path),
        "video_duration_s": manifest.video_duration_s,
        "video_fps": manifest.video_fps,
        "pipeline_version": manifest.pipeline_version,
        "calibration_profile_version": manifest.calibration_profile_version,
        "reviewer": manifest.reviewer,
        "reviewed_at": manifest.reviewed_at.isoformat() if manifest.reviewed_at else None,
        "provenance": source,
        "edit_count": manifest.edit_count,
        "frames_path": str(gold / "frames.parquet"),
        "segments_path": str(gold / "segments.csv"),
        "strip_path": str(gold / "strip.png"),
        "manifest_path": str(gold / "manifest.json"),
    }


def _write_index(accepted_dir: Path, index: pd.DataFrame) -> None:
    index_path = accepted_dir / INDEX_FILE
    temporary = index_path.with_suffix(".parquet.tmp")
    index.to_parquet(temporary, index=False)
    os.replace(temporary, index_path)  # atomic: a crash never leaves a half-written index


def _update_index(accepted_dir: Path, gold: Path, manifest: ManifestRecord, source: str, trial: Trial | None) -> None:
    new = pd.DataFrame([_index_row(gold, manifest, source, trial)])
    index_path = accepted_dir / INDEX_FILE
    if index_path.is_file():
        existing = pd.read_parquet(index_path)
        new = pd.concat([existing[existing["video_id"] != gold.name], new], ignore_index=True)
    _write_index(accepted_dir, new)


def rebuild_index(accepted_dir: Path, *, trials: Sequence[Trial] = ()) -> int:
    """Regenerate `accepted_index.parquet` from the ACCEPTED manifests on disk; returns the row count.

    The directories are the source of truth, so a lost, stale or hand-edited index is recoverable.
    `trials` (e.g. the catalog) supplies the workbook fields the manifest does not carry. Unreadable or
    non-ACCEPTED directories are skipped, and rows for directories that no longer exist are dropped.
    """
    accepted_dir = Path(accepted_dir)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    by_key = {(t.sex, t.subject_id): t for t in trials}
    previous: dict[str, dict] = {}
    if (accepted_dir / INDEX_FILE).is_file():
        try:
            previous = {r["video_id"]: r for r in pd.read_parquet(accepted_dir / INDEX_FILE).to_dict("records")}
        except (OSError, ValueError, KeyError):
            previous = {}  # an unreadable index is exactly what this function repairs
    rows = []
    for gold in sorted(p for p in accepted_dir.iterdir() if (p / "manifest.json").is_file()):
        try:
            manifest = load_manifest(gold)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if manifest.review_status is not ReviewStatus.ACCEPTED:
            continue
        source = FrameSource.MANUAL.value if manifest.edited else FrameSource.AUTO.value
        try:
            source = json.loads((gold / "provenance.json").read_text(encoding="utf-8"))["source"]
        except (OSError, ValueError, KeyError, TypeError):
            pass  # fall back to what the manifest implies
        row = _index_row(gold, manifest, source, by_key.get((manifest.sex, manifest.subject_id)))
        for field in _TRIAL_INDEX_FIELDS:  # workbook fields no longer available: keep what was indexed before
            if row[field] is None and previous.get(gold.name, {}).get(field) is not None:
                row[field] = previous[gold.name][field]
        rows.append(row)
    _write_index(accepted_dir, pd.DataFrame(rows, columns=INDEX_COLUMNS))
    return len(rows)
