"""Batch orchestration (T080): track -> features -> label -> consolidate -> flag -> render -> export, per video.

One video per worker process. A video that fails is reported and skipped (FR-004: one bad file never halts
a batch); reviewed work is never regenerated unless `force` (FR-016).
"""

from __future__ import annotations

import datetime as dt
import logging
import multiprocessing
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from prepds.consolidate import consolidate_states
from prepds.export import export_video
from prepds.features import derive_features
from prepds.labeling import classify_video
from prepds.models import ManifestRecord, Track, Trial
from prepds.review_flags import terminal_no_action_flag
from prepds.review_store import load_manifest, should_process
from prepds.tracking import track_video
from prepds.video_io import probe

LOGGER = logging.getLogger(__name__)
Tracker = Callable[[Path], list[Track]]


class Outcome(str, Enum):
    PROCESSED = "processed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class RunConfig:
    out_dir: Path
    thresholds: Mapping[str, Any]  # `calibration.profile.labeling_thresholds(...)` keyword arguments
    pipeline_version: str
    calibration_profile_version: str
    processed_at: dt.datetime
    force: bool = False
    undetermined_color_hex: str = "#FF00FF"


@dataclass(frozen=True)
class VideoResult:
    video_id: str
    outcome: Outcome
    message: str = ""
    seconds: float = 0.0


def video_id(trial: Trial) -> str:
    return f"{trial.sex}_{trial.subject_id}"


def existing_manifest(out_dir: Path, trial: Trial) -> ManifestRecord | None:
    try:
        return load_manifest(out_dir / video_id(trial))
    except (OSError, ValueError, KeyError, TypeError):
        return None  # absent or unreadable: treat as never processed


def is_pending(trial: Trial, config: RunConfig) -> bool:
    manifest = existing_manifest(config.out_dir, trial)
    return manifest is None or should_process(manifest.review_status, force=config.force)


def process_trial(trial: Trial, config: RunConfig, *, tracker: Tracker = track_video) -> VideoResult:
    """Run the whole per-video pipeline; never raises for a bad video."""
    vid = video_id(trial)
    started = time.perf_counter()
    try:
        if trial.video_path is None:
            raise ValueError("trial has no matched video")
        if not is_pending(trial, config):
            return VideoResult(vid, Outcome.SKIPPED, "already processed or reviewed (use --force to regenerate)")
        asset = probe(trial.video_path)
        tracks = tracker(trial.video_path)
        states = consolidate_states(classify_video(tracks, **config.thresholds))
        flag = terminal_no_action_flag(
            tracks,
            freeze_speed_floor_px_per_s=config.thresholds["freeze_speed_floor_px_per_s"],
            speed_lag_s=config.thresholds.get("speed_lag_s", 1.0),
        )
        (config.out_dir / vid / "detections.parquet").unlink(missing_ok=True)  # never beside artifacts of another run
        export_video(
            config.out_dir / vid,
            trial=trial,
            asset=asset,
            tracks=tracks,
            features=derive_features(tracks),
            states=states,
            pipeline_version=config.pipeline_version,
            calibration_profile_version=config.calibration_profile_version,
            processed_at=config.processed_at,
            review_flags=(flag,) if flag else (),
            undetermined_color_hex=config.undetermined_color_hex,
            overwrite=config.force,
        )
        _write_detections(config.out_dir / vid, getattr(tracker, "last_detections", None))
    except Exception as error:  # noqa: BLE001 - FR-004: any per-video failure is reported, the batch goes on
        LOGGER.warning("%s failed: %s", vid, error)
        return VideoResult(vid, Outcome.FAILED, f"{type(error).__name__}: {error}", time.perf_counter() - started)
    return VideoResult(vid, Outcome.PROCESSED, "", time.perf_counter() - started)


def _write_detections(video_dir: Path, detections) -> None:
    """Raw per-sample detections (score, box, keypoints) of a learned tracker, kept so later rules (e.g. Listing from
    the dorsal/ventral tilt) and thresholds can be tried without tracking the video again."""
    target = video_dir / "detections.parquet"
    if detections is None:
        target.unlink(missing_ok=True)  # a classical re-run must not leave a stale sidecar from an earlier model run
        return
    temporary = video_dir / "detections.parquet.tmp"
    try:
        detections.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _pool_task(args: tuple[Trial, RunConfig]) -> VideoResult:
    import cv2

    cv2.setNumThreads(1)  # one video per worker; avoid oversubscribing cores
    return process_trial(*args)


def dedupe_trials(trials: Sequence[Trial]) -> tuple[list[Trial], list[Trial]]:
    """Split into (first trial per video id, the later trials that share an id and would overwrite it)."""
    seen: set[str] = set()
    unique: list[Trial] = []
    duplicates: list[Trial] = []
    for trial in trials:
        (duplicates if video_id(trial) in seen else unique).append(trial)
        seen.add(video_id(trial))
    return unique, duplicates


def _run_isolated(trial: Trial, config: RunConfig, task: Callable) -> VideoResult:
    """Run one video in its own single-worker pool so a native crash can only take down that video."""
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            return pool.submit(task, (trial, config)).result()
    except BrokenProcessPool:
        return VideoResult(video_id(trial), Outcome.FAILED, "worker process crashed (native crash or out of memory)")
    except Exception as error:  # noqa: BLE001 - e.g. an unpicklable result; still only this video fails
        return VideoResult(video_id(trial), Outcome.FAILED, f"{type(error).__name__}: {error}")


def run_batch(
    trials: Sequence[Trial],
    config: RunConfig,
    *,
    workers: int,
    tracker: Tracker | None = None,
    on_result: Callable[[VideoResult], None] | None = None,
    _task: Callable = _pool_task,
) -> list[VideoResult]:
    """Process `trials`, returning one result per trial in input order. A `tracker` override is
    sequential-only (a closure cannot cross a process boundary)."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if tracker is not None and workers != 1:
        raise ValueError("a custom tracker requires workers=1")
    counts = Counter(video_id(t) for t in trials)
    seen: set[str] = set()
    runnable: list[tuple[int, Trial]] = []
    results: dict[int, VideoResult] = {}

    def record(index: int, result: VideoResult) -> None:
        results[index] = result
        if on_result is not None:
            try:
                on_result(result)
            except Exception:  # noqa: BLE001 - progress reporting must never abort the batch
                LOGGER.exception("on_result callback failed for %s", result.video_id)

    for index, trial in enumerate(trials):
        vid = video_id(trial)
        if counts[vid] > 1 and vid in seen:
            record(index, VideoResult(vid, Outcome.FAILED, f"duplicate video id {vid}: only its first trial is processed"))
            continue
        seen.add(vid)
        runnable.append((index, trial))

    if workers == 1:
        for index, trial in runnable:
            record(index, process_trial(trial, config, tracker=tracker or track_video))
    else:
        retry: list[tuple[int, Trial]] = []
        # spawn, not fork: forking a process that has OpenCV/threads running can deadlock
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [(index, trial, pool.submit(_task, (trial, config))) for index, trial in runnable]
            for index, trial, future in futures:
                try:
                    record(index, future.result())
                except BrokenProcessPool:
                    retry.append((index, trial))  # a worker died; which video did it is unknown
                except Exception as error:  # noqa: BLE001
                    record(index, VideoResult(video_id(trial), Outcome.FAILED, f"{type(error).__name__}: {error}"))
        for index, trial in retry:  # each alone, so only the crashing video fails
            record(index, _run_isolated(trial, config, _task))
    return [results[i] for i in range(len(trials))]
