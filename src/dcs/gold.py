"""Read the fish to train on from prepds output, with the contract checks of plan U3 (T1.7).

Two sources, chosen by `training.gold_source` (D-033):

- `accepted`: the reviewed gold folder, `<DCS_ACCEPTED_DIR>/accepted_index.parquet` plus one
  `<video_id>/` folder per fish (PRD §5.1).
- `processed`: the unreviewed pipeline output, `<DCS_PROCESSED_DIR>/trials_catalog.parquet` plus
  `processed/<video_id>/`. Used until videos are Accepted (owner, 2026-10-03); `Undetermined` is
  allowed there and every row is marked unreviewed.

Both give a `GoldSet`: one row per kept fish, a table of dropped fish with reasons (never silent,
EC-1), the single calibration profile and tracker (EC-21, D-003) and the frame-rate flag (EC-26).
Per fish, the manifest is read first so that status and profile drops come before the data checks.
Per-video files are found from the video id, never from stored paths (D-012).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from dcs import schema
from dcs.config import ConfigError, Settings
from dcs.config_rules import SOURCE_ACCEPTED, SOURCE_PROCESSED
from dcs.gold_checks import (
    DROP_MISSING_FILE,
    DROP_UNREADABLE,
    FileProblem,
    GoldDataError,
    check_manifest_keys,
    inspect_data,
    load_files,
    read_manifest,
)
from dcs.gold_rules import (
    check_dates,
    check_identity,
    check_labels,
    check_not_blank,
    check_plain_ids,
    check_tracker_evidence,
    check_unique,
    chosen_profile,
    fps_flag,
    read_table,
    whole_number,
)

__all__ = [
    "DROP_MISSING_FILE",
    "DROP_OTHER_PROFILE",
    "DROP_STATUS",
    "DROP_UNREADABLE",
    "SOURCE_ACCEPTED",
    "SOURCE_PROCESSED",
    "GoldDataError",
    "GoldSet",
    "ReadOptions",
    "Video",
    "load_video",
    "read_accepted",
    "read_gold",
    "read_processed",
    "read_videos",
    "SOURCE_VIDEOS",
]

SOURCE_VIDEOS = "videos"  # per-video folders without index or catalog, for predict (D-018); never trained on
TRACKER_CLASSICAL = "classical"
TRACKER_MODEL = "model"
DROP_STATUS = "review_status"  # REJECTED or NOT_PROCESSED in the unreviewed output
DROP_OTHER_PROFILE = "other_profile"  # not the calibration profile selected in ReadOptions
DEFAULT_MARKER = "-model"  # same as config/default_training.yaml (a test keeps them equal)
DEFAULT_FPS_TOLERANCE = 0.01
KEPT_STATUSES = (schema.REVIEW_PROCESSED_AUTO, schema.REVIEW_EDITED, schema.REVIEW_ACCEPTED)

VIDEO_COLUMNS = (
    "video_id",
    "subject_id",
    "sex",
    "compound",
    "concentration_mM",
    "date",
    "strain",
    "age",
    "agent_exposure_min",
    "video_fps",
    "video_duration_s",
    "calibration_profile_version",
    "pipeline_version",
    "review_status",
    "reviewed",
    "review_flag_count",
    "edit_count",
    "undetermined_share",
)
DROPPED_COLUMNS = ("video_id", "reason", "detail")
INDEX_FIELDS = tuple(name for name in VIDEO_COLUMNS if name in schema.INDEX_COLUMNS)
CHECKED_LABELS = ("compound", "concentration_mM")  # must read the same in the manifest and the index/catalog
NO_DATE_HINTS = {
    SOURCE_ACCEPTED: "Run `prepds catalog`, then `prepds export-index`, so the index gets the workbook fields.",
    SOURCE_PROCESSED: "Run `prepds catalog` so every processed video has a catalog row with its date.",
}

Dropped = list[tuple[str, str, str]]


@dataclass(frozen=True)
class ReadOptions:
    """How to read a set: the calibration profile to keep (None = the only one present), the text marking
    model-tracker profiles (D-003) and the relative fps spread that still counts as uniform (EC-26)."""

    profile: str | None = None
    marker: str = DEFAULT_MARKER
    fps_tolerance: float = DEFAULT_FPS_TOLERANCE

    def __post_init__(self) -> None:
        if not isinstance(self.marker, str) or not self.marker.strip():
            raise ConfigError(f"The model-tracker marker must be non-empty text, got {self.marker!r}.")
        tolerance = self.fps_tolerance
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 <= tolerance < math.inf:
            raise ConfigError(f"fps_tolerance must be a finite number >= 0, got {tolerance!r}.")


@dataclass(frozen=True, eq=False, kw_only=True)
class GoldSet:
    """The fish to train on. `videos` and `dropped` are fresh copies on every access."""

    source: str
    folder: Path  # holds one <video_id>/ folder per fish
    profile: str
    tracker: str
    fps_uniform: bool
    fps_range: tuple[float, float]
    tracker_checked: int  # fish whose tracker evidence (detections.parquet or its absence) was looked at
    video_ids: tuple[str, ...]  # kept fish, in table order
    _videos: pd.DataFrame = field(repr=False)
    _dropped: pd.DataFrame = field(repr=False)

    @property
    def videos(self) -> pd.DataFrame:
        """One row per kept fish, columns VIDEO_COLUMNS."""
        return self._videos.copy()

    @property
    def dropped(self) -> pd.DataFrame:
        """One row per dropped fish: video_id, reason, detail."""
        return self._dropped.copy()

    @property
    def allows_undetermined(self) -> bool:
        return self.source != SOURCE_ACCEPTED  # only Accepted videos are free of Undetermined (EC-22)

    def folder_of(self, video_id: str) -> Path:
        return self.folder / video_id


@dataclass(frozen=True)
class Video:
    video_id: str
    frames: pd.DataFrame
    segments: pd.DataFrame


def read_gold(settings: Settings, *, profile: str | None = None) -> GoldSet:
    """Read the source chosen by `training.gold_source`, with the marker and fps tolerance from the settings."""
    training = settings.training
    options = ReadOptions(profile, training["model_profile_marker"], training["fps_tolerance"])
    if training["gold_source"] == SOURCE_PROCESSED:
        return read_processed(settings.require("processed_dir"), options)
    return read_accepted(settings.require("accepted_dir"), options, processed_dir=settings.paths.processed_dir)


def read_accepted(accepted_dir: Path, options: ReadOptions | None = None, *, processed_dir: Path | None = None) -> GoldSet:
    """Read the reviewed gold folder; `processed_dir` (the prepds output folder) adds the tracker cross-check."""
    options = options or ReadOptions()
    accepted_dir = Path(accepted_dir)
    index_path = accepted_dir / schema.INDEX_FILE
    if not index_path.is_file():
        raise ConfigError(
            f"No {schema.INDEX_FILE} in {accepted_dir}. Run `prepds export-index` on the machine that holds "
            "the gold dataset, or point DCS_ACCEPTED_DIR at it."
        )
    index = read_table(index_path, schema.INDEX_COLUMNS)
    check_plain_ids(index["video_id"], schema.INDEX_FILE)
    check_unique(index, ["video_id"], schema.INDEX_FILE)
    check_unique(index, ["sex", "subject_id"], schema.INDEX_FILE)

    rows: list[dict[str, Any]] = []
    dropped: Dropped = []
    seen: Counter[str] = Counter()
    for record in index.to_dict("records"):
        video_id = str(record["video_id"])
        manifest = read_manifest(accepted_dir / video_id)
        if isinstance(manifest, FileProblem):
            dropped.append((video_id, manifest.reason, manifest.detail))
            continue
        check_identity(manifest, video_id)
        if manifest["review_status"] != schema.REVIEW_ACCEPTED:
            raise GoldDataError(
                f"{video_id}/{schema.MANIFEST_FILE}: review status {manifest['review_status']}, but the "
                f"fish is in {schema.INDEX_FILE}; re-run `prepds export-index`."
            )
        check_manifest_keys(manifest, video_id)
        check_labels(manifest, record, video_id, (*CHECKED_LABELS, "calibration_profile_version"), schema.INDEX_FILE)
        if not _keep_profile(record["calibration_profile_version"], video_id, options, seen, dropped):
            continue
        files = inspect_data(accepted_dir / video_id, allow_undetermined=False)
        if isinstance(files, FileProblem):
            dropped.append((video_id, files.reason, files.detail))
            continue
        row = {name: record[name] for name in INDEX_FIELDS}
        row.update(_review_fields(manifest, video_id), undetermined_share=files.undetermined_share)
        rows.append(row)
    return _assemble(SOURCE_ACCEPTED, accepted_dir, rows, dropped, seen, options, processed_dir)


def read_processed(output_dir: Path, options: ReadOptions | None = None) -> GoldSet:
    """Read the unreviewed prepds output folder (`trials_catalog.parquet` + `processed/<video_id>/`)."""
    options = options or ReadOptions()
    output_dir = Path(output_dir)
    catalog_path = output_dir / schema.CATALOG_FILE
    processed = output_dir / schema.PROCESSED_DIR_NAME
    if not catalog_path.is_file():
        raise ConfigError(
            f"No {schema.CATALOG_FILE} in {output_dir}. Run `prepds catalog` and `prepds run`, or point "
            "DCS_PROCESSED_DIR at the prepds output folder."
        )
    if not processed.is_dir():
        raise ConfigError(f"No {schema.PROCESSED_DIR_NAME}/ folder in {output_dir}. Run `prepds run` first.")
    catalog = read_table(catalog_path, schema.CATALOG_COLUMNS)
    catalog = catalog[catalog["match_status"] == schema.MATCH_MATCHED]
    check_not_blank(catalog, ["sex", "subject_id"], schema.CATALOG_FILE)
    check_unique(catalog, ["sex", "subject_id"], schema.CATALOG_FILE)
    trials = {schema.video_id(str(row["sex"]), str(row["subject_id"])): row for row in catalog.to_dict("records")}

    rows: list[dict[str, Any]] = []
    dropped: Dropped = []
    seen: Counter[str] = Counter()
    folders = sorted(path for path in processed.iterdir() if path.is_dir())
    for folder in folders:
        video_id = folder.name
        manifest = read_manifest(folder)
        if isinstance(manifest, FileProblem):
            dropped.append((video_id, manifest.reason, manifest.detail))
            continue
        check_identity(manifest, video_id)
        status = manifest["review_status"]
        if status not in schema.REVIEW_STATUSES:
            raise GoldDataError(f"{video_id}/{schema.MANIFEST_FILE}: unknown review status {status!r}")
        if status not in KEPT_STATUSES:
            dropped.append((video_id, DROP_STATUS, f"review status {status}"))
            continue
        check_manifest_keys(manifest, video_id)
        trial = trials.get(video_id, {})
        if trial:
            check_labels(manifest, trial, video_id, CHECKED_LABELS, schema.CATALOG_FILE)
        if not _keep_profile(manifest["calibration_profile_version"], video_id, options, seen, dropped):
            continue
        files = inspect_data(folder, allow_undetermined=True)
        if isinstance(files, FileProblem):
            dropped.append((video_id, files.reason, files.detail))
            continue
        row = {name: manifest.get(name) for name in INDEX_FIELDS if name not in schema.WORKBOOK_INDEX_FIELDS}
        row.update({name: trial.get(name) for name in schema.WORKBOOK_INDEX_FIELDS})
        row.update(video_id=video_id, **_review_fields(manifest, video_id), undetermined_share=files.undetermined_share)
        rows.append(row)
    run = {folder.name for folder in folders}
    for video_id in sorted(set(trials) - run):  # matched to a video but never run, or deleted since (EC-1)
        dropped.append((video_id, DROP_MISSING_FILE, f"no {schema.PROCESSED_DIR_NAME}/{video_id}/ folder (not run yet?)"))
    return _assemble(SOURCE_PROCESSED, processed, rows, dropped, seen, options, output_dir)


def read_videos(folder: Path, options: ReadOptions | None = None) -> GoldSet:
    """Per-video folders (`<folder>/<video_id>/` with manifest, frames, segments) without an index or catalog
    (D-018): for scoring new fish with `predict`. Labels come from the manifest when it has them; date and the
    workbook fields stay empty. Rejected and unprocessed videos are dropped; the rest count as unreviewed unless
    Accepted."""
    options = options or ReadOptions()
    folder = Path(folder)
    if not folder.is_dir():
        raise ConfigError(f"--videos {folder} is not a folder.")
    rows: list[dict[str, Any]] = []
    dropped: Dropped = []
    seen: Counter[str] = Counter()
    for video in sorted(path for path in folder.iterdir() if path.is_dir()):
        manifest = read_manifest(video)
        if isinstance(manifest, FileProblem):
            dropped.append((video.name, manifest.reason, manifest.detail))
            continue
        check_identity(manifest, video.name)
        if manifest["review_status"] not in KEPT_STATUSES:
            dropped.append((video.name, DROP_STATUS, f"review status {manifest['review_status']}"))
            continue
        check_manifest_keys(manifest, video.name)
        if not _keep_profile(manifest["calibration_profile_version"], video.name, options, seen, dropped):
            continue
        files = inspect_data(video, allow_undetermined=True)
        if isinstance(files, FileProblem):
            dropped.append((video.name, files.reason, files.detail))
            continue
        row = {name: manifest.get(name) for name in INDEX_FIELDS if name not in schema.WORKBOOK_INDEX_FIELDS}
        row.update(video_id=video.name, **_review_fields(manifest, video.name), undetermined_share=files.undetermined_share)
        rows.append(row)
    return _assemble(SOURCE_VIDEOS, folder, rows, dropped, seen, options, None, need_dates=False)


def load_video(gold: GoldSet, video_id: str) -> Video:
    """Read one kept fish's frames and segments; both are checked again, since files may change after the scan."""
    if video_id not in gold.video_ids:
        raise KeyError(f"{video_id} is not in this gold set")
    frames, segments = load_files(gold.folder_of(video_id), allow_undetermined=gold.allows_undetermined)
    return Video(video_id=video_id, frames=frames, segments=segments)


# --- assembly ----------------------------------------------------------------------------------


def _keep_profile(profile: Any, video_id: str, options: ReadOptions, seen: Counter[str], dropped: Dropped) -> bool:
    """Count the fish's profile (EC-21); False (and listed as dropped) when another profile was selected."""
    if not isinstance(profile, str) or not profile.strip():
        raise GoldDataError(f"calibration_profile_version is blank for {video_id}; cannot tell its tracker (D-003)")
    seen[profile] += 1
    if options.profile is not None and profile != options.profile:
        dropped.append((video_id, DROP_OTHER_PROFILE, f"profile {profile}"))
        return False
    return True


def _assemble(
    source: str,
    folder: Path,
    rows: list[dict[str, Any]],
    dropped: Dropped,
    seen: Counter[str],
    options: ReadOptions,
    evidence_dir: Path | None,
    need_dates: bool = True,
) -> GoldSet:
    """Set-level rules in order: one profile (EC-21), something left, dates (EC-27), tracker (EC-29), fps (EC-26)."""
    videos = pd.DataFrame(rows, columns=list(VIDEO_COLUMNS))
    profile = chosen_profile(seen, Counter(videos["calibration_profile_version"]), options.profile)
    if videos.empty:
        reasons = dict(Counter(reason for _, reason, _ in dropped))
        raise GoldDataError(f"No usable fish in {folder}; dropped by reason: {reasons}")
    if need_dates:
        check_dates(videos, NO_DATE_HINTS[source])
    is_model = options.marker in profile
    evidence = None if evidence_dir is None else Path(evidence_dir)
    checked = check_tracker_evidence(videos["video_id"], profile, is_model, evidence)
    uniform, fps_range = fps_flag(videos, options.fps_tolerance)
    return GoldSet(
        source=source,
        folder=folder,
        profile=profile,
        tracker=TRACKER_MODEL if is_model else TRACKER_CLASSICAL,
        fps_uniform=uniform,
        fps_range=fps_range,
        tracker_checked=checked,
        video_ids=tuple(str(video_id) for video_id in videos["video_id"]),
        _videos=videos,
        _dropped=pd.DataFrame(dropped, columns=list(DROPPED_COLUMNS)),
    )


def _review_fields(manifest: dict[str, Any], video_id: str) -> dict[str, Any]:
    status = manifest["review_status"]
    return {
        "review_status": status,
        "reviewed": status == schema.REVIEW_ACCEPTED,
        "review_flag_count": len(manifest.get("review_flags") or ()),
        "edit_count": whole_number(manifest, "edit_count", video_id),
    }
