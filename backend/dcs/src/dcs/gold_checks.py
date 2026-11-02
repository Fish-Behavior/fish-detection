"""Per-video file checks of the gold reader (plan U3): EC-1, EC-22, EC-25.

A fish whose files are missing or unreadable is dropped (EC-1, a `FileProblem`); a file that
reads but breaks the prepds contract (a missing key or column, a retyped column, an unknown or
empty state, `Undetermined` in reviewed gold) stops the run with a `GoldDataError` naming the
file and the column. Checking is split in two so the reader can drop a fish on its manifest
(review status, profile) before its data files are looked at: `read_manifest`, then `inspect_data`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dcs import schema
from dcs.config import ConfigError

DROP_MISSING_FILE = "missing_file"
DROP_UNREADABLE = "unreadable_file"
# Data files read after the status drop; strip.png and provenance.json are not needed.
DATA_FILES = (schema.FRAMES_FILE, schema.SEGMENTS_FILE)
# Manifest keys needed to decide a fish's identity and status; the rest are checked after the status drop.
IDENTITY_KEYS = ("sex", "subject_id", "review_status")
SEGMENT_TIMES = ("start_s", "end_s", "duration_s")

# How each pandas dtype of schema.FRAME_DTYPES looks in the parquet schema (read without loading rows).
ARROW_TYPE_CHECKS: dict[str, Callable[[pa.DataType], bool]] = {
    "int32": pa.types.is_int32,
    "float32": pa.types.is_float32,
    "bool": pa.types.is_boolean,
    "category": pa.types.is_dictionary,
}
PARQUET_ERRORS = (OSError, pa.ArrowException)  # ArrowInvalid is also a ValueError; ArrowIOError an OSError
CSV_ERRORS = (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError)


class GoldDataError(ConfigError):
    """The input breaks the prepds file contract; the message names the file, the column or the fish."""


@dataclass(frozen=True)
class FileProblem:
    """Why a fish is dropped (EC-1): `reason` is DROP_MISSING_FILE or DROP_UNREADABLE."""

    reason: str
    detail: str


@dataclass(frozen=True)
class VideoFiles:
    """What the reader keeps from a fish's data files once they passed the checks."""

    undetermined_share: float  # share of the recording labeled Undetermined (0 in reviewed gold)


def read_manifest(folder: Path) -> dict[str, Any] | FileProblem:
    """Step 1: the manifest present and readable, with the keys that decide identity and status."""
    video = folder.name
    if not folder.is_dir():
        return FileProblem(DROP_MISSING_FILE, f"folder {video}/ not found")
    if not (folder / schema.MANIFEST_FILE).is_file():
        return FileProblem(DROP_MISSING_FILE, f"{video}/: {schema.MANIFEST_FILE} missing")
    try:
        manifest = json.loads((folder / schema.MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return unreadable(video, schema.MANIFEST_FILE, error)
    if not isinstance(manifest, dict):
        return FileProblem(DROP_UNREADABLE, f"{video}/{schema.MANIFEST_FILE}: not a JSON object")
    check_manifest_keys(manifest, video, IDENTITY_KEYS)
    return manifest


def check_manifest_keys(manifest: dict[str, Any], video: str, keys: tuple[str, ...] = schema.MANIFEST_KEYS) -> None:
    """EC-25: every key prepds writes (by default) is present."""
    absent = [key for key in keys if key not in manifest]
    if absent:
        raise GoldDataError(f"{video}/{schema.MANIFEST_FILE}: key(s) {', '.join(absent)} missing (EC-25)")


def inspect_data(folder: Path, *, allow_undetermined: bool) -> VideoFiles | FileProblem:
    """Step 2: data files present; frames schema and state column; segments rows, columns, times and states."""
    video = folder.name
    missing = [name for name in DATA_FILES if not (folder / name).is_file()]
    if missing:
        return FileProblem(DROP_MISSING_FILE, f"{video}/: {', '.join(missing)} missing")
    frames_path = folder / schema.FRAMES_FILE
    try:
        frames_schema = pq.read_schema(frames_path)
        check_frames_schema(frames_schema, video)
        states = pq.read_table(frames_path, columns=["state"]).column("state").to_pandas()
    except PARQUET_ERRORS as error:  # the footer may read while a data page is damaged
        return unreadable(video, schema.FRAMES_FILE, error)
    try:
        segments = pd.read_csv(folder / schema.SEGMENTS_FILE)
    except CSV_ERRORS as error:
        return unreadable(video, schema.SEGMENTS_FILE, error)
    if segments.empty:  # a header without rows: every column reads as text, so no dtype check can say why
        return FileProblem(DROP_UNREADABLE, f"{video}/{schema.SEGMENTS_FILE}: no rows")
    check_states(states, f"{video}/{schema.FRAMES_FILE}", allow_undetermined=allow_undetermined)
    check_segments(segments, video, allow_undetermined=allow_undetermined)
    return VideoFiles(undetermined_share=undetermined_share(segments))


def load_files(folder: Path, *, allow_undetermined: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a fish's frames and segments in full and check them again (they may have changed since the scan)."""
    video = folder.name
    try:
        frames = pd.read_parquet(folder / schema.FRAMES_FILE)
    except PARQUET_ERRORS as error:
        raise GoldDataError(f"Cannot read {video}/{schema.FRAMES_FILE}: {one_line(error)}") from error
    check_frames(frames, f"{video}/{schema.FRAMES_FILE}", allow_undetermined=allow_undetermined)
    try:
        segments = pd.read_csv(folder / schema.SEGMENTS_FILE)
    except CSV_ERRORS as error:
        raise GoldDataError(f"Cannot read {video}/{schema.SEGMENTS_FILE}: {one_line(error)}") from error
    check_segments(segments, video, allow_undetermined=allow_undetermined)
    return frames, segments


def check_frames_schema(frames_schema: pa.Schema, video: str) -> None:
    """EC-25 from the parquet schema alone: every expected column present with its type."""
    where = f"{video}/{schema.FRAMES_FILE}"
    missing = [name for name in schema.FRAME_DTYPES if name not in frames_schema.names]
    if missing:
        raise GoldDataError(f"{where}: column(s) {', '.join(missing)} missing (EC-25)")
    for name, dtype in schema.FRAME_DTYPES.items():
        found = frames_schema.field(name).type
        if not ARROW_TYPE_CHECKS[dtype](found):
            raise GoldDataError(f"{where}: column {name} has type {found}, expected {dtype} (EC-25)")


def check_frames(frames: pd.DataFrame, where: str, *, allow_undetermined: bool) -> None:
    """EC-25 and EC-22 on loaded frames: columns, pandas dtypes and state names."""
    missing = [name for name in schema.FRAME_DTYPES if name not in frames.columns]
    if missing:
        raise GoldDataError(f"{where}: column(s) {', '.join(missing)} missing (EC-25)")
    for name, dtype in schema.FRAME_DTYPES.items():
        if str(frames[name].dtype) != dtype:
            raise GoldDataError(f"{where}: column {name} has dtype {frames[name].dtype}, expected {dtype} (EC-25)")
    check_states(frames["state"], where, allow_undetermined=allow_undetermined)


def check_segments(segments: pd.DataFrame, video: str, *, allow_undetermined: bool) -> None:
    """EC-25: the segment columns, numeric times; then the states."""
    where = f"{video}/{schema.SEGMENTS_FILE}"
    missing = [name for name in schema.SEGMENT_COLUMNS if name not in segments.columns]
    if missing:
        raise GoldDataError(f"{where}: column(s) {', '.join(missing)} missing (EC-25)")
    for name in SEGMENT_TIMES:
        if not pd.api.types.is_numeric_dtype(segments[name]):
            raise GoldDataError(f"{where}: column {name} is not numeric (dtype {segments[name].dtype}) (EC-25)")
    check_states(segments["state"], where, allow_undetermined=allow_undetermined)


def check_states(states: pd.Series, where: str, *, allow_undetermined: bool) -> None:
    """Only prepds state names, none empty; `Undetermined` only where allowed (unreviewed output, EC-22)."""
    empty = int(states.isna().sum())
    if empty:
        raise GoldDataError(f"{where}: empty state in {empty} row(s)")
    found = set(states.astype(str).unique())
    unknown = sorted(found - set(schema.STATES))
    if unknown:
        raise GoldDataError(f"{where}: unknown state(s) {', '.join(unknown)}; expected one of {', '.join(schema.STATES)}")
    if not allow_undetermined and schema.UNDETERMINED in found:
        raise GoldDataError(
            f"{where}: state {schema.UNDETERMINED} found in an Accepted video; the gold-data contract is "
            "violated (EC-22). Review the video again in the prepds review app."
        )


def undetermined_share(segments: pd.DataFrame) -> float:
    """Share of the recording's segment time labeled Undetermined (0 for an empty recording)."""
    total = float(segments["duration_s"].sum())
    if total <= 0:
        return 0.0
    return float(segments.loc[segments["state"] == schema.UNDETERMINED, "duration_s"].sum()) / total


def unreadable(video: str, name: str, error: Exception) -> FileProblem:
    """A dropped fish whose file exists but cannot be read; the detail keeps the first 200 characters."""
    return FileProblem(DROP_UNREADABLE, f"{video}/{name}: {one_line(error)[:200]}")


def one_line(error: Exception) -> str:
    return " ".join(str(error).split())
