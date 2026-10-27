"""Set-level rules of the gold reader (plan U3): one row per fish (EC-10), one calibration profile
(EC-21), dates (EC-27), frame rates (EC-26), tracker evidence (EC-29), and the small value helpers
they share. `dcs.gold` calls these; per-file checks live in `dcs.gold_checks`.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from dcs import schema
from dcs.config import ConfigError
from dcs.gold_checks import PARQUET_ERRORS, GoldDataError, one_line

LIST_LIMIT = 10  # ids shown in one error message


# --- tables -----------------------------------------------------------------------------------


def read_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """The index or the catalog, with every expected column (EC-25)."""
    try:
        table = pd.read_parquet(path)
    except PARQUET_ERRORS as error:
        raise GoldDataError(f"Cannot read {path}: {one_line(error)[:200]}") from error
    missing = [name for name in columns if name not in table.columns]
    if missing:
        raise GoldDataError(f"{path.name}: column(s) {', '.join(missing)} missing (EC-25)")
    return table


def check_not_blank(table: pd.DataFrame, columns: Sequence[str], source: str) -> None:
    """Key columns must hold a value in every row; a blank key would silently miss its fish."""
    for name in columns:
        blank_rows = int(table[name].map(blank).sum())
        if blank_rows:
            raise GoldDataError(f"{source}: blank {name} in {blank_rows} row(s)")


def check_unique(table: pd.DataFrame, key: list[str], source: str) -> None:
    """EC-10: one row per fish; a repeat is an error, never a silent choice."""
    repeated = table[table.duplicated(key, keep=False)]
    if repeated.empty:
        return
    shown = repeated[key].astype(str).agg("_".join, axis=1).unique()
    what = "video_id" if key == ["video_id"] else "subject (sex + subject_id)"
    raise GoldDataError(f"{source}: {what} appears more than once: {ids(shown)} (EC-10)")


def check_plain_ids(video_ids: Iterable[Any], source: str) -> None:
    """A video id names a folder inside the gold folder; anything with a path in it is rejected."""
    for value in video_ids:
        text = str(value)
        if text in ("", ".", "..") or Path(text).name != text:
            raise GoldDataError(f"{source}: video_id {text!r} is not a plain folder name")


# --- one fish --------------------------------------------------------------------------------


def check_identity(manifest: dict[str, Any], video_id: str) -> None:
    named = schema.video_id(str(manifest.get("sex")), str(manifest.get("subject_id")))
    if named != video_id:
        raise GoldDataError(f"{video_id}/{schema.MANIFEST_FILE} describes fish {named}, not {video_id} (EC-10)")


def check_labels(manifest: dict[str, Any], row: dict[str, Any], video_id: str, names: Sequence[str], source: str) -> None:
    """The manifest and the index/catalog row must give the same labels."""
    for name in names:
        if not same_label(manifest.get(name), row.get(name)):
            raise GoldDataError(
                f"{video_id}: {name} is {manifest.get(name)!r} in {schema.MANIFEST_FILE} but {row.get(name)!r} in {source}"
            )


def whole_number(manifest: dict[str, Any], key: str, video_id: str) -> int:
    value = manifest.get(key)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not float(value).is_integer():
        raise GoldDataError(f"{video_id}/{schema.MANIFEST_FILE}: {key} must be a whole number, got {value!r}")
    return int(value)


# --- the whole set ---------------------------------------------------------------------------


def chosen_profile(seen: Counter[str], kept: Counter[str], selected: str | None) -> str:
    """EC-21: the only profile among the kept fish, or the selected one (the others were dropped as they were read).

    `seen` counts every fish whose manifest was read (listed when a selection is not found); `kept` counts the
    fish that passed every per-fish check, so a profile whose fish were all dropped does not make a mix.
    """
    if selected is None:
        if len(kept) > 1:
            raise GoldDataError(
                f"The set mixes calibration profiles: {_counts(kept)}. Speeds are comparable only within one "
                "tracker and profile; select one with --profile <version>."
            )
        return next(iter(kept), "")
    if selected not in seen:
        raise GoldDataError(f"Calibration profile {selected!r} not found; profiles present: {_counts(seen) or 'none'}.")
    return selected


def _counts(profiles: Counter[str]) -> str:
    return ", ".join(f"{name} ({count} fish)" for name, count in profiles.most_common())


def check_dates(videos: pd.DataFrame, hint: str) -> None:
    """EC-27 / D-005: every kept fish needs its experiment date (it builds the validation groups)."""
    no_date = videos.loc[videos["date"].map(blank), "video_id"]
    if not no_date.empty:
        raise ConfigError(f"{len(no_date)} fish have no date ({ids(no_date)}). {hint}")


def fps_flag(videos: pd.DataFrame, tolerance: float) -> tuple[bool, tuple[float, float]]:
    """EC-26: (uniform, (min, max)); every fps must be a finite number > 0."""
    fps = pd.to_numeric(videos["video_fps"], errors="coerce").astype(float)
    bad = videos.loc[~np.isfinite(fps) | (fps <= 0), "video_id"]
    if not bad.empty:
        raise GoldDataError(f"video_fps must be a finite number > 0; not so for {ids(bad)}")
    low, high = float(fps.min()), float(fps.max())
    return (high - low) <= tolerance * low, (low, high)


def check_tracker_evidence(video_ids: Iterable[str], profile: str, is_model: bool, evidence_dir: Path | None) -> int:
    """EC-29: where prepds' per-video folder exists, detections.parquet must exist exactly for model runs.

    Returns how many fish were checked (0 when no prepds output folder is known).
    """
    if evidence_dir is None:
        return 0
    checked = 0
    disagree = []
    for video_id in video_ids:
        folder = evidence_dir / schema.PROCESSED_DIR_NAME / video_id
        if not folder.is_dir():
            continue
        checked += 1
        if (folder / schema.DETECTIONS_FILE).is_file() != is_model:
            disagree.append(video_id)
    if disagree:
        tracker = "model" if is_model else "classical"
        raise GoldDataError(
            f"Tracker evidence disagrees with profile {profile} ({tracker} tracker) for {ids(disagree)}: "
            f"a {schema.DETECTIONS_FILE} beside the frames marks a model-tracker run (D-003, EC-29)."
        )
    return checked


# --- values ------------------------------------------------------------------------------------


def blank(value: Any) -> bool:
    """None, NaN/NA or whitespace-only text."""
    if isinstance(value, str):
        return not value.strip()
    if not pd.api.types.is_scalar(value):
        return False
    return bool(pd.isna(value))


def compound_label(value: Any) -> str:
    """Trimmed, inner whitespace collapsed, upper case (EC-8): the one spelling rule for compounds."""
    return " ".join(str(value).split()).upper()


def same_label(left: Any, right: Any) -> bool:
    """Equal labels: both blank, equal numbers (1 == 1.0 == "1"), or equal text after trimming."""
    if blank(left) or blank(right):
        return blank(left) and blank(right)
    left_number, right_number = _number(left), _number(right)
    if left_number is not None and right_number is not None:
        return left_number == right_number
    return str(left).strip() == str(right).strip()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def ids(values: Iterable[Any]) -> str:
    """At most LIST_LIMIT ids, then a count of the rest."""
    shown = [str(value) for value in values]
    extra = len(shown) - LIST_LIMIT
    return ", ".join(shown[:LIST_LIMIT]) + (f" (+{extra} more)" if extra > 0 else "")
