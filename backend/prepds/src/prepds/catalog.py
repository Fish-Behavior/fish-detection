"""FR-001..004: workbook cleaning + video matching + exceptions report.

Matching normalization (FR-003, PRD Clarifications C11/C12, confirmed against
real data in docs/progress.md §0):

1. Strip a trailing parenthetical status suffix from the folder name
   (`FD-2-66 + methylone(DONE Compressed Only)` -> `FD-2-66 + methylone`) with
   a general regex, not a hardcoded string match - other suffix variants are
   handled the same way.
2. Normalize compound text, sex, and subject number the same way on both
   sides: `compound.strip().lower().replace(" ", "")`, sex uppercased,
   subject number compared as an int (independent of zero-padding width).

Exact full-row duplicate workbook rows (confirmed real: Subject #320, PRD
C13) are auto-deduplicated to one logical trial; both original row indices
are kept in the exceptions report for audit only, per C13 - no manual
disambiguation step.
"""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from prepds.models import CatalogExceptionsReport, MatchStatus, Trial
from prepds.video_io import probe

SHEET_NAME = "Sheet1"
COMPOUND_COLUMN = "Compund:"  # sic - matches the real workbook's own typo
EXCLUDED_COLUMN = "Body Tissue:"  # confirmed 100% empty across all real trials

FOLDER_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")
# {sex}_{subject}.mp4, case-insensitive, subject zero-padding not fixed width.
VIDEO_FILENAME_RE = re.compile(r"^([A-Za-z])_0*(\d+)\.mp4$", re.IGNORECASE)

DEFAULT_DURATION_TOLERANCE_S = 30.0


def load_workbook(db_path: Path) -> pd.DataFrame:
    """Read `Sheet1`, keep only real trial rows, drop the always-empty column.

    Header whitespace/embedded newlines (present on a few real columns, e.g.
    `"NTT \\nTime (min):"`) are stripped so header matching never depends on
    exact literal whitespace (FR-001).
    """
    df = pd.read_excel(db_path, sheet_name=SHEET_NAME)
    df.columns = [str(column).strip() for column in df.columns]
    df = df[df[COMPOUND_COLUMN].notna()].reset_index(drop=True)
    if EXCLUDED_COLUMN in df.columns:
        df = df.drop(columns=[EXCLUDED_COLUMN])
    return df


def dedupe_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Collapse exact full-row duplicates to one logical row each (FR-004, C13).

    Returns the deduplicated frame plus a `duplicate_groups` audit list (kept
    even though C13 says no manual disambiguation is needed - "still logged
    for visibility").
    """
    duplicate_mask = df.duplicated(keep=False)
    duplicate_groups: list[dict[str, Any]] = []
    for subject_id, group in df[duplicate_mask].groupby("Subject #:"):
        duplicate_groups.append(
            {
                "subject_id": _format_subject_id(subject_id),
                "row_indices": group.index.tolist(),
            }
        )
    deduped = df.drop_duplicates(keep="first").reset_index(drop=True)
    return deduped, duplicate_groups


def to_trials(df: pd.DataFrame) -> list[Trial]:
    """Convert cleaned+deduped rows into `Trial` objects (PRD §5.3).

    `match_status` defaults to `NO_VIDEO` pending `match_videos()` - every
    trial starts "not yet matched", not "confirmed unmatched".
    """
    trials = []
    for _, row in df.iterrows():
        trials.append(
            Trial(
                subject_id=_format_subject_id(row["Subject #:"]),
                sex=str(row["Sex (M/F):"]).strip().upper(),
                strain=str(row["Strain:"]).strip(),
                age=_to_float_or_none(row["Age (~):"]),
                compound=str(row[COMPOUND_COLUMN]).strip(),
                concentration_mM=_format_concentration(row["Conc. (mM):"]),
                date=_parse_yymmdd(row["Date of EXP:"]),
                agent_exposure_min=float(row["Agent Exposure Time (min):"]),  # may be NaN; guarded in find_duration_mismatches()
                video_path=None,
                match_status=MatchStatus.NO_VIDEO,
            )
        )
    return trials


def match_videos(
    trials: list[Trial], video_dir: Path
) -> tuple[list[Trial], list[Path], list[dict[str, Any]]]:
    """FR-003: normalized (compound, subject) matching against the local video mirror.

    Returns the trials with `match_status`/`video_path` resolved, the list of
    video files that matched no trial row (FR-004), and a third list of
    ambiguous-match collisions (see below) - never silently overwritten.

    Match key is (compound, subject number) - NOT sex. Empirically, against
    the real 337-video local mirror, the filename's leading letter is `F`/`f`
    on all 337 files regardless of the matching workbook row's actual
    `Sex (M/F):` value (198 M / 155 F rows) - it is not a reliable sex
    encoding (most likely short for "Fish", not "Female"), confirmed via
    docs/progress.md §0 inventory. Using it in the match key was tried first
    and silently dropped ~60% of real matches (141/352) before this was
    caught by actually running catalog against real data (T023), not just
    synthetic fixtures - see docs/progress.md §2 Design Decisions. Subject
    number is confirmed unique per compound in the *current* real workbook
    snapshot (checked: max 2 rows per (compound, subject) - exactly the known
    C13 duplicate, already collapsed before matching; zero subjects span >1
    compound). That check is a one-time empirical fact about today's data,
    not a guarantee about a future workbook update - so it is also actively
    enforced here: any (compound, subject) key that would resolve to two
    different video files, or that two non-identical trial rows both claim,
    is routed to `ambiguous_matches` instead of being silently overwritten or
    double-assigned.

    File discovery is case-insensitive on the `.mp4` extension (`glob("*.mp4")`
    alone would miss a real `.MP4`/`.Mp4` file on a case-sensitive filesystem,
    even though `VIDEO_FILENAME_RE` below already tolerates that case via
    `re.IGNORECASE` - the two were inconsistent before this fix).
    """
    video_index: dict[tuple[str, int], Path] = {}
    all_video_paths: set[Path] = set()
    ambiguous_matches: list[dict[str, Any]] = []
    # A compound folder is any directory that directly holds .mp4 files, at any depth (the real mirror is
    # video_dir/Phase_N/<compound folder>/<letter>_<subject>.mp4).
    compound_dirs = sorted({p.parent for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"})
    for compound_dir in compound_dirs:
        normalized_compound = _normalize_compound(FOLDER_SUFFIX_RE.sub("", compound_dir.name))
        video_files = sorted(
            p for p in compound_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"
        )
        for video_path in video_files:
            all_video_paths.add(video_path)
            match = VIDEO_FILENAME_RE.match(video_path.name)
            if match is None:
                continue  # not a {letter}_{subject}.mp4-shaped file; can't key it
            _leading_letter, subject_str = match.groups()
            key = (normalized_compound, int(subject_str))
            existing = video_index.get(key)
            if existing is not None and existing != video_path:
                ambiguous_matches.append(
                    {
                        "kind": "duplicate_video_files",
                        "compound": normalized_compound,
                        "subject_id": _format_subject_id(subject_str),
                        "video_paths": [str(existing), str(video_path)],
                    }
                )
                continue  # keep the first one found; both are reported either way
            video_index[key] = video_path

    matched_video_paths: set[Path] = set()
    claimed_keys: dict[tuple[str, int], str] = {}  # key -> subject_id that already claimed it
    resolved_trials = []
    for trial in trials:
        key = (_normalize_compound(trial.compound), int(trial.subject_id))
        video_path = video_index.get(key)
        prior_subject_id = claimed_keys.get(key)
        if video_path is not None and prior_subject_id is not None:
            ambiguous_matches.append(
                {
                    "kind": "duplicate_trial_key",
                    "compound": trial.compound,
                    "subject_ids": [prior_subject_id, trial.subject_id],
                    "video_path": str(video_path),
                }
            )
            resolved_trials.append(trial)  # left as NO_VIDEO - never double-assigned
            continue
        if video_path is not None:
            claimed_keys[key] = trial.subject_id
            matched_video_paths.add(video_path)
            resolved_trials.append(
                dataclasses.replace(trial, video_path=video_path, match_status=MatchStatus.MATCHED)
            )
        else:
            resolved_trials.append(trial)  # already NO_VIDEO from to_trials()

    unmatched_videos = sorted(all_video_paths - matched_video_paths)
    return resolved_trials, unmatched_videos, ambiguous_matches


def find_duration_mismatches(
    trials: list[Trial], tolerance_s: float = DEFAULT_DURATION_TOLERANCE_S
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """FR-004: flag matched videos whose measured duration deviates from expectation.

    Only probes trials that are actually MATCHED (with a video_path). A video
    that fails to open/read (corrupt or zero-byte, §3.3 edge case) is
    reported in the second return list, never raised - one bad file must not
    halt the batch.
    """
    mismatches: list[dict[str, Any]] = []
    corrupt: list[dict[str, Any]] = []
    for trial in trials:
        if trial.match_status != MatchStatus.MATCHED or trial.video_path is None:
            continue
        if math.isnan(trial.agent_exposure_min):
            # NaN comparisons are always False in Python - `abs(x - nan) >
            # tolerance` would silently look "within tolerance" rather than
            # flagging a genuinely missing expected value. Report it
            # explicitly instead (as a mismatch, since it can't be verified).
            mismatches.append(
                {
                    "subject_id": trial.subject_id,
                    "video_path": str(trial.video_path),
                    "expected_s": None,
                    "measured_s": None,
                    "error": "agent_exposure_min is missing (NaN) - cannot check duration",
                }
            )
            continue
        expected_s = trial.agent_exposure_min * 60.0
        try:
            asset = probe(trial.video_path)
        except ValueError as error:
            corrupt.append(
                {
                    "subject_id": trial.subject_id,
                    "video_path": str(trial.video_path),
                    "error": str(error),
                }
            )
            continue
        if abs(asset.duration_s - expected_s) > tolerance_s:
            mismatches.append(
                {
                    "subject_id": trial.subject_id,
                    "video_path": str(trial.video_path),
                    "expected_s": expected_s,
                    "measured_s": asset.duration_s,
                    "measured_fps": asset.fps,
                }
            )
    return mismatches, corrupt


def build_exceptions_report(
    trials: list[Trial],
    duplicate_groups: list[dict[str, Any]],
    unmatched_videos: list[Path],
    duration_mismatches: list[dict[str, Any]],
    corrupt_videos: list[dict[str, Any]],
    ambiguous_matches: list[dict[str, Any]] | None = None,
) -> CatalogExceptionsReport:
    """Assemble the final, JSON-serializable exceptions report (FR-004)."""
    unmatched_trials = [
        {
            "subject_id": trial.subject_id,
            "compound": trial.compound,
            "sex": trial.sex,
        }
        for trial in trials
        if trial.match_status != MatchStatus.MATCHED
    ]
    return CatalogExceptionsReport(
        unmatched_trials=unmatched_trials,
        unmatched_videos=[{"path": str(path)} for path in unmatched_videos],
        duplicate_trials=duplicate_groups,
        duration_mismatches=duration_mismatches,
        corrupt_videos=corrupt_videos,
        ambiguous_matches=ambiguous_matches or [],
    )


# --- small private helpers ------------------------------------------------------


def _normalize_compound(compound: str) -> str:
    return compound.strip().lower().replace(" ", "")


def _format_subject_id(value: Any) -> str:
    return f"{int(value):04d}"


def _to_float_or_none(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def _format_concentration(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()  # combo doses like "0.03 + 0.01" aren't numeric
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _parse_yymmdd(value: Any):
    """Parse the workbook's `Date of EXP:` (int/float YYMMDD, 21st century)."""
    import datetime as dt

    digits = f"{int(value):06d}"
    year = 2000 + int(digits[0:2])
    month = int(digits[2:4])
    day = int(digits[4:6])
    return dt.date(year, month, day)
