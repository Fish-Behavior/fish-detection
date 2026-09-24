"""Workbook catalog: clean the trial workbook and match every trial to its video(s).

This is the first data step of the pipeline. It answers two questions before any
video is processed:

1. *Which trials are real and what do we know about them?*
   The workbook is read, its headers are mapped to stable column names, and the
   cleaning rules from docs/zebrafish_drug_detection_scope.md (Section 5) are applied.
2. *Which video file belongs to which trial?*
   Video file names (e.g. ``F_0042.mp4``) are parsed for their subject number and
   matched to the workbook. Recordings split into parts (``0012a``/``0012b``) are joined into one subject,
   with the videos kept in part order.

Outputs (written to ``<FISH_OUTPUT_DIR>/catalog/``, which Git ignores):
    trials.csv             one row per subject, cleaned values + matched video paths
    videos.csv             one row per video file found, with what was parsed from its name
    validation_report.md   human-readable list of everything that needs attention

Later steps read trials.csv back with `load_trials` and filter it with `select_subjects`.
"""

from __future__ import annotations

import logging
import numbers
import re
import warnings
from datetime import date
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from fishbehavior.config import ConfigError, Settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
# Workbook header (after `normalize_header`) -> stable name used in the code.
# Keeping this in one table means a renamed or re-typed header only needs one edit here.
COLUMN_MAP = {
    "date of exp": "exp_date",
    "subject #": "subject_id",
    "strain": "strain",
    "sex (m/f)": "sex",
    "age (~)": "age",
    "compund": "compound",  # the workbook header has this typo
    "compound": "compound",  # ...and the correct spelling, in case it is fixed later
    "conc. (mm)": "concentration_raw",
    "agent exposure time (min)": "exposure_min",
    "ntt time (min)": "ntt_min",
    "uv exposure time (min)": "uv_min",
    "tdm (full arena)": "tdm_full",
    "tdm (top half)": "tdm_top",
    "tdm (bot half)": "tdm_bottom",
    "velocity (full arena)": "velocity_full",
    "velocity (top half)": "velocity_top",
    "velocity (bot half)": "velocity_bottom",
    "time spent (top)": "time_top_s",
    "time spent (bot)": "time_bottom_s",
    "h2o (before)": "h2o_before",
    "h2o (after)": "h2o_after",
    "brain tissue": "brain_tissue",
    "body tissue": "body_tissue",
}

# The eight NTT movement measurements (scope §5 calls these "the eight movement columns").
MOVEMENT_COLUMNS = [
    "tdm_full", "tdm_top", "tdm_bottom",
    "velocity_full", "velocity_top", "velocity_bottom",
    "time_top_s", "time_bottom_s",
]
# Only the top-/bottom-half columns may contain "-", meaning "zero time in that zone" (scope §5).
ZONE_COLUMNS = ["tdm_top", "tdm_bottom", "velocity_top", "velocity_bottom", "time_top_s", "time_bottom_s"]

# Columns without which the catalog cannot be built at all.
REQUIRED_COLUMNS = ["subject_id", "sex", "compound", *MOVEMENT_COLUMNS]

# Columns dropped on purpose (scope §5: empty for every real trial).
DROPPED_COLUMNS = ["body_tissue"]

# Numeric columns other than the movement ones; text such as "N/A" becomes missing.
OTHER_NUMERIC_COLUMNS = ["age", "exposure_min", "ntt_min", "uv_min"]

# Possible values of `video_status` in trials.csv.
STATUS_MATCHED = "matched"  # every expected video found
STATUS_MISSING = "missing"  # no video for this subject
STATUS_PART_MISMATCH = "part_mismatch"  # split recording, but the part files do not line up
STATUS_DUPLICATE = "duplicate_videos"  # several files for a single-part subject
STATUS_NOT_CHECKED = "not_checked"  # FISH_VIDEO_DIR not configured


class CatalogError(RuntimeError):
    """The workbook cannot be used at all (e.g. a required column is missing)."""


@dataclass
class CatalogResult:
    """Everything `build_catalog` produces, kept together for the CLI and later steps."""

    trials: pd.DataFrame  # one row per subject
    videos: pd.DataFrame  # one row per video file
    issues: dict[str, list[str]] = field(default_factory=dict)  # section title -> lines
    notes: dict[str, Any] = field(default_factory=dict)  # counts for the summary
    output_dir: Path | None = None

    @property
    def has_issues(self) -> bool:
        """True when anything was put in the report as needing attention."""
        return any(self.issues.values())


# ---------------------------------------------------------------------------
# 1. Reading the workbook
# ---------------------------------------------------------------------------


def normalize_header(header: Any) -> str:
    """Make a header comparable: lowercase, no ':' and no line breaks, single spaces.

    Example: 'NTT \\nTime (min):' -> 'ntt time (min)'.
    """
    text = str(header).replace("\n", " ").replace(":", " ").lower()
    return re.sub(r"\s+", " ", text).strip()


def read_workbook(path: Path, sheet: str | int = 0) -> tuple[pd.DataFrame, list[str]]:
    """Read the trial sheet and rename its columns to the stable names in COLUMN_MAP.

    Every cell is read as-is (``dtype=object``, and pandas' automatic "N/A -> empty"
    conversion switched off) so that values such as '0042', '-', 'N/A' or '10*'
    survive untouched until the cleaning step decides what they mean.

    Returns the table and the list of headers that were not recognized (kept, but reported).
    """
    with warnings.catch_warnings():
        # openpyxl warns that Excel drop-down lists ("data validation") are not loaded; harmless here.
        warnings.filterwarnings("ignore", message="Data Validation extension", category=UserWarning)
        raw = pd.read_excel(path, sheet_name=sheet, dtype=object, keep_default_na=False)

    renamed: dict[str, str] = {}
    unknown: list[str] = []
    for header in raw.columns:
        key = normalize_header(header)
        if key in COLUMN_MAP:
            renamed[header] = COLUMN_MAP[key]
        elif key.startswith("unnamed"):
            continue  # pandas' name for an empty header cell; dropped below
        else:
            # Unknown column: keep it under a snake_case name so no information is lost.
            renamed[header] = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
            unknown.append(str(header).strip())

    table = raw[list(renamed)].rename(columns=renamed)

    missing = [name for name in REQUIRED_COLUMNS if name not in table.columns]
    if missing:
        raise CatalogError(f"{path.name}: required column(s) not found: {', '.join(missing)}")
    return table, unknown


# ---------------------------------------------------------------------------
# 2. Cleaning (scope §5 rules), one small function per rule
# ---------------------------------------------------------------------------


def _is_blank(value: Any) -> bool:
    """True for empty cells: None/NaN or text made only of spaces."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return bool(pd.isna(value))


def _to_number(value: Any) -> float:
    """Convert a cell to a float; anything that is not a plain number becomes NaN."""
    if _is_blank(value) or isinstance(value, bool):
        return float("nan")
    if isinstance(value, numbers.Number):
        return float(value)
    try:
        return float(str(value).strip())  # numbers stored as text, e.g. "3.5"
    except ValueError:
        return float("nan")  # "N/A", "10*", "-", notes, ...


def _parse_date(value: Any) -> str | None:
    """YYMMDD number (240109) or a real Excel date -> 'YYYY-MM-DD'; anything else -> None."""
    if isinstance(value, date):  # the cell may be formatted as a date in Excel
        return value.strftime("%Y-%m-%d")
    number = _to_number(value)
    if pd.isna(number) or number % 1 != 0:
        return None
    parsed = pd.to_datetime(f"{int(number):06d}", format="%y%m%d", errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def drop_template_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rule: rows with a blank compound are unused template rows and are removed."""
    return table[~table["compound"].map(_is_blank)].copy()


def parse_subject(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add `subject_num` (int) and a zero-padded `subject_id` (text, e.g. '0042').

    Rows whose subject cell is not a whole number are removed and reported, because
    they cannot be matched to a video.
    """
    numbers = table["subject_id"].map(_to_number)
    bad = numbers.isna() | (numbers % 1 != 0)
    problems = [
        f"Excel row {row}: subject {value!r} is not a whole number (row skipped)"
        for row, value in zip(table.loc[bad, "excel_row"], table.loc[bad, "subject_id"])
    ]
    table = table[~bad].copy()
    table["subject_num"] = numbers[~bad].astype(int)
    table["subject_id"] = table["subject_num"].map(lambda n: f"{n:04d}")  # same style as file names
    return table, problems


def parse_sex(table: pd.DataFrame) -> list[str]:
    """Normalize sex to 'M'/'F' (in place); report any other value."""
    table["sex"] = table["sex"].map(lambda v: "" if _is_blank(v) else str(v).strip().upper())
    bad = ~table["sex"].isin(["M", "F"])
    return [
        f"Excel row {row}: sex {value!r} is not M or F"
        for row, value in zip(table.loc[bad, "excel_row"], table.loc[bad, "sex"])
    ]


def parse_movement(table: pd.DataFrame) -> list[str]:
    """Rules for the 8 movement columns (in place):

    * '-' in a top/bottom column -> 0 (the fish spent no time in that zone);
    * numbers stay numbers;
    * 'N/A', blanks, and any other text -> missing (NaN).

    Unexpected text (anything except '-' in a zone column or 'N/A') is reported so
    that a new kind of note in the workbook does not silently become "missing".
    """
    problems: list[str] = []
    for column in MOVEMENT_COLUMNS:
        cleaned = []
        for row, value in zip(table["excel_row"], table[column]):
            text = value.strip() if isinstance(value, str) else None
            if text == "-" and column in ZONE_COLUMNS:
                cleaned.append(0.0)  # "-" = zero time/distance in this half of the tank
                continue
            number = _to_number(value)
            if text and pd.isna(number) and text.upper() != "N/A":
                problems.append(f"Excel row {row}: {column} = {value!r} (treated as missing)")
            cleaned.append(number)
        table[column] = cleaned
    return problems


def mark_untracked(table: pd.DataFrame) -> None:
    """Rule: if ALL 8 movement values are missing, the NTT trial was not tracked.

    These values stay missing (masked out of training later), they are never imputed.
    """
    table["ntt_tracked"] = ~table[MOVEMENT_COLUMNS].isna().all(axis=1)


def parse_other_columns(table: pd.DataFrame) -> None:
    """Convert the remaining columns (in place).

    * UV exposure time: the raw text is kept in `uv_min_raw`; values such as '10*'
      become missing, as decided with the research team.
    * age / exposure / NTT / UV times: numbers, text -> missing.
    * exp_date: YYMMDD number (e.g. 240109) -> ISO date text (2024-01-09).
    * concentration: raw text kept; `concentration_mm` is filled only when the cell
      is a single number (vehicle and combination doses stay text-only).
    * compound: surrounding spaces removed.
    """
    if "uv_min" in table.columns:
        table["uv_min_raw"] = table["uv_min"].map(lambda v: None if _is_blank(v) else str(v).strip())
    for column in OTHER_NUMERIC_COLUMNS:
        if column in table.columns:
            table[column] = table[column].map(_to_number)

    if "exp_date" in table.columns:
        table["exp_date"] = table["exp_date"].map(_parse_date)

    if "concentration_raw" in table.columns:
        table["concentration_mm"] = table["concentration_raw"].map(_to_number)
        table["concentration_raw"] = table["concentration_raw"].map(
            lambda v: None if _is_blank(v) else str(v).strip()
        )

    table["compound"] = table["compound"].map(lambda v: str(v).strip())


def clean_trials(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Apply every cleaning rule in order and collect what needs attention.

    Returns the cleaned table (one row per workbook row) and a dict of issues.
    """
    table = table.copy()
    # Excel row number (header is row 1) so every issue can be found in the workbook.
    table.insert(0, "excel_row", table.index + 2)

    table = drop_template_rows(table)
    table = table.drop(columns=[c for c in DROPPED_COLUMNS if c in table.columns])
    table, subject_problems = parse_subject(table)
    sex_problems = parse_sex(table)
    movement_problems = parse_movement(table)
    mark_untracked(table)
    parse_other_columns(table)

    issues = {
        "Unreadable subject numbers": subject_problems,
        "Unexpected sex values": sex_problems,
        "Unexpected text in movement columns": movement_problems,
    }
    return table.reset_index(drop=True), issues


# ---------------------------------------------------------------------------
# 3. Split recordings: several workbook rows with the same subject number
# ---------------------------------------------------------------------------


def merge_split_recordings(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Join rows that share a subject number into one subject.

    A repeated subject number means one fish whose recording was cut into parts
    (a, b, ...). The subject keeps the values of its first row; `n_workbook_rows`
    records how many parts are expected, and `excel_rows` lists where they came from.
    If the rows disagree on any value, the first row wins and the difference is reported.

    Returns the merged table, a list describing each split recording (for information),
    and a list of conflicts (these need attention).
    """
    groups = table.groupby("subject_num", sort=True)
    # First row of each subject; drop_duplicates keeps the column types intact.
    merged = table.drop_duplicates("subject_num", keep="first").sort_values("subject_num")
    merged = merged.set_index("subject_num", drop=False)
    merged["n_workbook_rows"] = groups.size()
    merged["excel_rows"] = groups["excel_row"].apply(lambda rows: ";".join(str(r) for r in rows))

    split_notes: list[str] = []
    conflicts: list[str] = []
    for subject_num, group in groups:
        if len(group) == 1:
            continue
        first = merged.loc[subject_num]
        split_notes.append(
            f"Subject {first['subject_id']}: {len(group)} rows (Excel rows {first['excel_rows']}) "
            f"-> joined as one subject with {len(group)} parts"
        )
        # Compare every column across the parts, ignoring the row-number bookkeeping.
        for column in group.columns.drop("excel_row"):
            distinct = {str(v) for v in group[column] if not _is_blank(v)}
            if len(distinct) > 1:
                conflicts.append(
                    f"Subject {first['subject_id']}: {column} differs between parts "
                    f"({', '.join(sorted(distinct))}); kept {first[column]!r}"
                )

    merged = merged.drop(columns=["excel_row"]).reset_index(drop=True)
    return merged, split_notes, conflicts


# ---------------------------------------------------------------------------
# 4. Videos: scan the folder and parse file names
# ---------------------------------------------------------------------------


def scan_videos(video_dir: Path, pattern: str, extensions: Iterable[str]) -> pd.DataFrame:
    """List every video file under `video_dir` (sub-folders included).

    Each file name (without extension) is searched with `pattern`; the result has
    one row per file with the parsed subject number and part letter ('' if none).
    Files whose name does not fit the pattern get `recognized=False`.
    """
    regex = re.compile(pattern, re.IGNORECASE)
    wanted = {ext.lower() for ext in extensions}
    records = []
    for path in sorted(video_dir.rglob("*")):
        # Skip folders, other file types, and hidden files such as macOS "._F_0042.mp4" copies.
        if not path.is_file() or path.suffix.lower() not in wanted or path.name.startswith("."):
            continue
        match = regex.search(path.stem)
        records.append(
            {
                "path": str(path),
                "file_name": path.name,
                "recognized": match is not None,
                "subject_num": int(match["subject"]) if match else None,
                "part": (match.groupdict().get("part") or "").lower() if match else None,
            }
        )
    columns = ["path", "file_name", "recognized", "subject_num", "part"]
    videos = pd.DataFrame(records, columns=columns)
    videos["subject_num"] = videos["subject_num"].astype("Int64")  # integer that allows "missing"
    return videos


# ---------------------------------------------------------------------------
# 5. Matching videos to subjects
# ---------------------------------------------------------------------------


def match_videos(trials: pd.DataFrame, videos: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Attach the video file(s) of each subject and give every subject a `video_status`.

    Matching key = subject number, which is unique across the whole workbook.
    Single-part subjects need exactly one file. Split subjects need one file per
    part, each with a different part letter; files are listed in part order (a, b, ...).
    """
    trials = trials.copy()
    recognized = videos[videos["recognized"]]
    by_subject = {num: group for num, group in recognized.groupby("subject_num")}

    statuses, paths, counts = [], [], []
    problems: list[str] = []
    for _, trial in trials.iterrows():
        found = by_subject.get(trial["subject_num"])
        label = f"Subject {trial['subject_id']} ({trial['sex']})"
        if found is None:
            statuses.append(STATUS_MISSING)
            paths.append("")
            counts.append(0)
            continue

        counts.append(len(found))
        expected_parts = int(trial["n_workbook_rows"])
        if expected_parts == 1 and len(found) > 1:
            status = STATUS_DUPLICATE
            problems.append(f"{label}: {len(found)} files for one recording: {', '.join(found['file_name'])}")
        elif expected_parts > 1 and (len(found) != expected_parts or found["part"].eq("").any()
                                     or found["part"].nunique() != len(found)):
            status = STATUS_PART_MISMATCH
            problems.append(
                f"{label}: expected {expected_parts} part files (a, b, ...), found: {', '.join(found['file_name'])}"
            )
        else:
            status = STATUS_MATCHED

        statuses.append(status)
        # Only fully matched subjects get paths, so later steps never use a doubtful file.
        ordered = found.sort_values("part")["path"] if status == STATUS_MATCHED else []
        paths.append(";".join(ordered))

    trials["video_status"] = statuses
    trials["video_paths"] = paths  # ';'-separated, in part order
    trials["n_videos_found"] = counts

    known = set(trials["subject_num"])
    orphans = [
        f"{row.file_name}: no workbook row for subject {row.subject_num}"
        for row in recognized.itertuples()
        if row.subject_num not in known
    ]
    unrecognized = [f"{name}: name does not match the video name pattern"
                    for name in videos.loc[~videos["recognized"], "file_name"]]
    return trials, {
        "Video problems": problems,
        "Videos without a workbook row": orphans,
        "Unrecognized video file names": unrecognized,
    }


# ---------------------------------------------------------------------------
# 6. Putting it together
# ---------------------------------------------------------------------------


def build_catalog(settings: Settings) -> CatalogResult:
    """Run every step, write the outputs, and return the result.

    Needs FISH_DB_PATH. FISH_VIDEO_DIR is optional: without it the workbook is still
    cleaned and checked, and every subject gets video_status = 'not_checked'.
    """
    params = settings.params.get("catalog", {})
    db_path = settings.require("db_path")

    table, unknown_columns = read_workbook(db_path, params.get("sheet", 0))
    n_rows_read = len(table)
    cleaned, issues = clean_trials(table)
    trials, split_notes, conflicts = merge_split_recordings(cleaned)
    issues["Conflicting values in split recordings"] = conflicts
    issues["Unknown workbook columns (kept as-is)"] = unknown_columns

    if settings.paths.video_dir is not None:
        video_dir = settings.require("video_dir")
        videos = scan_videos(video_dir, params["video_name_pattern"], params["video_extensions"])
        trials, video_issues = match_videos(trials, videos)
        issues.update(video_issues)
        issues["Subjects without a video"] = list(
            trials.loc[trials["video_status"] == STATUS_MISSING, "subject_id"]
        )
    else:
        log.warning("FISH_VIDEO_DIR is not set: videos are not checked.")
        videos = pd.DataFrame(columns=["path", "file_name", "recognized", "subject_num", "part"])
        trials["video_status"] = STATUS_NOT_CHECKED
        trials["video_paths"] = ""
        trials["n_videos_found"] = 0

    notes = {
        "rows_read": n_rows_read,
        "trial_rows": len(cleaned),
        "subjects": len(trials),
        "untracked_rows": int((~cleaned["ntt_tracked"]).sum()),
        "untracked_subjects": int((~trials["ntt_tracked"]).sum()),
        "video_files": len(videos),
        "status_counts": trials["video_status"].value_counts().to_dict(),
        "split_list": split_notes,
        "untracked_list": [
            f"Subject {s} (Excel rows {r})"
            for s, r in zip(trials.loc[~trials["ntt_tracked"], "subject_id"],
                            trials.loc[~trials["ntt_tracked"], "excel_rows"])
        ],
    }

    output_dir = settings.paths.output_dir / "catalog"
    result = CatalogResult(trials=trials, videos=videos, issues=issues, notes=notes, output_dir=output_dir)
    write_outputs(result)
    return result


def write_outputs(result: CatalogResult) -> None:
    """Write trials.csv, videos.csv and validation_report.md to `result.output_dir`."""
    assert result.output_dir is not None
    result.output_dir.mkdir(parents=True, exist_ok=True)
    result.trials.to_csv(result.output_dir / "trials.csv", index=False)
    result.videos.to_csv(result.output_dir / "videos.csv", index=False)
    (result.output_dir / "validation_report.md").write_text(render_report(result), encoding="utf-8")


def render_report(result: CatalogResult) -> str:
    """Build the human-readable validation report (Markdown)."""
    n = result.notes
    lines = [
        "# Catalog validation report",
        "",
        "## Summary",
        "",
        f"- Workbook rows read: {n['rows_read']}",
        f"- Trial rows (non-blank compound): {n['trial_rows']}",
        f"- Subjects after joining split recordings: {n['subjects']}",
        f"- NTT untracked (all 8 movement values missing): {n['untracked_rows']} rows / "
        f"{n['untracked_subjects']} subjects",
        f"- Video files found: {n['video_files']}",
        "- Video status: " + ", ".join(f"{k} {v}" for k, v in sorted(n["status_counts"].items())),
        "",
    ]
    for title, entries in result.issues.items():
        lines += [f"## {title} ({len(entries)})", ""]
        lines += [f"- {entry}" for entry in entries] if entries else ["- none"]
        lines.append("")
    # Expected situations, listed for reference only (they are not counted as issues).
    for title, entries in (
        ("Split recordings joined into one subject, for reference", n["split_list"]),
        ("NTT untracked subjects (scope §5), for reference", n["untracked_list"]),
    ):
        lines += [f"## {title} ({len(entries)})", ""]
        lines += [f"- {entry}" for entry in entries] or ["- none"]
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 7. Reading the catalog back (used by every later step)
# ---------------------------------------------------------------------------


def trials_path(settings: Settings) -> Path:
    """Where `validate` writes trials.csv."""
    return settings.paths.output_dir / "catalog" / "trials.csv"


def load_trials(settings: Settings) -> pd.DataFrame:
    """Read trials.csv written by `validate`, with `subject_id` kept as text ('0042').

    Raises ConfigError when the catalog has not been built yet, because every later
    step depends on it.
    """
    path = trials_path(settings)
    if not path.is_file():
        raise ConfigError(f"{path} not found: run `python -m fishbehavior validate` first")
    # dtype=str for the ID keeps its leading zeros; empty video_paths would otherwise be NaN.
    trials = pd.read_csv(path, dtype={"subject_id": str, "video_paths": str})
    trials["video_paths"] = trials["video_paths"].fillna("")
    return trials


def parse_subject_number(text: str) -> int:
    """'42', '0042', 'F_0042' and 'M_0042a' all mean subject 42 (the first run of digits)."""
    match = re.search(r"\d+", str(text))
    if match is None:
        raise ConfigError(f"--subjects: {text!r} does not contain a subject number")
    return int(match.group())


def select_subjects(trials: pd.DataFrame, wanted: Iterable[str] | None) -> pd.DataFrame:
    """Keep only the requested subjects (for --subjects); None or empty keeps all.

    Items may also be comma-separated ('42,43'). Unknown subjects raise ConfigError so
    a typo is not silently ignored.
    """
    if not wanted:
        return trials
    items = [part.strip() for item in wanted for part in str(item).split(",") if part.strip()]
    numbers = {parse_subject_number(item) for item in items}
    known = set(trials["subject_id"].astype(int))
    unknown = sorted(numbers - known)
    if unknown:
        raise ConfigError(
            "subject(s) not in trials.csv: " + ", ".join(f"{n:04d}" for n in unknown)
        )
    return trials[trials["subject_id"].astype(int).isin(numbers)]
