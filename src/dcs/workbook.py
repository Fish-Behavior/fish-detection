"""The 8 NTT columns from the trial workbook, one row per gold fish (plan U4, T1.9, D-004).

NTT is in neither the index nor the catalog, so it is read from the workbook itself
(`DCS_DB_PATH`) and joined on the video id `<Sex>_<Subject:04d>`. Rows are trial rows when they
name a compound, as in prepds. Exact duplicate rows collapse to one; two different rows for one
fish, or a row whose compound (spelled by the training set's rule) or date disagrees with the gold
set, stop the run (EC-28). A fish with no row, or with a blank NTT cell, gets `has_ntt = 0` (EC-2).
A `-` cell is measured, not missing (D-037): the fish never entered that half, so its distance is 0
and its speed NaN while `has_ntt` stays 1. Imputation is U9's job.
"""

from __future__ import annotations

import datetime as dt
import math
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from dcs import schema
from dcs.config import ENV_DB_PATH, ConfigError
from dcs.gold_checks import GoldDataError, one_line
from dcs.gold_rules import blank, compound_label, ids

NTT_COLUMNS = (
    "tdm_full",
    "tdm_top",
    "tdm_bottom",
    "velocity_full",
    "velocity_top",
    "velocity_bottom",
    "time_top_s",
    "time_bottom_s",
)
HAS_NTT = "has_ntt"
# The real workbook writes "-" in a half's cells when the fish never entered that half (its time
# there is 0 and the full-arena distance equals the other half's): distance 0, speed undefined.
NEVER_ENTERED = "-"
NEVER_ENTERED_VALUES = {
    schema.NTT_HEADERS[1]: 0.0,
    schema.NTT_HEADERS[2]: 0.0,
    schema.NTT_HEADERS[4]: math.nan,
    schema.NTT_HEADERS[5]: math.nan,
}


def read_ntt(workbook: Path | None, videos: pd.DataFrame) -> pd.DataFrame:
    """video_id, NTT_COLUMNS and has_ntt for every row of `videos` (a GoldSet table), in its order.

    `workbook` None (DCS_DB_PATH unset) gives NaN NTT and has_ntt = 0 for every fish.
    """
    out = pd.DataFrame({"video_id": videos["video_id"].astype(str).to_numpy()})
    if workbook is None:
        ntt = pd.DataFrame(index=out["video_id"], columns=[*NTT_COLUMNS, HAS_NTT], dtype=float)
    else:
        rows = _trial_rows(Path(workbook))
        _check_against(rows, videos)
        ntt = rows.reindex(out["video_id"])[[*NTT_COLUMNS, HAS_NTT]]
    out[list(NTT_COLUMNS)] = ntt[list(NTT_COLUMNS)].to_numpy(dtype=float)
    out[HAS_NTT] = ntt[HAS_NTT].fillna(0).to_numpy(dtype=int)  # no workbook row -> 0
    return out


def _trial_rows(path: Path) -> pd.DataFrame:
    """One row per fish, indexed by video id, with compound, date and the NTT columns."""
    try:
        table = pd.read_excel(path, sheet_name=schema.WORKBOOK_SHEET)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ConfigError(
            f"Cannot read sheet {schema.WORKBOOK_SHEET} of the workbook {path} ({ENV_DB_PATH}): {one_line(error)[:200]}"
        ) from error
    table.columns = [_header(column) for column in table.columns]
    wanted = {_header(h): h for h in (schema.WB_SEX, schema.WB_SUBJECT, schema.WB_COMPOUND, schema.WB_DATE, *schema.NTT_HEADERS)}
    missing = [real for header, real in wanted.items() if header not in table.columns]
    if missing:
        raise GoldDataError(f"Workbook {path.name}: header(s) {', '.join(map(repr, missing))} missing")

    table = table[~table[_header(schema.WB_COMPOUND)].map(blank)].drop_duplicates()
    sex, subject = _header(schema.WB_SEX), _header(schema.WB_SUBJECT)
    for name in (sex, subject):
        blank_rows = int(table[name].map(blank).sum())
        if blank_rows:
            raise GoldDataError(f"Workbook {path.name}: blank {name!r} in {blank_rows} trial row(s)")
    keys = [schema.video_id(str(s).strip().upper(), _subject(n, path)) for s, n in zip(table[sex], table[subject])]
    cells = table[[_header(h) for h in schema.NTT_HEADERS]]
    rows = pd.DataFrame(
        {
            "compound": table[_header(schema.WB_COMPOUND)].to_numpy(),
            "date": [_yymmdd(value, path) for value in table[_header(schema.WB_DATE)]],
            **{name: _ntt_values(cells[_header(h)], h, path) for name, h in zip(NTT_COLUMNS, schema.NTT_HEADERS)},
            HAS_NTT: (~cells.map(blank)).all(axis=1).astype(int).to_numpy(),  # all eight measured (EC-2)
        },
        index=pd.Index(keys, name="video_id"),
    )
    repeated = rows.index[rows.index.duplicated()].unique()
    if len(repeated):
        raise GoldDataError(f"Workbook {path.name}: different rows for the same fish: {ids(repeated)} (EC-28)")
    return rows


def _check_against(rows: pd.DataFrame, videos: pd.DataFrame) -> None:
    """A gold fish's workbook row must name its compound and date (EC-28)."""
    for record in videos.to_dict("records"):
        video_id = str(record["video_id"])
        if video_id not in rows.index:
            continue
        row = rows.loc[video_id]
        if compound_label(row["compound"]) != compound_label(record["compound"]):  # same rule as the training set
            raise GoldDataError(
                f"{video_id}: workbook compound {row['compound']!r} disagrees with the gold set's {record['compound']!r} (EC-28)"
            )
        if row["date"] != _as_date(record["date"]):
            raise GoldDataError(f"{video_id}: workbook date {row['date']} disagrees with the gold set's {record['date']} (EC-28)")


def _header(value: Any) -> str:
    """Real headers carry stray spaces and embedded newlines; compare them with whitespace collapsed."""
    return " ".join(str(value).split())


def _ntt_values(cells: pd.Series, header: str, path: Path) -> list[float]:
    """Numbers stay numbers, blanks become NaN, "-" follows NEVER_ENTERED_VALUES; any other text stops the run."""
    values = []
    for cell in cells:
        if blank(cell):
            values.append(math.nan)
        elif isinstance(cell, str) and cell.strip() == NEVER_ENTERED and header in NEVER_ENTERED_VALUES:
            values.append(NEVER_ENTERED_VALUES[header])
        else:
            number = pd.to_numeric(cell, errors="coerce")
            if pd.isna(number):
                raise GoldDataError(f"Workbook {path.name}: {header!r} holds {cell!r}, not a number")
            values.append(float(number))
    return values


def _subject(value: Any, path: Path) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or float(number) != int(number):
        raise GoldDataError(f"Workbook {path.name}: subject {value!r} is not a whole number")
    return f"{int(number):04d}"


def _yymmdd(value: Any, path: Path) -> dt.date | None:
    """`Date of EXP:` is a YYMMDD number (21st century), as prepds reads it."""
    if blank(value):
        return None
    try:
        digits = f"{int(value):06d}"
        return dt.date(2000 + int(digits[:2]), int(digits[2:4]), int(digits[4:]))
    except (TypeError, ValueError) as error:
        raise GoldDataError(f"Workbook {path.name}: date {value!r} is not a YYMMDD number") from error


def _as_date(value: Any) -> dt.date | None:
    return None if blank(value) else pd.Timestamp(value).date()
