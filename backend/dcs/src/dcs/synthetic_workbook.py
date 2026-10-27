"""Synthetic trial workbook (plan U2): the real header names, one row per fish, placeholder values.

As in the real workbook, a plain dose is a number cell and a combination dose
is text. The 8 NTT columns carry a weak copy of the fish's compound speed
effect and are all missing for a configurable share of fish (EC-2).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dcs import schema

NTT_SESSION_S = 360.0  # the novel tank test lasts 6 minutes
NTT_TIME_MIN = 6.0
UV_EXPOSURE_MIN = 10.0
BASE_TDM = 1500.0
BASE_NTT_SPEED = 5.0
NTT_NOISE = 0.1  # sigma of the lognormal noise on TDM and speed
TOP_SHARE_MEAN, TOP_SHARE_SD = 0.4, 0.08  # share of the distance travelled in the top half
SHARE_BOUNDS = (0.05, 0.95)
HALF_SPEED_RANGE = (0.8, 1.2)  # top/bottom-half speed relative to the full-arena speed
TIME_TOP_MEAN, TIME_TOP_SD = 0.35, 0.1  # share of the session spent in the top half
H2O_RANGE = (0.5, 1.5)
BRAIN_TISSUE = "N"
H2O_BEFORE, H2O_AFTER, BRAIN_TISSUE_HEADER, BODY_TISSUE_HEADER = schema.WB_OTHER_HEADERS


@dataclass(frozen=True)
class WorkbookFish:
    """What the workbook says about one fish, before the random NTT values are added."""

    date: dt.date
    subject: int
    sex: str
    strain: str
    age: float
    compound_label: str
    dose_label: str
    agent_exposure_min: float
    speed_multiplier: float


def workbook_record(fish: WorkbookFish, ntt_missing_fraction: float, rng: np.random.Generator) -> dict[str, Any]:
    """One workbook row keyed by the real headers; `rng` is the fish's own NTT stream."""
    missing = rng.random() < ntt_missing_fraction
    ntt = _ntt_values(fish.speed_multiplier, rng)
    return {
        schema.WB_DATE: float(fish.date.strftime("%y%m%d")),  # YYMMDD number, as in the real workbook
        schema.WB_SUBJECT: fish.subject,
        schema.WB_STRAIN: fish.strain,
        schema.WB_SEX: fish.sex,
        schema.WB_AGE: fish.age,
        schema.WB_COMPOUND: fish.compound_label,
        schema.WB_DOSE: dose_cell(fish.dose_label),
        schema.WB_AGENT_EXPOSURE: fish.agent_exposure_min,
        schema.WB_NTT_TIME: NTT_TIME_MIN,
        schema.WB_UV_TIME: UV_EXPOSURE_MIN,
        **{header: (np.nan if missing else value) for header, value in ntt.items()},
        H2O_BEFORE: round(float(rng.uniform(*H2O_RANGE)), 3),
        H2O_AFTER: round(float(rng.uniform(*H2O_RANGE)), 3),
        BRAIN_TISSUE_HEADER: BRAIN_TISSUE,
        BODY_TISSUE_HEADER: np.nan,  # always empty in the real workbook
    }


def dose_cell(label: str) -> float | str:
    """A plain dose is stored as a number (prepds turns 1.0 back into "1"); a combination stays text."""
    try:
        return float(label)
    except ValueError:
        return label


def write_workbook(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write the rows to the workbook's single sheet, with the real headers in the real order."""
    for record in records:
        if set(record) != set(schema.WORKBOOK_HEADERS):  # a misspelt key would silently become an empty column
            raise ValueError(f"workbook record keys differ from the headers: {sorted(set(record) ^ set(schema.WORKBOOK_HEADERS))}")
    pd.DataFrame(list(records), columns=list(schema.WORKBOOK_HEADERS)).to_excel(
        path, sheet_name=schema.WORKBOOK_SHEET, index=False
    )


def _ntt_values(speed_multiplier: float, rng: np.random.Generator) -> dict[str, float]:
    """The 8 NTT values; drawn even when they end up missing, so the stream stays stable."""
    full_tdm = BASE_TDM * speed_multiplier * rng.lognormal(0.0, NTT_NOISE)
    top_share = float(np.clip(rng.normal(TOP_SHARE_MEAN, TOP_SHARE_SD), *SHARE_BOUNDS))
    full_speed = BASE_NTT_SPEED * speed_multiplier * rng.lognormal(0.0, NTT_NOISE)
    time_top = NTT_SESSION_S * float(np.clip(rng.normal(TIME_TOP_MEAN, TIME_TOP_SD), 0.0, 1.0))
    values = (
        full_tdm,
        full_tdm * top_share,
        full_tdm * (1 - top_share),
        full_speed,
        full_speed * rng.uniform(*HALF_SPEED_RANGE),
        full_speed * rng.uniform(*HALF_SPEED_RANGE),
        time_top,
        NTT_SESSION_S - time_top,
    )
    return {header: round(float(value), 3) for header, value in zip(schema.NTT_HEADERS, values)}
