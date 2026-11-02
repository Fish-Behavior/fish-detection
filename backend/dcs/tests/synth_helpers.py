"""Helpers shared by the synthetic-data tests (test_dcs_synthetic*.py); not a test module itself.

Imported as `tests.dcs.synth_helpers` (the repository root is on the test path via pyproject's
`pythonpath`), which works in pytest's default and importlib import modes alike.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pandas as pd

from dcs.schema import FRAMES_FILE, LISTING_LORR
from dcs.synthetic import SynthResult, make_gold_dataset
from dcs.synthetic_config import SynthConfig

# Small enough to keep tests/dcs fast: 20 fish of 20 s at 10 fps.
SMALL = SynthConfig(
    seed=0, n_compounds=2, doses_per_compound=2, fish_per_dose_date=2, n_dates=4, vehicle_per_date=1, duration_s=20.0
)
KINEMATIC = ("x", "y", "velocity", "acceleration", "angular_velocity", "meander")
LISTING = LISTING_LORR
PATH_COLUMNS = ["frames_path", "segments_path", "strip_path", "manifest_path"]  # differ between output folders


def make(tmp_path: Path, **changes: Any) -> SynthResult:
    """SMALL with some fields changed, written to a new folder under tmp_path."""
    return make_gold_dataset(tmp_path / "synth", dataclasses.replace(SMALL, **changes))


def index(result: SynthResult) -> pd.DataFrame:
    return pd.read_parquet(result.index_path)


def frames(result: SynthResult, video_id: str) -> pd.DataFrame:
    return pd.read_parquet(result.accepted_dir / video_id / FRAMES_FILE)


def read_json(result: SynthResult, video_id: str, name: str) -> dict[str, Any]:
    return json.loads((result.accepted_dir / video_id / name).read_text(encoding="utf-8"))
