"""Frozen calibration profiles (T046).

A profile is an immutable YAML file (`config/calibration_profiles/cal-<date>.yaml`) holding the
labeling thresholds found by `calibrate.search`, the agreement metrics they were validated with, and
provenance. It has the same `labeling:` shape as `config/default_thresholds.yaml`, so it can be used
as a `PDS_CONFIG` override or copied into the defaults. Profiles are never edited: a recalibration
writes a new file (NFR-002 - a run is reproducible from its profile version).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from prepds.config import ConfigError
from prepds.features import DEFAULT_SPEED_LAG_S

REQUIRED_THRESHOLDS = (
    "freeze_speed_floor_px_per_s",
    "min_freeze_bout_s",
    "erratic_speed_threshold_px_per_s",
    "surface_breach_depth_threshold_px",
    "min_dead_bout_s",
)


def write_calibration_profile(
    path: Path,
    *,
    version: str,
    created_at: str,
    thresholds: Mapping[str, float],
    metrics: Mapping[str, Any],
    notes: str = "",
    speed_lag_s: float = DEFAULT_SPEED_LAG_S,
) -> Path:
    """Write a new, complete profile. Refuses to overwrite: a frozen profile is immutable.

    Listing/LORR is written as disabled (`listing_orientation_deviation_deg: null`) - it is labeled
    manually in review, not calibrated.
    """
    missing = [key for key in REQUIRED_THRESHOLDS if thresholds.get(key) is None]
    if missing:
        raise ValueError(f"profile is missing thresholds: {', '.join(missing)}")
    invalid = [key for key in REQUIRED_THRESHOLDS if not _is_positive_finite(thresholds[key])]
    if not invalid and thresholds["erratic_speed_threshold_px_per_s"] <= thresholds["freeze_speed_floor_px_per_s"]:
        raise ValueError("erratic_speed_threshold_px_per_s must exceed freeze_speed_floor_px_per_s")
    if invalid or not _is_positive_finite(speed_lag_s):
        raise ValueError(f"thresholds must be finite and positive: {', '.join(invalid or ['speed_lag_s'])}")
    document = {
        "calibration_profile": {
            "version": version,
            "created_at": created_at,
            "metrics": dict(metrics),
            "notes": notes,
        },
        "labeling": {
            **{key: float(thresholds[key]) for key in REQUIRED_THRESHOLDS},
            "speed_lag_s": float(speed_lag_s),
            "listing_orientation_deviation_deg": None,
        },
    }
    text = yaml.safe_dump(document, sort_keys=False)  # serialise first: a failure must not leave a partial file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:  # "x": raise FileExistsError instead of overwriting
        handle.write(text)
    return path


def labeling_thresholds(params: Mapping[str, Any]) -> dict[str, float | None]:
    """The merged config's `labeling:` block as `classify_video` keyword arguments.

    Raises `ConfigError` while any required threshold is still `null` (the packaged placeholder), so
    an uncalibrated run fails loudly instead of labeling with made-up numbers.
    """
    section = params.get("labeling") or {}
    missing = [key for key in REQUIRED_THRESHOLDS if section.get(key) is None]
    if missing:
        raise ConfigError(
            f"labeling thresholds are uncalibrated (null): {', '.join(missing)}. "
            "Pass a frozen calibration profile via PDS_CONFIG / --config."
        )
    kwargs: dict[str, float | None] = {key: float(section[key]) for key in REQUIRED_THRESHOLDS}
    kwargs["speed_lag_s"] = float(section.get("speed_lag_s", DEFAULT_SPEED_LAG_S))
    listing = section.get("listing_orientation_deviation_deg")
    kwargs["listing_orientation_deviation_deg"] = None if listing is None else float(listing)
    return kwargs


def _is_positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0
