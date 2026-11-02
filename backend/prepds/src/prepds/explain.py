"""'Why this label' for one second of a processed video (review app).

Replays the labeling rules on the video's stored track: it recomputes the smoothed speed, classifies every frame
with the thresholds of the video's calibration profile, applies the same 1 s consolidation, and reports for the
chosen second which rules the frames fell under, the measured numbers next to their thresholds, and whether a
reviewer changed the automatic label. It reads only; it never changes a state.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from prepds.consolidate import DEFAULT_BIN_S, DEFAULT_MIN_DETERMINED_FRACTION, consolidate_labels
from prepds.labeling import classify_prepared_states, prepare_video
from prepds.models import BehaviorState as B, Track

_EPSILON = 1e-9


class VideoExplainer:
    """Built once per video (recomputes speeds and states over the whole track), then asked about any second."""

    def __init__(self, frames: pd.DataFrame, thresholds: Mapping[str, float | None] | None) -> None:
        self._frames = frames.reset_index(drop=True)
        self._t = self._frames["t_sec"].to_numpy(float)
        self._bins = np.floor(self._t / DEFAULT_BIN_S + _EPSILON).astype(np.int64)
        self._thresholds = None if thresholds is None else dict(thresholds)
        tracks = [_track(row) for row in self._frames.itertuples(index=False)]
        speed_lag = (self._thresholds or {}).get("speed_lag_s", 1.0)
        self._prepared = prepare_video(tracks, speed_lag_s=speed_lag)
        self._raw: list[B] | None = None
        self._auto: list[B] | None = None
        if self._thresholds is not None:
            args = {k: v for k, v in self._thresholds.items() if k != "speed_lag_s"}
            self._raw = classify_prepared_states(self._prepared, **args)
            self._auto = consolidate_labels(self._raw, self._t.tolist())

    def explain(self, second: int) -> dict[str, Any]:
        mask = self._bins == second
        if not mask.any():
            raise KeyError(second)
        idx = np.flatnonzero(mask)
        frames = self._frames.iloc[idx]
        stored_state = str(frames["state"].iloc[0])
        stored_source = str(frames["source"].iloc[0])
        detected = frames["detected"].to_numpy(bool)
        speeds = np.array(self._prepared.speeds)[idx]
        valid = np.array(self._prepared.speed_valid)[idx] & detected
        y_top = frames["depth_from_surface"].to_numpy(float)
        y_min = float(np.nanmin(y_top)) if np.isfinite(y_top).any() else None
        speed_median = float(np.median(speeds[valid])) if valid.any() else None
        result: dict[str, Any] = {
            "second": second, "start_s": float(second * DEFAULT_BIN_S), "end_s": float((second + 1) * DEFAULT_BIN_S),
            "n_frames": int(len(idx)), "n_detected": int(detected.sum()),
            "stored_state": stored_state, "stored_source": stored_source,
            "speed_median_px_per_s": speed_median, "y_min_px": y_min,
            "thresholds_available": self._thresholds is not None,
            "auto_state": None, "matches_stored": None, "votes": {}, "rules": [],
        }
        if self._raw is None or self._auto is None:
            return result
        auto_state = self._auto[idx[0]].value
        votes = Counter(self._raw[i].value for i in idx)
        result.update(auto_state=auto_state, matches_stored=auto_state == stored_state, votes=dict(votes))
        result["rules"] = self._rules(result, votes, auto_state)
        return result

    def _rules(self, r: Mapping[str, Any], votes: Counter, winner: str) -> list[dict[str, Any]]:
        t = self._thresholds or {}
        n, det = r["n_frames"], r["n_detected"]
        v, y = r["speed_median_px_per_s"], r["y_min_px"]
        speed = "no measurable speed" if v is None else f"median speed {v:.1f} px/s"
        floor, erratic = t["freeze_speed_floor_px_per_s"], t["erratic_speed_threshold_px_per_s"]
        listing = t.get("listing_orientation_deviation_deg")
        details = [
            (B.UNDETERMINED, f"{det} of {n} frames have the fish detected; the second is undetermined when fewer than "
                             f"{DEFAULT_MIN_DETERMINED_FRACTION:.0%} of its frames are determined"),
            (B.DEAD, f"still for at least {t['min_dead_bout_s']:.0f} s until the end of the video (Dead never reverts)"),
            (B.SURFACE_BREACH, ("fish not detected" if y is None else f"highest position y = {y:.0f} px from the top of the frame")
                               + f"; needs y <= {t['surface_breach_depth_threshold_px']:.0f} px"),
            (B.LISTING_LORR, "never set by the rules (Listing/LORR is a manual label; see the possible-Listing hints)"
                             if listing is None else f"orientation deviates at least {listing:.0f} deg"),
            (B.FREEZING_DRIFT, f"{speed}; freezing needs speed <= {floor:.1f} px/s for at least {t['min_freeze_bout_s']:g} s"),
            (B.ERRATIC_MOVEMENT, f"{speed}; erratic needs speed >= {erratic:.1f} px/s"),
            (B.CONTROLLED_SWIM, f"{speed}; between {floor:.1f} and {erratic:.1f} px/s (the swimming default)"),
        ]
        return [{"state": s.value, "frames": int(votes.get(s.value, 0)), "wins": s.value == winner, "detail": d} for s, d in details]


def _track(row: Any) -> Track:
    detected = bool(row.detected)
    orientation = None if not detected or pd.isna(row.orientation_deg) else float(row.orientation_deg)
    y_top = None if not detected or pd.isna(row.depth_from_surface) else float(row.depth_from_surface)
    return Track(int(row.frame_idx), float(row.t_sec), float(row.x), float(row.y), orientation, y_top, detected)
