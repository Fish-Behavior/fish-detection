"""'Why this label' for one second of a video (review app): rules in order, the numbers behind them, the votes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prepds.consolidate import consolidate_labels
from prepds.explain import VideoExplainer
from prepds.labeling import classify_video
from prepds.models import BehaviorState as B, Track

FPS = 30.0
THRESHOLDS = {"freeze_speed_floor_px_per_s": 6.0, "min_freeze_bout_s": 1.0, "erratic_speed_threshold_px_per_s": 40.0,
              "surface_breach_depth_threshold_px": 20.0, "min_dead_bout_s": 60.0, "speed_lag_s": 1.0,
              "listing_orientation_deviation_deg": None}


def _tracks(n=300, speed=0.0, detected=True, y=150.0):
    return [Track(i, i / FPS, 10.0 + speed * i / FPS, y, 90.0 if detected else None, y if detected else None, detected) for i in range(n)]


def _frames(tracks, stored=None):
    states = consolidate_labels([s.state for s in classify_video(tracks, **THRESHOLDS)], [t.t_sec for t in tracks])
    return pd.DataFrame({
        "frame_idx": [t.frame_idx for t in tracks], "t_sec": [t.t_sec for t in tracks], "x": [t.x for t in tracks],
        "y": [t.y for t in tracks], "orientation_deg": [np.nan if t.orientation_deg is None else t.orientation_deg for t in tracks],
        "depth_from_surface": [np.nan if t.y_from_frame_top is None else t.y_from_frame_top for t in tracks],
        "detected": [t.detected for t in tracks], "state": [s.value for s in (stored or states)],
        "source": ["auto"] * len(tracks),
    })


def _rule(explanation, state):
    return next(r for r in explanation["rules"] if r["state"] == state)


def test_a_still_fish_is_explained_as_freezing_with_its_speed_and_threshold() -> None:
    explanation = VideoExplainer(_frames(_tracks(speed=0.0)), THRESHOLDS).explain(5)
    assert explanation["stored_state"] == "Freezing/Drift" and explanation["auto_state"] == "Freezing/Drift"
    freezing = _rule(explanation, "Freezing/Drift")
    assert freezing["wins"] and "0.0" in freezing["detail"] and "6.0" in freezing["detail"]
    assert not _rule(explanation, "Erratic Movement")["wins"]
    assert [r["state"] for r in explanation["rules"]][0] == "Undetermined"
    assert explanation["votes"]["Freezing/Drift"] == explanation["n_frames"] == 30


def test_a_fast_fish_is_explained_as_erratic() -> None:
    explanation = VideoExplainer(_frames(_tracks(speed=80.0)), THRESHOLDS).explain(5)
    assert explanation["stored_state"] == "Erratic Movement"
    assert _rule(explanation, "Erratic Movement")["wins"] and explanation["speed_median_px_per_s"] == pytest.approx(80.0, rel=0.05)


def test_a_second_without_the_fish_is_undetermined_and_says_why() -> None:
    tracks = _tracks(n=300)[:150] + _tracks(n=300, detected=False)[150:]
    explanation = VideoExplainer(_frames(tracks), THRESHOLDS).explain(8)
    assert explanation["stored_state"] == "Undetermined" and explanation["n_detected"] == 0
    assert _rule(explanation, "Undetermined")["wins"] and "0 of 30" in _rule(explanation, "Undetermined")["detail"]


def test_a_reviewer_edit_is_reported_against_the_automatic_label() -> None:
    frames = _frames(_tracks(speed=0.0))
    frames.loc[(frames.t_sec >= 5) & (frames.t_sec < 6), ["state", "source"]] = ["Erratic Movement", "manual"]
    explanation = VideoExplainer(frames, THRESHOLDS).explain(5)
    assert explanation["stored_state"] == "Erratic Movement" and explanation["auto_state"] == "Freezing/Drift"
    assert explanation["stored_source"] == "manual" and explanation["matches_stored"] is False


def test_without_thresholds_the_measurements_are_shown_but_no_rules() -> None:
    explanation = VideoExplainer(_frames(_tracks(speed=80.0)), None).explain(5)
    assert explanation["rules"] == [] and explanation["auto_state"] is None and explanation["thresholds_available"] is False
    assert explanation["speed_median_px_per_s"] == pytest.approx(80.0, rel=0.05)


def test_a_second_outside_the_video_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        VideoExplainer(_frames(_tracks()), THRESHOLDS).explain(999)


def test_listing_is_reported_as_a_manual_label_when_the_rule_is_off() -> None:
    explanation = VideoExplainer(_frames(_tracks()), THRESHOLDS).explain(5)
    assert "manual" in _rule(explanation, "Listing/LORR")["detail"].lower()
