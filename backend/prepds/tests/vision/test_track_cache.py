"""Calibration video selection and the on-disk Track cache - FR-008 support.

Tracking (~43s/video) is by far the most expensive pipeline step and does not
depend on any parameter the calibration search varies (advisor: tracking
parameters are deliberately left out of the search as a cost-driven deviation
from T045's "calibrate together" note), so tracks are computed once per
calibration video and cached; the search only re-runs features/labeling.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from prepds.calibration.track_cache import group_key, load_cached_tracks, save_cached_tracks, select_calibration_trials
from prepds.models import MatchStatus, Track, Trial


def _trial(compound, conc, subject, *, video=True) -> Trial:
    return Trial(
        subject_id=f"{subject:04d}",
        sex="F",
        strain="Casper",
        age=None,
        compound=compound,
        concentration_mM=conc,
        date=dt.date(2026, 1, 1),
        agent_exposure_min=20.0,
        video_path=Path(f"/videos/{compound}/F_{subject:04d}.mp4") if video else None,
        match_status=MatchStatus.MATCHED if video else MatchStatus.NO_VIDEO,
    )


def test_group_key_maps_concentration_mM_to_uM_and_normalizes_compound() -> None:
    assert group_key(_trial("Fentanyl", "0.03", 1)) == ("fentanyl", 30)
    assert group_key(_trial("Fentanyl", "0.1", 1)) == ("fentanyl", 100)
    assert group_key(_trial("DOB", "0.03", 1)) == ("dob", 30)
    assert group_key(_trial("Veh", "1% DMSO", 1)) == ("veh", 0)


def test_group_key_is_none_for_non_reference_compounds_and_combos() -> None:
    assert group_key(_trial("FD-2-45", "0.03", 1)) is None
    assert group_key(_trial("Fentanyl", "0.03 + 0.01", 1)) is None
    assert group_key(_trial("Fentanyl", "0.5", 1)) is None  # a concentration with no reference panel


def test_select_calibration_trials_is_deterministic_capped_and_skips_unmatched() -> None:
    trials = [_trial("Veh", "1% DMSO", n) for n in range(1, 21)]
    trials += [_trial("Fentanyl", "0.03", n) for n in range(30, 34)]
    trials += [_trial("Fentanyl", "0.03", 99, video=False)]
    trials += [_trial("FD-2-45", "0.03", 5)]

    selected = select_calibration_trials(trials, per_group=6)

    assert set(selected) == {("veh", 0), ("fentanyl", 30)}
    assert len(selected[("veh", 0)]) == 6
    assert len(selected[("fentanyl", 30)]) == 4  # fewer available than the cap
    assert all(t.video_path is not None for group in selected.values() for t in group)
    assert selected == select_calibration_trials(list(reversed(trials)), per_group=6)  # input-order independent


def test_select_calibration_trials_spreads_picks_across_the_subject_range() -> None:
    trials = [_trial("Veh", "1% DMSO", n) for n in range(1, 61)]
    picked = [int(t.subject_id) for t in select_calibration_trials(trials, per_group=6)[("veh", 0)]]
    assert picked[0] == 1 and picked[-1] >= 50  # not just the first six subjects


def test_track_cache_round_trip(tmp_path: Path) -> None:
    tracks = [
        Track(0, 0.0, 1.0, 2.0, 90.0, 2.0, True),
        Track(1, 0.033, 0.0, 0.0, None, None, False),
    ]
    save_cached_tracks(tmp_path, "veh_0_0001", tracks)
    assert load_cached_tracks(tmp_path, "veh_0_0001") == tracks


def test_load_cached_tracks_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_cached_tracks(tmp_path, "missing") is None


def test_select_rejects_nonpositive_cap() -> None:
    with pytest.raises(ValueError):
        select_calibration_trials([], per_group=0)
