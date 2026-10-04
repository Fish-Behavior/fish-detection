"""Per-fish features on hand-built segments and frames (plan U5, T1.10): PRD §5.3, §5.4, EC-3."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from dcs import schema
from dcs.featurize import BEHAVIOR_STATES, kinematic_features, segment_features, slug

CS, ER, FD = schema.CONTROLLED_SWIM, schema.ERRATIC_MOVEMENT, schema.FREEZING_DRIFT
UNKNOWN = schema.UNDETERMINED


def segments(*rows: tuple[float, float, str] | tuple[float, float, str, str]) -> pd.DataFrame:
    """(start_s, end_s, state[, source]) rows -> a segments.csv table."""
    records = [(start, end, end - start, state, *(rest or (schema.SOURCE_AUTO,))) for start, end, state, *rest in rows]
    return pd.DataFrame(records, columns=list(schema.SEGMENT_COLUMNS))


def share(state: str) -> str:
    return f"state_{slug(state)}_share"


def bouts(state: str) -> str:
    return f"state_{slug(state)}_bouts"


def mean_bout(state: str) -> str:
    return f"state_{slug(state)}_mean_bout_s"


def latency(state: str) -> str:
    return f"state_{slug(state)}_latency_s"


def transition(a: str, b: str) -> str:
    return f"trans_{slug(a)}_to_{slug(b)}"


# --- segments --------------------------------------------------------------------------------

SIMPLE = segments((0, 4, CS), (4, 5, ER), (5, 8, CS), (8, 10, FD))


def test_behavior_states_are_the_six_prepds_states_without_undetermined() -> None:
    assert BEHAVIOR_STATES == tuple(state for state in schema.STATES if state != UNKNOWN)
    assert len({slug(state) for state in BEHAVIOR_STATES}) == 6
    assert all(slug(state).isidentifier() for state in BEHAVIOR_STATES)


def test_share_bouts_mean_bout_and_latency_per_state() -> None:
    features = segment_features(SIMPLE)
    assert features[share(CS)] == pytest.approx(0.7)
    assert features[share(ER)] == pytest.approx(0.1)
    assert features[share(FD)] == pytest.approx(0.2)
    assert (features[bouts(CS)], features[bouts(ER)], features[bouts(FD)]) == (2, 1, 1)
    assert features[mean_bout(CS)] == pytest.approx(3.5)
    assert features[mean_bout(FD)] == pytest.approx(2.0)
    assert (features[latency(CS)], features[latency(ER)], features[latency(FD)]) == (0, 4, 8)
    assert features["recording_s"] == pytest.approx(10.0)


def test_ec3_state_never_shown_gives_zero_bouts_zero_mean_and_latency_of_the_whole_recording() -> None:
    features = segment_features(SIMPLE)
    for state in (schema.LISTING_LORR, schema.SURFACE_BREACH, schema.DEAD):
        assert features[share(state)] == 0
        assert features[bouts(state)] == 0
        assert features[mean_bout(state)] == 0
        assert features[latency(state)] == pytest.approx(10.0)


def test_one_column_set_per_state_and_no_undetermined_feature() -> None:
    features = segment_features(SIMPLE)
    for state in BEHAVIOR_STATES:
        assert {share(state), bouts(state), mean_bout(state), latency(state)} <= set(features)
    assert not [name for name in features if slug(UNKNOWN) in name]


def test_adjacent_segments_of_one_state_are_one_bout() -> None:
    """prepds starts a new segment when the label source changes (auto -> manual) even if the state does not."""
    features = segment_features(segments((0, 2, CS), (2, 4, CS, schema.SOURCE_MANUAL), (4, 6, ER)))
    assert features[bouts(CS)] == 1
    assert features[mean_bout(CS)] == pytest.approx(4.0)
    assert features[transition(CS, ER)] == 1


def test_transition_counts_between_states() -> None:
    features = segment_features(SIMPLE)
    names = [name for name in features if name.startswith("trans_")]
    assert len(names) == 6 * 5  # every ordered pair of different states
    assert features[transition(CS, ER)] == 1
    assert features[transition(ER, CS)] == 1
    assert features[transition(CS, FD)] == 1
    assert sum(features[name] for name in names) == 3


def test_undetermined_is_unknown_time_not_a_state() -> None:
    """Every unreviewed video has Undetermined time (U3 note): shares are of the known time, transitions are
    not bridged across it, and bouts on both sides of it stay separate."""
    features = segment_features(segments((0, 2, CS), (2, 6, UNKNOWN), (6, 8, CS), (8, 10, ER)))
    assert features[share(CS)] == pytest.approx(4 / 6)
    assert features[share(ER)] == pytest.approx(2 / 6)
    assert features[bouts(CS)] == 2
    assert features[transition(CS, ER)] == 1
    assert features[latency(ER)] == pytest.approx(8.0)
    assert features[latency(FD)] == pytest.approx(10.0)  # still the whole recording
    assert features["recording_s"] == pytest.approx(10.0)

    across = segment_features(segments((0, 2, CS), (2, 4, UNKNOWN), (4, 6, ER)))
    assert across[transition(CS, ER)] == 0


def test_latency_counts_from_the_recording_start() -> None:
    features = segment_features(segments((1, 3, CS), (3, 5, ER)))
    assert features[latency(ER)] == pytest.approx(2.0)
    assert features[latency(FD)] == pytest.approx(4.0)


def test_recording_with_no_known_state_gives_none() -> None:
    assert segment_features(segments((0, 5, UNKNOWN))) is None


# --- frames ----------------------------------------------------------------------------------


def frames(detected: list[bool], **columns: list[float]) -> pd.DataFrame:
    """Frames with prepds' undetected conventions: kinematic 0.0 sentinel, null depth."""
    n = len(detected)
    table = pd.DataFrame({"detected": detected, "is_immobile": columns.pop("is_immobile", [False] * n)})
    for name in schema.KINEMATIC_COLUMNS:
        table[name] = np.asarray(columns.pop(name, [1.0] * n), dtype="float32")
    table["depth_from_surface"] = np.asarray(columns.pop("depth", [10.0] * n), dtype="float32")
    table.loc[~table["detected"], list(schema.KINEMATIC_COLUMNS)] = 0.0
    table.loc[~table["detected"], "depth_from_surface"] = np.nan
    table["source"] = schema.SOURCE_AUTO
    assert not columns
    return table


MIXED = frames(
    [True, True, False, True, False],
    velocity=[2.0, 4.0, 99.0, 6.0, 99.0],
    acceleration=[-2.0, 2.0, 99.0, 5.0, 99.0],
    angular_velocity=[-3.0, 0.0, 99.0, 3.0, 99.0],
    meander=[0.0, 0.3, 99.0, 0.6, 99.0],
    is_immobile=[True, False, True, False, False],
    depth=[10.0, 20.0, 0.0, 60.0, 0.0],
)


def test_kinematics_use_detected_frames_only() -> None:
    features = kinematic_features(MIXED)
    assert features["velocity_mean"] == pytest.approx(4.0)
    assert features["velocity_median"] == pytest.approx(4.0)
    assert features["velocity_cv"] == pytest.approx(np.std([2.0, 4.0, 6.0]) / 4.0)
    assert features["abs_acceleration_mean"] == pytest.approx(3.0)
    assert features["abs_angular_velocity_mean"] == pytest.approx(2.0)
    assert features["meander_mean"] == pytest.approx(0.3)
    assert features["immobile_share"] == pytest.approx(1 / 3)
    assert features["detected_share"] == pytest.approx(3 / 5)


def test_depth_features_ignore_undetected_frames() -> None:
    features = kinematic_features(MIXED)
    depth = [10.0, 20.0, 60.0]
    assert features["depth_mean"] == pytest.approx(30.0)
    assert features["depth_min"] == pytest.approx(10.0)
    for percent in (1, 10, 50, 90, 99):
        assert features[f"depth_p{percent:02d}"] == pytest.approx(np.percentile(depth, percent))


def test_velocity_cv_is_zero_when_the_fish_never_moves() -> None:
    still = frames([True, True], velocity=[0.0, 0.0])
    assert kinematic_features(still)["velocity_cv"] == 0


def test_kinematic_features_are_finite_numbers() -> None:
    assert all(isinstance(value, float) and math.isfinite(value) for value in kinematic_features(MIXED).values())


def test_no_detected_frame_gives_none() -> None:
    assert kinematic_features(frames([False, False, False])) is None
