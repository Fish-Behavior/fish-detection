"""Per-frame behavior-state classification (T047-T052) - FR-007/FR-007a.

Design decisions these tests lock in (docs/progress.md Phase 7 has the narrative):

- Rule precedence: `detected=False` -> Undetermined (absolute) > Dead (whole-video
  terminal-suffix pass) > Surface Breach (per-frame) > Listing/LORR (opt-in, sustained)
  > Freezing/Drift (sustained low speed) > Erratic Movement (speed at/above a threshold)
  > Controlled Swim (moving, below it) > Undetermined (no computable speed).
- Every speed-based rule uses smoothed speed (`features.compute_smoothed_speed`, displacement
  over a 1 s lag), never raw frame-to-frame velocity, which is centroid-jitter noise on real
  footage. Controlled vs Erratic is a speed band (calibrated threshold), not a mixture model:
  the pooled GMM it replaced only saw the fastest frames and could not be calibrated.
- No gap-bridging: "sustained for >= N seconds" is N seconds of contiguous detected frames; an
  undetected frame resets the clock.
- Listing/LORR is OFF unless `listing_orientation_deviation_deg` is given: body angle does not
  separate LORR on real footage (docs/progress.md), so it is labeled manually in review.
- A frame whose speed is not computable is Undetermined, never guessed.
"""

from __future__ import annotations

import pytest

from prepds.labeling import (
    DEFAULT_ERRATIC_SPEED_THRESHOLD_PX_PER_S,
    DEFAULT_FREEZE_SPEED_FLOOR_PX_PER_S,
    DEFAULT_MIN_DEAD_BOUT_S,
    DEFAULT_MIN_FREEZE_BOUT_S,
    DEFAULT_SURFACE_BREACH_DEPTH_THRESHOLD_PX,
    classify_prepared,
    classify_prepared_states,
    classify_video,
    prepare_video,
)
from prepds.models import BehaviorState, FrameSource, Track

FPS = 30.0
FREEZE_SPEED = DEFAULT_FREEZE_SPEED_FLOOR_PX_PER_S * 0.3
CONTROLLED_SPEED = (DEFAULT_FREEZE_SPEED_FLOOR_PX_PER_S + DEFAULT_ERRATIC_SPEED_THRESHOLD_PX_PER_S) / 2
ERRATIC_SPEED = DEFAULT_ERRATIC_SPEED_THRESHOLD_PX_PER_S * 2.5


def _track(frame_idx, x=0.0, y=100.0, orientation_deg=90.0, detected=True) -> Track:
    return Track(
        frame_idx=frame_idx,
        t_sec=frame_idx / FPS,
        x=x if detected else 0.0,
        y=y if detected else 0.0,
        orientation_deg=orientation_deg if detected else None,
        y_from_frame_top=y if detected else None,
        detected=detected,
    )


def _run(n, speed, *, start=0, x0=0.0, y=100.0, orientation_deg=90.0) -> list[Track]:
    """`n` consecutive detected frames drifting along x at `speed` px/s."""
    return [
        _track(start + i, x=x0 + speed * i / FPS, y=y, orientation_deg=orientation_deg) for i in range(n)
    ]


def _states(tracks, **kwargs):
    return [s.state for s in classify_video(tracks, **kwargs)]


# --- undetected frames are Undetermined, never guessed -------------------------


def test_undetected_frames_are_undetermined_not_guessed() -> None:
    tracks = _run(90, CONTROLLED_SPEED)
    tracks[40] = _track(40, detected=False)
    result = classify_video(tracks)
    assert result[40].state == BehaviorState.UNDETERMINED
    assert result[40].confidence is None
    assert result[40].source == FrameSource.AUTO


def test_undetected_overrides_every_other_rule() -> None:
    assert _states([_track(0, y=0.0, detected=False)]) == [BehaviorState.UNDETERMINED]


# --- Controlled vs Erratic: a calibrated speed band --------------------------------


def test_moderate_speed_is_controlled_and_fast_speed_is_erratic() -> None:
    controlled = _states(_run(120, CONTROLLED_SPEED))
    erratic = _states(_run(120, ERRATIC_SPEED))
    assert set(controlled) == {BehaviorState.CONTROLLED_SWIM}
    assert set(erratic) == {BehaviorState.ERRATIC_MOVEMENT}


def test_the_erratic_threshold_is_a_parameter() -> None:
    tracks = _run(120, CONTROLLED_SPEED)
    assert set(_states(tracks, erratic_speed_threshold_px_per_s=CONTROLLED_SPEED / 2)) == {
        BehaviorState.ERRATIC_MOVEMENT
    }


def test_speed_band_frames_carry_no_confidence() -> None:
    assert all(s.confidence is None for s in classify_video(_run(120, ERRATIC_SPEED)))


def test_a_detected_frame_with_no_computable_speed_is_undetermined() -> None:
    # 10 frames is under the 1 s lag: no partner frame exists for any of them.
    assert set(_states(_run(10, CONTROLLED_SPEED))) == {BehaviorState.UNDETERMINED}


def test_jittery_centroid_does_not_look_like_erratic_swimming() -> None:
    # +-5 px alternating jitter on a fish drifting at 6 px/s: raw frame-to-frame speed is ~300 px/s,
    # the smoothed speed is the drift.
    tracks = [
        _track(i, x=6.0 * i / FPS + (5.0 if i % 2 == 0 else -5.0)) for i in range(int(DEFAULT_MIN_FREEZE_BOUT_S * FPS) + 60)
    ]
    assert set(_states(tracks)) == {BehaviorState.FREEZING_DRIFT}


# --- Freezing/Drift -----------------------------------------------------------------


def test_freezing_rule_sustained_low_speed() -> None:
    tracks = _run(int(DEFAULT_MIN_FREEZE_BOUT_S * FPS) + 60, FREEZE_SPEED)
    states = _states(tracks)
    assert states[5] == BehaviorState.FREEZING_DRIFT  # the whole qualifying bout is labeled
    assert states[-1] == BehaviorState.FREEZING_DRIFT


def test_freezing_rule_does_not_fire_before_minimum_bout_duration() -> None:
    short = _run(int((DEFAULT_MIN_FREEZE_BOUT_S - 0.5) * FPS), FREEZE_SPEED)
    tail = _run(90, ERRATIC_SPEED, start=len(short), x0=short[-1].x)
    assert BehaviorState.FREEZING_DRIFT not in _states(short + tail)


def test_freezing_bout_resets_across_an_undetected_gap_no_bridging() -> None:
    half = int(DEFAULT_MIN_FREEZE_BOUT_S * FPS * 0.6)
    tracks = _run(half, FREEZE_SPEED) + [_track(half, detected=False)] + _run(half, FREEZE_SPEED, start=half + 1)
    assert BehaviorState.FREEZING_DRIFT not in _states(tracks)


def test_qualifying_freezing_bout_is_labeled_from_its_first_frame() -> None:
    bout = _run(int((DEFAULT_MIN_FREEZE_BOUT_S + 1.0) * FPS), FREEZE_SPEED)
    tail = _run(90, ERRATIC_SPEED, start=len(bout), x0=bout[-1].x)  # not a Dead-eligible suffix
    states = _states(bout + tail)
    assert all(states[i] == BehaviorState.FREEZING_DRIFT for i in range(len(bout) - 5))


# --- Surface Breach ----------------------------------------------------------------


def test_surface_breach_is_a_per_frame_threshold_crossing() -> None:
    threshold = DEFAULT_SURFACE_BREACH_DEPTH_THRESHOLD_PX
    tracks = _run(90, CONTROLLED_SPEED, y=200.0)
    tracks[45] = _track(45, x=tracks[45].x, y=threshold * 0.5)
    states = _states(tracks)
    assert states[45] == BehaviorState.SURFACE_BREACH
    assert states[44] != BehaviorState.SURFACE_BREACH and states[46] != BehaviorState.SURFACE_BREACH


def test_surface_breach_needs_no_computable_speed() -> None:
    assert set(_states(_run(3, 0.0, y=DEFAULT_SURFACE_BREACH_DEPTH_THRESHOLD_PX * 0.3))) == {
        BehaviorState.SURFACE_BREACH
    }


def test_surface_breach_outranks_freezing() -> None:
    tracks = _run(150, FREEZE_SPEED, y=DEFAULT_SURFACE_BREACH_DEPTH_THRESHOLD_PX * 0.3)
    assert set(_states(tracks)) == {BehaviorState.SURFACE_BREACH}


# --- Listing/LORR: opt-in only ------------------------------------------------------


def test_listing_is_off_by_default_even_for_a_vertical_fish() -> None:
    tracks = _run(150, CONTROLLED_SPEED, orientation_deg=0.0)  # cv2.fitEllipse: 0 == vertical
    assert BehaviorState.LISTING_LORR not in _states(tracks)


def test_listing_fires_when_enabled_and_labels_the_whole_bout() -> None:
    bout = _run(int(3.0 * FPS), CONTROLLED_SPEED, orientation_deg=0.0)
    tail = _run(90, CONTROLLED_SPEED, start=len(bout), x0=bout[-1].x)
    states = _states(bout + tail, listing_orientation_deviation_deg=45.0, min_listing_bout_s=2.0)
    assert all(states[i] == BehaviorState.LISTING_LORR for i in range(len(bout)))
    assert states[len(bout) + 10] != BehaviorState.LISTING_LORR


def test_listing_abstains_when_orientation_is_none() -> None:
    tracks = [
        Track(t.frame_idx, t.t_sec, t.x, t.y, None, t.y_from_frame_top, True)
        for t in _run(150, FREEZE_SPEED)
    ]
    states = _states(tracks, listing_orientation_deviation_deg=45.0)
    assert states[-1] == BehaviorState.FREEZING_DRIFT


def test_listing_bout_resets_across_an_undetected_gap() -> None:
    half = int(2.0 * FPS * 0.6)
    tracks = (
        _run(half, CONTROLLED_SPEED, orientation_deg=0.0)
        + [_track(half, detected=False)]
        + _run(half, CONTROLLED_SPEED, start=half + 1, orientation_deg=0.0)
    )
    states = _states(tracks, listing_orientation_deviation_deg=45.0, min_listing_bout_s=2.0)
    assert BehaviorState.LISTING_LORR not in states


# --- Dead (FR-007a) -----------------------------------------------------------------


def test_dead_fires_on_sustained_terminal_stillness_and_is_monotonic() -> None:
    tracks = _run(int(DEFAULT_MIN_DEAD_BOUT_S * FPS) + 60, FREEZE_SPEED)
    states = _states(tracks)
    assert states[-1] == BehaviorState.DEAD
    onset = states.index(BehaviorState.DEAD)
    assert all(s == BehaviorState.DEAD for s in states[onset:])


def test_dead_does_not_fire_if_the_video_tail_is_undetected() -> None:
    n = int(DEFAULT_MIN_DEAD_BOUT_S * FPS) + 60
    tracks = _run(n - 5, FREEZE_SPEED) + [_track(i, detected=False) for i in range(n - 5, n)]
    states = _states(tracks)
    assert BehaviorState.DEAD not in states
    assert states[-1] == BehaviorState.UNDETERMINED


def test_dead_does_not_fire_if_the_fish_recovers_before_the_end() -> None:
    still = _run(int(DEFAULT_MIN_DEAD_BOUT_S * FPS) + 60, FREEZE_SPEED)
    moving = _run(90, ERRATIC_SPEED, start=len(still), x0=still[-1].x)
    assert BehaviorState.DEAD not in _states(still + moving)


# --- general behavior --------------------------------------------------------------


def test_classify_video_returns_one_state_per_frame_in_order() -> None:
    tracks = _run(100, CONTROLLED_SPEED)
    result = classify_video(tracks)
    assert [s.frame_idx for s in result] == [t.frame_idx for t in tracks]
    assert [s.t_sec for s in result] == [t.t_sec for t in tracks]


def test_classify_video_empty_input() -> None:
    assert classify_video([]) == []


def test_classify_video_is_deterministic() -> None:
    tracks = _run(60, FREEZE_SPEED) + _run(60, ERRATIC_SPEED, start=60, x0=10.0)
    assert classify_video(tracks) == classify_video(tracks)


# --- prepared-video path (calibration performance) -----------------------------------


def _mixed_video() -> list[Track]:
    tracks = _run(90, FREEZE_SPEED) + _run(90, CONTROLLED_SPEED, start=90) + _run(90, ERRATIC_SPEED, start=180)
    tracks[100] = _track(100, detected=False)
    tracks[200] = _track(200, x=tracks[200].x, y=5.0)
    return tracks


def test_classify_prepared_matches_classify_video() -> None:
    tracks = _mixed_video()
    assert classify_prepared(prepare_video(tracks)) == classify_video(tracks)


def test_classify_prepared_respects_threshold_overrides() -> None:
    tracks = _mixed_video()
    overrides = dict(
        freeze_speed_floor_px_per_s=2.0,
        erratic_speed_threshold_px_per_s=20.0,
        min_freeze_bout_s=0.5,
        surface_breach_depth_threshold_px=2.0,
        listing_orientation_deviation_deg=20.0,
        min_listing_bout_s=0.2,
    )
    assert classify_prepared(prepare_video(tracks), **overrides) == classify_video(tracks, **overrides)


def test_classify_prepared_states_returns_just_the_states() -> None:
    prepared = prepare_video(_mixed_video())
    assert classify_prepared_states(prepared) == [s.state for s in classify_prepared(prepared)]


def test_prepared_video_speed_lag_is_a_parameter() -> None:
    # 20 frames: undetermined at the default 1 s lag, computable at a 0.2 s lag.
    tracks = _run(20, CONTROLLED_SPEED)
    assert set(classify_prepared_states(prepare_video(tracks))) == {BehaviorState.UNDETERMINED}
    assert set(classify_prepared_states(prepare_video(tracks, speed_lag_s=0.2))) == {
        BehaviorState.CONTROLLED_SWIM
    }


def test_erratic_threshold_must_exceed_the_freeze_floor() -> None:
    # Otherwise Freezing (higher precedence) swallows the whole Controlled band and "Controlled"
    # degenerates into brief slow blips - a configuration that fits data but means nothing.
    with pytest.raises(ValueError, match="erratic_speed_threshold_px_per_s"):
        classify_video(_run(60, CONTROLLED_SPEED), freeze_speed_floor_px_per_s=50.0, erratic_speed_threshold_px_per_s=20.0)
