"""Per-frame kinematic feature derivation from a `Track` list (T036-T039) - FR-006.

Design decisions this module's tests lock in (see docs/progress.md Phase 5 for
the full narrative - PRD §9.5.4 leaves several of these genuinely open):

- All four kinematic fields (`velocity`, `acceleration`, `angular_velocity`,
  `meander`) use the same `0.0`-sentinel-on-not-computable convention as
  `Track.x`/`Track.y` (never `None` - `frames.parquet` types them
  non-nullable). A value is only ever real when there is enough *consecutive,
  detected* history behind it: velocity needs 2 consecutive detected frames,
  acceleration needs 3, angular_velocity needs 2 consecutive detected frames
  that both also have a non-`None` `orientation_deg` (a detected frame can
  still have `orientation_deg=None` - too small a contour for
  `cv2.fitEllipse`, a real Phase 4 output shape).
- `orientation_deg` (from `cv2.fitEllipse`) is periodic with period **180**,
  not 360 - verified empirically (an ellipse's major axis doesn't distinguish
  head from tail: a synthetic ellipse rotated 0deg and 180deg both fit to the
  same angle). `angular_velocity` must use a period-180 circular difference,
  or it's wrong by up to 2x at the wraparound boundary.
- `meander` ("path curvature normalized by distance", PRD §9.5.4) is
  implemented as *trajectory* tortuosity - turning angle between consecutive
  position-derived heading vectors, summed over a rolling window and divided
  by the total path length in that window - not body-orientation curvature.
  This keeps it independent from `angular_velocity` (which already carries
  the body-orientation signal), consistent with the labeling GMM needing
  3 genuinely distinct input dimensions (PRD §9.5.6). Heading is a full
  360-periodic quantity (unlike body orientation - "moving left" and "moving
  right" are genuinely different headings), so its circular diff uses
  period 360.
"""

from __future__ import annotations

import math

import pytest

from prepds.features import (
    DEFAULT_IMMOBILITY_VELOCITY_FLOOR_PX_PER_S,
    DEFAULT_IMMOBILITY_WINDOW_FRAMES,
    _circular_diff,
    compute_feature_validity,
    consecutive_detected_run_length,
    derive_features,
)
from prepds.models import Track


def _track(frame_idx, t_sec, x, y, orientation_deg=None, detected=True) -> Track:
    return Track(
        frame_idx=frame_idx,
        t_sec=t_sec,
        x=x,
        y=y,
        orientation_deg=orientation_deg,
        y_from_frame_top=y if detected else None,
        detected=detected,
    )


def test_circular_diff_period_360_shortest_path() -> None:
    # Heading wraps at 360 -> 0deg; the shortest turn from 350deg to 10deg is
    # +20deg, not -340deg.
    assert _circular_diff(10, 350, period=360) == pytest.approx(20.0)
    assert _circular_diff(350, 10, period=360) == pytest.approx(-20.0)
    # Exactly half a period apart is a genuine ambiguity (+180 and -180 are
    # the same rotation on a circle) - only the magnitude is well-defined.
    assert abs(_circular_diff(180, 0, period=360)) == pytest.approx(180.0)


def test_circular_diff_period_180_shortest_path() -> None:
    # Body-orientation wraps at 180 -> 0deg (fitEllipse periodicity, verified
    # empirically - an ellipse's major axis has no head/tail distinction).
    assert _circular_diff(5, 175, period=180) == pytest.approx(10.0)
    assert _circular_diff(175, 5, period=180) == pytest.approx(-10.0)


def test_velocity_matches_finite_difference() -> None:
    # T036: a straight horizontal move of 30px over exactly 1.0s -> 30 px/s.
    tracks = [
        _track(0, 0.0, 0.0, 0.0),
        _track(1, 1.0, 30.0, 0.0),
    ]
    features = derive_features(tracks)
    assert features[0].velocity == 0.0  # no prior frame - not computable
    assert features[1].velocity == pytest.approx(30.0)


def test_velocity_is_zero_sentinel_across_an_undetected_gap() -> None:
    # A detected frame immediately after an undetected one must not compute
    # a "velocity" using the undetected frame's fabricated x=y=0.0 position.
    tracks = [
        _track(0, 0.0, 10.0, 10.0),
        _track(1, 1.0, 0.0, 0.0, detected=False),
        _track(2, 2.0, 500.0, 500.0),  # real position, far from frame 0
    ]
    features = derive_features(tracks)
    assert features[1].velocity == 0.0  # undetected frame itself
    assert features[2].velocity == 0.0  # first-detected-after-gap: no valid prior


def test_velocity_is_zero_sentinel_across_a_frame_idx_gap() -> None:
    # Even if both frames are detected, a non-adjacent frame_idx (e.g. a
    # caller passing a stride-sampled or filtered track list) must not
    # produce a fabricated velocity across the skipped frames.
    tracks = [
        _track(0, 0.0, 0.0, 0.0),
        _track(5, 5.0, 100.0, 0.0),  # frame_idx jumps by 5, not 1
    ]
    features = derive_features(tracks)
    assert features[1].velocity == 0.0


def test_acceleration_matches_finite_difference_of_velocity() -> None:
    # T036 extension: constant acceleration a=10 px/s^2 over unit steps.
    # x(t) = 0.5 * a * t^2 -> velocities at t=1,2 are 5, 15 (finite diff),
    # acceleration at t=2 is (15-5)/1 = 10.
    tracks = [
        _track(0, 0.0, 0.0, 0.0),
        _track(1, 1.0, 5.0, 0.0),
        _track(2, 2.0, 20.0, 0.0),
    ]
    features = derive_features(tracks)
    assert features[0].acceleration == 0.0  # no history
    assert features[1].acceleration == 0.0  # only 1 prior velocity - not enough
    assert features[2].acceleration == pytest.approx(10.0)


def test_acceleration_is_zero_sentinel_right_after_a_gap_even_with_two_valid_frames() -> None:
    # Frames 2,3 are a valid consecutive pair (real velocity at frame 3), but
    # frame 1 was undetected, so frame 3 doesn't have 3 consecutive detected
    # frames of history - acceleration must stay a sentinel, not silently
    # difference against a velocity computed from before the gap.
    tracks = [
        _track(0, 0.0, 0.0, 0.0),
        _track(1, 1.0, 0.0, 0.0, detected=False),
        _track(2, 2.0, 10.0, 10.0),
        _track(3, 3.0, 20.0, 10.0),
    ]
    features = derive_features(tracks)
    assert features[3].velocity != 0.0  # real velocity, frame 2->3
    assert features[3].acceleration == 0.0  # but not enough history for acceleration


def test_angular_velocity_uses_period_180_wraparound() -> None:
    # 175deg -> 5deg is a +10deg turn (through the 180/0 seam), not -170deg.
    tracks = [
        _track(0, 0.0, 0.0, 0.0, orientation_deg=175.0),
        _track(1, 1.0, 1.0, 0.0, orientation_deg=5.0),
    ]
    features = derive_features(tracks)
    assert features[1].angular_velocity == pytest.approx(10.0)


def test_angular_velocity_is_zero_sentinel_when_orientation_is_none() -> None:
    # A detected frame can still have orientation_deg=None (contour too
    # small for cv2.fitEllipse - a real Phase 4 output shape). This must not
    # poison the neighboring frames' own angular_velocity computation.
    tracks = [
        _track(0, 0.0, 0.0, 0.0, orientation_deg=10.0),
        _track(1, 1.0, 1.0, 0.0, orientation_deg=None),  # detected, but no ellipse fit
        _track(2, 2.0, 2.0, 0.0, orientation_deg=30.0),
    ]
    features = derive_features(tracks)
    assert features[1].angular_velocity == 0.0  # this frame's own orientation is None
    assert features[2].angular_velocity == 0.0  # no valid orientation at frame 1 to diff against
    # But velocity (position-based) is unaffected by the missing orientation.
    assert features[1].velocity == pytest.approx(math.hypot(1.0, 0.0))
    assert features[2].velocity == pytest.approx(math.hypot(1.0, 0.0))


def test_meander_is_zero_for_a_straight_line_path() -> None:
    # T038: no turning at all -> meander (turn per unit distance) is 0.
    tracks = [_track(i, float(i), float(i) * 10.0, 0.0) for i in range(10)]
    features = derive_features(tracks)
    assert features[-1].meander == pytest.approx(0.0, abs=1e-9)


def test_meander_is_positive_and_larger_for_a_tighter_zigzag() -> None:
    # A path that reverses direction every step has much higher turning per
    # unit distance than one that only turns gently.
    sharp_zigzag = []
    x = 0.0
    for i in range(10):
        x += 10.0 if i % 2 == 0 else -10.0
        sharp_zigzag.append(_track(i, float(i), x, 0.0))

    gentle_path = [_track(i, float(i), float(i) * 10.0, math.sin(i) * 0.5) for i in range(10)]

    sharp_features = derive_features(sharp_zigzag)
    gentle_features = derive_features(gentle_path)

    assert sharp_features[-1].meander > 0.0
    assert sharp_features[-1].meander > gentle_features[-1].meander


def test_meander_zero_path_length_in_window_does_not_divide_by_zero() -> None:
    # A fish that hasn't moved at all within the window: 0 distance, 0 turn -
    # must report 0.0, not raise or return NaN/inf.
    tracks = [_track(i, float(i), 5.0, 5.0) for i in range(5)]
    features = derive_features(tracks)
    for f in features:
        assert f.meander == pytest.approx(0.0)
        assert math.isfinite(f.meander)


def test_immobility_flag_on_sustained_low_velocity() -> None:
    # T037: velocity below the floor for the full rolling window -> True;
    # a single frame of real movement partway through -> False for frames
    # that don't yet have a full low-velocity window behind them.
    floor = DEFAULT_IMMOBILITY_VELOCITY_FLOOR_PX_PER_S
    window = DEFAULT_IMMOBILITY_WINDOW_FRAMES
    tracks = [_track(i, float(i), 5.0, 5.0) for i in range(window + 5)]  # stationary throughout
    features = derive_features(tracks)

    # Not enough history yet for the first `window` frames' worth of run.
    assert features[window - 1].is_immobile is False
    # Once `window` consecutive real, sub-floor-velocity frames exist:
    assert features[window].is_immobile is True
    assert features[-1].is_immobile is True
    assert floor >= 0.0  # sanity: the constant itself is a non-negative threshold


def test_immobility_flag_false_when_history_includes_a_gap() -> None:
    # A rolling window that includes an undetected frame can't confirm
    # sustained low velocity - the flag must fail open to False, not assume
    # immobility across an unknown stretch.
    window = DEFAULT_IMMOBILITY_WINDOW_FRAMES
    tracks = [_track(i, float(i), 5.0, 5.0) for i in range(window + 5)]
    tracks[3] = _track(3, 3.0, 0.0, 0.0, detected=False)
    features = derive_features(tracks)
    # The window ending just past the gap still includes the undetected
    # frame's broken run, so immobility can't be confirmed there.
    assert features[3 + window - 1].is_immobile is False


def test_immobility_flag_false_when_genuinely_moving() -> None:
    tracks = [_track(i, float(i), float(i) * 100.0, 0.0) for i in range(DEFAULT_IMMOBILITY_WINDOW_FRAMES + 2)]
    features = derive_features(tracks)
    assert all(f.is_immobile is False for f in features)


def test_derive_features_returns_one_row_per_track_frame() -> None:
    tracks = [_track(i, float(i), float(i), 0.0) for i in range(20)]
    features = derive_features(tracks)
    assert len(features) == 20
    assert [f.frame_idx for f in features] == [t.frame_idx for t in tracks]


def test_derive_features_empty_input() -> None:
    assert derive_features([]) == []


def test_velocity_and_acceleration_guard_against_non_positive_dt() -> None:
    # Defensive: t_sec should be strictly increasing for consecutive
    # frame_idx (video_io guarantees this in practice), but a zero or
    # negative dt must never reach a division - sentinel out, not a
    # ZeroDivisionError or a fabricated inf/nan.
    tracks = [
        _track(0, 0.0, 0.0, 0.0),
        _track(1, 0.0, 10.0, 0.0),  # same t_sec as frame 0 - dt == 0
        _track(2, 1.0, 20.0, 0.0),
    ]
    features = derive_features(tracks)
    assert features[1].velocity == 0.0
    assert all(math.isfinite(f.velocity) for f in features)
    assert all(math.isfinite(f.acceleration) for f in features)

    # Same guard, but tripped on acceleration's own (later) step specifically
    # - 3 consecutive detected frames (enough history for acceleration), but
    # the last t_sec step is non-positive.
    tracks_accel = [
        _track(0, 0.0, 0.0, 0.0),
        _track(1, 1.0, 10.0, 0.0),
        _track(2, 1.0, 20.0, 0.0),  # same t_sec as frame 1 - dt == 0
    ]
    accel_features = derive_features(tracks_accel)
    assert accel_features[2].acceleration == 0.0
    assert all(math.isfinite(f.acceleration) for f in accel_features)


def test_meander_is_sentinel_zero_on_undetected_frames_not_a_stale_value() -> None:
    # Regression test (code review, CRITICAL): meander's rolling-window sum
    # was indexed by list position only, with no run-boundary check, so an
    # undetected frame right after a real zigzag reported the zigzag's
    # leftover turn/distance ratio instead of the documented 0.0 sentinel.
    tracks = []
    x = 0.0
    for i in range(6):
        x += 10.0 if i % 2 == 0 else -10.0
        tracks.append(_track(i, float(i), x, 0.0))  # sharp zigzag, frames 0-5
    tracks.append(_track(6, 6.0, 0.0, 0.0, detected=False))  # undetected

    features = derive_features(tracks)
    assert features[5].meander > 0.0  # real zigzag turning established first
    assert features[6].meander == 0.0  # undetected frame: sentinel, not stale


def test_meander_does_not_blend_pre_gap_and_post_gap_trajectory() -> None:
    # Same root cause as above, the other direction: a detected frame soon
    # after reacquisition must not still be averaging in turn/distance from
    # before the gap.
    tracks = []
    x = 0.0
    for i in range(6):
        x += 10.0 if i % 2 == 0 else -10.0
        tracks.append(_track(i, float(i), x, 0.0))  # sharp zigzag, frames 0-5
    tracks.append(_track(6, 6.0, 0.0, 0.0, detected=False))  # gap
    # Reacquire far away, moving in a perfectly straight line (true meander
    # of this segment, on its own, is exactly 0).
    for i, fx in enumerate(range(1000, 1000 + 4 * 50, 50)):
        tracks.append(_track(7 + i, 7.0 + i, float(fx), 500.0))

    features = derive_features(tracks)
    # Index 9 (frame_idx 9) is the 3rd frame of the post-gap run - the first
    # point with a real turn value computed purely from post-gap steps.
    assert features[9].meander == pytest.approx(0.0, abs=1e-9)


def test_acceleration_does_not_treat_a_dt_sentinel_velocity_as_real_history() -> None:
    # Regression test (code review, MEDIUM): run_length[i] >= 3 is necessary
    # but not sufficient for "velocities[i-1] is a real measurement" - a
    # dt<=0 step produces a 0.0 velocity sentinel that run_length can't see,
    # and acceleration must not difference against it as if it were real.
    tracks = [
        _track(0, 0.0, 0.0, 0.0),
        _track(1, 0.0, 500.0, 0.0),  # same t_sec as frame 0 -> dt=0 -> sentinel velocity
        _track(2, 1.0, 520.0, 0.0),  # real small step, real dt
    ]
    features = derive_features(tracks)
    assert features[2].acceleration == 0.0


def test_immobility_flag_does_not_count_a_dt_sentinel_velocity_as_a_real_below_floor_sample() -> None:
    # Regression test (code review, MEDIUM): a dt<=0 step's 0.0 velocity
    # sentinel must not silently satisfy "<=floor" inside an immobility
    # window - the docstring promises "fails open to False," but a
    # dt-sentinel bypassed that.
    window = DEFAULT_IMMOBILITY_WINDOW_FRAMES
    floor = DEFAULT_IMMOBILITY_VELOCITY_FLOOR_PX_PER_S
    far_x = 5.0 + 10 * floor + 1000.0

    tracks = [_track(0, 0.0, 5.0, 5.0)]
    tracks.append(_track(1, 0.0, far_x, 5.0))  # same t_sec as frame 0 -> dt=0, huge real jump
    for i in range(2, window + 3):
        tracks.append(_track(i, float(i - 1), far_x, 5.0))  # stationary from here, real dt

    features = derive_features(tracks)
    # A window ending here spans indices [1, window] - includes the
    # dt-sentinel frame.
    assert features[window].is_immobile is False


# --- compute_feature_validity (added for Phase 7's code review - labeling.py
# needs a public, correct validity mask instead of re-deriving a partial one) ---


def test_compute_feature_validity_flags_dt_sentinel_velocity_as_invalid() -> None:
    # Regression test (Phase 7 code review, HIGH): labeling.py's own
    # re-derivation of "is this velocity real" checked frame_idx adjacency
    # but not dt>0, silently treating a dt<=0 sentinel as a real
    # below-floor measurement. The fix is for labeling.py to consume this
    # function instead - lock in that the function itself gets it right.
    tracks = [
        _track(0, 1.0, 0.0, 100.0),
        _track(1, 1.0, 500.0, 100.0),  # same t_sec as frame 0 -> dt=0 -> sentinel
    ]
    validity = compute_feature_validity(tracks)
    assert validity.velocity == [False, False]


def test_compute_feature_validity_flags_dt_sentinel_angular_velocity_as_invalid() -> None:
    tracks = [
        _track(0, 1.0, 0.0, 0.0, orientation_deg=10.0),
        _track(1, 1.0, 0.0, 0.0, orientation_deg=80.0),  # same t_sec -> dt=0 -> sentinel
    ]
    validity = compute_feature_validity(tracks)
    assert validity.angular_velocity == [False, False]


def test_compute_feature_validity_real_values_marked_valid() -> None:
    tracks = [
        _track(0, 0.0, 0.0, 0.0, orientation_deg=10.0),
        _track(1, 1.0, 5.0, 0.0, orientation_deg=20.0),
        _track(2, 2.0, 10.0, 0.0, orientation_deg=30.0),
    ]
    validity = compute_feature_validity(tracks)
    assert validity.velocity == [False, True, True]
    assert validity.angular_velocity == [False, True, True]
    assert validity.meander == [False, True, True]


def test_compute_feature_validity_meander_invalid_on_undetected_and_run_start() -> None:
    tracks = [
        _track(0, 0.0, 0.0, 0.0),
        _track(1, 1.0, 0.0, 0.0, detected=False),
        _track(2, 2.0, 5.0, 0.0),  # first frame of a new run - not yet 2 real steps
        _track(3, 3.0, 10.0, 0.0),
    ]
    validity = compute_feature_validity(tracks)
    assert validity.meander == [False, False, False, True]


def test_compute_feature_validity_empty_input() -> None:
    validity = compute_feature_validity([])
    assert validity.velocity == []
    assert validity.angular_velocity == []
    assert validity.meander == []


def test_consecutive_detected_run_length_is_publicly_importable() -> None:
    tracks = [_track(0, 0.0, 0.0, 0.0), _track(1, 1.0, 1.0, 1.0, detected=False), _track(2, 2.0, 2.0, 2.0)]
    assert consecutive_detected_run_length(tracks) == [1, 0, 1]


# --- smoothed speed (labeling input): displacement over a lag, robust to centroid jitter ---


def _speed_track(frame_idx: int, x: float, *, detected: bool = True, fps: float = 30.0) -> Track:
    return Track(
        frame_idx=frame_idx,
        t_sec=frame_idx / fps,
        x=x if detected else 0.0,
        y=100.0 if detected else 0.0,
        orientation_deg=90.0 if detected else None,
        y_from_frame_top=100.0 if detected else None,
        detected=detected,
    )


def test_smoothed_speed_is_displacement_over_the_lag_not_frame_to_frame_jitter() -> None:
    from prepds.features import compute_smoothed_speed

    # True drift 30 px/s (1 px/frame) plus +-4 px centroid jitter every frame: the raw per-frame
    # velocity is ~240 px/s, the 1 s displacement recovers ~30 px/s.
    tracks = [_speed_track(i, i + (4.0 if i % 2 == 0 else -4.0)) for i in range(90)]

    speeds, valid = compute_smoothed_speed(tracks, lag_s=1.0)

    assert all(valid)
    assert speeds[45] == pytest.approx(30.0, abs=0.5)  # even lag: jitter cancels exactly


def test_smoothed_speed_uses_the_forward_frame_when_no_backward_history_exists() -> None:
    from prepds.features import compute_smoothed_speed

    tracks = [_speed_track(i, 2.0 * i) for i in range(60)]  # 60 px/s
    speeds, valid = compute_smoothed_speed(tracks, lag_s=1.0)

    assert valid[0] and speeds[0] == pytest.approx(60.0)  # first frame borrows the frame 1 s ahead


def test_smoothed_speed_is_not_computable_across_a_detection_gap_and_never_bridges_it() -> None:
    from prepds.features import compute_smoothed_speed

    tracks = [_speed_track(i, float(i), detected=(i not in range(20, 60))) for i in range(100)]
    speeds, valid = compute_smoothed_speed(tracks, lag_s=1.0)

    for i in range(20, 60):
        assert not valid[i] and speeds[i] == 0.0  # undetected frames have no speed
    assert not valid[70]  # frame 40 (its backward partner) undetected and frame 100 does not exist
    assert valid[65]  # backward partner 35 undetected, forward partner 95 detected -> real speed


def test_smoothed_speed_requires_a_contiguous_frame_index_lag() -> None:
    from prepds.features import compute_smoothed_speed

    # frame_idx jumps 39 -> 70: the frame 30 list-positions back from position 40 is frame 10, which is
    # 60 frames (2 s) earlier, not the 1 s lag, so it must not be used as a partner.
    idx = list(range(40)) + list(range(70, 80))
    tracks = [_speed_track(i, float(i)) for i in idx]
    speeds, valid = compute_smoothed_speed(tracks, lag_s=1.0)

    assert valid[5] and speeds[5] == pytest.approx(30.0)  # forward partner frame 35 is exactly 1 s away
    assert not valid[10] and not valid[40]  # their only in-range partners sit across the frame_idx jump


def test_smoothed_speed_empty_and_single_frame_inputs() -> None:
    from prepds.features import compute_smoothed_speed

    assert compute_smoothed_speed([], lag_s=1.0) == ([], [])
    speeds, valid = compute_smoothed_speed([_speed_track(0, 0.0)], lag_s=1.0)
    assert speeds == [0.0] and valid == [False]


def test_smoothed_speed_spans_a_gap_shorter_than_the_lag_because_both_endpoints_are_real() -> None:
    from prepds.features import compute_smoothed_speed

    # frames 40..44 undetected; frame 50's partner (frame 20) and frame 20's partner (frame 50) are real
    tracks = [_speed_track(i, 2.0 * i, detected=(i not in range(40, 45))) for i in range(90)]
    speeds, valid = compute_smoothed_speed(tracks, lag_s=1.0)

    assert valid[50] and speeds[50] == pytest.approx(60.0)
    assert not valid[42]  # an undetected frame never gets a speed
