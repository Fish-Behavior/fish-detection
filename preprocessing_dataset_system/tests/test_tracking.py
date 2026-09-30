"""Per-frame fish localization from foreground extraction (T031/T032) - FR-005.

Uses the synthetic moving-dot video for position-accuracy ground truth (its
motion function is deterministic and known - see fixtures/make_synth_video.py)
and blank/background-only frames for the detected=false case.

Tracking uses `cv2.createBackgroundSubtractorKNN` (see tracking.py module
docstring for why, and for the real-video validation status - as of this
pass detection quality on real footage is still under active investigation,
not yet a settled win over the original approach). One real, expected
consequence of the KNN switch: the first few frames are never detected -
`apply()` has no prior history yet, so nothing can register as foreground
relative to "background" until the model has seen a handful of frames.
Measured directly (not assumed) at exactly 4 frames on both the synthetic
fixture and a 35865-frame real video, independent of video length - a
genuine, bounded algorithmic property, not a bug to route around, and
asserted on explicitly below rather than silently excluded from the
trajectory check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prepds.tracking import track_video, track_video_with_context
from tests.fixtures.make_synth_video import _dot_position

FIXTURES = Path(__file__).parent / "fixtures"
SYNTH_VIDEO = FIXTURES / "synth_tiny.mp4"


def test_tracks_synthetic_moving_dot_matches_known_trajectory() -> None:
    track = track_video(SYNTH_VIDEO)
    assert len(track) == 150  # 5.0s @ 30fps

    # Spot-check several frames against the fixture's own known ground-truth
    # position function, not just "some track exists." Kept well clear of
    # the KNN warmup window (frames 0-3, see below).
    for frame_idx in (30, 75, 120, 149):
        t = frame_idx / 30.0
        expected_x, expected_y = _dot_position(t)
        row = track[frame_idx]
        assert row.frame_idx == frame_idx
        assert row.detected is True
        assert row.x == pytest.approx(expected_x, abs=3.0)
        assert row.y == pytest.approx(expected_y, abs=3.0)


def test_leading_warmup_window_is_exactly_four_frames_knn_has_no_prior_history() -> None:
    # Measured directly, not assumed: KNN's first `apply()` calls have no
    # history to compare against, so nothing registers as foreground until
    # the model has accumulated a few samples. This is a fixed property of
    # the algorithm (confirmed the same width on a 35865-frame real video -
    # see tracking.py module docstring), not something that grows with video
    # length, so asserting an exact width here is deliberate, not fragile.
    track = track_video(SYNTH_VIDEO)
    assert [row.detected for row in track[:4]] == [False, False, False, False]
    assert track[4].detected is True


def test_tracks_synthetic_moving_dot_detects_at_least_95_percent_of_frames() -> None:
    track = track_video(SYNTH_VIDEO)
    detected_fraction = sum(1 for row in track if row.detected) / len(track)
    assert detected_fraction >= 0.95


def test_tracks_synthetic_moving_dot_t_sec_matches_frame_idx_over_fps() -> None:
    track = track_video(SYNTH_VIDEO)
    for row in track[::30]:
        assert row.t_sec == pytest.approx(row.frame_idx / 30.0, abs=0.01)


def test_detected_false_on_blank_frames(tmp_path: Path) -> None:
    import cv2

    blank_path = tmp_path / "blank.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(blank_path), fourcc, 30.0, (304, 240))
    background = np.full((240, 304, 3), 180, dtype=np.uint8)
    for _ in range(10):
        writer.write(background)
    writer.release()

    track = track_video(blank_path)
    assert len(track) == 10
    assert all(row.detected is False for row in track)
    # An undetected frame must never silently report a plausible-looking
    # position as if it were real (§3.3 - Undetermined, never guessed).
    assert all(row.x == 0.0 and row.y == 0.0 for row in track)
    # y_from_frame_top must not fabricate a surface-relative reading either
    # (0.0 would silently read as "at the surface" - see models.py docstring).
    assert all(row.y_from_frame_top is None for row in track)


def test_track_video_missing_file_raises() -> None:
    with pytest.raises(ValueError):
        track_video(FIXTURES / "does_not_exist.mp4")


def test_track_video_is_deterministic_across_repeated_calls() -> None:
    # NFR-002: "deterministic tracking/classification, no unseeded
    # randomness." `cv2.createBackgroundSubtractorKNN`'s internal
    # sample-replacement history update is stochastic (OpenCV's global RNG),
    # confirmed on real footage: two unseeded `track_video()` calls on the
    # identical file differed in `detected` on ~4% of frames. `track_video`
    # reseeds that RNG to a fixed default before each call - this must
    # produce byte-identical output, not just a similar detection rate.
    run_a = track_video(SYNTH_VIDEO)
    run_b = track_video(SYNTH_VIDEO)
    assert run_a == run_b


def test_track_video_with_context_matches_track_video_and_returns_valid_waterline() -> None:
    tracks_only = track_video(SYNTH_VIDEO)
    result = track_video_with_context(SYNTH_VIDEO)
    assert result.tracks == tracks_only
    # detect_waterline() always returns *some* row (it finds the strongest
    # gradient even on a flat background - see roi.py docstring), so the only
    # thing to check here is that it's a valid row index for this frame.
    assert 0 <= result.waterline_y < 240


def test_reacquires_after_long_gap_instead_of_permanently_rejecting_distant_position(
    tmp_path: Path,
) -> None:
    """Regression test for a real code-review finding: without a reset, the
    jump-rejection anchor (`last_detected_xy`) stayed pinned to the position
    seen before a long undetected gap, so every real detection afterwards
    kept getting rejected as "too far" once the fish had genuinely moved -
    a permanent-stranding failure mode, not a one-frame edge case. On real
    footage, undetected runs up to 2214 frames were observed (see
    tracking.py module docstring), so a gap this long is not a contrived
    scenario.

    Video: a dot drifts from x=40 to x=79 near y=50 over 40 frames (real
    motion - genuinely establishes `last_detected_xy` via legitimate
    detections, verified empirically; a dot held perfectly still from frame
    0 is learned as background immediately and is never detected at all -
    a KNN quirk found while building this test, see the follow-up code
    review notes in docs/progress.md Phase 4), then vanishes (blank
    background) for 40 frames - long enough to exceed
    `DEFAULT_MAX_CONSECUTIVE_MISSES_BEFORE_REACQUIRE` (30) - then reappears
    far away and stationary at position B (200, 150), more than
    `DEFAULT_MAX_JUMP_PX` (60) from the last A position, for 20 more frames.
    """
    import cv2

    width, height = 304, 240
    background = np.full((height, width, 3), 180, dtype=np.uint8)
    dot_color = (20, 20, 20)
    position_b = (200, 150)

    video_path = tmp_path / "reacquire.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
    for i in range(40):
        frame = background.copy()
        cv2.circle(frame, (40 + i, 50), 6, dot_color, thickness=-1)
        writer.write(frame)
    for _ in range(40):
        writer.write(background.copy())
    for _ in range(20):
        frame = background.copy()
        cv2.circle(frame, position_b, 6, dot_color, thickness=-1)
        writer.write(frame)
    writer.release()

    track = track_video(video_path)
    assert len(track) == 100

    # The A-phase drift must have actually been detected - otherwise this
    # test never establishes a real stale anchor to test the reset against
    # (a real risk with a synthetic-video test, verified empirically before
    # relying on it here).
    assert any(row.detected for row in track[:40]), (
        "A-phase drift was never detected - this test isn't exercising a "
        "real stale anchor"
    )

    # The reappearance at B must be detected, not stuck rejecting against
    # the stale anchor from A - check the tail of the video, past any
    # re-detection settling time.
    tail = track[90:]
    assert any(row.detected for row in tail), (
        "fish never re-detected at the new position after the gap - "
        "the jump-rejection anchor is stuck on the stale pre-gap position"
    )
    for row in tail:
        if row.detected:
            assert row.x == pytest.approx(position_b[0], abs=5.0)
            assert row.y == pytest.approx(position_b[1], abs=5.0)


def test_reacquires_after_gap_made_of_rejected_jumps_not_just_missing_contours(
    tmp_path: Path,
) -> None:
    """Companion to the test above, covering the other half of what the
    `DEFAULT_MAX_CONSECUTIVE_MISSES_BEFORE_REACQUIRE` comment documents: "no
    contour, OR a contour rejected as an implausible jump." The previous test
    exercises the no-contour case (blank frames during the gap); this one
    keeps a real, moving contour present throughout the gap - far from the
    anchor - so every frame gets jump-rejected, not dropped for lack of a
    contour. (A *stationary* dot doesn't work for this: verified empirically
    that KNN absorbs a constant-position object into its background model
    within well under 30 frames, so it stops producing a contour at all long
    before the miss-counter reset could fire - see the drift design below
    and docs/progress.md Phase 4.)

    Video: a dot drifts near A (x=40..79, y=50) for 40 frames - real motion,
    establishes a genuine stale anchor - then drifts locally near B
    (x=190..209, y=150) for 40 more frames, always more than
    `DEFAULT_MAX_JUMP_PX` (60) from the stale A anchor.
    """
    import cv2

    width, height = 304, 240
    background = np.full((height, width, 3), 180, dtype=np.uint8)
    dot_color = (20, 20, 20)

    video_path = tmp_path / "reacquire_visible.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
    for i in range(40):
        frame = background.copy()
        cv2.circle(frame, (40 + i, 50), 6, dot_color, thickness=-1)
        writer.write(frame)
    for i in range(40):
        frame = background.copy()
        cv2.circle(frame, (190 + i % 20, 150), 6, dot_color, thickness=-1)
        writer.write(frame)
    writer.release()

    track = track_video(video_path)
    assert len(track) == 80
    assert any(row.detected for row in track[:40]), (
        "A-phase drift was never detected - this test isn't exercising a "
        "real stale anchor"
    )

    # The first 30 frames after the jump to B must be rejected (a real
    # contour is present every frame - just too far from the stale A
    # anchor) - confirms the counter is fed by the rejected-jump path, not
    # skipping it because no contour was found.
    post_jump = track[40:70]
    assert all(not row.detected for row in post_jump), (
        "expected the first 30 post-jump frames to be jump-rejected, not "
        "detected - the rejected-jump path may not be incrementing "
        "consecutive_misses"
    )

    # Once the anchor resets (at the 30th consecutive miss), detection must
    # resume at B and track its continued local drift there.
    tail = track[70:]
    assert all(row.detected for row in tail), (
        "fish not reliably re-detected at B after the reset point"
    )
    for i, row in enumerate(tail):
        expected_x = 190 + (30 + i) % 20
        assert row.x == pytest.approx(expected_x, abs=5.0)
        assert row.y == pytest.approx(150, abs=5.0)


def _speckled_bar(shape: tuple[int, int], center: tuple[int, int], length: int, thickness: int, vertical: bool, seed: int = 0) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cx, cy = center
    half_l, half_t = length // 2, thickness // 2
    if vertical:
        mask[cy - half_l : cy + half_l, cx - half_t : cx + half_t] = 255
    else:
        mask[cy - half_t : cy + half_t, cx - half_l : cx + half_l] = 255
    rng = np.random.default_rng(seed)
    mask[(rng.random(shape) < 0.25) & (mask > 0)] = 0  # punch holes: speckle
    return mask


def test_detect_fish_orientation_horizontal_body_is_near_90_even_when_speckled() -> None:
    from prepds.tracking import _detect_fish

    mask = _speckled_bar((240, 320), (160, 100), length=40, thickness=8, vertical=False)
    centroid, orientation = _detect_fish(mask, min_area_px2=15, max_area_px2=5000)
    assert centroid is not None and orientation is not None
    assert abs(orientation - 90.0) <= 10.0
    assert centroid[0] == pytest.approx(160, abs=3)


def test_detect_fish_orientation_vertical_body_is_near_0_or_180_even_when_speckled() -> None:
    from prepds.tracking import _detect_fish

    mask = _speckled_bar((240, 320), (160, 100), length=40, thickness=8, vertical=True)
    _, orientation = _detect_fish(mask, min_area_px2=15, max_area_px2=5000)
    assert orientation is not None
    assert min(orientation, 180.0 - orientation) <= 10.0


def test_detect_fish_speckled_fragments_are_merged_into_one_body() -> None:
    from prepds.tracking import _detect_fish

    mask = _speckled_bar((240, 320), (160, 100), length=40, thickness=8, vertical=False)
    # A second faint fragment far away must not pull the centroid.
    mask[200:203, 20:23] = 255
    centroid, _ = _detect_fish(mask, min_area_px2=15, max_area_px2=5000)
    assert centroid is not None
    assert centroid[0] == pytest.approx(160, abs=3) and centroid[1] == pytest.approx(100, abs=3)
