"""Reviewer flags computed from tracks (docs/progress.md, Dead investigation).

A fish that stops moving for the rest of the recording is either Dead or in LORR / deep sedation, and
neither the tracker (a motionless fish is absorbed by the background model) nor the video alone separates
them (FR-015a). The pipeline therefore does not decide: it flags the terminal no-action stretch so the
reviewer confirms it. "Movement" is a detected frame with a real smoothed speed above the freeze floor;
undetected frames and frames without a computable speed are neither movement nor stillness evidence.
"""

from __future__ import annotations

import pytest

from prepds.models import Track
from prepds.review_flags import DEFAULT_MIN_TERMINAL_NO_ACTION_S, terminal_no_action_flag

FPS = 30.0
FLOOR = 20.0
FAST = 100.0  # px/s
SLOW = 2.0


def _t(frame_idx, x=0.0, detected=True) -> Track:
    return Track(frame_idx, frame_idx / FPS, x if detected else 0.0, 100.0 if detected else 0.0,
                 90.0 if detected else None, 100.0 if detected else None, detected)


def _moving(n, speed, *, start=0, x0=0.0):
    return [_t(start + i, x0 + speed * i / FPS) for i in range(n)]


def _undetected(n, *, start):
    return [_t(start + i, detected=False) for i in range(n)]


def _flag(tracks, **kw):
    return terminal_no_action_flag(tracks, freeze_speed_floor_px_per_s=FLOOR, **kw)


def test_moving_then_an_undetected_tail_is_flagged_from_the_last_movement() -> None:
    moving = _moving(600, FAST)  # 20 s
    tail = _undetected(int(200 * FPS), start=len(moving))
    flag = _flag(moving + tail)

    assert flag is not None and flag.kind == "terminal_no_action"
    assert flag.start_s == pytest.approx(moving[-1].t_sec, abs=1.5)  # last frame with a real >floor speed
    assert flag.end_s == pytest.approx((moving + tail)[-1].t_sec)
    assert "Dead" in flag.message and "LORR" in flag.message


def test_a_detected_but_slow_tail_is_flagged_too() -> None:
    moving = _moving(600, FAST)
    tail = _moving(int(200 * FPS), SLOW, start=len(moving), x0=moving[-1].x)
    assert _flag(moving + tail) is not None


def test_a_tail_shorter_than_the_minimum_is_not_flagged() -> None:
    moving = _moving(600, FAST)
    tail = _undetected(int((DEFAULT_MIN_TERMINAL_NO_ACTION_S - 20) * FPS), start=len(moving))
    assert _flag(moving + tail) is None


def test_a_fish_that_moves_to_the_end_is_not_flagged() -> None:
    assert _flag(_moving(int(400 * FPS), FAST)) is None


def test_a_fish_that_never_moves_is_flagged_from_the_start() -> None:
    tracks = _moving(int(200 * FPS), SLOW)
    flag = _flag(tracks)
    assert flag is not None and flag.start_s == 0.0


def test_a_video_shorter_than_the_minimum_is_not_flagged_even_if_still() -> None:
    assert _flag(_moving(int(30 * FPS), SLOW)) is None


def test_frames_without_a_computable_speed_do_not_count_as_movement() -> None:
    moving = _moving(600, FAST)
    tail = _undetected(int(200 * FPS), start=len(moving))
    # an isolated detected frame in the tail: no partner 1 s away -> no speed -> neither movement nor stillness
    tail[100] = _t(tail[100].frame_idx, x=500.0)
    flag = _flag(moving + tail)
    assert flag is not None and flag.start_s == pytest.approx(moving[-1].t_sec, abs=1.5)


def test_movement_late_in_the_tail_resets_the_flag() -> None:
    moving = _moving(600, FAST)
    gap = _undetected(int(200 * FPS), start=len(moving))
    late = _moving(int(3 * FPS), FAST, start=len(moving) + len(gap), x0=0.0)
    assert _flag(moving + gap + late) is None


def test_the_minimum_duration_is_a_parameter() -> None:
    moving = _moving(600, FAST)
    tail = _undetected(int(60 * FPS), start=len(moving))
    assert _flag(moving + tail) is None
    assert _flag(moving + tail, min_no_action_s=30.0) is not None


def test_empty_input_is_not_flagged() -> None:
    assert _flag([]) is None
