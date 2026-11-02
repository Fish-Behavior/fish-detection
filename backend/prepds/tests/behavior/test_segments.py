"""Run-length encoding of per-frame states into segments.csv rows (T053) - FR-007a.

`frames_to_segments` also re-validates the Dead monotonicity invariant
independently of `labeling.classify_video` (which already guarantees it by
construction - see labeling.py's module docstring): defense in depth against
a `StateFrame` sequence assembled some other way (e.g. after manual review
edits), per FR-007a's "reject or flag" semantics - validate, never silently
repair.
"""

from __future__ import annotations

import pytest

from prepds.models import BehaviorState, FrameSource, StateFrame, StateSegment
from prepds.segments import DeadMonotonicityError, frames_to_segments, segments_to_frames


def _sf(frame_idx, t_sec, state, source=FrameSource.AUTO, confidence=None) -> StateFrame:
    return StateFrame(frame_idx=frame_idx, t_sec=t_sec, state=state, source=source, confidence=confidence)


def test_frames_to_segments_merges_consecutive_same_state() -> None:
    frames = [
        _sf(0, 0.0, BehaviorState.FREEZING_DRIFT),
        _sf(1, 0.1, BehaviorState.FREEZING_DRIFT),
        _sf(2, 0.2, BehaviorState.FREEZING_DRIFT),
        _sf(3, 0.3, BehaviorState.CONTROLLED_SWIM),
        _sf(4, 0.4, BehaviorState.CONTROLLED_SWIM),
    ]
    segments = frames_to_segments(frames)
    assert len(segments) == 2
    assert segments[0].state == BehaviorState.FREEZING_DRIFT
    assert segments[0].start_s == pytest.approx(0.0)
    assert segments[0].end_s == pytest.approx(0.3)  # up to the next segment's start
    assert segments[1].state == BehaviorState.CONTROLLED_SWIM
    assert segments[1].start_s == pytest.approx(0.3)


def test_frames_to_segments_splits_on_source_change_even_if_state_matches() -> None:
    frames = [
        _sf(0, 0.0, BehaviorState.CONTROLLED_SWIM, source=FrameSource.AUTO),
        _sf(1, 0.1, BehaviorState.CONTROLLED_SWIM, source=FrameSource.MANUAL),
    ]
    segments = frames_to_segments(frames)
    assert len(segments) == 2
    assert segments[0].source == FrameSource.AUTO
    assert segments[1].source == FrameSource.MANUAL


def test_frames_to_segments_single_frame_segment_has_extrapolated_duration() -> None:
    frames = [
        _sf(0, 0.0, BehaviorState.CONTROLLED_SWIM),
        _sf(1, 0.1, BehaviorState.SURFACE_BREACH),
        _sf(2, 0.2, BehaviorState.CONTROLLED_SWIM),
    ]
    segments = frames_to_segments(frames)
    assert len(segments) == 3
    breach = segments[1]
    assert breach.start_s == pytest.approx(0.1)
    assert breach.end_s == pytest.approx(0.2)
    assert breach.duration_s == pytest.approx(0.1)


def test_frames_to_segments_last_segment_extrapolates_end_from_frame_spacing() -> None:
    frames = [
        _sf(0, 0.0, BehaviorState.CONTROLLED_SWIM),
        _sf(1, 0.1, BehaviorState.CONTROLLED_SWIM),
        _sf(2, 0.2, BehaviorState.FREEZING_DRIFT),
    ]
    segments = frames_to_segments(frames)
    last = segments[-1]
    assert last.start_s == pytest.approx(0.2)
    assert last.end_s == pytest.approx(0.3)  # 0.2 + median spacing (0.1)
    assert last.duration_s == pytest.approx(0.1)


def test_frames_to_segments_empty_input() -> None:
    assert frames_to_segments([]) == []


def test_frames_to_segments_single_frame_total() -> None:
    segments = frames_to_segments([_sf(0, 0.0, BehaviorState.UNDETERMINED)])
    assert len(segments) == 1
    assert segments[0].start_s == pytest.approx(0.0)
    assert segments[0].end_s == pytest.approx(0.0)
    assert segments[0].duration_s == pytest.approx(0.0)


# --- FR-007a Dead monotonicity validation ------------------------------------


def test_frames_to_segments_accepts_dead_running_to_the_end() -> None:
    frames = [
        _sf(0, 0.0, BehaviorState.CONTROLLED_SWIM),
        _sf(1, 0.1, BehaviorState.DEAD),
        _sf(2, 0.2, BehaviorState.DEAD),
    ]
    segments = frames_to_segments(frames)
    assert segments[-1].state == BehaviorState.DEAD


def test_frames_to_segments_raises_on_non_dead_frame_after_dead() -> None:
    frames = [
        _sf(0, 0.0, BehaviorState.CONTROLLED_SWIM),
        _sf(1, 0.1, BehaviorState.DEAD),
        _sf(2, 0.2, BehaviorState.FREEZING_DRIFT),  # violates FR-007a
    ]
    with pytest.raises(DeadMonotonicityError) as exc_info:
        frames_to_segments(frames)
    message = str(exc_info.value)
    assert "2" in message  # offending frame_idx
    assert "1" in message  # dead onset frame_idx
    assert BehaviorState.FREEZING_DRIFT.value in message


def test_frames_to_segments_raises_even_on_undetermined_after_dead() -> None:
    # Undetermined is the one state that might look "safe" to allow after
    # Dead, but FR-007a's invariant is Dead-is-terminal, full stop - even
    # Undetermined after Dead is a data-quality error to flag, not something
    # to wave through.
    frames = [
        _sf(0, 0.0, BehaviorState.DEAD),
        _sf(1, 0.1, BehaviorState.UNDETERMINED),
    ]
    with pytest.raises(DeadMonotonicityError):
        frames_to_segments(frames)


# --- T054: round trip ---------------------------------------------------------


def test_segments_round_trip_preserves_state_and_source() -> None:
    frames = [
        _sf(0, 0.0, BehaviorState.CONTROLLED_SWIM),
        _sf(1, 0.1, BehaviorState.CONTROLLED_SWIM),
        _sf(2, 0.2, BehaviorState.FREEZING_DRIFT),
        _sf(3, 0.3, BehaviorState.FREEZING_DRIFT),
        _sf(4, 0.4, BehaviorState.FREEZING_DRIFT),
        _sf(5, 0.5, BehaviorState.SURFACE_BREACH),
        _sf(6, 0.6, BehaviorState.DEAD),
        _sf(7, 0.7, BehaviorState.DEAD),
    ]
    segments = frames_to_segments(frames)
    reconstructed = segments_to_frames(segments, [f.frame_idx for f in frames], [f.t_sec for f in frames])

    assert len(reconstructed) == len(frames)
    for original, rebuilt in zip(frames, reconstructed):
        assert rebuilt.frame_idx == original.frame_idx
        assert rebuilt.t_sec == pytest.approx(original.t_sec)
        assert rebuilt.state == original.state
        assert rebuilt.source == original.source
        # Confidence is genuinely lost crossing segments.csv - never fabricated back.
        assert rebuilt.confidence is None


def test_segments_to_frames_empty_input() -> None:
    assert segments_to_frames([], [], []) == []


def test_segments_to_frames_raises_on_frames_with_no_segments() -> None:
    with pytest.raises(ValueError):
        segments_to_frames([], [0], [0.0])


def test_segments_to_frames_raises_on_non_monotonic_t_secs() -> None:
    # Regression test (code review, MEDIUM): the documented "non-decreasing
    # t_sec" precondition must be validated, not silently violated into
    # misattributed frames.
    frames = [_sf(0, 0.0, BehaviorState.CONTROLLED_SWIM), _sf(1, 0.1, BehaviorState.FREEZING_DRIFT)]
    segments = frames_to_segments(frames)
    with pytest.raises(ValueError):
        segments_to_frames(segments, [0, 1], [0.1, 0.0])  # decreasing


def test_frames_to_segments_splits_on_frame_idx_gap_even_if_state_and_source_match() -> None:
    # Regression test (code review, LOW): a frame_idx gap must start a new
    # segment even when state/source are unchanged across it - merging
    # across a gap would fabricate continuity nobody observed, the same
    # principle labeling.py enforces for bout duration.
    frames = [
        _sf(0, 0.0, BehaviorState.CONTROLLED_SWIM),
        _sf(1, 0.1, BehaviorState.CONTROLLED_SWIM),
        _sf(5, 0.5, BehaviorState.CONTROLLED_SWIM),  # frame_idx gap: 1 -> 5
        _sf(6, 0.6, BehaviorState.CONTROLLED_SWIM),
    ]
    segments = frames_to_segments(frames)
    # Splitting into two segments (rather than merging into one that would
    # claim continuous CONTROLLED_SWIM across the gap frames 2-4, which
    # were never observed) is the point of this test - the tiling
    # convention (each segment's end_s is the next segment's start_s) still
    # applies across the split, same as any other boundary.
    assert len(segments) == 2
    assert segments[0].state == BehaviorState.CONTROLLED_SWIM
    assert segments[1].state == BehaviorState.CONTROLLED_SWIM
    assert segments[0].end_s == pytest.approx(0.5) == segments[1].start_s
