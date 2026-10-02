"""Temporal consolidation of per-frame labels into reviewable bins.

Real footage flickers: F_332 alone produced ~10,000 segments in 1200 s (median 2 frames) because the
tracker's detection toggles ~5,000 times. The reference figures are drawn at ~1 s resolution, and FR-015
forces the reviewer to resolve every Undetermined segment, so per-frame output is unusable. Each fixed
time bin takes the majority of its DETERMINED frames (Undetermined abstains), is Undetermined only when
too few frames were determined, and keeps any Surface Breach spike (a one-frame event by definition).
"""

from __future__ import annotations

import pytest

from prepds.consolidate import consolidate_states
from prepds.models import BehaviorState as B, FrameSource, StateFrame
from prepds.segments import frames_to_segments

FPS = 30.0


def _frames(states: list[B], *, start=0) -> list[StateFrame]:
    return [
        StateFrame(frame_idx=start + i, t_sec=(start + i) / FPS, state=s, source=FrameSource.AUTO, confidence=None)
        for i, s in enumerate(states)
    ]


def _states(frames):
    return [f.state for f in frames]


def test_frame_level_flicker_inside_a_bin_becomes_its_majority_state() -> None:
    raw = ([B.FREEZING_DRIFT] * 2 + [B.ERRATIC_MOVEMENT]) * 10  # 30 frames = one 1 s bin
    assert set(_states(consolidate_states(_frames(raw)))) == {B.FREEZING_DRIFT}


def test_undetermined_frames_abstain_rather_than_vote() -> None:
    raw = [B.UNDETERMINED] * 15 + [B.CONTROLLED_SWIM] * 9 + [B.ERRATIC_MOVEMENT] * 6
    assert set(_states(consolidate_states(_frames(raw)))) == {B.CONTROLLED_SWIM}


def test_a_bin_with_too_few_determined_frames_is_undetermined() -> None:
    raw = [B.UNDETERMINED] * 27 + [B.CONTROLLED_SWIM] * 3  # 10 % determined
    assert set(_states(consolidate_states(_frames(raw), min_determined_fraction=0.2))) == {B.UNDETERMINED}
    assert set(_states(consolidate_states(_frames(raw), min_determined_fraction=0.05))) == {B.CONTROLLED_SWIM}


def test_a_single_surface_breach_frame_keeps_its_bin() -> None:
    raw = [B.CONTROLLED_SWIM] * 29 + [B.SURFACE_BREACH]
    assert set(_states(consolidate_states(_frames(raw)))) == {B.SURFACE_BREACH}


def test_ties_break_deterministically_by_precedence() -> None:
    raw = [B.CONTROLLED_SWIM] * 15 + [B.FREEZING_DRIFT] * 15
    assert set(_states(consolidate_states(_frames(raw)))) == {B.FREEZING_DRIFT}


def test_bins_are_time_based_and_each_bin_is_labeled_independently() -> None:
    raw = [B.FREEZING_DRIFT] * 30 + [B.ERRATIC_MOVEMENT] * 30
    out = _states(consolidate_states(_frames(raw)))
    assert out[:30] == [B.FREEZING_DRIFT] * 30 and out[30:] == [B.ERRATIC_MOVEMENT] * 30


def test_output_preserves_frame_identity_source_and_order_and_drops_confidence() -> None:
    raw = [B.CONTROLLED_SWIM] * 45
    frames = _frames(raw, start=100)
    out = consolidate_states(frames)
    assert [f.frame_idx for f in out] == [f.frame_idx for f in frames]
    assert [f.t_sec for f in out] == [f.t_sec for f in frames]
    assert all(f.source == FrameSource.AUTO and f.confidence is None for f in out)


def test_a_frame_index_gap_does_not_merge_bins() -> None:
    a = _frames([B.FREEZING_DRIFT] * 30)
    b = _frames([B.ERRATIC_MOVEMENT] * 30, start=300)  # 10 s later
    out = consolidate_states(a + b)
    assert _states(out) == [B.FREEZING_DRIFT] * 30 + [B.ERRATIC_MOVEMENT] * 30


def test_dead_suffix_stays_monotonic_and_segments_accept_it() -> None:
    raw = [B.ERRATIC_MOVEMENT] * 40 + [B.DEAD] * 80
    out = consolidate_states(_frames(raw))
    segments = frames_to_segments(out)  # raises DeadMonotonicityError if Dead is ever followed by non-Dead
    assert segments[-1].state == B.DEAD


def test_flicker_no_longer_explodes_the_segment_count() -> None:
    raw = ([B.FREEZING_DRIFT, B.UNDETERMINED, B.ERRATIC_MOVEMENT, B.UNDETERMINED] * 300)[:1200]  # 40 s
    assert len(frames_to_segments(_frames(raw))) == 1200
    assert len(frames_to_segments(consolidate_states(_frames(raw)))) <= 40


def test_empty_input_and_invalid_parameters() -> None:
    assert consolidate_states([]) == []
    with pytest.raises(ValueError, match="bin_s"):
        consolidate_states(_frames([B.DEAD]), bin_s=0.0)
    with pytest.raises(ValueError, match="min_determined_fraction"):
        consolidate_states(_frames([B.DEAD]), min_determined_fraction=1.5)


def test_a_lone_dead_frame_does_not_turn_a_surface_breach_bin_into_dead() -> None:
    from prepds.consolidate import consolidate_labels

    states = [B.SURFACE_BREACH] * 5 + [B.DEAD]
    out = consolidate_labels(states, [i / 10 for i in range(6)])
    assert set(out) == {B.SURFACE_BREACH}


def test_dead_still_wins_a_surface_bin_when_it_holds_at_least_half_the_determined_frames() -> None:
    from prepds.consolidate import consolidate_labels

    states = [B.SURFACE_BREACH] * 3 + [B.DEAD] * 3
    assert set(consolidate_labels(states, [i / 10 for i in range(6)])) == {B.DEAD}
