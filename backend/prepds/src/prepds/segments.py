"""Run-length encoding between per-frame states and segments.csv rows - FR-007a.

`frames_to_segments` merges consecutive `StateFrame`s that share both
`state` and `source` into one `StateSegment`. A segment's `end_s` is the
*next* segment's `start_s` (frames tile the video with no gaps between
segments - PRD's schema doesn't spell out this convention, so it is fixed
here); the final segment has no "next" frame to borrow a boundary from, so
its `end_s` is extrapolated using the median inter-frame spacing observed
across the whole sequence (median, not the last observed gap, so one
anomalous final interval can't skew it).

**FR-007a Dead monotonicity is re-validated here**, independently of
`labeling.classify_video` (which already guarantees it by construction -
see labeling.py's module docstring): this is defense in depth for any
`StateFrame` sequence assembled some other way (e.g. after manual review
edits reorder or overwrite states). "Reject or flag" - `frames_to_segments`
raises `DeadMonotonicityError`, carrying the offending frame's index and
state plus the Dead-onset frame's index, rather than silently repairing or
dropping frames.

A `frame_idx` gap between two otherwise-mergeable frames also starts a new
segment (Phase 7 code review, LOW), even though the current sole producer
(`tracking.py`'s `track_video`, one `Track` per `frame_idx` with no skips)
can't trigger this today: `labeling.py` treats `frame_idx` gaps as
first-class everywhere else (no bridging across a gap - see its module
docstring), and a `StateFrame` sequence assembled some other way (e.g.
after manual review edits, or a partial reprocessing run) could have one -
silently merging across it would fabricate the same continuity `labeling.py`
was carefully built to avoid.

`segments_to_frames` is the inverse: given a segment list and the
(frame_idx, t_sec) pairs to reconstruct, it looks up which segment's
`[start_s, end_s)` window each `t_sec` falls in. `confidence` is always
`None` on the reconstruction - `StateSegment` (matching segments.csv's
schema) carries no per-frame confidence column, so it is genuinely lost
crossing this boundary, never fabricated back. Its documented "non-decreasing
`t_sec`" input precondition is validated explicitly (raises `ValueError`,
"reject or flag" - Phase 7 code review, MEDIUM) rather than silently
producing misattributed frames on violation. Note this doesn't fully resolve
an exact-tie `t_sec` sitting precisely on a segment boundary (segments.csv's
schema stores only float second boundaries, not a `frame_idx` grid, so two
frames sharing one `t_sec` value exactly at a state transition are
inherently unresolvable from segments alone) - not reachable from the
current producer's strictly-increasing `frame_idx`-derived timestamps, but
worth this explicit caveat rather than a false sense of full correctness.
"""

from __future__ import annotations

import statistics

from prepds.models import BehaviorState, StateFrame, StateSegment


class DeadMonotonicityError(ValueError):
    """A non-Dead frame follows a Dead frame in a StateFrame sequence - FR-007a."""


def frames_to_segments(state_frames: list[StateFrame]) -> list[StateSegment]:
    _validate_dead_monotonicity(state_frames)

    n = len(state_frames)
    if n == 0:
        return []

    frame_duration_s = _median_frame_duration(state_frames)

    boundaries = [0]
    for i in range(1, n):
        if (
            state_frames[i].state != state_frames[i - 1].state
            or state_frames[i].source != state_frames[i - 1].source
            or state_frames[i].frame_idx != state_frames[i - 1].frame_idx + 1
        ):
            boundaries.append(i)
    boundaries.append(n)

    segments = []
    for seg_i in range(len(boundaries) - 1):
        start_idx = boundaries[seg_i]
        end_idx = boundaries[seg_i + 1]  # exclusive
        start_s = state_frames[start_idx].t_sec
        end_s = state_frames[end_idx].t_sec if end_idx < n else state_frames[end_idx - 1].t_sec + frame_duration_s
        segments.append(
            StateSegment(
                start_s=start_s,
                end_s=end_s,
                duration_s=end_s - start_s,
                state=state_frames[start_idx].state,
                source=state_frames[start_idx].source,
            )
        )
    return segments


def segments_to_frames(
    segments: list[StateSegment], frame_idxs: list[int], t_secs: list[float]
) -> list[StateFrame]:
    """`frame_idxs`/`t_secs` must be given in non-decreasing `t_sec` order
    (the same order `frames_to_segments` originally consumed them in) -
    validated explicitly, raises `ValueError` on violation rather than
    silently misattributing frames (see module docstring's caveat on the
    exact-tie-at-a-boundary case this still can't fully resolve).
    """
    if not frame_idxs:
        return []
    if not segments:
        raise ValueError("segments_to_frames: no segments given for a non-empty frame list")
    for i in range(1, len(t_secs)):
        if t_secs[i] < t_secs[i - 1]:
            raise ValueError(
                f"segments_to_frames: t_secs must be non-decreasing, got {t_secs[i - 1]} then "
                f"{t_secs[i]} at position {i}"
            )

    frames = []
    seg_i = 0
    for frame_idx, t_sec in zip(frame_idxs, t_secs):
        # '>=' is correct for the normal case: a segment's end_s IS the next
        # segment's start_s by construction (frames_to_segments), so the
        # first frame of segment k+1 always has t_sec == segments[k].end_s
        # and must advance here. This only under/over-attributes in the
        # inherently-unresolvable case of two DIFFERENT frames sharing the
        # exact same t_sec straddling a boundary - see module docstring.
        while seg_i < len(segments) - 1 and t_sec >= segments[seg_i].end_s:
            seg_i += 1
        segment = segments[seg_i]
        frames.append(
            StateFrame(
                frame_idx=frame_idx,
                t_sec=t_sec,
                state=segment.state,
                source=segment.source,
                confidence=None,
            )
        )
    return frames


def _median_frame_duration(state_frames: list[StateFrame]) -> float:
    if len(state_frames) < 2:
        return 0.0
    diffs = [state_frames[i].t_sec - state_frames[i - 1].t_sec for i in range(1, len(state_frames))]
    return statistics.median(diffs)


def _validate_dead_monotonicity(state_frames: list[StateFrame]) -> None:
    dead_onset_idx: int | None = None
    for sf in state_frames:
        if sf.state == BehaviorState.DEAD:
            if dead_onset_idx is None:
                dead_onset_idx = sf.frame_idx
        elif dead_onset_idx is not None:
            raise DeadMonotonicityError(
                f"frame_idx={sf.frame_idx} has state={sf.state.value}, but Dead began at "
                f"frame_idx={dead_onset_idx} - FR-007a requires Dead to be terminal and "
                "monotonic once observed; this is a data-quality error to reject or flag, "
                "never silently repaired."
            )
