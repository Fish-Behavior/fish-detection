"""Temporal consolidation of per-frame labels into reviewable, reference-resolution bins.

Per-frame labels flicker on real footage: the tracker's detection toggles thousands of times per video
(F_332: ~5,000 on/off transitions, ~10,000 segments, median 2 frames), and the reference figures are drawn
at ~1 s resolution. This runs between `labeling.classify_video` and `segments.frames_to_segments`:

- each fixed time bin (default 1 s, by `t_sec`, so frame_idx gaps never merge bins) takes the majority of
  its DETERMINED frames - Undetermined abstains instead of voting;
- a bin is Undetermined only when fewer than `min_determined_fraction` of its frames are determined, so
  a bin is never labeled from a handful of stray frames;
- any Surface Breach frame keeps its bin (a one-frame spike is the definition of that event) unless Dead
  holds at least half of the bin's determined frames;
- ties break by fixed precedence (Dead, Listing, Freezing, Erratic, Controlled) for determinism.

This does not fabricate long bouts across dropouts: it never crosses a bin boundary, so a stretch is at
most as long as the frames that actually voted for it. Dead stays a monotonic suffix (every bin after the
onset bin is entirely Dead). Bin size and fraction are structural (they match the reference figure
resolution), not calibrated thresholds.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from prepds.models import BehaviorState, FrameSource, StateFrame

DEFAULT_BIN_S = 1.0
DEFAULT_MIN_DETERMINED_FRACTION = 0.2

_STATES = tuple(BehaviorState)
_INDEX = {state: i for i, state in enumerate(_STATES)}
# Higher = wins ties. Surface Breach and Undetermined are resolved before the majority vote.
_TIE_PRIORITY = {
    BehaviorState.DEAD: 4,
    BehaviorState.LISTING_LORR: 3,
    BehaviorState.FREEZING_DRIFT: 2,
    BehaviorState.ERRATIC_MOVEMENT: 1,
    BehaviorState.CONTROLLED_SWIM: 0,
}
_EPSILON = 1e-9


def consolidate_states(
    frames: Sequence[StateFrame],
    *,
    bin_s: float = DEFAULT_BIN_S,
    min_determined_fraction: float = DEFAULT_MIN_DETERMINED_FRACTION,
) -> list[StateFrame]:
    """One consolidated `StateFrame` per input frame (same frame_idx/t_sec/order, source AUTO, no confidence)."""
    states = consolidate_labels(
        [f.state for f in frames],
        [f.t_sec for f in frames],
        bin_s=bin_s,
        min_determined_fraction=min_determined_fraction,
    )
    return [
        StateFrame(frame_idx=f.frame_idx, t_sec=f.t_sec, state=state, source=FrameSource.AUTO, confidence=None)
        for f, state in zip(frames, states)
    ]


def consolidate_labels(
    states: Sequence[BehaviorState],
    t_secs: Sequence[float],
    *,
    bin_s: float = DEFAULT_BIN_S,
    min_determined_fraction: float = DEFAULT_MIN_DETERMINED_FRACTION,
) -> list[BehaviorState]:
    """The consolidation on bare labels (vectorised; calibration calls it once per search sample)."""
    if bin_s <= 0:
        raise ValueError(f"bin_s must be positive, got {bin_s}")
    if not 0.0 <= min_determined_fraction <= 1.0:
        raise ValueError(f"min_determined_fraction must be in [0, 1], got {min_determined_fraction}")
    if not states:
        return []

    codes = np.fromiter((_INDEX[s] for s in states), dtype=np.int64, count=len(states))
    bins = np.floor(np.asarray(t_secs, dtype=float) / bin_s + _EPSILON).astype(np.int64)
    _, bin_of_frame = np.unique(bins, return_inverse=True)
    n_bins = int(bin_of_frame.max()) + 1
    counts = np.zeros((n_bins, len(_STATES)), dtype=np.int64)
    np.add.at(counts, (bin_of_frame, codes), 1)

    undetermined = _INDEX[BehaviorState.UNDETERMINED]
    surface, dead = _INDEX[BehaviorState.SURFACE_BREACH], _INDEX[BehaviorState.DEAD]
    total = counts.sum(axis=1)
    determined = total - counts[:, undetermined]
    # Surface Breach keeps its bin unless Dead holds at least half of the determined frames: a lone Dead
    # frame must not turn a bin of Surface Breach frames into Dead.
    surface_wins = (counts[:, surface] > 0) & (counts[:, dead] * 2 < determined)

    votes = counts.copy()
    votes[:, undetermined] = 0
    votes[:, surface] = 0
    scores = votes * 10  # counts dominate; the tie-break priority (< 10) only separates equal counts
    for state, priority in _TIE_PRIORITY.items():
        scores[:, _INDEX[state]] += priority
    winner = np.where(votes.sum(axis=1) > 0, scores.argmax(axis=1), undetermined)
    winner = np.where(surface_wins, surface, winner)
    winner = np.where((determined == 0) | (determined / total < min_determined_fraction), undetermined, winner)
    return [_STATES[i] for i in winner[bin_of_frame]]
