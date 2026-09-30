"""Per-frame kinematic feature derivation from tracking output (PRD §9.5.4) - FR-006.

Every kinematic field (`velocity`, `acceleration`, `angular_velocity`,
`meander`) uses the same `0.0`-sentinel-on-not-computable convention as
`Track.x`/`Track.y` (§3.3 - never fabricate a value from missing data). A
value is only ever real when enough *consecutive, detected* history exists
behind it - see `tests/test_features.py`'s module docstring for the exact
history-depth rule per field, and `FeatureFrame`'s docstring in models.py for
why downstream code must gate on `detected`, not on a feature being `0.0`.

**Provisional, uncalibrated constants** (same status as tracking.py's
`DEFAULT_*` values): `DEFAULT_IMMOBILITY_VELOCITY_FLOOR_PX_PER_S`,
`DEFAULT_IMMOBILITY_WINDOW_FRAMES`, `DEFAULT_MEANDER_WINDOW_FRAMES` are
starting points pending Phase 6's calibration search (T045), not finished
thresholds. `velocity`/`acceleration` are in px/s (time-normalized via each
Track's own `t_sec`, so already fps-independent), but the two window sizes
are frame counts, not durations - the same fps-normalization gap already
flagged for `tracking.py`'s `DEFAULT_MAX_CONSECUTIVE_MISSES_BEFORE_REACQUIRE`
(see docs/progress.md Phase 4/5) - and neither velocity's pixel units nor the
windows are resolution-normalized (same open risk as Phase 4's contour-area/
jump-distance constants). All deferred to the same Phase 6/T045 pass.

`_compute_meanders` is O(n x meander_window_frames) (a fresh window sum per
frame, not a true sliding-window accumulator). Measured directly at 36,000
synthetic frames with the default window: ~0.12s - negligible next to video
decode/tracking cost even across hundreds of videos, so left as-is. Revisit
with a prefix-sum/sliding accumulator only if Phase 12's actual batch timing
ever shows this mattering.

**`compute_feature_validity`** is the public companion to `derive_features`:
per-index validity masks for `velocity`/`angular_velocity`/`meander`, since
`FeatureFrame` itself (matching `frames.parquet`'s schema) has no room for a
per-field validity flag and a `0.0` is otherwise indistinguishable from a
real near-zero measurement. Added after a Phase 7 code review caught
`labeling.py` re-deriving a *partial* copy of this same validity logic
locally (missing the `dt > 0` guard `_compute_velocities`/
`_compute_angular_velocities` already had, and missing `_compute_meanders`'s
gap-clamp for a rolling window) - reintroducing, in a second module, exactly
the sentinel-treated-as-real bug class Phase 5's own review caught and fixed
here. Any future consumer that needs to know which `FeatureFrame` values are
real should call this instead of re-deriving validity from scratch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from prepds.models import FeatureFrame, Track

DEFAULT_IMMOBILITY_VELOCITY_FLOOR_PX_PER_S = 5.0
DEFAULT_IMMOBILITY_WINDOW_FRAMES = 30
DEFAULT_MEANDER_WINDOW_FRAMES = 30


@dataclass(frozen=True)
class FeatureValidity:
    """Per-index `True`/`False` for whether `FeatureFrame.velocity` /
    `.angular_velocity` / `.meander` at that index is a REAL measurement, as
    opposed to the not-computable `0.0` sentinel. Not part of `frames.parquet`'s
    schema (no `to_dict`/`from_dict` - never persisted), purely an in-process
    companion to `derive_features`'s output for consumers that must tell the
    two cases apart (see module docstring).
    """

    velocity: list[bool]
    angular_velocity: list[bool]
    meander: list[bool]


def derive_features(
    tracks: list[Track],
    *,
    immobility_velocity_floor_px_per_s: float = DEFAULT_IMMOBILITY_VELOCITY_FLOOR_PX_PER_S,
    immobility_window_frames: int = DEFAULT_IMMOBILITY_WINDOW_FRAMES,
    meander_window_frames: int = DEFAULT_MEANDER_WINDOW_FRAMES,
) -> list[FeatureFrame]:
    """Derive one `FeatureFrame` per input `Track`, in the same order.

    Never raises on bad tracking data - an undetected frame, a missing
    `orientation_deg`, or a frame_idx gap all just mean less history is
    available, producing the `0.0` sentinel where a real value can't be
    computed (see module docstring).
    """
    if not tracks:
        return []

    run_length = consecutive_detected_run_length(tracks)
    velocities, velocity_valid = _compute_velocities(tracks, run_length)
    accelerations = _compute_accelerations(tracks, velocities, velocity_valid)
    angular_velocities, _ = _compute_angular_velocities(tracks)
    meanders = _compute_meanders(tracks, run_length, meander_window_frames)
    immobility = _compute_immobility(
        velocities, velocity_valid, immobility_velocity_floor_px_per_s, immobility_window_frames
    )

    return [
        FeatureFrame(
            frame_idx=tracks[i].frame_idx,
            velocity=velocities[i],
            acceleration=accelerations[i],
            angular_velocity=angular_velocities[i],
            meander=meanders[i],
            is_immobile=immobility[i],
        )
        for i in range(len(tracks))
    ]


def compute_feature_validity(tracks: list[Track]) -> FeatureValidity:
    """Companion to `derive_features` - see `FeatureValidity`'s docstring."""
    if not tracks:
        return FeatureValidity(velocity=[], angular_velocity=[], meander=[])

    run_length = consecutive_detected_run_length(tracks)
    _, velocity_valid = _compute_velocities(tracks, run_length)
    _, angular_velocity_valid = _compute_angular_velocities(tracks)
    meander_valid = [run_length[i] >= 2 for i in range(len(tracks))]
    return FeatureValidity(velocity=velocity_valid, angular_velocity=angular_velocity_valid, meander=meander_valid)


def _circular_diff(a: float, b: float, *, period: float) -> float:
    """Shortest signed difference `a - b` on a circle of the given period.

    Result is in `[-period/2, period/2)` (Python's `%` on floats returns a
    value in `[0, period)`, so the seam lands on the negative side). Used
    for both body-orientation (`period=180` - `cv2.fitEllipse`'s angle has
    no head/tail distinction, verified empirically) and trajectory heading
    (`period=360` - a genuine direction, "left" and "right" are different
    headings). Exactly half a period apart is a genuine ambiguity (+period/2
    and -period/2 are the same rotation) - only the magnitude is meaningful
    there, not the sign this function happens to return.
    """
    return (a - b + period / 2) % period - period / 2


def consecutive_detected_run_length(tracks: list[Track]) -> list[int]:
    """`run_length[i]` = length of the run of detected, frame_idx-adjacent frames ending at `i` (inclusive)."""
    run_length = [0] * len(tracks)
    for i, track in enumerate(tracks):
        if not track.detected:
            run_length[i] = 0
            continue
        if i > 0 and tracks[i - 1].detected and tracks[i].frame_idx == tracks[i - 1].frame_idx + 1:
            run_length[i] = run_length[i - 1] + 1
        else:
            run_length[i] = 1
    return run_length


DEFAULT_SPEED_LAG_S = 1.0


def compute_smoothed_speed(tracks: list[Track], *, lag_s: float = DEFAULT_SPEED_LAG_S) -> tuple[list[float], list[bool]]:
    """Speed (px/s) as displacement between a detected frame and its partner `lag_s` away, and validity.

    Raw frame-to-frame velocity is dominated by centroid jitter from the speckled foreground mask
    (real footage: tens to hundreds of px/s for a fish that is barely moving), so behavior rules
    that need "how fast is the fish really moving" use displacement over a lag instead. The partner
    is the frame `lag` frames earlier if it is detected and exactly `lag` frame indices away, else
    the one `lag` frames later; a frame with neither is not computable (`0.0`, `valid=False`). Only the
    two endpoints must be real detections - a gap shorter than the lag between them does not stop the
    displacement being measured, but a gap that leaves no detected partner (or a frame_idx
    discontinuity) is never bridged. `lag` is `lag_s` over the median
    consecutive-frame spacing, so it is fps-independent.

    Known limits of a displacement measure: the backward partner is preferred, so a bout's onset/offset
    is smeared by up to `lag_s`; and net displacement underestimates the speed of tight circling or
    back-and-forth motion, which can therefore read as slow.
    """
    n = len(tracks)
    speeds = [0.0] * n
    valid = [False] * n
    spacings = sorted(
        tracks[i].t_sec - tracks[i - 1].t_sec
        for i in range(1, n)
        if tracks[i].frame_idx == tracks[i - 1].frame_idx + 1 and tracks[i].t_sec > tracks[i - 1].t_sec
    )
    if not spacings:
        return speeds, valid
    lag = max(1, round(lag_s / spacings[len(spacings) // 2]))

    for i, track in enumerate(tracks):
        if not track.detected:
            continue
        for j in (i - lag, i + lag):
            if not 0 <= j < n or not tracks[j].detected:
                continue
            first, second = (tracks[j], track) if j < i else (track, tracks[j])
            elapsed = second.t_sec - first.t_sec
            if second.frame_idx - first.frame_idx != lag or elapsed <= 0:
                continue
            speeds[i] = math.hypot(second.x - first.x, second.y - first.y) / elapsed
            valid[i] = True
            break
    return speeds, valid


def _compute_velocities(tracks: list[Track], run_length: list[int]) -> tuple[list[float], list[bool]]:
    """Return `(velocities, velocity_valid)` - `run_length[i] >= 2` is necessary but not
    sufficient for "`velocities[i]` is a real measurement": two consecutive detected,
    frame_idx-adjacent frames can still share a non-increasing `t_sec` (defensive
    guard, not expected from `video_io` in practice), which also yields the `0.0`
    sentinel. Downstream consumers that need to know *which* zeros are real must use
    `velocity_valid`, not re-derive it from `run_length` alone (code review finding -
    both `_compute_accelerations` and `_compute_immobility` originally did exactly
    that and silently treated a `dt<=0` sentinel as a real below-floor/prior-velocity
    sample).
    """
    velocities = [0.0] * len(tracks)
    velocity_valid = [False] * len(tracks)
    for i in range(len(tracks)):
        if run_length[i] < 2:
            continue
        prev, cur = tracks[i - 1], tracks[i]
        dt = cur.t_sec - prev.t_sec
        if dt <= 0:
            continue
        velocities[i] = math.hypot(cur.x - prev.x, cur.y - prev.y) / dt
        velocity_valid[i] = True
    return velocities, velocity_valid


def _compute_accelerations(
    tracks: list[Track], velocities: list[float], velocity_valid: list[bool]
) -> list[float]:
    accelerations = [0.0] * len(tracks)
    for i in range(len(tracks)):
        if i == 0 or not velocity_valid[i] or not velocity_valid[i - 1]:
            continue
        # velocity_valid[i] already guarantees tracks[i].t_sec - tracks[i-1].t_sec
        # > 0 (same computation as in _compute_velocities) - no separate dt
        # guard needed here.
        dt = tracks[i].t_sec - tracks[i - 1].t_sec
        accelerations[i] = (velocities[i] - velocities[i - 1]) / dt
    return accelerations


def _compute_angular_velocities(tracks: list[Track]) -> tuple[list[float], list[bool]]:
    """Return `(angular_velocities, angular_velocity_valid)` - same real-vs-sentinel
    split as `_compute_velocities` (see its docstring for why consumers must use
    the validity mask, not re-derive it from a `0.0` value)."""
    n = len(tracks)
    angular_velocities = [0.0] * n
    angular_velocity_valid = [False] * n
    orientation_run_length = 0
    for i, track in enumerate(tracks):
        has_orientation = track.detected and track.orientation_deg is not None
        if not has_orientation:
            orientation_run_length = 0
            continue
        if (
            i > 0
            and orientation_run_length > 0
            and tracks[i].frame_idx == tracks[i - 1].frame_idx + 1
        ):
            orientation_run_length += 1
        else:
            orientation_run_length = 1

        if orientation_run_length >= 2:
            dt = track.t_sec - tracks[i - 1].t_sec
            if dt > 0:
                diff = _circular_diff(track.orientation_deg, tracks[i - 1].orientation_deg, period=180.0)
                angular_velocities[i] = diff / dt
                angular_velocity_valid[i] = True
    return angular_velocities, angular_velocity_valid


def _compute_meanders(tracks: list[Track], run_length: list[int], window_frames: int) -> list[float]:
    """Trajectory tortuosity: sum of |turning angle| / sum of path length, over a rolling window.

    Not body-orientation curvature - see module docstring for why heading
    (position-derived) is used instead of `orientation_deg`.

    The window is clamped to `[i - run_length[i] + 1, i]` in addition to the
    plain `window_frames` bound (code review finding, CRITICAL): without
    this, the window was indexed by list position only, so a detection gap
    didn't stop it from summing turn/distance values left over from before
    the gap - an undetected frame could report a stale nonzero value instead
    of the documented `0.0` sentinel, and a frame soon after reacquisition
    could silently blend pre-gap and post-gap trajectory into one ratio.
    Clamping to the current run makes the window empty (and therefore
    `meanders[i] == 0.0` via the zero-distance guard below) for any frame
    with `run_length[i] < 2`, without a separate special case.
    """
    n = len(tracks)
    step_dx = [0.0] * n
    step_dy = [0.0] * n
    step_dist = [0.0] * n
    step_valid = [False] * n
    heading = [0.0] * n
    for i in range(n):
        if run_length[i] < 2:
            continue
        prev, cur = tracks[i - 1], tracks[i]
        dx, dy = cur.x - prev.x, cur.y - prev.y
        step_dx[i], step_dy[i] = dx, dy
        step_dist[i] = math.hypot(dx, dy)
        step_valid[i] = True
        heading[i] = math.degrees(math.atan2(dy, dx))

    turn = [0.0] * n
    turn_valid = [False] * n
    for i in range(1, n):
        if step_valid[i] and step_valid[i - 1] and run_length[i] >= 3:
            turn[i] = abs(_circular_diff(heading[i], heading[i - 1], period=360.0))
            turn_valid[i] = True

    meanders = [0.0] * n
    for i in range(n):
        lo = max(0, i - window_frames + 1, i - run_length[i] + 1)
        total_turn = sum(turn[k] for k in range(lo, i + 1) if turn_valid[k])
        total_dist = sum(step_dist[k] for k in range(lo, i + 1) if step_valid[k])
        meanders[i] = total_turn / total_dist if total_dist > 1e-9 else 0.0
    return meanders


def _compute_immobility(
    velocities: list[float],
    velocity_valid: list[bool],
    velocity_floor_px_per_s: float,
    window_frames: int,
) -> list[bool]:
    """`True` only when `window_frames` consecutive REAL velocities are all below the floor.

    Fails open to `False` (not confirmed immobile) if the window isn't fully
    backed by real, gated velocity values (e.g. an undetected frame, or a
    `dt<=0` step, inside it) - never infers sustained stillness across an
    unknown stretch. Checking `velocity_valid` directly (rather than
    re-deriving "is this a real velocity" from `run_length >= 2`, code
    review finding) matters because `run_length[i] >= 2` is necessary but
    not sufficient - a `dt<=0` step also produces a `0.0` velocity sentinel
    that `run_length` alone can't distinguish from a genuine below-floor
    measurement.
    """
    immobility = [False] * len(velocities)
    for i in range(len(velocities)):
        if i < window_frames - 1:
            continue
        lo = i - window_frames + 1
        if not all(velocity_valid[lo : i + 1]):
            continue
        immobility[i] = all(v <= velocity_floor_px_per_s for v in velocities[lo : i + 1])
    return immobility
