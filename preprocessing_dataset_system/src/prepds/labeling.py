"""Per-frame behavior-state classification (PRD §9.5.6) - FR-007/FR-007a.

**Rule precedence** (PRD lists the rule families but never states an order - fixed here and
enforced by the implementation, not left to be inferred from call order):

    detected=False              -> Undetermined (absolute, overrides everything)
    > Dead                      (whole-video terminal-suffix pass, FR-007a)
    > Surface Breach            (per-frame spike, no sustained-duration gate)
    > Listing/LORR              (opt-in; sustained orientation deviation - see below)
    > Freezing/Drift            (sustained low speed)
    > Erratic Movement          (speed at or above `erratic_speed_threshold_px_per_s`)
    > Controlled Swim           (moving, below that threshold)
    > Undetermined              (detected, but no computable speed)

**Speed, not velocity.** Every speed rule uses `features.compute_smoothed_speed` (displacement
over a ~1 s lag). The per-frame `FeatureFrame.velocity` is centroid-jitter noise on real footage
(tens to hundreds of px/s for a nearly still fish), which is why the first calibration attempts
pushed the freeze floor to its search cap. Controlled vs Erratic is a calibrated speed band
rather than the earlier pooled Gaussian mixture, which only ever saw the fastest frames and whose
Controlled:Erratic ratio could not be moved by any calibration parameter that mattered (see
docs/progress.md, Phase 6/7). Frames carry no confidence value: nothing probabilistic remains.

**No gap-bridging** (deliberate). Every "sustained for >= N seconds" rule (Freezing/Drift,
Listing/LORR, Dead) measures elapsed `t_sec` over a run of CONTIGUOUS DETECTED frames only. An
undetected frame - or a frame_idx gap - resets the bout clock; it never bridges across the gap.
Fabricating continuity across a tracking dropout would turn a tracking deficiency into a claim
about a duration nobody observed. Consequence, accepted rather than worked around: on footage with
frequent gaps these rules may fire rarely or never; FR-015's reviewer-correction workflow is the
designed absorber.

**Dead (FR-007a)** is a whole-video terminal-suffix pass: the maximal contiguous run of
real, below-floor speed that reaches the LAST frame. If no such run reaches the last frame (e.g.
the tail is undetected, common on real footage) no Dead is emitted; a later fast frame rules it
out entirely. Monotonic by construction, matching the invariant `segments.py` re-validates. Dead
reuses the freeze speed floor with a much longer minimum bout.

**Listing/LORR is off by default.** Body angle from the tracker's ellipse fit does not separate
LORR on real footage (a fish rolled on its side still looks horizontal from the side, and the
measured deviation anti-correlates with the reference), and no other automatic proxy held up
(docs/progress.md). `listing_orientation_deviation_deg=None` disables the rule so Listing/LORR is
labeled manually in review; the rule stays available for opt-in experiments.

**Placeholder thresholds.** The `DEFAULT_*` constants are structurally-motivated placeholders
until a calibration profile is frozen (Phase 6, T046); they are never hand-picked to look
reasonable on footage. `DEFAULT_MIN_DEAD_BOUT_S` is deliberately an order of magnitude above the
Freezing bout to keep the PRD's structural relationship.
"""

from __future__ import annotations

from dataclasses import dataclass

from prepds.features import DEFAULT_SPEED_LAG_S, compute_smoothed_speed
from prepds.models import BehaviorState, FrameSource, StateFrame, Track

DEFAULT_FREEZE_SPEED_FLOOR_PX_PER_S = 10.0
DEFAULT_MIN_FREEZE_BOUT_S = 2.0
DEFAULT_ERRATIC_SPEED_THRESHOLD_PX_PER_S = 40.0
DEFAULT_SURFACE_BREACH_DEPTH_THRESHOLD_PX = 20.0
DEFAULT_LISTING_ORIENTATION_DEVIATION_DEG: float | None = None
DEFAULT_MIN_LISTING_BOUT_S = 2.0
DEFAULT_MIN_DEAD_BOUT_S = 60.0


@dataclass(frozen=True)
class PreparedVideo:
    """The threshold-independent work for one video (smoothed speed and its validity), done once
    so calibration re-runs only the cheap rule pass per threshold sample."""

    tracks: list[Track]
    speeds: list[float]
    speed_valid: list[bool]
    t_secs: list[float]


def prepare_video(tracks: list[Track], *, speed_lag_s: float = DEFAULT_SPEED_LAG_S) -> PreparedVideo:
    speeds, speed_valid = compute_smoothed_speed(tracks, lag_s=speed_lag_s)
    return PreparedVideo(tracks=tracks, speeds=speeds, speed_valid=speed_valid, t_secs=[t.t_sec for t in tracks])


def classify_video(
    tracks: list[Track],
    *,
    speed_lag_s: float = DEFAULT_SPEED_LAG_S,
    **thresholds: float | None,
) -> list[StateFrame]:
    """Classify every frame of one video, in precedence order (see module docstring)."""
    return classify_prepared(prepare_video(tracks, speed_lag_s=speed_lag_s), **thresholds)


def classify_prepared(prepared: PreparedVideo, **thresholds: float | None) -> list[StateFrame]:
    states = classify_prepared_states(prepared, **thresholds)
    return [
        StateFrame(
            frame_idx=track.frame_idx,
            t_sec=track.t_sec,
            state=state,
            source=FrameSource.AUTO,
            confidence=None,
        )
        for track, state in zip(prepared.tracks, states)
    ]


def classify_prepared_states(
    prepared: PreparedVideo,
    *,
    freeze_speed_floor_px_per_s: float = DEFAULT_FREEZE_SPEED_FLOOR_PX_PER_S,
    min_freeze_bout_s: float = DEFAULT_MIN_FREEZE_BOUT_S,
    erratic_speed_threshold_px_per_s: float = DEFAULT_ERRATIC_SPEED_THRESHOLD_PX_PER_S,
    surface_breach_depth_threshold_px: float = DEFAULT_SURFACE_BREACH_DEPTH_THRESHOLD_PX,
    listing_orientation_deviation_deg: float | None = DEFAULT_LISTING_ORIENTATION_DEVIATION_DEG,
    min_listing_bout_s: float = DEFAULT_MIN_LISTING_BOUT_S,
    min_dead_bout_s: float = DEFAULT_MIN_DEAD_BOUT_S,
) -> list[BehaviorState]:
    """Per-frame states for a prepared video."""
    if erratic_speed_threshold_px_per_s <= freeze_speed_floor_px_per_s:
        raise ValueError(
            f"erratic_speed_threshold_px_per_s ({erratic_speed_threshold_px_per_s}) must exceed "
            f"freeze_speed_floor_px_per_s ({freeze_speed_floor_px_per_s}); otherwise Freezing swallows "
            "the Controlled band"
        )
    tracks, speeds, speed_valid = prepared.tracks, prepared.speeds, prepared.speed_valid
    n = len(tracks)
    freeze_holds = [speed_valid[i] and speeds[i] <= freeze_speed_floor_px_per_s for i in range(n)]
    is_freezing = _sustained_run_flags(tracks, freeze_holds, min_freeze_bout_s)
    is_dead = _compute_dead_suffix(tracks, freeze_holds, min_dead_bout_s)
    if listing_orientation_deviation_deg is None:
        is_listing = [False] * n
    else:
        listing_holds = _listing_condition(tracks, listing_orientation_deviation_deg)
        is_listing = _sustained_run_flags(tracks, listing_holds, min_listing_bout_s)

    states: list[BehaviorState] = []
    for i, track in enumerate(tracks):
        if not track.detected:
            states.append(BehaviorState.UNDETERMINED)
        elif is_dead[i]:
            states.append(BehaviorState.DEAD)
        elif track.y_from_frame_top is not None and track.y_from_frame_top <= surface_breach_depth_threshold_px:
            states.append(BehaviorState.SURFACE_BREACH)
        elif is_listing[i]:
            states.append(BehaviorState.LISTING_LORR)
        elif is_freezing[i]:
            states.append(BehaviorState.FREEZING_DRIFT)
        elif not speed_valid[i]:
            states.append(BehaviorState.UNDETERMINED)
        elif speeds[i] >= erratic_speed_threshold_px_per_s:
            states.append(BehaviorState.ERRATIC_MOVEMENT)
        else:
            states.append(BehaviorState.CONTROLLED_SWIM)
    return states


def _sustained_run_flags(tracks: list[Track], condition_holds: list[bool], min_bout_s: float) -> list[bool]:
    """True for EVERY frame of a maximal run of contiguous (frame_idx-adjacent) frames where
    `condition_holds`, provided the run spans at least `min_bout_s` of `t_sec` - a qualifying
    bout is labeled in full from its first frame, not only after the minimum has elapsed.
    (An earlier per-frame "elapsed since onset >= minimum" test truncated every bout by the
    minimum duration: a 3s bout with a 2s minimum only had its last 1s labeled. Found by the
    Phase 6 calibration toy test; earlier tests only asserted the last frame.) A run that is
    broken by a gap or a `frame_idx` discontinuity never bridges - see module docstring.
    """
    n = len(tracks)
    flags = [False] * n
    i = 0
    while i < n:
        if not condition_holds[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and condition_holds[j + 1] and tracks[j + 1].frame_idx == tracks[j].frame_idx + 1:
            j += 1
        if tracks[j].t_sec - tracks[i].t_sec >= min_bout_s:
            for k in range(i, j + 1):
                flags[k] = True
        i = j + 1
    return flags


def _listing_condition(tracks: list[Track], deviation_threshold_deg: float) -> list[bool]:
    """True at `i` when the frame is detected, has a real `orientation_deg`,
    and its deviation from horizontal meets the threshold. `orientation_deg`
    is a direct per-frame measurement (not derived from history the way
    velocity is), so no cross-frame validity re-derivation is needed here -
    its own `None`-ness already is the validity signal.

    `cv2.fitEllipse`'s angle convention is `90.0` == horizontal, `0.0`
    (equivalently `~180.0`, by its period-180 wraparound) == vertical - the
    OPPOSITE of the naive "0 == horizontal" assumption (verified empirically:
    a synthetic wide/short blob fits to `90.0`, a tall/narrow one to `0.0`,
    and a blob rotated 45 deg off horizontal fits to `135.0`). An earlier
    version of this function used `min(orientation_deg, 180 - orientation_deg)`,
    which is only correct under the naive assumption - it silently inverted
    the rule (a normally-horizontal-swimming fish, `orientation_deg~=90`,
    scored *maximal* deviation, while an actually-vertical/rolled fish,
    `orientation_deg~=0`, scored *zero* deviation). Caught on real footage:
    a vehicle-control (undrugged) video showed 27% Listing/LORR, which should
    be ~0% per the reference figures (no drug panel shows meaningful pink
    outside Fentanyl) - a calibration-search red flag traced back to this
    inverted formula, not a threshold that needed tuning.
    `abs(orientation_deg - 90.0)` is correct given `fitEllipse`'s `[0, 180)`
    output range: `0.0` at `orientation_deg=90` (horizontal), rising to its
    max of `90.0` at either boundary (`orientation_deg` near `0` or near
    `180`, both "vertical").
    """
    holds = [False] * len(tracks)
    for i, track in enumerate(tracks):
        if not track.detected or track.orientation_deg is None:
            continue
        deviation = abs(track.orientation_deg - 90.0)
        holds[i] = deviation >= deviation_threshold_deg
    return holds


def _compute_dead_suffix(tracks: list[Track], freeze_holds: list[bool], min_dead_bout_s: float) -> list[bool]:
    """Whole-video terminal-suffix pass (FR-007a) - see module docstring."""
    n = len(tracks)
    is_dead = [False] * n
    if n == 0 or not freeze_holds[-1]:
        return is_dead

    i = n - 1
    while i >= 0 and freeze_holds[i] and (i == n - 1 or tracks[i + 1].frame_idx == tracks[i].frame_idx + 1):
        i -= 1
    onset_idx = i + 1

    if tracks[-1].t_sec - tracks[onset_idx].t_sec >= min_dead_bout_s:
        for k in range(onset_idx, n):
            is_dead[k] = True
    return is_dead
