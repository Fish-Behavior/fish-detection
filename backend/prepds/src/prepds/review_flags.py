"""Reviewer flags computed from tracks.

The pipeline cannot tell Dead from Listing/LORR or deep sedation: a motionless fish is absorbed by the
background model, so both look like "nothing moves to the end" (docs/progress.md, Dead investigation;
PRD FR-015a). Instead of guessing, it flags the terminal no-action stretch so the reviewer confirms it.

"Movement" is a detected frame with a real smoothed speed above the freeze floor. Undetected frames and
frames with no computable speed are neither movement nor stillness evidence, so a tail of tracking
dropouts is flagged the same as a tail of detected-but-still frames.
"""

from __future__ import annotations

from prepds.features import DEFAULT_SPEED_LAG_S, compute_smoothed_speed
from prepds.models import ReviewFlag, Track

DEFAULT_MIN_TERMINAL_NO_ACTION_S = 120.0
TERMINAL_NO_ACTION = "terminal_no_action"


def terminal_no_action_flag(
    tracks: list[Track],
    *,
    freeze_speed_floor_px_per_s: float,
    speed_lag_s: float = DEFAULT_SPEED_LAG_S,
    min_no_action_s: float = DEFAULT_MIN_TERMINAL_NO_ACTION_S,
) -> ReviewFlag | None:
    """A flag when no movement is seen from some time until the end of the video, for >= `min_no_action_s`."""
    if not tracks:
        return None
    speeds, valid = compute_smoothed_speed(tracks, lag_s=speed_lag_s)
    last_movement = next(
        (i for i in range(len(tracks) - 1, -1, -1) if valid[i] and speeds[i] > freeze_speed_floor_px_per_s), None
    )
    start_s = 0.0 if last_movement is None else tracks[last_movement].t_sec
    end_s = tracks[-1].t_sec
    if end_s - start_s < min_no_action_s:
        return None
    return ReviewFlag(
        kind=TERMINAL_NO_ACTION,
        start_s=start_s,
        end_s=end_s,
        message=(
            f"No movement detected from {_mmss(start_s)} to the end of the video ({end_s - start_s:.0f} s). "
            "The pipeline cannot tell Dead from Listing/LORR or deep sedation here: check the video and label this stretch."
        ),
    )


def _mmss(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 60:02d}:{whole % 60:02d}"
