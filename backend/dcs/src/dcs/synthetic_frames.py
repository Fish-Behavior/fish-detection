"""One synthetic fish: frames.parquet rows and their segments.csv (plan U2).

Values are plausible, not realistic: pixel speeds per state, a position that
follows the state (surface near the top, freezing near the bottom), and the
prepds conventions later units depend on (schema.py): 0.0 / null sentinels on
undetected frames, 0.0 kinematics where a detected run has too little history,
and no confidence on any frame. All random draws happen up front in a fixed
order, so a knob changes only the fish it is applied to.

Simplification: undetected frames keep the state of the bout around them. In
prepds they are Undetermined until consolidation or review relabels them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dcs import schema

# Typical speed per state in pixels per second (classical tracker scale, 304x240 frames).
BASE_SPEED_PX_S = {
    schema.CONTROLLED_SWIM: 20.0,
    schema.ERRATIC_MOVEMENT: 45.0,
    schema.FREEZING_DRIFT: 1.5,
    schema.SURFACE_BREACH: 15.0,
    schema.LISTING_LORR: 3.0,
    schema.UNDETERMINED: 10.0,
}
# States a fish draws its bouts from; Listing/LORR, Dead and Undetermined appear only through knobs.
BASE_STATE_WEIGHTS = {
    schema.CONTROLLED_SWIM: 0.6,
    schema.ERRATIC_MOVEMENT: 0.15,
    schema.FREEZING_DRIFT: 0.15,
    schema.SURFACE_BREACH: 0.1,
}
# Pixel row (from the frame top) a fish drifts towards in each state; others use the middle.
TARGET_ROW_PX = {schema.SURFACE_BREACH: 10.0, schema.FREEZING_DRIFT: 200.0}
MIDDLE_ROW_PX = 120.0
FRAME_WIDTH_PX = 304
FRAME_HEIGHT_PX = 240
SWIM_AMPLITUDE_PX = 100.0  # side-to-side swim around the frame centre
SWIM_RATE_RAD_S = 0.3
POSITION_NOISE_PX = 5.0
ROW_NOISE_PX = 20.0

MEAN_BOUT_S = 1.0
FISH_SPEED_NOISE = 0.1  # sigma of the per-fish lognormal speed factor
FRAME_SPEED_NOISE = 0.25  # sigma of the per-frame lognormal speed factor
TURN_SD_DEG_S = 40.0
ERRATIC_TURN_FACTOR = 2.0
# prepds immobility rule (features.py defaults): the last 30 frames all have a real velocity at or
# below 5 px/s. Calibrated in prepds, so a real gold set may use other values.
IMMOBILITY_WINDOW_FRAMES = 30
IMMOBILITY_FLOOR_PX_S = 5.0
MIN_SPEED_FOR_MEANDER = 1.0

EDIT_WINDOW_SHARE = 0.1  # share of frames a reviewer relabelled in an edited fish
RARE_STATE_WINDOW_SHARE = 0.2  # share of frames shown as Listing/LORR by a rare-state fish
UNDETERMINED_WINDOW_SHARE = 0.05


@dataclass(frozen=True)
class FishBehavior:
    """Everything that shapes one fish's frames; built by dcs.synthetic from the design and the knobs."""

    fps: float
    n_frames: int
    speed_multiplier: float  # compound x dose x date effect
    preferred_state: str | None  # state the compound makes more frequent (None for vehicle)
    preference_boost: float  # weight multiplier minus 1 for the preferred state
    depth_offset_px: float  # camera framing shift of the fish's date
    miss_rate: float  # share of undetected frames
    manual_fraction: float  # chance that a reviewer edited this fish
    rare_state: bool = False
    undetermined: bool = False
    constant_meander: bool = False


@dataclass(frozen=True)
class _Draws:
    """Every random number one fish uses, drawn in one fixed order."""

    edited: bool
    edit_start: int
    fish_factor: float
    states: np.ndarray
    speed_noise: np.ndarray
    turn: np.ndarray
    row_noise: np.ndarray
    phase: float
    position_noise: np.ndarray
    detected: np.ndarray


def make_frames(behavior: FishBehavior, rng: np.random.Generator) -> pd.DataFrame:
    """frames.parquet rows for one fish, with the prepds column order and dtypes."""
    draws = _draw(behavior, rng)
    states = _apply_state_knobs(draws.states, behavior)
    t_sec = np.arange(behavior.n_frames) / behavior.fps
    speed = (
        np.array([BASE_SPEED_PX_S[state] for state in states])
        * behavior.speed_multiplier
        * draws.fish_factor
        * draws.speed_noise
    )
    turn = draws.turn * np.where(states == schema.ERRATIC_MOVEMENT, ERRATIC_TURN_FACTOR, 1.0)
    target_row = np.array([TARGET_ROW_PX.get(state, MIDDLE_ROW_PX) for state in states])
    y = np.clip(target_row + draws.row_noise, 0.0, FRAME_HEIGHT_PX - 1)
    sway = SWIM_AMPLITUDE_PX * np.sin(SWIM_RATE_RAD_S * t_sec + draws.phase)
    x = np.clip(FRAME_WIDTH_PX / 2 + sway + draws.position_noise, 0.0, FRAME_WIDTH_PX - 1)
    meander = np.zeros_like(speed) if behavior.constant_meander else np.abs(turn) / np.maximum(speed, MIN_SPEED_FOR_MEANDER)
    columns = {
        "frame_idx": np.arange(behavior.n_frames),
        "t_sec": t_sec,
        "x": x,
        "y": y,
        "orientation_deg": np.mod(np.cumsum(turn) / behavior.fps, 360.0),
        "depth_from_surface": y + behavior.depth_offset_px,
        "detected": draws.detected,
        "velocity": speed,
        "acceleration": np.diff(speed, prepend=speed[0]) * behavior.fps,
        "angular_velocity": turn,
        "meander": meander,
        "is_immobile": np.zeros(behavior.n_frames, dtype=bool),  # set by _apply_prepds_conventions
        "state": states,
        "source": np.where(_edit_mask(behavior.n_frames, draws), schema.SOURCE_MANUAL, schema.SOURCE_AUTO),
        "confidence": np.full(behavior.n_frames, np.nan),  # prepds labels carry no confidence
    }
    return _typed_frame(_apply_prepds_conventions(columns))


def frames_to_segments(frames: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Run-length encoding of (state, source), as prepds writes segments.csv: a segment ends where the next
    one starts, and the last one ends one frame after the last t_sec."""
    key = frames["state"].astype(str) + "|" + frames["source"].astype(str)
    starts = np.flatnonzero(key.ne(key.shift()).to_numpy())
    t_sec = frames["t_sec"].to_numpy(dtype=float)
    ends = [t_sec[stop] if stop < len(frames) else t_sec[-1] + 1.0 / fps for stop in [*starts[1:], len(frames)]]
    start_s = t_sec[starts]
    return pd.DataFrame(
        {
            "start_s": start_s,
            "end_s": ends,
            "duration_s": np.asarray(ends) - start_s,
            "state": frames["state"].astype(str).to_numpy()[starts],
            "source": frames["source"].astype(str).to_numpy()[starts],
        },
        columns=list(schema.SEGMENT_COLUMNS),
    )


def _draw(behavior: FishBehavior, rng: np.random.Generator) -> _Draws:
    n = behavior.n_frames
    edited = bool(rng.random() < behavior.manual_fraction)
    edit_start = int(rng.integers(0, max(1, n - _edit_length(n))))
    fish_factor = float(rng.lognormal(0.0, FISH_SPEED_NOISE))
    states = _state_sequence(behavior, rng)
    return _Draws(
        edited=edited,
        edit_start=edit_start,
        fish_factor=fish_factor,
        states=states,
        speed_noise=rng.lognormal(0.0, FRAME_SPEED_NOISE, n),
        turn=rng.normal(0.0, TURN_SD_DEG_S, n),
        row_noise=rng.normal(0.0, ROW_NOISE_PX, n),
        phase=float(rng.uniform(0.0, 2 * np.pi)),
        position_noise=rng.normal(0.0, POSITION_NOISE_PX, n),
        detected=rng.random(n) >= behavior.miss_rate,
    )


def _state_sequence(behavior: FishBehavior, rng: np.random.Generator) -> np.ndarray:
    names = list(BASE_STATE_WEIGHTS)
    weights = np.array([BASE_STATE_WEIGHTS[name] for name in names])
    if behavior.preferred_state is not None:
        weights = np.where(np.array(names) == behavior.preferred_state, weights * (1 + behavior.preference_boost), weights)
    weights = weights / weights.sum()
    states = np.empty(behavior.n_frames, dtype=object)
    filled = 0
    while filled < behavior.n_frames:
        length = 1 + int(rng.exponential(MEAN_BOUT_S * behavior.fps))
        states[filled : filled + length] = names[int(rng.choice(len(names), p=weights))]
        filled += length
    return states


def _apply_state_knobs(states: np.ndarray, behavior: FishBehavior) -> np.ndarray:
    """Overwrite fixed windows for the rare-state and Undetermined knobs (no random draws)."""
    n = len(states)
    result = states.copy()
    if behavior.rare_state:
        start = n // 3
        result[start : start + max(1, int(n * RARE_STATE_WINDOW_SHARE))] = schema.LISTING_LORR
    if behavior.undetermined:
        start = n // 2
        result[start : start + max(1, int(n * UNDETERMINED_WINDOW_SHARE))] = schema.UNDETERMINED
    return result


def _edit_length(n_frames: int) -> int:
    return max(1, int(n_frames * EDIT_WINDOW_SHARE))


def _edit_mask(n_frames: int, draws: _Draws) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    if draws.edited:
        mask[draws.edit_start : draws.edit_start + _edit_length(n_frames)] = True
    return mask


def _apply_prepds_conventions(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return new columns with the prepds rules for missing values applied (inputs are not changed)."""
    detected = columns["detected"]
    run = _detected_run_length(detected)
    result = dict(columns)
    for name in schema.ZERO_ON_FIRST_FRAME:
        result[name] = np.where(run == 1, 0.0, result[name])
    for name in schema.ZERO_ON_FIRST_TWO_FRAMES:
        result[name] = np.where((run == 1) | (run == 2), 0.0, result[name])
    for name in schema.ZERO_ON_UNDETECTED:
        result[name] = np.where(detected, result[name], 0.0)
    for name in schema.NULL_ON_UNDETECTED:
        result[name] = np.where(detected, result[name], np.nan)
    result["is_immobile"] = _immobile(result["velocity"], run)
    return result


def _immobile(velocity: np.ndarray, run: np.ndarray) -> np.ndarray:
    """True where the last IMMOBILITY_WINDOW_FRAMES frames all have a real velocity (run length >= 2)
    at or below IMMOBILITY_FLOOR_PX_S, as prepds computes it."""
    slow = ((run >= 2) & (velocity <= IMMOBILITY_FLOOR_PX_S)).astype(int)
    slow_in_window = np.convolve(slow, np.ones(IMMOBILITY_WINDOW_FRAMES, dtype=int))[: len(slow)]
    return slow_in_window == IMMOBILITY_WINDOW_FRAMES


def _detected_run_length(detected: np.ndarray) -> np.ndarray:
    """1 on the first frame of each detected run, 2 on the second, ...; 0 on undetected frames."""
    positions = np.arange(len(detected))
    last_gap = np.maximum.accumulate(np.where(detected, -1, positions))
    return np.where(detected, positions - last_gap, 0)


def _typed_frame(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame({name: columns[name] for name in schema.FRAME_DTYPES})
    for name, dtype in schema.FRAME_DTYPES.items():
        if name == "state":
            frame[name] = pd.Categorical(frame[name], categories=schema.STATES)
        elif name == "source":
            frame[name] = pd.Categorical(frame[name], categories=schema.SOURCES)
        else:
            frame[name] = frame[name].astype(dtype)
    return frame
