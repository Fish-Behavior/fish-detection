"""The fixed reference colour palette (PRD §5.2) and pixel classification.

Shared by calibration (digitizing the reference figures) and, later,
rendering. `Undetermined` deliberately has no entry: it is pipeline-internal
and has no colour in the original figures. White is `Dead / No Action
Recorded` (Clarification C16), never "no data".

`classify_pixels` uses hue/saturation/value rules rather than nearest RGB
distance. Measured on the real reference figures: they are heavily
blurred/darkened (plot-body blue is ~(17,7,176), not the legend's
(0,0,255)), and nearest-RGB matching snaps every red/blue blend onto the
gray that sits between them in RGB space - a systematic bias toward
Controlled Swim. The HSV rules tolerate brightness shifts and reject
blends (returning `-1` / `None`) instead of guessing.
"""

from __future__ import annotations

import numpy as np

from prepds.models import BehaviorState

PALETTE_HEX: dict[BehaviorState, str] = {
    BehaviorState.CONTROLLED_SWIM: "#BEBEBE",
    BehaviorState.ERRATIC_MOVEMENT: "#FF0000",
    BehaviorState.FREEZING_DRIFT: "#0000FF",
    BehaviorState.LISTING_LORR: "#FFC0CB",
    BehaviorState.SURFACE_BREACH: "#00FF00",
    BehaviorState.DEAD: "#FFFFFF",
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


PALETTE_RGB: dict[BehaviorState, tuple[int, int, int]] = {
    state: _hex_to_rgb(hex_color) for state, hex_color in PALETTE_HEX.items()
}

PALETTE_STATES: tuple[BehaviorState, ...] = tuple(PALETTE_RGB)

WHITE_MAX_SATURATION = 0.10
GRAY_MAX_SATURATION = 0.25
# Real pink cells measured in the Fentanyl panels: saturation 0.19-0.31, value >= 0.94.
# The 1-2px "light reddish" overshoot band at red<->gray/white row boundaries has higher
# saturation and lower value, and was manufacturing ~4.5% phantom pink on vehicle panels;
# this window drops that to <=0.6% while keeping ~92% of genuine pink pixels (a small,
# known low bias on pink - see docs/progress.md Phase 6).
PINK_MAX_SATURATION = 0.32
PINK_MIN_VALUE = 0.94


def classify_pixels(rgb: np.ndarray) -> np.ndarray:
    """`(N, 3)` uint8 RGB -> `(N,)` int index into `PALETTE_STATES`, or `-1` if rejected."""
    x = rgb.astype(float) / 255.0
    maxc = x.max(axis=1)
    minc = x.min(axis=1)
    delta = maxc - minc
    value = maxc
    saturation = np.where(maxc > 0, delta / np.where(maxc > 0, maxc, 1), 0.0)

    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    safe_delta = np.where(delta > 0, delta, 1)
    hue = np.where(
        maxc == r,
        ((g - b) / safe_delta) % 6,
        np.where(maxc == g, (b - r) / safe_delta + 2, (r - g) / safe_delta + 4),
    ) * 60.0
    hue = np.where(delta > 0, hue, 0.0)

    is_reddish = (hue <= 15) | (hue >= 335)

    result = np.full(len(x), -1, dtype=int)
    idx = {state: i for i, state in enumerate(PALETTE_STATES)}
    result[(saturation <= WHITE_MAX_SATURATION) & (value >= 0.92)] = idx[BehaviorState.DEAD]
    result[(saturation <= GRAY_MAX_SATURATION) & (value >= 0.60) & (value <= 0.86)] = idx[BehaviorState.CONTROLLED_SWIM]
    result[(saturation >= 0.75) & (value >= 0.75) & is_reddish] = idx[BehaviorState.ERRATIC_MOVEMENT]
    result[(saturation >= 0.75) & (value >= 0.55) & (hue >= 225) & (hue <= 255)] = idx[BehaviorState.FREEZING_DRIFT]
    result[(saturation >= 0.75) & (value >= 0.75) & (hue >= 105) & (hue <= 135)] = idx[BehaviorState.SURFACE_BREACH]
    result[(saturation >= 0.12) & (saturation <= PINK_MAX_SATURATION) & (value >= PINK_MIN_VALUE) & is_reddish] = idx[BehaviorState.LISTING_LORR]
    return result


def classify_pixel(rgb: tuple[int, int, int]) -> BehaviorState | None:
    index = int(classify_pixels(np.array([rgb], dtype=np.uint8))[0])
    return PALETTE_STATES[index] if index >= 0 else None
