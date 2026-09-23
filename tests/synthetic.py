"""Synthetic test videos: a drawn beaker scene with a moving dark "fish" (no real data).

Reusable by every step's tests. The scene, from top to bottom:

* light background (the lab wall);
* a darker rectangle = the beaker, with a horizontal brightness step at `waterline_y`
  (glass above is lighter than the water below), so the waterline is a clear edge;
* a dark orange ellipse = the (colored) fish, placed each frame by
  ``fish_path(t) -> (x, y, angle_deg)``, with a black eye near its head. The head is the
  end the drawing angle points to (angle 0 = facing right, 180 = facing left);
* optionally a faint, slightly tinted mirrored copy of the fish BELOW the beaker bottom
  = the reflection (much weaker than the fish, like the real one);
* optionally a whole-frame brightness drop in some frames = a light flicker / camera
  exposure change, which must not count as fish movement;
* optionally time windows in which the fish is not drawn at all (hidden), to test gaps;
* optionally a shorter fish at some times (turned toward the camera), and a gray "decoy"
  blob that is not the fish (stands in for a mirror image of the fish in the glass).

Videos are written as MJPG .avi, which OpenCV can read and write on every OS.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

FishPath = Callable[[float], tuple[float, float, float]]

# Gray levels of the drawn scene (0-255).
WALL = 205
GLASS = 175  # beaker above the water
WATER = 140  # beaker below the waterline: the 35-level step is the waterline edge
FISH_BGR = (20, 60, 160)  # dark orange, like the real (colored) fish
FISH = 85  # its approximate gray level, for tests that look for dark pixels
EYE_BGR = (15, 15, 15)  # the eye: much darker than the body, like the real fish
EYE_POSITION = 0.6  # eye centre, as a fraction of the half-length from the body centre
REFLECTION_DROP = 10  # the reflection darkens the wall by only this much (under scene.diff_threshold)
REFLECTION_TINT = (0, 12, 30)  # ...and tints it toward the fish color (BGR added)
DECOY = 100  # gray level of the decoy: clearly darker than the water, lighter than the fish


def default_beaker(size: tuple[int, int], waterline_y: int) -> tuple[int, int, int, int]:
    """Beaker rectangle (x0, y0, x1, y1) as fractions of the frame, with the rim well above the water."""
    width, height = size
    return round(0.15 * width), round(0.08 * height), round(0.85 * width), round(0.75 * height)


def lissajous_path(beaker: tuple[int, int, int, int], waterline_y: int, fish_len: float,
                   duration_s: float, margin: float = 1.0) -> FishPath:
    """A fish path that sweeps the whole water volume, so no pixel is covered in most frames.

    x and y oscillate at different, unrelated speeds (a Lissajous curve); `margin`
    (in fish lengths) keeps the fish body inside the water.
    """
    x0, _, x1, y1 = beaker
    left, right = x0 + margin * fish_len, x1 - margin * fish_len
    top, bottom = waterline_y + 0.6 * fish_len, y1 - 0.6 * fish_len

    def path(t: float) -> tuple[float, float, float]:
        u = t / duration_s * 2 * math.pi
        x = (left + right) / 2 + (right - left) / 2 * math.sin(3 * u)
        y = (top + bottom) / 2 + (bottom - top) / 2 * math.sin(5 * u + 0.7)
        angle = 20 * math.sin(7 * u)  # gentle tilt
        return x, y, angle

    return path


def head_position(fish: tuple[float, float, float], fish_axes: tuple[float, float]) -> tuple[float, float]:
    """Centre of the drawn eye: along the drawing angle (clockwise on screen, y down)."""
    x, y, angle = fish
    theta = math.radians(angle)
    offset = EYE_POSITION * fish_axes[0]
    return x + offset * math.cos(theta), y + offset * math.sin(theta)


def draw_frame(size: tuple[int, int], beaker: tuple[int, int, int, int], waterline_y: int,
               fish: tuple[float, float, float] | None, fish_axes: tuple[float, float],
               reflection: bool, brightness: int = 0, reflection_drop: int = REFLECTION_DROP,
               decoy: tuple[float, float, float, tuple[float, float]] | None = None) -> np.ndarray:
    """One BGR frame of the synthetic scene; `brightness` is added to every pixel.

    `fish` None draws the empty scene (fish hidden). `decoy` = (x, y, angle, axes) of
    a gray ellipse that is not the fish.
    """
    width, height = size
    gray = np.full((height, width), WALL, np.uint8)
    x0, y0, x1, y1 = beaker
    gray[y0:y1, x0:x1] = GLASS
    gray[waterline_y:y1, x0:x1] = WATER
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if decoy is not None:
        dx, dy, dangle, daxes = decoy
        cv2.ellipse(frame, (round(dx), round(dy)), (round(daxes[0]), round(daxes[1])), dangle, 0, 360,
                    (DECOY, DECOY, DECOY), -1, cv2.LINE_AA)
    if fish is None:
        return np.clip(frame.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

    x, y, angle = fish
    axes = (max(1, round(fish_axes[0])), max(1, round(fish_axes[1])))
    cv2.ellipse(frame, (round(x), round(y)), axes, angle, 0, 360, FISH_BGR, -1, cv2.LINE_AA)
    head = head_position(fish, fish_axes)
    cv2.circle(frame, (round(head[0]), round(head[1])), max(1, round(fish_axes[1] * 0.5)), EYE_BGR, -1, cv2.LINE_AA)
    if reflection:
        # Mirror the fish about the beaker bottom and draw it faintly on the wall below.
        mask = np.zeros((height, width), np.uint8)
        cv2.ellipse(mask, (round(x), round(2 * y1 - y)), axes, -angle, 0, 360, 255, -1)
        mask[:y1, :] = 0  # only below the beaker
        change = np.array(REFLECTION_TINT, np.int16) - reflection_drop
        frame = np.where(mask[..., None] > 0, frame.astype(np.int16) + change, frame)
    return np.clip(frame.astype(np.int16) + brightness, 0, 255).astype(np.uint8)


def make_video(
    path: str | Path,
    fps: float = 30.0,
    size: tuple[int, int] = (320, 240),
    duration_s: float = 10.0,
    waterline_y: int | None = None,
    fish_path: FishPath | None = None,
    beaker: tuple[int, int, int, int] | None = None,
    fish_length: float | None = None,
    reflection: bool = True,
    flicker: tuple[int, int] | None = None,
    hidden: list[tuple[float, float]] | None = None,
    reflection_drop: int = REFLECTION_DROP,
    length_factor: Callable[[float], float] | None = None,
    decoy: FishPath | None = None,
    decoy_axes: tuple[float, float] | None = None,
) -> dict:
    """Write a synthetic MJPG .avi and return its ground truth.

    Sizes default to fractions of the frame so tests can run at any resolution.
    ``flicker=(every, amount)`` darkens every `every`-th frame by `amount` gray levels.
    ``hidden=[(start_s, end_s), ...]`` leaves the fish out of frames with start <= t < end.
    `reflection_drop` sets how much the reflection darkens the wall (default: faint).
    ``length_factor(t)`` shortens the drawn fish (1 = full length, e.g. 0.5 = turned
    toward the camera). ``decoy(t) -> (x, y, angle)`` with `decoy_axes` draws a gray
    blob that is not the fish in every frame.
    Returns a dict with the path, fps, frame_count, size, waterline_y, beaker box,
    fish axes, the list of fish positions (x, y, angle) per frame, and `visible` per frame.
    """
    path = Path(path)
    width, height = size
    waterline_y = round(0.35 * height) if waterline_y is None else waterline_y
    beaker = default_beaker(size, waterline_y) if beaker is None else beaker
    fish_length = 0.12 * width if fish_length is None else fish_length
    fish_axes = (fish_length / 2, fish_length / 7)  # slender body: half-length, half-height
    fish_path = fish_path or lissajous_path(beaker, waterline_y, fish_length, duration_s)

    n_frames = round(duration_s * fps)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot write MJPG video to {path}")
    positions, visible = [], []
    try:
        for index in range(n_frames):
            t = index / fps
            fish = fish_path(t)
            positions.append(fish)
            shown = not any(start <= t < end for start, end in hidden or [])
            visible.append(shown)
            dimmed = flicker is not None and index % flicker[0] == 0
            brightness = -flicker[1] if dimmed else 0
            factor = length_factor(t) if length_factor else 1.0
            axes = (fish_axes[0] * factor, fish_axes[1])
            decoy_shape = (*decoy(t), decoy_axes) if decoy else None
            writer.write(draw_frame(size, beaker, waterline_y, fish if shown else None, axes, reflection,
                                    brightness, reflection_drop, decoy_shape))
    finally:
        writer.release()
    return {
        "path": path,
        "fps": fps,
        "frame_count": n_frames,
        "size": size,
        "waterline_y": waterline_y,
        "beaker": beaker,
        "fish_axes": fish_axes,
        "positions": positions,
        "visible": visible,
    }
