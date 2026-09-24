"""Generate tests/fixtures/synth_tiny.mp4: a deterministic ~5s moving-dot video.

304x240 (matches the real sample video's measured resolution, not the scope
doc's approximate 320x240) at 30fps, plain gray background, a few direction
and speed changes so tracking/feature tests (Phase 3-5) have real motion to
detect - not restricted data, safe to commit (PRD Clarification C14 only
restricts the real trial videos/workbook/reference images).

Run directly to (re)generate the committed fixture:
    python tests/fixtures/make_synth_video.py
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 304, 240
FPS = 30.0
DURATION_S = 5.0
BACKGROUND_GRAY = 180
DOT_RADIUS = 6
DOT_COLOR = (20, 20, 20)  # dark dot, clearly separable from the gray background

OUTPUT_PATH = Path(__file__).with_name("synth_tiny.mp4")


def _dot_position(t: float) -> tuple[int, int]:
    """Piecewise motion: drift right, then a fast diagonal dash, then a slow arc.

    Deterministic (pure function of t, no RNG) so the fixture is reproducible
    byte-for-byte across regenerations on the same OpenCV/codec version.
    """
    margin = DOT_RADIUS + 2
    if t < 1.5:
        # slow horizontal drift - a plausible "Controlled Swim" -like segment
        x = margin + (t / 1.5) * (WIDTH * 0.4)
        y = HEIGHT * 0.5
    elif t < 2.5:
        # fast diagonal dash - a plausible "Erratic Movement" -like segment
        frac = (t - 1.5) / 1.0
        x = WIDTH * 0.4 + frac * (WIDTH * 0.5 - WIDTH * 0.4)
        y = HEIGHT * 0.5 - frac * (HEIGHT * 0.35)
    else:
        # slow arc back down near the "surface" - a plausible "Surface Breach" -like moment
        frac = (t - 2.5) / (DURATION_S - 2.5)
        x = WIDTH * 0.5 + frac * (WIDTH * 0.3)
        y = HEIGHT * 0.15 + math.sin(frac * math.pi) * (HEIGHT * 0.1)
    x = min(max(x, margin), WIDTH - margin)
    y = min(max(y, margin), HEIGHT - margin)
    return int(round(x)), int(round(y))


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    frame_count = int(round(DURATION_S * FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")
    try:
        background = np.full((HEIGHT, WIDTH, 3), BACKGROUND_GRAY, dtype=np.uint8)
        for frame_idx in range(frame_count):
            t = frame_idx / FPS
            frame = background.copy()
            cx, cy = _dot_position(t)
            cv2.circle(frame, (cx, cy), DOT_RADIUS, DOT_COLOR, thickness=-1)
            writer.write(frame)
    finally:
        writer.release()
    return output_path


if __name__ == "__main__":
    path = generate()
    print(f"wrote {path}")
