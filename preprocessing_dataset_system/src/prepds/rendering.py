"""Single-subject behavior strip (PRD §5.2, FR-009).

One horizontal strip per video in the reference figures' style: a row of flat colour blocks, one per
segment, over a time axis in seconds, with the subject ID on the left. Colours are the exact reference
palette (`palette.PALETTE_RGB`); pixels are drawn with PIL rather than an antialiasing plotting library so
a sampled pixel is exactly the legend colour.

The axis always spans the ACTUAL video duration, so a 1773 s recording is not squeezed into 1200 s.

`Undetermined` (a pipeline-internal marker with no colour in the original figures) is magenta
`#FF00FF`, the PRD's own placeholder suggestion: it is distinct from all six legend colours, including
white (Dead) and pink (Listing/LORR), and impossible to mistake for a real ethogram label. Configurable
via `rendering.undetermined_color_hex`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepds.models import BehaviorState, StateSegment
from prepds.palette import PALETTE_RGB

DEFAULT_UNDETERMINED_COLOR_HEX = "#FF00FF"

IMAGE_SIZE = (1100, 130)
STRIP_BOX = (110, 30, 1070, 90)  # x0, y0, x1, y1 (x1, y1 exclusive): the coloured strip
BACKGROUND = (255, 255, 255)
INK = (0, 0, 0)
_TICK_STEPS = (10, 20, 50, 100, 200, 300, 500, 1000)
_MAX_TICKS = 8
_FLOAT_TOLERANCE_S = 1e-6


def seconds_to_x(seconds: float, duration_s: float) -> int:
    """Column of `seconds` on an axis spanning `duration_s` (0 -> strip left edge, duration -> right edge)."""
    x0, _, x1, _ = STRIP_BOX
    return x0 + round(seconds / duration_s * (x1 - x0))


def render_strip(
    segments: Sequence[StateSegment],
    *,
    duration_s: float,
    subject_id: str,
    output_path: Path | None = None,
    undetermined_color_hex: str = DEFAULT_UNDETERMINED_COLOR_HEX,
) -> Image.Image:
    """Draw the strip; also write a PNG when `output_path` is given. Returns the RGB image."""
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    for segment in segments:
        if segment.end_s > duration_s + _FLOAT_TOLERANCE_S:
            raise ValueError(
                f"segment {segment.state.value} [{segment.start_s}, {segment.end_s}] runs past the video duration {duration_s}"
            )

    image = Image.new("RGB", IMAGE_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = STRIP_BOX
    colours = {**PALETTE_RGB, BehaviorState.UNDETERMINED: _hex_to_rgb(undetermined_color_hex)}

    for segment in segments:
        left = seconds_to_x(max(segment.start_s, 0.0), duration_s)
        right = min(seconds_to_x(segment.end_s, duration_s), x1)
        if right > left:
            draw.rectangle([left, y0, right - 1, y1 - 1], fill=colours[segment.state])

    font = ImageFont.load_default(size=13)
    draw.rectangle([x0 - 1, y0 - 1, x1, y1], outline=INK)
    for tick in _ticks(duration_s):
        x = seconds_to_x(tick, duration_s)
        draw.line([x, y1 + 1, x, y1 + 5], fill=INK)
        draw.text((x, y1 + 8), f"{tick:g}", fill=INK, font=font, anchor="mt")
    draw.text((x0 - 10, (y0 + y1) // 2), subject_id, fill=INK, font=font, anchor="rm")
    draw.text(((x0 + x1) // 2, y1 + 28), "seconds", fill=INK, font=font, anchor="mt")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
    return image


def _ticks(duration_s: float) -> list[float]:
    step = next((s for s in _TICK_STEPS if duration_s / s <= _MAX_TICKS), _TICK_STEPS[-1])
    ticks = [float(t) for t in range(0, int(duration_s) + 1, step)]
    return ticks


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected a #RRGGBB colour, got {hex_color!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
