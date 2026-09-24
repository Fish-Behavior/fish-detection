"""Generate tests/fixtures/legend_swatches.png: 5 solid legend-swatch crops.

Crops just the 5 solid-color legend squares from the real reference image
`sample_data/sample_labeled_1.png` (restricted data, local-only per PRD
Clarification C14) - no subject/timeline data, only the small color-key
squares at the bottom of the figure. This lets palette-matching unit tests
(Phase 6, T041) run against pixels that genuinely came from the reference
image without depending on or committing the restricted source image itself.

Swatch x-ranges below were located once by scanning the reference image for
near-exact matches to the PRD §5.2 palette hex values, restricted to the
legend's y-band, then taking the widest contiguous column run per color
(rules out stray anti-aliased text pixels) - see docs/progress.md Phase 1
notes for the discovery script. Re-run this file to regenerate the fixture if
the source reference image or its layout changes.

Run directly to (re)generate the committed fixture:
    python tests/fixtures/make_legend_swatches.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

REFERENCE_IMAGE = Path(__file__).parents[3] / "sample_data" / "sample_labeled_1.png"
OUTPUT_PATH = Path(__file__).with_name("legend_swatches.png")

SWATCH_Y = (1058, 1088)  # interior of the legend row, avoids anti-aliased edges
SWATCH_HEIGHT = SWATCH_Y[1] - SWATCH_Y[0]
SWATCH_WIDTH = 26
GAP_PX = 4

# (label, x_start) - widths trimmed to SWATCH_WIDTH from each run's left edge,
# in the fixed legend order left-to-right (PRD §5.2).
SWATCHES = [
    ("Controlled Swim", 504),
    ("Erratic Movement", 716),
    ("Freezing/Drift", 936),
    ("Listing/LORR", 1119),
    ("Surface Breach", 1313),
]


def generate(reference_path: Path = REFERENCE_IMAGE, output_path: Path = OUTPUT_PATH) -> Path:
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"{reference_path} not found - this script must run where the restricted "
            "sample_data/ reference images are locally available (not committed to git)."
        )
    source = Image.open(reference_path).convert("RGB")
    canvas_width = len(SWATCHES) * SWATCH_WIDTH + (len(SWATCHES) - 1) * GAP_PX
    canvas = Image.new("RGB", (canvas_width, SWATCH_HEIGHT), (255, 255, 255))
    x_cursor = 0
    for _label, x_start in SWATCHES:
        crop = source.crop((x_start, SWATCH_Y[0], x_start + SWATCH_WIDTH, SWATCH_Y[1]))
        canvas.paste(crop, (x_cursor, 0))
        x_cursor += SWATCH_WIDTH + GAP_PX
    canvas.save(output_path)
    return output_path


if __name__ == "__main__":
    path = generate()
    print(f"wrote {path}")
