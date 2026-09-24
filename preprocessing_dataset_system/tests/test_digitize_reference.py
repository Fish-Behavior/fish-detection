"""Reference-image digitization (T040/T041) - FR-008, PRD §9.5.5.

Design decisions locked in here (advisor consultation, pre-Phase-6):

- Plot geometry is HARDCODED per image (two images, ten panels, never
  re-parsed) - auto-detection was tried and found fragile (missed panels,
  picked up row-label text), see docs/progress.md Phase 6.
- x is mapped once per image from the shared 0s/1200s pixel positions, never
  from a per-row right edge: rows that end in Dead end white, so a per-row
  edge would truncate exactly the rows that matter most.
- Pixels are classified by hue/saturation/value rules with blends rejected
  (see the T041 section below for why nearest-RGB was abandoned).
- The group-level target is the pixel fraction over each panel's plot
  rectangle - no per-subject OCR (pytesseract isn't installed, and the
  group aggregate is the primary calibration target per PRD §9.5.5).
- White is `Dead`, not "no data" (Clarification C16) - so any margin/gap
  sampled by mistake would silently inflate Dead. Bounds are tight, and
  blended/antialiased pixels are rejected rather than snapped to whichever
  colour they sit closest to; the rejected fraction is reported per panel
  so a bad crop is visible, not silent.
- Tests here use synthetic images plus `legend_swatches.png` (real palette
  pixels, committed fixture). The restricted reference PNGs themselves are
  only ever read at runtime from `sample_data/` (C14), never in tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from prepds.calibration.digitize_reference import (
    IMAGE_GEOMETRIES,
    ImageGeometry,
    PanelGeometry,
    digitize_panel,
    pixel_x_to_seconds,
    seconds_to_pixel_x,
)
from prepds.models import BehaviorState
from prepds.palette import PALETTE_RGB, classify_pixel

FIXTURES = Path(__file__).parent / "fixtures"

SWATCH_ORDER = [
    BehaviorState.CONTROLLED_SWIM,
    BehaviorState.ERRATIC_MOVEMENT,
    BehaviorState.FREEZING_DRIFT,
    BehaviorState.LISTING_LORR,
    BehaviorState.SURFACE_BREACH,
]


# --- T040: axis pixel -> seconds ------------------------------------------------


def test_axis_pixel_to_seconds_mapping() -> None:
    assert pixel_x_to_seconds(332, x_zero_px=332, x_end_px=1553) == pytest.approx(0.0)
    assert pixel_x_to_seconds(1553, x_zero_px=332, x_end_px=1553) == pytest.approx(1200.0)
    assert pixel_x_to_seconds(942.5, x_zero_px=332, x_end_px=1553) == pytest.approx(600.0)


def test_seconds_to_pixel_x_is_the_inverse_mapping() -> None:
    for seconds in (0.0, 1.0, 300.5, 1199.0, 1200.0):
        x = seconds_to_pixel_x(seconds, x_zero_px=345, x_end_px=1705)
        assert pixel_x_to_seconds(x, x_zero_px=345, x_end_px=1705) == pytest.approx(seconds)


def test_known_geometry_constants_are_sane() -> None:
    assert len(IMAGE_GEOMETRIES) == 2
    for geometry in IMAGE_GEOMETRIES:
        assert geometry.x_zero_px < geometry.x_end_px
        assert len(geometry.panels) == 5
        previous_bottom = -1
        for panel in geometry.panels:
            assert panel.y0 < panel.y1
            assert panel.y0 > previous_bottom, "panels must not overlap or be out of order"
            previous_bottom = panel.y1
    concentrations = {g.concentration_uM for g in IMAGE_GEOMETRIES}
    assert concentrations == {30, 100}


# --- T041: palette colour classification ---------------------------------------
#
# Classification is by hue/saturation/value, not nearest RGB distance: the real
# reference figures are heavily blurred and darkened (plot-body blue is ~(17,7,176),
# not (0,0,255)), and nearest-RGB matching snaps red/blue blends onto gray - a
# systematic bias, found by measuring the real image (docs/progress.md Phase 6).


def test_color_nearest_match_to_palette_on_real_legend_swatches() -> None:
    swatches = np.array(Image.open(FIXTURES / "legend_swatches.png").convert("RGB"))
    swatch_width, gap = 26, 4
    for i, expected in enumerate(SWATCH_ORDER):
        x_mid = i * (swatch_width + gap) + swatch_width // 2
        pixel = tuple(int(v) for v in swatches[swatches.shape[0] // 2, x_mid])
        assert classify_pixel(pixel) == expected


def test_white_is_dead_not_no_data() -> None:
    assert classify_pixel((255, 255, 255)) == BehaviorState.DEAD


def test_palette_has_no_undetermined_entry() -> None:
    # Undetermined is pipeline-internal only - it has no reference colour.
    assert BehaviorState.UNDETERMINED not in PALETTE_RGB
    assert len(PALETTE_RGB) == 6


def test_pixel_far_from_every_palette_colour_is_rejected_not_snapped() -> None:
    assert classify_pixel((0, 0, 0)) is None
    # Half-way between red and white: antialiasing, not a real state.
    assert classify_pixel((255, 128, 128)) is None


def test_darkened_blue_and_red_from_the_blurred_real_figure_still_classify() -> None:
    # Observed plot-body colours in sample_labeled_1.png (JPEG-blurred, darker
    # than the ideal legend swatches).
    assert classify_pixel((17, 7, 176)) == BehaviorState.FREEZING_DRIFT
    assert classify_pixel((247, 0, 10)) == BehaviorState.ERRATIC_MOVEMENT
    assert classify_pixel((178, 178, 180)) == BehaviorState.CONTROLLED_SWIM


def test_red_blue_blend_is_rejected_not_snapped_to_gray() -> None:
    # The blend between an adjacent red row and blue row must not become
    # "Controlled Swim" just because gray sits between them in RGB space.
    assert classify_pixel((126, 74, 183)) is None
    assert classify_pixel((161, 54, 138)) is None


def test_slightly_off_pink_stays_pink_not_white() -> None:
    # Pink and white are only ~82 RGB units apart; a lightly JPEG-perturbed
    # pink must still resolve to pink.
    assert classify_pixel((255, 200, 208)) == BehaviorState.LISTING_LORR


# --- digitize_panel on synthetic images ----------------------------------------


def _synthetic_geometry() -> ImageGeometry:
    return ImageGeometry(
        filename="synthetic.png",
        concentration_uM=30,
        x_zero_px=10,
        x_end_px=1209,  # 1199 px -> ~1 px per second
        panels=(PanelGeometry(label="VEH", compound="veh", concentration_uM=0, y0=5, y1=24),),
    )


def _blank_canvas() -> np.ndarray:
    return np.full((40, 1300, 3), 255, dtype=np.uint8)


def test_digitize_panel_proportions_on_a_two_colour_panel() -> None:
    canvas = _blank_canvas()
    geometry = _synthetic_geometry()
    panel = geometry.panels[0]
    mid_x = (geometry.x_zero_px + geometry.x_end_px) // 2
    canvas[panel.y0 : panel.y1 + 1, geometry.x_zero_px : mid_x] = PALETTE_RGB[BehaviorState.FREEZING_DRIFT]
    canvas[panel.y0 : panel.y1 + 1, mid_x : geometry.x_end_px + 1] = PALETTE_RGB[BehaviorState.ERRATIC_MOVEMENT]

    target = digitize_panel(canvas, geometry, panel)

    assert target.proportions[BehaviorState.FREEZING_DRIFT] == pytest.approx(0.5, abs=0.02)
    assert target.proportions[BehaviorState.ERRATIC_MOVEMENT] == pytest.approx(0.5, abs=0.02)
    assert target.proportions[BehaviorState.DEAD] == pytest.approx(0.0, abs=0.01)
    assert target.rejected_fraction == pytest.approx(0.0, abs=0.01)


def test_digitize_panel_proportions_sum_to_one_over_accepted_pixels() -> None:
    canvas = _blank_canvas()
    geometry = _synthetic_geometry()
    panel = geometry.panels[0]
    canvas[panel.y0 : panel.y1 + 1, geometry.x_zero_px : geometry.x_end_px + 1] = PALETTE_RGB[BehaviorState.CONTROLLED_SWIM]
    canvas[panel.y0 + 8 : panel.y0 + 11, :] = (0, 0, 0)  # an interior black stripe: rejected, not counted

    target = digitize_panel(canvas, geometry, panel)

    assert sum(target.proportions.values()) == pytest.approx(1.0)
    assert target.rejected_fraction > 0.1


def test_digitize_panel_counts_white_inside_the_plot_as_dead() -> None:
    canvas = _blank_canvas()
    geometry = _synthetic_geometry()
    panel = geometry.panels[0]
    canvas[panel.y0 : panel.y1 + 1, geometry.x_zero_px : geometry.x_end_px + 1] = PALETTE_RGB[BehaviorState.FREEZING_DRIFT]
    tail_start = geometry.x_zero_px + int(0.75 * (geometry.x_end_px - geometry.x_zero_px))
    canvas[panel.y0 : panel.y1 + 1, tail_start : geometry.x_end_px + 1] = 255  # a subject that dies at 75%

    target = digitize_panel(canvas, geometry, panel)

    assert target.proportions[BehaviorState.DEAD] == pytest.approx(0.25, abs=0.03)


def test_digitize_panel_does_not_sample_outside_the_panel_rectangle() -> None:
    canvas = _blank_canvas()  # all white everywhere - would read as 100% Dead if the margins leaked in
    geometry = _synthetic_geometry()
    panel = geometry.panels[0]
    canvas[panel.y0 : panel.y1 + 1, geometry.x_zero_px : geometry.x_end_px + 1] = PALETTE_RGB[BehaviorState.CONTROLLED_SWIM]

    target = digitize_panel(canvas, geometry, panel)

    assert target.proportions[BehaviorState.DEAD] == pytest.approx(0.0, abs=0.01)
    assert target.proportions[BehaviorState.CONTROLLED_SWIM] == pytest.approx(1.0, abs=0.01)


def test_digitize_panel_raises_on_panel_outside_the_image() -> None:
    geometry = ImageGeometry(
        filename="synthetic.png",
        concentration_uM=30,
        x_zero_px=10,
        x_end_px=1209,
        panels=(PanelGeometry(label="VEH", compound="veh", concentration_uM=0, y0=5, y1=500),),
    )
    with pytest.raises(ValueError):
        digitize_panel(_blank_canvas(), geometry, geometry.panels[0])


# --- bias check: known proportions through blur + JPEG --------------------------


def _blurred_jpeg_figure(seed: int, sigma: float, quality: int) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic panel of random state cells with KNOWN pixel proportions, then Gaussian-blurred and
    JPEG-compressed like the real reference figures. Returns (image, true proportions in palette order)."""
    import io

    import cv2

    from prepds.palette import PALETTE_STATES

    rng = np.random.default_rng(seed)
    weights = np.array([0.30, 0.25, 0.30, 0.08, 0.02, 0.05])
    rows, row_h, width = 9, 9, 1221
    image = np.full((rows * row_h, width, 3), 255, np.uint8)
    truth = np.zeros(len(PALETTE_STATES))
    for r in range(rows):
        x = 0
        while x < width:
            w = min(int(rng.integers(5, 80)), width - x)
            k = int(rng.choice(len(PALETTE_STATES), p=weights))
            image[r * row_h : (r + 1) * row_h, x : x + w] = PALETTE_RGB[PALETTE_STATES[k]]
            truth[k] += w * row_h
            x += w
    image = cv2.GaussianBlur(image, (0, 0), sigma)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "JPEG", quality=quality)
    return np.array(Image.open(buffer).convert("RGB")), truth / truth.sum()


def test_digitizer_recovers_known_proportions_from_a_blurred_jpeg_figure() -> None:
    # End-to-end regression test through digitize_panel. At the original
    # gray-saturation tolerance (0.10), JPEG chroma leaking from adjacent
    # red/blue rows tinted gray cores past the cutoff, under-counting
    # Controlled Swim by ~5-6pp and inflating red/blue - a systematic bias no
    # rejected-fraction report would reveal.
    from prepds.palette import PALETTE_STATES

    estimates, truths = [], []
    for seed in range(6):
        image, truth = _blurred_jpeg_figure(seed, sigma=2.0, quality=60)
        geometry = ImageGeometry(
            filename="synthetic.png",
            concentration_uM=30,
            x_zero_px=0,
            x_end_px=image.shape[1] - 1,
            panels=(PanelGeometry("SYN", "veh", 0, 0, image.shape[0] - 1),),
        )
        target = digitize_panel(image, geometry, geometry.panels[0])
        estimates.append([target.proportions[state] for state in PALETTE_STATES])
        truths.append(truth)

    worst_error = np.abs(np.mean(estimates, axis=0) - np.mean(truths, axis=0)).max()
    assert worst_error < 0.03


def test_red_gray_boundary_overshoot_band_is_not_counted_as_pink() -> None:
    # Regression test: at a red->gray row boundary the real (JPEG-blurred)
    # figure passes through a 1-2px "light reddish" band like (245,154,169) or
    # (233,177,185) that a loose pink window accepts - ~4.5% phantom Listing/LORR
    # on the vehicle panels, which have essentially none. Real pink cells were
    # measured at saturation 0.19-0.31 and value >= 0.94; the overshoot band sits
    # at higher saturation / lower value.
    assert classify_pixel((245, 154, 169)) is None
    assert classify_pixel((233, 177, 185)) is None
    assert classify_pixel((230, 172, 174)) is None
    assert classify_pixel((255, 192, 203)) == BehaviorState.LISTING_LORR  # the true legend pink


def test_a_genuine_full_height_pink_cell_is_still_counted() -> None:
    canvas = _blank_canvas()
    geometry = _synthetic_geometry()
    panel = geometry.panels[0]
    x0, x1 = geometry.x_zero_px, geometry.x_end_px + 1
    canvas[panel.y0 : panel.y1 + 1, x0:x1] = PALETTE_RGB[BehaviorState.LISTING_LORR]

    target = digitize_panel(canvas, geometry, panel)

    assert target.proportions[BehaviorState.LISTING_LORR] == pytest.approx(1.0, abs=0.01)


# --- group targets (duplicate VEH panels) ---------------------------------------


def _target(label, compound, conc, source, **props) -> "PanelTarget":
    from prepds.calibration.digitize_reference import PanelTarget

    proportions = {state: 0.0 for state in PALETTE_RGB}
    for name, value in props.items():
        proportions[BehaviorState[name]] = value
    return PanelTarget(label, compound, conc, source, proportions, rejected_fraction=0.3, n_samples=1000)


def test_group_targets_merges_duplicate_vehicle_panels_by_averaging() -> None:
    from prepds.calibration.digitize_reference import group_targets

    veh_a = _target("VEH", "veh", 0, "a.png", CONTROLLED_SWIM=0.30, ERRATIC_MOVEMENT=0.40, FREEZING_DRIFT=0.30)
    veh_b = _target("VEH", "veh", 0, "b.png", CONTROLLED_SWIM=0.32, ERRATIC_MOVEMENT=0.36, FREEZING_DRIFT=0.32)
    mdma = _target("MDMA 30uM", "mdma", 30, "a.png", CONTROLLED_SWIM=1.0)

    groups = group_targets([veh_a, veh_b, mdma])

    assert set(groups) == {("veh", 0), ("mdma", 30)}
    assert groups[("veh", 0)][BehaviorState.CONTROLLED_SWIM] == pytest.approx(0.31)
    assert sum(groups[("veh", 0)].values()) == pytest.approx(1.0)


def test_group_targets_rejects_duplicate_panels_that_disagree() -> None:
    from prepds.calibration.digitize_reference import group_targets

    veh_a = _target("VEH", "veh", 0, "a.png", CONTROLLED_SWIM=1.0)
    veh_b = _target("VEH", "veh", 0, "b.png", ERRATIC_MOVEMENT=1.0)  # TV distance 1.0: not the same data
    with pytest.raises(ValueError, match="disagree"):
        group_targets([veh_a, veh_b])
