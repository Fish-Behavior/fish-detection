"""Single-subject strip rendering (T055-T057) - FR-009.

The strip must use the exact reference palette (PRD §5.2) - verified by sampling pixels, not by
reading the code - and its time axis must map seconds to columns linearly over the ACTUAL video
duration (a 1773 s recording is not squeezed into a 1200 s axis).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from prepds.models import BehaviorState, FrameSource, StateSegment
from prepds.palette import PALETTE_RGB
from prepds.rendering import DEFAULT_UNDETERMINED_COLOR_HEX, STRIP_BOX, render_strip, seconds_to_x

STATES = [
    BehaviorState.CONTROLLED_SWIM,
    BehaviorState.ERRATIC_MOVEMENT,
    BehaviorState.FREEZING_DRIFT,
    BehaviorState.LISTING_LORR,
    BehaviorState.SURFACE_BREACH,
    BehaviorState.DEAD,
]


def _seg(start, end, state, source=FrameSource.AUTO) -> StateSegment:
    return StateSegment(start_s=start, end_s=end, duration_s=end - start, state=state, source=source)


def _six_state_segments(each=100.0) -> list[StateSegment]:
    return [_seg(i * each, (i + 1) * each, s) for i, s in enumerate(STATES)]


def _mid_pixel(image: Image.Image, t: float, duration: float) -> tuple[int, int, int]:
    x0, y0, x1, y1 = STRIP_BOX
    return image.getpixel((seconds_to_x(t, duration), (y0 + y1) // 2))[:3]


def test_strip_uses_exact_reference_palette() -> None:
    image = render_strip(_six_state_segments(), duration_s=600.0, subject_id="F_332")
    for i, state in enumerate(STATES):
        assert _mid_pixel(image, i * 100.0 + 50.0, 600.0) == PALETTE_RGB[state]


def test_undetermined_has_its_own_color_distinct_from_all_six() -> None:
    image = render_strip([_seg(0.0, 600.0, BehaviorState.UNDETERMINED)], duration_s=600.0, subject_id="F_332")
    colour = _mid_pixel(image, 300.0, 600.0)
    assert colour == tuple(int(DEFAULT_UNDETERMINED_COLOR_HEX[i : i + 2], 16) for i in (1, 3, 5))
    assert colour not in PALETTE_RGB.values()


def test_undetermined_colour_is_configurable() -> None:
    image = render_strip(
        [_seg(0.0, 100.0, BehaviorState.UNDETERMINED)], duration_s=100.0, subject_id="s", undetermined_color_hex="#123456"
    )
    assert _mid_pixel(image, 50.0, 100.0) == (0x12, 0x34, 0x56)


def test_time_axis_maps_seconds_to_columns_linearly_over_the_actual_duration() -> None:
    x0, _, x1, _ = STRIP_BOX
    assert seconds_to_x(0.0, 600.0) == x0
    assert seconds_to_x(600.0, 600.0) == x1
    assert seconds_to_x(300.0, 600.0) == pytest.approx((x0 + x1) / 2, abs=1)
    # the same second lands at a different column on a longer recording
    assert seconds_to_x(600.0, 1773.0) < seconds_to_x(600.0, 1200.0)


def test_a_segment_boundary_lands_on_its_time_column() -> None:
    image = render_strip(
        [_seg(0.0, 300.0, BehaviorState.CONTROLLED_SWIM), _seg(300.0, 600.0, BehaviorState.FREEZING_DRIFT)],
        duration_s=600.0,
        subject_id="F_332",
    )
    _, y0, _, y1 = STRIP_BOX
    boundary = seconds_to_x(300.0, 600.0)
    y = (y0 + y1) // 2
    assert image.getpixel((boundary - 1, y))[:3] == PALETTE_RGB[BehaviorState.CONTROLLED_SWIM]
    assert image.getpixel((boundary, y))[:3] == PALETTE_RGB[BehaviorState.FREEZING_DRIFT]


def test_the_last_segment_fills_to_the_end_of_the_axis_and_nothing_is_drawn_beyond_it() -> None:
    duration = 1773.0
    image = render_strip([_seg(0.0, duration, BehaviorState.ERRATIC_MOVEMENT)], duration_s=duration, subject_id="x")
    x0, y0, x1, y1 = STRIP_BOX
    y = (y0 + y1) // 2
    assert image.getpixel((x1 - 1, y))[:3] == PALETTE_RGB[BehaviorState.ERRATIC_MOVEMENT]
    assert image.getpixel((x1 + 3, y))[:3] == (255, 255, 255)  # plain background past the axis


def test_axis_has_tick_marks_and_labels_below_the_strip() -> None:
    image = render_strip(_six_state_segments(), duration_s=600.0, subject_id="F_332")
    _, _, _, y1 = STRIP_BOX
    tick_x = seconds_to_x(200.0, 600.0)  # a 100 s step on a 600 s axis
    assert image.getpixel((tick_x, y1 + 3))[:3] == (0, 0, 0)  # tick mark
    label_band = image.crop((STRIP_BOX[0], y1 + 8, STRIP_BOX[2] + 20, image.height))
    white_band = Image.new("RGB", label_band.size, (255, 255, 255))
    assert label_band.convert("RGB").tobytes() != white_band.tobytes()  # tick labels are drawn


def test_subject_label_is_drawn_left_of_the_strip_and_changes_with_the_subject() -> None:
    a = render_strip(_six_state_segments(), duration_s=600.0, subject_id="F_332")
    b = render_strip(_six_state_segments(), duration_s=600.0, subject_id="M_0069")
    left = (0, STRIP_BOX[1], STRIP_BOX[0] - 2, STRIP_BOX[3])
    assert a.crop(left).tobytes() != Image.new("RGB", (left[2], left[3] - left[1]), (255, 255, 255)).tobytes()
    assert a.crop(left).tobytes() != b.crop(left).tobytes()


def test_writes_a_png_that_reopens_at_the_same_size(tmp_path: Path) -> None:
    out = tmp_path / "strip.png"
    image = render_strip(_six_state_segments(), duration_s=600.0, subject_id="F_332", output_path=out)
    with Image.open(out) as reopened:
        assert reopened.format == "PNG" and reopened.size == image.size
        assert reopened.convert("RGB").getpixel((seconds_to_x(50.0, 600.0), STRIP_BOX[1] + 5)) == PALETTE_RGB[STATES[0]]


@pytest.mark.parametrize("duration", [0.0, -5.0])
def test_a_non_positive_duration_is_rejected(duration: float) -> None:
    with pytest.raises(ValueError, match="duration"):
        render_strip([], duration_s=duration, subject_id="x")


def test_a_segment_running_past_the_duration_is_rejected_not_clipped_silently() -> None:
    with pytest.raises(ValueError, match="past"):
        render_strip([_seg(0.0, 700.0, BehaviorState.DEAD)], duration_s=600.0, subject_id="x")
