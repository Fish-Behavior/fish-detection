"""Background model + waterline detection from a sparse frame sample (T026/T027).

PRD §9.5.3: static camera + plain background -> per-video background
estimated via median of a sparse frame sample; waterline located as a stable
horizontal edge (row-wise gradient peak) in that background frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prepds.roi import WATERLINE_EDGE_MARGIN_PX, detect_waterline, estimate_background

FIXTURES = Path(__file__).parent / "fixtures"


def test_background_model_from_sparse_sample_removes_moving_dot() -> None:
    background = estimate_background(FIXTURES / "synth_tiny.mp4", stride=30)
    assert background.shape == (240, 304, 3)
    # The synthetic dot (dark, radius 6) visits 5 well-separated positions
    # across the 5 sampled frames (t=0,1,2,3,4s) - at any single pixel it
    # covers at most 1 of those 5 samples, so the per-pixel median must
    # reconstruct the plain gray background (180) essentially everywhere,
    # not an average that's pulled toward the dot's dark color.
    background_gray = background.astype(np.int32).mean(axis=-1)
    near_180 = np.isclose(background_gray, 180, atol=3)
    assert near_180.mean() > 0.99  # at least 99% of pixels are clean background


def test_waterline_detection_on_synthetic_frame() -> None:
    # A sharp horizontal step: bright "above surface" band, darker "water"
    # band below, step at row 50 of a 240-row frame. Exact match, not just
    # "close to 50" - the off-by-one between transition-space and row-space
    # that code review caught means this must land exactly on the row that
    # changed, not the row before it.
    frame = np.full((240, 304, 3), 200, dtype=np.uint8)
    frame[50:, :, :] = 140
    assert detect_waterline(frame) == 50


def test_waterline_detection_ignores_near_edge_step_favoring_stronger_interior_step() -> None:
    # A real near-edge step (rows 0-1 vs 2+) that would win a naive argmax,
    # plus a genuine interior step at row 100. WATERLINE_EDGE_MARGIN_PX must
    # suppress the near-edge one and report the interior one - a uniform
    # frame (no step anywhere) doesn't actually exercise this, since argmax
    # on all-zero gradients trivially returns index 0 regardless of the
    # margin logic being correct.
    frame = np.full((240, 304, 3), 180, dtype=np.uint8)
    frame[2:, :, :] = 178  # small near-edge step, inside the margin
    frame[100:, :, :] = 140  # larger interior step
    assert detect_waterline(frame) == 100


def test_waterline_detection_frame_too_short_raises() -> None:
    tiny_frame = np.full((2 * WATERLINE_EDGE_MARGIN_PX, 304, 3), 180, dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_waterline(tiny_frame)


def test_estimate_background_missing_file_raises() -> None:
    with pytest.raises(ValueError):
        estimate_background(FIXTURES / "does_not_exist.mp4")


def test_estimate_background_no_frames_readable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Distinct from the missing-file case above: a video that opens fine but
    # yields zero frames (e.g. corrupt past the header) - frames() itself
    # wouldn't raise (it just yields nothing), so estimate_background()'s own
    # empty-sample check is what must catch this.
    import prepds.roi as roi_module

    monkeypatch.setattr(roi_module, "frames", lambda video_path, stride=30: iter(()))
    with pytest.raises(ValueError, match="No frames could be read"):
        estimate_background(FIXTURES / "synth_tiny.mp4")
