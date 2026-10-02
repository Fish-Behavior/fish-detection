"""Real fps/duration/frame_count from the container, never assumed constants (T024/T025).

`probe()` was pulled forward into Phase 2 (catalog.py's FR-004 duration-
mismatch check needed it); `frames()` completes T025's original scope here in
Phase 3. ROI/waterline/background model (T026-T030) live in roi.py - see
docs/progress.md §1 sequencing note.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prepds.video_io import frames, probe

FIXTURES = Path(__file__).parent / "fixtures"


def test_probe_reads_actual_fps_and_duration_not_hardcoded() -> None:
    asset = probe(FIXTURES / "synth_tiny.mp4")
    # The fixture is 150 frames @ 30fps = exactly 5.0s - asserting the exact
    # measured values (not "close to 30/1200") is the point: nothing here is
    # allowed to fall back to an assumed nominal 30fps/1200s constant.
    assert asset.fps == pytest.approx(30.0, abs=0.5)
    assert asset.frame_count == 150
    assert asset.duration_s == pytest.approx(5.0, abs=0.2)
    assert asset.resolution == (304, 240)


def test_probe_missing_file_raises() -> None:
    with pytest.raises(ValueError):
        probe(FIXTURES / "does_not_exist.mp4")


def test_probe_zero_byte_file_raises(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"")
    with pytest.raises(ValueError):
        probe(corrupt)


def test_frames_yields_every_frame_in_order() -> None:
    indices = [idx for idx, _frame in frames(FIXTURES / "synth_tiny.mp4")]
    assert indices == list(range(150))


def test_frames_yields_correctly_shaped_bgr_arrays() -> None:
    _idx, frame = next(frames(FIXTURES / "synth_tiny.mp4"))
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (240, 304, 3)  # (height, width, channels) - matches probe()'s resolution


def test_frames_stride_samples_every_nth_frame() -> None:
    indices = [idx for idx, _frame in frames(FIXTURES / "synth_tiny.mp4", stride=30)]
    assert indices == [0, 30, 60, 90, 120]


def test_frames_missing_file_raises() -> None:
    with pytest.raises(ValueError):
        next(frames(FIXTURES / "does_not_exist.mp4"))
