"""Background model + waterline/ROI detection (PRD §9.5.3) - supports FR-005.

Classical, GPU-free approach per §9.2 "simplicity first": static camera and
plain background mean a per-video background estimate from a sparse frame
sample is enough, without a learned or per-frame-adaptive model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from prepds.video_io import frames

DEFAULT_BACKGROUND_SAMPLE_STRIDE = 30  # "every 30th frame" per PRD §9.5.3
# Ignore this many rows at the very top/bottom when picking the strongest
# gradient, so a lighting vignette or compression artifact at the frame edge
# is never mistaken for the real waterline.
WATERLINE_EDGE_MARGIN_PX = 5


def estimate_background(video_path: Path, stride: int = DEFAULT_BACKGROUND_SAMPLE_STRIDE) -> np.ndarray:
    """Per-video background estimate: per-pixel median over a sparse frame sample.

    A moving foreground object (the fish) only covers any given pixel in a
    small minority of the sampled frames, so the median at that pixel is the
    background value, not the object's - this is what makes later foreground
    extraction (Phase 4 tracking.py) work via simple frame differencing.
    """
    sampled = [frame for _idx, frame in frames(video_path, stride=stride)]
    if not sampled:
        raise ValueError(f"No frames could be read from {video_path}")
    stacked = np.stack(sampled, axis=0)  # (n_samples, height, width, channels)
    # np.round before the uint8 cast: a plain .astype(np.uint8) truncates a
    # median that lands on a .5 boundary (even sample count) instead of
    # rounding, biasing the background reference down by up to 1 gray level.
    return np.round(np.median(stacked, axis=0)).astype(np.uint8)


def detect_waterline(background: np.ndarray) -> int:
    """Return the row index of the single strongest horizontal edge in `background`.

    Simple row-wise gradient peak (PRD §9.5.3): convert to grayscale, take
    the mean intensity of each row, and return the row where that mean
    changes most sharply from the row above it - a plain, flat background
    with no real step will still return *some* row (peak of whatever noise
    exists), which is expected for degenerate input.

    IMPORTANT caveat found during T029's real-footage spot-check (48 videos
    spanning all 16 matched compounds, see docs/strain_tracking_notes.md):
    the detected edge is always strong relative to the frame's own noise
    floor (SNR 9.6-110.5x the median row-to-row gradient across the sample -
    this is not a low-confidence/noisy-signal problem). But *which physical
    feature* that strongest edge corresponds to is not consistent across
    videos - the peak row lands anywhere from ~1% to ~90% of frame height,
    clustering bimodally near the very top (beaker rim / possible true water
    surface) in roughly a quarter of sampled videos and in the lower half
    (beaker bottom, confirmed by direct visual inspection of 3 videos) in
    most others. This function name is therefore aspirational, not yet
    validated: it finds *a* strong edge, not confirmed to be *the* water
    surface, on this camera setup/framing. Do not treat its return value as
    a trusted water-surface reference without the further per-video visual
    validation T029 recommends deferring to Phase 4's T035 debug-overlay
    spot-check (which can look at actual fish position, not just the
    background frame).
    """
    grayscale = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.float64)
    row_means = grayscale.mean(axis=1)
    # row_gradient[i] is the transition BETWEEN row i and row i+1 (np.diff
    # shortens by one), so the row that "changed from the row above it" is
    # i+1, not i - returning `lo + peak_offset` without the `+ 1` was an
    # off-by-one against this function's own docstring, caught by code
    # review: a step at row 50 was returning 49, only passing the original
    # test because of its +-2px tolerance.
    row_gradient = np.abs(np.diff(row_means))
    n_transitions = row_gradient.shape[0]  # == height - 1
    # Symmetric margin in transition-index space (previously mixed row-space
    # height with transition-space slicing, excluding 5 transitions at the
    # low end but only 4 at the high end).
    lo, hi = WATERLINE_EDGE_MARGIN_PX, n_transitions - WATERLINE_EDGE_MARGIN_PX
    if hi <= lo:
        raise ValueError(f"Frame too short ({row_means.shape[0]}px) to detect a waterline")
    peak_offset = int(np.argmax(row_gradient[lo:hi]))
    return lo + peak_offset + 1
