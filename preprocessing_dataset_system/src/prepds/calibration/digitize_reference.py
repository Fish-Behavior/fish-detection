"""Digitize the two reference behavior-strip PNGs into group-level targets - FR-008, PRD §9.5.5.

Produces, per panel (compound/concentration group), the fraction of plot
pixels in each of the six reference states. That aggregate is the primary
calibration target; per-subject-row extraction (and OCR of the tiny row
labels) is deliberately deferred - pytesseract isn't installed, and exact
subject-to-video matching from a low-resolution label crop isn't
guaranteed anyway (PRD §9.5.5's own caveat).

**Plot geometry is hardcoded** per image (two images, ten panels, never
re-parsed). An automatic panel/axis detector was tried first and found
fragile (it missed panels and picked up the row-label text), so the bounds
below were measured once from the images' colour-band structure and are
recorded here as constants. Each panel rectangle is shrunk 1px inside the
detected colour band: white counts as `Dead`, so a margin pixel sampled by
mistake silently inflates Dead. `x_zero_px`/`x_end_px` are the pixel
columns of the 0s and 1200s ticks, shared by every panel in an image -
never taken per row, since rows that end in Dead end white and a per-row
right edge would truncate exactly those rows.

Pixels that match no palette colour under `palette.classify_pixels`'s HSV
rules (blends between adjacent rows, antialiased edges) are rejected
rather than snapped; `rejected_fraction` is reported per panel so a bad
crop is visible. Proportions are over accepted pixels only.

Each panel's top/bottom `EDGE_INSET_PX` rows are skipped: the white margin
bleeds into them under blur and would inflate Dead (2.0% -> 0.6% on the
first vehicle panel). A vertical-stability filter (accept a pixel only if
its label is stable over +-2 rows) was tried to also suppress the phantom
pink at red/gray row boundaries, and abandoned: it removes boundary
artifacts but biases proportions toward frequent states (a frequent state
has more same-state neighbours, so more of its pixels survive), ~4.6pp
worst-case on a synthetic figure. The phantom pink is handled instead by a
tighter pink window in `palette.py`.

The restricted reference PNGs (`sample_data/`, Clarification C14) are read
at runtime only; nothing here is written back into the repo except the
derived numeric targets JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from prepds.models import BehaviorState
from prepds.palette import PALETTE_STATES, classify_pixels

REFERENCE_DURATION_S = 1200.0
EDGE_INSET_PX = 3

_STATES = PALETTE_STATES


@dataclass(frozen=True)
class PanelGeometry:
    label: str
    compound: str
    concentration_uM: int
    y0: int
    y1: int  # inclusive


@dataclass(frozen=True)
class ImageGeometry:
    filename: str
    concentration_uM: int
    x_zero_px: int
    x_end_px: int  # inclusive column of the 1200s tick
    panels: tuple[PanelGeometry, ...]


@dataclass(frozen=True)
class PanelTarget:
    label: str
    compound: str
    concentration_uM: int
    source_image: str
    proportions: dict[BehaviorState, float]
    rejected_fraction: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "compound": self.compound,
            "concentration_uM": self.concentration_uM,
            "source_image": self.source_image,
            "proportions": {state.name: value for state, value in self.proportions.items()},
            "rejected_fraction": self.rejected_fraction,
            "n_samples": self.n_samples,
        }


IMAGE_GEOMETRIES: tuple[ImageGeometry, ...] = (
    ImageGeometry(
        filename="sample_labeled_1.png",
        concentration_uM=30,
        x_zero_px=332,
        x_end_px=1553,
        panels=(
            PanelGeometry("VEH", "veh", 0, 246, 326),
            PanelGeometry("MDMA 30uM", "mdma", 30, 397, 475),
            PanelGeometry("METHYLONE 30uM", "methylone", 30, 547, 624),
            PanelGeometry("DOB 30uM", "dob", 30, 698, 776),
            PanelGeometry("FENT 30uM", "fentanyl", 30, 850, 929),
        ),
    ),
    ImageGeometry(
        filename="sample_labeled_2.png",
        concentration_uM=100,
        x_zero_px=345,
        x_end_px=1705,
        panels=(
            PanelGeometry("VEH", "veh", 0, 255, 345),
            PanelGeometry("MDMA 100uM", "mdma", 100, 411, 502),
            PanelGeometry("METHYLONE 100uM", "methylone", 100, 570, 658),
            PanelGeometry("DOB 100uM", "dob", 100, 726, 815),
            PanelGeometry("FENTANYL 100uM", "fentanyl", 100, 883, 970),
        ),
    ),
)


def pixel_x_to_seconds(x_px: float, *, x_zero_px: float, x_end_px: float) -> float:
    return (x_px - x_zero_px) / (x_end_px - x_zero_px) * REFERENCE_DURATION_S


def seconds_to_pixel_x(seconds: float, *, x_zero_px: float, x_end_px: float) -> float:
    return x_zero_px + seconds / REFERENCE_DURATION_S * (x_end_px - x_zero_px)


def digitize_panel(
    rgb: np.ndarray,
    geometry: ImageGeometry,
    panel: PanelGeometry,
) -> PanelTarget:
    height, width, _ = rgb.shape
    if not (0 <= panel.y0 < panel.y1 < height and 0 <= geometry.x_zero_px < geometry.x_end_px < width):
        raise ValueError(
            f"panel {panel.label!r} rectangle (y {panel.y0}-{panel.y1}, x {geometry.x_zero_px}-{geometry.x_end_px}) "
            f"falls outside the {width}x{height} image"
        )

    seconds = np.arange(int(REFERENCE_DURATION_S)) + 0.5
    columns = np.rint(
        [seconds_to_pixel_x(s, x_zero_px=geometry.x_zero_px, x_end_px=geometry.x_end_px) for s in seconds]
    ).astype(int)
    columns = np.clip(columns, geometry.x_zero_px, geometry.x_end_px)

    region = rgb[panel.y0 + EDGE_INSET_PX : panel.y1 + 1 - EDGE_INSET_PX][:, columns]
    if region.shape[0] == 0:
        raise ValueError(f"panel {panel.label!r} is too short for an inset of {EDGE_INSET_PX}px")
    labels = classify_pixels(region.reshape(-1, 3))
    accepted_labels = labels[labels >= 0]

    n_samples = len(labels)
    n_accepted = len(accepted_labels)
    if n_accepted == 0:
        raise ValueError(f"panel {panel.label!r}: no pixel matched any palette colour")

    counts = np.bincount(accepted_labels, minlength=len(_STATES))
    proportions = {state: float(counts[i]) / n_accepted for i, state in enumerate(_STATES)}
    return PanelTarget(
        label=panel.label,
        compound=panel.compound,
        concentration_uM=panel.concentration_uM,
        source_image=geometry.filename,
        proportions=proportions,
        rejected_fraction=1.0 - n_accepted / n_samples,
        n_samples=n_samples,
    )


def digitize_reference_images(sample_dir: Path) -> list[PanelTarget]:
    targets: list[PanelTarget] = []
    for geometry in IMAGE_GEOMETRIES:
        path = sample_dir / geometry.filename
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found - the restricted reference images (Clarification C14) must be present locally"
            )
        rgb = np.array(Image.open(path).convert("RGB"))
        targets.extend(digitize_panel(rgb, geometry, panel) for panel in geometry.panels)
    return targets


MAX_DUPLICATE_PANEL_TV = 0.05


def total_variation(a: dict[BehaviorState, float], b: dict[BehaviorState, float]) -> float:
    return 0.5 * sum(abs(a[state] - b[state]) for state in a)


def group_targets(
    targets: list[PanelTarget], *, max_duplicate_tv: float = MAX_DUPLICATE_PANEL_TV
) -> dict[tuple[str, int], dict[BehaviorState, float]]:
    """One target per (compound, concentration_uM). The two vehicle panels (one per
    figure) show the same control group, so they are averaged - but only after
    checking they actually agree: measured TV distance between them is ~2.8pp,
    consistent with digitization noise, and a large gap would mean they are NOT
    the same data and silently averaging them would hide it.
    """
    by_group: dict[tuple[str, int], list[PanelTarget]] = {}
    for target in targets:
        by_group.setdefault((target.compound, target.concentration_uM), []).append(target)

    groups: dict[tuple[str, int], dict[BehaviorState, float]] = {}
    for key, panels in by_group.items():
        for other in panels[1:]:
            distance = total_variation(panels[0].proportions, other.proportions)
            if distance > max_duplicate_tv:
                raise ValueError(
                    f"duplicate panels for group {key} disagree (TV distance {distance:.3f} > {max_duplicate_tv}): "
                    f"{panels[0].source_image} vs {other.source_image}"
                )
        groups[key] = {
            state: sum(p.proportions[state] for p in panels) / len(panels) for state in panels[0].proportions
        }
    return groups


def write_reference_targets(targets: list[PanelTarget], path: Path) -> None:
    payload = {
        "panels": [t.to_dict() for t in targets],
        "groups": [
            {
                "compound": compound,
                "concentration_uM": conc,
                "proportions": {state.name: value for state, value in proportions.items()},
            }
            for (compound, conc), proportions in sorted(group_targets(targets).items())
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    targets = digitize_reference_images(repo_root / "sample_data")
    output = Path(__file__).with_name("reference_targets.json")
    write_reference_targets(targets, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
