"""Digitize the reference ethogram figures into approximate per-subject timelines.

The reference PDF (FISH_REFERENCE_PDF) shows, on each ethogram page, one raster image
with stacked panels (one per group). Every panel has one thin row per subject, x axis
0..axis_seconds, colored by behavior state; white means no data. The subject labels on
the left overlap and cannot be read, so which subject each row is comes from a
hand-filled `mapping.yaml` (a template is written on the first run).

Steps:
1. the largest image of each ethogram page is saved as `page<N>.png`;
2. the plot area: columns / rows where palette-colored pixels dominate. Runs of such
   rows are the panels; all panels of a page share one x extent (mapped linearly to
   0..axis_seconds);
3. each panel is split into len(subject_ids) equal rows (mapping.yaml). The center line
   of each row is classified pixel by pixel to the nearest palette color (CIELAB
   distance, lightness weighted; farther than that color's max_color_distance = unknown;
   a pink run without a near-exact pink pixel takes its next-best color, see
   require_seeds) and resampled to 1 s bins by majority vote.

The bar charts of the summary page (seconds per state per group) are not read
automatically; `slide18_values.yaml` is a template to fill in by eye, and once filled it
is compared with the digitized group means (a check of the digitizer).

Outputs (in ``<FISH_OUTPUT_DIR>/reference/``, git-ignored):
    page<N>.png           the extracted ethogram images, and the bar-chart page rendered
    mapping.yaml          panel -> group and subject rows (template, filled in by hand)
    timelines.csv         subject_id, group, second, label (one row per second)
    group_means.csv       mean seconds per label per group
    digitize_check.png    original panel crops next to the redrawn timelines
    slide18_values.yaml   group x state -> seconds read by eye (template, optional)
    slide18_check.csv     those values next to the digitized group means
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pymupdf
import yaml

from fishbehavior.catalog import parse_subject_number
from fishbehavior.config import ConfigError, Settings
from fishbehavior.labeling import STATES

NO_DATA, UNKNOWN = "no_data", "unknown"
PALETTE_NAMES = (*STATES, NO_DATA)  # palette order = class index; UNKNOWN is the index after them
LABEL_NAMES = (*PALETTE_NAMES, UNKNOWN)
MAPPING_FILE, SLIDE18_FILE = "mapping.yaml", "slide18_values.yaml"
TIMELINES_FILE, MEANS_FILE, CHECK_FILE, SLIDE18_CHECK_FILE = (
    "timelines.csv", "group_means.csv", "digitize_check.png", "slide18_check.csv")
VOTES_PER_SECOND = 10  # sub-samples per second for the majority vote (plots have < or > 1 px per second)

MAPPING_HEADER = """\
# Which subjects are the rows of each ethogram panel (see README, "Reference ethograms").
# Fill in every panel, then run `python -m fishbehavior reference` again.
#   page, panel  where the panel is: PDF page, and panel number from the top of page<N>.png
#   rows         number of subject rows. ESTIMATED from the image: count the rows in
#                page<N>.png and correct it if needed
#   group        the panel's group as it should appear in the outputs, e.g. VEH or COMPOUND_A 30uM
#   subject_ids  the panel's subjects from TOP to BOTTOM, exactly `rows` of them,
#                e.g. [F_0048, F_0045, 42]; "42", "0042" and "F_0042" all mean subject 42
#   skip         true for a panel that repeats another one (e.g. the same control group on
#                both pages); it is not digitized and needs no group or subject_ids
"""
SLIDE18_HEADER = """\
# Optional: mean seconds per state per group, read BY EYE from the bar charts (page{page}.png).
# Leave null where you cannot read a bar. When any value is filled in, `reference` compares
# them with the digitized means in group_means.csv (slide18_check.csv): a check of the
# digitizer, not an input to it. Group names must match mapping.yaml.
"""


@dataclass
class Panel:
    """One panel of an ethogram image: its pixel box (sub-pixel edges) and estimated row count."""

    page: int
    index: int  # 1 = top panel of the page
    y0: float
    y1: float
    x0: float  # the page's shared plot extent: x0 = 0 s, x1 = axis_seconds
    x1: float
    est_rows: int


@dataclass
class ReferenceResult:
    """What `run_reference` did, for the command-line summary."""

    reference_dir: Path
    panels: list[Panel]
    mapping_created: bool = False
    cached: bool = False
    timelines: pd.DataFrame | None = None
    group_means: pd.DataFrame | None = None
    skipped_panels: int = 0
    slide18_created: bool = False
    slide18_check: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------


def reference_params(settings: Settings) -> dict[str, Any]:
    """The `reference:` settings, checked so a typo in a state name fails early."""
    params = settings.params["reference"]
    for key in ("palette", "max_color_distance"):
        names = set(params[key])
        if names != set(PALETTE_NAMES):
            raise ConfigError(f"reference.{key} must have exactly the entries {', '.join(PALETTE_NAMES)}; "
                              f"got {', '.join(sorted(names))}")
    unknown = set(params["seed_color_distance"]) - set(PALETTE_NAMES)
    if unknown:
        raise ConfigError(f"reference.seed_color_distance has unknown colors {sorted(unknown)}; "
                          f"use {', '.join(PALETTE_NAMES)}")
    return params


# ---------------------------------------------------------------------------
# 2. Images from the PDF
# ---------------------------------------------------------------------------


def pdf_page(doc: pymupdf.Document, number: int, pdf: Path) -> pymupdf.Page:
    """1-based page `number`, or a clear error when the PDF is shorter."""
    if not 1 <= number <= doc.page_count:
        raise ConfigError(f"{pdf} has {doc.page_count} pages, no page {number}; check the reference: page settings")
    return doc[number - 1]


def largest_image(doc: pymupdf.Document, number: int, pdf: Path) -> np.ndarray:
    """The largest embedded raster image of a page, as an RGB array (the ethogram figure)."""
    infos = pdf_page(doc, number, pdf).get_images(full=True)
    if not infos:
        raise ConfigError(f"page {number} of {pdf} has no embedded image; check reference.ethogram_pages")
    xref = max(infos, key=lambda info: info[2] * info[3])[0]  # info = (xref, smask, width, height, ...)
    pix = pymupdf.Pixmap(doc, xref)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)  # drop transparency
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)  # gray / CMYK figures -> RGB
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()


def save_rgb(path: Path, image: np.ndarray) -> None:
    """Write an RGB array as PNG (OpenCV expects BGR)."""
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def load_rgb(path: Path) -> np.ndarray:
    """Read a PNG as an RGB array."""
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# 3. Colors -> classes, plot area and panels
# ---------------------------------------------------------------------------


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB (L 0-100, a/b about -128..127), where distances match perceived color differences."""
    return cv2.cvtColor(np.asarray(rgb, np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def classify(image: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pixel: index of the nearest palette color (PALETTE_NAMES order, or UNKNOWN's index if
    none is close), the distance to it, and the nearest color that needs no seed (see require_seeds).

    Lightness differences are weighted up: JPEG keeps brightness sharp but smears color
    over ~2 px, so a thin gray row under a pink one turns pinkish while staying gray-bright.
    """
    palette = to_lab(np.array([[params["palette"][name] for name in PALETTE_NAMES]], np.uint8))[0]
    weight = np.array([float(params["lightness_weight"]), 1.0, 1.0], np.float32)
    distance = np.linalg.norm((to_lab(image)[:, :, None, :] - palette[None, None]) * weight, axis=-1)
    limit = np.array([float(params["max_color_distance"][name]) for name in PALETTE_NAMES])

    def match(distance: np.ndarray) -> np.ndarray:
        """Nearest color, or unknown when it is too far (JPEG edge mixes, text)."""
        classes = distance.argmin(-1)
        classes[distance.min(-1) > limit[classes]] = len(PALETTE_NAMES)
        return classes

    seeded = [PALETTE_NAMES.index(name) for name in params["seed_color_distance"]]
    others = distance.copy()
    others[..., seeded] = np.inf  # a rejected pinkish pixel takes its next-best color
    return match(distance), distance.min(-1), match(others)


def require_seeds(line: np.ndarray, nearest: np.ndarray, fallback: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Along one pixel line: runs of a `seed_color_distance` color with no pixel that close take their next-best color.

    Hysteresis, as in edge detection: pink is almost a red + white mix, so the blurred
    edge of a red bout (next to gray or white) looks pinkish for 1-3 px, but never close
    to the palette pink, while a real lorr stretch always has some near-exact pink pixels.
    """
    line = line.copy()
    for name, seed in params["seed_color_distance"].items():
        for start, stop in runs_of(line == PALETTE_NAMES.index(name)):
            if nearest[start:stop].min() > float(seed):
                line[start:stop] = fallback[start:stop]
    return line


def runs_of(mask: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    """[start, stop) of each run of True, joining runs separated by at most `max_gap` False."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    return [(int(part[0]), int(part[-1]) + 1) for part in np.split(idx, breaks + 1)]


def refine_edges(profile: np.ndarray, start: int, stop: int) -> tuple[float, float]:
    """Sub-pixel edges of a run: a blurred edge pixel counts by its share of the run's typical level.

    JPEG blurs edges over 1-2 px; with rows only ~4 px tall, an edge off by one pixel
    would move every row center toward its neighbour.
    """
    level = float(np.median(profile[start:stop]))
    inside = np.flatnonzero(profile[start:stop] >= level / 2) + start
    first, last = int(inside[0]), int(inside[-1])
    before = profile[first - 1] / level if first > 0 else 0.0
    after = profile[last + 1] / level if last + 1 < len(profile) else 0.0
    return first - min(before, 1.0), last + 1 + min(after, 1.0)


def estimate_rows(image: np.ndarray, y0: float, y1: float, c0: int, c1: int) -> int:
    """Row count whose boundaries best match where the color changes vertically (vs row centers).

    Uses the raw RGB change between pixel rows (JPEG blur turns a boundary into
    in-between colors that no palette class catches). A true row count puts its
    boundaries on changes and its centers on steady color; half the count puts
    "centers" on boundaries and twice the count puts "boundaries" on centers, so both
    score lower.
    """
    top = int(np.floor(y0))
    band = image[top:int(np.ceil(y1)), c0:c1].astype(float)
    change = np.abs(np.diff(band, axis=0)).sum(2).mean(1)  # change[j] = edge at y = top + j + 1
    edges = top + 1.0 + np.arange(len(change))
    height = y1 - y0
    best, best_n = -np.inf, 1
    for n in range(3, int(height // 2) + 1):  # rows of at least 2 px
        step = height / n
        at_bounds = np.interp(y0 + step * np.arange(1, n), edges, change).mean()
        at_centers = np.interp(y0 + step * (np.arange(n) + 0.5), edges, change).mean()
        if at_bounds - at_centers > best:
            best, best_n = at_bounds - at_centers, n
    return best_n


def find_panels(image: np.ndarray, classes: np.ndarray, page: int, params: dict[str, Any]) -> list[Panel]:
    """Plot columns and panel rows of one ethogram image, top panel first."""
    colored = classes < len(STATES)  # a behavior color: finds the plot, ignoring text
    ink = classes != PALETTE_NAMES.index(NO_DATA)  # anything not white: places the edges (blurred edge pixels too)
    height = classes.shape[0]
    # Plot columns: the longest run of columns with enough colored pixels compared with the most
    # colored column (the subject labels and the legend have far fewer).
    column_count = colored.sum(0)
    column_runs = runs_of(column_count >= float(params["plot_min_column_fraction"]) * column_count.max())
    if not column_runs:
        raise ConfigError(f"page {page}: no plot area found; check reference.palette and ethogram_pages")
    c0, c1 = max(column_runs, key=lambda run: run[1] - run[0])
    # Panel rows: runs of rows mostly colored inside the plot columns; small white gaps are joined.
    max_gap = int(round(float(params["panel_max_gap_fraction"]) * height))
    min_height = float(params["panel_min_height_fraction"]) * height
    row_runs = [run for run in runs_of(colored[:, c0:c1].mean(1) >= float(params["panel_min_row_fraction"]), max_gap)
                if run[1] - run[0] >= min_height]  # drops legend boxes
    if not row_runs:
        raise ConfigError(f"page {page}: no panels found; check the reference: settings")
    # One shared x extent for the page (all panels have the same axis), from the panel rows only.
    in_panels = np.zeros(height, bool)
    for r0, r1 in row_runs:
        in_panels[r0:r1] = True
    x0, x1 = refine_edges(ink[in_panels].mean(0), c0, c1)
    row_ink = ink[:, c0:c1].mean(1)
    panels = []
    for index, (r0, r1) in enumerate(row_runs, start=1):
        y0, y1 = refine_edges(row_ink, r0, r1)
        panels.append(Panel(page, index, y0, y1, x0, x1, estimate_rows(image, y0, y1, c0, c1)))
    return panels


# ---------------------------------------------------------------------------
# 4. mapping.yaml
# ---------------------------------------------------------------------------


def mapping_template(panels: list[Panel]) -> str:
    """The mapping.yaml text for the user to fill in: one entry per panel, estimated rows."""
    entries = [{"page": p.page, "panel": p.index, "rows": p.est_rows, "group": "", "subject_ids": [], "skip": False}
               for p in panels]
    return MAPPING_HEADER + "\n" + yaml.safe_dump({"panels": entries}, sort_keys=False)


def subject_text(value: Any, where: str) -> str:
    """'42', '0042', 'F_0042' -> '0042' (the catalog's zero-padded subject_id)."""
    try:
        return f"{parse_subject_number(str(value)):04d}"
    except ConfigError:
        raise ConfigError(f"{where}: {value!r} is not a subject id") from None


def load_mapping(path: Path, panels: list[Panel]) -> list[dict[str, Any]]:
    """The filled-in mapping, one entry per detected panel (page/panel order), or ConfigError listing every problem."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = {(int(e["page"]), int(e["panel"])): e for e in data.get("panels") or []}
    detected = [(p.page, p.index) for p in panels]
    if set(entries) != set(detected):
        raise ConfigError(f"{path} lists panels {sorted(entries)} but the figures have {detected} (page, panel); "
                          f"delete it to get a new template")
    mapping, problems, empty, seen = [], [], [], {}
    for key in detected:
        entry, where = entries[key], f"page {key[0]} panel {key[1]}"
        if entry.get("skip"):
            mapping.append({"page": key[0], "panel": key[1], "skip": True})
            continue
        ids = [subject_text(value, where) for value in entry.get("subject_ids") or []]
        rows = int(entry.get("rows") or 0)
        if not entry.get("group") or not ids:
            empty.append(key)
        elif len(ids) != rows:
            problems.append(f"{where}: {len(ids)} subject_ids but rows: {rows}. Count the rows in "
                            f"page{key[0]}.png, then fix subject_ids or rows so they agree")
        for subject in ids:
            if subject in seen:  # would be counted twice in the group means
                problems.append(f"{where}: subject {subject} is also in {seen[subject]}; "
                                f"set skip: true on a repeated panel")
            seen[subject] = where
        mapping.append({"page": key[0], "panel": key[1], "skip": False, "group": str(entry["group"]),
                        "subject_ids": ids})
    if empty:  # one clause for all unfilled panels, e.g. "page 16 panels 1, 2, 3"
        pages = {page: [str(panel) for pg, panel in empty if pg == page] for page, _ in empty}
        listed = "; ".join(f"page {page} panel{'s' * (len(panels) > 1)} {', '.join(panels)}"
                           for page, panels in pages.items())
        problems.insert(0, f"not filled in yet: {listed}. Give each a group and its subject_ids "
                           f"(top -> bottom), or skip: true (README, step 7)")
    if problems:
        raise ConfigError(f"{path}: " + "; ".join(problems))
    return mapping


# ---------------------------------------------------------------------------
# 5. Rows -> 1 s timelines
# ---------------------------------------------------------------------------


def row_centers(panel: Panel, n_rows: int) -> np.ndarray:
    """Pixel row through the middle of each of the panel's `n_rows` equal rows, top first."""
    step = (panel.y1 - panel.y0) / n_rows
    return np.floor(panel.y0 + step * (np.arange(n_rows) + 0.5)).astype(int)


def to_seconds(line: np.ndarray, x0: float, x1: float, seconds: int) -> np.ndarray:
    """Class per second [s, s+1) along one pixel line: majority of VOTES_PER_SECOND evenly spaced samples.

    Weighs each pixel by how much of the second it covers, whether a second spans
    less or more than one pixel. Unknown pixels abstain: a second is unknown only when
    no pixel of it matched a palette color.
    """
    t = (np.arange(seconds * VOTES_PER_SECOND) + 0.5) / VOTES_PER_SECOND
    columns = np.clip(np.floor(x0 + t / seconds * (x1 - x0)).astype(int), 0, len(line) - 1)
    votes = line[columns].reshape(seconds, VOTES_PER_SECOND)
    counts = (votes[:, :, None] == np.arange(len(PALETTE_NAMES))).sum(1)  # palette colors only
    return np.where(counts.any(1), counts.argmax(1), len(PALETTE_NAMES))


def digitize_panel(classes: np.ndarray, nearest: np.ndarray, fallback: np.ndarray, panel: Panel, n_rows: int,
                   params: dict[str, Any]) -> np.ndarray:
    """Class per row (top first) and second: an (n_rows, axis_seconds) array (arrays from classify)."""
    seconds = int(params["axis_seconds"])
    return np.stack([to_seconds(require_seeds(classes[y], nearest[y], fallback[y], params), panel.x0, panel.x1,
                                seconds) for y in row_centers(panel, n_rows)])


def timelines_table(digitized: list[dict[str, Any]], seconds: int) -> pd.DataFrame:
    """Long table: one row per subject and second."""
    frames = []
    for item in digitized:
        n = len(item["subject_ids"])
        frames.append(pd.DataFrame({
            "subject_id": np.repeat(item["subject_ids"], seconds),
            "group": item["group"],
            "second": np.tile(np.arange(seconds), n),
            "label": np.asarray(LABEL_NAMES, dtype=object)[item["labels"].ravel()],
        }))
    return pd.concat(frames, ignore_index=True)


def group_means(timelines: pd.DataFrame) -> pd.DataFrame:
    """Mean seconds per label per group (mean over the group's subjects), groups in mapping order."""
    per_subject = pd.crosstab([timelines["group"], timelines["subject_id"]], timelines["label"])
    per_subject = per_subject.reindex(columns=list(LABEL_NAMES), fill_value=0)
    order = list(dict.fromkeys(timelines["group"]))
    means = per_subject.groupby(level="group").mean().reindex(order)
    means.insert(0, "n_subjects", per_subject.groupby(level="group").size().reindex(order))
    return means.round(1).reset_index().rename_axis(columns=None)


# ---------------------------------------------------------------------------
# 6. The check image
# ---------------------------------------------------------------------------

PLOT_W, ROW_H, LABEL_W, PAD = 900, 12, 64, 16  # redrawn plot width, px per subject row, label column, margin
FONT = cv2.FONT_HERSHEY_SIMPLEX
INK, AXIS = (30, 30, 30), (120, 120, 120)
UNKNOWN_RGB = (0, 0, 0)  # drawn for pixels that matched no palette color


def label_colors(params: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    """RGB per label for every drawing (digitize check, plots): the figures' legend palette
    (`reference.palette`, the one place colors are defined) plus black for unknown."""
    return {**{name: tuple(int(c) for c in params["palette"][name]) for name in PALETTE_NAMES}, UNKNOWN: UNKNOWN_RGB}


def put_text(canvas: np.ndarray, text: str, x: float, y: float, scale: float = 0.4,
             align: str = "left", bold: bool = False) -> None:
    """Text with its baseline at y; align left / center / right at x."""
    thickness = 2 if bold else 1
    width = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
    x = {"left": x, "center": x - width / 2, "right": x - width}[align]
    cv2.putText(canvas, text, (int(x), int(y)), FONT, scale, INK, thickness, cv2.LINE_AA)


def draw_axis(canvas: np.ndarray, x: int, y: int, seconds: int) -> None:
    """x axis under a plot: at most 12 ticks at a round step (100 s for 1200 s, 200 s for 2200 s)."""
    step = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10**9) if seconds / s <= 12)
    cv2.line(canvas, (x, y), (x + PLOT_W - 1, y), AXIS, 1)
    for s in range(0, seconds + 1, step):
        px = x + round(s / seconds * (PLOT_W - 1))
        cv2.line(canvas, (px, y), (px, y + 4), AXIS, 1)
        put_text(canvas, str(s), px, y + 16, 0.35, "center")


def draw_check(images: dict[int, np.ndarray], digitized: list[dict[str, Any]], params: dict[str, Any]) -> np.ndarray:
    """Every digitized panel as [original crop | subject ids | redrawn timelines], rows aligned; RGB.

    The redrawn side looks like the reference: subjects top to bottom, time 0..axis
    left to right, colored by state, one titled panel per group, legend at the bottom.
    """
    seconds = int(params["axis_seconds"])
    colors = np.array([label_colors(params)[name] for name in LABEL_NAMES], np.uint8)
    left, right = PAD, PAD + PLOT_W + 2 * PAD + LABEL_W  # x of the original crop / the redrawn plot
    width = right + PLOT_W + PAD
    header, legend_h = 56, 64
    block_h = [26 + len(item["subject_ids"]) * ROW_H + 28 for item in digitized]
    canvas = np.full((header + sum(block_h) + legend_h, width, 3), 255, np.uint8)

    put_text(canvas, "Digitized reference ethograms", PAD, 22, 0.6, bold=True)
    put_text(canvas, "original figure crop", left + PLOT_W / 2, 48, 0.45, "center", bold=True)
    put_text(canvas, "digitized, 1 s per column", right + PLOT_W / 2, 48, 0.45, "center", bold=True)
    y = header
    for item, height in zip(digitized, block_h):
        panel, n = item["panel"], len(item["subject_ids"])
        put_text(canvas, f"{item['group']}   (page {panel.page}, panel {panel.index}, {n} subjects)",
                 right + PLOT_W / 2, y + 18, 0.5, "center", bold=True)
        top, body_h = y + 26, n * ROW_H
        # Original crop, stretched so its rows line up with the redrawn rows.
        crop = images[panel.page][int(np.floor(panel.y0)):int(np.ceil(panel.y1)),
                                  int(np.floor(panel.x0)):int(np.ceil(panel.x1))]
        canvas[top:top + body_h, left:left + PLOT_W] = cv2.resize(crop, (PLOT_W, body_h), interpolation=cv2.INTER_AREA)
        # Redrawn: one color per second, each subject row ROW_H tall.
        canvas[top:top + body_h, right:right + PLOT_W] = cv2.resize(
            colors[item["labels"]], (PLOT_W, body_h), interpolation=cv2.INTER_NEAREST)
        for row, subject in enumerate(item["subject_ids"]):
            put_text(canvas, subject, right - 6, top + row * ROW_H + ROW_H - 2, 0.35, "right")
        draw_axis(canvas, left, top + body_h, seconds)
        draw_axis(canvas, right, top + body_h, seconds)
        y += height

    put_text(canvas, "Duration (s)", right + PLOT_W / 2, y + 12, 0.45, "center")
    x = PAD
    for name, color in zip(LABEL_NAMES, colors):  # legend, like the reference's
        cv2.rectangle(canvas, (x, y + 30), (x + 16, y + 46), tuple(int(c) for c in color), -1)
        cv2.rectangle(canvas, (x, y + 30), (x + 16, y + 46), AXIS, 1)  # outline, so white "no data" shows
        put_text(canvas, name.replace("_", " "), x + 22, y + 43, 0.45)
        x += 30 + cv2.getTextSize(name, FONT, 0.45, 1)[0][0] + 16
    return canvas


# ---------------------------------------------------------------------------
# 7. Slide 18 (bar charts): template and comparison
# ---------------------------------------------------------------------------


def slide18_template(groups: list[str], page: int) -> str:
    """slide18_values.yaml text: every group x state set to null."""
    values = {group: {state: None for state in STATES} for group in groups}
    return SLIDE18_HEADER.format(page=page) + "\n" + yaml.safe_dump(values, sort_keys=False)


def compare_slide18(path: Path, means: pd.DataFrame) -> pd.DataFrame | None:
    """Filled-in slide-18 values next to the digitized means; None while nothing is filled in."""
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    digitized = means.set_index("group")
    rows = []
    for group, states in values.items():
        if group not in digitized.index:
            raise ConfigError(f"{path}: group {group!r} is not in mapping.yaml ({', '.join(digitized.index)})")
        for state, seconds in (states or {}).items():
            if state not in STATES:
                raise ConfigError(f"{path}: {group}: unknown state {state!r}; use {', '.join(STATES)}")
            if seconds is not None:
                ours = float(digitized.loc[group, state])
                rows.append({"group": group, "state": state, "slide18_s": float(seconds), "digitized_s": ours,
                             "difference_s": round(ours - float(seconds), 1)})
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# 8. The step
# ---------------------------------------------------------------------------


def reference_dir(settings: Settings) -> Path:
    """Where this step writes (git-ignored, like every output)."""
    return settings.paths.output_dir / "reference"


def page_images(settings: Settings, params: dict[str, Any], out: Path, force: bool) -> dict[int, np.ndarray]:
    """Ethogram images by page (extracted once to page<N>.png), plus the bar-chart page rendered for reading by eye."""
    pages = [int(p) for p in params["ethogram_pages"]]
    barchart = int(params["barchart_page"])
    paths = {page: out / f"page{page}.png" for page in pages}
    barchart_path = out / f"page{barchart}.png"
    if force or not all(path.is_file() for path in [*paths.values(), barchart_path]):
        pdf = settings.require("reference_pdf")
        with pymupdf.open(pdf) as doc:
            for page, path in paths.items():
                save_rgb(path, largest_image(doc, page, pdf))
            pdf_page(doc, barchart, pdf).get_pixmap(dpi=150).save(barchart_path)  # several charts: the whole page
    return {page: load_rgb(path) for page, path in paths.items()}


def run_reference(settings: Settings, force: bool = False) -> ReferenceResult:
    """Extract the figures, write/read mapping.yaml, digitize, and compare with slide 18 when filled in."""
    params = reference_params(settings)
    seconds = int(params["axis_seconds"])
    out = reference_dir(settings)
    out.mkdir(parents=True, exist_ok=True)

    images = page_images(settings, params, out, force)
    matched = {page: classify(image, params) for page, image in images.items()}  # (classes, distance, fallback)
    panels = [panel for page in images for panel in find_panels(images[page], matched[page][0], page, params)]
    result = ReferenceResult(out, panels)

    mapping_path = out / MAPPING_FILE
    if not mapping_path.is_file():
        mapping_path.write_text(mapping_template(panels), encoding="utf-8")  # never overwritten: the user fills it
        result.mapping_created = True
        return result
    mapping = load_mapping(mapping_path, panels)
    result.skipped_panels = sum(entry["skip"] for entry in mapping)

    # Cache: redo when forced, when an output is missing, or when mapping.yaml was edited since.
    outputs = [out / TIMELINES_FILE, out / MEANS_FILE, out / CHECK_FILE]
    if (not force and all(path.is_file() for path in outputs)
            and outputs[0].stat().st_mtime >= mapping_path.stat().st_mtime):
        result.cached = True
        result.timelines = pd.read_csv(outputs[0], dtype={"subject_id": str})
        result.group_means = pd.read_csv(outputs[1])
    else:
        by_key = {(p.page, p.index): p for p in panels}
        digitized = []
        for entry in mapping:
            if not entry["skip"]:
                panel = by_key[entry["page"], entry["panel"]]
                labels = digitize_panel(*matched[panel.page], panel, len(entry["subject_ids"]), params)
                digitized.append({**entry, "panel": panel, "labels": labels})
        result.timelines = timelines_table(digitized, seconds)
        result.group_means = group_means(result.timelines)
        result.timelines.to_csv(outputs[0], index=False)
        result.group_means.to_csv(outputs[1], index=False)
        save_rgb(outputs[2], draw_check(images, digitized, params))

    slide18_path = out / SLIDE18_FILE
    if not slide18_path.is_file():
        groups = list(result.group_means["group"])
        slide18_path.write_text(slide18_template(groups, int(params["barchart_page"])), encoding="utf-8")
        result.slide18_created = True
    else:
        result.slide18_check = compare_slide18(slide18_path, result.group_means)
        if result.slide18_check is not None:
            result.slide18_check.to_csv(out / SLIDE18_CHECK_FILE, index=False)
    return result
