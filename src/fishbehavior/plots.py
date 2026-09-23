"""Figures for review and presentation, in the style of the reference slides.

Outputs (``<FISH_OUTPUT_DIR>/plots/``, git-ignored):

    ethogram_<group>.png   one per workbook group (compound + concentration): one row per
                           subject (highest number on top, like the reference), x = 0..axis
                           seconds, colored by label, white = untracked / no data. When any
                           subject of the group has a digitized reference row, the reference
                           is drawn next to ours with the rows lined up.
    bars_<state>.png       seconds in that state per group: mean +/- SEM with every subject as a dot
    overlay/<subject>_<start>-<end>s.mp4
                           (--overlay) the original frames with ROI (green), waterline (blue),
                           fitted ellipse (yellow), track point (red), the current label, and a
                           timeline bar with a moving cursor under the picture

Colors come from the one palette in `reference.label_colors` (the legend colors of the figures).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")  # no display needed: works headless (servers, Colab, tests)
import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend choice
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from fishbehavior.calibrate import per_second  # noqa: E402
from fishbehavior.catalog import STATUS_MATCHED  # noqa: E402
from fishbehavior.config import ConfigError, Settings  # noqa: E402
from fishbehavior.labeling import STATES, SUMMARY_FILE, UNTRACKED, labeled_bins_path, labels_dir  # noqa: E402
from fishbehavior.priors import TIMELINES_FILE  # noqa: E402
from fishbehavior.reference import NO_DATA, label_colors, reference_dir  # noqa: E402
from fishbehavior.tracking import track_path, tracks_dir  # noqa: E402
from fishbehavior.video import iter_frames  # noqa: E402

log = logging.getLogger(__name__)
LEGEND = (*STATES, UNTRACKED)  # labels shown in our legends (untracked drawn white like no data)


def plots_dir(settings: Settings) -> Path:
    """The step's output folder."""
    return settings.paths.output_dir / "plots"


def group_of(trials: pd.DataFrame) -> pd.Series:
    """Workbook group per subject: compound + concentration as written (e.g. 'Veh 1% DMSO')."""
    return (trials["compound"].astype(str) + " " + trials["concentration_raw"].astype(str)).set_axis(
        trials["subject_id"])


def file_safe(name: str) -> str:
    """A group name usable in a file name on every OS."""
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", name).strip("_")


def rgb(colors: dict[str, tuple[int, int, int]], labels: np.ndarray) -> np.ndarray:
    """(..., 3) float image for an array of labels; untracked and '' (after the video) white."""
    lookup = {**colors, UNTRACKED: colors[NO_DATA], "": colors[NO_DATA]}
    return np.array([[lookup.get(x, colors[NO_DATA]) for x in row] for row in labels], float) / 255.0


# ---------------------------------------------------------------------------
# 1. Ethograms
# ---------------------------------------------------------------------------


def draw_ethogram(title: str, subjects: list[str], ours: np.ndarray, reference: np.ndarray | None,
                  has_reference: list[bool], colors: dict[str, tuple[int, int, int]], seconds: int,
                  path: Path, dpi: int) -> None:
    """One group: our rows (and the reference rows next to them), legend at the bottom."""
    panels = [("video labels", ours)] + ([("digitized reference", reference)] if reference is not None else [])
    height = 1.2 + 0.18 * len(subjects)  # inches: thin rows like the reference
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), height), squeeze=False, sharey=True)
    for ax, (name, labels) in zip(axes[0], panels):
        ax.imshow(rgb(colors, labels), aspect="auto", interpolation="nearest",
                  extent=(0, seconds, len(subjects), 0))
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Duration (s)", fontsize=8)
        ax.set_xlim(0, seconds)
        ax.tick_params(labelsize=7)
    marks = ["" if ok or reference is None else "  (no ref)" for ok in has_reference]
    axes[0][0].set_yticks(np.arange(len(subjects)) + 0.5, [s + m for s, m in zip(subjects, marks)])
    fig.suptitle(title, fontsize=10, fontweight="bold")
    shown = [*LEGEND, "unknown"] if reference is not None else list(LEGEND)
    handles = [Patch(facecolor=np.array(colors[NO_DATA if n == UNTRACKED else n]) / 255, edgecolor="0.4",
                     label=n.replace("_", " ")) for n in shown]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.3 / height, 1, 1))  # keep 0.3 in at the bottom for the legend
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def our_seconds(settings: Settings, subject_id: str, seconds: int) -> np.ndarray:
    """Our label per second 0..seconds-1 ('' where the video is shorter)."""
    labeled = pd.read_csv(labeled_bins_path(labels_dir(settings), subject_id), usecols=["label"])
    return per_second(labeled, float(settings.params["features"]["bin_s"]), seconds)


def ethograms(settings: Settings, trials: pd.DataFrame, wanted: set[str] | None) -> list[Path]:
    """ethogram_<group>.png for every group (or the groups of `wanted` subjects)."""
    params, ref_params = settings.params["plots"], settings.params["reference"]
    seconds, colors = int(ref_params["axis_seconds"]), label_colors(ref_params)
    groups = group_of(trials)
    labeled = [s for s in trials.loc[trials["video_status"] == STATUS_MATCHED, "subject_id"]
               if labeled_bins_path(labels_dir(settings), s).is_file()]
    timelines_path = reference_dir(settings) / TIMELINES_FILE
    reference = {}
    if timelines_path.is_file():
        table = pd.read_csv(timelines_path, dtype={"subject_id": str}).sort_values(["subject_id", "second"])
        reference = {s: rows["label"].to_numpy(object)[:seconds] for s, rows in table.groupby("subject_id")}
    chosen = set(groups[list(wanted)]) if wanted else set(groups[labeled])
    out, paths = plots_dir(settings), []
    for group in sorted(chosen):
        subjects = sorted([s for s in labeled if groups[s] == group], key=int, reverse=True)  # highest on top
        if not subjects:
            continue
        ours = np.array([our_seconds(settings, s, seconds) for s in subjects], dtype=object)
        has_ref = [s in reference for s in subjects]
        ref = None
        if any(has_ref):
            blank = np.full(seconds, NO_DATA, dtype=object)
            ref = np.array([np.r_[reference[s], blank][:seconds] if s in reference else blank for s in subjects],
                           dtype=object)
        path = out / f"ethogram_{file_safe(group)}.png"
        draw_ethogram(f"{group}  ({len(subjects)} subjects)", subjects, ours, ref, has_ref, colors, seconds,
                      path, int(params["dpi"]))
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 2. Bar charts
# ---------------------------------------------------------------------------


def bar_charts(settings: Settings, trials: pd.DataFrame) -> list[Path]:
    """bars_<state>.png: per group mean +/- SEM of seconds in that state, subjects as dots."""
    summary_path = labels_dir(settings) / SUMMARY_FILE
    if not summary_path.is_file():
        raise ConfigError(f"{summary_path} not found: run `python -m fishbehavior label` first")
    summary = pd.read_csv(summary_path, dtype={"subject_id": str})
    summary["group"] = summary["subject_id"].map(group_of(trials))
    order = sorted(summary["group"].dropna().unique())
    colors, dpi = label_colors(settings.params["reference"]), int(settings.params["plots"]["dpi"])
    rng = np.random.default_rng(0)  # fixed jitter: the same data gives the same picture
    paths = []
    for state in STATES:
        values = [summary.loc[summary["group"] == g, f"{state}_s"].to_numpy(float) for g in order]
        means = [v.mean() for v in values]
        sems = [v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in values]
        fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(order) + 1.5), 4.0))
        ax.bar(range(len(order)), means, yerr=sems, capsize=2, color=np.array(colors[state]) / 255,
               edgecolor="0.25", linewidth=0.6, error_kw={"linewidth": 0.8})
        for i, v in enumerate(values):
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=6, color="0.15", alpha=0.6, zorder=3)
        ax.set_xticks(range(len(order)), order, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("seconds", fontsize=8)
        ax.set_title(f"{state.replace('_', ' ')}: mean ± SEM per group", fontsize=10)
        fig.tight_layout()
        path = plots_dir(settings) / f"bars_{state}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 3. Overlay review video
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Overlay:
    """What is drawn on every frame of one subject (whole-video data, original pixels)."""

    track: pd.DataFrame  # indexed by (part, part_frame)
    labels: np.ndarray  # label per bin
    confidence: np.ndarray
    bin_s: float
    duration_s: float
    colors: dict[str, tuple[int, int, int]]


def timeline_strip(overlay: Overlay, width: int, height: int) -> np.ndarray:
    """BGR strip: the whole video's labels left to right (column = its moment in time)."""
    times = (np.arange(width) + 0.5) / width * overlay.duration_s
    index = np.minimum((times / overlay.bin_s).astype(int), len(overlay.labels) - 1)
    row = rgb(overlay.colors, overlay.labels[index][None, :])[0]
    return np.repeat((row[:, ::-1] * 255).astype(np.uint8)[None], height, axis=0)  # RGB -> BGR


def draw_frame(frame: np.ndarray, row: pd.Series, time_s: float, scene: dict[str, Any], overlay: Overlay,
               strip: np.ndarray) -> np.ndarray:
    """One output frame: the annotated picture above the timeline strip with its cursor."""
    height, width = frame.shape[:2]
    unit = max(1, round(height / 240))  # line width / text scale grow with the resolution
    x0, y0, x1, y1 = scene["roi"]
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 0), unit)
    cv2.line(frame, (x0, scene["waterline_y"]), (x1 - 1, scene["waterline_y"]), (255, 120, 0), unit)
    if np.isfinite(row["x"]):  # NaN: fish not found in this frame
        if np.isfinite(row["major_axis"]) and np.isfinite(row["angle_deg"]):
            # cv2.ellipse turns clockwise on screen, our tilt counter-clockwise.
            cv2.ellipse(frame, (round(row["x"]), round(row["y"])),
                        (max(1, round(row["major_axis"] / 2)), max(1, round(row["minor_axis"] / 2))),
                        -float(row["angle_deg"]), 0, 360, (0, 230, 255), unit)
        cv2.circle(frame, (round(row["x"]), round(row["y"])), 2 * unit, (0, 0, 255), -1)
    b = min(int(time_s / overlay.bin_s), len(overlay.labels) - 1)
    label, confidence = str(overlay.labels[b]), overlay.confidence[b]
    color = overlay.colors.get(label, overlay.colors[NO_DATA]) if label != UNTRACKED else (255, 255, 255)
    text = f"{time_s:.1f} s  {label.replace('_', ' ')}" + (f"  {confidence:.2f}" if np.isfinite(confidence) else "")
    band = 18 * unit
    frame[:band] = frame[:band] // 3  # darkened band: white text stays readable on any picture
    cv2.rectangle(frame, (4 * unit, 4 * unit), (14 * unit, 14 * unit), tuple(int(c) for c in color[::-1]), -1)
    cv2.putText(frame, text, (18 * unit, 13 * unit), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * unit, (255, 255, 255), unit,
                cv2.LINE_AA)
    bar = strip.copy()
    cursor = min(width - 1, int(time_s / overlay.duration_s * width))
    cv2.line(bar, (cursor, 0), (cursor, bar.shape[0] - 1), (0, 0, 0), 3 * unit)  # black edge ...
    cv2.line(bar, (cursor, 0), (cursor, bar.shape[0] - 1), (255, 255, 255), unit)  # ... white cursor
    return np.vstack([frame, bar])


def overlay_video(settings: Settings, subject_id: str, video_paths: list[str], start_s: float,
                  end_s: float | None, out: Path) -> int:
    """Write one subject's review video for start_s <= t < end_s; returns the number of frames written."""
    params = settings.params["plots"]
    labeled = pd.read_csv(labeled_bins_path(labels_dir(settings), subject_id), usecols=["label", "confidence"])
    track = pd.read_csv(track_path(tracks_dir(settings), subject_id),
                        usecols=["part", "part_frame", "time_s", "x", "y", "major_axis", "minor_axis", "angle_deg"])
    bin_s = float(settings.params["features"]["bin_s"])
    frame_gap = float(np.median(np.diff(track["time_s"]))) if len(track) > 1 else bin_s
    overlay = Overlay(track.set_index(["part", "part_frame"]), labeled["label"].to_numpy(object),
                      labeled["confidence"].to_numpy(float), bin_s, float(track["time_s"].iloc[-1]) + frame_gap,
                      label_colors(settings.params["reference"]))
    end_s = overlay.duration_s if end_s is None else end_s
    speed = float(params["overlay_speed"])
    step = max(1, round(speed))  # faster than real time: keep every step-th frame ...
    writer, kept, strip = None, 0, None
    try:
        for part, path in enumerate(video_paths, start=1):
            scene_file = settings.paths.output_dir / "scene" / f"{Path(path).stem}.json"
            scene = json.loads(scene_file.read_text(encoding="utf-8"))
            in_part = overlay.track.loc[part]
            if in_part["time_s"].iloc[-1] < start_s:
                continue  # this part ends before the range: skip without reading it
            for index, _, frame in iter_frames(path, gray=False):
                if index not in in_part.index:
                    continue
                row = in_part.loc[index]
                if row["time_s"] >= end_s:
                    break
                if row["time_s"] < start_s:
                    continue
                if kept % step == 0:
                    if writer is None:
                        height, width = frame.shape[:2]
                        strip = timeline_strip(overlay, width, max(8, round(float(params["timeline_fraction"]) * height)))
                        fps = 1.0 / frame_gap * speed / step  # ... and play them so the video runs at `speed`
                        out.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                 (width, height + strip.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError(f"OpenCV cannot write mp4v video to {out}")
                    writer.write(draw_frame(frame, row, float(row["time_s"]), scene, overlay, strip))
                kept += 1
            if in_part["time_s"].iloc[-1] >= end_s:
                break  # the range ends in this part
    finally:
        if writer is not None:
            writer.release()
    return -(-kept // step)  # frames written


# ---------------------------------------------------------------------------
# 4. Whole run (called by the CLI)
# ---------------------------------------------------------------------------


@dataclass
class PlotResult:
    """What `run_plots` wrote, for the CLI summary."""

    ethograms: list[Path]
    bars: list[Path]
    overlays: list[dict[str, Any]]  # {"subject_id", "path", "frames" | "cached" | "error"}
    out: Path


def overlay_name(subject_id: str, start_s: float, end_s: float | None) -> str:
    """overlay/<subject>_<start>-<end>s.mp4 ('end' when the range runs to the end of the video)."""
    return f"{subject_id}_{start_s:g}-{'end' if end_s is None else f'{end_s:g}'}s.mp4"


def run_plots(settings: Settings, trials: pd.DataFrame, selected: pd.DataFrame | None, overlay: bool = False,
              start_s: float = 0.0, end_s: float | None = None, force: bool = False) -> PlotResult:
    """Ethograms (all groups, or those of `selected`), bar charts, and optional overlay videos."""
    out = plots_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(selected["subject_id"]) if selected is not None else None
    result = PlotResult(ethograms(settings, trials, wanted), bar_charts(settings, trials), [], out)
    if overlay:
        if selected is None:
            raise ConfigError("--overlay needs --subjects (one review video per subject)")
        if end_s is not None and end_s <= start_s:
            raise ConfigError(f"--end ({end_s}) must be after --start ({start_s})")
        for row in selected.itertuples():
            path = out / "overlay" / overlay_name(row.subject_id, start_s, end_s)
            record: dict[str, Any] = {"subject_id": row.subject_id, "path": path}
            if row.video_status != STATUS_MATCHED or not labeled_bins_path(labels_dir(settings), row.subject_id).is_file():
                record["error"] = "no labeled video (run the pipeline up to `label` first)"
            elif path.is_file() and not force:
                record["cached"] = True
            else:
                try:
                    record["frames"] = overlay_video(settings, row.subject_id, row.video_paths.split(";"),
                                                     start_s, end_s, path)
                except (OSError, RuntimeError, KeyError, ValueError) as error:
                    record["error"] = str(error)
            result.overlays.append(record)
    return result
