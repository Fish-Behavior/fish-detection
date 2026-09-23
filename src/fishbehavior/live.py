"""Live view: process one video step by step while a local browser page shows every stage.

`python -m fishbehavior live` opens a page (served only on this computer, 127.0.0.1) where
you pick a catalog video or upload one, check its scene (waterline + fish region), and watch:

1. scene      background, waterline and ROI (roi.detect_scene; drag to correct them)
2. tracking   every frame: the fish outline, track point and head (tracking.track_frames)
3. features   speed, turning, depth, posture per 1 s bin (features.frame_features / bin_features)
4. labels     one ethogram state per bin (labeling.label_bins, pooled swim model, calibrated values)
5. saved      the results in <FISH_OUTPUT_DIR>/live/<name>/

About once per second the features and labels are recomputed over the track so far, with the
same functions as the batch steps. The last few seconds are PROVISIONAL: the sustained rules
(freeze_min_s, lorr_min_s) and the bout cleanup still wait for later frames. After the last frame
one final pass over the whole track gives exactly what `all` gives for that video. The page only
shows how the process works; `all` remains the way to process every subject.

Outputs (``<FISH_OUTPUT_DIR>/live/<name>/``, git-ignored): scene.json, background.png,
track.csv.gz, bins.csv (labeled), segments.csv, summary.csv. Uploads go to live/uploads/.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import pandas as pd

from fishbehavior.catalog import STATUS_MATCHED, parse_subject_number, trials_path
from fishbehavior.config import ConfigError, Settings
from fishbehavior.features import bin_features, body_length_px, frame_features, meta_path
from fishbehavior.features import features_dir as get_features_dir
from fishbehavior.labeling import (
    LABELS, MODEL_FILE, STATES, SWIM, UNTRACKED, fit_swim_model, label_bins, labeling_params, labels_dir,
    make_segments, moving, rule_labels, summarize,
)
from fishbehavior.reference import label_colors, reference_dir
from fishbehavior.roi import SceneDetection, detect_scene, load_overrides, override_for, scene_dir
from fishbehavior.tracking import fill_gaps, finalize, join_parts, track_frames, track_table
from fishbehavior.tracking import summarize as track_summary
from fishbehavior.video import _prepare, iter_frames

log = logging.getLogger(__name__)

STAGES = ("scene", "tracking", "features", "labels", "saved")
PAGE_FILE = "live_page.html"
# Plain-words meaning of every label, shown on the page.
STATE_TEXT = {
    "controlled_swim": "normal swimming: steady speed, smooth path, or slow cruising",
    "erratic": "darting and zig-zagging: bursts of speed, sharp and frequent turns",
    "freeze_drift": "motionless, or only drifting",
    "lorr": "listing / loss of righting: hangs steeply and barely moves (not detected yet from this side view)",
    "surface_breach": "head pushed up at the water surface, body angled nose-up",
    "untracked": "not a behavior: the fish was not seen well enough to judge",
}
# Bin features sent to the page for the traces and the "why this label" panel.
TRACE_FEATURES = ("tracked_fraction", "speed_mean_bl_s", "speed_median_bl_s", "nose_up_surface_fraction",
                  "tilt_fraction", "depth_bl_min", "speed_cv", "turn_rate_var_deg2_s2", "distance_bl")
OUTPUT_FILES = ("scene.json", "background.png", "track.csv.gz", "bins.csv", "segments.csv", "summary.csv")


def live_dir(settings: Settings) -> Path:
    """The live view's output folder."""
    return settings.paths.output_dir / "live"


def safe_name(name: str) -> str:
    """A file or folder name that cannot leave its folder (no path parts, only plain characters)."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(name)).name).strip("._")
    return stem or "video"


def jpeg_data(image: np.ndarray, quality: int = 80) -> str:
    """A BGR/gray image as a base64 JPEG (for JSON)."""
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer.tobytes()).decode("ascii") if ok else ""


def plain(value: Any) -> Any:
    """numpy / pandas values -> JSON-safe Python values (NaN -> None)."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), 4)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# 1. What to process
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """One video to process: a catalog subject (all its parts) or an uploaded file."""

    name: str  # output folder name
    video_paths: list[Path]
    subject_id: str | None = None  # catalog subject, when known (for workbook details + reference row)
    details: dict[str, Any] = field(default_factory=dict)  # the subject's workbook row


def load_catalog(settings: Settings) -> pd.DataFrame | None:
    """trials.csv, or None when `validate` has not run."""
    path = trials_path(settings)
    return pd.read_csv(path, dtype={"subject_id": str, "video_paths": str}) if path.is_file() else None


def list_sources(settings: Settings) -> dict[str, Any]:
    """Catalog subjects with a matched video, and the uploaded files."""
    trials = load_catalog(settings)
    catalog = []
    if trials is not None:
        for row in trials[trials["video_status"] == STATUS_MATCHED].itertuples():
            files = [Path(p).name for p in str(row.video_paths).split(";") if p]
            catalog.append({"subject_id": row.subject_id, "files": files,
                            "group": f"{getattr(row, 'compound', '')} {getattr(row, 'concentration_raw', '')}".strip()})
    uploads = live_dir(settings) / "uploads"
    files = sorted(p.name for p in uploads.iterdir() if p.is_file()) if uploads.is_dir() else []
    return {"catalog": catalog, "uploads": files}


def workbook_details(trials: pd.DataFrame | None, subject_id: str | None) -> dict[str, Any]:
    """The subject's workbook row (without local paths), or {}."""
    if trials is None or subject_id is None or subject_id not in set(trials["subject_id"]):
        return {}
    row = trials[trials["subject_id"] == subject_id].iloc[0].drop(labels=["video_paths", "excel_rows"], errors="ignore")
    return plain(row.to_dict())


def resolve_source(settings: Settings, subject_id: str | None = None, upload: str | None = None) -> Source:
    """A catalog subject by id, or an uploaded file (matched to a subject by the number in its name)."""
    trials = load_catalog(settings)
    if subject_id:
        if trials is None:
            raise ConfigError("no catalog yet: run `python -m fishbehavior validate`, or upload a video")
        subject_id = f"{parse_subject_number(subject_id):04d}"
        rows = trials[(trials["subject_id"] == subject_id) & (trials["video_status"] == STATUS_MATCHED)]
        if rows.empty:
            raise ConfigError(f"subject {subject_id} has no matched video in the catalog")
        paths = [Path(p) for p in rows["video_paths"].iloc[0].split(";") if p]
        return Source(subject_id, paths, subject_id, workbook_details(trials, subject_id))
    path = live_dir(settings) / "uploads" / safe_name(upload or "")
    if not upload or not path.is_file():
        raise ConfigError(f"no uploaded video named {upload!r}")
    try:  # "F_0042.mp4" -> subject 0042, when it is in the catalog
        guess = f"{parse_subject_number(path.stem):04d}"
    except ConfigError:
        guess = None
    details = workbook_details(trials, guess)
    return Source(safe_name(path.stem), [path], guess if details else None, details)


def reference_row(settings: Settings, subject_id: str | None) -> dict[str, Any] | None:
    """The subject's digitized reference timeline (labels per second) and group, if it has one."""
    path = reference_dir(settings) / "timelines.csv"
    if subject_id is None or not path.is_file():
        return None
    table = pd.read_csv(path, dtype={"subject_id": str})
    rows = table[table["subject_id"] == subject_id].sort_values("second")
    return None if rows.empty else {"group": str(rows["group"].iloc[0]), "labels": rows["label"].tolist()}


# ---------------------------------------------------------------------------
# 2. Shared state between the processing thread and the page
# ---------------------------------------------------------------------------


class Session:
    """Everything the page shows, versioned so the event stream sends only changes."""

    def __init__(self) -> None:
        self.changed = threading.Condition()
        self.version = 0
        self.frame_version = 0
        self.state: dict[str, Any] = {"stage": "idle", "log": []}
        self.frame: dict[str, Any] | None = None
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        # Kept for review mode and restarts (not sent to the page).
        self.source: Source | None = None
        self.detections: list[SceneDetection] = []
        self.track: pd.DataFrame | None = None
        self.labeled: pd.DataFrame | None = None
        self.known_bl: float | None = None  # BL (px) from an earlier `features` run of this subject

    def reset(self, **state: Any) -> None:
        """Start over for a new video (the page clears everything it shows)."""
        with self.changed:
            self.state, self.frame = {"log": [], **plain(state)}, None
            self.version += 1
            self.frame_version += 1
            self.changed.notify_all()

    def update(self, **changes: Any) -> None:
        with self.changed:
            self.state.update(plain(changes))
            self.version += 1
            self.changed.notify_all()

    def say(self, message: str) -> None:
        """A timestamped line in the page's log (the last 200 are kept)."""
        with self.changed:
            self.state["log"] = [*self.state.get("log", []), f"{time.strftime('%H:%M:%S')}  {message}"][-200:]
            self.version += 1
            self.changed.notify_all()
        log.info(message)

    def set_frame(self, frame: dict[str, Any]) -> None:
        with self.changed:
            self.frame = frame
            self.frame_version += 1
            self.changed.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.changed:
            return json.loads(json.dumps(self.state))

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


# ---------------------------------------------------------------------------
# 3. Scene
# ---------------------------------------------------------------------------


def scene_info(detection: SceneDetection) -> dict[str, Any]:
    """What the page needs to show and edit one part's scene."""
    info = detection.info
    return {"file": info.path.name, "fps": info.fps, "frames": info.frame_count, "duration_s": info.duration_s,
            "width": info.width, "height": info.height, "waterline_y": detection.waterline_y,
            "roi": list(detection.roi), "confidence": detection.waterline_confidence, "method": detection.method,
            "flags": detection.flags, "activity_box": list(detection.activity_box) if detection.activity_box else None}


def open_source(settings: Settings, session: Session, source: Source) -> None:
    """Scene setup for every part (existing scene-review corrections are used); the page then
    shows part 1 to accept or correct."""
    session.cancel.set()
    if session.thread is not None:
        session.thread.join()
    session.cancel.clear()
    session.reset(stage="scene", message=f"Finding the scene of {source.name}...")
    session.source, session.track, session.labeled = source, None, None
    session.known_bl = known_body_length(settings, source)
    session.say(f"opened {source.name}: {len(source.video_paths)} part file(s) "
                f"({', '.join(p.name for p in source.video_paths)})")
    if session.known_bl:
        session.say(f"body length {session.known_bl:.1f} px from the earlier `features` run (until the final pass)")
    overrides = load_overrides(scene_dir(settings))
    session.detections = []
    for path in source.video_paths:
        start = time.perf_counter()
        detection = detect_scene(path, settings.params["scene"], override_for(overrides, path))
        session.detections.append(detection)
        session.say(f"scene {path.name}: waterline y={detection.waterline_y} (confidence "
                    f"{detection.waterline_confidence:.1f}, {detection.method}), ROI {list(detection.roi)} "
                    f"in {time.perf_counter() - start:.1f} s")
    first = session.detections[0]
    session.update(
        stage="scene", source={"name": source.name, "subject_id": source.subject_id, "details": source.details},
        parts=[scene_info(d) for d in session.detections], reference=reference_row(settings, source.subject_id),
        background=jpeg_data(first.background, 90),
        review_frames=[jpeg_data(f, 80) for f in first.review_frames[:4]],
        message="Check the waterline (blue) and fish region (green) of part 1, then press Start.",
    )


# ---------------------------------------------------------------------------
# 4. Processing: track frame by frame, relabel about once per second
# ---------------------------------------------------------------------------


def swim_model(settings: Settings, bins: pd.DataFrame, params: dict[str, Any], bin_s: float) -> tuple[dict | None, str]:
    """The pooled model saved by `label`; otherwise one fitted on this video alone (with a warning)."""
    path = labels_dir(settings) / MODEL_FILE
    if path.is_file():
        model = json.loads(path.read_text(encoding="utf-8"))
        return model, f"pooled ({model['method']}, {model['n_bins']} swim bins of {model['n_subjects']} subjects)"
    swim = (rule_labels(bins, params, bin_s)[0] == SWIM) & moving(bins, params)
    if swim.sum() < 10:
        return None, "none yet (fewer than 10 swimming seconds)"
    return fit_swim_model([(bins, swim)], params), "THIS VIDEO ONLY (no pooled model: run `label` on all subjects)"


def known_body_length(settings: Settings, source: Source) -> float | None:
    """BL (px) of this subject from an earlier `features` run; None for a new video.

    BL is the median fitted length over the whole video. A few seconds in, that median can be
    far off (a fish facing the camera looks short), and every speed and depth, so every label,
    moves with it. While the video plays the page uses the known BL when there is one; the
    final pass always measures it on the whole track, exactly like `features`.
    """
    if source.subject_id is None:
        return None
    try:
        meta = json.loads(meta_path(get_features_dir(settings), source.subject_id).read_text(encoding="utf-8"))
        value = float(meta["endpoint"]["body_length_px"])
    except (OSError, ValueError, KeyError, TypeError):
        return None  # not run yet, or an unreadable file: measure it here
    return value if value > 0 else None


def provisional_seconds(params: dict[str, Any]) -> float:
    """How far back from the newest frame a label can still change (sustained rules, bout cleanup)."""
    bout = max(float(v) for v in params["min_bout_s"].values())
    return max(float(params["freeze_min_s"]), float(params["lorr_min_s"]), bout) + 1.0


def label_summary(segments: pd.DataFrame) -> dict[str, Any]:
    """Seconds, % and bouts per label from the segments (bouts across part boundaries joined)."""
    if segments.empty:
        return {}
    runs = segments.assign(new=segments["label"].ne(segments["label"].shift()).cumsum())
    bouts = runs.groupby("new")["label"].first().value_counts()
    row = summarize(segments).iloc[0]
    return {name: {"s": row[f"{name}_s"], "pct": row[f"{name}_pct"], "bouts": int(bouts.get(name, 0))}
            for name in LABELS}


def compute(settings: Settings, session: Session, source: Source, rows: list[list[dict[str, Any]]],
            infos: list[dict[str, Any]], final: bool) -> None:
    """Features and labels over the track so far (the batch functions), then update the page."""
    fparams, bin_s = settings.params["features"], float(settings.params["features"]["bin_s"])
    params, calibrated = labeling_params(settings)
    parts = [track_table(r) for r in rows if r]
    joined = fill_gaps(join_parts(parts, infos[:len(parts)], source.name), float(settings.params["tracking"]["max_gap_s"]))
    stats = track_summary(joined)  # the same numbers as tracks/summary.csv
    tracking = {**stats, "body_length_px": stats["median_body_length_px"]}
    track = finalize(joined)  # what the batch steps store and read back
    known = None if final else session.known_bl
    bl_px = known or body_length_px(track)
    session.track = track
    if not bl_px > 0:
        session.update(tracking=tracking)
        return
    frames = frame_features(track, bl_px, float(session.detections[0].waterline_y), fparams)
    bins = bin_features(frames, fparams)
    model, model_text = swim_model(settings, bins, params, bin_s)
    stage = "saved" if final else "labels"
    labeling = {"calibrated": str(calibrated) if calibrated else None, "swim_model": model_text,
                "params": params,
                # Everything while BL is still being measured (a new video), else only the last seconds.
                "provisional_from_s": None if final else 0.0 if known is None
                else max(0.0, float(track["time_s"].iloc[-1]) - provisional_seconds(params))}
    features = {"body_length_px": bl_px, "bins": len(bins),
                "body_length_from": "earlier `features` run (until the final pass)" if known
                else "median fitted length of the whole track" if final
                else "median fitted length of the frames so far (settles as the video plays)", "distance_bl": float(bins["distance_bl"].sum()),
                "mean_speed_bl_s": float(bins["speed_mean_bl_s"].mean()), "bin_s": bin_s}
    if model is None:
        session.update(stage="features", tracking=tracking, features=features, labeling=labeling,
                       traces={f: bins[f].tolist() for f in TRACE_FEATURES})
        return
    labeled = label_bins(bins, params, model, bin_s)
    segments = make_segments(labeled, track[["part", "part_frame", "time_s"]], bin_s)
    session.labeled = labeled
    session.update(
        stage=stage, tracking=tracking, features=features, labeling=labeling,
        labels=[LABELS.index(x) for x in labeled["label"]], confidence=labeled["confidence"].tolist(),
        traces={f: labeled[f].tolist() for f in TRACE_FEATURES}, summary=label_summary(segments),
        segments=segments.tail(400).drop(columns=["subject_id"]).to_dict("records"), n_segments=len(segments),
    )
    if final:
        write_outputs(settings, session, source, track, labeled, segments)


def write_outputs(settings: Settings, session: Session, source: Source, track: pd.DataFrame,
                  labeled: pd.DataFrame, segments: pd.DataFrame) -> None:
    """live/<name>/: the same formats as the batch steps."""
    out = live_dir(settings) / source.name
    out.mkdir(parents=True, exist_ok=True)
    first = session.detections[0]
    (out / "scene.json").write_text(json.dumps(plain({"parts": [scene_info(d) for d in session.detections],
                                                      "source": source.name}), indent=2), encoding="utf-8")
    cv2.imwrite(str(out / "background.png"), first.background)
    track.to_csv(out / "track.csv.gz", index=False)
    labeled.to_csv(out / "bins.csv", index=False, float_format="%.5g")
    segments.to_csv(out / "segments.csv", index=False)
    summarize(segments).to_csv(out / "summary.csv", index=False)
    session.update(outputs=list(OUTPUT_FILES), output_dir=str(out))
    session.say(f"saved {', '.join(OUTPUT_FILES)} to {out}")


def process(settings: Settings, session: Session, pace: float) -> None:
    """Track every part frame by frame (preview ~preview_fps per second), relabel every update_s."""
    source, lp = session.source, settings.params["live"]
    tparams = settings.params["tracking"]
    stride, scale = int(tparams["frame_stride"]), float(tparams["scale"])
    infos = [{"fps": d.info.fps, "frame_count": d.info.frame_count, "duration_s": d.info.duration_s, "stride": stride}
             for d in session.detections]
    offsets = np.cumsum([0.0] + [i["duration_s"] for i in infos[:-1]])
    total_frames = sum(-(-i["frame_count"] // stride) for i in infos)
    total_s = float(sum(i["duration_s"] for i in infos))
    rows: list[list[dict[str, Any]]] = [[] for _ in infos]
    preview_gap, update_gap = 1.0 / float(lp["preview_fps"]), float(lp["update_s"])
    start = last_preview = last_update = time.perf_counter()
    done = 0
    session.update(stage="tracking", message="Tracking the fish frame by frame.")
    session.say(f"tracking {total_frames} frames ({total_s:.0f} s of video), pace "
                f"{'as fast as possible' if pace <= 0 else f'{pace:g}x real time'}")
    trail: list[tuple[float, float]] = []
    for part, (detection, offset) in enumerate(zip(session.detections, offsets)):
        latest: dict[str, np.ndarray] = {}

        def frames(path: Path = detection.info.path):
            for index, time_s, raw in iter_frames(path, stride=stride, gray=False):
                latest["frame"] = raw  # the color frame, for the preview
                yield index, time_s, _prepare(raw, scale, True)  # exactly what iter_frames(gray=True) gives

        for row in track_frames(frames(), detection.background, list(detection.roi), tparams):
            rows[part].append(row)
            done += 1
            video_t = offset + row["part_time_s"]
            if pace > 0:  # slow down to the chosen multiple of real time
                wait = video_t / pace - (time.perf_counter() - start)
                if wait > 0:
                    time.sleep(wait)
            now = time.perf_counter()
            if "x" in row and np.isfinite(row.get("x", np.nan)):
                trail = [*trail, (row["x"], row["y"])][-int(2 * detection.info.fps / stride):]
            if now - last_preview >= preview_gap:
                last_preview = now
                session.set_frame({"jpeg": jpeg_data(latest["frame"]), "t": video_t, "part": part + 1,
                                   "frame": row["part_frame"], "row": plain(row), "trail": plain(trail),
                                   "roi": list(detection.roi), "waterline_y": detection.waterline_y})
            if now - last_update >= update_gap:
                last_update = now
                elapsed = now - start
                session.update(progress={"frames_done": done, "frames_total": total_frames, "video_s": video_t,
                                         "total_s": total_s, "elapsed_s": elapsed, "speed_x": video_t / max(elapsed, 1e-6),
                                         "eta_s": (total_s - video_t) / max(video_t / max(elapsed, 1e-6), 1e-6)})
                compute(settings, session, source, rows, infos, final=False)
            if session.cancel.is_set():
                session.update(stage="stopped", message="Stopped.")
                session.say("stopped by the user")
                return
        session.say(f"part {part + 1} done: {len(rows[part])} frames")
    elapsed = time.perf_counter() - start
    session.update(progress={"frames_done": done, "frames_total": total_frames, "video_s": total_s, "total_s": total_s,
                             "elapsed_s": elapsed, "speed_x": total_s / max(elapsed, 1e-6), "eta_s": 0.0},
                   message="Final pass over the whole track (same as the batch steps)...")
    compute(settings, session, source, rows, infos, final=True)
    session.update(stage="saved", message=f"Done in {elapsed:.1f} s ({total_s / max(elapsed, 1e-6):.0f}x real time). "
                                          f"Click the ethogram to look at any second.")


def start_processing(settings: Settings, session: Session, waterline_y: int | None, roi: list[int] | None,
                     pace: float) -> None:
    """Apply the page's scene corrections to part 1 (if any), then process in a background thread."""
    if session.source is None or not session.detections:
        raise ConfigError("open a video first")
    if session.busy():
        raise ConfigError("already processing; press Stop first")
    first = session.detections[0]
    if waterline_y is not None or roi is not None:
        override = {k: v for k, v in (("waterline_y", waterline_y), ("roi", roi)) if v is not None}
        if override != {"waterline_y": first.waterline_y, "roi": list(first.roi)}:
            session.detections[0] = detect_scene(first.info.path, settings.params["scene"], override)
            session.say(f"part 1 scene corrected on the page: {override}")
            session.update(parts=[scene_info(d) for d in session.detections])
    session.cancel.clear()

    def run() -> None:
        try:
            process(settings, session, pace)
        except Exception as error:  # noqa: BLE001 - shown on the page instead of killing the server
            log.exception("live processing failed")
            session.update(stage="error", message=f"Failed: {error}")
            session.say(f"ERROR: {error}")

    session.thread = threading.Thread(target=run, daemon=True)
    session.thread.start()


def review_frame(session: Session, t: float) -> dict[str, Any]:
    """The frame at t seconds (joined timeline) with its track row and bin, for review mode."""
    track = session.track
    if track is None or track.empty:
        raise ConfigError("nothing processed yet")
    i = int(np.clip(np.searchsorted(track["time_s"].to_numpy(float), t), 0, len(track) - 1))
    row = track.iloc[i]
    part = int(row["part"])
    detection = session.detections[part - 1]
    capture = cv2.VideoCapture(str(detection.info.path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["part_frame"]))  # seeking is fine for one frame
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise ConfigError(f"cannot read frame {int(row['part_frame'])} of {detection.info.path.name}")
    return {"jpeg": jpeg_data(image), "t": float(row["time_s"]), "part": part, "frame": int(row["part_frame"]),
            "row": plain(row.to_dict()), "trail": [], "roi": list(detection.roi), "waterline_y": detection.waterline_y}


# ---------------------------------------------------------------------------
# 5. The page and its server
# ---------------------------------------------------------------------------


def page_config(settings: Settings) -> dict[str, Any]:
    """Constants the page needs: labels, colors, plain-words definitions, stages."""
    colors = label_colors(settings.params["reference"])
    colors[UNTRACKED] = colors["no_data"]
    return {"labels": list(LABELS), "states": list(STATES), "colors": {k: list(v) for k, v in colors.items()},
            "text": STATE_TEXT, "stages": list(STAGES), "axis_seconds": int(settings.params["reference"]["axis_seconds"]),
            "max_upload_mb": float(settings.params["live"]["max_upload_mb"])}


def render_page(settings: Settings) -> str:
    """The self-contained page with its configuration filled in."""
    template = resources.files("fishbehavior").joinpath(PAGE_FILE).read_text(encoding="utf-8")
    return template.replace("/*CONFIG*/null", json.dumps(page_config(settings)))


def make_server(settings: Settings, port: int) -> ThreadingHTTPServer:
    """An HTTP server on 127.0.0.1 for the live page.

    GET  /                 the page
    GET  /api/sources      catalog subjects and uploads
    GET  /api/state        the current state (JSON)
    GET  /api/events       server-sent events: `state` and `frame` whenever they change
    GET  /api/frame?t=S    review mode: the frame at S seconds with its track row
    GET  /files/<name>     a result file of the current video (only OUTPUT_FILES)
    POST /api/upload?name= the raw video bytes (size limited)
    POST /api/open         {"subject_id": "42"} or {"upload": "file.mp4"}
    POST /api/start        {"waterline_y", "roi", "pace"}
    POST /api/stop
    """
    session, page = Session(), render_page(settings)
    limit = int(float(settings.params["live"]["max_upload_mb"]) * 1024 * 1024)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._reply(status, json.dumps(plain(data)).encode("utf-8"), "application/json")

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")

        def _events(self) -> None:
            """Stream `state` / `frame` events until the page closes the connection."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sent, sent_frame = -1, -1
            try:
                while True:
                    with session.changed:
                        session.changed.wait_for(lambda: session.version != sent or session.frame_version != sent_frame,
                                                 timeout=15)
                        version, frame_version = session.version, session.frame_version
                        state = json.dumps(session.state) if version != sent else None
                        frame = json.dumps(session.frame) if frame_version != sent_frame and session.frame else None
                    chunks = [f"event: state\ndata: {state}\n\n" if state else "",
                              f"event: frame\ndata: {frame}\n\n" if frame else ""]
                    self.wfile.write(("".join(chunks) or ": keep-alive\n\n").encode("utf-8"))
                    self.wfile.flush()
                    sent, sent_frame = version, frame_version
            except (BrokenPipeError, ConnectionResetError):
                return  # the page was closed

        def do_GET(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
            url = urlparse(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    self._reply(200, page.encode("utf-8"), "text/html; charset=utf-8")
                elif url.path == "/api/sources":
                    self._json(200, list_sources(settings))
                elif url.path == "/api/state":
                    self._json(200, session.snapshot())
                elif url.path == "/api/events":
                    self._events()
                elif url.path == "/api/frame":
                    self._json(200, review_frame(session, float(parse_qs(url.query).get("t", ["0"])[0])))
                elif url.path.startswith("/files/") and session.source is not None:
                    name = unquote(url.path[len("/files/"):])
                    path = live_dir(settings) / session.source.name / name
                    if name not in OUTPUT_FILES or not path.is_file():
                        self._reply(404, b"not found", "text/plain")
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Disposition", f'attachment; filename="{session.source.name}_{name}"')
                        self.send_header("Content-Length", str(path.stat().st_size))
                        self.end_headers()
                        self.wfile.write(path.read_bytes())
                else:
                    self._reply(404, b"not found", "text/plain")
            except (ConfigError, ValueError) as error:
                self._json(400, {"ok": False, "error": str(error)})

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            try:
                if url.path == "/api/upload":
                    self._json(200, {"ok": True, "upload": self._save_upload(url)})
                elif url.path == "/api/open":
                    body = self._body()
                    open_source(settings, session, resolve_source(settings, body.get("subject_id"), body.get("upload")))
                    self._json(200, {"ok": True})
                elif url.path == "/api/start":
                    body = self._body()
                    roi = [int(v) for v in body["roi"]] if body.get("roi") else None
                    waterline = int(body["waterline_y"]) if body.get("waterline_y") is not None else None
                    start_processing(settings, session, waterline, roi, float(body.get("pace") or 0))
                    self._json(200, {"ok": True})
                elif url.path == "/api/stop":
                    session.cancel.set()
                    self._json(200, {"ok": True})
                else:
                    self._json(404, {"ok": False, "error": "unknown address"})
            except (ConfigError, ValueError, KeyError) as error:  # ValueError includes bad JSON / bad scene values
                self._json(400, {"ok": False, "error": str(error)})

        def _save_upload(self, url: Any) -> str:
            """Write the request body to live/uploads/<safe name> in chunks; refuse too large or non-video files."""
            name = safe_name(parse_qs(url.query).get("name", [""])[0])
            extensions = [e.lower() for e in settings.params["catalog"]["video_extensions"]]
            length = int(self.headers.get("Content-Length", 0))
            problem = (f"{name}: not a video file ({', '.join(extensions)})" if Path(name).suffix.lower() not in extensions
                       else f"upload must be 1 byte to {limit // 2**20} MB (live.max_upload_mb)" if not 0 < length <= limit
                       else None)
            if problem:
                if 0 < length <= limit:  # read the body first, so the browser gets the message, not a reset
                    while length > 0:
                        length -= len(self.rfile.read(min(length, 1 << 20)) or b"x" * length)
                else:
                    self.close_connection = True
                raise ConfigError(problem)
            folder = live_dir(settings) / "uploads"
            folder.mkdir(parents=True, exist_ok=True)
            left = length
            with (folder / name).open("wb") as handle:
                while left:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        raise ConfigError("upload interrupted")
                    handle.write(chunk)
                    left -= len(chunk)
            return name

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            log.debug("live server: " + format, *args)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)  # this computer only
    server.daemon_threads = True  # open event streams must not keep the program alive
    server.session = session  # type: ignore[attr-defined] - for tests
    return server


def serve(server: ThreadingHTTPServer, open_browser: bool = True) -> None:
    """Serve until Ctrl+C."""
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Live view at {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.session.cancel.set()  # type: ignore[attr-defined]
        server.server_close()
