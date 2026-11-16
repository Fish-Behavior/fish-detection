"""Live capture: a camera (or a file played like one) through the same per-frame tracker as a video file.

A live session is just another frame source. `FishTracker` tracks each frame as it arrives, `OnlineLabeler` shows the
current behavior state, and every frame is recorded, so when the session ends the recording and its tracks go
through the same labeling and export as an uploaded video.

`OnlineLabeler` is display-only and uses past frames only. It applies the batch thresholds the way they can be applied
live, so it differs from `labeling.classify_video` in three known ways: a sustained state (Freezing/Drift,
Listing/LORR) shows once its minimum bout has elapsed rather than from the bout's first frame; speed uses only the
earlier partner frame (batch falls back to the later one); and Dead is never shown, because it needs the end of the
recording. The final labels always come from `classify_video` on the collected tracks.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from prepds.features import DEFAULT_SPEED_LAG_S
from prepds.labeling import (
    DEFAULT_ERRATIC_SPEED_THRESHOLD_PX_PER_S,
    DEFAULT_FREEZE_SPEED_FLOOR_PX_PER_S,
    DEFAULT_LISTING_ORIENTATION_DEVIATION_DEG,
    DEFAULT_MIN_FREEZE_BOUT_S,
    DEFAULT_MIN_LISTING_BOUT_S,
    DEFAULT_SURFACE_BREACH_DEPTH_THRESHOLD_PX,
)
from prepds.models import BehaviorState, Track
from prepds.tracking import FishTracker

MAX_CAMERA_INDEX = 5
# The tracker's area and jump limits are in pixels, set on 320x240 footage (tracking.py). Frames are scaled to this
# height, keeping the aspect ratio, before tracking and recording.
DEFAULT_FRAME_HEIGHT = 240
FALLBACK_FPS = 30.0
DEFAULT_MAX_DURATION_S = 2 * 3600.0
# Browser-playable first. mp4v always records but browsers do not play it, so it is the last resort.
RECORDING_CODECS = (("avc1", ".mp4"), ("VP80", ".webm"), ("mp4v", ".mp4"))


def list_cameras(max_index: int = MAX_CAMERA_INDEX, *, opener: Callable = cv2.VideoCapture) -> list[dict]:
    """Cameras that open and return a frame, by index; `[]` when there are none. OpenCV cannot read device names."""
    found = []
    for index in range(max_index):
        capture = opener(index)
        try:
            ok, frame = capture.read() if capture.isOpened() else (False, None)
            if ok and frame is not None:
                found.append({"index": index, "name": f"Camera {index}", "width": int(frame.shape[1]), "height": int(frame.shape[0])})
        finally:
            capture.release()
    return found


class OnlineLabeler:
    """Current behavior state per frame from past frames only, with the batch rule precedence minus Dead."""

    def __init__(
        self,
        fps: float,
        *,
        speed_lag_s: float = DEFAULT_SPEED_LAG_S,
        freeze_speed_floor_px_per_s: float = DEFAULT_FREEZE_SPEED_FLOOR_PX_PER_S,
        min_freeze_bout_s: float = DEFAULT_MIN_FREEZE_BOUT_S,
        erratic_speed_threshold_px_per_s: float = DEFAULT_ERRATIC_SPEED_THRESHOLD_PX_PER_S,
        surface_breach_depth_threshold_px: float = DEFAULT_SURFACE_BREACH_DEPTH_THRESHOLD_PX,
        listing_orientation_deviation_deg: float | None = DEFAULT_LISTING_ORIENTATION_DEVIATION_DEG,
        min_listing_bout_s: float = DEFAULT_MIN_LISTING_BOUT_S,
        min_dead_bout_s: float | None = None,  # accepted so the batch thresholds dict passes through; Dead is not live
    ) -> None:
        if erratic_speed_threshold_px_per_s <= freeze_speed_floor_px_per_s:
            raise ValueError("erratic_speed_threshold_px_per_s must exceed freeze_speed_floor_px_per_s")
        self.lag = max(1, round(speed_lag_s * fps))  # batch: lag_s over the median frame spacing, i.e. 1/fps
        self._freeze_floor, self._min_freeze = freeze_speed_floor_px_per_s, min_freeze_bout_s
        self._erratic, self._breach = erratic_speed_threshold_px_per_s, surface_breach_depth_threshold_px
        self._listing, self._min_listing = listing_orientation_deviation_deg, min_listing_bout_s
        self._recent: deque[Track] = deque(maxlen=self.lag + 1)
        self._freeze_since: float | None = None
        self._listing_since: float | None = None
        self.speed: float | None = None

    def update(self, track: Track) -> BehaviorState:
        if self._recent and track.frame_idx != self._recent[-1].frame_idx + 1:
            self._recent.clear()  # a frame_idx gap is never bridged, as in batch
        self._recent.append(track)
        partner = self._recent[0]
        valid = (
            track.detected and partner.detected and track.frame_idx - partner.frame_idx == self.lag
            and track.t_sec > partner.t_sec
        )
        self.speed = math.hypot(track.x - partner.x, track.y - partner.y) / (track.t_sec - partner.t_sec) if valid else None

        freezing = valid and self.speed <= self._freeze_floor
        self._freeze_since = (self._freeze_since if self._freeze_since is not None else track.t_sec) if freezing else None
        listing = (
            self._listing is not None and track.detected and track.orientation_deg is not None
            and abs(track.orientation_deg - 90.0) >= self._listing  # fitEllipse: 90 == horizontal, see labeling.py
        )
        self._listing_since = (self._listing_since if self._listing_since is not None else track.t_sec) if listing else None

        if not track.detected:
            return BehaviorState.UNDETERMINED
        if track.y_from_frame_top is not None and track.y_from_frame_top <= self._breach:
            return BehaviorState.SURFACE_BREACH
        if self._listing_since is not None and track.t_sec - self._listing_since >= self._min_listing:
            return BehaviorState.LISTING_LORR
        if self._freeze_since is not None and track.t_sec - self._freeze_since >= self._min_freeze:
            return BehaviorState.FREEZING_DRIFT
        if not valid:
            return BehaviorState.UNDETERMINED
        return BehaviorState.ERRATIC_MOVEMENT if self.speed >= self._erratic else BehaviorState.CONTROLLED_SWIM


def open_recorder(base: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    """The first codec in `RECORDING_CODECS` that opens; the suffix of `base` is replaced by the codec's."""
    for fourcc, suffix in RECORDING_CODECS:
        path = base.with_suffix(suffix)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
        path.unlink(missing_ok=True)
    raise RuntimeError("No video codec could open a recording file")


class LiveSession(threading.Thread):
    """Capture, track, label and record one fish until `stop()`, the end of a file, or `max_duration_s`.

    `source` is a camera index or a video file path. A file is read as fast as possible unless `realtime`, which
    paces it at its own fps so it behaves like a camera (useful without one). `status()` is safe to call from other
    threads; `tracks` and `video_path` are complete once the thread has finished.
    """

    def __init__(
        self,
        source: int | str | Path,
        recording: Path,
        *,
        thresholds: dict | None = None,
        fps: float | None = None,
        frame_height: int | None = DEFAULT_FRAME_HEIGHT,
        realtime: bool = False,
        max_duration_s: float = DEFAULT_MAX_DURATION_S,
        opener: Callable = cv2.VideoCapture,
    ) -> None:
        super().__init__(daemon=True)
        self.source, self.recording, self.thresholds = source, Path(recording), dict(thresholds or {})
        self.fps_override, self.frame_height, self.realtime = fps, frame_height, realtime
        self.max_duration_s, self._opener = max_duration_s, opener
        self.tracks: list[Track] = []
        self.video_path: Path | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, Track] | None = None
        self._status = {"phase": "starting", "fps": None, "frame_count": 0, "t_sec": 0.0, "state": None,
                        "detected": False, "speed_px_per_s": None, "seconds": [], "error": None}

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            return {**self._status, "seconds": list(self._status["seconds"])}

    def latest(self) -> tuple[np.ndarray, Track] | None:
        """The last processed frame (as recorded) and its track, for a preview."""
        with self._lock:
            return self._latest

    def run(self) -> None:
        try:
            self._run()
        except Exception as error:  # noqa: BLE001 - reported through status, the server keeps running
            with self._lock:
                self._status.update(phase="error", error=f"{type(error).__name__}: {error}")

    def _run(self) -> None:
        source = str(self.source) if isinstance(self.source, Path) else self.source
        capture = self._opener(source)
        writer = None
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open {'camera' if isinstance(source, int) else 'video'} {source!r}")
            reported = capture.get(cv2.CAP_PROP_FPS)
            # ponytail: webcams often report 0 or a nominal fps; t_sec = frame_idx / fps assumes it holds. Pass fps=
            # once a rig's real rate is measured.
            fps = self.fps_override or (reported if 1.0 <= reported <= 240.0 else FALLBACK_FPS)
            tracker = FishTracker(fps)
            labeler = OnlineLabeler(fps, **self.thresholds)
            with self._lock:
                self._status.update(phase="running", fps=fps)
            started, second, states = time.monotonic(), 0, Counter()
            frame_idx = 0
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame = self._resize(frame)
                if writer is None:
                    writer, self.video_path = open_recorder(self.recording, fps, (frame.shape[1], frame.shape[0]))
                writer.write(frame)
                track = tracker.step(frame_idx, frame)
                state = labeler.update(track)
                self.tracks.append(track)
                if int(track.t_sec) > second:  # close the finished second with its most frequent state
                    finished = states.most_common(1)[0][0].value
                    second, states = int(track.t_sec), Counter()
                    with self._lock:
                        self._status["seconds"].append(finished)
                states[state] += 1
                with self._lock:
                    self._latest = (frame, track)
                    self._status.update(frame_count=frame_idx + 1, t_sec=track.t_sec, state=state.value,
                                        detected=track.detected, speed_px_per_s=labeler.speed)
                frame_idx += 1
                if track.t_sec >= self.max_duration_s:
                    break
                if self.realtime and not isinstance(source, int):
                    time.sleep(max(0.0, frame_idx / fps - (time.monotonic() - started)))
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        with self._lock:
            self._status["phase"] = "stopped"

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if not self.frame_height or height == self.frame_height:
            return frame
        new_width = max(2, round(width * self.frame_height / height / 2) * 2)  # even width: some codecs need it
        return cv2.resize(frame, (new_width, self.frame_height), interpolation=cv2.INTER_AREA)
