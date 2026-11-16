"""Live capture (prepds.live): a live source must give exactly the batch tracks, and the online labels must agree
with the batch labels except for the documented confirmation delay."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from prepds.features import compute_smoothed_speed
from prepds.labeling import classify_prepared_states, prepare_video
from prepds.live import LiveSession, OnlineLabeler, list_cameras
from prepds.models import BehaviorState, Track
from prepds.tracking import track_video

SYNTH_VIDEO = Path(__file__).parents[3] / "shared/tests/fixtures/synth_tiny.mp4"
FPS = 10.0
THRESHOLDS = dict(freeze_speed_floor_px_per_s=10.0, min_freeze_bout_s=2.0, erratic_speed_threshold_px_per_s=40.0,
                  surface_breach_depth_threshold_px=5.0, speed_lag_s=1.0)


def _tracks() -> list[Track]:
    """0-3 s at 100 px/s (erratic), 3-8 s still (freezing), 8-12 s at 20 px/s (controlled), undetected 10.0-10.5 s."""
    out, x = [], 50.0
    for i in range(int(12 * FPS)):
        t = i / FPS
        x += (100.0 if t < 3 else 0.0 if t < 8 else 20.0) / FPS
        detected = not 10.0 <= t < 10.5
        out.append(Track(frame_idx=i, t_sec=t, x=x if detected else 0.0, y=120.0 if detected else 0.0,
                         orientation_deg=90.0 if detected else None, y_from_frame_top=120.0 if detected else None,
                         detected=detected))
    return out


def _run(session: LiveSession) -> LiveSession:
    session.start()
    session.join(timeout=30)
    assert not session.is_alive()
    return session


def test_live_file_source_gives_the_batch_tracks_and_records_every_frame(tmp_path) -> None:
    session = _run(LiveSession(SYNTH_VIDEO, tmp_path / "recording", thresholds=THRESHOLDS))
    assert session.status()["phase"] == "stopped", session.status()["error"]
    assert session.tracks == track_video(SYNTH_VIDEO)
    capture = cv2.VideoCapture(str(session.video_path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(session.tracks) == 150
    capture.release()
    status = session.status()
    assert status["frame_count"] == 150 and len(status["seconds"]) == 4  # 5 s: seconds 0-3 closed, 4 still open


def test_online_speed_equals_batch_speed_wherever_the_earlier_partner_exists() -> None:
    tracks = _tracks()
    batch_speeds, batch_valid = compute_smoothed_speed(tracks, lag_s=1.0)
    labeler = OnlineLabeler(FPS, **THRESHOLDS)
    checked = 0
    for i, track in enumerate(tracks):
        labeler.update(track)
        if labeler.speed is not None:
            assert batch_valid[i] and labeler.speed == pytest.approx(batch_speeds[i])
            checked += 1
    assert checked > 60


def test_online_labels_match_batch_except_the_freeze_confirmation_delay() -> None:
    tracks = _tracks()
    batch = classify_prepared_states(prepare_video(tracks, speed_lag_s=1.0), **{k: v for k, v in THRESHOLDS.items() if k != "speed_lag_s"})
    labeler = OnlineLabeler(FPS, **THRESHOLDS)
    live = [labeler.update(t) for t in tracks]

    freeze = BehaviorState.FREEZING_DRIFT
    live_freeze = [i for i, s in enumerate(live) if s == freeze]
    batch_freeze = [i for i, s in enumerate(batch) if s == freeze]
    assert live_freeze and set(live_freeze) <= set(batch_freeze)  # live never claims a freeze batch does not
    assert tracks[live_freeze[0]].t_sec - tracks[batch_freeze[0]].t_sec == pytest.approx(2.0)  # min_freeze_bout_s
    assert live_freeze[-1] == batch_freeze[-1]

    for i, (l, b) in enumerate(zip(live, batch)):
        if l not in (freeze, BehaviorState.UNDETERMINED) and b != freeze:
            assert l == b, f"frame {i}: live {l}, batch {b}"
    assert live[int(1.5 * FPS)] == BehaviorState.ERRATIC_MOVEMENT
    assert live[int(9.5 * FPS)] == BehaviorState.CONTROLLED_SWIM
    assert live[int(10.2 * FPS)] == BehaviorState.UNDETERMINED  # undetected
    assert live[int(10.7 * FPS)] == BehaviorState.CONTROLLED_SWIM  # only the endpoints must be detected, as in batch
    assert live[int(11.2 * FPS)] == BehaviorState.UNDETERMINED  # partner 1 s back is undetected: no speed


def test_online_labeler_never_shows_dead_and_rejects_overlapping_speed_bands() -> None:
    labeler = OnlineLabeler(FPS, min_dead_bout_s=1.0, **THRESHOLDS)
    assert BehaviorState.DEAD not in [labeler.update(t) for t in _tracks()]
    with pytest.raises(ValueError):
        OnlineLabeler(FPS, freeze_speed_floor_px_per_s=50.0, erratic_speed_threshold_px_per_s=40.0)


class _FakeCamera:
    def __init__(self, frames: int) -> None:
        self.frames = frames

    def isOpened(self) -> bool:
        return self.frames > 0

    def read(self):
        if self.frames <= 0:
            return False, None
        self.frames -= 1
        return True, np.full((720, 1280, 3), 180, np.uint8)

    def get(self, _prop) -> float:
        return 0.0  # webcams often report no fps

    def release(self) -> None:
        pass


def test_list_cameras_is_empty_without_devices_and_lists_the_ones_that_read() -> None:
    assert list_cameras(opener=lambda index: _FakeCamera(0)) == []
    cameras = list_cameras(opener=lambda index: _FakeCamera(5 if index in (0, 2) else 0))
    assert cameras == [{"index": 0, "name": "Camera 0", "width": 1280, "height": 720},
                       {"index": 2, "name": "Camera 2", "width": 1280, "height": 720}]


def test_camera_frames_are_scaled_to_the_tracker_height_and_unknown_fps_falls_back(tmp_path) -> None:
    session = _run(LiveSession(0, tmp_path / "rec", opener=lambda source: _FakeCamera(40), thresholds=THRESHOLDS))
    status = session.status()
    assert status["phase"] == "stopped" and status["fps"] == 30.0 and len(session.tracks) == 40
    frame, _track = session.latest()
    assert frame.shape == (240, 426, 3)  # 1280x720 -> height 240, even width


def test_unopenable_source_reports_an_error_instead_of_raising(tmp_path) -> None:
    session = _run(LiveSession(3, tmp_path / "rec", opener=lambda source: _FakeCamera(0)))
    status = session.status()
    assert status["phase"] == "error" and "Could not open camera 3" in status["error"]
    assert session.video_path is None


def test_stop_ends_a_realtime_session_early(tmp_path) -> None:
    session = LiveSession(SYNTH_VIDEO, tmp_path / "rec", realtime=True, thresholds=THRESHOLDS)
    session.start()
    session.stop()
    session.join(timeout=10)
    assert session.status()["phase"] == "stopped" and len(session.tracks) < 150


def test_online_breach_listing_and_frame_gaps_follow_the_batch_rules() -> None:
    labeler = OnlineLabeler(FPS, listing_orientation_deviation_deg=30.0, min_listing_bout_s=1.0, **THRESHOLDS)

    def step(i: int, y: float = 120.0, angle: float = 90.0) -> BehaviorState:
        return labeler.update(Track(frame_idx=i, t_sec=i / FPS, x=50.0, y=y, orientation_deg=angle,
                                    y_from_frame_top=y, detected=True))

    assert step(0, y=3.0) == BehaviorState.SURFACE_BREACH  # per frame, no bout needed
    tilted = [step(i, angle=150.0) for i in range(1, 13)]
    assert tilted[-1] == BehaviorState.LISTING_LORR and tilted[0] != BehaviorState.LISTING_LORR  # after 1 s
    assert step(40) == BehaviorState.UNDETERMINED  # frame_idx gap: no partner, the bout clock restarts


def test_a_listing_bout_never_bridges_a_frame_gap() -> None:
    labeler = OnlineLabeler(FPS, listing_orientation_deviation_deg=30.0, min_listing_bout_s=1.0, **THRESHOLDS)
    states = [labeler.update(Track(frame_idx=i, t_sec=i / FPS, x=50.0, y=120.0, orientation_deg=150.0,
                                   y_from_frame_top=120.0, detected=True))
              for i in (*range(0, 6), *range(40, 46))]
    assert BehaviorState.LISTING_LORR not in states  # two 0.5 s runs, each under the 1 s bout, as in batch


def test_codec_fallback_and_duration_cap(tmp_path, monkeypatch) -> None:
    import prepds.live as live
    from prepds.tracking import FishTracker

    monkeypatch.setattr(live, "RECORDING_CODECS", (("ZZZZ", ".mkv"), ("mp4v", ".mp4")))
    session = _run(LiveSession(SYNTH_VIDEO, tmp_path / "rec", max_duration_s=1.0, thresholds=THRESHOLDS))
    assert session.video_path == tmp_path / "rec.mp4" and not (tmp_path / "rec.mkv").exists()
    assert len(session.tracks) == 31  # frames 0..30: stops at t = 1.0 s
    monkeypatch.setattr(live, "RECORDING_CODECS", (("ZZZZ", ".mkv"),))
    with pytest.raises(RuntimeError, match="No video codec"):
        live.open_recorder(tmp_path / "x", 30.0, (304, 240))
    with pytest.raises(ValueError):
        FishTracker(0.0)
