"""Tests for fishbehavior.video on synthetic MJPG videos (see synthetic.py)."""

import math

import numpy as np
import pytest

from fishbehavior import video as video_module
from fishbehavior.video import VideoError, iter_frames, probe, sample_frames
from synthetic import make_video


@pytest.mark.parametrize("fps", [30.0, 60.0])
@pytest.mark.parametrize("size", [(320, 240), (640, 480)])
def test_probe_reads_timing_and_size_from_the_file(tmp_path, fps, size):
    truth = make_video(tmp_path / "clip.avi", fps=fps, size=size, duration_s=2.0)

    info = probe(truth["path"])

    assert info.fps == pytest.approx(fps, rel=1e-3)
    assert info.frame_count == truth["frame_count"]
    assert info.duration_s == pytest.approx(2.0, abs=1 / fps)
    assert (info.width, info.height) == size


def test_probe_missing_or_unreadable_file_is_a_clear_error(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        probe(tmp_path / "nope.avi")
    junk = tmp_path / "junk.avi"
    junk.write_bytes(b"this is not a video")
    with pytest.raises(VideoError):
        probe(junk)


class _NoFpsCapture:
    """Stands in for cv2.VideoCapture on a file whose header has no usable frame rate."""

    fps = math.nan

    def __init__(self, *_):
        pass

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == video_module.cv2.CAP_PROP_FPS else 10

    def release(self):
        pass


@pytest.mark.parametrize("fps_value", [math.nan, 0.0])
def test_probe_rejects_invalid_fps(tmp_path, monkeypatch, fps_value):
    path = tmp_path / "clip.avi"
    path.write_bytes(b"")
    monkeypatch.setattr(_NoFpsCapture, "fps", fps_value)
    monkeypatch.setattr(video_module.cv2, "VideoCapture", _NoFpsCapture)

    with pytest.raises(VideoError, match="frame rate"):
        probe(path)


def test_iter_frames_stride_times_and_gray(tmp_path):
    truth = make_video(tmp_path / "clip.avi", fps=60.0, duration_s=1.0)

    frames = list(iter_frames(truth["path"], stride=7))

    indices = [index for index, _, _ in frames]
    assert indices == list(range(0, truth["frame_count"], 7))
    # time comes from frame_index / fps, not from the container's timestamps
    assert [t for _, t, _ in frames] == pytest.approx([i / 60.0 for i in indices])
    assert frames[0][2].shape == (240, 320)  # grayscale by default


def test_iter_frames_scale_and_color(tmp_path):
    truth = make_video(tmp_path / "clip.avi", size=(640, 480), duration_s=0.5)

    _, _, small = next(iter_frames(truth["path"], scale=0.5))
    _, _, color = next(iter_frames(truth["path"], gray=False))

    assert small.shape == (240, 320)
    assert color.shape == (480, 640, 3)


def test_iter_frames_rejects_bad_arguments(tmp_path):
    truth = make_video(tmp_path / "clip.avi", duration_s=0.2)
    with pytest.raises(ValueError):
        next(iter_frames(truth["path"], stride=0))
    with pytest.raises(ValueError):
        next(iter_frames(truth["path"], scale=0))


def test_sample_frames_spreads_over_the_whole_video(tmp_path):
    # A fish moving left to right at constant speed: its x in a sampled frame tells the time.
    truth = make_video(tmp_path / "clip.avi", duration_s=2.0, reflection=False,
                       fish_path=lambda t: (40 + 120 * t, 150, 0))

    frames = sample_frames(truth["path"], 5)

    assert frames.shape == (5, 240, 320)
    fish_x = [np.nonzero(frame < 110)[1].mean() for frame in frames]  # centre of the dark pixels
    expected = [40 + 120 * t for t in np.linspace(0, (truth["frame_count"] - 1) / 30.0, 5)]
    assert fish_x == pytest.approx(expected, abs=3)


def test_sample_frames_never_returns_more_than_the_video_has(tmp_path):
    truth = make_video(tmp_path / "clip.avi", duration_s=0.2)  # 6 frames

    assert len(sample_frames(truth["path"], 100)) == truth["frame_count"]
