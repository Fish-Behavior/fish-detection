"""Per-frame tracks from detections made on every Nth frame (Phase 15 model tracker)."""

from __future__ import annotations

import pytest

from prepds.strided_tracks import Detection, expand_strided_detections

FPS = 30.0


def _det(x, y, angle=10.0):
    return Detection(x=x, y=y, orientation_deg=angle)


def test_every_frame_gets_a_track_with_time_from_fps() -> None:
    tracks = expand_strided_detections({0: _det(10, 20), 3: _det(40, 50)}, n_frames=7, stride=3, fps=FPS)
    assert [t.frame_idx for t in tracks] == list(range(7))
    assert tracks[3].t_sec == pytest.approx(3 / FPS)


def test_sampled_frames_keep_their_detection_exactly() -> None:
    tracks = expand_strided_detections({0: _det(10, 20, 30.0), 3: _det(40, 50, 60.0)}, n_frames=4, stride=3, fps=FPS)
    assert (tracks[0].x, tracks[0].y, tracks[0].orientation_deg) == (10, 20, 30.0)
    assert (tracks[3].x, tracks[3].y, tracks[3].y_from_frame_top) == (40, 50, 50)
    assert tracks[0].detected and tracks[3].detected


def test_frames_between_two_detections_are_linearly_interpolated() -> None:
    tracks = expand_strided_detections({0: _det(0, 0), 3: _det(30, 60)}, n_frames=4, stride=3, fps=FPS)
    assert (tracks[1].x, tracks[1].y) == pytest.approx((10, 20)) and tracks[1].detected
    assert (tracks[2].x, tracks[2].y) == pytest.approx((20, 40)) and tracks[2].detected


def test_frames_next_to_a_missed_detection_are_undetected() -> None:
    tracks = expand_strided_detections({0: _det(0, 0), 3: None, 6: _det(60, 60)}, n_frames=7, stride=3, fps=FPS)
    for i in (1, 2, 3, 4, 5):
        assert not tracks[i].detected and tracks[i].x == 0.0 and tracks[i].y_from_frame_top is None and tracks[i].orientation_deg is None
    assert tracks[0].detected and tracks[6].detected


def test_trailing_frames_after_the_last_sample_are_undetected() -> None:
    tracks = expand_strided_detections({0: _det(0, 0), 3: _det(30, 30)}, n_frames=6, stride=3, fps=FPS)
    assert tracks[3].detected and not tracks[4].detected and not tracks[5].detected


def test_stride_one_is_a_plain_per_frame_track() -> None:
    tracks = expand_strided_detections({i: _det(i, i) for i in range(4)}, n_frames=4, stride=1, fps=FPS)
    assert [t.x for t in tracks] == [0, 1, 2, 3]


def test_orientation_of_interpolated_frames_follows_the_nearest_sample() -> None:
    tracks = expand_strided_detections({0: _det(0, 0, 10.0), 4: _det(40, 0, 100.0)}, n_frames=5, stride=4, fps=FPS)
    assert tracks[1].orientation_deg == 10.0 and tracks[3].orientation_deg == 100.0


def test_missing_orientation_propagates_as_none() -> None:
    tracks = expand_strided_detections({0: Detection(0, 0, None), 2: Detection(2, 0, None)}, n_frames=3, stride=2, fps=FPS)
    assert all(t.detected and t.orientation_deg is None for t in tracks)


@pytest.mark.parametrize("kwargs", [{"n_frames": 0}, {"stride": 0}, {"fps": 0.0}])
def test_invalid_arguments(kwargs) -> None:
    args = {"n_frames": 4, "stride": 2, "fps": FPS, **kwargs}
    with pytest.raises(ValueError):
        expand_strided_detections({0: _det(0, 0)}, **args)
