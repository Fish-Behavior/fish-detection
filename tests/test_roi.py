"""Tests for fishbehavior.roi (scene setup) and the `scene` command, on synthetic videos.

The synthetic scene (synthetic.py) has a known waterline, beaker and fish path, and a
faint reflection of the fish below the beaker bottom that must stay out of the ROI.
"""

import json
import math

import cv2
import numpy as np
import pandas as pd
import pytest

from fishbehavior.cli import main
from fishbehavior.config import ConfigError, load_settings
from fishbehavior.roi import (
    FLAG_DURATION,
    FLAG_FPS,
    FLAG_LOW_CONFIDENCE,
    SceneJob,
    detect_scene,
    load_overrides,
    process_video,
    video_check,
)
from synthetic import FISH, make_video

PARAMS = load_settings(environ={}).params["scene"]


def fish_extent(position, axes):
    """Axis-aligned bounds (x0, y0, x1, y1) of the rotated fish ellipse."""
    x, y, angle = position
    a, b = axes
    theta = math.radians(angle)
    half_w = math.hypot(a * math.cos(theta), b * math.sin(theta))
    half_h = math.hypot(a * math.sin(theta), b * math.cos(theta))
    return x - half_w, y - half_h, x + half_w, y + half_h


@pytest.fixture(scope="module", params=[((320, 240), 30.0), ((640, 480), 60.0)], ids=["320x240@30", "640x480@60"])
def scene(request, tmp_path_factory):
    """One synthetic video per size/fps, and the detection result for it."""
    size, fps = request.param
    truth = make_video(tmp_path_factory.mktemp("scene") / "F_0042.avi", fps=fps, size=size, duration_s=8.0)
    return truth, detect_scene(truth["path"], PARAMS)


def test_background_has_no_fish(scene):
    truth, detection = scene
    # The fish is the only very dark thing in the scene; the median must have removed it.
    assert (detection.background < (FISH + 100)).sum() == 0
    assert detection.background.shape == truth["size"][::-1]


def test_waterline_is_found_within_two_pixels(scene):
    truth, detection = scene
    tolerance = 2 * truth["size"][1] / 240  # ±2 px at 240 rows, scaled with the frame height
    assert abs(detection.waterline_y - truth["waterline_y"]) <= tolerance
    assert detection.waterline_confidence >= PARAMS["min_waterline_confidence"]
    assert detection.flags == []
    assert detection.method == "auto"


def test_roi_contains_the_whole_fish_path_and_the_surface(scene):
    truth, detection = scene
    x0, y0, x1, y1 = detection.roi
    for position in truth["positions"]:
        fx0, fy0, fx1, fy1 = fish_extent(position, truth["fish_axes"])
        assert x0 <= fx0 and fx1 <= x1 and y0 <= fy0 and fy1 <= y1
    assert y0 < detection.waterline_y  # a surface breach is inside the ROI


def test_roi_excludes_the_reflection_below_the_beaker(scene):
    truth, detection = scene
    beaker_bottom = truth["beaker"][3]
    assert detection.roi[3] <= beaker_bottom  # the reflection is drawn only below this row


def test_no_movement_falls_back_to_the_whole_frame(tmp_path):
    truth = make_video(tmp_path / "still.avi", duration_s=2.0, fish_path=lambda t: (160, 150, 0))

    detection = detect_scene(truth["path"], PARAMS)

    assert "no_activity" in detection.flags
    assert detection.activity_box is None


def test_flat_image_gives_low_waterline_confidence(tmp_path):
    # No waterline step at all (glass and water the same): nothing stands out.
    truth = make_video(tmp_path / "flat.avi", duration_s=2.0, waterline_y=239, beaker=(0, 0, 320, 240))

    detection = detect_scene(truth["path"], PARAMS)

    assert FLAG_LOW_CONFIDENCE in detection.flags


# --- one video on disk: files, overrides, caching ------------------------------


@pytest.fixture
def one_video(tmp_path):
    truth = make_video(tmp_path / "F_0042.avi", duration_s=4.0)
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    return truth, scene_dir


def job_for(truth, scene_dir, override=None, force=False):
    return SceneJob("0042", truth["path"], scene_dir, PARAMS, override or {}, force)


def test_process_video_writes_json_background_and_qa(one_video):
    truth, scene_dir = one_video

    record = process_video(job_for(truth, scene_dir))

    saved = json.loads((scene_dir / "F_0042.json").read_text())
    assert saved["waterline_y"] == record["waterline_y"]
    assert saved["video"]["fps"] == pytest.approx(30.0)
    assert saved["method"] == "auto" and len(saved["roi"]) == 4
    background = cv2.imread(str(scene_dir / "F_0042_background.png"), cv2.IMREAD_UNCHANGED)
    assert background.shape == (240, 320)
    qa = cv2.imread(str(scene_dir / "F_0042_qa.png"))
    assert qa.shape[1] >= 640 and qa.shape[2] == 3  # enlarged, in color
    # The blue waterline and the green ROI are drawn.
    assert ((qa[..., 0] == 255) & (qa[..., 1] == 0) & (qa[..., 2] == 0)).any()
    assert ((qa[..., 0] == 0) & (qa[..., 1] == 200) & (qa[..., 2] == 0)).any()


def test_overrides_replace_automatic_values(one_video):
    truth, scene_dir = one_video
    (scene_dir / "overrides.yaml").write_text("F_0042.avi:\n  waterline_y: 100\n  roi: [10, 20, 300, 200]\n")
    override = load_overrides(scene_dir)["F_0042.avi"]

    record = process_video(job_for(truth, scene_dir, override))

    assert record["method"] == "override"
    assert record["waterline_y"] == 100
    assert record["roi"] == [10, 20, 300, 200]


def test_waterline_override_alone_still_raises_the_roi_top(one_video):
    truth, scene_dir = one_video

    record = process_video(job_for(truth, scene_dir, {"waterline_y": 60}))

    margin = round(PARAMS["waterline_margin_fraction"] * 240)
    assert record["roi"][1] == 60 - margin


def test_override_outside_the_frame_is_reported_not_raised(one_video):
    truth, scene_dir = one_video

    record = process_video(job_for(truth, scene_dir, {"roi": [0, 0, 999, 100]}))

    assert "outside" in record["error"]


@pytest.mark.parametrize("text", ["- just a list\n", "F_0042.avi:\n  waterline: 3\n",
                                  "F_0042.avi:\n  roi: [1, 2, 3]\n"])
def test_invalid_overrides_file_is_a_config_error(tmp_path, text):
    (tmp_path / "overrides.yaml").write_text(text)
    with pytest.raises(ConfigError):
        load_overrides(tmp_path)


def test_results_are_cached_until_force_or_new_overrides(one_video):
    truth, scene_dir = one_video
    assert process_video(job_for(truth, scene_dir))["cached"] is False
    assert process_video(job_for(truth, scene_dir))["cached"] is True
    assert process_video(job_for(truth, scene_dir, force=True))["cached"] is False
    # An edited override takes effect without --force.
    assert process_video(job_for(truth, scene_dir, {"waterline_y": 90}))["cached"] is False


# --- video_check.csv flags -------------------------------------------------------


def fake_record(subject, file, fps, duration_s, flags=()):
    return {"subject_id": subject, "file": file, "flags": list(flags),
            "video": {"fps": fps, "frame_count": round(fps * duration_s), "duration_s": duration_s,
                      "width": 320, "height": 240}}


def test_video_check_flags_duration_fps_and_low_confidence():
    records = [
        fake_record("0001", "F_0001.avi", 30.0, 1210.0),  # within 5% of 1200 s
        fake_record("0002", "F_0002.avi", 30.0, 1000.0),  # too short
        fake_record("0003", "F_0003a.avi", 30.0, 600.0),  # parts sum to 1200 s...
        fake_record("0003", "F_0003b.avi", 60.0, 600.0),  # ...but the fps differs
        fake_record("0004", "F_0004.avi", 30.0, 1200.0, [FLAG_LOW_CONFIDENCE]),
        {"subject_id": "0005", "file": "F_0005.avi", "error": "cannot open"},
    ]

    check = video_check(records, PARAMS).set_index("file")["flags"]

    assert check["F_0001.avi"] == ""
    assert check["F_0002.avi"] == FLAG_DURATION
    assert check["F_0003a.avi"] == check["F_0003b.avi"] == FLAG_FPS  # subject flags go on every part
    assert check["F_0004.avi"] == FLAG_LOW_CONFIDENCE
    assert check["F_0005.avi"] == "error"


# --- the `scene` command ------------------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project folder with a hand-written trials.csv (as `validate` would write) and 2 videos."""
    for name in ("FISH_VIDEO_DIR", "FISH_DB_PATH", "FISH_REFERENCE_PDF", "FISH_OUTPUT_DIR", "FISH_WORKERS", "FISH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("F_0042.avi", "F_0043.avi"):
        make_video(videos / name, duration_s=3.0)
    catalog = tmp_path / "outputs" / "catalog"
    catalog.mkdir(parents=True)
    pd.DataFrame({
        "subject_id": ["0042", "0043", "0044"],
        "video_status": ["matched", "matched", "missing"],
        "video_paths": [str(videos / "F_0042.avi"), str(videos / "F_0043.avi"), ""],
    }).to_csv(catalog / "trials.csv", index=False)
    (tmp_path / ".env").write_text("FISH_OUTPUT_DIR=outputs\nFISH_WORKERS=2\n")
    return tmp_path


def test_cli_scene_subjects_and_caching(project, capsys):
    scene_dir = project / "outputs" / "scene"

    assert main(["scene", "--subjects", "F_0042"]) == 0
    assert (scene_dir / "F_0042.json").is_file()
    assert not (scene_dir / "F_0043.json").exists()  # not requested
    assert "1 done (0 cached)" in capsys.readouterr().out

    # All subjects, run in 2 worker processes: 0042 comes from the cache, 0044 has no video.
    assert main(["scene"]) == 0
    out = capsys.readouterr().out
    assert "2 done (1 cached)" in out and "0044 (missing)" in out
    check = pd.read_csv(scene_dir / "video_check.csv", dtype={"subject_id": str})
    assert list(check["subject_id"]) == ["0042", "0043"]
    # Every 3-second test video is far from the expected 1200 s.
    assert (check["flags"] == FLAG_DURATION).all()

    assert main(["scene", "--subjects", "42", "--force"]) == 0
    assert "1 done (0 cached)" in capsys.readouterr().out


def test_cli_scene_unknown_subject_and_missing_catalog(project, capsys):
    assert main(["scene", "--subjects", "0099"]) == 2
    assert "0099" in capsys.readouterr().out

    (project / "outputs" / "catalog" / "trials.csv").unlink()
    assert main(["scene"]) == 2
    assert "validate" in capsys.readouterr().out


def test_qa_background_is_nearly_the_empty_scene(scene):
    """Extra guard: the background matches the drawn scene without the fish."""
    truth, detection = scene
    from synthetic import draw_frame

    empty = cv2.cvtColor(
        draw_frame(truth["size"], truth["beaker"], truth["waterline_y"], (-999, -999, 0), truth["fish_axes"], False),
        cv2.COLOR_BGR2GRAY,
    )
    diff = np.abs(detection.background.astype(int) - empty.astype(int))
    assert np.median(diff) <= 2
