"""Tests for fishbehavior.tracking and the `track` command, on synthetic videos.

The synthetic scene (synthetic.py) gives the true fish position and tilt in every
frame, can hide the fish for a while (gaps), and draws a reflection below the beaker
that must never be taken for the fish.
"""

import json
import math

import cv2
import numpy as np
import pandas as pd
import pytest

from fishbehavior.cli import main
from fishbehavior.config import load_settings
from fishbehavior.roi import detect_scene
from fishbehavior.tracking import (
    TRACK_COLUMNS,
    TrackJob,
    choose_blob,
    fill_gaps,
    join_parts,
    process_subject,
    summarize,
    track_part,
)
from synthetic import default_beaker, head_position, lissajous_path, make_video

SETTINGS = load_settings(environ={}).params
PARAMS, SCENE = SETTINGS["tracking"], SETTINGS["scene"]


def track_video(truth, params=None, roi=None, max_gap_s=None):
    """Scene + tracking of one synthetic video, joined and gap-filled like a one-part subject."""
    params = {**PARAMS, **(params or {})}
    scene = detect_scene(truth["path"], SCENE)
    part = track_part(truth["path"], scene.background, list(roi or scene.roi), params)
    info = {"fps": truth["fps"], "frame_count": truth["frame_count"],
            "duration_s": truth["frame_count"] / truth["fps"], "stride": int(params["frame_stride"])}
    gap = params["max_gap_s"] if max_gap_s is None else max_gap_s
    return fill_gaps(join_parts([part], [info], "0042"), gap)


def position_error(track, truth):
    """Distance (px) between tracked and true centroid, per frame (NaN where not tracked)."""
    positions = np.array(truth["positions"])
    return np.hypot(track["x"].to_numpy() - positions[:, 0], track["y"].to_numpy() - positions[:, 1])


# --- accuracy on a known path ----------------------------------------------------


@pytest.mark.parametrize("size, fps, scale", [((320, 240), 30.0, 1.0), ((640, 480), 60.0, 1.0),
                                              ((640, 480), 60.0, 0.5)],
                         ids=["320x240@30", "640x480@60", "640x480@60-scale0.5"])
def test_centroid_error_below_two_pixels_and_detection_above_95_percent(tmp_path, size, fps, scale):
    truth = make_video(tmp_path / "F_0042.avi", fps=fps, size=size, duration_s=5.0)

    track = track_video(truth, {"scale": scale})

    assert track["detected"].mean() > 0.95
    error = position_error(track, truth)[track["detected"].to_numpy()]
    assert error.max() < 2.0  # in ORIGINAL pixels, also when analysed at half size
    # Outline size and tilt: length close to the drawn ellipse, tilt sign as documented
    # (positive = right end higher on screen = counter-clockwise, the opposite of cv2.ellipse).
    length = 2 * truth["fish_axes"][0]
    assert track["major_axis"].median() == pytest.approx(length, rel=0.08)
    drawn = np.array(truth["positions"])[:, 2]
    tilt_error = (track["angle_deg"].to_numpy() + drawn + 90) % 180 - 90
    assert np.nanmax(np.abs(tilt_error)) < 5
    # The top/bottom rows enclose the centroid.
    assert (track["top_y"] <= track["y"]).all() and (track["y"] <= track["bottom_y"]).all()


def head_error(track, truth):
    """Distance (px) between the tracked head and the drawn eye, per frame."""
    heads = np.array([head_position(p, truth["fish_axes"]) for p in truth["positions"]])
    return np.hypot(track["head_x"].to_numpy() - heads[:, 0], track["head_y"].to_numpy() - heads[:, 1])


@pytest.mark.parametrize("size, scale", [((320, 240), 1.0), ((640, 480), 0.5)], ids=["320x240", "640x480-scale0.5"])
def test_head_is_the_eye_and_pitch_is_nose_up_positive(tmp_path, size, scale):
    truth = make_video(tmp_path / "F_0042.avi", size=size, duration_s=4.0)

    track = track_video(truth, {"scale": scale})

    seen = track["detected"].to_numpy()
    assert track.loc[seen, "head_x"].notna().mean() > 0.95 and track.loc[seen, "pitch_deg"].notna().mean() > 0.95
    assert np.nanmax(head_error(track, truth)[seen]) < 2.0 * size[0] / 320  # original pixels
    # The drawn fish faces right, so its nose points up when the drawing angle is negative.
    drawn = np.array(truth["positions"])[:, 2]
    assert np.nanmax(np.abs(track["pitch_deg"].to_numpy() + drawn)) < 5


@pytest.mark.parametrize("angle, pitch", [(190.0, 10.0), (170.0, -10.0), (-30.0, 30.0), (30.0, -30.0)])
def test_pitch_does_not_depend_on_which_way_the_fish_faces(tmp_path, angle, pitch):
    # angle 190 = facing left with the nose 10 deg up; the body tilt alone (angle_deg) cannot
    # tell that apart from facing right with the nose 10 deg down.
    truth = make_video(tmp_path / "F_0042.avi", duration_s=1.0,
                       fish_path=lambda t: (110 + 60 * t, 120.0, angle))

    track = track_video(truth, roi=[40, 60, 280, 190])

    assert track["pitch_deg"].median() == pytest.approx(pitch, abs=3)
    facing_left = math.cos(math.radians(angle)) < 0
    assert ((track["head_x"] < track["x"]) == facing_left).all()


def test_no_clear_eye_means_no_head(tmp_path):
    truth = make_video(tmp_path / "F_0042.avi", duration_s=1.0)

    track = track_video(truth, {"head_min_contrast": 200})  # no eye is that much darker

    assert track["detected"].all() and track["head_x"].isna().all() and track["pitch_deg"].isna().all()


def test_frame_stride_keeps_times_exact(tmp_path):
    truth = make_video(tmp_path / "F_0042.avi", fps=30.0, duration_s=2.0)

    track = track_video(truth, {"frame_stride": 3})

    assert list(track["part_frame"][:3]) == [0, 3, 6]
    assert track["time_s"].iloc[1] == pytest.approx(3 / 30)
    assert len(track) == 20


def test_brightness_flicker_does_not_break_detection(tmp_path):
    # Every 10th frame the whole picture is 30 gray levels darker (lights / auto exposure).
    truth = make_video(tmp_path / "F_0042.avi", duration_s=4.0, flicker=(10, 30))

    track = track_video(truth)

    assert track["detected"].mean() > 0.95
    assert (track["n_blobs"] <= 1).all()


# --- gaps ----------------------------------------------------------------------------


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_short_gap_is_interpolated_long_gap_stays_empty(tmp_path, fps):
    truth = make_video(tmp_path / "F_0042.avi", fps=fps, duration_s=6.0, hidden=[(1.0, 1.2), (3.0, 5.0)])
    time = np.arange(truth["frame_count"]) / fps

    track = track_video(truth)

    short = (time >= 1.0 - 1e-9) & (time < 1.2 - 1e-9)
    long = (time >= 3.0 - 1e-9) & (time < 5.0 - 1e-9)
    assert not track.loc[short, "detected"].any() and track.loc[short, "interpolated"].all()
    assert track.loc[short, "x"].notna().all()
    assert position_error(track, truth)[short].max() < 6  # a straight line over 0.2 s of smooth swimming
    assert not track.loc[long, "interpolated"].any() and track.loc[long, "x"].isna().all()
    assert track["interpolated"].sum() == short.sum()  # nothing else was filled

    summary = summarize(track)
    assert summary["untracked_s"] == pytest.approx(2.0, abs=1.5 / fps)
    assert summary["interpolated_pct"] == pytest.approx(100 * short.mean(), abs=0.01)


def test_gap_at_the_start_is_not_interpolated(tmp_path):
    truth = make_video(tmp_path / "F_0042.avi", duration_s=2.0, hidden=[(0.0, 0.2)])

    track = track_video(truth)

    assert not track["interpolated"].iloc[:6].any() and track["x"].iloc[:6].isna().all()


def test_tilt_is_interpolated_across_the_vertical():
    # -85 deg and +85 deg are nearly the same body direction: the fill must pass through
    # +-90, not through 0 (a horizontal fish).
    track = pd.DataFrame({
        "time_s": [0.0, 0.1, 0.2], "detected": [True, False, True], "row_s": [0.1] * 3,
        "angle_deg": [-85.0, np.nan, 85.0], "x": [1.0, np.nan, 3.0], "y": [1.0, np.nan, 1.0],
        "top_y": [0.0, np.nan, 0.0], "bottom_y": [2.0, np.nan, 2.0], "area": [5.0, np.nan, 5.0],
        "major_axis": [4.0, np.nan, 4.0], "minor_axis": [1.0, np.nan, 1.0],
        "head_x": [2.0, np.nan, 4.0], "head_y": [1.0, np.nan, 1.0], "pitch_deg": [10.0, np.nan, 20.0],
    })

    filled = fill_gaps(track, 0.5)

    assert filled["interpolated"].tolist() == [False, True, False]
    assert abs(filled["angle_deg"].iloc[1]) == pytest.approx(90, abs=0.5)
    assert filled["x"].iloc[1] == pytest.approx(2.0)
    assert filled["head_x"].iloc[1] == pytest.approx(3.0) and filled["pitch_deg"].iloc[1] == pytest.approx(15.0)


# --- a large fish that rests in one place (the background must not keep it) ---------


def resting_then_swimming(truth_args):
    """A fish path that rests at one spot for the first 60 % of the video, then swims."""
    size, duration = truth_args["size"], truth_args["duration_s"]
    beaker = default_beaker(size, round(0.35 * size[1]))
    # A small margin (in fish lengths) leaves a large fish room to swim away from the spot.
    swim = lissajous_path(beaker, round(0.35 * size[1]), truth_args["fish_length"], duration, margin=0.6)
    rest = swim(0.6 * duration)  # rest where the swimming starts, so the path is continuous

    def path(t):
        return rest if t < 0.6 * duration else swim(t)

    return path


@pytest.fixture(scope="module")
def resting_fish(tmp_path_factory):
    """A large fish (25 % of the frame width) resting in one place for 60 % of the video."""
    args = {"size": (320, 240), "duration_s": 5.0, "fish_length": 0.25 * 320}
    truth = make_video(tmp_path_factory.mktemp("rest") / "F_0042.avi", size=args["size"],
                       duration_s=args["duration_s"], fish_length=args["fish_length"],
                       fish_path=resting_then_swimming(args))
    resting = np.arange(truth["frame_count"]) / truth["fps"] < 0.6 * args["duration_s"]
    return truth, resting


def test_resting_large_fish_is_tracked_with_the_default_background(resting_fish):
    truth, resting = resting_fish

    track = track_video(truth)

    # With a median background the resting fish would BE the background and vanish.
    assert track.loc[resting, "detected"].mean() > 0.95
    assert track["detected"].mean() > 0.95
    error = position_error(track, truth)[track["detected"].to_numpy()]
    assert error.max() < 2.0  # also after it leaves: the empty resting spot is never tracked
    assert track["major_axis"].median() == pytest.approx(0.25 * 320, rel=0.08)


def test_median_background_loses_a_resting_fish(resting_fish):
    # The failure the percentile background fixes (see scene.background_percentile).
    truth, resting = resting_fish
    scene = detect_scene(truth["path"], {**SCENE, "background_percentile": 50})
    part = track_part(truth["path"], scene.background, list(scene.roi), {**PARAMS, "polarity": "both"})

    assert part.loc[resting, "detected"].mean() < 0.5


def test_unknown_polarity_is_refused(tmp_path):
    truth = make_video(tmp_path / "F_0042.avi", duration_s=0.5)
    with pytest.raises(ValueError, match="polarity"):
        track_video(truth, {"polarity": "lighter"})


# --- frame-rate independence --------------------------------------------------------


def test_same_path_at_30_and_60_fps_gives_the_same_trajectory_in_seconds(tmp_path):
    slow = track_video(make_video(tmp_path / "slow.avi", fps=30.0, duration_s=4.0))
    fast = track_video(make_video(tmp_path / "fast.avi", fps=60.0, duration_s=4.0))

    # Every other 60-fps frame is at a 30-fps time.
    assert np.allclose(fast["time_s"].to_numpy()[::2], slow["time_s"].to_numpy())
    both = slow["detected"].to_numpy() & fast["detected"].to_numpy()[::2]
    distance = np.hypot(slow["x"].to_numpy() - fast["x"].to_numpy()[::2],
                        slow["y"].to_numpy() - fast["y"].to_numpy()[::2])[both]
    assert both.mean() > 0.95 and distance.max() < 2.0
    assert slow["major_axis"].median() == pytest.approx(fast["major_axis"].median(), rel=0.05)


# --- blob choice and the reflection --------------------------------------------------


def test_reflection_below_the_beaker_is_never_selected(tmp_path):
    truth = make_video(tmp_path / "F_0042.avi", duration_s=6.0)
    beaker_bottom = truth["beaker"][3]

    track = track_video(truth)

    assert (track.loc[track["detected"], "bottom_y"] < beaker_bottom).all()


def test_strong_reflection_inside_the_roi_does_not_steal_the_track(tmp_path):
    # Harder case: the ROI is the whole frame and the reflection is strong enough to be
    # a blob. The blob choice (largest, near the previous position) must stay on the fish.
    truth = make_video(tmp_path / "F_0042.avi", duration_s=6.0, reflection_drop=40)

    track = track_video(truth, roi=[0, 0, 320, 240])

    assert track["n_blobs"].max() >= 2  # the reflection really was a candidate
    assert (track.loc[track["detected"], "y"] < truth["beaker"][3]).all()
    assert position_error(track, truth)[track["detected"].to_numpy()].max() < 2.0


def swim_with_decoy(tmp_path, decoy_fraction, length_factor, duration_s=3.0):
    """A fish swimming left to right with a gray decoy (a stand-in mirror image) just below it.

    The decoy has `decoy_fraction` of the full fish area and never touches the fish.
    """
    length = 0.12 * 320
    below = 0.6 * length  # decoy centre this far under the fish centre: a clear gap between them

    def fish(t):
        return 100 + 120 * t / duration_s, 110.0, 0.0

    def decoy(t):
        x, y, _ = fish(t)
        return x, y + below, 0.0

    return make_video(tmp_path / "F_0042.avi", duration_s=duration_s, fish_path=fish, reflection=False,
                      length_factor=length_factor, decoy=decoy,
                      decoy_axes=(decoy_fraction * length / 2, length / 7))


def test_fish_turned_toward_the_camera_is_kept_next_to_a_bigger_mirror_image(tmp_path):
    # From 1 s to 2 s the fish looks half as long (turned toward the camera), so the decoy
    # (70 % of the full fish) is 1.4x bigger than it: the largest blob is NOT the fish.
    truth = swim_with_decoy(tmp_path, 0.7, lambda t: 0.5 if 1.0 <= t < 2.0 else 1.0)

    track = track_video(truth, roi=[0, 0, 320, 240])

    assert (track["n_blobs"] == 2).mean() > 0.95  # the decoy really is a candidate throughout
    assert track["detected"].all()
    assert position_error(track, truth).max() < 2.0  # never on the decoy
    # Choosing the largest blob (overlap overruled at any size difference) jumps to the decoy.
    largest = track_video(truth, {"switch_area_ratio": 1.0}, roi=[0, 0, 320, 240])
    assert (position_error(largest, truth) > 10).sum() > 20


def test_track_that_starts_on_a_mirror_image_switches_to_the_fish(tmp_path):
    # The fish starts turned toward the camera (30 % of its length) and the decoy (60 % of
    # the fish) is the largest blob, so the track starts on the decoy. When the fish turns
    # side-on it is > switch_area_ratio (1.5) times bigger and must take the track over.
    truth = swim_with_decoy(tmp_path, 0.6, lambda t: 0.3 if t < 0.5 else 1.0)

    track = track_video(truth, roi=[0, 0, 320, 240])

    time = track["time_s"].to_numpy()
    error = position_error(track, truth)
    assert error[time < 0.4].min() > 10  # on the decoy at first (nothing better to go by)
    assert error[time >= 0.6].max() < 2.0  # then on the fish for good


def test_choose_blob_keeps_the_blob_that_overlaps_last_frames_fish():
    areas = np.array([100.0, 140.0, 400.0])
    centroids = np.array([[50.0, 50.0], [50.0, 80.0], [200.0, 50.0]])
    overlaps = np.array([60, 0, 0])

    # Blob 0 overlaps the previous fish: kept over the bigger blob 1 right next to it...
    assert choose_blob(areas[:2], centroids[:2], 10, (50, 50), 400, overlaps[:2], 1.5) == (0, 2)
    # ...but a blob more than switch_area_ratio times bigger takes over (no lock-in).
    assert choose_blob(areas, centroids, 10, (50, 50), 400, overlaps, 1.5) == (2, 3)
    assert choose_blob(areas, centroids, 10, (50, 50), 400, overlaps, 5.0) == (0, 3)
    # Nothing overlaps: back to the distance rule.
    assert choose_blob(areas, centroids, 10, (50, 50), 40, np.zeros(3, int), 1.5) == (1, 3)


def test_choose_blob_prefers_the_largest_blob_near_the_previous_position():
    areas = np.array([100.0, 400.0, 5.0])
    centroids = np.array([[50.0, 50.0], [200.0, 50.0], [52.0, 50.0]])

    # A bigger ripple far away does not steal the track...
    assert choose_blob(areas, centroids, 10, previous=(48, 50), max_jump_px=40) == (0, 2)
    # ...but with no previous position, or nothing near it, the largest blob wins.
    assert choose_blob(areas, centroids, 10, previous=None, max_jump_px=40) == (1, 2)
    assert choose_blob(areas, centroids, 10, previous=(120, 200), max_jump_px=40) == (1, 2)
    # Blobs below the minimum area are ignored entirely.
    assert choose_blob(areas, centroids, 500, previous=None, max_jump_px=40) == (None, 0)


# --- whole subjects: split recordings, files, caching -------------------------------


def make_job(tmp_path, paths, subject="0012", force=False, params=None):
    scene_dir, tracks_dir = tmp_path / "scene", tmp_path / "tracks"
    scene_dir.mkdir(exist_ok=True)
    return TrackJob(subject, tuple(paths), tracks_dir, scene_dir, {**PARAMS, **(params or {})}, SCENE, {},
                    force, progress=False)


def test_split_recording_joins_parts_with_the_right_time_offset(tmp_path):
    part_a = make_video(tmp_path / "M_0012a.avi", fps=30.0, duration_s=3.0)
    part_b = make_video(tmp_path / "M_0012b.avi", fps=30.0, duration_s=2.0)
    job = make_job(tmp_path, [part_a["path"], part_b["path"]])

    record = process_subject(job)

    assert "error" not in record
    # Scene setup ran for both parts because their results were missing.
    assert (job.scene_dir / "M_0012a.json").is_file() and (job.scene_dir / "M_0012b.json").is_file()
    track = pd.read_csv(job.tracks_dir / "0012.csv.gz", dtype={"subject_id": str})
    assert list(track.columns) == TRACK_COLUMNS
    assert len(track) == 90 + 60
    assert (track["subject_id"] == "0012").all()
    b = track[track["part"] == 2]
    assert (track["part"] == 1).sum() == 90 and len(b) == 60
    assert b["part_frame"].tolist() == list(range(60))
    assert b["frame"].iloc[0] == 90 and b["time_s"].iloc[0] == pytest.approx(3.0)
    assert track["frame"].tolist() == list(range(150))
    assert np.allclose(np.diff(track["time_s"]), 1 / 30, atol=1e-4)  # continuous across the join
    assert [p["offset_s"] for p in record["parts"]] == pytest.approx([0.0, 3.0])
    assert [p["file"] for p in record["parts"]] == ["M_0012a.avi", "M_0012b.avi"]
    assert (job.tracks_dir / "0012_track_qa.png").is_file()


def test_split_parts_with_different_fps_use_each_files_own_timing(tmp_path):
    part_a = make_video(tmp_path / "M_0012a.avi", fps=30.0, duration_s=2.0)
    part_b = make_video(tmp_path / "M_0012b.avi", fps=60.0, duration_s=1.0)

    process_subject(make_job(tmp_path, [part_a["path"], part_b["path"]]))

    track = pd.read_csv(tmp_path / "tracks" / "0012.csv.gz")
    b = track[track["part"] == 2]
    # time_s is stored to 0.1 ms.
    assert b["time_s"].iloc[0] == pytest.approx(2.0) and b["time_s"].iloc[-1] == pytest.approx(2.0 + 59 / 60, abs=1e-4)
    assert b["frame"].iloc[0] == 60


def test_qa_image_draws_the_track_in_time_colors(tmp_path):
    truth = make_video(tmp_path / "F_0042.avi", duration_s=3.0)

    process_subject(make_job(tmp_path, [truth["path"]], subject="0042"))

    qa = cv2.imread(str(tmp_path / "tracks" / "0042_track_qa.png"))
    assert qa.shape[1] >= 640
    # The synthetic scene is gray; a colored (saturated) line can only be the drawn track.
    saturation = cv2.cvtColor(qa[44:], cv2.COLOR_BGR2HSV)[..., 1]
    assert (saturation > 120).sum() > 500


def test_subject_results_are_cached_until_force_or_new_settings(tmp_path):
    truth = make_video(tmp_path / "F_0042.avi", duration_s=2.0)
    job = make_job(tmp_path, [truth["path"]], subject="0042")

    assert process_subject(job)["cached"] is False
    assert process_subject(job)["cached"] is True
    assert process_subject(make_job(tmp_path, [truth["path"]], subject="0042", force=True))["cached"] is False
    assert process_subject(make_job(tmp_path, [truth["path"]], "0042", params={"diff_threshold": 25}))["cached"] is False
    # A new ROI from the scene step (e.g. a correction in overrides.yaml) also redoes the track.
    override_job = TrackJob("0042", (truth["path"],), job.tracks_dir, job.scene_dir, {**PARAMS, "diff_threshold": 25},
                            SCENE, {"F_0042.avi": {"roi": [10, 10, 310, 200]}}, False, False)
    assert process_subject(override_job)["cached"] is False


def test_unreadable_video_is_reported_not_raised(tmp_path):
    bad = tmp_path / "F_0042.avi"
    bad.write_bytes(b"not a video")

    record = process_subject(make_job(tmp_path, [bad], subject="0042"))

    assert "scene setup failed" in record["error"]


# --- the `track` command --------------------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project folder with a hand-written trials.csv: one single video, one split, one missing."""
    for name in ("FISH_VIDEO_DIR", "FISH_DB_PATH", "FISH_REFERENCE_PDF", "FISH_OUTPUT_DIR", "FISH_WORKERS", "FISH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    videos = tmp_path / "videos"
    videos.mkdir()
    for name, seconds in (("F_0042.avi", 2.0), ("M_0012a.avi", 1.5), ("M_0012b.avi", 1.0)):
        make_video(videos / name, duration_s=seconds)
    catalog = tmp_path / "outputs" / "catalog"
    catalog.mkdir(parents=True)
    pd.DataFrame({
        "subject_id": ["0042", "0012", "0044"],
        "video_status": ["matched", "matched", "missing"],
        "video_paths": [str(videos / "F_0042.avi"), f"{videos / 'M_0012a.avi'};{videos / 'M_0012b.avi'}", ""],
    }).to_csv(catalog / "trials.csv", index=False)
    (tmp_path / ".env").write_text("FISH_OUTPUT_DIR=outputs\nFISH_WORKERS=2\n")
    return tmp_path


def test_cli_track_subjects_caching_and_force(project, capsys):
    tracks = project / "outputs" / "tracks"

    assert main(["track", "--subjects", "F_0042"]) == 0
    out = capsys.readouterr().out
    assert "1 done (0 cached)" in out and "0042: detected" in out
    assert (tracks / "0042.csv.gz").is_file() and (tracks / "0042_track_qa.png").is_file()
    assert not (tracks / "0012.csv.gz").exists()  # not requested
    assert (project / "outputs" / "scene" / "F_0042.json").is_file()  # scene ran on the way

    # All subjects (2 worker processes when allowed): 0042 from the cache, 0044 has no video.
    assert main(["track"]) == 0
    out = capsys.readouterr().out
    assert "2 done (1 cached)" in out and "0044 (missing)" in out
    summary = pd.read_csv(tracks / "summary.csv", dtype={"subject_id": str})
    assert list(summary["subject_id"]) == ["0042", "0012"]
    assert list(summary.columns) == ["subject_id", "frames", "detected_pct", "interpolated_pct", "untracked_s",
                                     "median_body_length_px", "mean_n_blobs", "head_pct", "pitch_pct"]
    assert summary.set_index("subject_id").loc["0012", "frames"] == 45 + 30  # both parts
    assert (summary["detected_pct"] > 95).all()

    assert main(["track", "--subjects", "42", "--force"]) == 0
    assert "1 done (0 cached)" in capsys.readouterr().out
    meta = json.loads((tracks / "0042_track.json").read_text())
    assert meta["summary"]["frames"] == 60


def test_cli_track_unknown_subject(project, capsys):
    assert main(["track", "--subjects", "0099"]) == 2
    assert "0099" in capsys.readouterr().out
