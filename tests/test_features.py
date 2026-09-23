"""Tests for fishbehavior.features on synthetic track tables (no video needed).

Tracks are built directly in the tracking.py output format, with a body length of
BL_PX pixels, so the expected speeds in BL/s are known exactly.
"""

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fishbehavior.config import load_settings
from fishbehavior.features import (
    BIN_COLUMNS,
    FRAME_COLUMNS,
    QA_COLUMNS,
    FeatureJob,
    bin_features,
    body_length_px,
    endpoint_row,
    frame_features,
    odd_window,
    process_subject,
    qa_subject,
)
from fishbehavior.tracking import TRACK_COLUMNS

PARAMS = load_settings(environ={}).params["features"]
BL_PX = 20.0
WATERLINE_Y = 50.0
ROI_BOTTOM_Y = 250.0  # top half = depth <= (250 - 50) / 2 / 20 = 5 BL


def make_track(x, y, fps, **columns):
    """A one-part track table with the given centroid path; the fish is BL_PX long."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    seen = np.isfinite(x)
    table = pd.DataFrame({
        "subject_id": "0042", "part": 1, "part_frame": np.arange(n), "frame": np.arange(n),
        "time_s": np.arange(n) / fps, "detected": seen, "x": x, "y": y,
        "top_y": np.where(seen, y - 3, np.nan), "bottom_y": np.where(seen, y + 3, np.nan),
        "area": np.where(seen, 80.0, np.nan), "major_axis": np.where(seen, BL_PX, np.nan),
        "minor_axis": np.where(seen, 5.0, np.nan), "angle_deg": np.where(seen, 0.0, np.nan),
        "head_x": np.nan, "head_y": np.nan, "pitch_deg": np.nan, "n_blobs": seen.astype(int),
        "interpolated": False,
    })
    for name, values in columns.items():
        table[name] = values
    return table[TRACK_COLUMNS]


def features(track, **params):
    """Frames, bins and endpoint row of one synthetic track."""
    params = {**PARAMS, **params}
    bl = body_length_px(track)
    frames = frame_features(track, bl, WATERLINE_Y, params)
    return frames, bin_features(frames, params), endpoint_row(frames, bl, WATERLINE_Y, ROI_BOTTOM_Y, params)


def straight(fps, speed_bl_s=2.0, duration_s=5.0):
    """Horizontal swim at a constant speed, mid-water."""
    t = np.arange(round(duration_s * fps)) / fps
    return make_track(30 + speed_bl_s * BL_PX * t, np.full(len(t), 200.0), fps)


# --- speed ---------------------------------------------------------------------------


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_straight_line_gives_the_true_speed_in_bl_per_second_at_any_fps(fps):
    frames, bins, endpoint = features(straight(fps, speed_bl_s=2.0))
    speed = frames["speed_bl_s"].iloc[1:]
    assert speed.to_numpy() == pytest.approx(2.0, rel=1e-6)
    assert bins["speed_mean_bl_s"].to_numpy() == pytest.approx(2.0, rel=1e-6)
    assert frames["accel_bl_s2"].abs().max() < 1e-6
    assert endpoint["distance_bl"] == pytest.approx(2.0 * (5.0 - 1 / fps), abs=0.01)  # endpoints keep 2 decimals
    fast = endpoint_row(frames, BL_PX, WATERLINE_Y, ROI_BOTTOM_Y, {**PARAMS, "high_mobility_bl_s": 1.5})
    slow = endpoint_row(frames, BL_PX, WATERLINE_Y, ROI_BOTTOM_Y, {**PARAMS, "high_mobility_bl_s": 2.5})
    assert fast["highly_mobile_s"] == pytest.approx(5.0 - 1 / fps, abs=0.01) and slow["highly_mobile_s"] == 0.0


def test_frame_stride_rows_still_give_the_true_speed():
    track = straight(60.0).iloc[::3].reset_index(drop=True)  # like tracking.frame_stride = 3
    frames, _, _ = features(track)
    assert frames["speed_bl_s"].iloc[1:].to_numpy() == pytest.approx(2.0, rel=1e-6)


def test_stationary_fish_has_zero_speed_and_is_immobile_the_whole_time():
    n = 150
    frames, bins, endpoint = features(make_track(np.full(n, 100.0), np.full(n, 200.0), 30.0))
    assert frames["speed_bl_s"].max() == pytest.approx(0.0, abs=1e-9)
    assert frames["turn_deg"].isna().all()  # no heading when not moving
    assert endpoint["immobile_s"] == pytest.approx(n / 30 - 1 / 30, abs=0.01)  # first row has no speed
    assert endpoint["distance_bl"] == pytest.approx(0.0)
    assert bins["speed_max_bl_s"].max() == pytest.approx(0.0, abs=1e-9)


# --- turning -------------------------------------------------------------------------


def test_zigzag_turns_more_than_a_straight_path():
    fps, n = 30.0, 300
    t = np.arange(n) / fps
    x = 30 + 3 * BL_PX * t
    zig = 200 + 2 * BL_PX * np.abs((t % 1.0) - 0.5)  # triangle wave: up/down every half second
    _, zig_bins, zig_end = features(make_track(x, zig, fps))
    _, line_bins, line_end = features(make_track(x, np.full(n, 200.0), fps))
    assert zig_bins["turn_rate_var_deg2_s2"].mean() > 100 * max(line_bins["turn_rate_var_deg2_s2"].mean(), 1e-6)
    assert zig_end["meander_deg_per_bl"] > 10 * max(line_end["meander_deg_per_bl"], 1e-6)
    assert zig_end["mean_angular_velocity_deg_s"] > line_end["mean_angular_velocity_deg_s"]


def test_turn_angle_wraps_around_180_degrees():
    fps = 30.0
    angles = np.radians(np.r_[np.full(30, 178.0), np.full(30, -178.0)])  # heading 178 -> -178: a 4 degree turn
    steps = 2 * BL_PX / fps
    x = 500 + np.cumsum(steps * np.cos(angles))
    y = 200 - np.cumsum(steps * np.sin(angles))
    frames, _, _ = features(make_track(x, y, fps), smooth_s=0.0)
    assert frames["turn_deg"].abs().max() < 10


# --- gaps, surface, tilt, bins -------------------------------------------------------


def test_nan_gap_does_not_crash_and_lowers_tracked_fraction():
    track = straight(30.0, duration_s=6.0)
    gap = (track["time_s"] >= 2.0) & (track["time_s"] < 4.0)
    track.loc[gap, ["x", "y", "top_y", "bottom_y", "area", "major_axis", "minor_axis", "angle_deg"]] = np.nan
    track.loc[gap, "detected"] = False
    frames, bins, endpoint = features(track)
    assert list(bins["tracked_fraction"]) == pytest.approx([1, 1, 0, 0, 1, 1])
    assert frames.loc[gap, "speed_bl_s"].isna().all()
    good = frames.loc[~gap & (frames["time_s"] > 4.2), "speed_bl_s"]
    assert good.to_numpy() == pytest.approx(2.0, rel=1e-6)  # speed after the gap is unaffected
    assert endpoint["tracked_s"] == pytest.approx(4.0, abs=1e-6)


def test_surface_flag_and_tilt_fraction_on_crafted_rows():
    fps = 4.0  # 4 rows = one 1-second bin
    # Surface limit = waterline 50 + 0.25 BL * 20 px = 55.
    track = make_track(np.full(4, 100.0), np.full(4, 80.0), fps,
                       top_y=[40.0, 55.0, 56.0, 120.0], angle_deg=[-80.0, 10.0, 50.0, -30.0])
    frames, bins, _ = features(track)
    assert list(frames["at_surface"]) == [True, True, False, False]
    assert list(frames["tilt_deg"]) == [80.0, 10.0, 50.0, 30.0]
    assert bins.loc[0, "surface_fraction"] == pytest.approx(0.5)
    assert bins.loc[0, "tilt_fraction"] == pytest.approx(0.5)  # 80 and 50 are above 45
    assert bins.loc[0, "tilt_median_deg"] == pytest.approx(40.0)
    assert frames["depth_bl"].to_numpy() == pytest.approx((80 - 50) / BL_PX)


@pytest.mark.parametrize("n, fps, bin_s", [(95, 30.0, 1.0), (95, 30.0, 0.5), (120, 60.0, 1.0), (61, 60.0, 0.25)])
def test_bin_count_is_ceil_of_duration_over_bin_size(n, fps, bin_s):
    _, bins, _ = features(make_track(np.linspace(0, 100, n), np.full(n, 200.0), fps), bin_s=bin_s)
    assert len(bins) == math.ceil(n / fps / bin_s - 1e-9)
    assert list(bins["t_start_s"]) == pytest.approx([i * bin_s for i in range(len(bins))])
    assert bins["n_frames"].sum() == n
    assert list(bins.columns) == BIN_COLUMNS


def test_top_half_share_and_latency():
    fps = 10.0
    # 3 s deep (y = 230 -> 9 BL below the waterline), then 2 s high (y = 100 -> 2.5 BL).
    y = np.r_[np.full(30, 230.0), np.full(20, 100.0)]
    _, _, endpoint = features(make_track(np.full(50, 100.0), y, fps))
    assert endpoint["latency_top_half_s"] == pytest.approx(3.0)
    assert endpoint["top_half_pct"] == pytest.approx(40.0)


# --- one subject on disk, with caching -----------------------------------------------


def subject_on_disk(tmp_path, track):
    """Write a track + scene result like `track` does; return the job for subject 0042."""
    tracks, scenes, out = tmp_path / "tracks", tmp_path / "scene", tmp_path / "features"
    tracks.mkdir(parents=True, exist_ok=True)
    scenes.mkdir(exist_ok=True)
    track.to_csv(tracks / "0042.csv.gz", index=False)
    (scenes / "F_0042.json").write_text(json.dumps({"waterline_y": 50, "roi": [0, 30, 320, 250]}))
    return FeatureJob("0042", tmp_path / "F_0042.mp4", tracks, scenes, out, PARAMS, force=False)


def test_process_subject_writes_outputs_and_reuses_them(tmp_path):
    job = subject_on_disk(tmp_path, straight(30.0))
    out = job.features_dir

    first = process_subject(job)
    assert "error" not in first and not first["cached"]
    frames = pd.read_csv(out / "0042_frames.csv.gz", dtype={"subject_id": str})
    assert list(frames.columns) == FRAME_COLUMNS and frames["subject_id"].iloc[0] == "0042"
    assert len(pd.read_csv(out / "0042_bins.csv.gz")) == 5
    assert process_subject(job)["cached"]
    assert not process_subject(replace(job, params={**PARAMS, "bin_s": 2.0}))["cached"]  # changed setting: redo


def test_missing_track_is_reported_not_raised(tmp_path):
    job = FeatureJob("0043", tmp_path / "F_0043.mp4", tmp_path, tmp_path, tmp_path / "f", PARAMS, force=False)
    assert "run `track` first" in process_subject(job)["error"]


# --- smoothing window and QA ---------------------------------------------------------


@pytest.mark.parametrize("fps, expected", [(15.0, 5), (30.0, 7), (60.0, 13)])
def test_smoothing_window_never_shrinks_to_a_no_op(fps, expected):
    assert odd_window(PARAMS["smooth_s"], 1 / fps, polyorder=2) == expected


def test_qa_passes_a_clean_swim_and_flags_turning_on_a_jittery_one(tmp_path):
    fps, n = 30.0, 900
    t = np.arange(n) / fps
    x = 30 + 2 * BL_PX * t
    y = 200 + BL_PX * np.sin(2 * np.pi * t / 6)  # a slow, real wave: 6 s period

    clean = {row["feature"]: row for row in qa_subject(subject_on_disk(tmp_path / "a", make_track(x, y, fps)))["rows"]}
    assert clean["speed_mean_bl_s"]["stable"] and clean["distance_bl"]["stable"]
    assert clean["turn_rate_var_deg2_s2"]["stable"]
    assert clean["speed_mean_bl_s"]["jitter_bl"] < 0.01
    assert list(clean["speed_mean_bl_s"]) == QA_COLUMNS

    noise = np.random.default_rng(0).normal(0, 0.05 * BL_PX, (2, n))  # 0.05 BL centroid jitter
    noisy = {row["feature"]: row for row in
             qa_subject(subject_on_disk(tmp_path / "b", make_track(x + noise[0], y + noise[1], fps)))["rows"]}
    assert not noisy["turn_rate_var_deg2_s2"]["stable"]
    assert noisy["speed_mean_bl_s"]["jitter_bl"] > clean["speed_mean_bl_s"]["jitter_bl"]


def test_qa_reports_a_missing_track(tmp_path):
    job = FeatureJob("0043", tmp_path / "F_0043.mp4", tmp_path, tmp_path, tmp_path / "f", PARAMS, force=False)
    assert "run `track` first" in qa_subject(job)["error"]
