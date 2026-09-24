"""Tests for fishbehavior.plots on a synthetic project (headless, Agg backend)."""

import json

import cv2
import matplotlib
import pandas as pd
import pytest
import yaml

from fishbehavior.cli import main
from fishbehavior.features import BIN_COLUMNS
from fishbehavior.labeling import CALM, ERRATIC, FREEZE, STATES, SURFACE, UNTRACKED, make_segments, summarize
from synthetic import make_video

FPS, SIZE, DPI = 10.0, (160, 120), 40
# subject -> (part durations in s, per-bin labels); both in one group, 0012 is a split recording.
SUBJECTS = {"0012": ((3.5, 2.5), [CALM, CALM, ERRATIC, FREEZE, FREEZE, UNTRACKED]),
            "0042": ((4.0,), [SURFACE, SURFACE, CALM, CALM])}


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Videos, scene results, tracks, labels, a reference row and trials.csv; returns (out folder, CLI options)."""
    root = tmp_path_factory.mktemp("plots")
    out = root / "out"
    for folder in ("catalog", "tracks", "labels", "scene", "reference"):
        (out / folder).mkdir(parents=True)
    trials, segments = [], []
    for subject_id, (durations, labels) in SUBJECTS.items():
        paths, frames, offset = [], [], 0.0
        for part, duration in enumerate(durations, start=1):
            video = make_video(root / f"F_{subject_id}_{'ab'[part - 1]}.avi", fps=FPS, size=SIZE, duration_s=duration)
            paths.append(str(video["path"]))
            (out / "scene" / f"{video['path'].stem}.json").write_text(json.dumps(
                {"roi": list(video["beaker"]), "waterline_y": video["waterline_y"]}), encoding="utf-8")
            for k, (x, y, angle) in enumerate(video["positions"]):
                frames.append({"part": part, "part_frame": k, "time_s": round(offset + k / FPS, 4), "x": x, "y": y,
                               "major_axis": 2 * video["fish_axes"][0], "minor_axis": 2 * video["fish_axes"][1],
                               "angle_deg": angle})
            offset += duration
        track = pd.DataFrame(frames)
        track.to_csv(out / "tracks" / f"{subject_id}.csv.gz", index=False)
        labeled = pd.DataFrame({"subject_id": subject_id, "bin": range(len(labels))}).reindex(columns=BIN_COLUMNS)
        labeled = labeled.assign(label=labels, confidence=0.9)
        labeled.to_csv(out / "labels" / f"{subject_id}_bins.csv", index=False)
        segments.append(make_segments(labeled, track, bin_s=1.0))
        trials.append({"subject_id": subject_id, "compound": "CPD_A", "concentration_raw": "0.03",
                       "video_status": "matched", "video_paths": ";".join(paths)})
    trials.append({"subject_id": "0099", "compound": "Veh", "concentration_raw": "1% DMSO",
                   "video_status": "missing", "video_paths": ""})
    pd.DataFrame(trials).to_csv(out / "catalog" / "trials.csv", index=False)
    summarize(pd.concat(segments)).to_csv(out / "labels" / "summary.csv", index=False)
    pd.DataFrame({"subject_id": "0042", "group": "REF_A", "second": range(1200),
                  "label": [FREEZE] * 600 + ["no_data"] * 600}).to_csv(out / "reference" / "timelines.csv", index=False)
    (root / ".env").write_text(f"FISH_OUTPUT_DIR={out}\n", encoding="utf-8")
    (root / "config.yaml").write_text(yaml.safe_dump({"plots": {"dpi": DPI, "overlay_speed": 1.0}}), encoding="utf-8")
    return out, ["--env-file", str(root / ".env"), "--config", str(root / "config.yaml")]


def frames_in(path):
    """Frame count and frame size of a video, by reading it."""
    capture, count, shape = cv2.VideoCapture(str(path)), 0, None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        count, shape = count + 1, frame.shape
    capture.release()
    return count, shape


def test_plotting_runs_headless():
    import fishbehavior.plots  # noqa: F401 - importing selects the backend

    assert matplotlib.get_backend().lower() == "agg"


def test_ethograms_and_bar_charts_are_written_with_the_expected_sizes(project):
    out, cli = project

    assert main([*cli, "plot"]) == 0

    ethogram = cv2.imread(str(out / "plots" / "ethogram_CPD_A_0.03.png"))
    # Two panels (ours + reference, since 0042 has a reference row) of 6.5 in each; rows 0.18 in + 1.2 in.
    assert ethogram.shape[:2] == (round((1.2 + 0.18 * 2) * DPI), round(2 * 6.5 * DPI))
    assert not (out / "plots" / "ethogram_Veh_1_DMSO.png").exists()  # no labeled subject in that group
    for state in STATES:
        bars = cv2.imread(str(out / "plots" / f"bars_{state}.png"))
        assert bars is not None and bars.shape[1] >= 6 * DPI


def test_overlay_has_one_frame_per_video_frame_in_the_range_across_the_part_boundary(project):
    out, cli = project

    assert main([*cli, "plot", "--subjects", "12", "--overlay", "--start", "2", "--end", "5"]) == 0

    count, shape = frames_in(out / "plots" / "overlay" / "0012_2-5s.mp4")
    assert count == round(3 * FPS)  # 2.0 <= t < 5.0 s: 15 frames of part a + 15 of part b
    assert shape[1] == SIZE[0] and shape[0] > SIZE[1]  # the timeline strip is added under the picture


def test_faster_overlay_keeps_every_nth_frame(project, tmp_path):
    out, cli = project
    config = tmp_path / "fast.yaml"
    config.write_text(yaml.safe_dump({"plots": {"dpi": DPI, "overlay_speed": 3.0}}), encoding="utf-8")

    assert main([cli[0], cli[1], "--config", str(config), "plot", "--subjects", "42", "--overlay",
                 "--end", "3", "--force"]) == 0

    assert frames_in(out / "plots" / "overlay" / "0042_0-3s.mp4")[0] == 10  # 30 frames, every 3rd


def test_overlay_needs_subjects(project):
    _, cli = project
    assert main([*cli, "plot", "--overlay"]) == 2  # one-line config error, not a traceback
