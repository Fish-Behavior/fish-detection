"""Tests for fishbehavior.export on a synthetic project: one single-part subject, one split
recording (two part files), and one subject without video."""

import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from fishbehavior.cli import main
from fishbehavior.features import BIN_COLUMNS
from fishbehavior.export import behavior_column_names, behavior_columns
from fishbehavior.labeling import CALM, ERRATIC, FREEZE, LABELS, STATES, SURFACE, UNTRACKED, make_segments
from synthetic import make_video

FPS, SIZE = 10.0, (160, 120)
# subject -> (part durations in s, per-bin labels, per-bin confidence)
SUBJECTS = {
    "0012": ((3.5, 2.5), [CALM, CALM, ERRATIC, FREEZE, FREEZE, UNTRACKED], [0.9, 0.9, 0.8, 0.3, 0.9, 1.0]),
    "0042": ((4.0,), [SURFACE, SURFACE, CALM, CALM], [0.7, 0.7, 0.9, 0.9]),
}


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Videos, tracks, labels, endpoints and trials.csv of the synthetic project; returns (root, CLI options)."""
    root = tmp_path_factory.mktemp("export")
    out = root / "out"
    for folder in ("catalog", "tracks", "labels", "features"):
        (out / folder).mkdir(parents=True)
    trials, endpoints, segments = [], [], []
    for subject_id, (durations, labels, confidence) in SUBJECTS.items():
        paths, frames, offset = [], [], 0.0
        for part, duration in enumerate(durations, start=1):
            video = make_video(root / f"F_{subject_id}_{'ab'[part - 1]}.avi", fps=FPS, size=SIZE, duration_s=duration)
            paths.append(str(video["path"]))
            for k, (x, y, _) in enumerate(video["positions"]):
                frames.append({"part": part, "part_frame": k, "time_s": round(offset + k / FPS, 4), "x": x, "y": y})
            offset += duration
        track = pd.DataFrame(frames).assign(subject_id=subject_id)
        track.to_csv(out / "tracks" / f"{subject_id}.csv.gz", index=False)
        labeled = pd.DataFrame({"subject_id": subject_id, "bin": range(len(labels)), "t_start_s": range(len(labels)),
                                "tracked_fraction": 1.0, "speed_mean_bl_s": 1.0})
        labeled = labeled.reindex(columns=BIN_COLUMNS).assign(label=labels, confidence=confidence)
        labeled.to_csv(out / "labels" / f"{subject_id}_bins.csv", index=False)
        segments.append(make_segments(labeled, track, bin_s=1.0))
        trials.append({"subject_id": subject_id, "sex": "F", "compound": "CPD_A", "concentration_raw": "0.03",
                       "concentration_mm": 0.03, "tdm_full": 100.0, "video_status": "matched",
                       "video_paths": ";".join(paths)})
        endpoints.append({"subject_id": subject_id, "body_length_px": 0.12 * SIZE[0], "duration_s": offset,
                          "distance_bl": 12.0})
    trials.append({"subject_id": "0099", "sex": "M", "compound": "Veh", "concentration_raw": "1% DMSO",
                   "concentration_mm": np.nan, "tdm_full": 250.0, "video_status": "missing", "video_paths": ""})
    pd.DataFrame(trials).to_csv(out / "catalog" / "trials.csv", index=False)
    pd.DataFrame(endpoints).to_csv(out / "features" / "endpoints.csv", index=False)
    pd.concat(segments).to_csv(out / "labels" / "segments.csv", index=False)
    (root / ".env").write_text(f"FISH_OUTPUT_DIR={out}\n", encoding="utf-8")
    (root / "config.yaml").write_text(yaml.safe_dump({"export": {"folds": 2, "clip_frames": 4, "crop_size": 32}}),
                                      encoding="utf-8")
    return root, ["--env-file", str(root / ".env"), "--config", str(root / "config.yaml")]


@pytest.fixture(scope="module")
def exported(project):
    """Run `export --clips` once; returns the datasets folder."""
    root, cli = project
    assert main([*cli, "export", "--clips"]) == 0
    return root / "out" / "datasets"


def test_one_row_per_subject_with_every_column_and_seconds_summing_to_the_video(exported):
    dataset = pd.read_csv(exported / "behavior_dataset.csv", dtype={"subject_id": str}).set_index("subject_id")
    segments = pd.read_csv(exported / "segments.csv", dtype={"subject_id": str})

    assert sorted(dataset.index) == ["0012", "0042", "0099"]
    assert set(behavior_column_names()) <= set(dataset.columns)
    assert {"tdm_full", "distance_bl", "body_length_px"} <= set(dataset.columns)  # workbook + endpoints
    assert {"sex", "compound", "concentration_raw"} <= set(segments.columns)
    for subject_id in SUBJECTS:
        total = segments.loc[segments["subject_id"] == subject_id, "duration_s"].sum()
        assert dataset.loc[subject_id, [f"{s}_s" for s in STATES] + ["untracked_s"]].sum() == pytest.approx(total)
    missing = dataset.loc["0099"]
    assert missing["tdm_full"] == 250 and missing[behavior_column_names()].isna().all()  # workbook kept, behavior empty


def test_bouts_latency_and_transitions():
    # A bout cut in two by a part boundary still counts as one bout.
    segments = pd.DataFrame({"label": [CALM, FREEZE, FREEZE, CALM, ERRATIC],
                             "start_s": [0, 10, 14, 20, 25], "duration_s": [10, 4, 6, 5, 5]})
    row = behavior_columns(segments)
    assert row["freeze_drift_bouts"] == 1 and row["freeze_drift_s"] == 10 and row["freeze_drift_latency_s"] == 10
    assert row["controlled_swim_bouts"] == 2 and row["controlled_swim_mean_bout_s"] == 7.5
    assert row["controlled_swim_to_freeze_drift"] == 1 and row["freeze_drift_to_controlled_swim"] == 1
    assert row["controlled_swim_to_erratic"] == 1 and np.isnan(row["lorr_latency_s"])  # never shown: empty
    assert len([k for k in row if "_to_" in k]) == len(LABELS) * (len(LABELS) - 1)


def test_windows_use_part_local_frames_and_folds_never_split_a_subject(exported, project):
    root, _ = project
    windows = pd.read_csv(exported / "behavior_windows.csv", dtype={"subject_id": str})

    assert windows.groupby("subject_id")["fold"].nunique().eq(1).all()  # one fish, one fold
    assert windows["fold"].nunique() == 2
    split = windows[windows["subject_id"] == "0012"]
    # Second 3-4 is cut where part a ends at 3.5 s: frames 30-34 of part a, then frames 0-4 of part b.
    at_boundary = split[split["start_s"].between(3.0, 3.9)]
    assert at_boundary[["part", "part_start_frame", "part_end_frame"]].values.tolist() == [[1, 30, 34], [2, 0, 4]]
    assert at_boundary["video_path"].tolist() == [str(root / "F_0012_a.avi"), str(root / "F_0012_b.avi")]
    assert split["part_end_frame"].max() <= 34 and split[split["part"] == 2]["part_end_frame"].max() == 24
    usable = dict(zip(split["start_s"].round(1), split["use_for_training"]))
    assert usable[0.0] and not usable[3.0] and not usable[5.0]  # confidence 0.3 < 0.5; untracked


def test_per_second_labels_and_xlsx_sheets(exported):
    seconds = pd.read_csv(exported / "per_second_labels.csv", dtype={"subject_id": str})
    assert seconds[seconds["subject_id"] == "0012"]["label"].tolist() == SUBJECTS["0012"][1]
    assert {"confidence", "speed_mean_bl_s", "tracked_fraction"} <= set(seconds.columns)
    sheets = pd.read_excel(exported / "behavior_dataset.xlsx", sheet_name=None)
    assert list(sheets) == ["behavior_dataset", "column_guide"]
    guide = sheets["column_guide"].set_index("column")["meaning"]
    assert list(guide.index) == list(sheets["behavior_dataset"].columns) and guide.notna().all()


def test_clips_have_the_expected_shape_and_are_centered_on_the_fish(exported):
    windows = pd.read_csv(exported / "behavior_windows.csv", dtype={"subject_id": str})
    clips = np.load(exported / "clips" / "0012.npz")

    n = int((windows["subject_id"] == "0012").sum())
    assert clips["frames"].shape == (n, 4, 32, 32) and clips["frames"].dtype == np.uint8
    assert list(clips["window"]) == list(range(n)) and len(clips["label"]) == n
    center = clips["frames"][:, :, 14:18, 14:18].astype(float).mean()
    assert center < clips["frames"][:, :, :4, :4].astype(float).mean() - 20  # dark fish in the middle


def test_another_labels_folder_is_exported_next_to_the_main_one(exported, project):
    root, cli = project
    backup = root / "out" / "labels_backup"
    shutil.copytree(root / "out" / "labels", backup)
    segments = pd.read_csv(backup / "segments.csv", dtype={"subject_id": str})
    segments.loc[segments["label"] == CALM, "label"] = ERRATIC  # a different labeling
    segments.to_csv(backup / "segments.csv", index=False)

    assert main([*cli, "export", "--labels", str(backup)]) == 0

    main_rows = pd.read_csv(exported / "behavior_dataset.csv", dtype={"subject_id": str}).set_index("subject_id")
    other = pd.read_csv(root / "out" / "datasets_labels_backup" / "behavior_dataset.csv",
                        dtype={"subject_id": str}).set_index("subject_id")
    assert other.loc["0042", "controlled_swim_s"] == 0 and main_rows.loc["0042", "controlled_swim_s"] > 0
    assert main([*cli, "export", "--labels", str(root / "nowhere")]) == 2  # clear config error
