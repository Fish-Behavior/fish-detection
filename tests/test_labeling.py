"""Tests for fishbehavior.labeling: crafted bins tables, and one scripted synthetic video end to end."""

import json
import math
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from fishbehavior.cli import main
from fishbehavior.config import ConfigError, load_settings
from fishbehavior.features import BIN_COLUMNS
from fishbehavior.labeling import (
    CALM,
    ERRATIC,
    FREEZE,
    LABELS,
    LORR,
    SEGMENT_COLUMNS,
    SURFACE,
    SWIM,
    UNTRACKED,
    apply_label_overrides,
    check_label_overrides,
    clean_bouts,
    erratic_probability,
    fit_swim_model,
    label_bins,
    labeling_params,
    make_segments,
    rule_labels,
    summarize,
)
from synthetic import make_video

PARAMS = load_settings(environ={}).params["labeling"]
CALM_BIN = {"tracked_fraction": 1.0, "nose_up_surface_fraction": 0.0, "tilt_fraction": 0.0, "speed_median_bl_s": 1.0,
            "speed_mean_bl_s": 1.0,
            "speed_cv": 0.3, "jerk_abs_mean_bl_s3": 40.0, "turn_rate_var_deg2_s2": 1e3, "meander_deg_per_bl": 20.0}
JERKY_BIN = {**CALM_BIN, "speed_cv": 1.4, "jerk_abs_mean_bl_s3": 1500.0, "turn_rate_var_deg2_s2": 5e6,
             "meander_deg_per_bl": 600.0}


def make_bins(stretches, bin_s=1.0):
    """A bins table from [(n_bins, {column: value}), ...]; unset columns are a calm swim bin."""
    rows = [{**CALM_BIN, **values} for n, values in stretches for _ in range(n)]
    table = pd.DataFrame(rows).reindex(columns=BIN_COLUMNS)
    table["subject_id"], table["bin"] = "0042", range(len(table))
    table["t_start_s"] = table["bin"] * bin_s
    return table


def smooth_and_jerky(n, seed=0):
    """n calm and n jerky swim bins with a little noise, alternating in stretches of 5."""
    rng = np.random.default_rng(seed)
    stretches = [(5, CALM_BIN if (i // 5) % 2 == 0 else JERKY_BIN) for i in range(0, 2 * n, 5)]
    table = make_bins(stretches)
    for column in ("speed_cv", "jerk_abs_mean_bl_s3", "turn_rate_var_deg2_s2", "meander_deg_per_bl"):
        table[column] *= rng.uniform(0.7, 1.3, len(table))
    truth = np.array([ERRATIC if (i // 5) % 2 else CALM for i in range(len(table))])
    return table, truth


# --- rules ---------------------------------------------------------------------------------


def test_rules_follow_the_priority_order():
    slow, tilted = {"speed_median_bl_s": 0.05}, {"tilt_fraction": 0.9, "speed_median_bl_s": 0.3}
    bins = make_bins([
        (2, {"tracked_fraction": 0.2, "nose_up_surface_fraction": 1.0}),  # untracked beats surface
        (3, {"nose_up_surface_fraction": 0.6, "tilt_fraction": 1.0, "speed_median_bl_s": 0.0}),  # surface beats lorr/freeze
        (4, {**tilted, "speed_median_bl_s": 0.0}),  # tilted + still: lorr beats freeze
        (1, {}),
        (2, tilted),  # tilted but only 2 s (< lorr_min_s 3): swimming
        (3, slow),  # still for 3 s: freeze
        (1, {}),
        (1, slow),  # still for 1 s (< freeze_min_s 2): swimming
        (2, {}),
    ])
    label, confidence = rule_labels(bins, PARAMS, bin_s=1.0)

    assert list(label) == ([UNTRACKED] * 2 + [SURFACE] * 3 + [LORR] * 4 + [SWIM] * 3 + [FREEZE] * 3 + [SWIM] * 4)
    assert confidence[:2] == pytest.approx(0.8)  # 80% of frames not seen
    assert confidence[2:5] == pytest.approx(0.6)  # 60% of frames at the surface
    assert confidence[5:9] == pytest.approx(0.9)  # 90% of frames tilted
    assert confidence[12:15] == pytest.approx(0.5)  # 0.05 BL/s is half of the 0.1 freeze threshold
    assert np.isnan(confidence[label == SWIM]).all()


def test_sustained_rules_count_seconds_not_bins():
    bins = make_bins([(4, {"speed_median_bl_s": 0.0})], bin_s=0.5)  # 2 s at 0.5 s bins
    assert list(rule_labels(bins, {**PARAMS, "freeze_min_s": 2}, bin_s=0.5)[0]) == [FREEZE] * 4
    assert list(rule_labels(bins, {**PARAMS, "freeze_min_s": 2.5}, bin_s=0.5)[0]) == [SWIM] * 4


# --- bout cleanup, human relabels and segments ---------------------------------------------


def test_short_bouts_merge_into_the_longer_neighbour_but_untracked_stays():
    label = np.array([CALM] * 3 + [ERRATIC] + [FREEZE] * 5 + [UNTRACKED] + [CALM] * 2, dtype=object)
    confidence = np.full(len(label), 0.9)
    min_bout = {**PARAMS["min_bout_s"], ERRATIC: 2, CALM: 3}

    cleaned, conf = clean_bouts(label, confidence, 1.0, min_bout)

    # The 1 s erratic bout joins the 5 s freeze (longer than the 3 s swim before it); the
    # 1 s untracked gap is never hidden, so the 2 s calm bout after it joins it instead.
    assert list(cleaned) == [CALM] * 3 + [FREEZE] * 6 + [UNTRACKED] * 3
    assert conf[3] == 0 and conf[10] == 0 and conf[0] == 0.9  # merged bins lose their confidence


def test_human_relabels_win_over_the_automatic_labels_and_bad_rows_are_refused():
    labeled = make_bins([(6, {})], bin_s=0.5).assign(label=[CALM] * 6, confidence=0.4)
    overrides = pd.DataFrame({"subject_id": "0042", "start_s": [0.5, 1.0], "end_s": [2.0, 1.5], "label": [FREEZE, LORR]})

    out = apply_label_overrides(labeled, check_label_overrides(overrides, "test"), bin_s=0.5)

    # Bin middles 0.75, 1.25, 1.75 s are in 0.5-2.0 s: freeze, except 1.25 s, which the later row makes lorr.
    assert list(out["label"]) == [CALM, FREEZE, LORR, FREEZE, CALM, CALM]
    assert list(out["auto_label"]) == [CALM] * 6
    assert list(out["label_source"]) == ["auto", "human", "human", "human", "auto", "auto"]
    assert list(out["confidence"]) == [0.4, 1.0, 1.0, 1.0, 0.4, 0.4]  # a person decided: confidence 1
    for bad in ({"label": "sleeping"}, {"start_s": 3.0}, {"end_s": "soon"}):
        with pytest.raises(ConfigError, match="bad relabel"):
            check_label_overrides(overrides.assign(**bad), "test")


def test_segments_tile_the_video_and_split_at_the_part_boundary():
    fps, part_a_s, part_b_s = 30.0, 5.5, 4.5
    labeled = make_bins([(10, {})]).assign(label=[CALM] * 4 + [FREEZE] * 3 + [CALM] * 3, confidence=0.8)
    na, nb = round(part_a_s * fps), round(part_b_s * fps)
    timeline = pd.DataFrame({"part": [1] * na + [2] * nb, "part_frame": [*range(na), *range(nb)],
                             "time_s": np.arange(na + nb) / fps})

    segments = make_segments(labeled, timeline, bin_s=1.0)

    assert list(segments.columns) == SEGMENT_COLUMNS
    # freeze 4-7 s crosses the part boundary at 5.5 s, so it is two segments.
    assert list(zip(segments["label"], segments["part"])) == [(CALM, 1), (FREEZE, 1), (FREEZE, 2), (CALM, 2)]
    assert segments["start_s"].tolist() == pytest.approx([0.0, 4.0, 5.5, 7.0])
    assert segments["end_s"].tolist() == pytest.approx([4.0, 5.5, 7.0, 10.0])
    assert segments["duration_s"].sum() == pytest.approx(10.0)
    assert (segments["start_s"].iloc[1:].to_numpy() == segments["end_s"].iloc[:-1].to_numpy()).all()  # no gaps
    # Frame numbers are inside each part's own file: part 2 restarts at 0.
    assert segments[["start_frame", "end_frame"]].values.tolist() == [[0, 119], [120, 164], [0, 44], [45, 134]]
    assert segments["mean_confidence"].tolist() == pytest.approx([0.8] * 4)

    summary = summarize(segments).iloc[0]
    assert summary["duration_s"] == pytest.approx(10.0)
    assert summary[f"{CALM}_s"] == pytest.approx(7.0) and summary[f"{FREEZE}_pct"] == pytest.approx(30.0)
    assert sum(summary[f"{name}_pct"] for name in LABELS) == pytest.approx(100.0)


# --- swim split ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["gmm", "hmm"])
def test_swim_model_separates_smooth_from_jerky_bins(method):
    params = {**PARAMS, "swim_split": method}
    bins, truth = smooth_and_jerky(60)
    swim = rule_labels(bins, params, 1.0)[0] == SWIM
    assert swim.all()

    model = json.loads(json.dumps(fit_swim_model([(bins, swim)], params)))  # survives the JSON file
    labeled = label_bins(bins, params, model, bin_s=1.0)

    assert (labeled["label"].to_numpy() == truth).mean() > 0.95
    assert labeled["confidence"].min() >= 0.5  # posterior of the chosen state
    # A new subject is labeled with the saved model, without refitting.
    other, other_truth = smooth_and_jerky(20, seed=1)
    assert (label_bins(other, params, model, 1.0)["label"].to_numpy() == other_truth).mean() > 0.95


def test_slow_swimming_is_controlled_and_stays_out_of_the_swim_model():
    bins, _ = smooth_and_jerky(20)
    slow = make_bins([(3, {**JERKY_BIN, "speed_median_bl_s": 0.2, "speed_mean_bl_s": 0.25})])  # not frozen
    model = fit_swim_model([(bins, np.ones(len(bins), bool))], PARAMS)

    labeled = label_bins(slow, PARAMS, model, bin_s=1.0)

    assert list(labeled["label"]) == [CALM] * 3  # too slow to measure turning, whatever its jerk
    assert labeled["confidence"].tolist() == pytest.approx([0.5] * 3)  # half the 0.5 BL/s gate


def test_missing_turn_values_do_not_break_the_swim_split():
    bins, truth = smooth_and_jerky(30)
    bins.loc[::7, ["turn_rate_var_deg2_s2", "meander_deg_per_bl"]] = np.nan  # e.g. too slow for a heading
    model = fit_swim_model([(bins, np.ones(len(bins), bool))], PARAMS)
    p = erratic_probability(bins, np.ones(len(bins), bool), model)
    assert np.isfinite(p).all() and ((p >= 0.5) == (truth == ERRATIC)).mean() > 0.9


# --- settings ------------------------------------------------------------------------------


def settings_in(tmp_path, calibrated=None):
    if calibrated is not None:
        (tmp_path / "calibration").mkdir(parents=True, exist_ok=True)
        (tmp_path / "calibration" / "calibrated.yaml").write_text(yaml.safe_dump(calibrated))
    return load_settings(environ={"FISH_OUTPUT_DIR": str(tmp_path)})


def test_calibrated_values_override_the_config(tmp_path):
    params, used = labeling_params(settings_in(tmp_path))
    assert used is None and params["surface_breach_fraction"] == PARAMS["surface_breach_fraction"]

    params, used = labeling_params(settings_in(tmp_path, {"labeling": {"surface_breach_fraction": 0.9}}))
    assert used == tmp_path / "calibration" / "calibrated.yaml"
    assert params["surface_breach_fraction"] == 0.9 and params["freeze_min_s"] == PARAMS["freeze_min_s"]
    bins = make_bins([(3, {"nose_up_surface_fraction": 0.5})])
    assert list(rule_labels(bins, params, 1.0)[0]) == [SWIM] * 3  # 50% at the surface is no longer enough


def test_unknown_swim_split_or_state_name_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="swim_split"):
        labeling_params(settings_in(tmp_path, {"labeling": {"swim_split": "kmeans"}}))
    with pytest.raises(ConfigError, match="freezing"):
        labeling_params(settings_in(tmp_path, {"labeling": {"min_bout_s": {"freezing": 2}}}))


# --- end to end: scripted video -> scene -> track -> features -> label ---------------------------

FPS, SIZE, PHASE_S = 30.0, (320, 240), 10.0
WATERLINE = round(0.35 * SIZE[1])
FISH_LEN = 0.12 * SIZE[0]
SCRIPT = [CALM, ERRATIC, FREEZE, SURFACE, LORR]  # one PHASE_S phase each, in this order


def scripted_path():
    """Calm circling, jerky darting, stop, rise nose-up to the surface and stay, tilted and still."""
    rng = np.random.default_rng(3)
    waypoints = np.column_stack([rng.uniform(100, 220, 41), rng.uniform(112, 162, 41)])  # a new target every 0.25 s
    # Nose-up at 35 degrees (facing right: a negative drawing angle points the head up on screen),
    # placed so the highest point of the tilted body is 1 px under the waterline.
    a, b, up = FISH_LEN / 2, FISH_LEN / 7, math.radians(35)
    surface_y = WATERLINE + math.hypot(a * math.sin(up), b * math.cos(up)) + 1

    def path(t):
        phase, u = divmod(t, PHASE_S)
        if phase == 0:  # constant speed (2 BL/s) around a circle: smooth, steady turning
            angle = 2 * FISH_LEN / 30 * u
            return 160 + 30 * math.cos(angle), 140 + 30 * math.sin(angle), 0.0
        if phase == 1:  # straight dashes between random points, new speed and direction every 0.25 s
            k, f = divmod(u / 0.25, 1)
            a, b = waypoints[int(k)], waypoints[int(k) + 1]
            return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), 0.0
        if phase == 2:
            return 160.0, 150.0, 0.0
        if phase == 3:  # rise to the surface in 1 s, then stay there, head up
            return 130.0, 150 + min(u, 1.0) * (surface_y - 150), -35.0 * min(u, 1.0)
        return 190.0, 140.0, 70.0  # steep and still: lost its righting

    return path


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A project whose only subject is the scripted video, labeled once through the CLI."""
    root = tmp_path_factory.mktemp("label")
    make_video(root / "F_0042.avi", fps=FPS, size=SIZE, duration_s=len(SCRIPT) * PHASE_S,
               waterline_y=WATERLINE, fish_path=scripted_path(), reflection=False)
    catalog = root / "outputs" / "catalog"
    catalog.mkdir(parents=True)
    pd.DataFrame({"subject_id": ["0042"], "video_status": ["matched"],
                  "video_paths": [str(root / "F_0042.avi")]}).to_csv(catalog / "trials.csv", index=False)
    (root / ".env").write_text("FISH_OUTPUT_DIR=outputs\nFISH_WORKERS=1\n")
    env = ["--env-file", str(root / ".env")]
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        for name in ("FISH_VIDEO_DIR", "FISH_DB_PATH", "FISH_REFERENCE_PDF", "FISH_OUTPUT_DIR", "FISH_CONFIG"):
            patch.delenv(name, raising=False)
        assert main([*env, "track"]) == 0 and main([*env, "features"]) == 0
        assert main([*env, "label"]) == 0
    return root, env


def test_scripted_video_recovers_every_state(project):
    root, _ = project
    labels = pd.read_csv(root / "outputs" / "labels" / "0042_bins.csv", dtype={"subject_id": str})
    assert len(labels) == len(SCRIPT) * PHASE_S

    for i, state in enumerate(SCRIPT):
        core = labels["label"].iloc[int(i * PHASE_S) + 1:int((i + 1) * PHASE_S) - 1]  # 1 s off each switch
        assert (core == state).mean() >= 0.75, f"{state}: {core.value_counts().to_dict()}"

    segments = pd.read_csv(root / "outputs" / "labels" / "segments.csv", dtype={"subject_id": str})
    assert segments["duration_s"].sum() == pytest.approx(len(SCRIPT) * PHASE_S, abs=0.05)
    summary = pd.read_csv(root / "outputs" / "labels" / "summary.csv", dtype={"subject_id": str})
    assert summary["subject_id"].tolist() == ["0042"]
    assert summary[[f"{name}_pct" for name in LABELS]].sum(axis=1).iloc[0] == pytest.approx(100.0)


def test_label_is_cached_and_calibrated_values_are_used(project, capsys):
    root, env = project
    capsys.readouterr()
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        assert main([*env, "label", "--subjects", "F_0042"]) == 0
        out = capsys.readouterr().out
        assert "reused" in out and "1 done (1 cached)" in out

        # Calibration says a surface breach needs >100% of frames: never. The model is refit
        # (other swim bins) and the subject relabeled without --force.
        (root / "outputs" / "calibration").mkdir()
        (root / "outputs" / "calibration" / "calibrated.yaml").write_text("labeling:\n  surface_breach_fraction: 1.01\n")
        assert main([*env, "label", "--subjects", "42"]) == 0
        out = capsys.readouterr().out
    assert "calibrated values" in out and "fitted now" in out and "1 done (0 cached)" in out
    labels = pd.read_csv(root / "outputs" / "labels" / "0042_bins.csv")
    assert SURFACE not in set(labels["label"])

    # New features (e.g. `features --force` after a settings change) also refit the model.
    bins_file = root / "outputs" / "features" / "0042_bins.csv.gz"
    later = (root / "outputs" / "labels" / "swim_model.json").stat().st_mtime + 10
    os.utime(bins_file, (later, later))
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        assert main([*env, "label"]) == 0
    assert "fitted now" in capsys.readouterr().out


def test_human_relabels_in_overrides_csv_reach_the_labels_and_segments(project):
    root, env = project
    labels = root / "outputs" / "labels"
    before = pd.read_csv(labels / "0042_bins.csv")["label"].tolist()
    pd.DataFrame({"subject_id": ["0042"], "start_s": [2.0], "end_s": [5.0], "label": [LORR]}).to_csv(
        labels / "overrides.csv", index=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        assert main([*env, "label"]) == 0  # no --force: the relabels are newer than the outputs

    bins = pd.read_csv(labels / "0042_bins.csv")
    assert bins["label"].iloc[2:5].tolist() == [LORR] * 3 and bins["label_source"].iloc[2:5].eq("human").all()
    assert bins["auto_label"].tolist() == before and bins["label_source"].eq("human").sum() == 3
    segments = pd.read_csv(labels / "segments.csv")
    human = segments[(segments["label"] == LORR) & (segments["end_s"] < 6)]
    assert human[["start_s", "end_s"]].values.tolist() == [pytest.approx([2.0, 5.0], abs=0.05)]
