"""Tests for fishbehavior.calibrate and fishbehavior.priors on synthetic bins, references and tables."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from fishbehavior.calibrate import (
    candidate_params,
    confusion,
    evaluate,
    f1_scores,
    fold_split,
    group_agreement,
    macro_f1,
    per_second,
    sample_candidates,
    scores,
)
from fishbehavior.cli import main
from fishbehavior.config import load_settings
from fishbehavior.features import BIN_COLUMNS
from fishbehavior.labeling import CALM, ERRATIC, FREEZE, LABELS, STATES, UNTRACKED, fit_swim_model, label_bins
from fishbehavior.labeling import labeling_params, moving
from fishbehavior.priors import direction_checks, expectations_template, residual_flags

PARAMS = load_settings(environ={}).params["labeling"]
# The "true" thresholds the synthetic reference is labeled with.
TRUTH = {"freeze_speed_bl_s": 0.2, "surface_breach_fraction": 0.4, "lorr_tilt_fraction": 0.6}
SEARCH = {"freeze_speed_bl_s": {"low": 0.03, "high": 0.4}, "surface_breach_fraction": {"low": 0.1, "high": 0.7},
          "lorr_tilt_fraction": {"low": 0.2, "high": 0.9}}


def synthetic_bins(subject_id, n_bins, rng):
    """Bins in runs of 3-8 s: swimming, slow (freeze candidate), tilted, or nose-up at the surface.

    Each run's key value is drawn uniformly around the thresholds, so every threshold matters.
    """
    rows = []
    while len(rows) < n_bins:
        kind, length = rng.choice(["swim", "slow", "tilt", "surface"]), int(rng.integers(3, 9))
        run = {"tracked_fraction": 1.0, "nose_up_surface_fraction": 0.0, "tilt_fraction": 0.0,
               "speed_median_bl_s": rng.uniform(0.6, 3.0)}
        if kind == "slow":
            run["speed_median_bl_s"] = rng.uniform(0.0, 0.4)
        elif kind == "tilt":
            run.update(tilt_fraction=rng.uniform(0.2, 1.0), speed_median_bl_s=rng.uniform(0.0, 0.4))
        elif kind == "surface":
            run["nose_up_surface_fraction"] = rng.uniform(0.0, 0.8)
        for _ in range(length):
            jerky = rng.random() < 0.5
            rows.append({**run, "speed_mean_bl_s": run["speed_median_bl_s"],
                         "speed_cv": rng.uniform(1.0, 1.6) if jerky else rng.uniform(0.1, 0.5),
                         "jerk_abs_mean_bl_s3": 1500.0 if jerky else 40.0,
                         "turn_rate_var_deg2_s2": 5e6 if jerky else 1e3, "meander_deg_per_bl": 600.0 if jerky else 20.0})
    table = pd.DataFrame(rows[:n_bins]).reindex(columns=BIN_COLUMNS)
    table["subject_id"], table["bin"] = subject_id, range(n_bins)
    table["t_start_s"] = table["bin"].astype(float)
    return table


@pytest.fixture(scope="module")
def synthetic_study():
    """12 subjects: bins, swim model, and reference timelines labeled with TRUTH (two groups)."""
    rng = np.random.default_rng(3)
    bins = [synthetic_bins(f"{9000 + i:04d}", 400, rng) for i in range(12)]
    truth = {**PARAMS, **TRUTH}
    model = fit_swim_model([(b, moving(b, truth)) for b in bins], truth)
    references = [label_bins(b, truth, model, 1.0)["label"].to_numpy(object) for b in bins]
    groups = np.array(["G1" if i % 2 else "G2" for i in range(12)], dtype=object)
    return bins, model, references, groups


def reference_means(references, groups):
    """group_means.csv of the synthetic reference."""
    rows = []
    for group in np.unique(groups):
        refs = [r for r, g in zip(references, groups) if g == group]
        rows.append({"group": group, "n_subjects": len(refs),
                     **{s: float(np.mean([(r == s).sum() for r in refs])) for s in STATES}})
    return pd.DataFrame(rows)


# --- score math ---------------------------------------------------------------------------


def test_confusion_and_f1_math():
    reference = np.array([CALM, CALM, ERRATIC, FREEZE, "no_data"], dtype=object)
    ours = np.array([CALM, ERRATIC, ERRATIC, UNTRACKED, CALM], dtype=object)

    matrix = confusion(reference, ours)

    assert matrix.shape == (len(STATES), len(LABELS)) and matrix.sum() == 4  # the no_data second is left out
    assert matrix[STATES.index(FREEZE), LABELS.index(UNTRACKED)] == 1
    f1 = f1_scores(matrix)
    assert f1[STATES.index(CALM)] == pytest.approx(2 / 3)  # tp 1, fn 1
    assert f1[STATES.index(ERRATIC)] == pytest.approx(2 / 3)  # tp 1, fp 1
    assert f1[STATES.index(FREEZE)] == 0
    assert np.isnan(f1[STATES.index("lorr")])  # absent on both sides: not averaged
    assert macro_f1(matrix) == pytest.approx(4 / 9)


def test_per_second_uses_the_bin_covering_each_second():
    labeled = pd.DataFrame({"label": [CALM, ERRATIC, FREEZE]})  # 0.5 s bins -> 1.5 s of video
    assert list(per_second(labeled, 0.5, 3)) == [ERRATIC, "", ""]  # second 0 middle = 0.5 s -> bin 1
    assert list(per_second(labeled, 1.0, 4)) == [CALM, ERRATIC, FREEZE, ""]


def test_group_agreement_is_one_for_the_same_budget_and_lower_otherwise():
    ref = pd.DataFrame([{"group": "G", "n_subjects": 1, **{s: 0.0 for s in STATES}, CALM: 60.0, FREEZE: 40.0}])
    same = np.zeros((2, len(LABELS)))
    same[:, LABELS.index(CALM)], same[:, LABELS.index(FREEZE)] = 30, 20
    assert group_agreement(same, np.array(["G", "G"]), ref) == pytest.approx(1.0)
    shifted = same.copy()
    shifted[:, LABELS.index(FREEZE)], shifted[:, LABELS.index(UNTRACKED)] = 10, 10
    assert group_agreement(shifted, np.array(["G", "G"]), ref) == pytest.approx(0.8)  # 20% of time moved


def test_folds_never_split_a_subject():
    folds = fold_split(23, 5, seed=0)
    together = np.concatenate(folds)
    assert len(folds) == 5 and sorted(together) == list(range(23))  # each subject in exactly one test fold
    assert {len(f) for f in folds} <= {4, 5}


def test_candidates_are_reproducible_and_min_bout_applies_to_every_state():
    search = {"freeze_min_s": [1, 3], "min_bout_s": [2], "lorr_tilt_fraction": {"low": 0.2, "high": 0.4}}
    first, again = sample_candidates(search, 20, seed=1), sample_candidates(search, 20, seed=1)
    assert first == again and first[0] == {} and len(first) == 20
    assert all(0.2 <= c["lorr_tilt_fraction"] <= 0.4 and c["freeze_min_s"] in (1, 3) for c in first[1:])
    params = candidate_params(PARAMS, first[1])
    assert all(params["min_bout_s"][s] == 2 for s in STATES) and params["min_bout_s"][UNTRACKED] == 0


# --- the search ---------------------------------------------------------------------------


def test_search_recovers_the_true_thresholds(synthetic_study):
    bins, model, references, groups = synthetic_study
    candidates = sample_candidates(SEARCH, 300, seed=0)

    ev = evaluate(candidates, PARAMS, bins, references, model, 1.0)
    score, f1, _ = scores(ev, np.arange(len(bins)), groups, reference_means(references, groups), 0.5)
    best = candidates[int(np.argmax(score))]

    assert f1.max() > 0.9 and f1.max() > f1[0]  # clearly better than the (wrong) default thresholds
    for key, true_value in TRUTH.items():
        width = SEARCH[key]["high"] - SEARCH[key]["low"]
        assert abs(best[key] - true_value) <= 0.15 * width, (key, best[key])


def test_calibrate_writes_calibrated_yaml_that_label_uses(synthetic_study, tmp_path, monkeypatch):
    bins, model, references, groups = synthetic_study
    out = tmp_path / "out"
    for folder in ("features", "reference", "labels"):
        (out / folder).mkdir(parents=True)
    for table in bins:
        table.to_csv(out / "features" / f"{table['subject_id'].iloc[0]}_bins.csv.gz", index=False)
    timelines = pd.concat([pd.DataFrame({"subject_id": b["subject_id"].iloc[0], "group": g,
                                         "second": range(len(r)), "label": r})
                           for b, r, g in zip(bins, references, groups)])
    timelines = pd.concat([timelines, timelines.head(400).assign(subject_id="9999")])  # a placeholder row: no video
    timelines.to_csv(out / "reference" / "timelines.csv", index=False)
    reference_means(references, groups).to_csv(out / "reference" / "group_means.csv", index=False)
    (out / "labels" / "swim_model.json").write_text(json.dumps(model), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"FISH_OUTPUT_DIR={out}\n", encoding="utf-8")
    search = {**{k: None for k in load_settings(environ={}).params["calibration"]["search"]}, **SEARCH,
              "min_bout_s": [1, 2]}  # null = untuned; min_bout_s is written per state
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"calibration": {"n_trials": 60, "folds": 3,
                                                                          "search": search}}), encoding="utf-8")
    cli = ["--env-file", str(tmp_path / ".env"), "--config", str(tmp_path / "config.yaml")]

    assert main([*cli, "calibrate"]) == 0

    written = yaml.safe_load((out / "calibration" / "calibrated.yaml").read_text(encoding="utf-8"))["labeling"]
    assert set(written) == {*SEARCH, "min_bout_s"}  # only the tuned keys
    assert set(written["min_bout_s"]) == set(STATES)  # one value per state, untracked left as configured
    report = (out / "calibration" / "calibration_report.md").read_text(encoding="utf-8")
    assert "Subjects used (bins + reference timeline): **12**" in report and "Warning" in report  # 12 < 30
    settings = load_settings(env_file=tmp_path / ".env", config_file=tmp_path / "config.yaml")
    params, used = labeling_params(settings)  # what `label` runs with
    assert used is not None and all(params[k] == v for k, v in written.items() if k != "min_bout_s")
    assert params["min_bout_s"] == {**written["min_bout_s"], UNTRACKED: 0}


# --- priors -------------------------------------------------------------------------------


def test_residual_flags_find_the_subject_off_the_trend():
    n = 40
    table = pd.DataFrame({"subject_id": [f"{i:04d}" for i in range(n)], "tdm_full": np.arange(n, dtype=float),
                          "distance_bl": np.arange(n, dtype=float), "freeze_drift_s": np.nan,
                          "velocity_full": np.nan, "surface_breach_s": np.nan, "time_top_s": np.nan})
    table.loc[5, "distance_bl"] = 1000.0  # moved a lot in the video, little in the NTT

    flags = residual_flags(table, z_limit=2)

    assert list(flags["subject_id"]) == ["0005"] and flags["video"].iloc[0] == "distance_bl"


def test_expectations_template_links_groups_and_directions_are_checked():
    trials = pd.DataFrame({"subject_id": ["0001", "0002", "0003"], "compound": ["Veh", "CPD_A", "CPD_A"],
                           "concentration_raw": ["1%", "0.03", "0.03"]})
    timelines = pd.DataFrame({"subject_id": ["0002", "0003", "9001"], "group": ["A 30", "A 30", "VEH"]})
    ref_means = pd.DataFrame([{"group": "VEH", "n_subjects": 1, **{s: 10.0 for s in STATES}},
                              {"group": "A 30", "n_subjects": 2, **{s: 10.0 for s in STATES}, "lorr": 300.0}])

    groups = yaml.safe_load(expectations_template(ref_means, timelines, trials))["groups"]

    assert groups["A 30"] == {"compound": "CPD_A", "concentration": "0.03"}
    assert groups["VEH"] == {"compound": "", "concentration": ""}  # only a placeholder subject: fill by hand
    video = pd.DataFrame([{"group": "A 30", "n_subjects": 2, **{s: 5.0 for s in STATES}}])
    checks = direction_checks([{"group": "A 30", "state": "lorr", "relation": "greater_than", "than": "VEH"}],
                              ref_means, video)
    assert checks[["reference", "video"]].values.tolist() == [[True, None]]  # VEH not linked
