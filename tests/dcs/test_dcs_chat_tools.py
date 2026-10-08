"""Research query tools for the chat (plan U19): every number a tool returns can be checked by hand."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dcs.chat_tools import TOOLS, ResearchData, call_tool, tool_schemas
from dcs.config import load_settings
from dcs.gold_rules import compound_label
from dcs.synthetic import SynthResult
from dcs.train import run_training
from dcs.trainset import load_table


@pytest.fixture(scope="module")
def run_folder(tiny_table: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One small `dcs train` run with ablations, built from an empty .env so no real setting leaks in."""
    base = tmp_path_factory.mktemp("chat_run")
    (base / "empty.env").write_text("", encoding="utf-8")
    settings = load_settings(env_file=base / "empty.env", environ={"DCS_TABLE": str(tiny_table), "DCS_OUTPUT_DIR": str(base / "out")})
    training = {**settings.training, "folds": 3, "repeats": 2, "stage": "compound", "models": ("logreg",)}
    return run_training(settings, training, "test", device="cpu", log=lambda message: None)


@pytest.fixture(scope="module")
def data(tiny_table: Path, tiny_synth: SynthResult, run_folder: Path) -> ResearchData:
    table, described = load_table(tiny_table)
    return ResearchData(table, described, dict(load_settings().training), tiny_synth.accepted_dir, run_folder)


def call(data: ResearchData, name: str, **arguments: Any) -> dict[str, Any]:
    result = call_tool(data, name, json.dumps(arguments))
    json.dumps(result)  # every answer must be JSON-safe for the model and the API
    return result


def test_every_tool_has_a_valid_schema() -> None:
    schemas = tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert len(names) == len(set(names)) == len(TOOLS)
    for schema in schemas:
        parameters = schema["function"]["parameters"]
        assert schema["type"] == "function" and parameters["type"] == "object"
        assert set(parameters.get("required", [])) <= set(parameters["properties"])
        assert schema["function"]["description"]


def test_compounds_lists_each_compound_and_marks_the_vehicle(data: ResearchData) -> None:
    result = call(data, "list_compounds")
    names = {row["compound"]: row for row in result["compounds"]}
    assert set(names) == {"COMPOUND_A", "COMPOUND_B", "VEHICLE"} and result["vehicle"] == "VEHICLE"
    assert names["COMPOUND_A"]["fish"] == 8 and len(names["COMPOUND_A"]["doses"]) == 2


def test_feature_search_explains_names(data: ResearchData) -> None:
    result = call(data, "find_features", search="freez")
    names = {row["name"]: row["meaning"] for row in result["features"]}
    assert "state_freezing_drift_share" in names and "Freezing/Drift" in names["state_freezing_drift_share"]
    assert all("freez" in (n + m).lower() for n, m in names.items())


def test_compare_gives_the_numbers_a_researcher_would_compute(data: ResearchData) -> None:
    result = call(data, "compare_to_vehicle", feature="velocity_mean", compound="compound_a")
    table = data.table
    compound = table["compound"].map(compound_label)
    group = table.loc[compound == "COMPOUND_A", "velocity_mean"]
    dates = set(table.loc[compound == "COMPOUND_A", "date"])
    control = table.loc[(compound == "VEHICLE") & table["date"].isin(dates), "velocity_mean"]
    assert result["group"]["n"] == 8 and result["group"]["mean"] == pytest.approx(group.mean())
    assert result["control"]["n"] == len(control) and result["control"]["which"].startswith("vehicle fish on the same dates")
    pooled = math.sqrt(((len(group) - 1) * group.var() + (len(control) - 1) * control.var()) / (len(group) + len(control) - 2))
    g = (group.mean() - control.mean()) / pooled * (1 - 3 / (4 * (len(group) + len(control)) - 9))
    assert result["hedges_g"] == pytest.approx(g)
    assert result["p_value"] == pytest.approx(stats.mannwhitneyu(group, control).pvalue)
    assert result["meaning"] and result["caveats"]


def test_compare_falls_back_to_all_vehicle_fish_and_says_so(data: ResearchData) -> None:
    """One dose of COMPOUND_A sits on two dates; with no vehicle fish there, the other dates' vehicle fish are used."""
    compound = data.table["compound"].map(compound_label)
    dose = str(data.table.loc[compound == "COMPOUND_A", "concentration_mM"].iloc[0])
    dates = set(data.table.loc[(compound == "COMPOUND_A") & (data.table["concentration_mM"].astype(str) == dose), "date"])
    thin = data.table[~((compound == "VEHICLE") & data.table["date"].isin(dates))]
    sparse = ResearchData(thin, data.described, data.training, data.gold_dir, data.run)
    result = call(sparse, "compare_to_vehicle", feature="velocity_mean", compound="COMPOUND_A", dose=dose)
    assert result["control"]["which"].startswith("all vehicle fish")
    assert any("same dates" in caveat for caveat in result["caveats"])


def test_bad_arguments_come_back_as_an_error_with_the_choices(data: ResearchData) -> None:
    assert "COMPOUND_A" in call(data, "compare_to_vehicle", feature="velocity_mean", compound="nope")["error"]
    assert "find_features" in call(data, "compare_to_vehicle", feature="nope", compound="COMPOUND_A")["error"]
    assert "unknown tool" in call(data, "no_such_tool")["error"]
    assert "error" in call_tool(data, "list_compounds", "{not json")


def test_top_differences_rank_by_effect_with_fdr(data: ResearchData) -> None:
    result = call(data, "top_differences", compound="COMPOUND_A", n=5)
    rows = result["features"]
    assert len(rows) == 5
    effects = [abs(row["hedges_g"]) for row in rows]
    assert effects == sorted(effects, reverse=True)
    assert all(row["q_value"] >= row["p_value"] - 1e-12 for row in rows)
    assert result["tested"] > 5 and "Benjamini-Hochberg" in result["note"]


def test_fish_profile_has_labels_percentiles_and_out_of_fold_predictions(data: ResearchData) -> None:
    video_id = str(data.table["video_id"].iloc[0])
    result = call(data, "fish_profile", video_id=video_id)
    assert result["video_id"] == video_id and result["compound"]
    assert all(0 <= row["percentile_in_compound"] <= 100 for row in result["features"])
    assert result["predictions"]["model"] and result["predictions"]["repeats"] >= 1


def test_timeline_bins_cover_the_recording(data: ResearchData) -> None:
    video_id = str(data.table["video_id"].iloc[0])
    result = call(data, "fish_timeline", video_id=video_id, bin_s=5)
    assert len(result["bins"]) == math.ceil(result["duration_s"] / 5)
    for row in result["bins"]:
        if row["known_s"] > 0:
            assert sum(row["state_shares"].values()) == pytest.approx(1)
    assert result["bouts"] and {"start_s", "end_s", "state"} <= set(result["bouts"][0])
    assert "first_bout_s" in result


def test_unknown_fish_is_an_error(data: ResearchData) -> None:
    assert "F_9999" in call(data, "fish_timeline", video_id="F_9999")["error"]


def test_model_results_read_the_run_folder(data: ResearchData) -> None:
    result = call(data, "model_results")
    rows = {row["model"]: row for row in result["models"]}
    assert {"majority", "date_only", "logreg"} <= set(rows)
    assert rows["logreg"]["verdict"] and rows["logreg"]["reason"] and "scheme_A" in rows["logreg"]
    assert result["saved_model"]["model"] == "logreg" and result["run_id"] == data.run.name


def test_class_scores_and_confusions(data: ResearchData) -> None:
    result = call(data, "class_scores")
    assert {row["label"] for row in result["classes"]} == {"COMPOUND_A", "COMPOUND_B", "VEHICLE"}
    assert isinstance(result["top_confusions"], list)


def test_ablations_from_the_run(data: ResearchData) -> None:
    result = call(data, "ablation_results")
    assert "use_ntt=false" in {row["ablation"] for row in result["ablations"]}


def test_audit_facts_and_notes(data: ResearchData) -> None:
    result = call(data, "audit_facts")
    assert result["facts"]["fish in the table"] == 24 and isinstance(result["notes"], list)


def test_without_a_run_the_model_tools_say_how_to_get_one(data: ResearchData) -> None:
    bare = ResearchData(data.table, data.described, data.training, data.gold_dir, None)
    assert "dcs train" in call(bare, "model_results")["error"]


def test_research_data_from_settings_finds_the_latest_run(tiny_table: Path, run_folder: Path, tmp_path: Path) -> None:
    (tmp_path / "empty.env").write_text("", encoding="utf-8")
    environ = {"DCS_TABLE": str(tiny_table), "DCS_OUTPUT_DIR": str(run_folder.parent.parent)}
    loaded = ResearchData.from_settings(load_settings(env_file=tmp_path / "empty.env", environ=environ))
    assert loaded.run == run_folder and len(loaded.table) == 24
    assert np.isfinite(loaded.table["velocity_mean"]).all()
