"""Stage 2, dose within compound (plan U15, T4.1; PRD §6.5, FR-7): one model per compound, FR-9 variant, caveats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from dcs.cli import main
from dcs.config import load_settings
from dcs.gold_rules import compound_label
from dcs.stage2 import run_stage2, stage_key
from dcs.trainset import load_table

FAST = ["majority", "date_only", "logreg"]


@pytest.fixture(scope="module")
def training() -> dict[str, Any]:
    return {**load_settings().training, "folds": 3, "repeats": 2, "min_class_size": 2}


def stage2(table: pd.DataFrame, described: dict[str, Any], training: dict[str, Any]) -> tuple[list[Any], list[str]]:
    return run_stage2(table, described, training, FAST, "cpu", {}, lambda message: None)


@pytest.fixture(scope="module")
def dose_models(tiny_table: Path, training: dict[str, Any]) -> list[Any]:
    """Computed once: the default stage-2 result on the tiny table."""
    table, described = load_table(tiny_table)
    return stage2(table, described, training)[0]


def test_one_dose_model_per_compound_and_never_the_vehicle(dose_models: list[Any]) -> None:
    models = dose_models
    assert [m.compound for m in models] == ["COMPOUND_A", "COMPOUND_B"]
    for model in models:
        labels = set(model.raw.trainset.y)
        assert len(labels) == 2 and all(label.startswith(model.compound + " @ ") for label in labels)


def test_a_compound_left_with_one_dose_gets_no_model(tiny_table: Path, training: dict[str, Any]) -> None:
    table, described = load_table(tiny_table)
    b = table["compound"].map(compound_label) == "COMPOUND_B"
    one_dose = table[~b | (table["concentration_mM"] == table.loc[b, "concentration_mM"].iloc[0])]
    models, _ = stage2(one_dose, described, training)
    assert [m.compound for m in models] == ["COMPOUND_A"]


def test_every_dose_model_shows_date_only_and_the_permuted_score(dose_models: list[Any]) -> None:
    for model in dose_models:
        summary = model.raw.evaluation.summary
        assert ("date_only", "B") in summary.index and ("logreg", "B_permuted") in summary.index


def test_fr9_variant_runs_on_the_same_folds(dose_models: list[Any]) -> None:
    for model in dose_models:
        assert model.normalized is not None
        assert model.normalized.folds is model.raw.folds
        assert set(model.normalized.trainset.y) == set(model.raw.trainset.y)


def test_without_the_vehicle_the_fr9_variant_is_a_note(tiny_table: Path, training: dict[str, Any]) -> None:
    table, described = load_table(tiny_table)
    models, _ = stage2(table, described, {**training, "vehicle_compound": "NOT_HERE"})
    assert all(m.normalized is None and "vehicle_compound" in m.note for m in models)


def test_no_dose_class_left_is_a_note_not_a_stop(tiny_table: Path, training: dict[str, Any]) -> None:
    table, described = load_table(tiny_table)
    models, notes = stage2(table, described, {**training, "min_class_size": 6})
    assert models == [] and notes and "dose" in notes[0]


def test_stage_key_names_the_compound() -> None:
    assert stage_key("COMPOUND_A") == "dose COMPOUND_A"


def test_train_writes_one_dose_section_per_compound(tiny_table: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "fast.yaml"
    config.write_text(yaml.safe_dump({"training": {"folds": 3, "repeats": 1, "min_class_size": 2}}), encoding="utf-8")
    monkeypatch.setenv("DCS_TABLE", str(tiny_table))
    monkeypatch.setenv("DCS_OUTPUT_DIR", str(tmp_path / "out"))
    assert main(["--config", str(config), "train", "--models", "logreg", "--no-ablations"]) == 0
    (run,) = (tmp_path / "out" / "training").iterdir()
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    assert info["stages"] == ["compound", "dose COMPOUND_A", "dose COMPOUND_B"]
    metrics = pd.read_csv(run / "metrics.csv")
    assert {"main", "non-vehicle, vehicle-normalized"} <= set(metrics.loc[metrics["stage"] == "dose COMPOUND_A", "ablation"])
    report = (run / "report.md").read_text(encoding="utf-8")
    assert report.count("**Exploratory (PRD §6.5).**") == 2 and "## Stage: dose COMPOUND_B" in report
    assert (run / "confusion_dose_compound_a.png").is_file()
