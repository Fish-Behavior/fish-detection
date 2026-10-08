"""`dcs train` end to end on synthetic data (plan U12, T2.7; PRD §7.2, §7.4, US-1, NFR-1)."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dcs.cli import main
from dcs.config import load_settings
from dcs.featurize import featurize, write_outputs
from dcs.folds import make_folds
from dcs.gold import read_accepted
from dcs.synthetic import make_gold_dataset
from dcs.trainset import build_trainset, load_table
from tests.dcs.synth_helpers import SMALL

RUN_FILES = {
    "run_info.json", "config_used.yaml", "audit.md", "folds.csv", "metrics.csv", "predictions.csv", "report.md",
    "confusion_compound.png",
}  # fmt: skip


@pytest.fixture(scope="module")
def table_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """24 fish: two compounds and vehicle, 8 each on 4 dates; every dose class has 4 fish, so the dose stage
    cannot be built with min_class_size 6."""
    synth = make_gold_dataset(tmp_path_factory.mktemp("train"), dataclasses.replace(SMALL, vehicle_per_date=2))
    result = featurize(read_accepted(synth.accepted_dir), synth.workbook_path, load_settings().training)
    return write_outputs(result, tmp_path_factory.mktemp("table") / "training_table.parquet")[0]


@pytest.fixture
def setup(table_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    config = tmp_path / "fast.yaml"
    config.write_text(yaml.safe_dump({"training": {"folds": 3, "repeats": 2}}), encoding="utf-8")
    monkeypatch.setenv("DCS_TABLE", str(table_path))
    monkeypatch.setenv("DCS_OUTPUT_DIR", str(tmp_path / "out"))
    return {"config": config, "runs": tmp_path / "out" / "training"}


def train(setup: dict[str, Path], *options: str) -> int:
    return main(["--config", str(setup["config"]), "train", *options])


def only_run(setup: dict[str, Path]) -> Path:
    (run,) = setup["runs"].iterdir()
    return run


def test_train_writes_the_run_folder(setup: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert train(setup, "--models", "majority,logreg") == 0
    run = only_run(setup)
    assert re.fullmatch(r"\d{8}T\d{6}Z-\w+", run.name)
    assert RUN_FILES <= {path.name for path in run.iterdir()}
    out = capsys.readouterr().out
    assert "scheme A repeat 1/2" in out and str(run / "report.md") in out
    assert "dose stage cannot be built" in (run / "report.md").read_text(encoding="utf-8")


def test_run_info_records_what_reproduces_the_run(setup: dict[str, Path]) -> None:
    assert train(setup, "--models", "logreg") == 0
    info = json.loads((only_run(setup) / "run_info.json").read_text(encoding="utf-8"))
    assert {"run_id", "git_commit", "git_dirty", "command", "seed", "folds", "repeats", "stages", "models",
            "skipped_models", "gold_source", "table", "versions", "hardware", "started_utc", "seconds"} <= set(info)  # fmt: skip
    assert {"python", "scikit-learn", "numpy", "pandas", "torch", "cuda"} <= set(info["versions"])
    assert {"system", "machine", "cpus"} <= set(info["hardware"])
    assert info["stages"] == ["compound"] and info["models"] == ["majority", "date_only", "logreg"]


def test_command_line_overrides_land_in_config_used(setup: dict[str, Path]) -> None:
    assert train(setup, "--stage", "compound", "--models", "logreg", "--seed", "3", "--repeats", "1") == 0
    used = yaml.safe_load((only_run(setup) / "config_used.yaml").read_text(encoding="utf-8"))["training"]
    assert (used["stage"], used["models"], used["seed"], used["repeats"], used["folds"]) == ("compound", ["logreg"], 3, 1, 3)


def test_folds_csv_is_the_shared_split(setup: dict[str, Path], table_path: Path) -> None:
    assert train(setup, "--stage", "compound", "--models", "logreg") == 0
    saved = pd.read_csv(only_run(setup) / "folds.csv", dtype={"date": str})
    table, described = load_table(table_path)
    training = {**load_settings().training, "folds": 3, "repeats": 2}
    ts = build_trainset(table, described, training, "compound")
    expected = make_folds(ts.y, ts.groups, ts.ids, training).table.assign(stage="compound")
    pd.testing.assert_frame_equal(saved[expected.columns], expected, check_dtype=False)


def test_metrics_and_predictions_carry_stage_and_scheme(setup: dict[str, Path]) -> None:
    assert train(setup, "--models", "logreg") == 0
    run = only_run(setup)
    metrics = pd.read_csv(run / "metrics.csv")
    assert set(metrics["scheme"]) == {"A", "B", "B_permuted"} and set(metrics["stage"]) == {"compound"}
    predictions = pd.read_csv(run / "predictions.csv")
    assert {"stage", "model", "scheme", "repeat", "fold", "video_id", "label", "predicted"} <= set(predictions.columns)
    assert any(column.startswith("p:") for column in predictions.columns)


def test_mlp_is_skipped_with_a_message_until_it_exists(setup: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert train(setup, "--models", "logreg,mlp") == 0
    assert "mlp" in capsys.readouterr().out
    info = json.loads((only_run(setup) / "run_info.json").read_text(encoding="utf-8"))
    assert info["skipped_models"] == ["mlp"] and "mlp" not in info["models"]


def test_same_seed_gives_identical_metrics(setup: dict[str, Path]) -> None:
    """EC-14 / AC-7 through the command (two runs started within a second get distinct folders)."""
    assert train(setup, "--models", "logreg") == 0
    assert train(setup, "--models", "logreg") == 0
    first, second = sorted(setup["runs"].iterdir())
    pd.testing.assert_frame_equal(pd.read_csv(first / "metrics.csv"), pd.read_csv(second / "metrics.csv"))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--models", "logreg,bogus"], "--models must be"),
        (["--repeats", "0"], "--repeats must be"),
        (["--stage", "dose"], "dose"),  # asked for alone, a stage that cannot be built stops the run
    ],
)
def test_bad_requests_stop_with_a_message_and_write_nothing(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str], options: list[str], message: str
) -> None:
    assert train(setup, *options) == 2
    out = capsys.readouterr().out
    error = out.splitlines()[-1]  # progress lines may come first
    assert error.startswith("Configuration error:") and message in error
    assert not setup["runs"].exists()


def test_missing_table_is_a_configuration_error(setup: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DCS_TABLE", str(tmp_path / "absent.parquet"))
    assert train(setup) == 2

