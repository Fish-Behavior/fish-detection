"""Shared fixtures for the classifier (`dcs`) tests. Synthetic data only."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from dcs.config import ENV_CONFIG, PATH_VARIABLES, load_settings
from dcs.featurize import featurize, write_outputs
from dcs.gold import read_accepted
from dcs.synthetic import SynthResult, make_gold_dataset
from tests.dcs.synth_helpers import SMALL

DCS_VARIABLES = (*PATH_VARIABLES.values(), ENV_CONFIG)


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every dcs test from an empty directory, with no real .env and no DCS_* variables.

    load_settings falls back to `Path.cwd() / ".env"` and the CLI reads
    `os.environ`, so a developer's own setup must never leak into a test.
    """
    monkeypatch.chdir(tmp_path)
    for name in DCS_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def small(tmp_path_factory: pytest.TempPathFactory) -> SynthResult:
    """One small synthetic gold dataset, written once per session. Tests only read its files;
    `truth` hands out copies and `targets` is read-only, so a test cannot change what others see."""
    return make_gold_dataset(tmp_path_factory.mktemp("small"), SMALL)


@pytest.fixture(scope="session")
def tiny_table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic training table and its schema: 24 fish, two compounds and vehicle, 8 each on 4 dates. Every
    dose class has 4 fish, so the dose stage cannot be built with min_class_size 6."""
    synth = make_gold_dataset(tmp_path_factory.mktemp("tiny"), dataclasses.replace(SMALL, vehicle_per_date=2))
    result = featurize(read_accepted(synth.accepted_dir), synth.workbook_path, load_settings().training)
    return write_outputs(result, tmp_path_factory.mktemp("tiny_table") / "training_table.parquet")[0]
