"""Shared fixtures for the classifier (`dcs`) tests. Synthetic data only."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from dcs.config import ENV_CONFIG, PATH_VARIABLES, load_settings
from dcs.featurize import featurize, write_outputs
from dcs.gold import read_accepted
from dcs.synthetic import SynthResult, make_gold_dataset
from backend.dcs.tests.synth_helpers import SMALL

DCS_VARIABLES = (*PATH_VARIABLES.values(), ENV_CONFIG)


@contextmanager
def _isolated(folder: Path) -> Iterator[None]:
    """Work from an empty folder with no DCS_* variables, then restore both (other test packages use relative paths)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(folder)
        for name in DCS_VARIABLES:
            patch.delenv(name, raising=False)
        yield


@pytest.fixture(scope="module", autouse=True)
def _isolated_module(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """The same isolation for module-scoped fixtures, which are built before any per-test fixture runs."""
    with _isolated(tmp_path_factory.mktemp("cwd")):
        yield


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
def tiny_synth(tmp_path_factory: pytest.TempPathFactory) -> SynthResult:
    """The synthetic gold folder behind `tiny_table` (read only)."""
    return make_gold_dataset(tmp_path_factory.mktemp("tiny"), dataclasses.replace(SMALL, vehicle_per_date=2))


@pytest.fixture(scope="session")
def tiny_table(tiny_synth: SynthResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic training table and its schema: 24 fish, two compounds and vehicle, 8 each on 4 dates. Every
    dose class has 4 fish, so the dose stage cannot be built with min_class_size 6."""
    synth = tiny_synth
    with _isolated(tmp_path_factory.mktemp("cwd")):  # session fixtures are built before the module isolation
        training = load_settings().training
    result = featurize(read_accepted(synth.accepted_dir), synth.workbook_path, training)
    return write_outputs(result, tmp_path_factory.mktemp("tiny_table") / "training_table.parquet")[0]
