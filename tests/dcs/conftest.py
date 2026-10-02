"""Shared fixtures for the classifier (`dcs`) tests. Synthetic data only."""

from __future__ import annotations

from pathlib import Path

import pytest

from dcs.config import ENV_CONFIG, PATH_VARIABLES

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
