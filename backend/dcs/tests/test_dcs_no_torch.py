"""EC-16 (plan U13): without torch, the baselines run and the MLP is skipped with a message."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from dcs.cli import main
from dcs.models import torch_available


@pytest.fixture
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide torch: `import torch` fails and its spec cannot be found, as on a machine without it."""
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.mark.usefixtures("no_torch")
def test_torch_hidden_reads_as_not_installed() -> None:
    assert not torch_available()


@pytest.mark.usefixtures("no_torch")
def test_train_without_torch_runs_the_baselines_and_skips_the_mlp(
    tiny_table: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "fast.yaml"
    config.write_text(yaml.safe_dump({"training": {"folds": 3, "repeats": 1, "stage": "compound"}}), encoding="utf-8")
    monkeypatch.setenv("DCS_TABLE", str(tiny_table))
    monkeypatch.setenv("DCS_OUTPUT_DIR", str(tmp_path / "out"))
    assert main(["--config", str(config), "train", "--models", "logreg,mlp", "--device", "cuda"]) == 0
    assert "mlp: skipped, PyTorch is not installed" in capsys.readouterr().out
    (run,) = (tmp_path / "out" / "training").iterdir()
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    assert info["skipped_models"] == ["mlp"] and info["models"] == ["majority", "date_only", "logreg"]
