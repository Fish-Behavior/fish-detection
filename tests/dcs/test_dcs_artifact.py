"""Model artifact (plan U16, T5.1; PRD FR-6, §6.9, §7.4; D-006, EC-18): save, reload, reproduce, version warning."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dcs.artifact import MODEL_DIR, REFERENCE_FILE, load_model
from dcs.cli import main
from dcs.config import ConfigError
from dcs.trainset import load_table

RELOAD = """
import sys
import pandas as pd
from dcs.artifact import load_model
table = pd.read_parquet(sys.argv[2])
load_model(sys.argv[1]).predict(table).to_csv(sys.argv[3], index=False)
"""


def trained(tiny_table: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models: str, extra: dict | None = None) -> Path:
    config = tmp_path / "fast.yaml"
    config.write_text(yaml.safe_dump({"training": {"folds": 3, "repeats": 1, **(extra or {})}}), encoding="utf-8")
    monkeypatch.setenv("DCS_TABLE", str(tiny_table))
    monkeypatch.setenv("DCS_OUTPUT_DIR", str(tmp_path / "out"))
    assert main(["--config", str(config), "train", "--stage", "compound", "--models", models, "--no-ablations", "--device", "cpu"]) == 0
    (run,) = (tmp_path / "out" / "training").iterdir()
    return run


def fresh_process_predictions(run: Path, table: Path, out: Path) -> pd.DataFrame:
    """EC-18: a new Python process that has never seen the training objects."""
    subprocess.run([sys.executable, "-c", RELOAD, str(run), str(table), str(out)], check=True, capture_output=True)
    return pd.read_csv(out)


def test_model_folder_holds_what_prd_7_4_lists(tiny_table: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = trained(tiny_table, tmp_path, monkeypatch, "logreg")
    files = {path.name for path in (run / MODEL_DIR).iterdir()}
    assert files == {"preprocess.json", "classes.json", "sklearn.joblib", "model_info.json", REFERENCE_FILE}
    info = json.loads((run / MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    assert info["model"] == "logreg" and info["stage"] == "compound" and info["fish"] == 24
    assert {"python", "scikit-learn", "numpy", "pandas"} <= set(info["versions"])
    classes = json.loads((run / MODEL_DIR / "classes.json").read_text(encoding="utf-8"))
    reference = pd.read_csv(run / MODEL_DIR / REFERENCE_FILE)
    assert [c for c in reference.columns if c.startswith("p:")] == [f"p:{c}" for c in classes]
    assert "model/" in (run / "report.md").read_text(encoding="utf-8")
    assert json.loads((run / "run_info.json").read_text(encoding="utf-8"))["saved_model"]["model"] == "logreg"


def test_reload_in_a_fresh_process_reproduces_the_reference(tiny_table: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """EC-18 / AC-9 (D-006): the reloaded model gives the saved reference predictions."""
    run = trained(tiny_table, tmp_path, monkeypatch, "logreg")
    reference = pd.read_csv(run / MODEL_DIR / REFERENCE_FILE)
    again = fresh_process_predictions(run, tiny_table, tmp_path / "again.csv")
    pd.testing.assert_frame_equal(again.set_index("video_id").loc[reference["video_id"]].reset_index(), reference)


def test_mlp_reloads_in_a_fresh_process(tiny_table: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    small = {"mlp": {"hidden_sizes": [8], "max_epochs": 15, "seeds": 2}}
    run = trained(tiny_table, tmp_path, monkeypatch, "mlp", small)
    assert (run / MODEL_DIR / "mlp.pt").is_file()
    reference = pd.read_csv(run / MODEL_DIR / REFERENCE_FILE)
    again = fresh_process_predictions(run, tiny_table, tmp_path / "again.csv").set_index("video_id").loc[reference["video_id"]]
    columns = [c for c in reference.columns if c.startswith("p:")]
    np.testing.assert_allclose(again[columns].to_numpy(), reference[columns].to_numpy(), atol=1e-6)


def test_version_mismatch_is_a_warning_not_an_error(tiny_table: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = trained(tiny_table, tmp_path, monkeypatch, "logreg")
    info_path = run / MODEL_DIR / "model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["versions"]["scikit-learn"] = "0.0.1"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    loaded = load_model(run)
    assert any("scikit-learn" in warning and "0.0.1" in warning for warning in loaded.version_warnings)


def test_missing_feature_column_is_an_error_naming_it(tiny_table: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = trained(tiny_table, tmp_path, monkeypatch, "logreg")
    loaded = load_model(run)
    table, _ = load_table(tiny_table)
    gone = loaded.preprocess.features[0]
    with pytest.raises(ConfigError, match=gone):
        loaded.predict(table.drop(columns=[gone]))


def test_no_model_folder_is_an_error_with_a_hint(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="model"):
        load_model(tmp_path)


def test_a_changed_model_file_is_refused_before_it_is_unpickled(tiny_table: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = trained(tiny_table, tmp_path, monkeypatch, "logreg")
    with (run / MODEL_DIR / "sklearn.joblib").open("ab") as file:
        file.write(b"x")
    with pytest.raises(ConfigError, match="checksum"):
        load_model(run)


def test_a_model_saved_without_a_checksum_is_refused(tiny_table: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = trained(tiny_table, tmp_path, monkeypatch, "logreg")
    info_path = run / MODEL_DIR / "model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    del info["sha256"]
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ConfigError, match="no checksum"):
        load_model(run)
