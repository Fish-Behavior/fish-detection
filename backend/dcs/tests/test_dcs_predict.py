"""`dcs predict` and `dcs featurize --videos` (plan U17, T5.3; PRD FR-8, AC-9; D-018)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from dcs import schema
from dcs.artifact import MODEL_DIR, REFERENCE_FILE
from dcs.cli import main
from dcs.synthetic import SynthResult


@pytest.fixture
def env(tiny_table: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    config = tmp_path / "fast.yaml"
    config.write_text(yaml.safe_dump({"training": {"folds": 3, "repeats": 1}}), encoding="utf-8")
    monkeypatch.setenv("DCS_TABLE", str(tiny_table))
    monkeypatch.setenv("DCS_OUTPUT_DIR", str(tmp_path / "out"))
    return {"config": config, "out": tmp_path / "out", "table": tiny_table}


def dcs(env: dict[str, Path], *args: str) -> int:
    return main(["--config", str(env["config"]), *args])


@pytest.fixture
def run(env: dict[str, Path]) -> Path:
    assert dcs(env, "train", "--stage", "compound", "--models", "logreg", "--no-ablations") == 0
    (folder,) = (env["out"] / "training").iterdir()
    return folder


def test_predict_reproduces_the_saved_reference(env: dict[str, Path], run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """AC-9: predicting the training table with the saved model gives the reference predictions, and says so."""
    out = env["out"] / "pred.csv"
    assert dcs(env, "predict", "--model", str(run), "--input", str(env["table"]), "--out", str(out)) == 0
    printed = capsys.readouterr().out
    assert "matches the saved reference predictions for 24 fish" in printed
    reference = pd.read_csv(run / MODEL_DIR / REFERENCE_FILE)
    pd.testing.assert_frame_equal(pd.read_csv(out).set_index("video_id").loc[reference["video_id"]].reset_index(), reference)


def test_predict_writes_next_to_the_outputs_by_default_and_reads_csv(env: dict[str, Path], run: Path) -> None:
    as_csv = env["out"] / "table.csv"
    pd.read_parquet(env["table"]).to_csv(as_csv, index=False)
    assert dcs(env, "predict", "--model", str(run), "--input", str(as_csv)) == 0
    written = pd.read_csv(env["out"] / "table_predictions.csv")
    assert len(written) == 24 and written.columns[0] == "video_id"


def test_missing_columns_stop_with_their_names(env: dict[str, Path], run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    feature = json.loads((run / MODEL_DIR / "preprocess.json").read_text(encoding="utf-8"))["features"][0]
    broken = env["out"] / "broken.parquet"
    pd.read_parquet(env["table"]).drop(columns=[feature]).to_parquet(broken)
    assert dcs(env, "predict", "--model", str(run), "--input", str(broken)) == 2
    assert feature in capsys.readouterr().out


def test_version_warnings_are_printed(env: dict[str, Path], run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    info_path = run / MODEL_DIR / "model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["versions"]["numpy"] = "0.0.1"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    assert dcs(env, "predict", "--model", str(run), "--input", str(env["table"])) == 0
    assert "warning: saved with numpy 0.0.1" in capsys.readouterr().out


def copied_videos(synth: SynthResult, tmp_path: Path) -> Path:
    """The synthetic per-video folders without accepted_index.parquet; one fish marked as not yet reviewed."""
    folder = tmp_path / "videos"
    shutil.copytree(synth.accepted_dir, folder, ignore=shutil.ignore_patterns(schema.INDEX_FILE))
    first = sorted(p for p in folder.iterdir() if p.is_dir())[0]
    manifest = json.loads((first / schema.MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["review_status"] = schema.REVIEW_PROCESSED_AUTO
    (first / schema.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def test_featurize_videos_needs_no_index_and_marks_unreviewed_fish(
    env: dict[str, Path], tiny_synth: SynthResult, tmp_path: Path
) -> None:
    videos = copied_videos(tiny_synth, tmp_path)
    assert dcs(env, "featurize", "--videos", str(videos)) == 0
    table = pd.read_parquet(env["out"] / "videos_table.parquet")
    assert len(table) == 24 and (~table["reviewed"]).sum() == 1
    assert table["date"].isna().all()  # no index or catalog: the date is unknown
    assert pd.read_parquet(env["table"]).shape[0] == 24  # the training table is untouched


def test_featurize_videos_with_the_workbook_keeps_the_ntt_values(
    env: dict[str, Path], tiny_synth: SynthResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DCS_DB_PATH", str(tiny_synth.workbook_path))
    out = tmp_path / "with_ntt.parquet"
    assert dcs(env, "featurize", "--videos", str(copied_videos(tiny_synth, tmp_path)), "--out", str(out)) == 0
    assert pd.read_parquet(out)["has_ntt"].sum() == pd.read_parquet(env["table"])["has_ntt"].sum()


def test_new_fish_from_folders_get_the_same_predictions(
    env: dict[str, Path], run: Path, tiny_synth: SynthResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-018 end to end: folders -> featurize --videos -> predict equals the reference for the same fish."""
    monkeypatch.setenv("DCS_DB_PATH", str(tiny_synth.workbook_path))
    table = tmp_path / "new.parquet"
    assert dcs(env, "featurize", "--videos", str(copied_videos(tiny_synth, tmp_path)), "--out", str(table)) == 0
    out = tmp_path / "new_pred.csv"
    assert dcs(env, "predict", "--model", str(run), "--input", str(table), "--out", str(out)) == 0
    reference = pd.read_csv(run / MODEL_DIR / REFERENCE_FILE).set_index("video_id")
    got: Any = pd.read_csv(out).set_index("video_id").loc[reference.index]
    pd.testing.assert_frame_equal(got, reference)
