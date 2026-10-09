"""Training report (plan U12, T2.7; PRD §6.4-6.7, AC-5, US-2): baselines beside every model, date effect, caveats."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dcs.audit import Audit
from dcs.config import load_settings
from dcs.evaluate import date_effects, evaluate
from dcs.folds import make_folds
from dcs.report import StageResult, best_model, render_report, save_confusion, scores_table
from dcs.trainset import TrainSet
from tests.dcs.eval_helpers import CLASSES, as_trainset, signal_set

FAST = ["majority", "date_only", "logreg"]
INFO = {
    "run_id": "20001231T000000Z-abc1234",
    "command": "python -m dcs train",
    "git_commit": "abc1234",
    "git_dirty": False,
    "gold_source": "processed",
    "seed": 0,
    "folds": 3,
    "repeats": 2,
    "models": FAST,
    "skipped_models": [],
}


@pytest.fixture(scope="module")
def training() -> dict[str, Any]:
    return {**load_settings().training, "folds": 3, "repeats": 2}


def stage_result(ts: TrainSet, training: dict[str, Any]) -> StageResult:
    folds = make_folds(ts.y, ts.groups, ts.ids, training)
    return StageResult(ts, folds, evaluate(ts, folds, FAST, training))


def audit(**facts: Any) -> Audit:
    base = {"source": "UNREVIEWED pipeline output", "framing differs between dates": False}
    return Audit(facts={**base, **facts}, tables={}, notes=())


@pytest.fixture(scope="module")
def compound(training: dict[str, Any]) -> StageResult:
    return stage_result(signal_set(signal=4.0), training)


def test_baselines_are_shown_next_to_every_model(compound: StageResult) -> None:
    """AC-5: majority, date-only and the within-date permuted score sit on every model's row."""
    table = scores_table(compound.evaluation)
    assert list(table["model"]) == FAST
    assert {"scheme A", "scheme B", "B permuted", "date effect (B - A)", "verdict"} <= set(table.columns)


def test_date_effect_is_scheme_b_minus_scheme_a(compound: StageResult) -> None:
    """With no date-confounded class both schemes score the same fish, so the effect is the plain difference."""
    summary = compound.evaluation.summary["balanced_accuracy"]["mean"]
    row = scores_table(compound.evaluation).set_index("model").loc["logreg"]
    assert row["date effect (B - A)"] == f"{summary[('logreg', 'B')] - summary[('logreg', 'A')]:+.3f}"


def test_date_effect_leaves_out_fish_that_scheme_a_does_not_score() -> None:
    """A pinned (date-confounded) class is scored in B only; its fish must not move the B - A gap."""
    rows = [("A", "f1", "X", "X"), ("A", "f2", "Y", "X"), ("B", "f1", "X", "X"), ("B", "f2", "Y", "Y"), ("B", "f3", "Z", "X")]
    predictions = pd.DataFrame(
        [{"model": "m", "repeat": 0, "scheme": s, "video_id": v, "label": t, "predicted": p} for s, v, t, p in rows]
    )
    assert date_effects(predictions)["m"] == pytest.approx(0.5)  # B on f1, f2 = 1.0; A = 0.5; naive B - A would be 0.0


def test_no_plain_accuracy_is_reported(compound: StageResult) -> None:
    text = render_report(INFO, {"compound": compound}, audit(), [])
    header = [line for line in text.splitlines() if line.startswith("| model")]
    assert header and all(" accuracy |" not in line.replace("balanced", "").replace("top-3", "") for line in header)


def test_unreviewed_data_is_called_temporarily_accepted(compound: StageResult) -> None:
    text = render_report(INFO, {"compound": compound}, audit(), [])
    assert "UNREVIEWED" in text and "temporarily accepted" in text
    reviewed = render_report({**INFO, "gold_source": "accepted"}, {"compound": compound}, audit(), [])
    assert "UNREVIEWED" not in reviewed and "Accepted gold folder" in reviewed


def test_report_names_very_small_and_date_confounded_classes(training: dict[str, Any]) -> None:
    ts = signal_set()
    y = ts.y.where(~((ts.groups == "2000-01-01") & (ts.y == "VEHICLE")), "COMPOUND_C")
    ts = dataclasses.replace(as_trainset(ts.X, list(y), list(ts.groups)), very_small=("COMPOUND_B",))
    text = render_report(INFO, {"compound": stage_result(ts, training)}, audit(), [])
    assert "date-confounded" in text and "COMPOUND_C" in text
    assert "very small" in text.lower() and "COMPOUND_B" in text


def test_dose_stage_carries_the_date_confound_caveat(training: dict[str, Any]) -> None:
    result = stage_result(dataclasses.replace(signal_set(), stage="dose"), training)
    text = render_report(INFO, {"dose": result}, audit(), [])
    assert "Exploratory" in text and "§6.5" in text


def test_framing_and_normalization_caveats(compound: StageResult) -> None:
    text = render_report(INFO, {"compound": compound}, audit(**{"framing differs between dates": True}), [])
    assert "camera" in text.lower() and "FR-9" in text
    calm = render_report(INFO, {"compound": compound}, audit(), [])
    assert "Camera framing differs" not in calm


def test_notes_and_skipped_models_are_shown(compound: StageResult) -> None:
    text = render_report({**INFO, "skipped_models": ["mlp"]}, {"compound": compound}, audit(), ["dose stage cannot be built: x"])
    assert "dose stage cannot be built" in text and "mlp" in text


def test_skipped_scheme_a_shows_as_skipped(training: dict[str, Any]) -> None:
    ts = signal_set(n_dates=3)
    dates = [f"2000-01-0{CLASSES.index(label) + 1}" for label in ts.y]
    result = stage_result(as_trainset(ts.X, list(ts.y), dates), training)
    row = scores_table(result.evaluation).set_index("model").loc["logreg"]
    assert row["scheme A"] == "skipped" and row["verdict"] == "undecided"
    assert best_model(result.evaluation) == ("logreg", "B")


def test_best_model_is_the_top_scheme_a_model_beside_the_references(compound: StageResult) -> None:
    assert best_model(compound.evaluation) == ("logreg", "A")


def test_confusion_matrix_is_saved_as_png(compound: StageResult, tmp_path: Path) -> None:
    path = tmp_path / "confusion_compound.png"
    save_confusion(compound, path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
