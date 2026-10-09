"""Cross-validation folds (plan U7, T1.14): EC-5, EC-6, EC-13, EC-14 (folds part), K reduced when dates are few."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dcs.config import load_settings
from dcs.featurize import featurize, write_outputs
from dcs.folds import PINNED, SCHEME_A, SCHEME_B, Folds, make_folds
from dcs.gold import read_accepted
from dcs.synthetic import make_gold_dataset
from dcs.trainset import build_trainset, load_table
from tests.dcs.synth_helpers import SMALL


@pytest.fixture(scope="module")
def defaults() -> dict[str, Any]:
    return dict(load_settings().training)


def fish(*cells: tuple[str, str, int]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(label, date, fish) cells -> y, groups, ids on a gappy index, as a filtered training set has."""
    rows = [(label, date) for label, date, n in cells for _ in range(n)]
    index = pd.Index(range(0, 3 * len(rows), 3))
    y = pd.Series([label for label, _ in rows], index=index, name="label")
    groups = pd.Series([date for _, date in rows], index=index, name="date")
    ids = pd.Series([f"F_{i:04d}" for i in range(len(rows))], index=index, name="video_id")
    return y, groups, ids


def design(seed: int = 0, n_dates: int = 10, two_date_classes: int = 4, per_cell: int = 3) -> list[tuple[str, str, int]]:
    """Vehicle on every date, each other class on two dates (the PRD §2.3 structure); dates shuffled by seed."""
    rng = np.random.default_rng(seed)
    dates = [f"<date {d}>" for d in rng.permutation(n_dates)]
    cells = [("VEHICLE", date, 2) for date in dates]
    for c in range(two_date_classes):
        cells += [(f"COMPOUND_{c}", dates[(2 * c) % n_dates], per_cell), (f"COMPOUND_{c}", dates[(2 * c + 5) % n_dates], per_cell)]
    return cells


def build(cells: list[tuple[str, str, int]], training: dict[str, Any], **changes: Any) -> Folds:
    return make_folds(*fish(*cells), {**training, **changes})


def scheme(result: Folds, name: str) -> pd.DataFrame:
    return result.table[result.table["scheme"] == name]


def collided(result: Folds) -> set[tuple[int, str]]:
    """(repeat, label) whose two or more dates all fell in one scheme-A fold."""
    held = scheme(result, SCHEME_A)
    held = held[held["fold"] != PINNED]
    out = set()
    for (repeat, label), rows in held.groupby(["repeat", "label"]):
        if rows["date"].nunique() >= 2 and rows["fold"].nunique() == 1:
            out.add((int(repeat), str(label)))
    return out


# --- table layout -----------------------------------------------------------------------------


def test_every_fish_has_one_row_per_scheme_and_repeat(defaults: dict[str, Any]) -> None:
    cells = design()
    result = build(cells, defaults, repeats=3)
    assert list(result.table.columns) == ["video_id", "label", "date", "scheme", "repeat", "fold"]
    n = sum(n for _, _, n in cells)
    assert len(result.table) == n * 2 * 3
    assert not result.table.duplicated(["video_id", "scheme", "repeat"]).any()
    assert set(result.table["repeat"]) == {0, 1, 2}


def test_labels_and_dates_travel_with_the_fish(defaults: dict[str, Any]) -> None:
    y, groups, ids = fish(*design())
    result = make_folds(y, groups, ids, defaults)
    row = result.table.set_index(["video_id", "scheme", "repeat"]).loc[(ids.iloc[7], SCHEME_B, 0)]
    assert row["label"] == y.iloc[7] and row["date"] == groups.iloc[7]


def test_misaligned_inputs_are_a_caller_bug(defaults: dict[str, Any]) -> None:
    y, groups, ids = fish(*design())
    with pytest.raises(ValueError, match="index"):
        make_folds(y, groups.reset_index(drop=True), ids, defaults)


# --- scheme B ---------------------------------------------------------------------------------


def test_scheme_b_uses_k_folds_and_spreads_each_class(defaults: dict[str, Any]) -> None:
    result = build([("A", "<d0>", 10), ("B", "<d1>", 10), ("C", "<d2>", 10)], defaults)
    rows = scheme(result, SCHEME_B)
    assert result.k[SCHEME_B] == 5
    assert (rows.groupby(["repeat", "fold"])["label"].nunique() == 3).all()  # stratified: every class in every fold
    assert (rows["fold"] != PINNED).all()


# --- scheme A, EC-5: a class on one date ------------------------------------------------------


def test_single_date_class_is_pinned_to_training_and_flagged(defaults: dict[str, Any]) -> None:
    cells = [*design(), ("COMPOUND_S", "<date 3>", 6)]
    result = build(cells, defaults, repeats=2)
    a, b = scheme(result, SCHEME_A), scheme(result, SCHEME_B)
    assert result.date_confounded == ("COMPOUND_S",)
    assert (a.loc[a["label"] == "COMPOUND_S", "fold"] == PINNED).all()
    assert (a.loc[a["label"] != "COMPOUND_S", "fold"] != PINNED).all()
    assert (b.loc[b["label"] == "COMPOUND_S", "fold"] != PINNED).all()  # scored in scheme B
    assert any("COMPOUND_S" in note for note in result.notes)


def test_scheme_a_is_skipped_when_every_class_is_on_one_date(defaults: dict[str, Any]) -> None:
    # the dose stage on real data: doses of one compound never share a date (PRD §6.5)
    result = build([("A @ 0.1", "<d0>", 6), ("A @ 0.3", "<d1>", 6), ("B @ 0.1", "<d2>", 6)], defaults, repeats=1)
    assert result.k[SCHEME_A] == 0
    assert scheme(result, SCHEME_A).empty
    assert len(scheme(result, SCHEME_B)) == 18
    assert any("scheme A" in note for note in result.notes)


# --- scheme A, EC-6: a class on two dates -----------------------------------------------------


def test_dates_that_cannot_be_split_are_named_in_a_note(defaults: dict[str, Any]) -> None:
    # three classes on the three pairs of three dates: with 2 folds two dates must share a fold
    cells = [("A", "<d0>", 4), ("A", "<d1>", 4), ("B", "<d1>", 4), ("B", "<d2>", 4), ("C", "<d0>", 4), ("C", "<d2>", 4)]
    result = build(cells, defaults, folds=2, repeats=4)
    hit = collided(result)
    assert hit
    named = {
        (repeat, label)
        for repeat in range(4)
        for label in "ABC"
        if any(f"repeat {repeat}: the dates of {label} " in note for note in result.notes)
    }
    assert named == hit


# --- EC-13 (and EC-6 again): a fish or a date never in two folds, over 200 seeds --------------


@pytest.mark.parametrize("seed", range(200))
def test_no_fish_or_date_in_two_folds(defaults: dict[str, Any], seed: int) -> None:
    cells = [*design(seed, n_dates=8 + seed % 5), ("COMPOUND_S", "<date 1>", 6)]
    result = build(cells, defaults, seed=seed, repeats=2)
    assert collided(result) == set()  # EC-6: each two-date class has its dates in two folds
    assert not [note for note in result.notes if "one fold" in note]
    assert not result.table.duplicated(["video_id", "scheme", "repeat"]).any()
    held = scheme(result, SCHEME_A)
    held = held[held["fold"] != PINNED]
    assert (held.groupby(["repeat", "date"])["fold"].nunique() == 1).all()
    for name in (SCHEME_A, SCHEME_B):
        rows = scheme(result, name)
        used = rows.loc[rows["fold"] != PINNED].groupby("repeat")["fold"].apply(set)
        assert all(folds == set(range(result.k[name])) for folds in used)  # no empty fold


# --- EC-14: same seed, same folds -------------------------------------------------------------


def test_same_seed_gives_the_same_folds(defaults: dict[str, Any]) -> None:
    first, second = build(design(), defaults, seed=7), build(design(), defaults, seed=7)
    pd.testing.assert_frame_equal(first.table, second.table)


def test_other_seed_and_other_repeat_give_other_folds(defaults: dict[str, Any]) -> None:
    first, other = build(design(), defaults, seed=7, repeats=2), build(design(), defaults, seed=8, repeats=2)
    assert not first.table["fold"].equals(other.table["fold"])
    a = scheme(first, SCHEME_A)
    assert not a.loc[a["repeat"] == 0, "fold"].reset_index(drop=True).equals(a.loc[a["repeat"] == 1, "fold"].reset_index(drop=True))


# --- fewer dates than K -----------------------------------------------------------------------


def test_fewer_dates_than_k_reduces_k_with_a_note(defaults: dict[str, Any]) -> None:
    cells = [("A", "<d0>", 6), ("A", "<d1>", 6), ("B", "<d1>", 6), ("B", "<d2>", 6)]
    result = build(cells, defaults)
    assert result.k == {SCHEME_A: 3, SCHEME_B: 5}
    assert set(scheme(result, SCHEME_A)["fold"]) == {0, 1, 2}
    assert any("3" in note and "5" in note for note in result.notes)


def test_pinned_dates_do_not_count_towards_k(defaults: dict[str, Any]) -> None:
    cells = [("A", "<d0>", 6), ("A", "<d1>", 6), ("S", "<d2>", 6), ("T", "<d3>", 6)]  # S, T pinned
    assert build(cells, defaults).k[SCHEME_A] == 2


def test_default_k_draws_no_note(defaults: dict[str, Any]) -> None:
    assert build(design(), defaults).notes == ()


# --- a synthetic set end to end ---------------------------------------------------------------


def test_synthetic_set_end_to_end(tmp_path_factory: pytest.TempPathFactory, defaults: dict[str, Any]) -> None:
    config = dataclasses.replace(SMALL, single_date_compound=True)
    synth = make_gold_dataset(tmp_path_factory.mktemp("folds"), config)
    result = featurize(read_accepted(synth.accepted_dir), synth.workbook_path, defaults)
    table_path, _ = write_outputs(result, tmp_path_factory.mktemp("out") / "training_table.parquet")
    table, described = load_table(table_path)
    training = {**defaults, "min_class_size": 4, "min_state_fish": 4}
    trainset = build_trainset(table, described, training, "compound")
    folds = make_folds(trainset.y, trainset.groups, trainset.ids, training)
    assert folds.date_confounded == ("COMPOUND_S",)
    assert folds.k == {SCHEME_A: 4, SCHEME_B: 5}  # SMALL has 4 dates
    assert collided(folds) == set()
    assert set(folds.table["video_id"]) == set(trainset.ids)
