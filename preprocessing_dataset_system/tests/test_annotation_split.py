"""Frozen subject-level train/held-out split (Phase 15): decided before any frame is sampled."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prepds.annotation.split import load_split, make_split, write_split

GROUPS = {f"F_{i:04d}": ("Veh" if i < 30 else "MDMA" if i < 50 else "Tiny") for i in range(1, 53)}
GROUPS["F_0100"] = "Tiny"  # 3 videos in total: too few to hold one out


def test_split_is_deterministic_and_independent_of_input_order() -> None:
    a = make_split(GROUPS, heldout_fraction=0.2, seed=15)
    b = make_split(dict(reversed(list(GROUPS.items()))), heldout_fraction=0.2, seed=15)
    assert a == b
    assert make_split(GROUPS, heldout_fraction=0.2, seed=16) != a


def test_every_video_is_assigned_and_heldout_is_stratified_by_group() -> None:
    split = make_split(GROUPS, heldout_fraction=0.2, seed=15)
    assert set(split) == set(GROUPS) and set(split.values()) == {"train", "heldout"}
    for group in ("Veh", "MDMA"):
        members = [v for v, g in GROUPS.items() if g == group]
        heldout = sum(split[v] == "heldout" for v in members)
        assert abs(heldout - 0.2 * len(members)) <= 1 and heldout >= 1


def test_groups_too_small_to_hold_out_stay_in_train() -> None:
    split = make_split(GROUPS, heldout_fraction=0.2, seed=15)
    assert all(split[v] == "train" for v, g in GROUPS.items() if g == "Tiny")


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_heldout_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError):
        make_split(GROUPS, heldout_fraction=fraction, seed=15)


def test_write_split_is_immutable_and_round_trips(tmp_path: Path) -> None:
    split = make_split(GROUPS, heldout_fraction=0.2, seed=15)
    path = tmp_path / "split.json"
    write_split(path, split, seed=15, heldout_fraction=0.2)
    assert load_split(path) == split
    assert json.loads(path.read_text())["seed"] == 15
    with pytest.raises(FileExistsError):
        write_split(path, split, seed=15, heldout_fraction=0.2)


def test_load_split_rejects_unknown_labels(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"seed": 1, "heldout_fraction": 0.2, "assignments": {"F_0001": "validation"}}))
    with pytest.raises(ValueError):
        load_split(path)
