"""Synthetic gold dataset (plan U2, T1.4-T1.5): config checks, the result object and `dcs synth`."""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
from typing import Any

import pytest

from dcs.cli import main
from dcs.config import load_settings
from dcs.schema import INDEX_FILE
from dcs.synthetic import DEFAULT_PROFILE, EFFECT_OFFSETS, FishFacts, SynthResult, design, make_gold_dataset
from dcs.synthetic_config import COMPOUND_LETTERS, SINGLE_DATE_FISH, TINY_CLASS_FISH, SynthConfig
from tests.dcs.synth_helpers import SMALL

# --- result object ------------------------------------------------------------------------------


def test_result_is_read_only(small: SynthResult) -> None:
    with pytest.raises(TypeError):
        small.targets["extra"] = ()  # type: ignore[index]
    copy = small.truth
    copy.loc[0, "compound"] = "CHANGED"
    assert small.truth.loc[0, "compound"] != "CHANGED"


def test_a_non_empty_folder_is_refused(tmp_path: Path) -> None:
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        make_gold_dataset(tmp_path / "busy", SMALL)


def test_an_unknown_knob_name_fails_loudly() -> None:
    facts = FishFacts(
        plan=design(SMALL)[0],
        knobs=frozenset(),
        fps=SMALL.fps,
        n_frames=1,
        profile=DEFAULT_PROFILE,
        speed_multiplier=1.0,
        compound_label="COMPOUND_A",
        dose_label="0.1",
    )
    with pytest.raises(KeyError):
        facts.has("mixed_fsp")


# --- config ------------------------------------------------------------------------------------


def test_group_sizes_follow_the_default_class_filter() -> None:
    """The single-date class must survive the filter (scheme A has to pin it); the tiny class must not."""
    min_class_size = load_settings(environ={}).training["min_class_size"]
    assert SINGLE_DATE_FISH == min_class_size
    assert TINY_CLASS_FISH == min_class_size - 1


def test_every_compound_has_an_effect_offset() -> None:
    assert len(EFFECT_OFFSETS) >= len(COMPOUND_LETTERS) + 3  # main compounds + single-date, tiny and combo groups


def test_a_small_design_without_knobs_is_valid() -> None:
    config = SynthConfig(n_compounds=1, doses_per_compound=1, fish_per_dose_date=2, n_dates=2, vehicle_per_date=1)
    assert config.main_fish == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"n_dates": 3, "doses_per_compound": 2},
        {"n_compounds": 0},
        {"n_compounds": 11},
        {"seed": -1},
        {"fish_per_dose_date": 2000},
        {"rare_state_fish": -1},
        {"rare_state_fish": 100},
        {"ntt_missing_fraction": 1.5},
        {"missing_file": "strip"},
        {"missing_frame_column": "no_such_column"},
        {"wrong_dtype_column": "no_such_column"},
        {"fps": 0},
        {"duration_s": 0},
        {"fps": math.nan},
        {"duration_s": math.inf},
        {"duration_s": 1e9},  # too many frames per fish
        {"compound_effect": math.nan},
        {"date_effect": math.inf},
        {"framing_shift_px": math.nan},
    ],
)
def test_invalid_config_is_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SMALL, **changes)


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_writes_a_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli_out"
    assert main(["synth", "--out", str(out), "--seed", "3"]) == 0
    assert (out / "accepted" / INDEX_FILE).is_file()
    assert (out / "synthetic_db.xlsx").is_file()
    assert "synthetic" in capsys.readouterr().out.lower()


def test_cli_synth_ignores_broken_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DCS_CONFIG", str(tmp_path / "absent.yaml"))
    assert main(["synth", "--out", str(tmp_path / "out")]) == 0


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs a non-root POSIX user")
def test_cli_unwritable_folder_is_a_one_line_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        assert main(["synth", "--out", str(locked / "out")]) == 2
    finally:
        locked.chmod(0o700)
    assert capsys.readouterr().out.startswith("Configuration error:")


def test_cli_bad_seed_is_a_config_error_and_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "never"
    assert main(["synth", "--out", str(out), "--seed", "-1"]) == 2
    assert capsys.readouterr().out.startswith("Configuration error:")
    assert not out.exists()


def test_cli_refuses_a_non_empty_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "busy"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    assert main(["synth", "--out", str(out)]) == 2
    assert "not empty" in capsys.readouterr().out
    assert sorted(path.name for path in out.iterdir()) == ["keep.txt"]


def test_cli_unknown_home_in_out_is_a_one_line_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["synth", "--out", "~no_such_user_dcs_test/out"]) == 2
    assert capsys.readouterr().out.startswith("Configuration error:")


def test_cli_refuses_a_file_as_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "a_file.txt"
    out.write_text("x")
    assert main(["synth", "--out", str(out)]) == 2
    assert "not a folder" in capsys.readouterr().out
