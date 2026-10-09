"""Synthetic gold dataset, edge-case knobs (plan U2, T1.4): each knob changes only the fish it names."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dcs.schema import (
    FRAMES_FILE,
    MANIFEST_FILE,
    PROVENANCE_FILE,
    SEGMENTS_FILE,
    UNDETERMINED,
    WB_DATE,
    WB_SEX,
    WB_SUBJECT,
    WORKBOOK_SHEET,
)
from dcs.synthetic import DEFAULT_PROFILE, MODEL_PROFILE, SynthResult, design
from dcs.synthetic_config import GROUP_KNOBS, SINGLE_DATE_FISH, SINGLE_FISH_KNOBS, TINY_CLASS_FISH, SynthConfig
from dcs.synthetic_design import GROUP_COMBO, GROUP_VEHICLE
from tests.dcs.synth_helpers import LISTING, PATH_COLUMNS, SMALL, frames, index, make, read_json

# Every knob that targets named fish, with a value that switches it on.
KNOBS_ON: dict[str, Any] = {
    "missing_file": "frames",
    "undetermined": True,
    "null_date": True,
    "missing_frame_column": "velocity",
    "wrong_dtype_column": "velocity",
    "mixed_fps": True,
    "low_detected_fish": 1,
    "rare_state_fish": 2,
    "messy_labels": True,
    "mixed_profiles": True,
    "single_date_compound": True,
    "tiny_class": True,
}


def test_targeted_knobs_are_config_fields() -> None:
    names = {field.name for field in dataclasses.fields(SynthConfig)}
    assert set(KNOBS_ON) == set(SINGLE_FISH_KNOBS) | set(GROUP_KNOBS)
    assert set(KNOBS_ON) <= names


def _workbook_by_fish(result: SynthResult) -> pd.DataFrame:
    workbook = pd.read_excel(result.workbook_path, sheet_name=WORKBOOK_SHEET)
    keys = workbook[WB_SEX] + "_" + workbook[WB_SUBJECT].map(lambda value: f"{int(value):04d}")
    return workbook.set_axis(keys, axis=0)


@pytest.mark.parametrize("knob", sorted(KNOBS_ON))
def test_a_knob_changes_only_the_fish_it_names(tmp_path: Path, small: SynthResult, knob: str) -> None:
    knobbed = make(tmp_path, **{knob: KNOBS_ON[knob]})
    targets = set(knobbed.targets[knob])
    assert targets
    base_index = index(small).set_index("video_id").drop(columns=PATH_COLUMNS)
    new_index = index(knobbed).set_index("video_id").drop(columns=PATH_COLUMNS)
    base_workbook, new_workbook = _workbook_by_fish(small), _workbook_by_fish(knobbed)
    for video_id in set(small.truth["video_id"]) - targets:
        pd.testing.assert_frame_equal(frames(small, video_id), frames(knobbed, video_id))
        for name in (SEGMENTS_FILE, MANIFEST_FILE, PROVENANCE_FILE):
            before = (small.accepted_dir / video_id / name).read_text(encoding="utf-8")
            assert (knobbed.accepted_dir / video_id / name).read_text(encoding="utf-8") == before
        pd.testing.assert_series_equal(base_index.loc[video_id], new_index.loc[video_id])
        # a knob may turn a workbook column from numbers into mixed cells, so compare values, not dtypes
        pd.testing.assert_series_equal(base_workbook.loc[video_id], new_workbook.loc[video_id], check_dtype=False)


def test_single_fish_knob_targets_are_disjoint_with_everything_on(tmp_path: Path) -> None:
    result = make(tmp_path, **KNOBS_ON)
    seen: set[str] = set()
    for knob in SINGLE_FISH_KNOBS:
        targets = set(result.targets[knob])
        assert targets and not targets & seen, knob
        seen |= targets


def test_knob_groups_never_renumber_the_main_design() -> None:
    plain = [plan.video_id for plan in design(SMALL)]
    grouped = design(dataclasses.replace(SMALL, single_date_compound=True, tiny_class=True, messy_labels=True))
    assert [plan.video_id for plan in grouped[: len(plain)]] == plain


@pytest.mark.parametrize("which", ["frames", "segments"])
def test_missing_file_knob(tmp_path: Path, which: str) -> None:
    result = make(tmp_path, missing_file=which)
    (target,) = result.targets["missing_file"]
    name = FRAMES_FILE if which == "frames" else SEGMENTS_FILE
    assert not (result.accepted_dir / target / name).exists()
    assert target in set(index(result)["video_id"])


def test_undetermined_knob(tmp_path: Path) -> None:
    result = make(tmp_path, undetermined=True)
    (target,) = result.targets["undetermined"]
    assert UNDETERMINED in set(frames(result, target)["state"].astype(str))
    segments = pd.read_csv(result.accepted_dir / target / SEGMENTS_FILE)
    assert UNDETERMINED in set(segments["state"])


def test_mixed_profiles_knob(tmp_path: Path) -> None:
    result = make(tmp_path, mixed_profiles=True)
    table = index(result).set_index("video_id")
    assert set(table["calibration_profile_version"]) == {DEFAULT_PROFILE, MODEL_PROFILE}
    targets = result.targets["mixed_profiles"]
    assert set(table.index[table["calibration_profile_version"] == MODEL_PROFILE]) == set(targets)
    assert table.loc[list(targets), "date"].nunique() == 1  # one date switched tracker
    for video_id in targets:
        assert read_json(result, video_id, MANIFEST_FILE)["calibration_profile_version"] == MODEL_PROFILE
        assert read_json(result, video_id, PROVENANCE_FILE)["calibration_profile_version"] == MODEL_PROFILE


def test_null_date_knob_leaves_the_workbook_fields_empty_in_the_index(tmp_path: Path) -> None:
    result = make(tmp_path, null_date=True)
    (target,) = result.targets["null_date"]
    table = index(result).set_index("video_id")
    assert all(pd.isna(table.loc[target, field]) for field in ("date", "strain", "age", "agent_exposure_min"))
    assert table.drop(index=target)["date"].notna().all()
    workbook = pd.read_excel(result.workbook_path, sheet_name=WORKBOOK_SHEET)
    assert workbook[WB_DATE].notna().all()  # the workbook still has it (PRD change C4)


def test_missing_column_knob(tmp_path: Path) -> None:
    result = make(tmp_path, missing_frame_column="velocity")
    (target,) = result.targets["missing_frame_column"]
    assert "velocity" not in frames(result, target).columns


def test_wrong_dtype_knob_stores_text(tmp_path: Path) -> None:
    result = make(tmp_path, wrong_dtype_column="velocity")
    (target,) = result.targets["wrong_dtype_column"]
    assert pd.api.types.is_string_dtype(frames(result, target)["velocity"])


def test_mixed_fps_knob(tmp_path: Path) -> None:
    result = make(tmp_path, mixed_fps=True)
    (target,) = result.targets["mixed_fps"]
    table = index(result).set_index("video_id")
    assert table["video_fps"].nunique() == 2
    fps = table.loc[target, "video_fps"]
    assert fps != SMALL.fps
    step = frames(result, target)["t_sec"].diff().dropna()
    assert np.allclose(step, 1 / fps, atol=1e-4)


def test_single_date_and_tiny_class_knobs(tmp_path: Path) -> None:
    result = make(tmp_path, single_date_compound=True, tiny_class=True)
    truth = result.truth.set_index("video_id")
    single = truth.loc[list(result.targets["single_date_compound"])]
    assert single["compound"].nunique() == 1 and single["date"].nunique() == 1
    assert len(single) == SINGLE_DATE_FISH  # kept by the class filter, so the scheme-A pinning rule is exercised
    tiny = truth.loc[list(result.targets["tiny_class"])]
    assert tiny["compound"].nunique() == 1
    assert len(tiny) == TINY_CLASS_FISH  # one below the default class filter, so it is dropped
    assert tiny["date"].nunique() == 2


def test_constant_feature_knob(tmp_path: Path, small: SynthResult) -> None:
    detected = frames(small, small.truth["video_id"].iloc[0]).query("detected")
    assert detected["meander"].nunique() > 1  # control: not constant by default
    result = make(tmp_path, constant_meander=True)
    for video_id in result.truth["video_id"]:
        assert (frames(result, video_id)["meander"] == 0.0).all()


def test_low_detected_knob(tmp_path: Path) -> None:
    result = make(tmp_path, low_detected_fish=1)
    (target,) = result.targets["low_detected_fish"]
    shares = {video_id: frames(result, video_id)["detected"].mean() for video_id in result.truth["video_id"]}
    assert shares[target] < 0.6
    assert min(share for video_id, share in shares.items() if video_id != target) > 0.9


def test_rare_state_knob(tmp_path: Path) -> None:
    result = make(tmp_path, rare_state_fish=3)
    showing = {
        video_id
        for video_id in result.truth["video_id"]
        if LISTING in set(frames(result, video_id)["state"].astype(str))
    }
    assert showing == set(result.targets["rare_state_fish"])
    assert len(showing) == 3


def test_no_rare_state_by_default(small: SynthResult) -> None:
    for video_id in small.truth["video_id"]:
        assert LISTING not in set(frames(small, video_id)["state"].astype(str))


def test_messy_labels_knob(tmp_path: Path) -> None:
    result = make(tmp_path, messy_labels=True)
    table = index(result)
    targets = set(result.targets["messy_labels"])
    groups = set(result.truth.set_index("video_id").loc[list(targets), "group"])
    assert groups == {GROUP_VEHICLE, GROUP_COMBO}  # never a fish another single-fish knob uses
    messy = table[table["video_id"].isin(targets)]
    assert (messy["compound"] != messy["compound"].str.upper()).any()  # case variants
    assert (messy["compound"] == messy["compound"].str.strip()).all()  # prepds strips spaces
    combos = messy[messy["concentration_mM"].str.contains(r"\+")]
    assert combos["concentration_mM"].nunique() == 2
    assert combos["concentration_mM"].str.replace(" ", "", regex=False).nunique() == 1


def test_framing_shift_knob(tmp_path: Path, small: SynthResult) -> None:
    def date_spread(result: SynthResult) -> float:
        rows = [
            (row.date, float(frames(result, row.video_id)["depth_from_surface"].median()))
            for row in result.truth.itertuples()
        ]
        by_date = pd.DataFrame(rows, columns=["date", "depth"]).groupby("date")["depth"].median()
        return float(by_date.max() - by_date.min())

    shifted = make(tmp_path, framing_shift_px=40.0)
    # 4 dates shifted by 0, 40, 80, 120 px: the spread grows by about 120 over the unshifted control
    assert date_spread(shifted) - date_spread(small) > 100.0


@pytest.mark.parametrize(("fraction", "expected"), [(0.0, {"auto"}), (1.0, {"manual"})])
def test_manual_fraction_extremes(tmp_path: Path, fraction: float, expected: set[str]) -> None:
    assert set(index(make(tmp_path, manual_fraction=fraction))["provenance"]) == expected
