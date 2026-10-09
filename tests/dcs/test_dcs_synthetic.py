"""Synthetic gold dataset and workbook (plan U2, T1.4): the prepds file contract, determinism,
planted structure and the workbook. Edge-case knobs: test_dcs_synthetic_knobs.py; config, result
object and CLI: test_dcs_synthetic_config.py.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dcs.schema import (
    FRAME_DTYPES,
    FRAMES_FILE,
    INDEX_COLUMNS,
    INDEX_FILE,
    KINEMATIC_COLUMNS,
    MANIFEST_FILE,
    MANIFEST_KEYS,
    NTT_HEADERS,
    PROVENANCE_FILE,
    PROVENANCE_KEYS,
    SEGMENT_COLUMNS,
    SEGMENTS_FILE,
    SOURCES,
    STATES,
    UNDETERMINED,
    WB_AGENT_EXPOSURE,
    WB_COMPOUND,
    WB_DATE,
    WB_DOSE,
    WB_SEX,
    WB_SUBJECT,
    WORKBOOK_SHEET,
    ZERO_ON_FIRST_FRAME,
    ZERO_ON_FIRST_TWO_FRAMES,
    ZERO_ON_UNDETECTED,
)
from dcs.synthetic import DEFAULT_PROFILE, VEHICLE, SynthResult, make_gold_dataset
from dcs.synthetic_frames import IMMOBILITY_FLOOR_PX_S, IMMOBILITY_WINDOW_FRAMES
from dcs.synthetic_workbook import write_workbook
from tests.dcs.synth_helpers import PATH_COLUMNS, SMALL, frames, index, make, read_json

# --- layout and the prepds contract --------------------------------------------------------


def test_layout_index_and_truth(small: SynthResult) -> None:
    table = index(small)
    assert small.index_path == small.accepted_dir / INDEX_FILE
    assert list(table.columns) == list(INDEX_COLUMNS)
    assert table["video_id"].is_unique
    assert all(re.fullmatch(r"[MF]_\d{4}", video_id) for video_id in table["video_id"])
    assert (table["video_id"] == table["sex"] + "_" + table["subject_id"]).all()
    folders = sorted(path.name for path in small.accepted_dir.iterdir() if path.is_dir())
    assert folders == sorted(table["video_id"])
    for video_id in folders:
        for name in (FRAMES_FILE, SEGMENTS_FILE, MANIFEST_FILE, PROVENANCE_FILE):
            assert (small.accepted_dir / video_id / name).is_file()
    assert sorted(small.truth["video_id"]) == folders
    assert {"video_id", "compound", "dose", "date"} <= set(small.truth.columns)
    # 2 compounds x 2 doses x 2 dates x 2 fish + 1 vehicle on each of 4 dates
    assert len(table) == 2 * 2 * 2 * 2 + 4


def test_index_paths_point_into_the_gold_folder(small: SynthResult) -> None:
    row = index(small).iloc[0]
    folder = small.accepted_dir / row["video_id"]
    assert row["frames_path"] == str(folder / FRAMES_FILE)
    assert row["segments_path"] == str(folder / SEGMENTS_FILE)
    assert row["manifest_path"] == str(folder / MANIFEST_FILE)


def test_frames_match_the_prepds_schema(small: SynthResult) -> None:
    table = frames(small, small.truth["video_id"].iloc[0])
    assert list(table.columns) == list(FRAME_DTYPES)
    assert {column: str(dtype) for column, dtype in table.dtypes.items()} == dict(FRAME_DTYPES)
    assert list(table["state"].cat.categories) == list(STATES)
    assert list(table["source"].cat.categories) == list(SOURCES)
    assert (table["frame_idx"].diff().dropna() == 1).all()
    assert UNDETERMINED not in set(table["state"].astype(str))


def _all_frames(result: SynthResult) -> pd.DataFrame:
    return pd.concat([frames(result, video_id) for video_id in result.truth["video_id"]], ignore_index=True)


def test_undetected_frames_use_the_prepds_sentinels(small: SynthResult) -> None:
    table = _all_frames(small)
    missed = table[~table["detected"]]
    assert len(missed) > 0
    assert (missed[list(ZERO_ON_UNDETECTED)] == 0.0).all().all()
    assert missed["orientation_deg"].isna().all() and missed["depth_from_surface"].isna().all()
    assert table.loc[table["detected"], "depth_from_surface"].notna().all()


def _detected_run_length(detected: pd.Series) -> np.ndarray:
    """1 on the first frame of each detected run, 2 on the second, ...; 0 on undetected frames."""
    run = np.zeros(len(detected), dtype=int)
    count = 0
    for position, flag in enumerate(detected):
        count = count + 1 if flag else 0
        run[position] = count
    return run


def test_kinematics_are_zero_without_history(small: SynthResult) -> None:
    """As prepds features.py: speed and turn need one earlier frame of the run, acceleration and meander two."""
    second_frames = 0
    for video_id in small.truth["video_id"]:
        table = frames(small, video_id)
        run = _detected_run_length(table["detected"])
        assert (table.loc[run == 1, list(ZERO_ON_FIRST_FRAME)] == 0.0).all().all()
        assert (table.loc[(run == 1) | (run == 2), list(ZERO_ON_FIRST_TWO_FRAMES)] == 0.0).all().all()
        assert (table.loc[run >= 3, list(KINEMATIC_COLUMNS)] != 0.0).all().all()  # real values once history exists
        assert (table.loc[run == 2, list(ZERO_ON_FIRST_FRAME)] != 0.0).all().all()
        second_frames += int((run == 2).sum())
    assert second_frames > len(small.truth)  # restarts after gaps were exercised, not only frame 0


def _prepds_immobile(velocity: np.ndarray, run: np.ndarray, window: int, floor: float) -> np.ndarray:
    """Reference for prepds features._compute_immobility: the last `window` frames all have a real
    velocity (run length >= 2) at or below `floor`."""
    slow = (run >= 2) & (velocity <= floor)
    return np.array([i >= window - 1 and bool(slow[i - window + 1 : i + 1].all()) for i in range(len(velocity))])


def test_immobility_follows_the_prepds_window_rule(tmp_path: Path) -> None:
    result = make(tmp_path, duration_s=60.0)
    immobile_frames = 0
    for video_id in result.truth["video_id"]:
        table = frames(result, video_id)
        run = _detected_run_length(table["detected"])
        expected = _prepds_immobile(
            table["velocity"].to_numpy(), run, IMMOBILITY_WINDOW_FRAMES, IMMOBILITY_FLOOR_PX_S
        )
        assert (table["is_immobile"].to_numpy() == expected).all(), video_id
        immobile_frames += int(expected.sum())
    assert immobile_frames > 0  # the rule is exercised, so a share-immobile feature has something to see


def test_confidence_is_always_empty(small: SynthResult) -> None:
    """prepds labeling sets no confidence on any frame (labeling.py: nothing probabilistic remains)."""
    assert _all_frames(small)["confidence"].isna().all()


def test_segments_are_the_run_length_encoding_of_the_frames(small: SynthResult) -> None:
    for video_id in small.truth["video_id"][:5]:
        table = frames(small, video_id)
        segments = pd.read_csv(small.accepted_dir / video_id / SEGMENTS_FILE)
        assert list(segments.columns) == list(SEGMENT_COLUMNS)
        key = table["state"].astype(str) + "|" + table["source"].astype(str)
        starts = table.index[key.ne(key.shift())]
        assert len(segments) == len(starts)
        assert list(segments["state"]) == list(table.loc[starts, "state"].astype(str))
        assert list(segments["source"]) == list(table.loc[starts, "source"].astype(str))
        assert np.allclose(segments["start_s"], table.loc[starts, "t_sec"].to_numpy(), atol=1e-4)
        assert np.allclose(segments["end_s"].iloc[:-1], segments["start_s"].iloc[1:], atol=1e-4)
        fps = read_json(small, video_id, MANIFEST_FILE)["video_fps"]
        assert segments["end_s"].iloc[-1] == pytest.approx(table["t_sec"].iloc[-1] + 1 / fps, abs=1e-4)
        assert np.allclose(segments["duration_s"], segments["end_s"] - segments["start_s"], atol=1e-4)


def test_manifest_provenance_and_index_agree(small: SynthResult) -> None:
    table = index(small).set_index("video_id")
    fields = ("compound", "concentration_mM", "calibration_profile_version", "video_fps", "video_duration_s", "edit_count")
    for video_id in small.truth["video_id"]:
        manifest = read_json(small, video_id, MANIFEST_FILE)
        provenance = read_json(small, video_id, PROVENANCE_FILE)
        assert tuple(manifest) == MANIFEST_KEYS
        assert tuple(provenance) == PROVENANCE_KEYS
        assert manifest["review_status"] == "ACCEPTED"
        row = table.loc[video_id]
        for field in fields:
            assert manifest[field] == row[field], field
        edited = bool((frames(small, video_id)["source"] == "manual").any())
        assert manifest["edited"] is edited
        assert (manifest["edit_count"] > 0) is edited
        assert provenance["source"] == row["provenance"] == ("manual" if edited else "auto")


def test_some_fish_have_reviewer_edits(small: SynthResult) -> None:
    assert index(small)["provenance"].eq("manual").any()
    assert index(small)["provenance"].eq("auto").any()


def test_only_placeholder_names(small: SynthResult) -> None:
    table = index(small)
    assert all(re.fullmatch(r"COMPOUND_[A-Z]( \+ COMPOUND_[A-Z])?|VEHICLE", name) for name in table["compound"])
    assert set(table["reviewer"]) == {"reviewer_1"}
    assert set(table["strain"]) == {"STRAIN_1"}
    assert set(table["calibration_profile_version"]) == {DEFAULT_PROFILE}


# --- design and planted structure ------------------------------------------------------------


def test_design_mimics_the_date_confound(small: SynthResult) -> None:
    truth = small.truth
    dates = sorted(truth["date"].unique())
    assert dates[0] == "2000-01-01"
    vehicle = truth[truth["compound"] == VEHICLE]
    assert sorted(vehicle["date"].unique()) == dates  # vehicle on every date
    drugs = truth[truth["compound"] != VEHICLE]
    for _, group in drugs.groupby("compound"):
        date_sets = [set(dose_group["date"]) for _, dose_group in group.groupby("dose")]
        assert all(len(date_set) == 2 for date_set in date_sets)  # each dose on two dates
        assert not set.intersection(*date_sets)  # doses of one compound never share a date


def test_same_seed_same_content_other_seed_different(tmp_path: Path, small: SynthResult) -> None:
    again = make_gold_dataset(tmp_path / "again", SMALL)
    pd.testing.assert_frame_equal(index(small).drop(columns=PATH_COLUMNS), index(again).drop(columns=PATH_COLUMNS))
    pd.testing.assert_frame_equal(small.truth, again.truth)
    pd.testing.assert_frame_equal(
        pd.read_excel(small.workbook_path, sheet_name=WORKBOOK_SHEET),
        pd.read_excel(again.workbook_path, sheet_name=WORKBOOK_SHEET),
    )
    video_id = small.truth["video_id"].iloc[3]
    pd.testing.assert_frame_equal(frames(small, video_id), frames(again, video_id))
    other = make_gold_dataset(tmp_path / "other", dataclasses.replace(SMALL, seed=1))
    assert not frames(small, video_id).equals(frames(other, video_id))


def _mean_speed(result: SynthResult) -> pd.Series:
    rows = []
    for row in result.truth.itertuples():
        table = frames(result, row.video_id)
        rows.append((row.compound, row.date, float(table.loc[table["detected"], "velocity"].mean())))
    return pd.DataFrame(rows, columns=["compound", "date", "speed"]).set_index(["compound", "date"])["speed"]


def test_compound_effect_is_planted_and_can_be_switched_off(tmp_path: Path) -> None:
    def spread(result: SynthResult) -> float:
        by_compound = _mean_speed(result).groupby(level="compound").mean()
        return float(by_compound.max() / by_compound.min())

    with_signal = make(tmp_path / "on", compound_effect=1.0, date_effect=0.0, duration_s=40.0)
    without = make(tmp_path / "off", compound_effect=0.0, date_effect=0.0, duration_s=40.0)
    assert spread(with_signal) > 1.5
    assert spread(without) < 1.4


def test_date_effect_is_planted(tmp_path: Path) -> None:
    result = make(tmp_path, compound_effect=0.0, date_effect=1.0, duration_s=40.0)
    vehicle = _mean_speed(result).xs(VEHICLE, level="compound")
    assert vehicle.max() / vehicle.min() > 1.3


# --- workbook ---------------------------------------------------------------------------------


def _prepds_dose(value: Any) -> str:
    """How prepds turns a workbook dose cell into the index string (numbers lose a trailing .0)."""
    if isinstance(value, str):
        return value.strip()
    return str(int(value)) if float(value).is_integer() else str(value)


def test_workbook_rows_match_the_index(small: SynthResult) -> None:
    workbook = pd.read_excel(small.workbook_path, sheet_name=WORKBOOK_SHEET)
    headers = [str(column) for column in workbook.columns]
    assert "NTT \nTime (min):" in headers  # real header, embedded newline kept
    for header in (WB_DATE, WB_SUBJECT, WB_SEX, WB_COMPOUND, WB_DOSE, *NTT_HEADERS):
        assert header in headers
    keys = workbook[WB_SEX] + "_" + workbook[WB_SUBJECT].map(lambda value: f"{int(value):04d}")
    table = index(small).set_index("video_id").loc[keys]
    assert sorted(keys) == sorted(small.truth["video_id"])
    assert list(workbook[WB_COMPOUND]) == list(table["compound"])
    assert [_prepds_dose(value) for value in workbook[WB_DOSE]] == list(table["concentration_mM"])
    assert list(workbook[WB_AGENT_EXPOSURE]) == list(table["agent_exposure_min"])
    dates = [f"{int(value):06d}" for value in workbook[WB_DATE]]  # YYMMDD numbers, as in the real workbook
    assert [f"20{d[:2]}-{d[2:4]}-{d[4:]}" for d in dates] == list(table["date"])


def test_workbook_messy_spellings_match_the_index(tmp_path: Path) -> None:
    result = make(tmp_path, messy_labels=True)
    workbook = pd.read_excel(result.workbook_path, sheet_name=WORKBOOK_SHEET)
    keys = workbook[WB_SEX] + "_" + workbook[WB_SUBJECT].map(lambda value: f"{int(value):04d}")
    table = index(result).set_index("video_id").loc[keys]
    assert list(workbook[WB_COMPOUND]) == list(table["compound"])
    assert [_prepds_dose(value) for value in workbook[WB_DOSE]] == list(table["concentration_mM"])


def test_workbook_doses_are_numbers_unless_written_as_text(small: SynthResult, tmp_path: Path) -> None:
    assert pd.api.types.is_numeric_dtype(pd.read_excel(small.workbook_path, sheet_name=WORKBOOK_SHEET)[WB_DOSE])
    messy = pd.read_excel(make(tmp_path, messy_labels=True).workbook_path, sheet_name=WORKBOOK_SHEET)[WB_DOSE]
    kinds = {type(value).__name__ for value in messy}
    assert "str" in kinds and kinds & {"float", "int"}  # mixed column, as in the real workbook


def test_workbook_rejects_a_misspelt_header(tmp_path: Path, small: SynthResult) -> None:
    record = pd.read_excel(small.workbook_path, sheet_name=WORKBOOK_SHEET).iloc[0].to_dict()
    record["Compound:"] = record.pop(WB_COMPOUND)  # the real header is spelt "Compund:"
    with pytest.raises(ValueError, match="Compound:"):
        write_workbook(tmp_path / "bad.xlsx", [record])
    assert not (tmp_path / "bad.xlsx").exists()


def test_workbook_ntt_missing_for_some_fish(tmp_path: Path) -> None:
    result = make(tmp_path, ntt_missing_fraction=0.5)
    workbook = pd.read_excel(result.workbook_path, sheet_name=WORKBOOK_SHEET)
    ntt = workbook[list(NTT_HEADERS)]
    all_missing = ntt.isna().all(axis=1)
    assert all_missing.any() and (~all_missing).any()
    assert (ntt.isna().any(axis=1) == all_missing).all()  # a fish has all 8 values or none


