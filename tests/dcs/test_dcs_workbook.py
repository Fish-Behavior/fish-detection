"""NTT columns from the trial workbook (plan U4, T1.8): EC-28, EC-2 (flag part), D-004."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from dcs import schema
from dcs.config import ConfigError
from dcs.gold import GoldDataError, read_accepted, read_processed
from dcs.synthetic import SynthResult
from dcs.workbook import HAS_NTT, NTT_COLUMNS, read_ntt
from tests.dcs.gold_helpers import to_processed

Change = Callable[[pd.DataFrame], pd.DataFrame]


@pytest.fixture(scope="module")
def videos(small: SynthResult) -> pd.DataFrame:
    return read_accepted(small.accepted_dir).videos


@pytest.fixture(scope="module")
def clean(small: SynthResult, videos: pd.DataFrame) -> pd.DataFrame:
    return read_ntt(small.workbook_path, videos)


def raw(result: SynthResult) -> pd.DataFrame:
    """The synthetic workbook as written, real headers and all."""
    return pd.read_excel(result.workbook_path, sheet_name=schema.WORKBOOK_SHEET)


def rewritten(result: SynthResult, tmp_path: Path, change: Change) -> Path:
    path = tmp_path / "changed.xlsx"
    change(raw(result)).to_excel(path, sheet_name=schema.WORKBOOK_SHEET, index=False)
    return path


def row_of(table: pd.DataFrame, video_id: str) -> pd.Series:
    sex, subject = video_id.split("_")
    match = table[(table[schema.WB_SEX] == sex) & (table[schema.WB_SUBJECT] == int(subject))]
    assert len(match) == 1
    return match.iloc[0]


def is_row(table: pd.DataFrame, video_id: str) -> pd.Series:
    sex, subject = video_id.split("_")
    return (table[schema.WB_SEX] == sex) & (table[schema.WB_SUBJECT] == int(subject))


def complete_fish(clean: pd.DataFrame) -> str:
    return str(clean.loc[clean[HAS_NTT] == 1, "video_id"].iloc[0])


# --- a clean workbook ---------------------------------------------------------------------


def test_one_row_per_gold_fish_in_gold_order(clean: pd.DataFrame, videos: pd.DataFrame) -> None:
    assert list(clean.columns) == ["video_id", *NTT_COLUMNS, HAS_NTT]
    assert list(clean["video_id"]) == list(videos["video_id"])


def test_values_come_from_the_fish_own_row(small: SynthResult, clean: pd.DataFrame) -> None:
    table = raw(small)
    for record in clean.to_dict("records"):
        source = row_of(table, record["video_id"])
        expected = [source[header] for header in schema.NTT_HEADERS]
        np.testing.assert_allclose([record[name] for name in NTT_COLUMNS], expected, equal_nan=True)


def test_has_ntt_is_one_only_when_all_eight_are_present(small: SynthResult, clean: pd.DataFrame) -> None:
    table = raw(small)
    expected = [int(row_of(table, video_id)[list(schema.NTT_HEADERS)].notna().all()) for video_id in clean["video_id"]]
    assert list(clean[HAS_NTT]) == expected
    assert set(expected) == {0, 1}  # SMALL leaves NTT out for some fish (EC-2)


def test_partly_missing_ntt_keeps_the_values_but_flags_the_fish(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    fish = complete_fish(clean)

    def blank_one(table: pd.DataFrame) -> pd.DataFrame:
        table.loc[is_row(table, fish), schema.NTT_HEADERS[0]] = np.nan
        return table

    result = read_ntt(rewritten(small, tmp_path, blank_one), read_accepted(small.accepted_dir).videos).set_index("video_id")
    assert result.loc[fish, HAS_NTT] == 0
    assert np.isnan(result.loc[fish, NTT_COLUMNS[0]])
    assert result.loc[fish, NTT_COLUMNS[1:]].notna().all()


def test_dash_means_never_in_that_half(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    """The real workbook writes "-" for a half the fish never entered (its time there is 0):
    distance 0, speed undefined, and the NTT row still counts as measured."""
    fish = complete_fish(clean)
    tdm_top, velocity_top, time_top = schema.NTT_HEADERS[1], schema.NTT_HEADERS[4], schema.NTT_HEADERS[6]

    def never_top(table: pd.DataFrame) -> pd.DataFrame:
        table[[tdm_top, velocity_top]] = table[[tdm_top, velocity_top]].astype(object)
        table.loc[is_row(table, fish), [tdm_top, velocity_top]] = " - "
        table.loc[is_row(table, fish), time_top] = 0.0
        return table

    result = read_ntt(rewritten(small, tmp_path, never_top), read_accepted(small.accepted_dir).videos).set_index("video_id")
    assert result.loc[fish, "tdm_top"] == 0.0
    assert np.isnan(result.loc[fish, "velocity_top"])
    assert result.loc[fish, HAS_NTT] == 1


@pytest.mark.parametrize(("header", "text"), [(schema.NTT_HEADERS[1], "absent"), (schema.NTT_HEADERS[0], "-")])
def test_other_text_in_an_ntt_cell_is_an_error(small: SynthResult, clean: pd.DataFrame, tmp_path: Path, header: str, text: str) -> None:
    fish = complete_fish(clean)

    def write_text(table: pd.DataFrame) -> pd.DataFrame:
        table[header] = table[header].astype(object)
        table.loc[is_row(table, fish), header] = text
        return table

    with pytest.raises(GoldDataError, match=header.split()[0]):
        read_ntt(rewritten(small, tmp_path, write_text), read_accepted(small.accepted_dir).videos)


def test_no_workbook_gives_every_fish_has_ntt_zero(videos: pd.DataFrame) -> None:
    result = read_ntt(None, videos)
    assert list(result["video_id"]) == list(videos["video_id"])
    assert result[list(NTT_COLUMNS)].isna().all().all()
    assert (result[HAS_NTT] == 0).all()


def test_processed_source_dates_match_too(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    processed = read_processed(to_processed(small, tmp_path / "out")).videos
    result = read_ntt(small.workbook_path, processed)
    pd.testing.assert_frame_equal(
        result.set_index("video_id").sort_index(), clean.set_index("video_id").sort_index()
    )


# --- headers and keys -------------------------------------------------------------------------


def test_header_whitespace_is_ignored(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    def pad(table: pd.DataFrame) -> pd.DataFrame:
        return table.rename(columns=lambda header: "  " + header.replace(" ", " \n ") + "\t")  # no \ in an f-string field: CI runs 3.11

    pd.testing.assert_frame_equal(read_ntt(rewritten(small, tmp_path, pad), read_accepted(small.accepted_dir).videos), clean)


def test_key_is_sex_and_four_digit_subject(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    def loosen(table: pd.DataFrame) -> pd.DataFrame:
        table[schema.WB_SEX] = " " + table[schema.WB_SEX].str.lower() + " "
        table[schema.WB_SUBJECT] = table[schema.WB_SUBJECT].astype(float)
        return table

    pd.testing.assert_frame_equal(read_ntt(rewritten(small, tmp_path, loosen), read_accepted(small.accepted_dir).videos), clean)


def test_non_whole_subject_is_an_error(small: SynthResult, tmp_path: Path) -> None:
    def fraction(table: pd.DataFrame) -> pd.DataFrame:
        table[schema.WB_SUBJECT] = table[schema.WB_SUBJECT].astype(float)
        table.loc[0, schema.WB_SUBJECT] = 7.5
        return table

    with pytest.raises(GoldDataError, match="7.5"):
        read_ntt(rewritten(small, tmp_path, fraction), read_accepted(small.accepted_dir).videos)


def test_blank_sex_on_a_trial_row_is_an_error(small: SynthResult, tmp_path: Path) -> None:
    def no_sex(table: pd.DataFrame) -> pd.DataFrame:
        table.loc[0, schema.WB_SEX] = np.nan
        return table

    with pytest.raises(GoldDataError, match="blank"):
        read_ntt(rewritten(small, tmp_path, no_sex), read_accepted(small.accepted_dir).videos)


def test_rows_without_a_compound_are_not_trials(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    def add_note(table: pd.DataFrame) -> pd.DataFrame:
        note = pd.DataFrame([{schema.WB_STRAIN: "note row"}], columns=table.columns)
        return pd.concat([table, note], ignore_index=True)

    pd.testing.assert_frame_equal(read_ntt(rewritten(small, tmp_path, add_note), read_accepted(small.accepted_dir).videos), clean)


@pytest.mark.parametrize("header", [schema.WB_SUBJECT, schema.WB_DATE, schema.NTT_HEADERS[3]])
def test_missing_header_is_an_error_naming_it(small: SynthResult, tmp_path: Path, header: str) -> None:
    path = rewritten(small, tmp_path, lambda table: table.drop(columns=[header]))
    with pytest.raises(GoldDataError, match=header.split()[0]):
        read_ntt(path, read_accepted(small.accepted_dir).videos)


def test_missing_file_or_sheet_is_a_config_error(small: SynthResult, videos: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="DCS_DB_PATH"):
        read_ntt(tmp_path / "nope.xlsx", videos)
    other_sheet = tmp_path / "other.xlsx"
    raw(small).to_excel(other_sheet, sheet_name="Other", index=False)
    with pytest.raises(ConfigError, match=schema.WORKBOOK_SHEET):
        read_ntt(other_sheet, videos)


# --- duplicates and disagreements (EC-28) -------------------------------------------------------


def test_exact_duplicate_rows_collapse(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    path = rewritten(small, tmp_path, lambda table: pd.concat([table, table.iloc[[0]]], ignore_index=True))
    pd.testing.assert_frame_equal(read_ntt(path, read_accepted(small.accepted_dir).videos), clean)


def test_conflicting_duplicate_rows_are_an_error(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    fish = complete_fish(clean)

    def conflict(table: pd.DataFrame) -> pd.DataFrame:
        copy = table[is_row(table, fish)].copy()
        copy[schema.NTT_HEADERS[0]] += 1.0
        return pd.concat([table, copy], ignore_index=True)

    with pytest.raises(GoldDataError, match=fish):
        read_ntt(rewritten(small, tmp_path, conflict), read_accepted(small.accepted_dir).videos)


def test_compound_disagreeing_with_the_index_is_an_error(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    fish = complete_fish(clean)

    def rename(table: pd.DataFrame) -> pd.DataFrame:
        table.loc[is_row(table, fish), schema.WB_COMPOUND] = "COMPOUND_Z"
        return table

    with pytest.raises(GoldDataError, match=f"{fish}.*compound"):
        read_ntt(rewritten(small, tmp_path, rename), read_accepted(small.accepted_dir).videos)


def test_date_disagreeing_with_the_index_is_an_error(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    fish = complete_fish(clean)

    def shift(table: pd.DataFrame) -> pd.DataFrame:
        table.loc[is_row(table, fish), schema.WB_DATE] = 991231.0
        return table

    with pytest.raises(GoldDataError, match=f"{fish}.*date"):
        read_ntt(rewritten(small, tmp_path, shift), read_accepted(small.accepted_dir).videos)


def test_fish_absent_from_the_workbook_gets_has_ntt_zero(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    fish = complete_fish(clean)
    path = rewritten(small, tmp_path, lambda table: table[~is_row(table, fish)])
    result = read_ntt(path, read_accepted(small.accepted_dir).videos)
    assert list(result["video_id"]) == list(clean["video_id"])
    gone = result["video_id"] == fish
    assert result.loc[gone, HAS_NTT].item() == 0
    assert result.loc[gone, list(NTT_COLUMNS)].isna().all().all()
    pd.testing.assert_frame_equal(result[~gone], clean[~gone])


def test_workbook_fish_without_a_video_are_ignored(small: SynthResult, clean: pd.DataFrame, tmp_path: Path) -> None:
    def add_trial(table: pd.DataFrame) -> pd.DataFrame:
        extra = table.iloc[[0]].copy()
        extra[schema.WB_SUBJECT] = 9999
        return pd.concat([table, extra], ignore_index=True)

    pd.testing.assert_frame_equal(read_ntt(rewritten(small, tmp_path, add_trial), read_accepted(small.accepted_dir).videos), clean)
