"""Data audit (plan U8, T1.16): AC-1 tables, EC-11, EC-24, EC-26 (fps), EC-31, the G1 stop rule, D-016."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from dcs.audit import GO, WAIT, Audit, build_audit, render
from dcs.cli import main
from dcs.config import ENV_ACCEPTED_DIR, ENV_DB_PATH, ENV_OUTPUT_DIR, ENV_TABLE, load_settings
from dcs.featurize import COLUMNS
from dcs.synthetic import SynthResult
from dcs.trainset import DROP_FEW_FISH, DROP_SINGLE_DOSE

FEATURES = [column.name for column in COLUMNS if column.role == "feature"]


@pytest.fixture(scope="module")
def defaults() -> dict[str, Any]:
    return dict(load_settings().training)


def table(*cells: tuple[str, str, str, int], seed: int = 0) -> pd.DataFrame:
    """(compound, dose, date, fish) cells -> a training table: features vary, every fish shows every state,
    every fish Accepted, nothing flagged, the same tank framing on every date."""
    rows = [(compound, dose, date) for compound, dose, date, n in cells for _ in range(n)]
    rng = np.random.default_rng(seed)
    n = len(rows)
    data: dict[str, Any] = {column.name: np.ones(n) for column in COLUMNS}
    data.update({name: rng.uniform(1, 2, n) for name in FEATURES})
    data.update(
        video_id=[f"F_{i:04d}" for i in range(n)],
        subject_id=[f"{i:04d}" for i in range(n)],
        compound=[compound for compound, _, _ in rows],
        concentration_mM=[dose for _, dose, _ in rows],
        date=[date for _, _, date in rows],
        sex="F",
        strain="S1",
        has_ntt=1,
        reviewed=True,
        review_status="ACCEPTED",
        manual_share=0.0,
        undetermined_share=0.0,
        flag_low_detected=0,
        flag_odd_duration=0,
        detected_share=0.95,
        video_duration_s=20.0,
        agent_exposure_min=30.0,
        depth_p01=rng.uniform(50, 52, n),
        depth_p99=rng.uniform(448, 450, n),
    )
    return pd.DataFrame(data, columns=[column.name for column in COLUMNS])


def described(**changes: Any) -> dict[str, Any]:
    """The schema JSON featurize writes, for an Accepted set read without trouble."""
    base = {
        "format": 1,
        "gold_source": "accepted",
        "profile": "cal-synthetic",
        "tracker": "classical",
        "fps_uniform": True,
        "fps_range": [10.0, 10.0],
        "tracker_checked": 0,
        "ntt_available": True,
        "flags": {"min_detected_fraction": 0.8, "duration_range_s": None},
        "dropped": [],
        "columns": [column.as_json() for column in COLUMNS],
    }
    return {**base, **changes}


# Vehicle on every date; A, B, C on two dates each with 6 fish; D on one date.
DATES = ["<d1>", "<d2>", "<d3>", "<d4>"]
BASE = (
    *(("VEHICLE", "0", date, 2) for date in DATES),
    ("COMPOUND_A", "0.1", "<d1>", 3), ("COMPOUND_A", "0.3", "<d2>", 3),
    ("COMPOUND_B", "0.1", "<d2>", 3), ("COMPOUND_B", "0.3", "<d3>", 3),
    ("COMPOUND_C", "0.1", "<d3>", 3), ("COMPOUND_C", "0.3", "<d4>", 3),
    ("COMPOUND_D", "1", "<d4>", 6),
)  # fmt: skip


def audit(frame: pd.DataFrame, training: dict[str, Any], schema: dict[str, Any] | None = None, **changes: Any) -> Audit:
    return build_audit(frame, schema or described(), {**training, "min_state_fish": 1, **changes})


@pytest.fixture(scope="module")
def base(defaults: dict[str, Any]) -> Audit:
    return audit(table(*BASE), defaults)


# --- filter steps, classes, dates --------------------------------------------------------------


def test_filter_steps_count_every_drop_in_order(defaults: dict[str, Any]) -> None:
    dropped = [
        {"video_id": "F_0901", "reason": "missing_file", "detail": "x"},
        {"video_id": "F_0902", "reason": "missing_file", "detail": "x"},
        {"video_id": "F_0903", "reason": "other_profile", "detail": "profile cal-2"},
    ]
    frame = table(*BASE, ("COMPOUND_E", "1", "<d1>", 2))
    steps = audit(frame, defaults, described(dropped=dropped)).tables["steps"].set_index("step")["fish"]
    assert steps["dropped before the table: missing_file"] == 2
    assert steps["dropped before the table: other_profile"] == 1
    assert steps["in the training table"] == len(frame) == 34
    assert steps[f"compound: dropped, {DROP_FEW_FISH}"] == 2  # COMPOUND_E
    assert steps["compound: kept"] == 32
    assert steps["compound: scored in scheme A"] == 26  # COMPOUND_D (one date) pinned
    assert list(steps.index).index("in the training table") < list(steps.index).index("compound: kept")


def test_class_table_shows_status_and_scheme_a(defaults: dict[str, Any]) -> None:
    result = audit(table(*BASE, ("COMPOUND_E", "1", "<d1>", 2)), defaults)
    classes = result.tables["classes_compound"].set_index("label")
    assert classes.loc["VEHICLE", "fish"] == 8 and classes.loc["VEHICLE", "dates"] == 4
    assert classes.loc["COMPOUND_A", "status"] == "very small"  # exactly min_class_size
    assert classes.loc["VEHICLE", "status"] == "kept"
    assert classes.loc["COMPOUND_E", "status"] == DROP_FEW_FISH
    assert classes.loc["COMPOUND_D", "scheme_a"] == "pinned"
    assert classes.loc["COMPOUND_A", "scheme_a"] == "scored"
    assert classes.loc["COMPOUND_E", "scheme_a"] == ""


def test_dose_stage_lists_single_dose_drops_and_skipped_scheme_a(base: Audit) -> None:
    doses = base.tables["classes_dose"].set_index("label")
    assert doses.loc["VEHICLE @ 0", "status"] == DROP_SINGLE_DOSE
    assert doses.loc["COMPOUND_A @ 0.1", "status"] == DROP_FEW_FISH  # 3 fish < 6
    assert base.facts["dose: classes kept"] == 0
    assert any(note.startswith("dose stage cannot be built") for note in base.notes)  # no crash


def test_per_date_and_compound_by_date_tables(base: Audit) -> None:
    dates = base.tables["dates"].set_index("date")
    assert dates.loc["<d2>", "fish"] == 2 + 3 + 3
    assert dates.loc["<d2>", "compounds"] == 3 and dates.loc["<d2>", "vehicle"] == 2
    cross = base.tables["compound_by_date"]
    assert cross.loc["COMPOUND_D", "<d4>"] == 6 and cross.loc["COMPOUND_D", "<d1>"] == 0
    assert cross.to_numpy().sum() == 32


def test_missing_values_counted_per_column(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    frame.loc[frame.index[:3], "tdm_full"] = np.nan
    missing = audit(frame, defaults).tables["missing"].set_index("column")
    assert missing.loc["tdm_full", "missing"] == 3
    assert missing.loc["tdm_full", "role"] == "feature"
    assert "velocity_mean" not in missing.index


# --- labeling process, flags (EC-11) -----------------------------------------------------------


def test_compound_table_reports_manual_share_flags_and_protocol(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    a = frame["compound"] == "COMPOUND_A"
    frame.loc[a, "manual_share"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.2]
    frame.loc[frame.index[a][:2], "flag_low_detected"] = 1
    frame.loc[a, "agent_exposure_min"] = 60.0
    compounds = audit(frame, defaults).tables["compounds"].set_index("compound")
    assert compounds.loc["COMPOUND_A", "manual_share"] == pytest.approx(0.2)
    assert compounds.loc["VEHICLE", "manual_share"] == 0
    assert compounds.loc["COMPOUND_A", "low_detected"] == 2
    assert compounds.loc["COMPOUND_A", "exposure_min"] == "60"
    assert compounds.loc["COMPOUND_A", "doses"] == 2


def test_flagged_fish_are_listed_and_counted(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    frame.loc[[0, 5], "flag_low_detected"] = 1
    frame.loc[[0, 1], "detected_share"] = 0.5
    frame.loc[7, "flag_odd_duration"] = 1
    result = audit(frame, defaults)
    flagged = result.tables["flagged"].set_index("video_id")
    assert set(flagged.index) == {"F_0000", "F_0005", "F_0007"}
    assert flagged.loc["F_0000", "detected_share"] == 0.5
    assert result.facts["fish flagged low detected share"] == 2
    assert result.facts["fish flagged odd duration"] == "not checked (training.duration_range_s is null)"


# --- confound statistics (PRD §2.3), EC-24, D-016 -----------------------------------------------


def test_confound_statistics(base: Audit) -> None:
    facts = base.facts
    assert facts["fish in the table"] == 32 and facts["Accepted fish"] == 32
    assert facts["dates"] == 4 and facts["fish per date"] == 8
    assert facts["compounds (with vehicle)"] == 5
    assert facts["vehicle fish"] == 8 and facts["vehicle dates"] == 4
    assert facts["compound+dose classes"] == 8
    assert facts["compound+dose classes on one date"] == 7  # all but VEHICLE @ 0
    assert facts["compound+dose classes below min_class_size"] == 6  # the 3-fish doses
    assert facts["compounds on one date"] == ["COMPOUND_D"]
    assert facts["dates where one compound has 2+ doses"] == 0
    assert facts["fish with NTT"] == 32


def test_dose_spelling_variants_count_as_one_dose(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    d = frame.index[frame["compound"] == "COMPOUND_D"]
    frame.loc[d[:3], "concentration_mM"] = " 1 "  # the training set reads it as "1" (EC-7)
    compounds = audit(frame, defaults).tables["compounds"].set_index("compound")
    assert compounds.loc["COMPOUND_D", "doses"] == 1


def test_doses_sharing_a_date_are_counted(defaults: dict[str, Any]) -> None:
    result = audit(table(*BASE, ("COMPOUND_A", "1", "<d1>", 2)), defaults)
    assert result.facts["dates where one compound has 2+ doses"] == 1


def test_compound_with_no_accepted_fish_shows_zero_ec24(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    frame.loc[frame["compound"] == "COMPOUND_B", "reviewed"] = False
    compounds = audit(frame, defaults).tables["compounds"].set_index("compound")
    assert compounds.loc["COMPOUND_B", "fish"] == 6 and compounds.loc["COMPOUND_B", "accepted"] == 0
    assert compounds.loc["COMPOUND_B", "accepted_dates"] == 0
    assert compounds.loc["COMPOUND_D", "dates"] == 1  # concentrated on one date


def test_share_of_fish_on_dates_with_two_classes_d016(base: Audit) -> None:
    # every date holds vehicle plus at least one compound
    assert base.facts["compound: share of fish on dates with 2+ classes"] == 1.0


def test_share_counts_single_class_dates_out(defaults: dict[str, Any]) -> None:
    frame = table(*BASE, ("VEHICLE", "0", "<d5>", 4))
    assert audit(frame, defaults).facts["compound: share of fish on dates with 2+ classes"] == pytest.approx(32 / 36)


# --- G1 stop rule (PRD §10) ---------------------------------------------------------------------


def test_stop_rule_go_with_three_compounds_on_two_dates(base: Audit) -> None:
    assert base.facts["G1 stop rule"] == GO
    assert base.facts["G1 compounds with enough Accepted fish on 2+ dates"] == ["COMPOUND_A", "COMPOUND_B", "COMPOUND_C"]


def test_stop_rule_ignores_vehicle_single_date_and_unaccepted(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    frame.loc[frame["compound"] == "COMPOUND_C", "reviewed"] = False
    result = audit(frame, defaults)
    assert result.facts["G1 stop rule"] == WAIT  # vehicle and one-date COMPOUND_D never count
    assert result.facts["G1 compounds with enough Accepted fish on 2+ dates"] == ["COMPOUND_A", "COMPOUND_B"]
    assert any("G1 stop rule: WAIT" in note for note in result.notes)


def test_stop_rule_on_unreviewed_source_also_counts_every_fish(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    frame["reviewed"] = False
    result = audit(frame, defaults, described(gold_source="processed"))
    assert result.facts["G1 stop rule"] == WAIT
    assert result.facts["G1 stop rule counting unreviewed fish (Q17)"] == GO
    assert any("UNREVIEWED" in note for note in result.notes)


def test_vehicle_name_comes_from_the_settings(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    frame["compound"] = frame["compound"].replace("VEHICLE", "control_x ")
    result = audit(frame, defaults, vehicle_compound="CONTROL_X")
    assert result.facts["vehicle fish"] == 8
    assert "CONTROL_X" not in result.facts["G1 compounds with enough Accepted fish on 2+ dates"]


def test_missing_vehicle_is_a_note(defaults: dict[str, Any]) -> None:
    result = audit(table(*BASE), defaults, vehicle_compound="OTHER")
    assert result.facts["vehicle fish"] == 0
    assert any("training.vehicle_compound" in note for note in result.notes)


# --- recording setup (EC-26, tracker), evaluable classes ---------------------------------------


def test_uniform_fps_and_unrecorded_resolution(base: Audit) -> None:
    assert base.facts["frame rate uniform"] is True
    assert base.facts["resolution"] == "not recorded upstream"
    assert not any("EC-26" in note for note in base.notes)


def test_mixed_fps_flags_pixel_features_ec26(defaults: dict[str, Any]) -> None:
    result = audit(table(*BASE), defaults, described(fps_uniform=False, fps_range=[10.0, 25.0]))
    assert result.facts["frame rate uniform"] is False
    assert any("EC-26" in note and "pixel" in note for note in result.notes)


def test_tracker_not_cross_checked_is_reported(base: Audit, defaults: dict[str, Any]) -> None:
    assert base.facts["tracker cross-checked"] == "0 of 32 fish"
    assert any("tracker" in note and "32 fish" in note for note in base.notes)
    checked = audit(table(*BASE), defaults, described(tracker_checked=32))
    assert not any("cross-check" in note for note in checked.notes)


def test_evaluable_classes_and_fold_notes(base: Audit) -> None:
    assert base.facts["compound: folds (A, B)"] == (4, 5)  # 4 dates to hold out; largest class 8
    assert base.facts["compound: date-confounded"] == ["COMPOUND_D"]
    assert any(note.startswith("compound: scheme A: only") for note in base.notes)


def test_states_table_shows_support_and_drops(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    frame["state_dead_bouts"] = 0.0
    frame.loc[:2, "state_dead_bouts"] = 1.0
    states = audit(frame, defaults, min_state_fish=5).tables["states"].set_index("state")
    assert states.loc["Dead", "fish"] == 3
    assert states.loc["Dead", "dropped in"] == "compound"
    assert states.loc["Controlled Swim", "fish"] == 32 and states.loc["Controlled Swim", "dropped in"] == ""


# --- camera framing (EC-31) ---------------------------------------------------------------------
# table() gives every date the same framing: top ~51 px, bottom ~449 px, height ~398, so the tolerance is ~40 px.


def shifted(frame: pd.DataFrame, dates: list[str], px: float) -> pd.DataFrame:
    """The tank sits `px` lower in the frame on `dates`."""
    frame.loc[frame["date"].isin(dates), ["depth_p01", "depth_p99"]] += px
    return frame


def framing_of(result: Audit) -> pd.DataFrame:
    return result.tables["framing"].set_index("date")


def test_same_framing_on_every_date_is_one_setup(base: Audit) -> None:
    framing = framing_of(base)
    assert len(framing) == 4 and set(framing["setup"]) == {1}
    assert base.facts["framing differs between dates"] is False
    assert base.facts["framing setups"] == 1
    assert not any("EC-31" in note for note in base.notes)


def test_shifted_framing_is_its_own_setup_with_a_recommendation_ec31(defaults: dict[str, Any]) -> None:
    result = audit(shifted(table(*BASE), ["<d3>"], 120), defaults)
    framing = framing_of(result)
    assert framing.loc["<d3>", "setup"] == 2 and set(framing.drop("<d3>")["setup"]) == {1}
    assert framing.loc["<d3>", "top"] - framing.loc["<d1>", "top"] == pytest.approx(120, abs=2)
    assert result.facts["framing differs between dates"] is True and result.facts["framing setups"] == 2
    assert any("2 setups" in note and "use_depth" in note and "pixel speeds" in note for note in result.notes)


def test_two_equally_common_setups_are_told_apart(defaults: dict[str, Any]) -> None:
    # 60 px = 15 % of the height: the median date falls between the two setups, 30 px from each
    result = audit(shifted(table(*BASE), ["<d3>", "<d4>"], 60), defaults)
    framing = framing_of(result)
    assert framing.loc["<d1>", "setup"] == framing.loc["<d2>", "setup"] == 1
    assert framing.loc["<d3>", "setup"] == framing.loc["<d4>", "setup"] == 2
    assert result.facts["framing differs between dates"] is True


def test_framing_drifting_in_small_steps_still_differs(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    for step, date in enumerate(DATES):
        shifted(frame, [date], 25 * step)  # 25 px a date, under the tolerance; 75 px from first to last
    result = audit(frame, defaults)
    assert result.facts["framing setups"] == 1  # each date is within reach of the next
    assert result.facts["framing differs between dates"] is True
    assert any("drifts" in note and "use_depth" in note for note in result.notes)


def test_one_fish_at_the_surface_does_not_move_its_date(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    frame.loc[0, "depth_p01"] = 0.0  # one fish hugs the surface; the camera did not move
    framing = framing_of(audit(frame, defaults))
    assert framing.loc["<d1>", "top_min"] == 0.0 and set(framing["setup"]) == {1}


def test_changed_framing_range_is_its_own_setup_ec31(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    zoomed = frame["date"] == "<d2>"
    frame.loc[zoomed, "depth_p99"] = frame.loc[zoomed, "depth_p99"] * 1.5  # zoomed in: the tank looks taller
    framing = framing_of(audit(frame, defaults))
    assert framing.loc["<d2>", "setup"] == 2 and framing.loc["<d1>", "setup"] == 1


def test_a_date_without_depth_data_is_listed_and_noted(defaults: dict[str, Any]) -> None:
    frame = table(*BASE)
    frame.loc[frame["date"] == "<d4>", ["depth_p01", "depth_p99"]] = np.nan
    result = audit(frame, defaults)
    framing = framing_of(result)
    assert len(framing) == 4 and framing.loc["<d4>", "fish"] == 0 and pd.isna(framing.loc["<d4>", "setup"])
    assert result.facts["dates without depth data"] == 1
    assert result.facts["framing differs between dates"] is False  # among the dates it could check
    assert any("no depth data" in note and "<d4>" in note for note in result.notes)
    assert "| <d4> | 0 | - |" in render(result)


# --- report and command -------------------------------------------------------------------------


def test_render_has_every_section_and_the_verdict(base: Audit) -> None:
    text = render(base)
    for heading in ("# Data audit", "## Notes", "## Filter steps", "## Compounds", "## Classes: compound",
                    "## Classes: dose", "## Dates", "## Compound by date", "## Missing values", "## Flagged fish",
                    "## States", "## Camera framing"):  # fmt: skip
        assert heading in text
    assert "G1 stop rule: GO" in text
    assert "| COMPOUND_A |" in text


def test_cli_audit_end_to_end_on_synthetic_data(
    small: SynthResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    override = tmp_path / "accepted.yaml"
    override.write_text(yaml.safe_dump({"training": {"gold_source": "accepted", "min_class_size": 2}}), encoding="utf-8")
    monkeypatch.setenv(ENV_ACCEPTED_DIR, str(small.accepted_dir))
    monkeypatch.setenv(ENV_DB_PATH, str(small.workbook_path))
    monkeypatch.setenv(ENV_OUTPUT_DIR, str(tmp_path / "out"))
    assert main(["--config", str(override), "featurize"]) == 0
    assert main(["--config", str(override), "audit"]) == 0
    report = (tmp_path / "out" / "audit.md").read_text(encoding="utf-8")
    assert "# Data audit" in report and "VEHICLE" in report
    out = capsys.readouterr().out
    assert "G1 stop rule" in out and str(tmp_path / "out" / "audit.md") in out


def test_cli_audit_without_a_table_says_to_run_featurize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_TABLE, str(tmp_path / "nowhere.parquet"))
    assert main(["audit"]) == 2
    out = capsys.readouterr().out
    assert "Configuration error" in out and "featurize" in out
