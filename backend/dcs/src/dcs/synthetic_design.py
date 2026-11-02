"""The experiment design of the synthetic gold dataset (plan U2): which fish exist, and which knob hits which.

Mimics the structure PRD §2.3 describes: vehicle fish on every date, each
main compound+dose on two dates, doses of one compound never on the same
date. Knob groups are appended after the main design and vehicle fish, so
switching them on never renumbers (or re-randomizes) the other fish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dcs import schema
from dcs.synthetic_config import (
    COMPOUND_LETTERS,
    DATES_PER_DOSE,
    DOSES,
    KNOB_LOW_DETECTED_FISH,
    KNOB_MESSY_LABELS,
    KNOB_MIXED_PROFILES,
    KNOB_RARE_STATE_FISH,
    KNOB_SINGLE_DATE_COMPOUND,
    KNOB_TINY_CLASS,
    SINGLE_DATE_FISH,
    TINY_CLASS_FISH,
    SynthConfig,
)

VEHICLE = "VEHICLE"
VEHICLE_DOSE = "0"
SINGLE_DATE_COMPOUND = "COMPOUND_S"
TINY_COMPOUND = "COMPOUND_T"
COMBO_COMPOUND = "COMPOUND_A + COMPOUND_B"
COMBO_DOSES = ("0.03 + 0.01", "0.03+0.01")  # the same dose spelled two ways (EC-7)

# Design groups (FishPlan.group).
GROUP_MAIN = "main"
GROUP_VEHICLE = "vehicle"
GROUP_SINGLE_DATE = "single_date"
GROUP_TINY = "tiny"
GROUP_COMBO = "combo"


@dataclass(frozen=True)
class FishPlan:
    number: int  # subject number, also the key of the fish's random stream
    compound: str  # canonical label
    dose: str  # canonical label
    dose_level: int
    effect_index: int | None  # None for vehicle
    date_index: int
    group: str  # one of the GROUP_* names

    @property
    def sex(self) -> str:
        return "F" if self.number % 2 else "M"

    @property
    def subject_id(self) -> str:
        return f"{self.number:04d}"

    @property
    def video_id(self) -> str:
        return schema.video_id(self.sex, self.subject_id)


def design(config: SynthConfig) -> list[FishPlan]:
    """Every fish, in subject-number order; knob groups come last so they never renumber the main design."""
    rows: list[tuple[str, str, int, int | None, int, str]] = []
    for k in range(config.n_compounds):
        for level in range(config.doses_per_compound):
            for date_index in dose_dates(k, level, config):
                cell = (f"COMPOUND_{COMPOUND_LETTERS[k]}", DOSES[level], level, k, date_index, GROUP_MAIN)
                rows += [cell] * config.fish_per_dose_date
    for date_index in range(config.n_dates):
        rows += [(VEHICLE, VEHICLE_DOSE, 0, None, date_index, GROUP_VEHICLE)] * config.vehicle_per_date
    extra = config.n_compounds  # effect indices after the main compounds
    if config.single_date_compound:
        rows += [(SINGLE_DATE_COMPOUND, DOSES[0], 0, extra, 0, GROUP_SINGLE_DATE)] * SINGLE_DATE_FISH
    if config.tiny_class:
        rows += [(TINY_COMPOUND, DOSES[0], 0, extra + 1, i % 2, GROUP_TINY) for i in range(TINY_CLASS_FISH)]
    if config.messy_labels:
        combo_fish = DATES_PER_DOSE * config.fish_per_dose_date
        rows += [(COMBO_COMPOUND, COMBO_DOSES[0], 0, extra + 2, i % 2, GROUP_COMBO) for i in range(combo_fish)]
    return [FishPlan(number, *row) for number, row in enumerate(rows, start=1)]


def knob_targets(plans: Sequence[FishPlan], config: SynthConfig) -> dict[str, tuple[str, ...]]:
    """Which fish each active knob changes. Single-fish knobs never share a fish (see synthetic_config)."""
    main = [plan.video_id for plan in plans if plan.group == GROUP_MAIN]
    # Each active head knob takes the next main fish (SynthConfig checks there are enough).
    targets: dict[str, tuple[str, ...]] = {
        knob: (main[position],) for position, knob in enumerate(config.active_head_knobs)
    }
    # The tail: the last main fish, low-detected first; empty when both counts are 0.
    tail = main[len(main) - config.low_detected_fish - config.rare_state_fish :]
    if config.low_detected_fish:
        targets[KNOB_LOW_DETECTED_FISH] = tuple(tail[: config.low_detected_fish])
    if config.rare_state_fish:
        targets[KNOB_RARE_STATE_FISH] = tuple(tail[config.low_detected_fish :])
    if config.messy_labels:
        vehicles = [plan.video_id for plan in plans if plan.group == GROUP_VEHICLE][::2]
        targets[KNOB_MESSY_LABELS] = (*vehicles, *(plan.video_id for plan in plans if plan.group == GROUP_COMBO))
    if config.mixed_profiles:
        targets[KNOB_MIXED_PROFILES] = tuple(plan.video_id for plan in plans if plan.date_index == config.n_dates - 1)
    for knob, group in ((KNOB_SINGLE_DATE_COMPOUND, GROUP_SINGLE_DATE), (KNOB_TINY_CLASS, GROUP_TINY)):
        if getattr(config, knob):
            targets[knob] = tuple(plan.video_id for plan in plans if plan.group == group)
    return targets


def dose_dates(compound_index: int, level: int, config: SynthConfig) -> tuple[int, ...]:
    """DATES_PER_DOSE dates per compound+dose, spread over the calendar; with n_dates >= 2 x doses,
    doses of one compound never share a date."""
    start = (compound_index * config.doses_per_compound + level) % config.n_dates
    step = config.n_dates // DATES_PER_DOSE
    return tuple((start + i * step) % config.n_dates for i in range(DATES_PER_DOSE))
