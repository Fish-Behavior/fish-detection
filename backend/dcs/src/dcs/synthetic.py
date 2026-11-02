"""Synthetic accepted gold dataset plus trial workbook, in the files prepds writes (plan U2, D-029).

Only for building and testing dcs: every name is a placeholder (COMPOUND_A,
VEHICLE, STRAIN_1, reviewer_1) and dates start at 2000-01-01. The design
mimics the structure the PRD (§2.3) describes, not its numbers: vehicle fish
on every date, each compound+dose on its own dates (doses of one compound
never share a date), a compound effect, a date effect, and one knob per
edge case. `SynthResult.targets` names the fish each knob was applied to.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from dcs import schema
from dcs.synthetic_config import (
    KNOB_LOW_DETECTED_FISH,
    KNOB_MESSY_LABELS,
    KNOB_MISSING_FILE,
    KNOB_MISSING_FRAME_COLUMN,
    KNOB_MIXED_FPS,
    KNOB_MIXED_PROFILES,
    KNOB_NULL_DATE,
    KNOB_RARE_STATE_FISH,
    KNOB_UNDETERMINED,
    KNOB_WRONG_DTYPE_COLUMN,
    MISSING_FRAMES,
    MISSING_SEGMENTS,
    MIXED_FPS_FACTOR,
    TARGETED_KNOBS,
    SynthConfig,
)
from dcs.synthetic_design import COMBO_DOSES, GROUP_COMBO, VEHICLE, FishPlan, design, knob_targets
from dcs.synthetic_frames import FishBehavior, frames_to_segments, make_frames
from dcs.synthetic_workbook import WorkbookFish, workbook_record, write_workbook

# Re-exported for callers: dcs.synthetic is the one import for building a dataset.
__all__ = ["DEFAULT_PROFILE", "MODEL_PROFILE", "VEHICLE", "SynthConfig", "SynthResult", "design", "make_gold_dataset"]

ACCEPTED_DIR = "accepted"
WORKBOOK_FILE = "synthetic_db.xlsx"

DEFAULT_PROFILE = "cal-synthetic"
MODEL_PROFILE = "cal-synthetic-model"  # model-tracker profiles carry "-model" (D-003)
PIPELINE_VERSION = "synthetic-0.1"
REVIEWER = "reviewer_1"
STRAIN = "STRAIN_1"
FIRST_DATE = dt.date(2000, 1, 1)
REVIEWED_AT = dt.datetime(2000, 6, 1, 12, 0, tzinfo=dt.timezone.utc)  # fixed, so the content is reproducible
PROCESSED_AT = REVIEWED_AT - dt.timedelta(days=1)

# Compound speed effects (log scale), alternating faster/slower; a dose level scales them up.
EFFECT_OFFSETS = (0.6, -0.6, 0.35, -0.35, 0.8, -0.8, 0.5, -0.5, 0.25, -0.25, 0.45, -0.45, 0.7)
DOSE_STEP = 0.4
PREFERRED_STATES = (schema.ERRATIC_MOVEMENT, schema.FREEZING_DRIFT, schema.SURFACE_BREACH)
PREFERENCE_PER_EFFECT = 2.0
BASE_MISS_RATE = 0.02
LOW_DETECTED_MISS_RATE = 0.5
BASE_AGE = 6.0
AGE_SPREAD = 3
BASE_EXPOSURE_MIN = 20.0
EXPOSURE_STEP_MIN = 5.0
DATE_STREAM = 0  # random stream of the per-date effects; fish streams use their subject number (>= 1)
NTT_STREAM = 1


@dataclass(frozen=True)
class FishFacts:
    """Everything decided about one fish before its files are written."""

    plan: FishPlan
    knobs: frozenset[str]  # targeted knobs applied to this fish
    fps: float
    n_frames: int
    profile: str
    speed_multiplier: float
    compound_label: str  # as written to the files (a messy spelling for messy-label fish)
    dose_label: str

    def has(self, knob: str) -> bool:
        """True when `knob` was applied to this fish; an unknown knob name is a programming error."""
        return _applies(self.knobs, knob)


def _applies(knobs: frozenset[str], knob: str) -> bool:
    if knob not in TARGETED_KNOBS:
        raise KeyError(f"unknown knob {knob!r}")  # a typo must fail loudly, not silently do nothing
    return knob in knobs


@dataclass(frozen=True, eq=False)
class SynthResult:
    """Where the dataset was written and what it contains. Read-only: `targets` cannot be changed and
    `truth` is a fresh copy on every access (built only by make_gold_dataset)."""

    out_dir: Path
    accepted_dir: Path
    index_path: Path
    workbook_path: Path
    targets: Mapping[str, tuple[str, ...]]  # knob name -> video_ids it was applied to (read-only)
    _truth: pd.DataFrame = field(repr=False)

    @property
    def truth(self) -> pd.DataFrame:
        """One row per fish (canonical labels, date, group, fps, profile); a fresh copy on every access."""
        return self._truth.copy()


def make_gold_dataset(out_dir: Path, config: SynthConfig | None = None) -> SynthResult:
    """Write <out>/accepted/<video_id>/..., <out>/accepted/accepted_index.parquet and <out>/synthetic_db.xlsx.

    Everything is decided before the first file is written; `out_dir` must be new or an empty folder.
    """
    config = config if config is not None else SynthConfig()
    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise FileExistsError(f"{out_dir} is not a folder; give a new or empty folder")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; give a new or empty folder (nothing is overwritten)")
    plans = design(config)
    targets = knob_targets(plans, config)
    date_z = _date_effects(config)
    all_facts = [_facts(plan, config, targets, date_z) for plan in plans]

    accepted_dir = out_dir / ACCEPTED_DIR
    accepted_dir.mkdir(parents=True)
    index_rows = []
    for facts in all_facts:
        gold = accepted_dir / facts.plan.video_id
        source = _write_fish(gold, facts, config)
        index_rows.append(_index_row(gold, facts, source))
    index_path = accepted_dir / schema.INDEX_FILE
    pd.DataFrame(index_rows, columns=list(schema.INDEX_COLUMNS)).to_parquet(index_path, index=False)
    workbook_path = out_dir / WORKBOOK_FILE
    write_workbook(workbook_path, [_workbook_row(facts, config) for facts in all_facts])
    truth = pd.DataFrame([_truth_row(facts) for facts in all_facts])
    return SynthResult(out_dir, accepted_dir, index_path, workbook_path, MappingProxyType(targets), truth)


# --- per-fish decisions -------------------------------------------------------------------------


def _facts(plan: FishPlan, config: SynthConfig, targets: Mapping[str, tuple[str, ...]], date_z: np.ndarray) -> FishFacts:
    knobs = frozenset(knob for knob, ids in targets.items() if plan.video_id in ids)
    fps = config.fps * MIXED_FPS_FACTOR if _applies(knobs, KNOB_MIXED_FPS) else config.fps
    compound_label, dose_label = _labels(plan, knobs)
    return FishFacts(
        plan=plan,
        knobs=knobs,
        fps=fps,
        n_frames=max(1, int(round(config.duration_s * fps))),
        profile=MODEL_PROFILE if _applies(knobs, KNOB_MIXED_PROFILES) else DEFAULT_PROFILE,
        speed_multiplier=_speed_multiplier(plan, config, date_z),
        compound_label=compound_label,
        dose_label=dose_label,
    )


def _labels(plan: FishPlan, knobs: frozenset[str]) -> tuple[str, str]:
    """Compound and dose as written to the files: canonical, or a messy spelling (prepds strips spaces,
    so the realistic variants are case and dose spacing)."""
    if not _applies(knobs, KNOB_MESSY_LABELS):
        return plan.compound, plan.dose
    if plan.group == GROUP_COMBO:
        return plan.compound, COMBO_DOSES[plan.number % len(COMBO_DOSES)]
    return plan.compound.lower(), plan.dose


def _speed_multiplier(plan: FishPlan, config: SynthConfig, date_z: np.ndarray) -> float:
    compound = 0.0 if plan.effect_index is None else EFFECT_OFFSETS[plan.effect_index] * _effect(plan, config)
    return float(np.exp(compound + config.date_effect * date_z[plan.date_index]))


def _effect(plan: FishPlan, config: SynthConfig) -> float:
    """Strength of the compound effect for this fish's dose (0 for vehicle)."""
    return 0.0 if plan.effect_index is None else config.compound_effect * (1 + DOSE_STEP * plan.dose_level)


def _behavior(facts: FishFacts, config: SynthConfig) -> FishBehavior:
    plan = facts.plan
    preferred = None if plan.effect_index is None else PREFERRED_STATES[plan.effect_index % len(PREFERRED_STATES)]
    return FishBehavior(
        fps=facts.fps,
        n_frames=facts.n_frames,
        speed_multiplier=facts.speed_multiplier,
        preferred_state=preferred,
        preference_boost=PREFERENCE_PER_EFFECT * _effect(plan, config),
        depth_offset_px=config.framing_shift_px * plan.date_index,
        miss_rate=LOW_DETECTED_MISS_RATE if facts.has(KNOB_LOW_DETECTED_FISH) else BASE_MISS_RATE,
        manual_fraction=config.manual_fraction,
        rare_state=facts.has(KNOB_RARE_STATE_FISH),
        undetermined=facts.has(KNOB_UNDETERMINED),
        constant_meander=config.constant_meander,
    )


# --- files --------------------------------------------------------------------------------------


def _write_fish(gold: Path, facts: FishFacts, config: SynthConfig) -> str:
    """Write the fish's gold files; returns its provenance source (auto or manual)."""
    frames = make_frames(_behavior(facts, config), np.random.default_rng([config.seed, facts.plan.number]))
    segments = frames_to_segments(frames, facts.fps)
    edited = bool((frames["source"] == schema.SOURCE_MANUAL).any())
    if facts.has(KNOB_MISSING_FRAME_COLUMN):
        frames = frames.drop(columns=[config.missing_frame_column])
    if facts.has(KNOB_WRONG_DTYPE_COLUMN):
        frames = frames.assign(**{config.wrong_dtype_column: frames[config.wrong_dtype_column].astype(str)})

    gold.mkdir()
    missing = config.missing_file if facts.has(KNOB_MISSING_FILE) else None
    if missing != MISSING_FRAMES:
        frames.to_parquet(gold / schema.FRAMES_FILE, index=False)
    if missing != MISSING_SEGMENTS:
        segments.to_csv(gold / schema.SEGMENTS_FILE, index=False)
    source = schema.SOURCE_MANUAL if edited else schema.SOURCE_AUTO
    _write_json(gold / schema.MANIFEST_FILE, _manifest(facts, edited))
    _write_json(
        gold / schema.PROVENANCE_FILE,
        {
            "reviewer": REVIEWER,
            "reviewed_at": REVIEWED_AT.isoformat(),
            "source": source,
            "calibration_profile_version": facts.profile,
            "pipeline_version": PIPELINE_VERSION,
        },
    )
    return source


def _manifest(facts: FishFacts, edited: bool) -> dict[str, Any]:
    plan = facts.plan
    return {
        "subject_id": plan.subject_id,
        "sex": plan.sex,
        "compound": facts.compound_label,
        "concentration_mM": facts.dose_label,
        "video_path": _video_path(plan),
        "video_duration_s": facts.n_frames / facts.fps,
        "video_fps": facts.fps,
        "pipeline_version": PIPELINE_VERSION,
        "calibration_profile_version": facts.profile,
        "processed_at": PROCESSED_AT.isoformat(),
        "review_status": schema.REVIEW_ACCEPTED,
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT.isoformat(),
        "edited": edited,
        "edit_count": 1 if edited else 0,
        "review_flags": [],
    }


def _index_row(gold: Path, facts: FishFacts, source: str) -> dict[str, Any]:
    plan = facts.plan
    no_workbook = facts.has(KNOB_NULL_DATE)  # accepted in the review app, `prepds export-index` not run yet
    workbook_fields = {
        "strain": STRAIN,
        "age": _age(plan),
        "date": _date(plan.date_index).isoformat(),
        "agent_exposure_min": _exposure(plan),
    }
    return {
        "video_id": plan.video_id,
        "subject_id": plan.subject_id,
        "sex": plan.sex,
        "compound": facts.compound_label,
        "concentration_mM": facts.dose_label,
        **{name: None if no_workbook else value for name, value in workbook_fields.items()},
        "video_path": _video_path(plan),
        "video_duration_s": facts.n_frames / facts.fps,
        "video_fps": facts.fps,
        "pipeline_version": PIPELINE_VERSION,
        "calibration_profile_version": facts.profile,
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT.isoformat(),
        "provenance": source,
        "edit_count": 1 if source == schema.SOURCE_MANUAL else 0,
        "frames_path": str(gold / schema.FRAMES_FILE),
        "segments_path": str(gold / schema.SEGMENTS_FILE),
        "strip_path": str(gold / schema.STRIP_FILE),  # prepds always writes one; the synthetic set does not
        "manifest_path": str(gold / schema.MANIFEST_FILE),
    }


def _workbook_row(facts: FishFacts, config: SynthConfig) -> dict[str, Any]:
    plan = facts.plan
    fish = WorkbookFish(
        date=_date(plan.date_index),
        subject=plan.number,
        sex=plan.sex,
        strain=STRAIN,
        age=_age(plan),
        compound_label=facts.compound_label,
        dose_label=facts.dose_label,
        agent_exposure_min=_exposure(plan),
        speed_multiplier=facts.speed_multiplier,
    )
    return workbook_record(fish, config.ntt_missing_fraction, np.random.default_rng([config.seed, plan.number, NTT_STREAM]))


def _truth_row(facts: FishFacts) -> dict[str, Any]:
    plan = facts.plan
    return {
        "video_id": plan.video_id,
        "subject_id": plan.subject_id,
        "sex": plan.sex,
        "compound": plan.compound,
        "dose": plan.dose,
        "date": _date(plan.date_index).isoformat(),
        "group": plan.group,
        "fps": facts.fps,
        "profile": facts.profile,
    }


# --- small helpers ------------------------------------------------------------------------------


def _date_effects(config: SynthConfig) -> np.ndarray:
    """One log-speed shift per date, evenly spread in [-1, 1] and shuffled by the seed."""
    rng = np.random.default_rng([config.seed, DATE_STREAM])
    return rng.permutation(np.linspace(-1.0, 1.0, config.n_dates))


def _date(date_index: int) -> dt.date:
    return FIRST_DATE + dt.timedelta(days=date_index)


def _age(plan: FishPlan) -> float:
    return BASE_AGE + plan.number % AGE_SPREAD


def _exposure(plan: FishPlan) -> float:
    """Protocol field that differs by compound (a label proxy the feature matrix must exclude)."""
    step = 0 if plan.effect_index is None else 1 + plan.effect_index % 2
    return BASE_EXPOSURE_MIN + EXPOSURE_STEP_MIN * step


def _video_path(plan: FishPlan) -> str:
    return f"videos/{plan.compound.replace(' ', '')}/{plan.video_id}.mp4"


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
