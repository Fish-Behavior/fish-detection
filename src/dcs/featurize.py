"""Gold set -> one row per fish (plan U5, T1.11; PRD FR-1, §5.3, §5.4).

`segment_features` and `kinematic_features` are pure functions of one fish's files. `featurize`
joins them with the gold set's labels and the workbook's NTT columns into the training table, and
describes every column in a schema (role, source, kind, feature group) that `trainset` reads.

`Undetermined` time (every unreviewed video has some, U3) is unknown time, not a behavior state:
state shares are of the known time, transitions are not bridged across it, and it gets no features
of its own. Kinematics use detected frames only. Nothing here filters features (U6) or fills
missing values (U9).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dcs import schema
from dcs.gold import GoldSet, load_video
from dcs.workbook import HAS_NTT, NTT_COLUMNS, read_ntt

BEHAVIOR_STATES = tuple(state for state in schema.STATES if state != schema.UNDETERMINED)
ROLES = ("feature", "label", "group", "id", "meta")
KINDS = ("count", "duration", "fraction", "continuous", "flag", "category")  # log1p on count and duration
DEPTH_PERCENTILES = (1, 10, 50, 90, 99)  # 1st and 99th feed the audit's framing check (EC-31)
DROP_NO_DETECTED = "no_detected_frames"
DROP_NO_KNOWN_STATE = "no_known_state_time"
SCHEMA_FORMAT = 1


def slug(state: str) -> str:
    """`Freezing/Drift` -> `freezing_drift`."""
    return "".join(c if c.isalnum() else "_" for c in state.lower())


@dataclass(frozen=True)
class Column:
    name: str
    role: str
    source: str
    kind: str | None = None
    feature_group: str | None = None
    states: tuple[str, ...] = ()

    def as_json(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"name": self.name, "role": self.role, "source": self.source}
        if self.kind:
            entry["kind"] = self.kind
        if self.feature_group:
            entry["feature_group"] = self.feature_group
        if self.states:
            entry["states"] = list(self.states)
        return entry


def _columns() -> tuple[Column, ...]:
    gold, seg, frames, wb = "gold", schema.SEGMENTS_FILE, schema.FRAMES_FILE, "workbook"
    out = [
        Column("video_id", "id", gold),
        Column("subject_id", "id", gold),
        Column("compound", "label", gold),
        Column("concentration_mM", "label", gold),
        Column("date", "group", gold),
    ]
    for state in BEHAVIOR_STATES:
        s = slug(state)
        out += [
            Column(f"state_{s}_share", "feature", seg, "fraction", "states", (state,)),
            Column(f"state_{s}_bouts", "feature", seg, "count", "states", (state,)),
            Column(f"state_{s}_mean_bout_s", "feature", seg, "duration", "states", (state,)),
            Column(f"state_{s}_latency_s", "feature", seg, "duration", "states", (state,)),
        ]
    for a, b in permutations(BEHAVIOR_STATES, 2):
        out.append(Column(f"trans_{slug(a)}_to_{slug(b)}", "feature", seg, "count", "transitions", (a, b)))
    for name, kind in (
        ("velocity_mean", "continuous"),
        ("velocity_median", "continuous"),
        ("velocity_cv", "continuous"),
        ("abs_acceleration_mean", "continuous"),
        ("abs_angular_velocity_mean", "continuous"),
        ("meander_mean", "continuous"),
        ("immobile_share", "fraction"),
        ("detected_share", "fraction"),
    ):
        out.append(Column(name, "feature", frames, kind, "kinematics"))
    depth = ["depth_mean", "depth_min", *(f"depth_p{p:02d}" for p in DEPTH_PERCENTILES)]
    out += [Column(name, "feature", frames, "continuous", "depth") for name in depth]  # off by default (D-015)
    for name in NTT_COLUMNS:
        out.append(Column(name, "feature", wb, "duration" if name.startswith("time_") else "continuous", "ntt"))
    out.append(Column(HAS_NTT, "feature", wb, "flag", "ntt"))
    out += [  # off by default; `use_demographics` is an ablation (PRD §5.3)
        Column("sex", "feature", gold, "category", "demographics"),
        Column("strain", "feature", gold, "category", "demographics"),
        Column("age", "feature", gold, "continuous", "demographics"),
    ]
    # Meta: reported by the audit, never a feature (PRD §5.3 exclusions; trainset enforces it, EC-12).
    out += [
        Column(name, "meta", source)
        for name, source in (
            ("recording_s", seg),
            ("undetermined_share", seg),
            ("manual_share", frames),
            ("flag_low_detected", frames),
            ("flag_odd_duration", gold),
            ("video_duration_s", gold),
            ("video_fps", gold),
            ("agent_exposure_min", gold),
            ("calibration_profile_version", gold),
            ("pipeline_version", gold),
            ("review_status", gold),
            ("reviewed", gold),
            ("review_flag_count", gold),
            ("edit_count", gold),
        )
    ]
    return tuple(out)


COLUMNS = _columns()


@dataclass(frozen=True)
class Featurized:
    table: pd.DataFrame
    schema: dict[str, Any]


# --- per-fish features ---------------------------------------------------------------------


def segment_features(segments: pd.DataFrame) -> dict[str, float] | None:
    """Share, bouts, mean bout and latency per state, transition counts, and the recording length.

    A state never shown has 0 bouts, mean bout 0 and latency = recording length (PRD §5.4, EC-3).
    None when the recording has no known (non-Undetermined) time.
    """
    segments = segments.sort_values("start_s")
    start = float(segments["start_s"].iloc[0])
    recording = float(segments["duration_s"].sum())
    # One bout per run of a state; prepds also splits a run where the label source changes.
    run = (segments["state"] != segments["state"].shift()).cumsum()
    bouts = segments.groupby(run, sort=False).agg(
        state=("state", "first"), start_s=("start_s", "first"), duration_s=("duration_s", "sum")
    )
    known = bouts[bouts["state"] != schema.UNDETERMINED]
    known_s = float(known["duration_s"].sum())
    if known_s <= 0:
        return None

    out: dict[str, float] = {}
    for state in BEHAVIOR_STATES:
        own = known.loc[known["state"] == state, ["start_s", "duration_s"]]
        s = slug(state)
        out[f"state_{s}_share"] = float(own["duration_s"].sum()) / known_s
        out[f"state_{s}_bouts"] = float(len(own))
        out[f"state_{s}_mean_bout_s"] = float(own["duration_s"].mean()) if len(own) else 0.0
        out[f"state_{s}_latency_s"] = float(own["start_s"].min()) - start if len(own) else recording
    pairs = list(zip(bouts["state"], bouts["state"].iloc[1:]))  # Undetermined in between breaks a pair
    for a, b in permutations(BEHAVIOR_STATES, 2):
        out[f"trans_{slug(a)}_to_{slug(b)}"] = float(pairs.count((a, b)))
    out["recording_s"] = recording
    return out


def kinematic_features(frames: pd.DataFrame) -> dict[str, float] | None:
    """Speed, turning, meander, immobility and depth on detected frames; None when none was detected.

    prepds writes 0.0 where a detected run has too little history (schema.ZERO_ON_FIRST_FRAME,
    ZERO_ON_FIRST_TWO_FRAMES); those frames are left out, so broken-up tracking does not slow a fish
    down (D-048). A fish whose runs are all too short gets NaN for those features, and one with no depth values NaN for depth.
    """
    detected = frames["detected"].astype(bool).to_numpy()
    if not detected.any():
        return None
    first = detected & ~np.r_[False, detected[:-1]]
    one_before = detected & ~first  # speed and turn are real here
    two_before = one_before & ~np.r_[False, first[:-1]]  # acceleration and meander too
    seen = frames[detected]
    velocity = frames["velocity"].to_numpy(dtype=float)[one_before]
    mean = _mean(velocity)
    depth = seen["depth_from_surface"].dropna().to_numpy(dtype=float)
    out = {
        "velocity_mean": mean,
        "velocity_median": float(np.median(velocity)) if len(velocity) else math.nan,
        # speeds are >= 0: mean 0 means no motion
        "velocity_cv": float(velocity.std()) / mean if mean > 0 else 0.0 if len(velocity) else math.nan,
        "abs_acceleration_mean": _mean(np.abs(frames["acceleration"].to_numpy(dtype=float)[two_before])),
        "abs_angular_velocity_mean": _mean(np.abs(frames["angular_velocity"].to_numpy(dtype=float)[one_before])),
        "meander_mean": _mean(frames["meander"].to_numpy(dtype=float)[two_before]),
        "immobile_share": float(seen["is_immobile"].astype(bool).mean()),
        "detected_share": len(seen) / len(frames),
        "depth_mean": _mean(depth),
        "depth_min": float(depth.min()) if len(depth) else math.nan,
    }
    out.update({f"depth_p{p:02d}": float(np.percentile(depth, p)) if len(depth) else math.nan for p in DEPTH_PERCENTILES})
    return out


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else math.nan


# --- the table -----------------------------------------------------------------------------


def featurize(gold: GoldSet, workbook: Path | None, training: Mapping[str, Any]) -> Featurized:
    """One row per kept fish of `gold`, in its order; fish that cannot be featurized are dropped and listed."""
    videos = gold.videos
    ntt = read_ntt(workbook, videos).set_index("video_id")
    dropped = gold.dropped.to_dict("records")
    rows = []
    for record in videos.to_dict("records"):
        video_id = str(record["video_id"])
        video = load_video(gold, video_id)
        kinematics = kinematic_features(video.frames)
        if kinematics is None:
            dropped.append({"video_id": video_id, "reason": DROP_NO_DETECTED, "detail": "no detected frame"})
            continue
        states = segment_features(video.segments)
        if states is None:
            dropped.append({"video_id": video_id, "reason": DROP_NO_KNOWN_STATE, "detail": "all Undetermined"})
            continue
        rows.append(
            {
                **record,
                **states,
                **kinematics,
                **ntt.loc[video_id].to_dict(),
                "manual_share": float((video.frames["source"] == schema.SOURCE_MANUAL).mean()),
                "flag_low_detected": int(kinematics["detected_share"] < training["min_detected_fraction"]),
                "flag_odd_duration": _odd_duration(record["video_duration_s"], training["duration_range_s"]),
            }
        )
    table = pd.DataFrame(rows, columns=[column.name for column in COLUMNS])
    table[HAS_NTT] = table[HAS_NTT].astype(int)
    return Featurized(table, _schema(gold, workbook is not None, dropped, training))


def schema_path(table_path: Path) -> Path:
    """`training_table.parquet` -> `training_table_schema.json`, beside it."""
    return table_path.with_name(f"{table_path.stem}_schema.json")


def write_outputs(result: Featurized, table_path: Path) -> tuple[Path, Path]:
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    result.table.to_parquet(table_path, index=False)
    json_path = schema_path(table_path)
    json_path.write_text(json.dumps(result.schema, indent=2) + "\n", encoding="utf-8")
    return table_path, json_path


def _odd_duration(duration: float, allowed: Any) -> int:
    """EC-11: 1 outside [min, max]; no range set (null) checks nothing (D-021)."""
    if allowed is None:
        return 0
    low, high = allowed
    return int(not low <= duration <= high)


def _schema(gold: GoldSet, ntt_available: bool, dropped: list[dict[str, Any]], training: Mapping[str, Any]) -> dict[str, Any]:
    low, high = gold.fps_range
    return {
        "format": SCHEMA_FORMAT,
        "gold_source": gold.source,
        "profile": gold.profile,
        "tracker": gold.tracker,
        "fps_uniform": gold.fps_uniform,
        "fps_range": [low, high],
        "tracker_checked": gold.tracker_checked,
        "ntt_available": ntt_available,
        "flags": {
            "min_detected_fraction": training["min_detected_fraction"],
            "duration_range_s": None if training["duration_range_s"] is None else list(training["duration_range_s"]),
        },
        "dropped": [{key: str(drop[key]) for key in ("video_id", "reason", "detail")} for drop in dropped],
        "columns": [column.as_json() for column in COLUMNS],
    }
