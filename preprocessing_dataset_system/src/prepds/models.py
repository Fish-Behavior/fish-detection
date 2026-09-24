"""Typed data contracts for the pipeline (PRD §5).

Scope note: only the entities named by Phase 1's T009/T010 are defined here
(Trial, VideoAsset, FrameRow, StateSegment, ManifestRecord, plus the enums
they use). The rest of §5.1's entity table (Track, FeatureFrame, StateFrame,
StripImage, CalibrationProfile, ReviewRecord, CatalogExceptionsReport) is
added in the phase that first needs it (tracking.py, features.py, labeling.py,
rendering.py, calibration/, review_store.py, catalog.py respectively) rather
than speculatively here - see docs/progress.md §2 Design Decisions.

Every dataclass is frozen (immutable) and carries `to_dict()`/`from_dict()`
for the exact JSON/Parquet-safe round-trip the frames/segments/manifest
export formats need (paths -> str, dates -> ISO strings, enums -> their
value), tested in tests/test_models.py.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class BehaviorState(str, Enum):
    """The 7 categories a frame can be classified into (PRD §5.4).

    Five real behavioral states, plus the terminal Dead outcome, plus the
    pipeline-internal Undetermined bookkeeping marker (PRD §5.2 - never a
    claim about the fish's biological state, unlike Dead).
    """

    CONTROLLED_SWIM = "Controlled Swim"
    ERRATIC_MOVEMENT = "Erratic Movement"
    FREEZING_DRIFT = "Freezing/Drift"
    LISTING_LORR = "Listing/LORR"
    SURFACE_BREACH = "Surface Breach"
    DEAD = "Dead"
    UNDETERMINED = "Undetermined"


class FrameSource(str, Enum):
    """Whether a frame/segment's label came from the classifier or a reviewer edit."""

    AUTO = "auto"
    MANUAL = "manual"


class MatchStatus(str, Enum):
    """Outcome of matching one trial row to a video file (PRD §5.3)."""

    MATCHED = "matched"
    NO_VIDEO = "no_video"
    NO_TRIAL_ROW = "no_trial_row"
    DUPLICATE_ROW = "duplicate_row"


class ReviewStatus(str, Enum):
    """Per-video review state machine (PRD §5.1 ReviewRecord, FR-016)."""

    NOT_PROCESSED = "NOT_PROCESSED"
    PROCESSED_AUTO = "PROCESSED_AUTO"
    EDITED = "EDITED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


def _path_to_str(value: Path | None) -> str | None:
    return str(value) if value is not None else None


def _str_to_path(value: str | None) -> Path | None:
    return Path(value) if value is not None else None


@dataclass(frozen=True)
class Trial:
    """One cleaned row from `00_NTT_DataBase.xlsx` (PRD §5.3)."""

    subject_id: str  # 4-digit zero-padded, normalized from workbook "Subject #:"
    sex: str  # "M" | "F"
    strain: str
    age: float | None
    compound: str  # raw "Compund:" value, incl. combo treatments
    concentration_mM: str  # kept as string; combo doses like "0.03 + 0.01" aren't numeric
    date: dt.date
    agent_exposure_min: float
    video_path: Path | None  # resolved local path after matching; None if unmatched
    match_status: MatchStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "sex": self.sex,
            "strain": self.strain,
            "age": self.age,
            "compound": self.compound,
            "concentration_mM": self.concentration_mM,
            "date": self.date.isoformat(),
            "agent_exposure_min": self.agent_exposure_min,
            "video_path": _path_to_str(self.video_path),
            "match_status": self.match_status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trial":
        return cls(
            subject_id=data["subject_id"],
            sex=data["sex"],
            strain=data["strain"],
            age=data["age"],
            compound=data["compound"],
            concentration_mM=data["concentration_mM"],
            date=dt.date.fromisoformat(data["date"]),
            agent_exposure_min=data["agent_exposure_min"],
            video_path=_str_to_path(data["video_path"]),
            match_status=MatchStatus(data["match_status"]),
        )


@dataclass(frozen=True)
class VideoAsset:
    """The located, validated video file for a trial (PRD §5.1)."""

    path: Path
    duration_s: float
    fps: float
    frame_count: int
    resolution: tuple[int, int]  # (width, height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "duration_s": self.duration_s,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "resolution": list(self.resolution),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoAsset":
        return cls(
            path=Path(data["path"]),
            duration_s=data["duration_s"],
            fps=data["fps"],
            frame_count=data["frame_count"],
            resolution=tuple(data["resolution"]),
        )


@dataclass(frozen=True)
class FrameRow:
    """One row of `frames.parquet` (PRD §5.4) - tracking + features + label, merged."""

    frame_idx: int
    t_sec: float
    x: float
    y: float
    orientation_deg: float | None
    depth_from_surface: float | None
    detected: bool
    velocity: float
    acceleration: float
    angular_velocity: float
    meander: float
    is_immobile: bool
    state: BehaviorState
    source: FrameSource
    confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "t_sec": self.t_sec,
            "x": self.x,
            "y": self.y,
            "orientation_deg": self.orientation_deg,
            "depth_from_surface": self.depth_from_surface,
            "detected": self.detected,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "angular_velocity": self.angular_velocity,
            "meander": self.meander,
            "is_immobile": self.is_immobile,
            "state": self.state.value,
            "source": self.source.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameRow":
        return cls(
            frame_idx=data["frame_idx"],
            t_sec=data["t_sec"],
            x=data["x"],
            y=data["y"],
            orientation_deg=data["orientation_deg"],
            depth_from_surface=data["depth_from_surface"],
            detected=data["detected"],
            velocity=data["velocity"],
            acceleration=data["acceleration"],
            angular_velocity=data["angular_velocity"],
            meander=data["meander"],
            is_immobile=data["is_immobile"],
            state=BehaviorState(data["state"]),
            source=FrameSource(data["source"]),
            confidence=data["confidence"],
        )


@dataclass(frozen=True)
class StateSegment:
    """One run-length-encoded contiguous block of one state (PRD §5.4 segments.csv)."""

    start_s: float
    end_s: float
    duration_s: float
    state: BehaviorState
    source: FrameSource

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "state": self.state.value,
            "source": self.source.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateSegment":
        return cls(
            start_s=data["start_s"],
            end_s=data["end_s"],
            duration_s=data["duration_s"],
            state=BehaviorState(data["state"]),
            source=FrameSource(data["source"]),
        )


@dataclass(frozen=True)
class ReviewFlag:
    """A note the pipeline attaches to a video for the reviewer (e.g. a stretch it cannot label itself)."""

    kind: str
    start_s: float
    end_s: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "start_s": self.start_s, "end_s": self.end_s, "message": self.message}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewFlag":
        return cls(kind=data["kind"], start_s=data["start_s"], end_s=data["end_s"], message=data["message"])


@dataclass(frozen=True)
class ManifestRecord:
    """`manifest.json` contents for one video (PRD §5.4)."""

    subject_id: str
    sex: str
    compound: str
    concentration_mM: str
    video_path: Path
    video_duration_s: float
    video_fps: float
    pipeline_version: str
    calibration_profile_version: str
    processed_at: dt.datetime
    review_status: ReviewStatus
    reviewer: str | None
    reviewed_at: dt.datetime | None
    edited: bool
    edit_count: int
    review_flags: tuple[ReviewFlag, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "sex": self.sex,
            "compound": self.compound,
            "concentration_mM": self.concentration_mM,
            "video_path": str(self.video_path),
            "video_duration_s": self.video_duration_s,
            "video_fps": self.video_fps,
            "pipeline_version": self.pipeline_version,
            "calibration_profile_version": self.calibration_profile_version,
            "processed_at": self.processed_at.isoformat(),
            "review_status": self.review_status.value,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at is not None else None,
            "edited": self.edited,
            "edit_count": self.edit_count,
            "review_flags": [flag.to_dict() for flag in self.review_flags],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestRecord":
        reviewed_at = data["reviewed_at"]
        return cls(
            subject_id=data["subject_id"],
            sex=data["sex"],
            compound=data["compound"],
            concentration_mM=data["concentration_mM"],
            video_path=Path(data["video_path"]),
            video_duration_s=data["video_duration_s"],
            video_fps=data["video_fps"],
            pipeline_version=data["pipeline_version"],
            calibration_profile_version=data["calibration_profile_version"],
            processed_at=dt.datetime.fromisoformat(data["processed_at"]),
            review_status=ReviewStatus(data["review_status"]),
            reviewer=data["reviewer"],
            reviewed_at=dt.datetime.fromisoformat(reviewed_at) if reviewed_at is not None else None,
            edited=data["edited"],
            edit_count=data["edit_count"],
            review_flags=tuple(ReviewFlag.from_dict(f) for f in data.get("review_flags", [])),
        )


@dataclass(frozen=True)
class Track:
    """Per-frame raw localization of the fish from foreground extraction (PRD §5.1) - Phase 4 addition.

    `y_from_frame_top` is the raw pixel y-coordinate (y=0 at the top of
    frame) - deliberately NOT named `depth_from_surface` (that name is
    reserved for `FrameRow`'s later, calibration-aware field) because T029's
    real-data finding was that the detected background edge is not reliably
    the true water surface across all videos, so no surface-relative claim
    is baked in at this stage. This is the top-frame-relative candidate from
    the two-signal design in docs/strain_tracking_notes.md §3;
    `track_video_with_context()` also returns the detected edge row
    separately for features.py to use as the second candidate if calibration
    favors it.

    `y_from_frame_top` is `None` whenever `detected` is `False` - mirroring
    `orientation_deg`'s existing gating, not `x`/`y` (which stay `0.0` as an
    explicit, documented, non-optional sentinel per §3.3/test coverage).
    When `detected` is `True`, `y_from_frame_top` holds exactly the same raw
    pixel row as `y` - the two fields carry identical values, not
    independent signals. The reason to still gate `y_from_frame_top`
    separately (code review, not a live bug today - nothing downstream
    consumes `Track` yet): a features.py/labeling.py author reading this
    field's name will reasonably treat it as a depth/surface-proximity
    signal in its own right and may check only `y_from_frame_top is None`
    without separately checking `detected`, whereas `x`/`y` read as plain
    position and are less likely to be trusted without that check. `y`
    carries the exact same fabricated-`0.0`-on-undetected risk as
    `y_from_frame_top` would if left ungated - any code consuming `x`/`y`
    must still gate on `detected`, not on `x`/`y`'s own value.
    """

    frame_idx: int
    t_sec: float
    x: float
    y: float
    orientation_deg: float | None
    y_from_frame_top: float | None
    detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "t_sec": self.t_sec,
            "x": self.x,
            "y": self.y,
            "orientation_deg": self.orientation_deg,
            "y_from_frame_top": self.y_from_frame_top,
            "detected": self.detected,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        return cls(
            frame_idx=data["frame_idx"],
            t_sec=data["t_sec"],
            x=data["x"],
            y=data["y"],
            orientation_deg=data["orientation_deg"],
            y_from_frame_top=data["y_from_frame_top"],
            detected=data["detected"],
        )


@dataclass(frozen=True)
class FeatureFrame:
    """Per-frame derived kinematics from a `Track` list (PRD §5.1, §9.5.4) - FR-006, Phase 5 addition.

    **PRD discrepancy, not a design decision (see docs/progress.md Phase 5
    for the full note):** §5.1's entity table lists `FeatureFrame` fields as
    `frame_idx, velocity, acceleration, angular_velocity, meander,
    smoothness, is_immobile` - including a `smoothness` field. §5.4's actual
    `frames.parquet` schema (the persisted, tested contract) has no
    `smoothness` column, and FR-006 itself says "a path-smoothness **or**
    meander measure" - phrasing the two as alternatives, not two features.
    This implementation produces only `meander`, matching the schema that is
    actually tested against. If a later phase needs a distinct `smoothness`
    field, this note is where that gap was first found.

    All four kinematic fields are non-nullable floats (matching
    `frames.parquet`'s `float32`, not `float32|null`) and use the same
    `0.0`-sentinel-on-unknown convention as `Track.x`/`Track.y`, NOT `None`
    - `0.0` here means "not computable from real data" (first frame, an
    undetected frame, or too few consecutive detected frames of history),
    never a fabricated real zero. Downstream code (Phase 7 labeling.py) must
    gate on the frame's own `detected` flag for the Undetermined case, never
    infer it from a `0.0` feature value alone - see `features.py` module
    docstring for the exact history-depth rule each field uses.
    """

    frame_idx: int
    velocity: float
    acceleration: float
    angular_velocity: float
    meander: float
    is_immobile: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "angular_velocity": self.angular_velocity,
            "meander": self.meander,
            "is_immobile": self.is_immobile,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureFrame":
        return cls(
            frame_idx=data["frame_idx"],
            velocity=data["velocity"],
            acceleration=data["acceleration"],
            angular_velocity=data["angular_velocity"],
            meander=data["meander"],
            is_immobile=data["is_immobile"],
        )


@dataclass(frozen=True)
class StateFrame:
    """Per-frame classification result (PRD §5.1, §9.5.6) - FR-007, Phase 7 addition.

    `confidence` is `None` for every deterministic rule-based state (Dead,
    Surface Breach, Listing/LORR, Freezing/Drift, and Undetermined-from-
    `detected=False`) - there is no natural probability for a threshold
    crossing. It is only set (to the winning component's posterior
    probability) for the GMM-driven states (Controlled Swim, Erratic
    Movement, and Undetermined-from-low-GMM-confidence) - see
    `labeling.py` module docstring for the full rule precedence order.
    """

    frame_idx: int
    t_sec: float
    state: BehaviorState
    source: FrameSource
    confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "t_sec": self.t_sec,
            "state": self.state.value,
            "source": self.source.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateFrame":
        return cls(
            frame_idx=data["frame_idx"],
            t_sec=data["t_sec"],
            state=BehaviorState(data["state"]),
            source=FrameSource(data["source"]),
            confidence=data["confidence"],
        )


@dataclass(frozen=True)
class CatalogExceptionsReport:
    """Output of the catalog matching step (PRD §5.1, FR-004) - Phase 2 addition.

    Every field is a plain-dict list (not nested dataclasses) so this can be
    written straight to `exceptions_report.json` with `json.dumps` and no
    custom encoder. `corrupt_videos` is not in the PRD §5.1 entity table by
    name, but is required by FR-004 read together with the §3.3 "corrupted or
    zero-byte video file" edge case (report it, never crash the batch).
    """

    unmatched_trials: list[dict[str, Any]]
    unmatched_videos: list[dict[str, Any]]
    duplicate_trials: list[dict[str, Any]]
    duration_mismatches: list[dict[str, Any]]
    corrupt_videos: list[dict[str, Any]]
    ambiguous_matches: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "unmatched_trials": self.unmatched_trials,
            "unmatched_videos": self.unmatched_videos,
            "duplicate_trials": self.duplicate_trials,
            "duration_mismatches": self.duration_mismatches,
            "corrupt_videos": self.corrupt_videos,
            "ambiguous_matches": self.ambiguous_matches,
        }
