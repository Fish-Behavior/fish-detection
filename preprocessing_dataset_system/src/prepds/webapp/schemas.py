"""Request/response models for the review API (T066)."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

from prepds.models import BehaviorState, FrameSource, ReviewStatus

MAX_EDITS_PER_REQUEST = 1000


def _clean_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned or any(ord(c) < 32 or ord(c) == 127 for c in cleaned):
        raise ValueError("reviewer must be a non-blank name without control characters")
    return cleaned


Reviewer = Annotated[str, Field(max_length=100), AfterValidator(_clean_name)]


class ReviewFlagOut(BaseModel):
    kind: str
    start_s: float
    end_s: float
    message: str


class ManifestOut(BaseModel):
    subject_id: str
    sex: str
    compound: str
    concentration_mM: str
    video_path: str
    video_duration_s: float
    video_fps: float
    pipeline_version: str
    calibration_profile_version: str
    processed_at: str
    review_status: ReviewStatus
    reviewer: str | None
    reviewed_at: str | None
    edited: bool
    edit_count: int
    review_flags: list[ReviewFlagOut]


class VideoSummary(BaseModel):
    video_id: str
    subject_id: str
    sex: str
    compound: str
    concentration_mM: str
    review_status: ReviewStatus
    edited: bool
    edit_count: int
    flag_count: int


class SegmentOut(BaseModel):
    start_s: float
    end_s: float
    duration_s: float
    state: BehaviorState
    source: FrameSource


class FrameSummary(BaseModel):
    seconds_by_state: dict[str, float]
    has_undetermined: bool
    has_dead: bool


class ListingFlagOut(BaseModel):
    start_s: float
    end_s: float
    max_score: float
    n_samples: int


class VideoDetail(BaseModel):
    video_id: str
    manifest: ManifestOut
    segments: list[SegmentOut]
    frame_summary: FrameSummary
    video_url: str
    strip_url: str | None
    listing_flags: list[ListingFlagOut] = []
    listing_flags_error: str | None = None


class EditItem(BaseModel):
    start_s: float = Field(allow_inf_nan=False)
    end_s: float = Field(allow_inf_nan=False)
    new_state: BehaviorState


class EditsRequest(BaseModel):
    reviewer: Reviewer
    edits: list[EditItem] = Field(min_length=1, max_length=MAX_EDITS_PER_REQUEST)


class AcceptRequest(BaseModel):
    reviewer: Reviewer
    force: bool = False


class RejectRequest(BaseModel):
    reviewer: Reviewer
