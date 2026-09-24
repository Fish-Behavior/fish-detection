"""Schema round-trip (serialize/deserialize) for the T009/T010 dataclasses (T014).

Round-trip here means: build an instance, call to_dict(), feed that dict back
through from_dict(), and get back an equal instance. This is what catches a
field silently dropped or mis-typed when frames.parquet/segments.csv/
manifest.json are written and re-read across process boundaries (T009/T010
scope: Trial, VideoAsset, FrameRow, StateSegment, ManifestRecord - the other
PRD §5.1 entities are added in the phase that first needs them, see
docs/progress.md §2).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from prepds.models import (
    BehaviorState,
    FeatureFrame,
    FrameRow,
    FrameSource,
    ManifestRecord,
    MatchStatus,
    ReviewStatus,
    StateFrame,
    StateSegment,
    Track,
    Trial,
    VideoAsset,
)


def test_trial_round_trip() -> None:
    trial = Trial(
        subject_id="0068",
        sex="F",
        strain="Casper (roya9; mitfaw2)",
        age=None,
        compound="FD-2-45",
        concentration_mM="0.03",
        date=dt.date(2026, 9, 30),
        agent_exposure_min=20.0,
        video_path=Path("data/videos/FD-2-45/F_0068.mp4"),
        match_status=MatchStatus.MATCHED,
    )
    assert Trial.from_dict(trial.to_dict()) == trial


def test_trial_round_trip_with_nulls() -> None:
    trial = Trial(
        subject_id="0320",
        sex="M",
        strain="Casper (roya9; mitfaw2)",
        age=None,
        compound="FD-2-66",
        concentration_mM="0.2",
        date=dt.date(2025, 12, 16),
        agent_exposure_min=20.0,
        video_path=None,
        match_status=MatchStatus.NO_VIDEO,
    )
    assert Trial.from_dict(trial.to_dict()) == trial


def test_video_asset_round_trip() -> None:
    asset = VideoAsset(
        path=Path("data/videos/DOB/F_0068.mp4"),
        duration_s=1202.7,
        fps=29.832876028934898,
        frame_count=35880,
        resolution=(304, 240),
    )
    assert VideoAsset.from_dict(asset.to_dict()) == asset


def test_frame_row_round_trip_detected() -> None:
    row = FrameRow(
        frame_idx=900,
        t_sec=30.0,
        x=100.5,
        y=80.2,
        orientation_deg=12.5,
        depth_from_surface=5.0,
        detected=True,
        velocity=0.01,
        acceleration=0.0,
        angular_velocity=0.0,
        meander=0.1,
        is_immobile=True,
        state=BehaviorState.FREEZING_DRIFT,
        source=FrameSource.AUTO,
        confidence=0.92,
    )
    assert FrameRow.from_dict(row.to_dict()) == row


def test_frame_row_round_trip_undetected_has_null_orientation_and_depth() -> None:
    row = FrameRow(
        frame_idx=1,
        t_sec=0.033,
        x=0.0,
        y=0.0,
        orientation_deg=None,
        depth_from_surface=None,
        detected=False,
        velocity=0.0,
        acceleration=0.0,
        angular_velocity=0.0,
        meander=0.0,
        is_immobile=False,
        state=BehaviorState.UNDETERMINED,
        source=FrameSource.AUTO,
        confidence=None,
    )
    assert FrameRow.from_dict(row.to_dict()) == row


@pytest.mark.parametrize("state", list(BehaviorState))
def test_frame_row_round_trip_every_state(state: BehaviorState) -> None:
    row = FrameRow(
        frame_idx=0,
        t_sec=0.0,
        x=0.0,
        y=0.0,
        orientation_deg=0.0,
        depth_from_surface=0.0,
        detected=True,
        velocity=0.0,
        acceleration=0.0,
        angular_velocity=0.0,
        meander=0.0,
        is_immobile=False,
        state=state,
        source=FrameSource.MANUAL,
        confidence=1.0,
    )
    assert FrameRow.from_dict(row.to_dict()) == row


def test_behavior_state_has_exactly_seven_values() -> None:
    # 5 behavioral states + Dead + Undetermined, per PRD §5.4.
    assert len(list(BehaviorState)) == 7


def test_state_segment_round_trip() -> None:
    segment = StateSegment(
        start_s=30.0,
        end_s=90.0,
        duration_s=60.0,
        state=BehaviorState.FREEZING_DRIFT,
        source=FrameSource.AUTO,
    )
    assert StateSegment.from_dict(segment.to_dict()) == segment


def test_manifest_record_round_trip() -> None:
    manifest = ManifestRecord(
        subject_id="0068",
        sex="F",
        compound="FD-2-45",
        concentration_mM="0.03",
        video_path=Path("data/videos/FD-2-45/F_0068.mp4"),
        video_duration_s=1202.7,
        video_fps=29.83,
        pipeline_version="0.1.0",
        calibration_profile_version="cal-2026-09-30",
        processed_at=dt.datetime(2026, 9, 30, 12, 0, 0, tzinfo=dt.timezone.utc),
        review_status=ReviewStatus.ACCEPTED,
        reviewer="owner",
        reviewed_at=dt.datetime(2026, 9, 30, 13, 0, 0, tzinfo=dt.timezone.utc),
        edited=True,
        edit_count=2,
    )
    assert ManifestRecord.from_dict(manifest.to_dict()) == manifest


def test_manifest_record_round_trip_not_yet_reviewed() -> None:
    manifest = ManifestRecord(
        subject_id="0069",
        sex="M",
        compound="Veh",
        concentration_mM="0",
        video_path=Path("data/videos/Veh/M_0069.mp4"),
        video_duration_s=1200.0,
        video_fps=29.9,
        pipeline_version="0.1.0",
        calibration_profile_version="cal-2026-09-30",
        processed_at=dt.datetime(2026, 9, 30, 12, 0, 0, tzinfo=dt.timezone.utc),
        review_status=ReviewStatus.PROCESSED_AUTO,
        reviewer=None,
        reviewed_at=None,
        edited=False,
        edit_count=0,
    )
    assert ManifestRecord.from_dict(manifest.to_dict()) == manifest


def test_track_round_trip_detected() -> None:
    track = Track(
        frame_idx=30,
        t_sec=1.0,
        x=100.5,
        y=80.2,
        orientation_deg=12.5,
        y_from_frame_top=80.2,
        detected=True,
    )
    assert Track.from_dict(track.to_dict()) == track


def test_track_round_trip_undetected() -> None:
    # Mirrors tracking.py's real undetected-frame shape: x/y stay the
    # documented 0.0 sentinel, but orientation_deg and y_from_frame_top are
    # both None - neither can be trusted from a rejected/missing contour,
    # and y_from_frame_top=0.0 would silently read as "at the surface"
    # (§3.3 - never a fabricated position).
    track = Track(
        frame_idx=0,
        t_sec=0.0,
        x=0.0,
        y=0.0,
        orientation_deg=None,
        y_from_frame_top=None,
        detected=False,
    )
    assert Track.from_dict(track.to_dict()) == track


def test_feature_frame_round_trip() -> None:
    feature = FeatureFrame(
        frame_idx=30,
        velocity=12.5,
        acceleration=-0.3,
        angular_velocity=4.2,
        meander=0.08,
        is_immobile=False,
    )
    assert FeatureFrame.from_dict(feature.to_dict()) == feature


def test_feature_frame_round_trip_unknown_sentinel() -> None:
    # Mirrors features.py's real not-computable-from-history shape: 0.0
    # sentinels, not None - frames.parquet types these non-nullable.
    feature = FeatureFrame(
        frame_idx=0,
        velocity=0.0,
        acceleration=0.0,
        angular_velocity=0.0,
        meander=0.0,
        is_immobile=False,
    )
    assert FeatureFrame.from_dict(feature.to_dict()) == feature


def test_state_frame_round_trip_rule_based() -> None:
    # Rule-based states (Dead, Surface Breach, Listing/LORR, Freezing/Drift,
    # Undetermined-from-undetected) carry no confidence - see StateFrame's
    # own docstring for why.
    state = StateFrame(
        frame_idx=30,
        t_sec=1.0,
        state=BehaviorState.FREEZING_DRIFT,
        source=FrameSource.AUTO,
        confidence=None,
    )
    assert StateFrame.from_dict(state.to_dict()) == state


def test_state_frame_round_trip_with_confidence() -> None:
    state = StateFrame(
        frame_idx=30,
        t_sec=1.0,
        state=BehaviorState.ERRATIC_MOVEMENT,
        source=FrameSource.AUTO,
        confidence=0.87,
    )
    assert StateFrame.from_dict(state.to_dict()) == state


@pytest.mark.parametrize("state", list(BehaviorState))
def test_state_frame_round_trip_every_state(state: BehaviorState) -> None:
    sf = StateFrame(frame_idx=0, t_sec=0.0, state=state, source=FrameSource.MANUAL, confidence=None)
    assert StateFrame.from_dict(sf.to_dict()) == sf


def _manifest(**overrides) -> ManifestRecord:
    base = dict(
        subject_id="0340",
        sex="F",
        compound="Fentanyl",
        concentration_mM="0.03",
        video_path=Path("data/videos/Fentanyl/F_340.mp4"),
        video_duration_s=1201.0,
        video_fps=29.84,
        pipeline_version="0.1.0",
        calibration_profile_version="cal-2026-09-23",
        processed_at=dt.datetime(2026, 9, 23, 12, 0, 0, tzinfo=dt.timezone.utc),
        review_status=ReviewStatus.PROCESSED_AUTO,
        reviewer=None,
        reviewed_at=None,
        edited=False,
        edit_count=0,
    )
    return ManifestRecord(**{**base, **overrides})


def test_review_flag_round_trip() -> None:
    from prepds.models import ReviewFlag

    flag = ReviewFlag(kind="terminal_no_action", start_s=842.0, end_s=1201.0, message="No movement to the end.")
    assert ReviewFlag.from_dict(flag.to_dict()) == flag


def test_manifest_record_carries_review_flags_through_json() -> None:
    from prepds.models import ReviewFlag

    flag = ReviewFlag(kind="terminal_no_action", start_s=842.0, end_s=1201.0, message="m")
    manifest = _manifest(review_flags=(flag,))
    data = manifest.to_dict()
    assert data["review_flags"] == [flag.to_dict()]
    assert ManifestRecord.from_dict(data) == manifest


def test_manifest_without_review_flags_key_still_loads_as_no_flags() -> None:
    data = _manifest().to_dict()
    del data["review_flags"]  # a manifest written before flags existed
    assert ManifestRecord.from_dict(data).review_flags == ()
