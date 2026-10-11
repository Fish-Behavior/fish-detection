"""Per-video export (T058-T060) - FR-010, FR-017.

Three artifacts (frames.parquet, segments.csv, strip.png) plus manifest.json must be mutually consistent:
the segments are the run-length encoding of exactly the frames written, and the strip is drawn from those
segments over the video's real duration. The manifest records provenance (pipeline + calibration version)
and the reviewer flags.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from prepds.export import export_video
from prepds.features import derive_features
from prepds.models import (
    BehaviorState as B,
    FrameSource,
    ManifestRecord,
    MatchStatus,
    ReviewFlag,
    ReviewStatus,
    StateFrame,
    Track,
    Trial,
    VideoAsset,
)
from prepds.segments import frames_to_segments

FPS = 30.0
PROCESSED_AT = dt.datetime(2026, 9, 23, 12, 0, tzinfo=dt.timezone.utc)


def _trial() -> Trial:
    return Trial("0332", "F", "Casper", None, "Fentanyl", "0.03", dt.date(2026, 3, 1), 20.0,
                 Path("videos/F_332.mp4"), MatchStatus.MATCHED)


def _inputs(n=180):
    tracks = [
        Track(i, i / FPS, i * 2.0, 100.0, 90.0, 100.0, detected=(i % 50 != 49)) for i in range(n)
    ]
    features = derive_features(tracks)
    plan = [B.CONTROLLED_SWIM] * (n // 3) + [B.FREEZING_DRIFT] * (n // 3) + [B.ERRATIC_MOVEMENT] * (n - 2 * (n // 3))
    states = [StateFrame(i, i / FPS, plan[i], FrameSource.AUTO, None) for i in range(n)]
    asset = VideoAsset(Path("videos/F_332.mp4"), n / FPS, FPS, n, (192, 240))
    return tracks, features, states, asset


def _export(tmp_path: Path, **kwargs):
    tracks, features, states, asset = _inputs()
    return export_video(
        tmp_path, trial=_trial(), asset=asset, tracks=tracks, features=features, states=states,
        pipeline_version="0.1.0", calibration_profile_version="cal-2026-09-23", processed_at=PROCESSED_AT, **kwargs,
    ), (tracks, features, states, asset)


def test_writes_three_consistent_artifacts_and_a_manifest(tmp_path: Path) -> None:
    paths, (tracks, _, states, asset) = _export(tmp_path)
    assert {p.name for p in (paths.frames, paths.segments, paths.strip, paths.manifest)} == {
        "frames.parquet", "segments.csv", "strip.png", "manifest.json",
    }
    assert all(p.is_file() for p in (paths.frames, paths.segments, paths.strip, paths.manifest))

    frames = pd.read_parquet(paths.frames)
    assert len(frames) == len(tracks)
    assert list(frames["frame_idx"]) == [t.frame_idx for t in tracks]

    # segments.csv is exactly the run-length encoding of the frames on disk
    written = pd.read_csv(paths.segments)
    reread = [
        StateFrame(int(r.frame_idx), float(r.t_sec), B(r.state), FrameSource(r.source), None)
        for r in frames.itertuples()
    ]
    expected = frames_to_segments(reread)
    assert list(written["state"]) == [s.state.value for s in expected]
    assert list(written["start_s"]) == pytest.approx([s.start_s for s in expected])
    assert list(written["end_s"]) == pytest.approx([s.end_s for s in expected])

    with Image.open(paths.strip) as strip:
        assert strip.format == "PNG"


def test_frames_parquet_follows_the_prd_schema(tmp_path: Path) -> None:
    paths, _ = _export(tmp_path)
    frames = pd.read_parquet(paths.frames)
    assert list(frames.columns) == [
        "frame_idx", "t_sec", "x", "y", "orientation_deg", "depth_from_surface", "detected", "velocity",
        "acceleration", "angular_velocity", "meander", "is_immobile", "state", "source", "confidence",
    ]
    assert str(frames["frame_idx"].dtype) == "int32" and str(frames["t_sec"].dtype) == "float32"
    assert str(frames["state"].dtype) == "category" and len(frames["state"].cat.categories) == 7
    assert list(frames["source"].cat.categories) == ["auto", "manual"]
    assert frames["confidence"].isna().all()


def test_manifest_records_pipeline_and_calibration_version_and_flags(tmp_path: Path) -> None:
    flag = ReviewFlag("terminal_no_action", 842.0, 1201.0, "check")
    paths, (_, _, _, asset) = _export(tmp_path, review_flags=(flag,))
    manifest = ManifestRecord.from_dict(json.loads(paths.manifest.read_text()))

    assert manifest.pipeline_version == "0.1.0" and manifest.calibration_profile_version == "cal-2026-09-23"
    assert manifest.review_status == ReviewStatus.PROCESSED_AUTO
    assert manifest.subject_id == "0332" and manifest.compound == "Fentanyl" and manifest.sex == "F"
    assert manifest.video_duration_s == asset.duration_s and manifest.video_fps == FPS
    assert manifest.processed_at == PROCESSED_AT
    assert manifest.review_flags == (flag,)
    assert manifest.reviewer is None and manifest.edited is False and manifest.edit_count == 0


def test_mismatched_input_lengths_are_rejected(tmp_path: Path) -> None:
    tracks, features, states, asset = _inputs()
    with pytest.raises(ValueError, match="same length"):
        export_video(tmp_path, trial=_trial(), asset=asset, tracks=tracks, features=features[:-1], states=states,
                     pipeline_version="0.1.0", calibration_profile_version="c", processed_at=PROCESSED_AT)


def test_empty_video_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no frames"):
        export_video(tmp_path, trial=_trial(), asset=_inputs()[3], tracks=[], features=[], states=[],
                     pipeline_version="0.1.0", calibration_profile_version="c", processed_at=PROCESSED_AT)


def test_re_export_overwrites_the_artifacts_in_place(tmp_path: Path) -> None:
    paths, _ = _export(tmp_path)
    again, _ = _export(tmp_path)
    assert again.frames == paths.frames and again.frames.is_file()


def test_segments_csv_is_the_exact_run_length_encoding_of_the_parquet_frames(tmp_path: Path) -> None:
    paths, _ = _export(tmp_path)
    frames = pd.read_parquet(paths.frames)
    reread = [
        StateFrame(int(r.frame_idx), float(r.t_sec), B(str(r.state)), FrameSource(str(r.source)), None)
        for r in frames.itertuples()
    ]
    written = pd.read_csv(paths.segments)
    expected = frames_to_segments(reread)
    assert list(written["start_s"]) == [s.start_s for s in expected]  # bit-identical, float32 rounding included
    assert list(written["end_s"]) == [s.end_s for s in expected]


def test_re_export_never_overwrites_reviewed_work_unless_asked(tmp_path: Path) -> None:
    import prepds.review_store as store
    from prepds.export import ReviewedWorkExists

    paths, (tracks, features, states, asset) = _export(tmp_path)
    store.save_edit(paths.frames.parent, [(1.0, 2.0, B.SURFACE_BREACH)], reviewer="lk", now=PROCESSED_AT)
    kwargs = dict(trial=_trial(), asset=asset, tracks=tracks, features=features, states=states,
                  pipeline_version="0.1.0", calibration_profile_version="c", processed_at=PROCESSED_AT)
    with pytest.raises(ReviewedWorkExists):
        export_video(tmp_path, **kwargs)
    assert store.load_manifest(paths.frames.parent).edit_count == 1

    export_video(tmp_path, overwrite=True, **kwargs)
    assert store.load_manifest(paths.frames.parent).edit_count == 0


def test_a_failed_forced_export_leaves_no_stale_manifest_behind(tmp_path, monkeypatch) -> None:
    import prepds.export as export_module

    _export(tmp_path)
    (tmp_path / "manifest.json").write_text(
        (tmp_path / "manifest.json").read_text().replace("PROCESSED_AUTO", "EDITED"))
    monkeypatch.setattr(export_module, "render_strip", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        _export(tmp_path, overwrite=True)
    assert not (tmp_path / "manifest.json").exists()  # never new frames under an old "EDITED" manifest
