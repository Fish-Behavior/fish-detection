"""Frame sampler for the tracker fine-tuning set (Phase 15)."""

from __future__ import annotations

import numpy as np
import pytest

from prepds.annotation.sampler import FrameRef, VideoInfo, sample_frames

FPS = 30.0
N = int(FPS * 600)  # 10 min


def _video(video_id: str, compound: str, gaps=(), flagged_from_s=None, states=None, n=N) -> VideoInfo:
    detected = np.ones(n, dtype=bool)
    for start_s, length_s in gaps:
        detected[int(start_s * FPS) : int((start_s + length_s) * FPS)] = False
    state = np.array(states if states is not None else ["Erratic Movement"] * n, dtype=object)
    return VideoInfo(video_id, compound, FPS, detected, state, flagged_from_s)


def _fleet() -> tuple[list[VideoInfo], dict[str, str]]:
    videos, split = [], {}
    for i in range(12):
        compound = ("Veh", "Fentanyl", "MDMA")[i % 3]
        gaps = [(20 + 40 * k, 3.0 + k) for k in range(6)] + [(400, 60)]  # several 1-10 s gaps and one long gap
        states = ["Freezing/Drift"] * N if compound == "Fentanyl" else None
        v = _video(f"F_{i:04d}", compound, gaps, flagged_from_s=300.0 if i == 4 else None, states=states)
        videos.append(v)
        split[v.video_id] = "heldout" if i in (0, 1, 2) else "train"
    return videos, split


def test_samples_are_deterministic_for_a_seed() -> None:
    videos, split = _fleet()
    a = sample_frames(videos, split, n_frames=60, seed=1)
    assert a == sample_frames(list(reversed(videos)), split, n_frames=60, seed=1)
    assert a != sample_frames(videos, split, n_frames=60, seed=2)


def test_only_the_requested_split_is_sampled() -> None:
    videos, split = _fleet()
    train = {r.video_id for r in sample_frames(videos, split, n_frames=80, seed=1, split_name="train")}
    held = {r.video_id for r in sample_frames(videos, split, n_frames=20, seed=1, split_name="heldout")}
    assert train and held and train.isdisjoint(held)
    assert all(split[v] == "train" for v in train) and all(split[v] == "heldout" for v in held)


def test_no_duplicates_spacing_and_per_video_cap() -> None:
    videos, split = _fleet()
    refs = sample_frames(videos, split, n_frames=90, seed=3, min_gap_s=2.0, max_per_video=6)
    assert len({(r.video_id, r.frame_idx) for r in refs}) == len(refs)
    by_video: dict[str, list[int]] = {}
    for r in refs:
        by_video.setdefault(r.video_id, []).append(r.frame_idx)
    for frames in by_video.values():
        assert len(frames) <= 6
        ordered = sorted(frames)
        assert all(b - a >= 2.0 * FPS for a, b in zip(ordered, ordered[1:]))


def test_reasons_describe_the_frame() -> None:
    videos, split = _fleet()
    by_id = {v.video_id: v for v in videos}
    refs = sample_frames(videos, split, n_frames=90, seed=3)
    assert {r.reason for r in refs} >= {"gap_1_10s", "gap_long", "detected"}
    for r in refs:
        v = by_id[r.video_id]
        if r.reason in ("gap_1_10s", "gap_long"):
            assert not v.detected[r.frame_idx]
        if r.reason == "detected":
            assert v.detected[r.frame_idx]
        if r.reason == "lorr_candidate":
            assert v.compound.lower() == "fentanyl" and r.frame_idx / FPS >= 300
        if r.reason == "flagged_tail":
            assert v.flagged_from_s is not None and r.frame_idx / FPS >= v.flagged_from_s


def test_gap_1_10s_is_the_largest_bucket_and_lorr_candidates_are_mined() -> None:
    videos, split = _fleet()
    refs = sample_frames(videos, split, n_frames=90, seed=3)
    counts = {reason: sum(r.reason == reason for r in refs) for reason in {r.reason for r in refs}}
    assert counts["gap_1_10s"] == max(counts.values())
    assert counts.get("lorr_candidate", 0) > 0 and counts.get("flagged_tail", 0) > 0


def test_a_shortfall_in_one_bucket_is_filled_from_the_others() -> None:
    videos = [_video(f"F_{i:04d}", "Veh") for i in range(6)]  # no gaps, no flags, no fentanyl
    split = {v.video_id: "train" for v in videos}
    refs = sample_frames(videos, split, n_frames=30, seed=1)
    assert len(refs) == 30 and {r.reason for r in refs} == {"detected"}


def test_compounds_are_spread() -> None:
    videos, split = _fleet()
    by_id = {v.video_id: v.compound for v in videos}
    refs = sample_frames(videos, split, n_frames=60, seed=3)
    assert {by_id[r.video_id] for r in refs} == {"Veh", "Fentanyl", "MDMA"}


def test_more_frames_than_available_returns_what_exists() -> None:
    v = _video("F_0001", "Veh", n=int(FPS * 10))
    refs = sample_frames([v], {"F_0001": "train"}, n_frames=500, seed=1, min_gap_s=2.0, max_per_video=6)
    assert 0 < len(refs) <= 6


def test_invalid_arguments_are_rejected() -> None:
    videos, split = _fleet()
    with pytest.raises(ValueError):
        sample_frames(videos, split, n_frames=0, seed=1)
    with pytest.raises(ValueError):
        sample_frames(videos, {}, n_frames=10, seed=1)  # videos missing from the split
