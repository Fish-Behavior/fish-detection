"""Mining likely-inverted (Listing) frames for the next labeling batch (Phase 15)."""

from __future__ import annotations

import pytest

from prepds.annotation.mining import Scanned, select_candidates

FPS = 30.0


def _s(video, frame, tilt, split="train", compound="DOB", score=0.9):
    return Scanned(video, frame, FPS, compound, split, score, tilt, (10.0, 10.0, 100.0, 80.0))


def _fleet():
    rows = []
    for v in range(6):
        for k in range(10):
            rows.append(_s(f"F_{v:04d}", k * 900, 130 + v + k, "train" if v < 4 else "heldout", "DOB" if v % 2 else "FD-2-67"))
        for k in range(10, 20):
            rows.append(_s(f"F_{v:04d}", k * 900, 80.0, "train" if v < 4 else "heldout", "DOB" if v % 2 else "FD-2-67"))
        for k in range(20, 30):
            rows.append(_s(f"F_{v:04d}", k * 900, 5.0, "train" if v < 4 else "heldout", "DOB" if v % 2 else "FD-2-67"))
    return rows


def test_candidates_have_a_high_tilt_and_ambiguous_ones_a_middling_tilt() -> None:
    picked = select_candidates(_fleet(), {}, quotas={"train": (6, 3), "heldout": (4, 2)}, seed=1)
    for c in picked:
        if c.reason == "listing_candidate":
            assert c.tilt >= 100.0
        else:
            assert c.reason == "listing_ambiguous" and 60.0 <= c.tilt < 100.0


def test_quotas_are_met_per_split_and_splits_are_kept() -> None:
    picked = select_candidates(_fleet(), {}, quotas={"train": (6, 3), "heldout": (4, 2)}, seed=1)
    count = lambda split, reason: sum(c.split == split and c.reason == reason for c in picked)  # noqa: E731
    assert (count("train", "listing_candidate"), count("train", "listing_ambiguous")) == (6, 3)
    assert (count("heldout", "listing_candidate"), count("heldout", "listing_ambiguous")) == (4, 2)
    fleet_split = {s.video_id: s.split for s in _fleet()}
    assert all(fleet_split[c.video_id] == c.split for c in picked)


def test_per_video_cap_and_spacing() -> None:
    picked = select_candidates(_fleet(), {}, quotas={"train": (12, 4), "heldout": (6, 2)}, seed=1, max_per_video=3, min_gap_s=20.0)
    by_video: dict[str, list[int]] = {}
    for c in picked:
        by_video.setdefault(c.video_id, []).append(c.frame_idx)
    for frames in by_video.values():
        assert len(frames) <= 3
        ordered = sorted(frames)
        assert all(b - a >= 20 * FPS for a, b in zip(ordered, ordered[1:]))


def test_frames_near_existing_samples_are_excluded() -> None:
    existing = {"F_0000": [9 * 900]}  # the best F_0000 frame is already sampled
    picked = select_candidates(_fleet(), existing, quotas={"train": (20, 0), "heldout": (0, 0)}, seed=1)
    assert all(not (c.video_id == "F_0000" and abs(c.frame_idx - 9 * 900) <= 2 * FPS) for c in picked)


def test_low_confidence_boxes_are_ignored() -> None:
    rows = [_s("F_0001", 0, 170.0, score=0.2), _s("F_0001", 9000, 170.0, score=0.95)]
    picked = select_candidates(rows, {}, quotas={"train": (5, 0), "heldout": (0, 0)}, seed=1)
    assert [c.frame_idx for c in picked] == [9000]


def test_highest_tilt_first_and_compounds_are_interleaved() -> None:
    picked = [c for c in select_candidates(_fleet(), {}, quotas={"train": (4, 0), "heldout": (0, 0)}, seed=1, max_per_video=1)
              if c.reason == "listing_candidate"]
    compounds = {"DOB" if int(c.video_id[-1]) % 2 else "FD-2-67" for c in picked}
    assert compounds == {"DOB", "FD-2-67"}


def test_deterministic_for_a_seed_and_input_order() -> None:
    a = select_candidates(_fleet(), {}, quotas={"train": (6, 3), "heldout": (4, 2)}, seed=3)
    assert a == select_candidates(list(reversed(_fleet())), {}, quotas={"train": (6, 3), "heldout": (4, 2)}, seed=3)


def test_fewer_available_than_requested_returns_what_exists() -> None:
    picked = select_candidates([_s("F_0001", 0, 170.0)], {}, quotas={"train": (10, 5), "heldout": (3, 3)}, seed=1)
    assert len(picked) == 1
