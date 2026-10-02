"""Advisory 'possible Listing' flags for the reviewer (Phase 15): merge classifier scores into time ranges,
store them beside the video's artifacts, show them in the review API. They never change states or acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prepds.listing_flags import ListingFlag, merge_flags, read_listing_flags, write_listing_flags
from prepds.webapp.app import create_app
from tests.test_webapp_api import VIDEO_BYTES, _make  # noqa: F401


def test_consecutive_high_scores_merge_into_one_range_padded_by_half_a_step() -> None:
    samples = [(0.0, 0.1), (3.0, 0.95), (6.0, 0.99), (9.0, 0.2), (12.0, 0.92)]
    flags = merge_flags(samples, threshold=0.9, step_s=3.0)
    assert flags == [ListingFlag(1.5, 7.5, 0.99, 2), ListingFlag(10.5, 13.5, 0.92, 1)]


def test_a_gap_of_missing_samples_splits_a_range() -> None:
    flags = merge_flags([(3.0, 0.95), (12.0, 0.95)], threshold=0.9, step_s=3.0)
    assert len(flags) == 2


def test_start_is_never_negative_and_unsorted_input_is_handled() -> None:
    flags = merge_flags([(3.0, 0.95), (0.0, 0.97)], threshold=0.9, step_s=3.0)
    assert flags == [ListingFlag(0.0, 4.5, 0.97, 2)]


def test_no_scores_above_threshold_gives_no_flags() -> None:
    assert merge_flags([(0.0, 0.1), (3.0, 0.5)], threshold=0.9, step_s=3.0) == []
    assert merge_flags([], threshold=0.9, step_s=3.0) == []


def test_sidecar_round_trip_and_missing_file(tmp_path: Path) -> None:
    assert read_listing_flags(tmp_path) == []
    flags = [ListingFlag(1.5, 7.5, 0.99, 2)]
    write_listing_flags(tmp_path, flags, meta={"classifier": "c1"})
    assert read_listing_flags(tmp_path) == flags
    assert json.loads((tmp_path / "listing_flags.json").read_text())["meta"] == {"classifier": "c1"}


def test_a_corrupt_sidecar_is_an_error_not_an_empty_list(tmp_path: Path) -> None:
    (tmp_path / "listing_flags.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        read_listing_flags(tmp_path)


@pytest.fixture
def client(tmp_path: Path):
    processed, source = tmp_path / "processed", tmp_path / "videos"
    processed.mkdir(), source.mkdir()
    video_id = _make(processed, source)
    app = create_app(processed, tmp_path / "accepted", video_dir=source, allowed_hosts=("testserver",))
    return TestClient(app), processed / video_id, video_id


def test_api_lists_listing_flags(client) -> None:
    c, directory, video_id = client
    assert c.get(f"/videos/{video_id}").json()["listing_flags"] == []
    write_listing_flags(directory, [ListingFlag(1.5, 7.5, 0.99, 2)], meta={})
    body = c.get(f"/videos/{video_id}").json()
    assert body["listing_flags"] == [{"start_s": 1.5, "end_s": 7.5, "max_score": 0.99, "n_samples": 2}]
    assert body["listing_flags_error"] is None


def test_api_reports_an_unreadable_sidecar_without_breaking_the_video(client) -> None:
    c, directory, video_id = client
    (directory / "listing_flags.json").write_text("{bad", encoding="utf-8")
    response = c.get(f"/videos/{video_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["listing_flags"] == [] and body["listing_flags_error"]


def test_min_samples_drops_isolated_hits() -> None:
    samples = [(3.0, 0.95), (6.0, 0.99), (12.0, 0.92), (30.0, 0.95)]
    flags = merge_flags(samples, threshold=0.9, step_s=3.0, min_samples=2)
    assert flags == [ListingFlag(1.5, 7.5, 0.99, 2)]


def test_non_finite_values_in_a_sidecar_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "listing_flags.json").write_text(
        '{"meta": {}, "flags": [{"start_s": NaN, "end_s": 5.0, "max_score": 0.9, "n_samples": 2}]}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_listing_flags(tmp_path)
