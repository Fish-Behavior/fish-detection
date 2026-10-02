"""Annotation records for the tracker fine-tuning set (Phase 15)."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest

from prepds.annotation.store import (
    KEYPOINT_NAMES,
    InvalidAnnotation,
    load_annotation,
    parse_annotation,
    save_annotation,
)

W, H = 320, 240
NOW = dt.datetime(2026, 9, 23, 12, tzinfo=dt.timezone.utc)


def _payload(**overrides):
    payload = {
        "fish_visible": True,
        "box": [50, 60, 200, 150],
        "keypoints": {"snout": [60, 100, 2], "tail_tip": [190, 110, 1]},
        "listing": "no",
        "annotator": "lk",
    }
    payload.update(overrides)
    return payload


def test_valid_annotation_round_trips(tmp_path: Path) -> None:
    annotation = parse_annotation(_payload(), width=W, height=H, now=NOW)
    save_annotation(tmp_path, "F_0001_000010", annotation)
    loaded = load_annotation(tmp_path, "F_0001_000010")
    assert loaded == annotation and loaded.box == (50.0, 60.0, 200.0, 150.0)
    assert loaded.keypoints["snout"] == (60.0, 100.0, 2)


def test_missing_annotation_loads_as_none(tmp_path: Path) -> None:
    assert load_annotation(tmp_path, "F_0001_000010") is None


def test_no_fish_visible_has_no_box_keypoints_or_listing() -> None:
    a = parse_annotation({"fish_visible": False, "annotator": "lk"}, width=W, height=H, now=NOW)
    assert a.box is None and a.keypoints == {} and a.listing is None
    for extra in ({"box": [1, 1, 10, 10]}, {"keypoints": {"snout": [1, 1, 2]}}, {"listing": "no"}):
        with pytest.raises(InvalidAnnotation):
            parse_annotation({"fish_visible": False, "annotator": "lk", **extra}, width=W, height=H, now=NOW)


@pytest.mark.parametrize(
    "bad",
    [
        {"box": None},
        {"box": [50, 60, 50, 150]},  # zero width
        {"box": [50, 60, 200, 61]},  # too thin
        {"box": [-5, 60, 200, 150]},
        {"box": [50, 60, 400, 150]},
        {"box": [50, 60, 200]},
        {"box": [50, 60, float("nan"), 150]},
        {"keypoints": {"whisker": [10, 10, 2]}},
        {"keypoints": {"snout": [10, 10, 3]}},
        {"keypoints": {"snout": [999, 10, 2]}},
        {"keypoints": {"snout": [10, 10]}},
        {"keypoints": {"snout": [float("inf"), 10, 2]}},
        {"listing": "maybe"},
        {"listing": None},
        {"annotator": "  "},
        {"annotator": "a\nb"},
    ],
)
def test_invalid_annotations_are_rejected(bad) -> None:
    with pytest.raises(InvalidAnnotation):
        parse_annotation(_payload(**bad), width=W, height=H, now=NOW)


def test_unlabeled_keypoint_is_dropped_and_the_names_are_the_listing_schema() -> None:
    a = parse_annotation(_payload(keypoints={"snout": [0, 0, 0], "tail_base": [100, 100, 2]}), width=W, height=H, now=NOW)
    assert set(a.keypoints) == {"tail_base"}
    assert KEYPOINT_NAMES == ("snout", "dorsal_fin_base", "ventral", "tail_base", "tail_tip")


def test_unknown_stem_characters_are_rejected(tmp_path: Path) -> None:
    annotation = parse_annotation(_payload(), width=W, height=H, now=NOW)
    for stem in ("../x", "a/b", "", ".hidden"):
        with pytest.raises(ValueError):
            save_annotation(tmp_path, stem, annotation)
        with pytest.raises(ValueError):
            load_annotation(tmp_path, stem)


def test_save_is_atomic_a_failed_replace_keeps_the_previous_annotation(tmp_path: Path, monkeypatch) -> None:
    first = parse_annotation(_payload(), width=W, height=H, now=NOW)
    save_annotation(tmp_path, "F_0001_000010", first)
    second = parse_annotation(_payload(listing="yes"), width=W, height=H, now=NOW)
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        save_annotation(tmp_path, "F_0001_000010", second)
    monkeypatch.undo()
    assert load_annotation(tmp_path, "F_0001_000010") == first
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_file_is_reported_not_returned_as_valid(tmp_path: Path) -> None:
    (tmp_path / "F_0001_000010.json").write_text("{nope")
    with pytest.raises(ValueError):
        load_annotation(tmp_path, "F_0001_000010")


def test_concurrent_saves_use_distinct_temp_files(tmp_path: Path, monkeypatch) -> None:
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(str(a)), real(a, b))[1])
    annotation = parse_annotation(_payload(), width=W, height=H, now=NOW)
    save_annotation(tmp_path, "F_0001_000010", annotation)
    save_annotation(tmp_path, "F_0001_000010", annotation)
    assert len(set(seen)) == 2
