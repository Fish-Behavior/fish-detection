"""Training examples from annotations (Phase 15): only the requested split, never the held-out frames."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from PIL import Image

from prepds.annotation.examples import Example, hflip, load_examples
from prepds.annotation.store import KEYPOINT_NAMES, parse_annotation, save_annotation

NOW = dt.datetime(2026, 9, 23, tzinfo=dt.timezone.utc)


def _work(tmp_path: Path) -> Path:
    work = tmp_path / "phase15"
    (work / "frames").mkdir(parents=True, exist_ok=True)
    samples = []
    for i, split in enumerate(["train", "train", "train", "heldout"]):
        image = f"F_{i:04d}_000010.png"
        Image.new("RGB", (320, 240)).save(work / "frames" / image)
        samples.append({"video_id": f"F_{i:04d}", "frame_idx": 10, "t_sec": 0.3, "reason": "detected", "split": split,
                        "compound": "Veh", "image": image, "classical": {"detected": True, "x": 1, "y": 1}})
    (work / "sample.json").write_text(json.dumps({"seed": 1, "samples": samples}))
    ann = work / "annotations"
    good = {"fish_visible": True, "box": [50, 60, 200, 150], "keypoints": {"snout": [60, 100, 2], "tail_tip": [190, 110, 1]},
            "listing": "yes", "annotator": "lk"}
    save_annotation(ann, "F_0000_000010", parse_annotation(good, width=320, height=240, now=NOW))
    save_annotation(ann, "F_0001_000010", parse_annotation({"fish_visible": False, "annotator": "lk"}, width=320, height=240, now=NOW))
    save_annotation(ann, "F_0003_000010", parse_annotation(good, width=320, height=240, now=NOW))  # held-out
    return work  # F_0002 is unlabeled


def test_only_labeled_frames_of_the_requested_split_are_loaded(tmp_path: Path) -> None:
    examples = load_examples(_work(tmp_path), "train")
    assert [e.frame_id for e in examples] == ["F_0000_000010", "F_0001_000010"]
    assert [e.frame_id for e in load_examples(_work(tmp_path), "heldout")] == ["F_0003_000010"]


def test_positive_and_negative_examples_carry_the_labels(tmp_path: Path) -> None:
    pos, neg = load_examples(_work(tmp_path), "train")
    assert pos.box == (50.0, 60.0, 200.0, 150.0) and pos.listing == "yes"
    assert pos.keypoints["snout"] == (60.0, 100.0, 2)
    assert neg.box is None and neg.keypoints == {} and neg.listing is None
    assert pos.image_path.name == "F_0000_000010.png" and pos.width == 320 and pos.height == 240


def test_unknown_split_and_corrupt_annotation_are_errors(tmp_path: Path) -> None:
    work = _work(tmp_path)
    with pytest.raises(ValueError):
        load_examples(work, "validation")
    (work / "annotations" / "F_0000_000010.json").write_text("{nope")
    with pytest.raises(ValueError):
        load_examples(work, "train")


def test_hflip_mirrors_x_only_and_keeps_visibility() -> None:
    e = Example("f", Path("f.png"), 320, 240, (50.0, 60.0, 200.0, 150.0), {"snout": (60.0, 100.0, 2)}, "no")
    f = hflip(e)
    assert f.box == (120.0, 60.0, 270.0, 150.0)
    assert f.keypoints["snout"] == (260.0, 100.0, 2) and f.listing == "no"
    assert hflip(f) == e
    assert set(KEYPOINT_NAMES) >= set(f.keypoints)


def test_training_examples_include_labeled_batch_frames(tmp_path: Path) -> None:
    work = _work(tmp_path)
    sample = {"video_id": "F_0009", "frame_idx": 90, "t_sec": 3.0, "reason": "listing_candidate", "split": "train",
              "compound": "DOB", "image": "F_0009_000090.png", "classical": {"detected": False, "x": 0.0, "y": 0.0}}
    Image.new("RGB", (320, 240)).save(work / "frames" / sample["image"])
    (work / "batch_002.json").write_text(json.dumps({"seed": 1, "samples": [sample]}))
    good = {"fish_visible": True, "box": [50, 60, 200, 150], "keypoints": {}, "listing": "yes", "annotator": "lk"}
    save_annotation(work / "annotations", "F_0009_000090", parse_annotation(good, width=320, height=240, now=NOW))
    assert "F_0009_000090" in [e.frame_id for e in load_examples(work, "train")]
