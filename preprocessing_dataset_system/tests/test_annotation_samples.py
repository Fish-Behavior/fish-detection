"""Loading the sampled frames: the original sample.json plus any later labeling batches (Phase 15)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prepds.annotation.samples import load_samples


def _sample(video: str, frame: int, split="train", **extra):
    return {"video_id": video, "frame_idx": frame, "t_sec": frame / 30, "reason": "detected", "split": split,
            "compound": "Veh", "image": f"{video}_{frame:06d}.png", "classical": {"detected": True, "x": 1.0, "y": 1.0}, **extra}


def _write(path: Path, samples) -> None:
    path.write_text(json.dumps({"seed": 1, "samples": samples}))


def test_original_sample_file_alone(tmp_path: Path) -> None:
    _write(tmp_path / "sample.json", [_sample("F_0001", 10)])
    (s,) = load_samples(tmp_path)
    assert s["batch"] == "sample" and s["image"] == "F_0001_000010.png"


def test_batches_are_appended_in_name_order_and_tagged(tmp_path: Path) -> None:
    _write(tmp_path / "sample.json", [_sample("F_0001", 10)])
    _write(tmp_path / "batch_003.json", [_sample("F_0003", 30)])
    _write(tmp_path / "batch_002.json", [_sample("F_0002", 20, prefill={"score": 0.9, "box": [1, 2, 30, 40]})])
    samples = load_samples(tmp_path)
    assert [(s["batch"], s["video_id"]) for s in samples] == [("sample", "F_0001"), ("batch_002", "F_0002"), ("batch_003", "F_0003")]
    assert samples[1]["prefill"]["score"] == 0.9


def test_a_frame_in_two_files_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path / "sample.json", [_sample("F_0001", 10)])
    _write(tmp_path / "batch_002.json", [_sample("F_0001", 10)])
    with pytest.raises(ValueError, match="duplicate"):
        load_samples(tmp_path)


def test_a_video_may_not_change_split_between_files(tmp_path: Path) -> None:
    _write(tmp_path / "sample.json", [_sample("F_0001", 10, split="train")])
    _write(tmp_path / "batch_002.json", [_sample("F_0001", 500, split="heldout")])
    with pytest.raises(ValueError, match="split"):
        load_samples(tmp_path)


def test_unrelated_json_files_are_ignored(tmp_path: Path) -> None:
    _write(tmp_path / "sample.json", [_sample("F_0001", 10)])
    (tmp_path / "split.json").write_text("{}")
    (tmp_path / "pilot_owlv2.json").write_text("[]")
    assert len(load_samples(tmp_path)) == 1
