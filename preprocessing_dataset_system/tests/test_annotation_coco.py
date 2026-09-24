"""COCO-keypoints export of annotated frames (Phase 15)."""

from __future__ import annotations

import datetime as dt

from prepds.annotation.coco import build_coco
from prepds.annotation.store import KEYPOINT_NAMES, parse_annotation

NOW = dt.datetime(2026, 9, 23, tzinfo=dt.timezone.utc)
SIZE = lambda image: (320, 240)  # noqa: E731
SAMPLES = [
    {"image": "F_0001_000010.png", "video_id": "F_0001", "frame_idx": 10, "split": "train", "compound": "Veh"},
    {"image": "F_0002_000020.png", "video_id": "F_0002", "frame_idx": 20, "split": "train", "compound": "MDMA"},
    {"image": "F_0003_000030.png", "video_id": "F_0003", "frame_idx": 30, "split": "heldout", "compound": "Veh"},
    {"image": "F_0004_000040.png", "video_id": "F_0004", "frame_idx": 40, "split": "train", "compound": "Veh"},
]


def _ann(**kw):
    payload = {"fish_visible": True, "box": [10, 20, 110, 80], "keypoints": {"snout": [15, 40, 2], "tail_tip": [100, 50, 1]},
               "listing": "yes", "annotator": "lk"}
    payload.update(kw)
    return parse_annotation(payload, width=320, height=240, now=NOW)


def _annotations():
    return {
        "F_0001_000010": _ann(),
        "F_0002_000020": parse_annotation({"fish_visible": False, "annotator": "lk"}, width=320, height=240, now=NOW),
        "F_0003_000030": _ann(listing="no"),
        # F_0004 has no annotation yet
    }


def test_only_annotated_frames_of_the_requested_split_are_exported() -> None:
    coco = build_coco(SAMPLES, _annotations(), split_name="train", size_of=SIZE)
    assert [i["file_name"] for i in coco["images"]] == ["F_0001_000010.png", "F_0002_000020.png"]


def test_visible_fish_becomes_a_keypoint_annotation() -> None:
    coco = build_coco(SAMPLES, _annotations(), split_name="train", size_of=SIZE)
    (ann,) = coco["annotations"]
    assert ann["bbox"] == [10.0, 20.0, 100.0, 60.0] and ann["area"] == 6000.0 and ann["iscrowd"] == 0
    assert len(ann["keypoints"]) == 3 * len(KEYPOINT_NAMES) and ann["num_keypoints"] == 2
    assert ann["keypoints"][:3] == [15.0, 40.0, 2] and ann["keypoints"][3:6] == [0, 0, 0]
    assert ann["listing"] == "yes"
    assert ann["image_id"] == coco["images"][0]["id"]


def test_no_fish_frames_are_kept_as_images_without_annotations() -> None:
    coco = build_coco(SAMPLES, _annotations(), split_name="train", size_of=SIZE)
    negative = coco["images"][1]
    assert not [a for a in coco["annotations"] if a["image_id"] == negative["id"]]


def test_categories_carry_the_keypoint_schema() -> None:
    coco = build_coco(SAMPLES, _annotations(), split_name="heldout", size_of=SIZE)
    (category,) = coco["categories"]
    assert category["name"] == "fish" and category["keypoints"] == list(KEYPOINT_NAMES)
    assert len(coco["images"]) == 1 and coco["annotations"][0]["listing"] == "no"


def test_each_image_gets_its_own_size() -> None:
    sizes = {"F_0001_000010.png": (320, 240), "F_0002_000020.png": (192, 240)}
    coco = build_coco(SAMPLES, _annotations(), split_name="train", size_of=lambda image: sizes[image])
    assert [(i["width"], i["height"]) for i in coco["images"]] == [(320, 240), (192, 240)]
