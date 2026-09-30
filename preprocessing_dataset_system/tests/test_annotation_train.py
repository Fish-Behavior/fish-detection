"""Smoke tests for the fine-tuning code (Phase 15). CPU, no pretrained weights, tiny images: they check the
plumbing (targets, flips, a training step, checkpoint round trip), not model quality."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from prepds.annotation.examples import Example  # noqa: E402
from prepds.annotation.train import FishDataset, build_model, load_model, predict, target_of, train  # noqa: E402

KP = {"snout": (20.0, 50.0, 2), "dorsal_fin_base": (50.0, 30.0, 2), "ventral": (50.0, 70.0, 1)}


def _examples(tmp_path: Path) -> list[Example]:
    out = []
    for i in range(3):
        path = tmp_path / f"f{i}.png"
        Image.new("RGB", (128, 96), (100 + i, 120, 90)).save(path)
        out.append(Example(f"f{i}", path, 128, 96, (10.0, 20.0, 100.0, 80.0) if i < 2 else None, KP if i < 2 else {}, "no" if i < 2 else None))
    return out


def test_positive_target_has_box_label_and_five_keypoints(tmp_path: Path) -> None:
    t = target_of(_examples(tmp_path)[0])
    assert t["boxes"].shape == (1, 4) and t["labels"].tolist() == [1] and t["keypoints"].shape == (1, 5, 3)
    assert t["keypoints"][0, 3].tolist() == [0.0, 0.0, 0.0]  # unlabeled tail_base has visibility 0
    assert t["keypoints"][0, 2].tolist() == [50.0, 70.0, 1.0]


def test_negative_target_is_empty(tmp_path: Path) -> None:
    t = target_of(_examples(tmp_path)[2])
    assert t["boxes"].shape == (0, 4) and t["labels"].shape == (0,) and t["keypoints"].shape == (0, 5, 3)


def test_dataset_never_flips_vertically_and_eval_mode_is_unaugmented(tmp_path: Path) -> None:
    examples = _examples(tmp_path)
    plain = FishDataset(examples, train=False)
    image, target = plain[0]
    assert image.shape == (3, 96, 128) and target["boxes"][0].tolist() == [10.0, 20.0, 100.0, 80.0]
    augmented = FishDataset(examples, train=True, seed=1)
    for _ in range(20):
        _, t = augmented[0]
        y0, y1 = t["boxes"][0][1].item(), t["boxes"][0][3].item()
        assert (y0, y1) == (20.0, 80.0)  # only x can change


def test_two_epochs_train_save_and_reload(tmp_path: Path) -> None:
    examples = _examples(tmp_path)
    info = train(examples, tmp_path / "run", epochs=2, batch_size=2, pretrained=False, device="cpu", min_size=128, max_size=160, log=lambda *_: None)
    assert len(info["loss_history"]) == 2 and info["n_train"] == 3
    assert (tmp_path / "run" / "model.pt").is_file()
    model = load_model(tmp_path / "run")
    preds = predict(model, examples, min_score=0.0)
    assert len(preds) == 3 and all(p is None or len(p["keypoints"]) == 5 for p in preds)


def test_a_run_directory_is_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        train(_examples(tmp_path), tmp_path / "run", epochs=1, pretrained=False, device="cpu", log=lambda *_: None)


def test_training_without_examples_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        train([], tmp_path / "run", epochs=1, pretrained=False, device="cpu")


def test_model_head_shapes() -> None:
    model = build_model(pretrained=False, min_size=128, max_size=160)
    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.roi_heads.keypoint_predictor.kps_score_lowres.out_channels == 5
