"""Crop-level 'inverted fish' classifier (Phase 15). CPU, tiny images, no pretrained weights: plumbing only."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from prepds.annotation.crop_classifier import (  # noqa: E402
    CropDataset,
    crop_window,
    load_classifier,
    score_boxes,
    train_classifier,
)
from prepds.annotation.examples import Example  # noqa: E402

BOX = (40.0, 30.0, 100.0, 70.0)


def _example(tmp_path: Path, name: str, listing: str | None, box=BOX, top_bright=True) -> Example:
    array = np.zeros((96, 128, 3), dtype=np.uint8)
    array[:48 if top_bright else 96, :, :] = 200 if top_bright else 30
    path = tmp_path / f"{name}.png"
    Image.fromarray(array).save(path)
    return Example(name, path, 128, 96, box, {}, listing)


def test_window_is_square_centered_and_padded_by_the_margin() -> None:
    x0, y0, x1, y1 = crop_window(BOX, margin=0.25)
    assert x1 - x0 == pytest.approx(y1 - y0)
    assert (x0 + x1) / 2 == pytest.approx(70.0) and (y0 + y1) / 2 == pytest.approx(50.0)
    assert x1 - x0 == pytest.approx(60.0 * 1.25)


def test_jitter_is_bounded_and_deterministic_for_a_seed() -> None:
    a = crop_window(BOX, margin=0.25, jitter=0.1, rng=random.Random(4))
    assert a == crop_window(BOX, margin=0.25, jitter=0.1, rng=random.Random(4))
    base = crop_window(BOX, margin=0.25)
    assert all(abs(p - q) <= 0.1 * 60.0 * 1.25 * 1.5 for p, q in zip(a, base))
    assert a != base


def test_dataset_keeps_only_yes_and_no_examples_with_a_box(tmp_path: Path) -> None:
    examples = [
        _example(tmp_path, "a", "yes"), _example(tmp_path, "b", "no"), _example(tmp_path, "c", "unsure"),
        _example(tmp_path, "d", None), _example(tmp_path, "e", "yes", box=None),
    ]
    data = CropDataset(examples, train=False, size=32)
    assert len(data) == 2
    assert [float(data[i][1]) for i in range(2)] == [1.0, 0.0]


def test_augmentation_never_flips_vertically(tmp_path: Path) -> None:
    example = _example(tmp_path, "a", "no")  # bright top half, dark bottom half
    data = CropDataset([example], train=True, size=32, seed=1, jitter=0.0)
    for _ in range(20):
        image, _label = data[0]
        # top rows stay brighter than bottom rows however brightness/contrast/hflip were drawn
        assert image[:, :6, :].mean() > image[:, -6:, :].mean()


def test_training_writes_a_run_once_and_scores_are_probabilities(tmp_path: Path) -> None:
    examples = [_example(tmp_path, f"p{i}", "yes", top_bright=False) for i in range(4)] + [
        _example(tmp_path, f"n{i}", "no") for i in range(4)
    ]
    run = tmp_path / "run"
    info = train_classifier(examples, run, epochs=2, batch_size=4, lr=1e-3, pretrained=False, device="cpu", size=32, seed=0)
    assert (run / "model.pt").is_file() and info["n_train"] == 8 and info["n_positive"] == 4
    with pytest.raises(FileExistsError):
        train_classifier(examples, run, epochs=1, batch_size=4, lr=1e-3, pretrained=False, device="cpu", size=32)
    model, size = load_classifier(run, device="cpu")
    probabilities = score_boxes(model, [(examples[0].image_path, BOX), (examples[5].image_path, BOX)], size=size, device="cpu")
    assert len(probabilities) == 2 and all(0.0 <= p <= 1.0 for p in probabilities)


def test_crop_near_the_image_edge_is_padded_not_an_error(tmp_path: Path) -> None:
    example = _example(tmp_path, "edge", "yes", box=(0.0, 0.0, 60.0, 40.0))
    image, label = CropDataset([example], train=True, size=32, seed=2, jitter=0.2)[0]
    assert image.shape == (3, 32, 32) and float(label) == 1.0
    assert score_boxes(*_tiny_model(), [(example.image_path, (100.0, 60.0, 128.0, 96.0))], size=32, device="cpu")


def _tiny_model():
    from prepds.annotation.crop_classifier import build_classifier

    return (build_classifier(pretrained=False).eval(),)


def test_a_degenerate_box_still_gives_a_usable_window() -> None:
    x0, y0, x1, y1 = crop_window((50.0, 40.0, 50.0, 40.0))
    assert x1 - x0 >= 8.0 and y1 - y0 >= 8.0


def test_a_zero_area_box_scores_without_crashing(tmp_path: Path) -> None:
    example = _example(tmp_path, "z", "yes")
    assert len(score_boxes(*_tiny_model(), [(example.image_path, (50.0, 40.0, 50.0, 40.0))], size=32, device="cpu")) == 1
