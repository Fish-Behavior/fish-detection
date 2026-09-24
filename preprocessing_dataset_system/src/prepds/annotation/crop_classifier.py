"""Crop-level 'inverted fish' (Listing/LORR) classifier (Phase 15). Needs the `ml` extra.

Keypoint tilt from the detector did not separate Listing from normal fish on held-out videos, so this asks the
question directly: given the crop around the fish box, is the fish belly-up? Trained on the human labels only
(`yes`/`no`; `unsure` and untagged frames are left out), with box jitter so it copes with the detector's boxes.
Augmentation is horizontal flip plus brightness/contrast; never a vertical flip, which would turn a normal fish
into an inverted one.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from prepds.annotation.examples import Example

DEFAULT_SIZE = 160
MIN_WINDOW_PX = 8.0  # a zero-area detector box must still give a croppable window
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_window(
    box: Sequence[float], *, margin: float = 0.25, jitter: float = 0.0, rng: random.Random | None = None
) -> tuple[float, float, float, float]:
    """Square window centred on the box, side = longest box side * (1 + margin); `jitter` (fraction of the side)
    randomly shifts the centre and rescales the side, imitating detector box error."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(max(box[2] - box[0], box[3] - box[1]) * (1 + margin), MIN_WINDOW_PX)
    if jitter and rng is not None:
        cx += rng.uniform(-jitter, jitter) * side
        cy += rng.uniform(-jitter, jitter) * side
        side *= 1 + rng.uniform(-jitter, jitter)
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def _crop_image(image: Image.Image, window: Sequence[float], size: int) -> np.ndarray:
    crop = image.convert("RGB").crop(tuple(round(v) for v in window)).resize((size, size), Image.BILINEAR)
    return np.asarray(crop, dtype=np.float32) / 255.0


def _crop_array(path: Path, window: Sequence[float], size: int) -> np.ndarray:
    with Image.open(path) as image:
        return _crop_image(image, window, size)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(((array - _MEAN) / _STD).transpose(2, 0, 1)))


class CropDataset(Dataset):
    def __init__(self, examples: Sequence[Example], *, train: bool, size: int = DEFAULT_SIZE, seed: int = 0,
                 jitter: float = 0.08, margin: float = 0.25) -> None:
        self.items = [e for e in examples if e.listing in ("yes", "no") and e.box is not None]
        self.train, self.size, self.jitter, self.margin = train, size, jitter, margin
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.items[index]
        rng = self._rng if self.train else None
        array = _crop_array(example.image_path, crop_window(example.box, margin=self.margin, jitter=self.jitter, rng=rng), self.size)
        if self.train:
            if rng.random() < 0.5:
                array = array[:, ::-1]
            array = np.clip((array - 0.5) * rng.uniform(0.75, 1.25) + 0.5 + rng.uniform(-0.1, 0.1), 0.0, 1.0)
        return _to_tensor(array.astype(np.float32)), torch.tensor(1.0 if example.listing == "yes" else 0.0)


def build_classifier(*, pretrained: bool) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def train_classifier(
    examples: Sequence[Example], run_dir: Path, *, epochs: int, batch_size: int, lr: float, pretrained: bool,
    device: str, size: int = DEFAULT_SIZE, seed: int = 0,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True)  # FileExistsError: a run name is never reused
    torch.manual_seed(seed)
    data = CropDataset(examples, train=True, size=size, seed=seed)
    n_positive = sum(e.listing == "yes" for e in data.items)
    if not 0 < n_positive < len(data):
        raise ValueError("training needs both 'yes' and 'no' examples")
    model = build_classifier(pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(data) - n_positive) / n_positive, device=device))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images.to(device)).squeeze(1), labels.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss) * len(labels)
        history.append(total / len(data))
    weights_tmp = run_dir / "model.pt.tmp"
    torch.save(model.state_dict(), weights_tmp)
    os.replace(weights_tmp, run_dir / "model.pt")
    info = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "pretrained": pretrained, "seed": seed, "size": size,
            "n_train": len(data), "n_positive": n_positive, "train_frame_ids": [e.frame_id for e in data.items],
            "loss_history": history}
    (run_dir / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def load_classifier(run_dir: Path, *, device: str) -> tuple[nn.Module, int]:
    info = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    model = build_classifier(pretrained=False)
    model.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location=device))
    return model.to(device).eval(), int(info["size"])


@torch.no_grad()
def score_images(model: nn.Module, items: Sequence[tuple[Image.Image, Sequence[float]]], *, size: int, device: str,
                 margin: float = 0.25) -> list[float]:
    """Probability that the fish in each (PIL image, box) is inverted."""
    model.eval()
    if not items:
        return []
    batch = torch.stack([_to_tensor(_crop_image(image, crop_window(box, margin=margin), size)) for image, box in items])
    return torch.sigmoid(model(batch.to(device)).squeeze(1)).cpu().tolist()


def score_boxes(model: nn.Module, items: Sequence[tuple[Path, Sequence[float]]], *, size: int, device: str,
                margin: float = 0.25) -> list[float]:
    """Probability that the fish in each (image path, box) is inverted."""
    images = []
    for path, box in items:
        with Image.open(path) as image:
            images.append((image.convert("RGB"), box))
    return score_images(model, images, size=size, device=device, margin=margin)
