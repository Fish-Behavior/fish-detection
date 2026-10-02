"""Fine-tune a torchvision Keypoint R-CNN (BSD) for fish box + 5 keypoints (Phase 15). Needs the `ml` extra.

One model, one annotation pass: the box replaces the classical tracker's centroid, and the keypoints (incl. the
dorsal and ventral points) give the tilt that Listing/LORR needs. Only the frozen *train* split is ever read here;
evaluation on the held-out split happens in `scripts/phase15_finetune.py` and is reported separately.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

from prepds.annotation.examples import Example, hflip
from prepds.annotation.store import KEYPOINT_NAMES

NUM_KEYPOINTS = len(KEYPOINT_NAMES)
DEFAULT_MIN_SIZE, DEFAULT_MAX_SIZE = 600, 800  # the frames are 320x240: upsample so a small fish spans enough pixels


def build_model(*, pretrained: bool, min_size: int = DEFAULT_MIN_SIZE, max_size: int = DEFAULT_MAX_SIZE):
    """COCO-pretrained (person keypoints) backbone with fresh heads: background+fish and 5 fish keypoints."""
    if pretrained:
        model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT, min_size=min_size, max_size=max_size)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
        model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
            model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels, NUM_KEYPOINTS
        )
        return model
    return keypointrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=2, num_keypoints=NUM_KEYPOINTS, min_size=min_size, max_size=max_size
    )


class FishDataset(Dataset):
    """Frames as [0,1] CHW tensors with torchvision detection targets. Training mode augments with a horizontal
    flip and brightness/contrast jitter only (never a vertical flip: it would swap dorsal and ventral)."""

    def __init__(self, examples: Sequence[Example], *, train: bool, seed: int = 0) -> None:
        self.examples = list(examples)
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        if self.train and self.rng.random() < 0.5:
            example = hflip(example)
        with Image.open(example.image_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        if self.train:
            array = np.clip((array - 0.5) * self.rng.uniform(0.8, 1.2) + 0.5 + self.rng.uniform(-0.1, 0.1), 0.0, 1.0)
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        return tensor, target_of(example)


def target_of(example: Example) -> dict[str, torch.Tensor]:
    if example.box is None:
        return {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64),
                "keypoints": torch.zeros((0, NUM_KEYPOINTS, 3), dtype=torch.float32)}
    keypoints = [list(example.keypoints.get(name, (0.0, 0.0, 0))) for name in KEYPOINT_NAMES]
    return {"boxes": torch.tensor([example.box], dtype=torch.float32), "labels": torch.ones((1,), dtype=torch.int64),
            "keypoints": torch.tensor([keypoints], dtype=torch.float32)}


def _collate(batch):
    return tuple(zip(*batch))


def train(
    examples: Sequence[Example],
    run_dir: Path,
    *,
    epochs: int,
    batch_size: int = 4,
    lr: float = 0.005,
    pretrained: bool = True,
    device: str = "cuda",
    seed: int = 0,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
    log=print,
) -> dict[str, Any]:
    """Train on `examples` (the caller passes the train split only); writes model.pt + run.json to a new `run_dir`."""
    if not examples:
        raise ValueError("no training examples")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)  # a run is never overwritten
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    model = build_model(pretrained=pretrained, min_size=min_size, max_size=max_size).to(device)
    loader = DataLoader(FishDataset(examples, train=True, seed=seed), batch_size=batch_size, shuffle=True,
                        collate_fn=_collate, generator=torch.Generator().manual_seed(seed))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    history: list[float] = []
    started = time.time()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for images, targets in loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = sum(model(images, targets).values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss)
        scheduler.step()
        history.append(total / len(loader))
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            log(f"epoch {epoch + 1}/{epochs} loss {history[-1]:.3f} ({time.time() - started:.0f}s)")
    torch.save(model.state_dict(), run_dir / "model.pt")
    info = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "pretrained": pretrained, "seed": seed,
            "min_size": min_size, "max_size": max_size, "n_train": len(examples),
            "train_frame_ids": [e.frame_id for e in examples], "loss_history": history}
    (run_dir / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def load_model(run_dir: Path, *, device: str = "cpu"):
    info = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    model = build_model(pretrained=False, min_size=info["min_size"], max_size=info["max_size"])
    model.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, examples: Sequence[Example], *, device: str = "cpu", min_score: float = 0.3) -> list[dict[str, Any] | None]:
    """Best detection per frame above `min_score`, or None."""
    model.eval()
    results: list[dict[str, Any] | None] = []
    for example in examples:
        with Image.open(example.image_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).to(device)
        out = model([tensor])[0]
        if len(out["scores"]) == 0 or float(out["scores"][0]) < min_score:
            results.append(None)
            continue
        keypoints = out["keypoints"][0].cpu().numpy()
        results.append({
            "box": tuple(float(v) for v in out["boxes"][0].cpu().tolist()),
            "score": float(out["scores"][0]),
            "keypoints": {name: (float(keypoints[i][0]), float(keypoints[i][1])) for i, name in enumerate(KEYPOINT_NAMES)},
        })
    return results
