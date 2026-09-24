"""Phase 15 / T094: train the crop-level 'inverted fish' classifier on TRAIN labels and evaluate it on HELD-OUT.

Usage (from preprocessing_dataset_system/):
    python scripts/phase15_listing_classifier.py --run-name c1 --detector m3 [--epochs 15]
Evaluated twice on held-out yes/no frames: on the human boxes, and on the detector's predicted boxes (what the
pipeline would see). Reports AUC and precision/recall at fixed probability thresholds; no threshold is tuned on
the held-out set. Writes outputs/phase15/listing_models/<run-name>/{model.pt,run.json,metrics.json}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prepds.annotation.crop_classifier import load_classifier, score_boxes, train_classifier
from prepds.annotation.examples import load_examples
from prepds.annotation.train import load_model, predict

WORK = Path("outputs/phase15")
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def summarize(scores: list[float], labels: list[bool]) -> dict:
    s, y = np.array(scores), np.array(labels)
    out = {"n_yes": int(y.sum()), "n_no": int((~y).sum()), "auc": round(auc(s, y), 3), "at_threshold": {}}
    for t in THRESHOLDS:
        tp, fp = int((s[y] >= t).sum()), int((s[~y] >= t).sum())
        out["at_threshold"][str(t)] = {"recall": round(tp / y.sum(), 3), "precision": round(tp / max(1, tp + fp), 3), "false_positives": fp}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--detector", default="m3")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_examples = load_examples(WORK, "train")
    run_dir = WORK / "listing_models" / args.run_name
    info = train_classifier(train_examples, run_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            pretrained=True, device=device, seed=args.seed)
    print(f"trained on {info['n_train']} crops ({info['n_positive']} yes); final loss {info['loss_history'][-1]:.3f}")

    model, size = load_classifier(run_dir, device=device)
    heldout = [e for e in load_examples(WORK, "heldout") if e.listing in ("yes", "no") and e.box is not None]
    labels = [e.listing == "yes" for e in heldout]
    human = score_boxes(model, [(e.image_path, e.box) for e in heldout], size=size, device=device)
    detector = load_model(WORK / "models" / args.detector, device=device)
    predictions = predict(detector, heldout, device=device, min_score=0.3)
    kept = [(e, p) for e, p in zip(heldout, predictions) if p is not None]
    detected = score_boxes(model, [(e.image_path, p["box"]) for e, p in kept], size=size, device=device)
    metrics = {"human_boxes": summarize(human, labels), "detector_boxes": summarize(detected, [e.listing == "yes" for e, _ in kept]),
               "detector": args.detector, "detector_missed": len(heldout) - len(kept)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
