"""Phase 15: fine-tune the fish box+keypoint model on the labeled TRAIN frames and evaluate it.

Usage (from preprocessing_dataset_system/, needs the [ml] extra and CUDA torch):
    python scripts/phase15_finetune.py --run-name r1 [--epochs 60] [--batch-size 4] [--lr 0.005]
Writes outputs/phase15/models/<run-name>/{model.pt,run.json,metrics.json}; a run name is never reused.
Only the frozen train split is used for training. Metrics are computed on the labeled HELD-OUT frames when there
are any; otherwise on the training frames, and the output says so loudly (that is a plumbing check, not an
evaluation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prepds.annotation.examples import load_examples
from prepds.annotation.metrics import evaluate
from prepds.annotation.train import load_model, predict, train

WORK = Path("outputs/phase15")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--from-scratch", action="store_true", help="do not use the COCO-pretrained weights")
    args = parser.parse_args()

    train_examples = load_examples(WORK, "train")
    heldout = load_examples(WORK, "heldout")
    print(f"{len(train_examples)} labeled train frames, {len(heldout)} labeled held-out frames")
    run_dir = WORK / "models" / args.run_name
    train(train_examples, run_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          pretrained=not args.from_scratch, device=args.device, seed=args.seed)

    model = load_model(run_dir, device=args.device)
    eval_set, label = (heldout, "HELD-OUT") if heldout else (train_examples, "TRAIN-SET (not a real evaluation)")
    metrics = evaluate(eval_set, predict(model, eval_set, device=args.device))
    metrics["evaluated_on"] = label
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    print(f"\nevaluated on: {label}  (n_positive={metrics['n_positive']}, n_negative={metrics['n_negative']})")
    print(f"detection rate {metrics['detection_rate']}  recall@IoU0.5 {metrics['recall_at_iou_0.5']}  mean IoU {metrics['mean_iou']}")
    for name, s in metrics["keypoint_error_px"].items():
        print(f"  {name:16s} n={s['n']:3d} mean error {s['mean'] if s['mean'] is None else round(s['mean'], 1)} px")
    for tag, s in metrics["tilt_deg_by_listing"].items():
        print(f"  listing={tag}: n={s['n']} tilt gt {s['gt_mean']} pred {s['pred_mean']}")


if __name__ == "__main__":
    main()
