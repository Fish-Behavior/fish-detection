"""Phase 15: mine likely-Listing (inverted fish) frames into a new labeling batch.

Usage (from preprocessing_dataset_system/):
    python scripts/phase15_mine_listing.py --run r1 [--stride 180] [--batch batch_002]
Scans every `--stride`-th frame of every processed video with the fine-tuned model (results cached in
outputs/phase15/scan_<run>.parquet, so re-running only re-selects), picks frames with a large predicted
dorsal->ventral tilt, and writes outputs/phase15/<batch>.json plus the frame PNGs. The box (not the keypoints) is
stored as the label prefill; the annotator's answer, not the model's tilt, is the label.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from prepds.annotation.inputs import extract_frame, load_video_infos
from prepds.annotation.metrics import tilt_deg
from prepds.annotation.mining import Scanned, select_candidates
from prepds.annotation.samples import load_samples
from prepds.annotation.split import load_split
from prepds.annotation.train import load_model
from prepds.review_store import load_manifest

OUT = Path("outputs/phase15")
PROCESSED = Path("outputs/processed")
MIN_SIZE = 400


@torch.no_grad()
def scan_video(model, video_path: Path, stride: int, device: str) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    rows: list[dict] = []
    try:
        n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in range(stride // 2, n, stride):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"could not read frame {index} of {video_path}")
            tensor = torch.from_numpy(np.ascontiguousarray(frame[:, :, ::-1])).permute(2, 0, 1).to(device).float() / 255.0
            out = model([tensor])[0]
            if len(out["scores"]) == 0:
                continue
            kp = out["keypoints"][0].cpu().numpy()
            rows.append({"frame_idx": index, "box_score": float(out["scores"][0]),
                         "box": [float(v) for v in out["boxes"][0].cpu().tolist()],
                         "tilt": tilt_deg((float(kp[1][0]), float(kp[1][1])), (float(kp[2][0]), float(kp[2][1])))})
    finally:
        capture.release()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="r1")
    parser.add_argument("--stride", type=int, default=180)
    parser.add_argument("--batch", default="batch_002")
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--quota", nargs=4, type=int, default=[60, 20, 45, 15],
                        metavar=("TRAIN_CAND", "TRAIN_AMB", "HELDOUT_CAND", "HELDOUT_AMB"))
    args = parser.parse_args()

    split = load_split(OUT / "split.json")
    videos = [v for v in load_video_infos(PROCESSED) if v.video_id in split]
    cache = OUT / f"scan_{args.run}_s{args.stride}.parquet"
    if cache.exists():
        scan = pd.read_parquet(cache)
    else:
        device = "cuda"
        model = load_model(OUT / "models" / args.run, device=device)
        model.transform.min_size = (MIN_SIZE,)
        model.transform.max_size = round(MIN_SIZE * 4 / 3)
        records = []
        for i, video in enumerate(videos, 1):
            manifest = load_manifest(PROCESSED / video.video_id)
            for row in scan_video(model, manifest.video_path, args.stride, device):
                records.append({"video_id": video.video_id, "compound": video.compound, "fps": video.fps, **row})
            print(f"[{i}/{len(videos)}] {video.video_id}", flush=True)
        scan = pd.DataFrame(records)
        scan.to_parquet(cache.with_suffix(".parquet.tmp"), index=False)
        os.replace(cache.with_suffix(".parquet.tmp"), cache)

    scanned = [Scanned(r.video_id, int(r.frame_idx), float(r.fps), r.compound, split[r.video_id], float(r.box_score),
                       float(r.tilt), tuple(r.box)) for r in scan.itertuples()]
    existing: dict[str, list[int]] = {}
    for s in load_samples(OUT):
        existing.setdefault(s["video_id"], []).append(s["frame_idx"])
    tc, ta, hc, ha = args.quota
    picked = select_candidates(scanned, existing, quotas={"train": (tc, ta), "heldout": (hc, ha)}, seed=args.seed)

    (OUT / "frames").mkdir(exist_ok=True)
    infos = {v.video_id: v for v in videos}
    samples = []
    for c in picked:
        manifest = load_manifest(PROCESSED / c.video_id)
        image = OUT / "frames" / f"{c.video_id}_{c.frame_idx:06d}.png"
        if not image.exists() and not cv2.imwrite(str(image), extract_frame(manifest.video_path, c.frame_idx)):
            raise OSError(f"could not write {image}")
        frames = pd.read_parquet(PROCESSED / c.video_id / "frames.parquet").set_index("frame_idx")
        if c.frame_idx not in frames.index:
            raise ValueError(f"{c.video_id} has no row for frame {c.frame_idx}")
        row = frames.loc[c.frame_idx]
        samples.append({"video_id": c.video_id, "frame_idx": c.frame_idx, "t_sec": float(row["t_sec"]), "reason": c.reason,
                        "split": c.split, "compound": infos[c.video_id].compound, "image": image.name,
                        "classical": {"detected": bool(row["detected"]), "x": float(row["x"]), "y": float(row["y"])},
                        "prefill": {"score": c.box_score, "box": list(c.box)}})
    target = OUT / f"{args.batch}.json"
    if target.exists():
        raise FileExistsError(f"{target} exists: batches are never overwritten")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"seed": args.seed, "samples": samples}, indent=1), encoding="utf-8")
    os.replace(temporary, target)
    for name in ("train", "heldout"):
        for reason in ("listing_candidate", "listing_ambiguous"):
            n = sum(s["split"] == name and s["reason"] == reason for s in samples)
            print(name, reason, n)
    print("videos", len({s["video_id"] for s in samples}), "compounds", sorted({s["compound"] for s in samples}))


if __name__ == "__main__":
    main()
