"""Phase 15: freeze the subject-level split, sample frames to annotate, and extract them.

Usage (from preprocessing_dataset_system/):
    python scripts/phase15_prepare.py [--train 250] [--heldout 100] [--seed 15]
Writes under outputs/phase15/ (restricted data, gitignored): split.json (written once, never overwritten),
sample.json, frames/<video_id>_<frame>.png. The classical tracker's centroid is stored with each sample for
pre-filling the labeling page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prepds.annotation.inputs import extract_frame, load_video_infos
from prepds.annotation.sampler import sample_frames
from prepds.annotation.split import load_split, make_split, write_split
from prepds.review_store import load_manifest

import cv2

OUT = Path("outputs/phase15")
PROCESSED = Path("outputs/processed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=int, default=250)
    parser.add_argument("--heldout", type=int, default=100)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)
    videos = load_video_infos(PROCESSED)
    split_path = OUT / "split.json"
    if split_path.exists():
        split = load_split(split_path)
        print(f"using the frozen split {split_path}")
    else:
        split = make_split({v.video_id: v.compound for v in videos}, heldout_fraction=args.heldout_fraction, seed=args.seed)
        write_split(split_path, split, seed=args.seed, heldout_fraction=args.heldout_fraction)
        print(f"froze a new split {split_path}")
    videos = [v for v in videos if v.video_id in split]

    samples = []
    for name, count in (("train", args.train), ("heldout", args.heldout)):
        for ref in sample_frames(videos, split, n_frames=count, seed=args.seed, split_name=name):
            directory = PROCESSED / ref.video_id
            manifest = load_manifest(directory)
            row = pd.read_parquet(directory / "frames.parquet").iloc[ref.frame_idx]
            image = OUT / "frames" / f"{ref.video_id}_{ref.frame_idx:06d}.png"
            if not image.exists():
                if not cv2.imwrite(str(image), extract_frame(manifest.video_path, ref.frame_idx)):
                    raise OSError(f"could not write {image}")
            samples.append(
                {
                    "video_id": ref.video_id, "frame_idx": ref.frame_idx, "t_sec": float(row["t_sec"]), "reason": ref.reason,
                    "split": name, "compound": manifest.compound, "image": image.name,
                    "classical": {"detected": bool(row["detected"]), "x": float(row["x"]), "y": float(row["y"])},
                }
            )
    with open(OUT / "sample.json", "x", encoding="utf-8") as handle:
        json.dump({"seed": args.seed, "samples": samples}, handle, indent=1)
    for name in ("train", "heldout"):
        subset = [s for s in samples if s["split"] == name]
        reasons = {r: sum(s["reason"] == r for s in subset) for r in sorted({s["reason"] for s in subset})}
        print(name, len(subset), "frames from", len({s["video_id"] for s in subset}), "videos", reasons)


if __name__ == "__main__":
    main()
