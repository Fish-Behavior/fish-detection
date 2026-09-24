"""Phase 15: scan processed videos for possible Listing/LORR and write `listing_flags.json` for the review app.

Usage (from preprocessing_dataset_system/):
    python scripts/phase15_flag_listing.py --detector m3 --classifier c1 [--step-s 3] [--threshold 0.9] [--limit N] [--force]
Every `--step-s` seconds the detector finds the fish, the crop classifier scores its box, and consecutive scores
>= threshold become one flagged range. Hints only: they never change states or acceptance. Videos that already
have a sidecar are skipped unless --force.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from prepds.annotation.crop_classifier import load_classifier, score_images
from prepds.annotation.train import load_model
from prepds.listing_flags import SIDECAR, merge_flags, write_listing_flags
from prepds.review_store import load_manifest

WORK = Path("outputs/phase15")
PROCESSED = Path("outputs/processed")
MIN_SIZE = 400
MIN_BOX_SCORE = 0.3


@torch.no_grad()
def scan_video(detector, classifier, size: int, video_path: Path, step_s: float, device: str) -> tuple[list[tuple[float, float]], float]:
    """(t_sec, score) per sampled frame where the fish was found, and the real sampling step in seconds.
    Frames are read, detected and scored 16 at a time so memory does not grow with video length."""
    capture = cv2.VideoCapture(str(video_path))
    samples: list[tuple[float, float]] = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not (fps > 0 and math.isfinite(fps) and n > 0):
            raise ValueError(f"unusable video metadata (fps={fps}, frames={n})")
        step = max(1, round(step_s * fps))
        indices = list(range(step // 2, n, step))
        for start in range(0, len(indices), 16):
            frames, times = [], []
            for index in indices[start:start + 16]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"could not read frame {index} of {video_path}")
                frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
                times.append(index / fps)
            outputs = detector([torch.from_numpy(f).permute(2, 0, 1).to(device).float() / 255.0 for f in frames])
            items, item_times = [], []
            for frame, out, t in zip(frames, outputs, times):
                if len(out["scores"]) and float(out["scores"][0]) >= MIN_BOX_SCORE:
                    items.append((Image.fromarray(frame), out["boxes"][0].cpu().tolist()))
                    item_times.append(t)
            samples.extend(zip(item_times, score_images(classifier, items, size=size, device=device)))
        if not indices:
            raise ValueError(f"no frames to sample in {video_path}")
        return samples, step / fps
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detector", default="m3")
    parser.add_argument("--classifier", default="c1")
    parser.add_argument("--step-s", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--min-samples", type=int, default=2, help="consecutive above-threshold samples needed for a flag")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = load_model(WORK / "models" / args.detector, device=device)
    detector.transform.min_size = (MIN_SIZE,)
    detector.transform.max_size = round(MIN_SIZE * 4 / 3)
    classifier, size = load_classifier(WORK / "listing_models" / args.classifier, device=device)

    directories = sorted(p for p in args.processed.iterdir() if (p / "manifest.json").is_file())
    done = 0
    failed: list[str] = []
    for i, directory in enumerate(directories, 1):
        if (directory / SIDECAR).exists() and not args.force:
            continue
        if args.limit is not None and done >= args.limit:
            break
        manifest = load_manifest(directory)
        try:
            samples, step_s = scan_video(detector, classifier, size, manifest.video_path, args.step_s, device)
        except (ValueError, OSError) as error:  # one bad video must not stop the scan, and must not look scanned
            failed.append(directory.name)
            print(f"[{i}/{len(directories)}] {directory.name}: FAILED, no sidecar written: {error}", file=sys.stderr, flush=True)
            continue
        flags = merge_flags(samples, threshold=args.threshold, step_s=step_s, min_samples=args.min_samples)
        write_listing_flags(directory, flags, meta={"detector": args.detector, "classifier": args.classifier,
                                                    "threshold": args.threshold, "min_samples": args.min_samples, "step_s": step_s, "n_scored": len(samples)})
        done += 1
        print(f"[{i}/{len(directories)}] {directory.name}: {len(samples)} scored, {len(flags)} flag(s)", flush=True)
    if failed:
        print(f"{len(failed)} video(s) failed and have no sidecar: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
