"""Re-track the calibration videos with the fine-tuned model, keeping the exact same videos (and cache names) as
the classical cache, so profile r3 can be fitted and compared like-for-like with r2.

Usage (from preprocessing_dataset_system/, after `prepds catalog`):
    python scripts/build_model_track_cache.py --run outputs/phase15/models/r1 [--stride 5]
Reads names from outputs/calibration/tracks/ and writes outputs/calibration/tracks_<run>/ (resumable).
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from prepds.calibration.track_cache import group_key, load_cached_tracks, save_cached_tracks
from prepds.models import Trial
from prepds.model_tracker import ModelTracker

CLASSICAL = Path("outputs/calibration/tracks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    out = Path("outputs/calibration") / f"tracks_{args.run.name}"
    frame = pd.read_parquet("outputs/trials_catalog.parquet")
    frame = frame.astype(object).where(frame.notna(), None)
    trials = [Trial.from_dict(r) for r in frame.to_dict("records")]
    by_key = {(group_key(t), int(t.subject_id)): t for t in trials if t.video_path is not None and group_key(t) is not None}
    tracker = ModelTracker(args.run, stride=args.stride)
    names = sorted(p.name[: -len(".json.gz")] for p in CLASSICAL.glob("*.json.gz"))
    started = time.time()
    for i, name in enumerate(names, 1):
        match = re.match(r"(.+)_(\d+)_(\d+)$", name)
        trial = by_key.get(((match[1], int(match[2])), int(match[3])))
        if trial is None:
            print(f"[{i}/{len(names)}] {name}: no catalog trial, skipped")
            continue
        if load_cached_tracks(out, name) is not None:
            continue
        save_cached_tracks(out, name, tracker(trial.video_path))
        print(f"[{i}/{len(names)}] {name} ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
