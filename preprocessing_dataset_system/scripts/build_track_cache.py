"""Track the calibration videos once and cache the tracks (~40 s per video), so threshold searches are fast.

Usage (from preprocessing_dataset_system/, with PDS_VIDEO_DIR and PDS_DB_PATH set):
    python scripts/build_track_cache.py [--per-group 6] [--workers 10]
Writes outputs/calibration/tracks/*.json.gz.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from prepds.calibration.track_cache import build_track_cache, select_calibration_trials
from prepds.catalog import dedupe_rows, load_workbook, match_videos, to_trials
from prepds.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    settings = load_settings()
    trials, _ = dedupe_rows(load_workbook(settings.require("db_path")))
    matched, _, _ = match_videos(to_trials(trials), settings.require("video_dir"))
    selected = select_calibration_trials(matched, per_group=args.per_group)
    for group, group_trials in sorted(selected.items()):
        print(group, [t.subject_id for t in group_trials])
    started = time.time()
    names = build_track_cache(selected, Path("outputs/calibration/tracks"), workers=args.workers)
    print(f"cached {len(names)} videos in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
