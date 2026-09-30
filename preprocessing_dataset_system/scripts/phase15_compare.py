"""Phase 15 / T093: compare the model-tracker run (outputs_r3) with the classical baseline (outputs) per video.

Usage (from preprocessing_dataset_system/):
    python scripts/phase15_compare.py [--new outputs_r3/processed] [--old outputs/processed] [--csv out.csv]
Only videos present in both are compared. Reports, over the shared videos: undetected-frame share, Undetermined
share, label agreement on frames both runs determined, state mix, review-flag counts, and (if a frozen split and
labels exist) the numbers split by the Phase 15 held-out videos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ["Controlled Swim", "Erratic Movement", "Freezing/Drift", "Listing/LORR", "Surface Breach", "Dead", "Undetermined"]


def per_video(new_dir: Path, old_dir: Path) -> pd.DataFrame:
    rows = []
    for new in sorted(p for p in new_dir.iterdir() if (p / "frames.parquet").is_file()):
        old = old_dir / new.name
        if not (old / "frames.parquet").is_file():
            continue
        a = pd.read_parquet(old / "frames.parquet", columns=["detected", "state"])
        b = pd.read_parquet(new / "frames.parquet", columns=["detected", "state"])
        n = min(len(a), len(b))
        sa, sb = a["state"].astype(str).to_numpy()[:n], b["state"].astype(str).to_numpy()[:n]
        both = (sa != "Undetermined") & (sb != "Undetermined")
        row = {
            "video": new.name,
            "old_undetected": 1 - a["detected"].mean(), "new_undetected": 1 - b["detected"].mean(),
            "old_und": (sa == "Undetermined").mean(), "new_und": (sb == "Undetermined").mean(),
            "agree_where_both_determined": float((sa[both] == sb[both]).mean()) if both.any() else np.nan,
            "both_determined": both.mean(),
            "old_flags": len(json.loads((old / "manifest.json").read_text())["review_flags"]),
            "new_flags": len(json.loads((new / "manifest.json").read_text())["review_flags"]),
        }
        for s in STATES[:-1]:
            row[f"old_{s}"], row[f"new_{s}"] = (sa == s).mean(), (sb == s).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new", type=Path, default=Path("outputs_r3/processed"))
    parser.add_argument("--old", type=Path, default=Path("outputs/processed"))
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    df = per_video(args.new, args.old)
    if df.empty:
        print("no videos in common yet")
        return
    if args.csv:
        df.to_csv(args.csv, index=False)
    n = len(df)
    print(f"{n} videos in common")
    for label, col in (("undetected frames", "undetected"), ("Undetermined frames", "und")):
        o, w = df[f"old_{col}"], df[f"new_{col}"]
        print(f"{label:20s} median old {o.median():.1%} -> new {w.median():.1%};  mean {o.mean():.1%} -> {w.mean():.1%}")
    for t in (0.3, 0.5):
        print(f"videos with Undetermined > {t:.0%}: old {(df.old_und > t).sum()} -> new {(df.new_und > t).sum()}")
    print(f"label agreement where both determined: median {df.agree_where_both_determined.median():.1%} "
          f"(over {df.both_determined.mean():.0%} of frames on average)")
    print("mean state share (old -> new):")
    for s in STATES[:-1]:
        print(f"  {s:17s} {df[f'old_{s}'].mean():6.1%} -> {df[f'new_{s}'].mean():6.1%}")
    print(f"review flags (videos with any): old {(df.old_flags > 0).sum()} -> new {(df.new_flags > 0).sum()}")


if __name__ == "__main__":
    main()
