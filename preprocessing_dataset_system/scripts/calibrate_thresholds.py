"""Search the speed-band thresholds against the digitized reference targets (FR-008) and validate leave-one-group-out.

Usage (from preprocessing_dataset_system/, after scripts/build_track_cache.py):
    python scripts/calibrate_thresholds.py [--seeds 1 2 3] [--cache outputs/calibration/tracks_r1]
Prints the best thresholds per seed, per-group total-variation distance, and leave-one-group-out (LOGO) scores.
It does NOT write a profile: freeze the chosen numbers with prepds.calibration.profile.write_calibration_profile
(profiles are immutable; add a new revision, never edit an old one) and point PDS_CONFIG at it.

Cohort selection is deliberate: groups without a matching local cohort are excluded (Fentanyl 100 uM has no local
videos; Fentanyl 30 uM keeps only subjects < 300, the cohort of the reference figure). Surface Breach and Dead have
~0% reference mass, so their thresholds are not searched (placeholders). Listing/LORR is excluded (manual).
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from prepds import labeling as L
from prepds.calibration.calibrate import (
    DEFAULT_SEARCH_SPACE,
    constrained_objective,
    evaluate_params,
    load_reference_targets,
    prepare_groups,
    search,
)
from prepds.calibration.track_cache import load_cached_tracks
from prepds.models import BehaviorState as B

CACHE = Path("outputs/calibration/tracks")
TARGETS = Path("src/prepds/calibration/reference_targets.json")
EXCLUDED_STATES = (B.LISTING_LORR,)
FIXED = {
    "surface_breach_depth_threshold_px": L.DEFAULT_SURFACE_BREACH_DEPTH_THRESHOLD_PX,
    "min_dead_bout_s": L.DEFAULT_MIN_DEAD_BOUT_S,
}
START = {
    "freeze_speed_floor_px_per_s": L.DEFAULT_FREEZE_SPEED_FLOOR_PX_PER_S,
    "min_freeze_bout_s": L.DEFAULT_MIN_FREEZE_BOUT_S,
    "erratic_speed_threshold_px_per_s": L.DEFAULT_ERRATIC_SPEED_THRESHOLD_PX_PER_S,
}


def load_videos(cache: Path) -> dict[tuple[str, int], list]:
    videos: dict[tuple[str, int], list] = {}
    for path in sorted(cache.glob("*.json.gz")):
        name = path.name[: -len(".json.gz")]
        match = re.match(r"(.+)_(\d+)_(\d+)$", name)
        group, subject = (match[1], int(match[2])), int(match[3])
        if group == ("fentanyl", 100) or (group == ("fentanyl", 30) and subject >= 300):
            continue
        videos.setdefault(group, []).append(load_cached_tracks(cache, name))
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--cache", type=Path, default=CACHE, help="track cache to calibrate on (classical by default)")
    args = parser.parse_args()
    videos = load_videos(args.cache)
    targets_all = load_reference_targets(TARGETS)
    targets = {key: targets_all[key] for key in videos}
    prepared = prepare_groups(videos)
    keys = sorted(videos)

    def objective(subset):
        return constrained_objective(
            {k: prepared[k] for k in subset}, {k: targets[k] for k in subset},
            exclude_states=EXCLUDED_STATES, fixed=FIXED, consolidate=True,
        )

    def run(subset, seed, n_random=600):
        return search(objective(subset), DEFAULT_SEARCH_SPACE, n_random=n_random, n_refine_rounds=5, seed=seed, initial=START)

    started = time.time()
    print("placeholder TV", round(objective(keys)(START), 4))
    results = {}
    for seed in args.seeds:
        results[seed] = run(keys, seed)
        best = results[seed]
        print(f"seed {seed}: TV {best.best_score:.4f} ({time.time() - started:.0f}s)", {k: round(v, 2) for k, v in best.best_params.items()})
    best = min(results.values(), key=lambda r: r.best_score)
    evaluation = evaluate_params(prepared, targets, {**FIXED, **best.best_params}, exclude_states=EXCLUDED_STATES, consolidate=True)
    for key, group in sorted(evaluation.groups.items()):
        print(f"{key[0]} {key[1]:>4}: TV {group.tv:.3f} coverage {group.coverage:.2f}")
    logo = {}
    for held_out in keys:
        rest = [k for k in keys if k != held_out]
        fit = run(rest, seed=1, n_random=300)
        logo[held_out] = evaluate_params(
            {held_out: prepared[held_out]}, {held_out: targets[held_out]}, {**FIXED, **fit.best_params},
            exclude_states=EXCLUDED_STATES, consolidate=True,
        ).mean_tv
        print("LOGO", held_out, round(logo[held_out], 3), flush=True)
    print("LOGO mean", round(sum(logo.values()) / len(logo), 3))


if __name__ == "__main__":
    main()
