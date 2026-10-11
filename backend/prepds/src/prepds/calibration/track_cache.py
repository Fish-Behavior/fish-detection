"""Calibration video selection and the on-disk Track cache - FR-008 support.

Tracking (~43s/video) is by far the most expensive pipeline step and does not
depend on any parameter the calibration search varies (tracking parameters
are deliberately left out of the search - a cost-driven deviation from
T045's "calibrate together" note, documented in docs/progress.md Phase 6),
so each calibration video is tracked once and its `Track` list cached; the
search only re-runs features/labeling on the cached tracks.
"""

from __future__ import annotations

import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from prepds.models import MatchStatus, Track, Trial

REFERENCE_COMPOUNDS = {"veh", "mdma", "methylone", "dob", "fentanyl"}
_MM_TO_UM = {"0.03": 30, "0.1": 100}
_VEH_CONCENTRATIONS = {"1% dmso"}


def group_key(trial: Trial) -> tuple[str, int] | None:
    """`(compound, concentration_uM)` reference group for a trial, or `None` if it has no reference panel."""
    compound = trial.compound.strip().lower().replace(" ", "")
    if compound not in REFERENCE_COMPOUNDS:
        return None
    concentration = trial.concentration_mM.strip().lower()
    if compound == "veh":
        return ("veh", 0) if concentration in _VEH_CONCENTRATIONS else None
    micromolar = _MM_TO_UM.get(concentration)
    return (compound, micromolar) if micromolar is not None else None


def select_calibration_trials(trials: list[Trial], *, per_group: int) -> dict[tuple[str, int], list[Trial]]:
    """Up to `per_group` matched-video trials per reference group, picked evenly across the sorted
    subject range (not just the first N subjects) and independent of input order."""
    if per_group < 1:
        raise ValueError("per_group must be >= 1")
    by_group: dict[tuple[str, int], list[Trial]] = {}
    for trial in trials:
        key = group_key(trial)
        if key is None or trial.video_path is None or trial.match_status != MatchStatus.MATCHED:
            continue
        by_group.setdefault(key, []).append(trial)

    selected: dict[tuple[str, int], list[Trial]] = {}
    for key, group in by_group.items():
        ordered = sorted(group, key=lambda t: (t.subject_id, str(t.video_path)))
        if len(ordered) <= per_group:
            selected[key] = ordered
        else:
            step = (len(ordered) - 1) / (per_group - 1) if per_group > 1 else 0
            selected[key] = [ordered[round(i * step)] for i in range(per_group)]
    return selected


def cache_name(group: tuple[str, int], trial: Trial) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", f"{group[0]}_{group[1]}_{trial.subject_id}".lower())


def _cache_file(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.json.gz"


def save_cached_tracks(cache_dir: Path, name: str, tracks: list[Track]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([t.to_dict() for t in tracks])
    with gzip.open(_cache_file(cache_dir, name), "wt") as handle:
        handle.write(payload)


def load_cached_tracks(cache_dir: Path, name: str) -> list[Track] | None:
    path = _cache_file(cache_dir, name)
    if not path.is_file():
        return None
    with gzip.open(path, "rt") as handle:
        return [Track.from_dict(d) for d in json.loads(handle.read())]


def _track_and_cache(args: tuple[str, str, str]) -> str:
    from prepds.tracking import track_video

    cache_dir, name, video_path = args
    if load_cached_tracks(Path(cache_dir), name) is None:
        save_cached_tracks(Path(cache_dir), name, track_video(Path(video_path)))
    return name


def build_track_cache(
    selected: dict[tuple[str, int], list[Trial]], cache_dir: Path, *, workers: int = 8
) -> list[str]:
    """Track every selected video that isn't cached yet, in parallel. Returns all cache names."""
    jobs = [
        (str(cache_dir), cache_name(group, trial), str(trial.video_path))
        for group, group_trials in selected.items()
        for trial in group_trials
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_track_and_cache, jobs))
