"""Threshold calibration search - FR-008, PRD §9.5.5.

Searches the rule thresholds in `labeling.py` to minimize the mean (over
reference groups) total-variation distance between the pipeline's aggregated
state proportions and the digitized reference targets.

Design decisions (advisor consultation before Phase 6; full narrative in
docs/progress.md):

- **Undetermined has no reference colour.** If it counted toward the
  proportions, the optimizer could shrink it by making frames un-computable.
  So proportions are renormalized over *determined* frames and **coverage**
  (the determined fraction) is reported next to the distance instead of being
  optimized away.
- **Unlabelable states are excluded** (`exclude_states`, e.g. Listing/LORR,
  which is labeled manually): dropped from the pipeline counts and from the
  target, which is renormalized over the remaining states.
- **Speed thresholds only.** Controlled/Erratic is a speed band, so the search
  moves the freeze floor, erratic threshold and bout/depth thresholds; there is
  no model to fit or freeze.
- **Tracking is not searched.** Tracks are computed once per calibration
  video and cached (`track_cache`); searching tracking parameters would mean
  re-tracking every video per sample (~43s each). This is a cost-driven
  deviation from T045's "calibrate tracking and features together" note.
- **The search is seeded** (NFR-002): a fixed seed reproduces the same
  history exactly.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prepds.consolidate import consolidate_labels
from prepds.labeling import PreparedVideo, classify_prepared_states
from prepds.models import BehaviorState

GroupKey = tuple[str, int]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"{self.name}: low ({self.low}) must be < high ({self.high})")
        if self.log and self.low <= 0:
            raise ValueError(f"{self.name}: a log-scale range needs a positive lower bound")

    def to_unit(self, value: float) -> float:
        lo, hi = (math.log(self.low), math.log(self.high)) if self.log else (self.low, self.high)
        v = math.log(value) if self.log else value
        return (v - lo) / (hi - lo)

    def from_unit(self, unit: float) -> float:
        unit = min(1.0, max(0.0, unit))
        lo, hi = (math.log(self.low), math.log(self.high)) if self.log else (self.low, self.high)
        v = lo + unit * (hi - lo)
        return math.exp(v) if self.log else v


@dataclass(frozen=True)
class SearchResult:
    best_params: dict[str, float]
    best_score: float
    history: list[tuple[dict[str, float], float]]


def search(
    objective: Callable[[dict[str, float]], float],
    specs: tuple[ParamSpec, ...],
    *,
    n_random: int,
    n_refine_rounds: int,
    seed: int,
    initial: dict[str, float] | None = None,
) -> SearchResult:
    """Seeded random search followed by coordinate-wise local refinement.

    Refinement tries `best +- step` per parameter (in log space for log-scaled
    ones), accepting strict improvements, with the step halving each round.
    `initial` (if given) is evaluated first, so the result is never worse than it.
    """
    rng = np.random.default_rng(seed)
    history: list[tuple[dict[str, float], float]] = []
    best_params: dict[str, float] = {}
    best_score = math.inf

    def consider(params: dict[str, float]) -> None:
        nonlocal best_params, best_score
        score = objective(params)
        history.append((dict(params), score))
        if score < best_score:
            best_params, best_score = dict(params), score

    if initial is not None:
        consider(dict(initial))
    for _ in range(n_random):
        consider({spec.name: spec.from_unit(float(rng.random())) for spec in specs})

    step = 0.25
    for _ in range(n_refine_rounds):
        for spec in specs:
            for direction in (-1.0, 1.0):
                unit = spec.to_unit(best_params[spec.name]) + direction * step
                candidate = dict(best_params)
                candidate[spec.name] = spec.from_unit(unit)
                if candidate[spec.name] != best_params[spec.name]:
                    consider(candidate)
        step /= 2.0

    return SearchResult(best_params=best_params, best_score=best_score, history=history)


def total_variation(a: Mapping, b: Mapping) -> float:
    return 0.5 * sum(abs(a[key] - b[key]) for key in a)


@dataclass(frozen=True)
class GroupScore:
    tv: float
    coverage: float  # fraction of all frames that are determined (not Undetermined)
    proportions: dict[BehaviorState, float]  # over determined frames only


@dataclass(frozen=True)
class Evaluation:
    mean_tv: float
    groups: dict[GroupKey, GroupScore]


def evaluate_params(
    prepared_by_group: Mapping[GroupKey, list[PreparedVideo]],
    targets: Mapping[GroupKey, Mapping[BehaviorState, float]],
    params: Mapping[str, float],
    *,
    exclude_states: Sequence[BehaviorState] = (),
    consolidate: bool = False,
) -> Evaluation:
    """Classify every prepared video with `params`, pool frames per group, and score against `targets`.

    `exclude_states` removes states that cannot be labelled automatically (e.g. Listing/LORR, which
    is labelled manually): pipeline frames carrying them leave the denominator and the target is
    renormalised over the remaining states, so the distance is measured on the identifiable states only.

    `consolidate=True` scores the 1 s-binned output (`consolidate.consolidate_labels`) that the pipeline
    actually emits, instead of the raw per-frame labels.
    """
    excluded = frozenset(exclude_states)
    scores: dict[GroupKey, GroupScore] = {}
    for key, videos in prepared_by_group.items():
        if key not in targets:
            raise ValueError(f"no reference target for group {key}")
        kept_states = [
            state for state in BehaviorState if state is not BehaviorState.UNDETERMINED and state not in excluded
        ]
        kept_mass = sum(targets[key].get(state, 0.0) for state in kept_states)
        counts: Counter[BehaviorState] = Counter()
        for video in videos:
            states = classify_prepared_states(video, **params)
            counts.update(consolidate_labels(states, video.t_secs) if consolidate else states)

        total = sum(counts.values())
        determined = total - counts[BehaviorState.UNDETERMINED]
        coverage = determined / total if total else 0.0
        scored = sum(counts[state] for state in kept_states)
        if scored == 0 or kept_mass <= 0:
            scores[key] = GroupScore(tv=1.0, coverage=coverage, proportions={s: 0.0 for s in kept_states})
            continue
        proportions = {state: counts[state] / scored for state in kept_states}
        target = {state: targets[key].get(state, 0.0) / kept_mass for state in kept_states}
        scores[key] = GroupScore(tv=total_variation(proportions, target), coverage=coverage, proportions=proportions)

    mean_tv = sum(score.tv for score in scores.values()) / len(scores) if scores else 1.0
    return Evaluation(mean_tv=mean_tv, groups=scores)


def load_reference_targets(path: Path) -> dict[GroupKey, dict[BehaviorState, float]]:
    payload = json.loads(path.read_text())
    return {
        (group["compound"], group["concentration_uM"]): {
            BehaviorState[name]: value for name, value in group["proportions"].items()
        }
        for group in payload["groups"]
    }


# Ranges anchored to measured real data (docs/progress.md Phase 6/7): 1 s-displacement speeds across
# the calibration groups have medians of ~14-31 px/s, so floor and erratic threshold are searched
# across that band rather than picked independently of the tracker's noise floor.
# Only the thresholds the reference data can identify. Surface Breach and Dead have ~0% reference mass in
# every calibratable group, so a search just switches Surface off (threshold -> its floor) and lets the
# Dead bout wander with no effect; those two stay at their placeholder defaults and are reported as
# uncalibrated in the profile.
DEFAULT_SEARCH_SPACE: tuple[ParamSpec, ...] = (
    ParamSpec("freeze_speed_floor_px_per_s", 3.0, 80.0, log=True),
    ParamSpec("min_freeze_bout_s", 0.03, 6.0, log=True),
    ParamSpec("erratic_speed_threshold_px_per_s", 10.0, 200.0, log=True),
)


def constrained_objective(
    prepared_by_group: Mapping[GroupKey, list[PreparedVideo]],
    targets: Mapping[GroupKey, Mapping[BehaviorState, float]],
    *,
    exclude_states: Sequence[BehaviorState] = (),
    fixed: Mapping[str, float] | None = None,
    consolidate: bool = False,
) -> Callable[[Mapping[str, float]], float]:
    """Search objective: mean TV, with `fixed` extra parameters merged in and an infeasible band
    ordering (erratic threshold not above the freeze floor) scored as the worst distance, 1.0."""

    def objective(params: Mapping[str, float]) -> float:
        merged = {**(fixed or {}), **params}
        floor = merged.get("freeze_speed_floor_px_per_s")
        erratic = merged.get("erratic_speed_threshold_px_per_s")
        if floor is not None and erratic is not None and erratic <= floor:
            return 1.0
        return evaluate_params(
            prepared_by_group, targets, merged, exclude_states=exclude_states, consolidate=consolidate
        ).mean_tv

    return objective


VideoInputs = list  # one video's tracks


def prepare_groups(videos_by_group: Mapping[GroupKey, list[VideoInputs]]) -> dict[GroupKey, list[PreparedVideo]]:
    from prepds.labeling import prepare_video

    return {key: [prepare_video(tracks) for tracks in videos] for key, videos in videos_by_group.items()}
