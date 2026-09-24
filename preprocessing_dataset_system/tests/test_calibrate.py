"""Calibration search (T044) - FR-008, PRD §9.5.5.

Design decisions locked in here (advisor consultation, pre-Phase-6):

- The objective is total-variation distance between the pipeline's aggregated
  state proportions per reference group and the digitized targets.
- Undetermined has no reference colour. If it counted toward the proportions,
  the optimizer could shrink it by making frames un-computable; so
  proportions are renormalized over DETERMINED frames only, and coverage
  (determined fraction) is reported alongside the distance instead of being
  optimized away.
- States that cannot be labeled automatically (Listing/LORR) are excluded
  from both pipeline and target and the target is renormalized.
- The search is seeded (NFR-002): same seed, same result.
"""

from __future__ import annotations

import pytest

from prepds.calibration.calibrate import (
    ParamSpec,
    evaluate_params,
    search,
    total_variation,
)
from prepds.labeling import prepare_video
from prepds.models import BehaviorState, Track

FPS = 30.0
SLOW_BLOCK, FAST_BLOCK = 90, 90  # 3s each


def _toy_video():
    """Alternating 3s blocks: slow (5 px/s) and fast (200 px/s), one detected run."""
    tracks = []
    frame, x = 0, 0.0
    for block in range(10):
        speed = 5.0 if block % 2 == 0 else 200.0
        for _ in range(SLOW_BLOCK if block % 2 == 0 else FAST_BLOCK):
            tracks.append(Track(frame, frame / FPS, x, 100.0, 90.0, 100.0, True))
            frame += 1
            x += speed / FPS
    return tracks


def _prepared_toy(*, undetermined_tail: bool = False):
    tracks = _toy_video()
    if undetermined_tail:
        # a 40-frame undetected gap (longer than the 1 s lag), then a 20-frame run that is detected
        # but has no detected partner 1 s away: no computable displacement
        gap = len(tracks)
        tracks += [Track(gap + i, (gap + i) / FPS, 0.0, 0.0, None, None, False) for i in range(40)]
        tracks += [Track(gap + 40 + i, (gap + 40 + i) / FPS, 0.0, 100.0, 90.0, 100.0, True) for i in range(20)]
    return prepare_video(tracks)


TARGET = {state: 0.0 for state in BehaviorState if state != BehaviorState.UNDETERMINED}
TARGET[BehaviorState.FREEZING_DRIFT] = 0.5
TARGET[BehaviorState.ERRATIC_MOVEMENT] = 0.5


def test_total_variation() -> None:
    a = {"x": 0.5, "y": 0.5}
    assert total_variation(a, a) == 0.0
    assert total_variation({"x": 1.0, "y": 0.0}, {"x": 0.0, "y": 1.0}) == 1.0
    assert total_variation({"x": 0.75, "y": 0.25}, {"x": 0.5, "y": 0.5}) == pytest.approx(0.25)


def test_evaluate_params_reports_distance_and_coverage() -> None:
    groups = {("toy", 0): [_prepared_toy()]}
    low_floor = evaluate_params(groups, {("toy", 0): TARGET}, {"freeze_speed_floor_px_per_s": 1.0})
    good_floor = evaluate_params(groups, {("toy", 0): TARGET}, {"freeze_speed_floor_px_per_s": 60.0, "erratic_speed_threshold_px_per_s": 100.0})

    assert good_floor.mean_tv < low_floor.mean_tv
    assert good_floor.mean_tv == pytest.approx(0.0, abs=0.12)  # residual: the 1 s displacement lag smears block edges
    assert 0.0 < good_floor.groups[("toy", 0)].coverage <= 1.0


def test_undetermined_is_excluded_from_proportions_and_reported_as_coverage() -> None:
    # A stretch with no computable speed is Undetermined: it must drop out of the proportions
    # (it has no reference colour) and show up as reduced coverage, not count as a state.
    groups = {("toy", 0): [_prepared_toy(undetermined_tail=True)]}
    result = evaluate_params(groups, {("toy", 0): TARGET}, {"freeze_speed_floor_px_per_s": 60.0, "erratic_speed_threshold_px_per_s": 100.0})

    group = result.groups[("toy", 0)]
    assert BehaviorState.UNDETERMINED not in group.proportions
    assert sum(group.proportions.values()) == pytest.approx(1.0)
    assert group.coverage == pytest.approx(900 / 960, abs=1e-3)  # the 60 tail frames are not determined


def test_a_group_with_no_determined_frames_scores_the_worst_distance() -> None:
    short = prepare_video([Track(i, i / FPS, 0.0, 100.0, 90.0, 100.0, True) for i in range(10)])
    result = evaluate_params({("toy", 0): [short]}, {("toy", 0): TARGET}, {})
    assert result.groups[("toy", 0)].tv == 1.0


# --- T044: the search reduces distance to the target -----------------------------


def test_search_reduces_distance_to_target() -> None:
    groups = {("toy", 0): [_prepared_toy()]}
    targets = {("toy", 0): TARGET}
    specs = (ParamSpec("freeze_speed_floor_px_per_s", 0.5, 400.0, log=True),)

    from prepds.calibration.calibrate import constrained_objective

    objective = constrained_objective(groups, targets, fixed={"erratic_speed_threshold_px_per_s": 100.0})
    initial = {"freeze_speed_floor_px_per_s": 1.0}
    result = search(objective, specs, n_random=25, n_refine_rounds=3, seed=0, initial=initial)

    assert result.best_score < objective(initial)
    assert result.best_score == pytest.approx(0.0, abs=0.12)
    # The known-optimal floor lies between the slow (5) and fast (200) speeds.
    assert 5.0 <= result.best_params["freeze_speed_floor_px_per_s"] < 100.0


def test_search_is_deterministic_for_a_fixed_seed() -> None:
    specs = (ParamSpec("x", 0.0, 10.0), ParamSpec("y", 1.0, 100.0, log=True))

    def objective(p: dict) -> float:
        return (p["x"] - 3.7) ** 2 + (p["y"] - 20.0) ** 2 / 100.0

    a = search(objective, specs, n_random=20, n_refine_rounds=2, seed=7)
    b = search(objective, specs, n_random=20, n_refine_rounds=2, seed=7)
    c = search(objective, specs, n_random=20, n_refine_rounds=2, seed=8)

    assert a.best_params == b.best_params and a.history == b.history
    assert a.history != c.history


def test_search_finds_a_known_optimum_of_a_smooth_objective() -> None:
    specs = (ParamSpec("x", 0.0, 10.0), ParamSpec("y", 1.0, 100.0, log=True))

    def objective(p: dict) -> float:
        return (p["x"] - 3.7) ** 2 + (p["y"] - 20.0) ** 2 / 100.0

    result = search(objective, specs, n_random=60, n_refine_rounds=6, seed=0)

    assert result.best_params["x"] == pytest.approx(3.7, abs=0.3)
    assert result.best_params["y"] == pytest.approx(20.0, abs=3.0)


def test_search_never_returns_worse_than_the_initial_point() -> None:
    specs = (ParamSpec("x", 0.0, 10.0),)
    result = search(lambda p: abs(p["x"] - 5.0), specs, n_random=3, n_refine_rounds=1, seed=1, initial={"x": 5.0})
    assert result.best_score == 0.0


def test_paramspec_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        ParamSpec("x", 5.0, 1.0)
    with pytest.raises(ValueError):
        ParamSpec("x", 0.0, 10.0, log=True)  # log scale needs a positive lower bound


# --- glue: prepared groups ---------------------------------------------------------


def test_prepare_groups_prepares_every_video_of_every_group() -> None:
    from prepds.calibration.calibrate import prepare_groups

    videos = {("toy", 0): [_toy_video()], ("other", 30): [_toy_video(), _toy_video()]}
    prepared = prepare_groups(videos)

    assert {k: len(v) for k, v in prepared.items()} == {("toy", 0): 1, ("other", 30): 2}
    assert prepared[("toy", 0)][0].speeds == prepared[("other", 30)][1].speeds


def test_excluded_states_are_dropped_from_target_and_pipeline_and_renormalized() -> None:
    groups = {("toy", 0): [_prepared_toy()]}
    target = {state: 0.0 for state in BehaviorState if state != BehaviorState.UNDETERMINED}
    target[BehaviorState.FREEZING_DRIFT] = 0.25
    target[BehaviorState.ERRATIC_MOVEMENT] = 0.25
    target[BehaviorState.LISTING_LORR] = 0.5  # reference says half the time is Listing
    params = {"freeze_speed_floor_px_per_s": 60.0, "erratic_speed_threshold_px_per_s": 100.0}

    with_listing = evaluate_params(groups, {("toy", 0): target}, params)
    without = evaluate_params(
        groups, {("toy", 0): target}, params, exclude_states=(BehaviorState.LISTING_LORR,)
    )

    assert with_listing.mean_tv > 0.4  # pipeline never emits Listing here
    assert without.mean_tv < 0.15  # renormalized target is 50/50 Freezing/Erratic
    assert BehaviorState.LISTING_LORR not in without.groups[("toy", 0)].proportions


def test_excluding_all_reference_mass_scores_worst_distance() -> None:
    groups = {("toy", 0): [_prepared_toy()]}
    target = {state: 0.0 for state in BehaviorState if state != BehaviorState.UNDETERMINED}
    target[BehaviorState.LISTING_LORR] = 1.0
    result = evaluate_params(
        groups, {("toy", 0): target}, {}, exclude_states=(BehaviorState.LISTING_LORR,)
    )
    assert result.groups[("toy", 0)].tv == 1.0


def test_states_the_pipeline_emits_but_the_target_omits_are_penalised_not_dropped() -> None:
    groups = {("toy", 0): [_prepared_toy()]}
    # a target with no Freezing entry at all: the pipeline's Freezing frames must count against it
    target = {BehaviorState.ERRATIC_MOVEMENT: 1.0}
    result = evaluate_params(groups, {("toy", 0): target}, {"freeze_speed_floor_px_per_s": 50.0, "erratic_speed_threshold_px_per_s": 100.0})
    assert result.groups[("toy", 0)].tv > 0.4


def test_default_search_space_covers_only_identifiable_thresholds() -> None:
    from prepds.calibration.calibrate import DEFAULT_SEARCH_SPACE

    # Surface Breach and Dead have ~0% reference mass in every calibratable group, so the data cannot
    # identify their thresholds (a search just switches them off / wanders): they stay placeholders.
    assert {s.name for s in DEFAULT_SEARCH_SPACE} == {
        "freeze_speed_floor_px_per_s",
        "min_freeze_bout_s",
        "erratic_speed_threshold_px_per_s",
    }


def test_constrained_objective_scores_an_infeasible_band_ordering_as_worst_and_merges_fixed_params() -> None:
    from prepds.calibration.calibrate import constrained_objective

    groups = {("toy", 0): [_prepared_toy()]}
    objective = constrained_objective(groups, {("toy", 0): TARGET}, fixed={"min_freeze_bout_s": 1.0})

    assert objective({"freeze_speed_floor_px_per_s": 50.0, "erratic_speed_threshold_px_per_s": 20.0}) == 1.0
    feasible = objective({"freeze_speed_floor_px_per_s": 50.0, "erratic_speed_threshold_px_per_s": 100.0})
    assert feasible == pytest.approx(0.0, abs=0.15)


def test_consolidated_evaluation_scores_the_binned_output_the_pipeline_actually_emits() -> None:
    # A fast fish detected on every other frame: per-frame half of all frames are Undetermined,
    # but the consolidated output labels every 1 s bin Erratic.
    tracks = [
        Track(i, i / FPS, i * 100.0 / FPS if i % 2 == 0 else 0.0, 100.0 if i % 2 == 0 else 0.0,
              90.0 if i % 2 == 0 else None, 100.0 if i % 2 == 0 else None, i % 2 == 0)
        for i in range(600)
    ]
    groups = {("toy", 0): [prepare_video(tracks)]}
    target = {state: 0.0 for state in BehaviorState if state != BehaviorState.UNDETERMINED}
    target[BehaviorState.ERRATIC_MOVEMENT] = 1.0
    params = {"freeze_speed_floor_px_per_s": 20.0, "erratic_speed_threshold_px_per_s": 50.0}

    per_frame = evaluate_params(groups, {("toy", 0): target}, params)
    binned = evaluate_params(groups, {("toy", 0): target}, params, consolidate=True)

    assert per_frame.groups[("toy", 0)].coverage == pytest.approx(0.5, abs=0.02)
    assert binned.groups[("toy", 0)].coverage == pytest.approx(1.0, abs=0.01)
