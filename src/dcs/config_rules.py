"""Value rules for the `training:` settings (D-023).

Types are checked while merging an override (`dcs.config`); these rules then
check the merged values: choices from fixed lists, counts and sizes within
range, fractions within bounds, no NaN or infinity. Each rule carries the
requirement text shown in the error, built from the same bounds it checks.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Callable, Mapping, NamedTuple

# Allowed choices (classifier PRD §6.1-6.3, §7.2; gold sources D-033).
SOURCE_PROCESSED = "processed"  # unreviewed prepds output (D-033)
SOURCE_ACCEPTED = "accepted"  # reviewed gold folder
GOLD_SOURCES = (SOURCE_PROCESSED, SOURCE_ACCEPTED)
STAGES = ("compound", "dose", "both")
MODEL_NAMES = ("majority", "date_only", "logreg", "random_forest", "hist_gb", "mlp")

# Smallest sensible values: K-fold needs 2 folds; a class needs 2 fish to be split.
MIN_FOLDS = 2
MIN_CLASS_SIZE_FLOOR = 2


class Rule(NamedTuple):
    check: Callable[[Any], bool]
    requirement: str  # completes "<key> must be ..."


def whole_at_least(minimum: int) -> Rule:
    return Rule(lambda value: _is_whole(value) and value >= minimum, f"a whole number >= {minimum}")


def number_in(low: float, high: float | None = None, *, low_open: bool = False, high_open: bool = False) -> Rule:
    """A finite number above `low` (and below `high`), each bound inclusive unless marked open."""

    def check(value: Any) -> bool:
        if not _is_number(value):
            return False
        above = value > low if low_open else value >= low
        below = high is None or (value < high if high_open else value <= high)
        return above and below

    bounds = [f"{'>' if low_open else '>='} {low}"]
    if high is not None:
        bounds.append(f"{'<' if high_open else '<='} {high}")
    return Rule(check, "a number " + " and ".join(bounds))


def one_of(choices: tuple[str, ...]) -> Rule:
    return Rule(lambda value: value in choices, f"one of {', '.join(choices)}")


def _is_whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """A real int or a finite float; ints are always finite (and may be too large for math.isfinite)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) or math.isfinite(value)


def _is_duration_range(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_is_number(part) for part in value):
        return False
    return 0 <= value[0] < value[1]


def _is_epoch_list(value: Any) -> bool:
    """null, or a non-empty list of strictly increasing dates (YAML date or `YYYY-MM-DD` text)."""
    if value is None:
        return True
    if not isinstance(value, (list, tuple)) or not value:
        return False
    try:
        days = [part if isinstance(part, dt.date) else dt.date.fromisoformat(part) for part in value]
    except (TypeError, ValueError):
        return False
    return all(a < b for a, b in zip(days, days[1:]))


def _is_optional_text(value: Any) -> bool:
    return value is None or NON_EMPTY_TEXT.check(value)


def is_http_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://")) and len(value) > len("https://")


def _is_model_list(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if not all(isinstance(name, str) and name in MODEL_NAMES for name in value):
        return False
    return len(set(value)) == len(value)


def _is_layer_list(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) > 0 and all(_is_whole(size) and size >= 1 for size in value)


NON_EMPTY_TEXT = Rule(lambda value: isinstance(value, str) and bool(value.strip()), "non-empty text")

# One rule per non-boolean default in config/default_training.yaml (a test keeps the two in step).
VALUE_RULES: tuple[tuple[str, Rule], ...] = (
    ("training.gold_source", one_of(GOLD_SOURCES)),
    ("training.model_profile_marker", NON_EMPTY_TEXT),
    ("training.fps_tolerance", number_in(0, 1)),
    ("training.seed", whole_at_least(0)),
    ("training.folds", whole_at_least(MIN_FOLDS)),
    ("training.repeats", whole_at_least(1)),
    ("training.min_class_size", whole_at_least(MIN_CLASS_SIZE_FLOOR)),
    ("training.min_state_fish", whole_at_least(1)),
    ("training.vehicle_compound", NON_EMPTY_TEXT),
    ("training.stage", one_of(STAGES)),
    ("training.models", Rule(_is_model_list, f"a non-empty list without repeats from {', '.join(MODEL_NAMES)}")),
    ("training.duration_range_s", Rule(_is_duration_range, "null or [min, max] seconds with 0 <= min < max")),
    ("training.camera_epochs", Rule(_is_epoch_list, "null or a list of increasing dates (YYYY-MM-DD)")),
    ("training.min_detected_fraction", number_in(0, 1)),
    ("training.mlp.hidden_sizes", Rule(_is_layer_list, "a non-empty list of whole numbers >= 1")),
    ("training.mlp.dropout", number_in(0, 1, high_open=True)),
    ("training.mlp.learning_rate", number_in(0, low_open=True)),
    ("training.mlp.weight_decay", number_in(0)),
    ("training.mlp.batch_size", whole_at_least(1)),
    ("training.mlp.max_epochs", whole_at_least(1)),
    ("training.mlp.patience", whole_at_least(1)),
    ("training.mlp.inner_val_fraction", number_in(0, 1, low_open=True, high_open=True)),
    ("training.mlp.seeds", whole_at_least(1)),
    ("chat.model", Rule(_is_optional_text, "null or the model's name")),
    ("chat.temperature", number_in(0, 2)),
    ("chat.max_tool_rounds", whole_at_least(1)),
    ("chat.timeout_s", number_in(0, low_open=True)),
)


def first_violation(params: Mapping[str, Any]) -> tuple[str, str, Any] | None:
    """The first (dotted key, requirement, value) that breaks a rule, or None when all values are valid."""
    for key, rule in VALUE_RULES:
        value: Any = params
        for part in key.split("."):
            value = value[part]
        if not rule.check(value):
            return key, rule.requirement, value
    return None
