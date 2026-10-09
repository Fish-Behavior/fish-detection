"""Training parameters of `dcs`: packaged defaults, YAML overrides and their validation (plan U1, T1.2).

Covers D-023: unknown keys, wrong types and out-of-range values are all a
ConfigError naming the dotted key; settings are read-only and picklable.
"""

from __future__ import annotations

import copy
import math
import pickle
from pathlib import Path
from typing import Any

import pytest
import yaml

from dcs.config import DEFAULT_CONFIG_FILE, MODEL_NAMES, STAGES, ConfigError, Settings, deep_merge, load_settings
from dcs.config_rules import VALUE_RULES


def _write_yaml(path: Path, data: Any) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _load_with(tmp_path: Path, training: dict[str, Any]) -> Settings:
    return load_settings(environ={}, config_file=_write_yaml(tmp_path / "o.yaml", {"training": training}))


# --- defaults and merging ------------------------------------------------------------


def test_packaged_defaults_are_loaded() -> None:
    assert DEFAULT_CONFIG_FILE.is_file()
    training = load_settings(environ={}).training
    assert training["seed"] == 0
    assert training["folds"] == 5
    assert training["min_class_size"] == 6
    assert training["min_state_fish"] == 10
    assert training["use_depth"] is False
    assert training["stage"] in STAGES
    assert training["mlp"]["hidden_sizes"] == (128, 64)
    assert set(training["models"]) <= set(MODEL_NAMES)
    assert {"majority", "date_only"} <= set(training["models"])


def test_override_changes_only_listed_keys(tmp_path: Path) -> None:
    training = _load_with(tmp_path, {"seed": 7, "mlp": {"dropout": 0.5}}).training
    assert training["seed"] == 7
    assert training["mlp"]["dropout"] == 0.5
    assert training["mlp"]["hidden_sizes"] == (128, 64)  # sibling key kept
    assert training["folds"] == 5


def test_dcs_config_variable_selects_the_override(tmp_path: Path) -> None:
    override = _write_yaml(tmp_path / "o.yaml", {"training": {"repeats": 2}})
    assert load_settings(environ={"DCS_CONFIG": str(override)}).training["repeats"] == 2


def test_explicit_config_file_wins_over_dcs_config(tmp_path: Path) -> None:
    from_env = _write_yaml(tmp_path / "env.yaml", {"training": {"repeats": 2}})
    explicit = _write_yaml(tmp_path / "explicit.yaml", {"training": {"repeats": 3}})
    settings = load_settings(environ={"DCS_CONFIG": str(from_env)}, config_file=explicit)
    assert settings.training["repeats"] == 3


def test_null_default_accepts_a_range_and_an_explicit_null(tmp_path: Path) -> None:
    assert _load_with(tmp_path, {"duration_range_s": [100, 200]}).training["duration_range_s"] == (100, 200)
    assert _load_with(tmp_path, {"duration_range_s": None}).training["duration_range_s"] is None


def test_camera_epochs_accept_yaml_dates_and_text(tmp_path: Path) -> None:
    """A YAML file reads an unquoted 2000-03-01 as a date; quoted, it is text. Both work (D-061)."""
    path = tmp_path / "epochs.yaml"
    path.write_text("training:\n  camera_epochs: [2000-03-01, '2000-06-01']\n", encoding="utf-8")
    assert len(load_settings(config_file=path).training["camera_epochs"]) == 2


def test_int_accepted_for_float_setting(tmp_path: Path) -> None:
    assert _load_with(tmp_path, {"min_detected_fraction": 1}).training["min_detected_fraction"] == 1


def test_empty_override_file_changes_nothing(tmp_path: Path) -> None:
    override = tmp_path / "empty.yaml"
    override.write_text("")
    assert load_settings(environ={}, config_file=override).training == load_settings(environ={}).training


def test_empty_section_changes_nothing(tmp_path: Path) -> None:
    override = tmp_path / "empty_section.yaml"
    override.write_text("training:\n")
    assert load_settings(environ={}, config_file=override).training == load_settings(environ={}).training


def test_empty_nested_section_changes_nothing(tmp_path: Path) -> None:
    override = tmp_path / "empty_mlp.yaml"
    override.write_text("training:\n  mlp:\n")
    assert load_settings(environ={}, config_file=override).training == load_settings(environ={}).training


def test_deep_merge_does_not_mutate_its_inputs() -> None:
    base = {"a": {"b": 1, "c": [1]}}
    override = {"a": {"b": 2}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 2, "c": [1]}}
    assert base == {"a": {"b": 1, "c": [1]}}
    merged["a"]["c"].append(2)
    assert base["a"]["c"] == [1]


# --- unknown keys and wrong types ---------------------------------------------------


@pytest.mark.parametrize(
    ("override", "name"),
    [
        ({"training": {"sead": 1}}, "training.sead"),
        ({"training": {"mlp": {"hidden": [8]}}}, "training.mlp.hidden"),
        ({"trainng": {"seed": 1}}, "trainng"),
    ],
)
def test_unknown_key_is_rejected_with_its_name(tmp_path: Path, override: dict[str, Any], name: str) -> None:
    path = _write_yaml(tmp_path / "o.yaml", override)
    with pytest.raises(ConfigError, match=name) as caught:
        load_settings(environ={}, config_file=path)
    assert "valid keys" in str(caught.value).lower()


@pytest.mark.parametrize(
    ("training", "name", "requirement"),
    [
        ({"folds": "five"}, "training.folds", "whole number"),
        ({"folds": 5.5}, "training.folds", "whole number"),
        ({"folds": True}, "training.folds", "whole number"),
        ({"use_ntt": "yes"}, "training.use_ntt", "true or false"),
        ({"use_ntt": 1}, "training.use_ntt", "true or false"),
        ({"models": "logreg"}, "training.models", "list"),
        ({"mlp": 3}, "training.mlp", "section"),
        ({"seed": None}, "training.seed", "whole number"),
        ({"min_detected_fraction": "high"}, "training.min_detected_fraction", "a number"),
        ({"min_detected_fraction": True}, "training.min_detected_fraction", "a number"),
        ({"stage": 5}, "training.stage", "text"),
        ({"mlp": {"hidden_sizes": 8}}, "training.mlp.hidden_sizes", "list"),
        ({"gold_source": 3}, "training.gold_source", "text"),
        ({"model_profile_marker": None}, "training.model_profile_marker", "text"),
        ({"fps_tolerance": "tight"}, "training.fps_tolerance", "a number"),
        ({"fps_tolerance": True}, "training.fps_tolerance", "a number"),
    ],
)
def test_wrong_type_is_rejected(tmp_path: Path, training: dict[str, Any], name: str, requirement: str) -> None:
    with pytest.raises(ConfigError, match=rf"{name} .*must be .*{requirement}"):
        _load_with(tmp_path, training)


# --- value rules (D-023 revised) -------------------------------------------------------


@pytest.mark.parametrize(
    ("training", "name"),
    [
        ({"stage": "bogus"}, "training.stage"),
        ({"models": ["nope"]}, "training.models"),
        ({"models": ["logreg", 3]}, "training.models"),
        ({"models": []}, "training.models"),
        ({"models": ["logreg", "logreg"]}, "training.models"),
        ({"folds": 1}, "training.folds"),
        ({"folds": -2}, "training.folds"),
        ({"repeats": 0}, "training.repeats"),
        ({"seed": -1}, "training.seed"),
        ({"min_class_size": 1}, "training.min_class_size"),
        ({"min_state_fish": 0}, "training.min_state_fish"),
        ({"min_detected_fraction": 1.5}, "training.min_detected_fraction"),
        ({"min_detected_fraction": math.nan}, "training.min_detected_fraction"),
        ({"duration_range_s": "banana"}, "training.duration_range_s"),
        ({"duration_range_s": [200, 100]}, "training.duration_range_s"),
        ({"duration_range_s": [100]}, "training.duration_range_s"),
        ({"duration_range_s": [-5, 100]}, "training.duration_range_s"),
        ({"mlp": {"hidden_sizes": []}}, "training.mlp.hidden_sizes"),
        ({"mlp": {"hidden_sizes": [64, 0]}}, "training.mlp.hidden_sizes"),
        ({"mlp": {"hidden_sizes": [64.5]}}, "training.mlp.hidden_sizes"),
        ({"mlp": {"dropout": 1.0}}, "training.mlp.dropout"),
        ({"mlp": {"dropout": math.nan}}, "training.mlp.dropout"),
        ({"mlp": {"learning_rate": 0}}, "training.mlp.learning_rate"),
        ({"mlp": {"weight_decay": -0.1}}, "training.mlp.weight_decay"),
        ({"mlp": {"batch_size": 0}}, "training.mlp.batch_size"),
        ({"mlp": {"max_epochs": 0}}, "training.mlp.max_epochs"),
        ({"mlp": {"patience": 0}}, "training.mlp.patience"),
        ({"mlp": {"inner_val_fraction": 0}}, "training.mlp.inner_val_fraction"),
        ({"mlp": {"inner_val_fraction": 1}}, "training.mlp.inner_val_fraction"),
        ({"mlp": {"seeds": 0}}, "training.mlp.seeds"),
        ({"min_detected_fraction": -0.1}, "training.min_detected_fraction"),
        ({"min_detected_fraction": 10**400}, "training.min_detected_fraction"),
        ({"mlp": {"dropout": -0.1}}, "training.mlp.dropout"),
        ({"mlp": {"learning_rate": -1}}, "training.mlp.learning_rate"),
        ({"mlp": {"learning_rate": math.inf}}, "training.mlp.learning_rate"),
        ({"mlp": {"weight_decay": math.inf}}, "training.mlp.weight_decay"),
        ({"duration_range_s": [0, math.inf]}, "training.duration_range_s"),
        ({"duration_range_s": [5, 5]}, "training.duration_range_s"),
        ({"duration_range_s": [1, 2, 3]}, "training.duration_range_s"),
        ({"duration_range_s": [True, 5]}, "training.duration_range_s"),
        ({"camera_epochs": []}, "training.camera_epochs"),
        ({"camera_epochs": ["2000-06-01", "2000-03-01"]}, "training.camera_epochs"),
        ({"camera_epochs": ["March"]}, "training.camera_epochs"),
        ({"camera_epochs": "2000-03-01"}, "training.camera_epochs"),
        ({"gold_source": "reviewed"}, "training.gold_source"),
        ({"model_profile_marker": ""}, "training.model_profile_marker"),
        ({"model_profile_marker": "   "}, "training.model_profile_marker"),
        ({"fps_tolerance": -0.01}, "training.fps_tolerance"),
        ({"fps_tolerance": 1.5}, "training.fps_tolerance"),
        ({"fps_tolerance": math.nan}, "training.fps_tolerance"),
    ],
)
def test_out_of_range_value_is_rejected(tmp_path: Path, training: dict[str, Any], name: str) -> None:
    with pytest.raises(ConfigError, match=rf"{name} must be"):
        _load_with(tmp_path, training)


@pytest.mark.parametrize(
    ("training", "key", "stored"),
    [
        ({"stage": "compound"}, "stage", "compound"),
        ({"models": ["logreg"]}, "models", ("logreg",)),
        ({"folds": 2}, "folds", 2),
        ({"repeats": 1}, "repeats", 1),
        ({"min_class_size": 2}, "min_class_size", 2),
        ({"min_state_fish": 1}, "min_state_fish", 1),
        ({"min_detected_fraction": 0}, "min_detected_fraction", 0),
        ({"min_detected_fraction": 1}, "min_detected_fraction", 1),
        ({"duration_range_s": [0, 100]}, "duration_range_s", (0, 100)),
        ({"duration_range_s": [1100.5, 1300]}, "duration_range_s", (1100.5, 1300)),
        ({"mlp": {"dropout": 0}}, "mlp.dropout", 0),
        ({"mlp": {"weight_decay": 0}}, "mlp.weight_decay", 0),
        ({"mlp": {"inner_val_fraction": 0.5}}, "mlp.inner_val_fraction", 0.5),
        ({"mlp": {"hidden_sizes": [8]}}, "mlp.hidden_sizes", (8,)),
        ({"gold_source": "accepted"}, "gold_source", "accepted"),
        ({"model_profile_marker": "_yolo"}, "model_profile_marker", "_yolo"),
        ({"fps_tolerance": 0}, "fps_tolerance", 0),
        ({"fps_tolerance": 1}, "fps_tolerance", 1),
    ],
)
def test_boundary_values_are_accepted(tmp_path: Path, training: dict[str, Any], key: str, stored: Any) -> None:
    value: Any = _load_with(tmp_path, training).training
    for part in key.split("."):
        value = value[part]
    assert value == stored


def test_every_default_value_has_exactly_one_rule() -> None:
    """A new numeric or text default must get a value rule; booleans are fully checked by type."""
    defaults = yaml.safe_load(DEFAULT_CONFIG_FILE.read_text(encoding="utf-8"))

    def leaves(section: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
        found: list[tuple[str, Any]] = []
        for key, value in section.items():
            if isinstance(value, dict):
                found.extend(leaves(value, f"{prefix}{key}."))
            else:
                found.append((f"{prefix}{key}", value))
        return found

    needs_rule = {key for key, value in leaves(defaults) if not isinstance(value, bool)}
    rule_keys = [key for key, _ in VALUE_RULES]
    assert len(rule_keys) == len(set(rule_keys))
    assert set(rule_keys) == needs_rule


# --- unreadable files --------------------------------------------------------------------


def test_malformed_yaml_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("training: [unclosed\n  seed: 1\n")
    with pytest.raises(ConfigError, match="bad.yaml.*not valid YAML"):
        load_settings(environ={}, config_file=path)


def test_yaml_error_message_is_one_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("training: [unclosed\n  seed: 1\n")
    with pytest.raises(ConfigError) as caught:
        load_settings(environ={}, config_file=path)
    assert "\n" not in str(caught.value)


def test_impossible_date_in_yaml_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "date.yaml"
    path.write_text("training:\n  seed: 2024-13-45\n")
    with pytest.raises(ConfigError, match="date.yaml.*not valid YAML"):
        load_settings(environ={}, config_file=path)


def test_non_utf8_yaml_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "latin1.yaml"
    path.write_bytes("training:\n  stage: caf\xe9\n".encode("latin-1"))
    with pytest.raises(ConfigError, match="latin1.yaml"):
        load_settings(environ={}, config_file=path)


def test_missing_override_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_settings(environ={}, config_file=tmp_path / "absent.yaml")


@pytest.mark.parametrize("text", ["- a\n- b\n", "[]\n", "0\n", "false\n", "''\n"])
def test_override_must_be_a_mapping(tmp_path: Path, text: str) -> None:
    path = tmp_path / "not_mapping.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(environ={}, config_file=path)


def test_override_that_is_a_folder_is_reported_as_such(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a file"):
        load_settings(environ={}, config_file=tmp_path)


def test_empty_override_name_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="empty"):
        load_settings(environ={}, config_file="")


# --- read-only, picklable -------------------------------------------------------------------


def test_sections_are_read_only() -> None:
    training = load_settings(environ={}).training
    with pytest.raises(TypeError):
        training["seed"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        training["mlp"]["dropout"] = 0.9  # type: ignore[index]


def test_lists_are_read_only() -> None:
    training = load_settings(environ={}).training
    with pytest.raises(AttributeError):
        training["models"].append("x")  # type: ignore[union-attr]
    with pytest.raises(AttributeError):
        training["mlp"]["hidden_sizes"].append(8)  # type: ignore[union-attr]


def test_settings_survive_pickle_and_deepcopy() -> None:
    settings = load_settings(environ={"DCS_DB_PATH": "book.xlsx"})
    for clone in (pickle.loads(pickle.dumps(settings)), copy.deepcopy(settings)):
        assert clone == settings
        assert clone.training["mlp"]["hidden_sizes"] == (128, 64)
        assert type(clone.training) is type(settings.training)
        with pytest.raises(TypeError):
            clone.training["seed"] = 1  # type: ignore[index]
