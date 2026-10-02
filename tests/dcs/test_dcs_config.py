"""Settings, env precedence, YAML overrides and the check-config command of `dcs` (plan U1, T1.2).

Every test runs from an empty tmp_path (autouse fixture) and passes `environ`
explicitly, because load_settings falls back to `Path.cwd() / ".env"`: a real
local `.env` must never leak into these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dcs import __version__
from dcs.cli import main
from dcs.config import DEFAULT_CONFIG_FILE, PATH_VARIABLES, ConfigError, deep_merge, load_settings

DCS_VARIABLES = ("DCS_ACCEPTED_DIR", "DCS_DB_PATH", "DCS_PROCESSED_DIR", "DCS_OUTPUT_DIR", "DCS_TABLE", "DCS_CONFIG")


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in this file from an empty directory, with no real .env and no DCS_* variables."""
    monkeypatch.chdir(tmp_path)
    for name in DCS_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- paths and precedence ---------------------------------------------------------


def test_defaults_when_nothing_set() -> None:
    settings = load_settings(environ={})
    assert settings.paths.accepted_dir == Path("accepted")
    assert settings.paths.output_dir == Path("outputs/dcs")
    assert settings.paths.table == Path("outputs/dcs/training_table.parquet")
    assert settings.paths.db_path is None
    assert settings.paths.processed_dir is None
    assert settings.env_file is None


def test_table_follows_output_dir_when_not_set() -> None:
    settings = load_settings(environ={"DCS_OUTPUT_DIR": "elsewhere"})
    assert settings.paths.table == Path("elsewhere/training_table.parquet")


def test_explicit_table_wins_over_output_dir() -> None:
    settings = load_settings(environ={"DCS_OUTPUT_DIR": "elsewhere", "DCS_TABLE": "t/table.parquet"})
    assert settings.paths.table == Path("t/table.parquet")


def test_env_file_is_read(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("DCS_ACCEPTED_DIR=from_file\nDCS_DB_PATH=book.xlsx\n")
    settings = load_settings(env_file=env_file, environ={})
    assert settings.paths.accepted_dir == Path("from_file")
    assert settings.paths.db_path == Path("book.xlsx")
    assert settings.env_file == env_file


def test_dot_env_in_working_directory_is_found(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("DCS_PROCESSED_DIR=processed\n")
    settings = load_settings(environ={})
    assert settings.paths.processed_dir == Path("processed")


def test_real_environment_takes_precedence_over_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("DCS_OUTPUT_DIR=from_file\n")
    settings = load_settings(env_file=env_file, environ={"DCS_OUTPUT_DIR": "from_environ"})
    assert settings.paths.output_dir == Path("from_environ")


def test_blank_value_counts_as_not_set() -> None:
    settings = load_settings(environ={"DCS_DB_PATH": "   ", "DCS_OUTPUT_DIR": ""})
    assert settings.paths.db_path is None
    assert settings.paths.output_dir == Path("outputs/dcs")


def test_prepds_variables_are_ignored() -> None:
    settings = load_settings(environ={"PDS_ACCEPTED_DIR": "pds_accepted", "PDS_DB_PATH": "pds.xlsx"})
    assert settings.paths.accepted_dir == Path("accepted")
    assert settings.paths.db_path is None


def test_missing_explicit_env_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_settings(env_file=tmp_path / "nope.env", environ={})


# --- require() ----------------------------------------------------------------------


def test_require_unset_path_names_the_variable() -> None:
    settings = load_settings(environ={})
    with pytest.raises(ConfigError, match="DCS_DB_PATH.*\\.env"):
        settings.require("db_path")


def test_require_missing_table_hints_featurize(tmp_path: Path) -> None:
    settings = load_settings(environ={"DCS_TABLE": str(tmp_path / "absent.parquet")})
    with pytest.raises(ConfigError, match="dcs featurize"):
        settings.require("table")


def test_require_missing_accepted_dir_hints_export_index(tmp_path: Path) -> None:
    settings = load_settings(environ={"DCS_ACCEPTED_DIR": str(tmp_path / "absent")})
    with pytest.raises(ConfigError, match="prepds export-index"):
        settings.require("accepted_dir")


def test_require_existing_path_returns_it(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    settings = load_settings(environ={"DCS_PROCESSED_DIR": str(processed)})
    assert settings.require("processed_dir") == processed


def test_require_unknown_name_is_a_programming_error() -> None:
    settings = load_settings(environ={})
    with pytest.raises(KeyError):
        settings.require("video_dir")


def test_path_variables_use_the_dcs_prefix() -> None:
    assert PATH_VARIABLES
    assert all(env_name.startswith("DCS_") for env_name in PATH_VARIABLES.values())


# --- training parameters --------------------------------------------------------------


def test_packaged_defaults_are_loaded() -> None:
    assert DEFAULT_CONFIG_FILE.is_file()
    training = load_settings(environ={}).training
    assert training["seed"] == 0
    assert training["folds"] == 5
    assert training["min_class_size"] == 6
    assert training["min_state_fish"] == 10
    assert training["use_depth"] is False
    assert training["mlp"]["hidden_sizes"] == [128, 64]
    assert "majority" in training["models"] and "date_only" in training["models"]


def test_override_changes_only_listed_keys(tmp_path: Path) -> None:
    override = _write_yaml(tmp_path / "o.yaml", {"training": {"seed": 7, "mlp": {"dropout": 0.5}}})
    training = load_settings(environ={}, config_file=override).training
    assert training["seed"] == 7
    assert training["mlp"]["dropout"] == 0.5
    assert training["mlp"]["hidden_sizes"] == [128, 64]  # sibling key kept
    assert training["folds"] == 5


def test_dcs_config_variable_selects_the_override(tmp_path: Path) -> None:
    override = _write_yaml(tmp_path / "o.yaml", {"training": {"repeats": 2}})
    assert load_settings(environ={"DCS_CONFIG": str(override)}).training["repeats"] == 2


def test_explicit_config_file_wins_over_dcs_config(tmp_path: Path) -> None:
    from_env = _write_yaml(tmp_path / "env.yaml", {"training": {"repeats": 2}})
    explicit = _write_yaml(tmp_path / "explicit.yaml", {"training": {"repeats": 3}})
    settings = load_settings(environ={"DCS_CONFIG": str(from_env)}, config_file=explicit)
    assert settings.training["repeats"] == 3


def test_null_allowed_where_default_is_null(tmp_path: Path) -> None:
    override = _write_yaml(tmp_path / "o.yaml", {"training": {"duration_range_s": [100, 200]}})
    assert load_settings(environ={}, config_file=override).training["duration_range_s"] == [100, 200]


def test_int_accepted_for_float_setting(tmp_path: Path) -> None:
    override = _write_yaml(tmp_path / "o.yaml", {"training": {"min_detected_fraction": 1}})
    assert load_settings(environ={}, config_file=override).training["min_detected_fraction"] == 1


def test_empty_override_file_changes_nothing(tmp_path: Path) -> None:
    override = tmp_path / "empty.yaml"
    override.write_text("")
    assert load_settings(environ={}, config_file=override).training == load_settings(environ={}).training


@pytest.mark.parametrize(
    ("override", "names"),
    [
        ({"training": {"sead": 1}}, "training.sead"),
        ({"training": {"mlp": {"hidden": [8]}}}, "training.mlp.hidden"),
        ({"trainng": {"seed": 1}}, "trainng"),
    ],
)
def test_unknown_key_is_rejected_with_its_name(tmp_path: Path, override: dict, names: str) -> None:
    path = _write_yaml(tmp_path / "o.yaml", override)
    with pytest.raises(ConfigError, match=names) as caught:
        load_settings(environ={}, config_file=path)
    assert "valid" in str(caught.value).lower()


@pytest.mark.parametrize(
    "override",
    [
        {"training": {"folds": "five"}},
        {"training": {"folds": 5.5}},
        {"training": {"folds": True}},
        {"training": {"use_ntt": "yes"}},
        {"training": {"use_ntt": 1}},
        {"training": {"models": "logreg"}},
        {"training": {"mlp": 3}},
        {"training": {"seed": None}},
    ],
)
def test_wrong_type_is_rejected(tmp_path: Path, override: dict) -> None:
    path = _write_yaml(tmp_path / "o.yaml", override)
    with pytest.raises(ConfigError, match="training\\."):
        load_settings(environ={}, config_file=path)


def test_missing_override_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_settings(environ={}, config_file=tmp_path / "absent.yaml")


def test_override_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(environ={}, config_file=path)


def test_deep_merge_does_not_mutate_its_inputs() -> None:
    base = {"a": {"b": 1, "c": [1]}}
    override = {"a": {"b": 2}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 2, "c": [1]}}
    assert base == {"a": {"b": 1, "c": [1]}}
    merged["a"]["c"].append(2)
    assert base["a"]["c"] == [1]


def test_settings_training_is_read_only() -> None:
    settings = load_settings(environ={})
    with pytest.raises(TypeError):
        settings.training["seed"] = 99  # type: ignore[index]


# --- CLI ------------------------------------------------------------------------------


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_check_config_prints_paths_and_training(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-config"]) == 0
    out = capsys.readouterr().out
    for name in ("DCS_ACCEPTED_DIR", "DCS_OUTPUT_DIR", "DCS_TABLE", "min_class_size"):
        assert name in out


def test_check_config_reports_a_set_but_missing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DCS_DB_PATH", str(tmp_path / "absent.xlsx"))
    assert main(["check-config"]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_config_error_is_one_line_and_exit_code_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = _write_yaml(tmp_path / "bad.yaml", {"training": {"sead": 1}})
    assert main(["--config", str(bad), "check-config"]) == 2
    out = capsys.readouterr().out
    assert out.startswith("Configuration error:")
    assert "Traceback" not in out
