"""Paths, env precedence, require() and the CLI of `dcs` (plan U1, T1.2).

Every test runs from an empty tmp_path with no DCS_* variables (conftest.py)
and passes `environ` explicitly where it calls load_settings directly.
YAML parameters and their validation are in test_dcs_config_params.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import dcs.config
from dcs import __version__
from dcs.cli import main
from dcs.config import PATH_VARIABLES, ConfigError, load_settings

# --- paths and precedence ---------------------------------------------------------


def test_defaults_when_nothing_set() -> None:
    settings = load_settings(environ={})
    assert settings.paths.accepted_dir == Path("accepted")
    assert settings.paths.output_dir == Path("outputs/dcs")
    assert settings.paths.table == Path("outputs/dcs/training_table.parquet")
    assert settings.paths.db_path is None
    assert settings.paths.processed_dir is None
    assert settings.env_file is None
    assert settings.explicit_paths == frozenset()


def test_table_follows_output_dir_when_not_set() -> None:
    settings = load_settings(environ={"DCS_OUTPUT_DIR": "elsewhere"})
    assert settings.paths.table == Path("elsewhere/training_table.parquet")
    assert settings.explicit_paths == frozenset({"output_dir"})


def test_explicit_table_wins_over_output_dir() -> None:
    settings = load_settings(environ={"DCS_OUTPUT_DIR": "elsewhere", "DCS_TABLE": "t/table.parquet"})
    assert settings.paths.table == Path("t/table.parquet")
    assert settings.explicit_paths == frozenset({"output_dir", "table"})


def test_env_file_is_read(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("DCS_ACCEPTED_DIR=from_file\nDCS_DB_PATH=book.xlsx\n")
    settings = load_settings(env_file=env_file, environ={})
    assert settings.paths.accepted_dir == Path("from_file")
    assert settings.paths.db_path == Path("book.xlsx")
    assert settings.env_file == env_file
    assert settings.explicit_paths == frozenset({"accepted_dir", "db_path"})


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


def test_whitespace_environment_value_does_not_hide_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("DCS_OUTPUT_DIR=from_file\n")
    settings = load_settings(env_file=env_file, environ={"DCS_OUTPUT_DIR": "   "})
    assert settings.paths.output_dir == Path("from_file")


def test_home_is_expanded_in_paths_and_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "home.env").write_text("DCS_DB_PATH=~/book.xlsx\n")
    settings = load_settings(env_file="~/home.env", environ={})
    assert settings.paths.db_path == tmp_path / "book.xlsx"
    assert settings.env_file == tmp_path / "home.env"


@pytest.mark.parametrize("use_env_file", [False, True])
def test_unknown_user_home_is_a_config_error(tmp_path: Path, use_env_file: bool) -> None:
    unknown = "~no_such_user_dcs_test/x"
    with pytest.raises(ConfigError, match="no_such_user_dcs_test"):
        if use_env_file:
            load_settings(env_file=unknown, environ={})
        else:
            load_settings(environ={"DCS_DB_PATH": unknown})


def test_prepds_variables_are_ignored() -> None:
    settings = load_settings(environ={"PDS_ACCEPTED_DIR": "pds_accepted", "PDS_DB_PATH": "pds.xlsx"})
    assert settings.paths.accepted_dir == Path("accepted")
    assert settings.paths.db_path is None


def test_missing_explicit_env_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_settings(env_file=tmp_path / "nope.env", environ={})


def test_unreadable_env_file_is_a_config_error(tmp_path: Path) -> None:
    env_file = tmp_path / "latin1.env"
    env_file.write_bytes("DCS_DB_PATH=caf\xe9.xlsx\n".encode("latin-1"))
    with pytest.raises(ConfigError, match="latin1.env"):
        load_settings(env_file=env_file, environ={})


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


def test_missing_packaged_defaults_is_a_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dcs.config, "DEFAULT_CONFIG_FILE", tmp_path / "absent.yaml")
    with pytest.raises(ConfigError, match="pip install -e"):
        load_settings(environ={})


# --- CLI ------------------------------------------------------------------------------


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_python_dash_m_runs_the_cli() -> None:
    """From a foreign working directory, so the packaged defaults must resolve without the repo as cwd."""
    for args, expected in ((["--version"], __version__), (["check-config"], "training.folds")):
        result = subprocess.run(
            [sys.executable, "-m", "dcs", *args], capture_output=True, text=True, timeout=60, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert expected in result.stdout


def test_importing_main_module_does_not_exit() -> None:
    import dcs.__main__  # noqa: F401  (must not raise SystemExit when imported)


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_check_config_prints_paths_and_training(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-config"]) == 0
    out = capsys.readouterr().out
    for name in ("DCS_ACCEPTED_DIR", "DCS_OUTPUT_DIR", "DCS_TABLE", "training.min_class_size", "training.mlp.dropout"):
        assert name in out


@pytest.mark.parametrize("variable", ["DCS_DB_PATH", "DCS_PROCESSED_DIR", "DCS_ACCEPTED_DIR", "DCS_TABLE"])
def test_check_config_flags_an_explicit_but_missing_path(
    variable: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(variable, str(tmp_path / "typo"))
    assert main(["check-config"]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_check_config_does_not_flag_a_new_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DCS_OUTPUT_DIR", str(tmp_path / "new_outputs"))
    assert main(["check-config"]) == 0
    assert "will be created" in capsys.readouterr().out


def test_check_config_reads_env_file_option(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("DCS_OUTPUT_DIR=from_env_option\n")
    assert main(["--env-file", str(env_file), "check-config"]) == 0
    assert "from_env_option" in capsys.readouterr().out


def test_check_config_reads_dcs_config_from_env_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    override = tmp_path / "o.yaml"
    override.write_text("training:\n  folds: 3\n")
    (tmp_path / ".env").write_text(f"DCS_CONFIG={override}\n")
    assert main(["check-config"]) == 0
    assert "training.folds: 3" in capsys.readouterr().out


def test_config_error_is_one_line_and_exit_code_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("training: [unclosed\n  seed: 1\n")
    assert main(["--config", str(bad), "check-config"]) == 2
    out = capsys.readouterr().out
    assert out.startswith("Configuration error:")
    assert "Traceback" not in out
