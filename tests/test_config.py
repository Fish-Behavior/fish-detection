"""Tests for fishbehavior.config (settings loading). Uses temporary files only — no real data."""

from pathlib import Path

import pytest

from fishbehavior.config import ConfigError, deep_merge, load_settings


def write(path: Path, text: str) -> Path:
    """Small helper: create a file with the given text and return its path."""
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_when_nothing_is_configured(tmp_path, monkeypatch):
    # Run from an empty folder with an empty environment: no .env, no variables.
    monkeypatch.chdir(tmp_path)
    settings = load_settings(environ={})

    assert settings.env_file is None
    assert settings.paths.video_dir is None
    assert settings.paths.db_path is None
    assert settings.paths.output_dir == Path("outputs")  # default output folder
    assert settings.workers == 1
    assert settings.params["run"]["log_level"] == "INFO"  # packaged default_config.yaml was read


def test_env_file_in_current_folder_is_picked_up(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / ".env", "FISH_VIDEO_DIR=videos\nFISH_WORKERS=4\n")

    settings = load_settings(environ={})

    assert settings.env_file == tmp_path / ".env"
    assert settings.paths.video_dir == Path("videos")
    assert settings.workers == 4


def test_environment_variables_override_env_file(tmp_path):
    # Colab/CI set real environment variables; those must win over the .env file.
    env_file = write(tmp_path / "local.env", "FISH_DB_PATH=from_file.xlsx\n")

    settings = load_settings(env_file=env_file, environ={"FISH_DB_PATH": "from_env.xlsx"})

    assert settings.paths.db_path == Path("from_env.xlsx")


def test_blank_values_count_as_not_set(tmp_path):
    env_file = write(tmp_path / "local.env", "FISH_VIDEO_DIR=\nFISH_OUTPUT_DIR=   \n")

    settings = load_settings(env_file=env_file, environ={})

    assert settings.paths.video_dir is None
    assert settings.paths.output_dir == Path("outputs")


def test_explicit_env_file_must_exist(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_settings(env_file=tmp_path / "missing.env", environ={})


@pytest.mark.parametrize("value", ["zero", "0", "-2"])
def test_invalid_worker_count_is_rejected(tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="FISH_WORKERS"):
        load_settings(environ={"FISH_WORKERS": value})


def test_user_yaml_overrides_only_the_given_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    override = write(tmp_path / "mine.yaml", "run:\n  log_level: DEBUG\nextra:\n  bin_s: 0.25\n")

    settings = load_settings(environ={"FISH_CONFIG": str(override)})

    assert settings.params["run"]["log_level"] == "DEBUG"  # overridden
    assert settings.params["extra"]["bin_s"] == 0.25  # new section added


def test_missing_user_yaml_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="does not exist"):
        load_settings(config_file=tmp_path / "nope.yaml", environ={})


def test_user_yaml_must_be_a_mapping(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad = write(tmp_path / "bad.yaml", "- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(config_file=bad, environ={})


def test_require_reports_unset_and_missing_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "videos"
    existing.mkdir()

    settings = load_settings(environ={"FISH_VIDEO_DIR": str(existing), "FISH_DB_PATH": str(tmp_path / "gone.xlsx")})

    assert settings.require("video_dir") == existing  # configured and present
    with pytest.raises(ConfigError, match="FISH_DB_PATH points to"):
        settings.require("db_path")  # configured but the file is missing
    with pytest.raises(ConfigError, match="FISH_REFERENCE_PDF is not set"):
        settings.require("reference_pdf")  # never configured
    with pytest.raises(KeyError):
        settings.require("not_a_setting")  # programming mistake, not a user error


def test_deep_merge_keeps_untouched_keys_and_does_not_mutate_inputs():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2]}
    override = {"a": {"y": 20}, "b": [3]}

    merged = deep_merge(base, override)

    assert merged == {"a": {"x": 1, "y": 20}, "b": [3]}  # nested merge, lists replaced
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2]}  # original left unchanged
