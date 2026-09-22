"""Tests for the command-line entry point. Uses temporary files only — no real data."""

import pytest

from fishbehavior.cli import main


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """Run each test from an empty folder with none of the FISH_* variables set."""
    for name in ("FISH_VIDEO_DIR", "FISH_DB_PATH", "FISH_REFERENCE_PDF", "FISH_OUTPUT_DIR", "FISH_WORKERS", "FISH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_check_config_all_present(clean_env, capsys):
    (clean_env / "videos").mkdir()
    (clean_env / ".env").write_text("FISH_VIDEO_DIR=videos\n", encoding="utf-8")

    code = main(["check-config"])

    out = capsys.readouterr().out
    assert code == 0
    assert "FISH_VIDEO_DIR" in out and "[ok]" in out
    assert "FISH_DB_PATH" in out and "[not set]" in out  # unset is allowed, not an error


def test_check_config_flags_missing_paths(clean_env, capsys):
    (clean_env / ".env").write_text("FISH_DB_PATH=missing.xlsx\n", encoding="utf-8")

    code = main(["check-config"])

    assert code == 1
    assert "[MISSING]" in capsys.readouterr().out


def test_config_errors_are_one_line_messages(clean_env, capsys):
    code = main(["--env-file", "does-not-exist.env", "check-config"])

    assert code == 2
    assert capsys.readouterr().out.startswith("Configuration error:")


def test_invalid_log_level_is_reported(clean_env, capsys):
    (clean_env / "mine.yaml").write_text("run:\n  log_level: LOUD\n", encoding="utf-8")

    code = main(["--config", "mine.yaml", "check-config"])

    assert code == 2
    assert "log_level" in capsys.readouterr().out


def test_a_command_is_required(clean_env):
    with pytest.raises(SystemExit):
        main([])
