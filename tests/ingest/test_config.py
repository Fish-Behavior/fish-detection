"""Env precedence and missing/invalid path errors for prepds.config (T006).

Mirrors fishbehavior's own test_config.py conventions but exercises the
independent PDS_* variable set (see PRD C10 - no shared dependency).

Every test chdir's into an empty tmp_path (via monkeypatch) before calling
load_settings with env_file=None, because load_settings falls back to
`Path.cwd() / ".env"` - without this isolation, a real local `.env` (which
README step 4 tells the owner to create via `cp .env.example .env`) would
leak into these tests and make them non-deterministic depending on the
directory pytest happens to run from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prepds.config import (
    default_workers,
    ConfigError,
    PATH_VARIABLES,
    load_settings,
)


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in this file from an empty directory with no real .env."""
    monkeypatch.chdir(tmp_path)


def test_defaults_when_nothing_set() -> None:
    settings = load_settings(environ={}, env_file=None)
    assert settings.paths.video_dir is None
    assert settings.paths.db_path is None
    assert settings.paths.reference_dir is None
    assert settings.paths.output_dir == Path("outputs")
    assert settings.paths.accepted_dir == Path("accepted")
    assert settings.workers == default_workers()


def test_env_var_sets_video_dir(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    settings = load_settings(environ={"PDS_VIDEO_DIR": str(video_dir)}, env_file=None)
    assert settings.paths.video_dir == video_dir


def test_real_environ_takes_precedence_over_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    file_video_dir = tmp_path / "from_file"
    real_video_dir = tmp_path / "from_environ"
    file_video_dir.mkdir()
    real_video_dir.mkdir()
    env_file.write_text(f"PDS_VIDEO_DIR={file_video_dir}\n")

    settings = load_settings(
        environ={"PDS_VIDEO_DIR": str(real_video_dir)},
        env_file=env_file,
    )
    assert settings.paths.video_dir == real_video_dir


def test_missing_env_file_raises_config_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.env"
    with pytest.raises(ConfigError):
        load_settings(env_file=missing, environ={})


def test_require_missing_path_raises_config_error() -> None:
    settings = load_settings(environ={}, env_file=None)
    with pytest.raises(ConfigError):
        settings.require("video_dir")


def test_require_nonexistent_path_raises_config_error(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nope"
    settings = load_settings(environ={"PDS_VIDEO_DIR": str(nonexistent)}, env_file=None)
    with pytest.raises(ConfigError):
        settings.require("video_dir")


def test_require_existing_path_returns_it(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    settings = load_settings(environ={"PDS_VIDEO_DIR": str(video_dir)}, env_file=None)
    assert settings.require("video_dir") == video_dir


def test_workers_must_be_positive_int() -> None:
    with pytest.raises(ConfigError):
        load_settings(environ={"PDS_WORKERS": "0"}, env_file=None)
    with pytest.raises(ConfigError):
        load_settings(environ={"PDS_WORKERS": "not-a-number"}, env_file=None)


def test_path_variables_cover_video_db_reference() -> None:
    assert set(PATH_VARIABLES) >= {"video_dir", "db_path", "reference_dir"}
