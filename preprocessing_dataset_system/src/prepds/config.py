"""Load pipeline settings from the environment (.env) and YAML files.

Same two-kind-of-setting split as `fishbehavior.config` (paths vs. tunable
parameters), with this package's own independent `PDS_*` variables (PRD
Clarification C10 - no shared dependency between the two packages).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import dotenv_values

# Built-in defaults live next to this file so they are available after `pip install`.
DEFAULT_CONFIG_FILE = Path(__file__).parent.parent.parent / "config" / "default_thresholds.yaml"

# Environment variable names, one place only, so a typo cannot hide in a step.
ENV_VIDEO_DIR = "PDS_VIDEO_DIR"
ENV_DB_PATH = "PDS_DB_PATH"
ENV_REFERENCE_DIR = "PDS_REFERENCE_DIR"
ENV_OUTPUT_DIR = "PDS_OUTPUT_DIR"
ENV_ACCEPTED_DIR = "PDS_ACCEPTED_DIR"
ENV_WORKERS = "PDS_WORKERS"
ENV_CONFIG = "PDS_CONFIG"

# Maps the short names used in code (settings.require("video_dir")) to their env var.
PATH_VARIABLES = {
    "video_dir": ENV_VIDEO_DIR,
    "db_path": ENV_DB_PATH,
    "reference_dir": ENV_REFERENCE_DIR,
}


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid (message explains the fix)."""


@dataclass(frozen=True)
class Paths:
    """Machine-specific locations. `None` means "not configured" (not every step needs every path)."""

    video_dir: Path | None
    db_path: Path | None
    reference_dir: Path | None
    output_dir: Path  # always set; defaults to ./outputs
    accepted_dir: Path  # always set; defaults to ./accepted


@dataclass(frozen=True)
class Settings:
    """Everything a pipeline step needs to know about its run environment."""

    paths: Paths
    workers: int
    params: dict[str, Any] = field(default_factory=dict)  # merged YAML parameters
    env_file: Path | None = None  # which .env was read (None = no file, env vars only)

    def require(self, name: str) -> Path:
        """Return a configured input path, or fail with a clear "how to fix" message.

        Steps call this for the inputs they truly need, e.g.
        `settings.require("video_dir")`, so a missing .env entry is reported
        by name instead of surfacing later as an obscure file error.
        """
        if name not in PATH_VARIABLES:
            raise KeyError(f"Unknown path setting {name!r}; expected one of {sorted(PATH_VARIABLES)}")
        value = getattr(self.paths, name)
        env_name = PATH_VARIABLES[name]
        if value is None:
            raise ConfigError(
                f"{env_name} is not set. Add it to your .env file (see .env.example) "
                f"or set it as an environment variable."
            )
        if not value.exists():
            raise ConfigError(f"{env_name} points to {value}, which does not exist.")
        return value


def load_settings(
    env_file: str | os.PathLike[str] | None = None,
    config_file: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Build the `Settings` for this run.

    Precedence (highest first) - real environment variables win over `.env`
    so that CI / batch runs can set values without editing files:

    1. `environ` (defaults to `os.environ`)
    2. the `.env` file (`env_file`, or `./.env` if it exists)
    3. built-in defaults

    Parameters are the packaged defaults deep-merged with `config_file`
    (or the PDS_CONFIG file) when given.
    """
    environ = os.environ if environ is None else environ

    # --- 1. Find and read the .env file (it is optional) -----------------------
    env_path = Path(env_file) if env_file is not None else Path.cwd() / ".env"
    if env_file is not None and not env_path.is_file():
        # The user asked for a specific file, so its absence is an error, not a silent fallback.
        raise ConfigError(f"Env file {env_path} does not exist.")
    file_values = _read_env_file(env_path) if env_path.is_file() else {}

    def lookup(name: str) -> str | None:
        """Environment first, then .env; blank values count as "not set"."""
        value = environ.get(name) or file_values.get(name)
        return value.strip() if value and value.strip() else None

    # --- 2. Paths ---------------------------------------------------------------
    paths = Paths(
        video_dir=_as_path(lookup(ENV_VIDEO_DIR)),
        db_path=_as_path(lookup(ENV_DB_PATH)),
        reference_dir=_as_path(lookup(ENV_REFERENCE_DIR)),
        output_dir=_as_path(lookup(ENV_OUTPUT_DIR)) or Path("outputs"),
        accepted_dir=_as_path(lookup(ENV_ACCEPTED_DIR)) or Path("accepted"),
    )

    # --- 3. Worker count (parallel videos) -----------------------------------------
    workers = _parse_workers(lookup(ENV_WORKERS))

    # --- 4. Parameters: packaged defaults, then the user's overrides ------------
    params = _read_yaml(DEFAULT_CONFIG_FILE)
    override_file = config_file if config_file is not None else lookup(ENV_CONFIG)
    if override_file is not None:
        override_path = Path(override_file).expanduser()
        if not override_path.is_file():
            raise ConfigError(f"Config file {override_path} does not exist.")
        params = deep_merge(params, _read_yaml(override_path))

    return Settings(
        paths=paths,
        workers=workers,
        params=params,
        env_file=env_path if env_path.is_file() else None,
    )


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of `base` with `override` applied, merging nested sections key by key.

    This lets a user YAML change one value (e.g. `labeling.min_freeze_bout_s`)
    without having to repeat the whole section.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)  # recurse into sub-sections
        else:
            merged[key] = copy.deepcopy(value)  # plain values / lists replace the default
    return merged


# --- small private helpers ------------------------------------------------------


def _read_env_file(path: Path) -> dict[str, str]:
    """Read KEY=VALUE pairs without touching os.environ (keeps runs and tests isolated)."""
    return {key: value for key, value in dotenv_values(path).items() if value is not None}


def _as_path(value: str | None) -> Path | None:
    """Turn a string into a Path, expanding `~`; keep None as None."""
    return Path(value).expanduser() if value is not None else None


def default_workers() -> int:
    """PRD §9.5.9: leave two cores for the OS and the review UI."""
    return max(1, (os.cpu_count() or 1) - 2)


def _parse_workers(value: str | None) -> int:
    """Parse PDS_WORKERS; default `default_workers()`, must be a positive integer."""
    if value is None:
        return default_workers()
    try:
        workers = int(value)
    except ValueError:
        raise ConfigError(f"{ENV_WORKERS} must be a whole number, got {value!r}.") from None
    if workers < 1:
        raise ConfigError(f"{ENV_WORKERS} must be at least 1, got {workers}.")
    return workers


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping; an empty file counts as no overrides."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping (key: value), not {type(data).__name__}.")
    return data
