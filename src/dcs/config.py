"""Load classifier settings from the environment (.env) and YAML files.

Same two-kind-of-setting split as `prepds.config` (paths vs. tunable
parameters), restated here with this package's own `DCS_*` variables: `dcs`
never imports `prepds` (classifier PRD NFR-6, D-013).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml
from dotenv import dotenv_values

from dcs.config_rules import MODEL_NAMES, STAGES, first_violation, is_http_url  # noqa: F401  (choices re-exported for callers)

# Defaults live in the repository's config/ folder, next to the prepds thresholds (PRD §7.1).
# Like prepds, this needs a source checkout or `pip install -e .`, not a wheel.
DEFAULT_CONFIG_FILE = Path(__file__).parent.parent.parent / "config" / "default_training.yaml"

# Environment variable names, one place only, so a typo cannot hide in a step.
ENV_ACCEPTED_DIR = "DCS_ACCEPTED_DIR"
ENV_DB_PATH = "DCS_DB_PATH"
ENV_PROCESSED_DIR = "DCS_PROCESSED_DIR"
ENV_OUTPUT_DIR = "DCS_OUTPUT_DIR"
ENV_TABLE = "DCS_TABLE"
ENV_CONFIG = "DCS_CONFIG"
ENV_CHAT_BASE_URL = "DCS_CHAT_BASE_URL"  # the chat server's address: machine-specific, so never in a committed file
ENV_CHAT_API_KEY = "DCS_CHAT_API_KEY"  # key for a hosted chat server: a secret, so only in .env or the environment
# Chat settings read only from .env / the environment; the same name in a YAML file is refused.
ENV_ONLY_SETTINGS = {"chat.base_url": ENV_CHAT_BASE_URL, "chat.api_key": ENV_CHAT_API_KEY}

DEFAULT_ACCEPTED_DIR = Path("accepted")
DEFAULT_OUTPUT_DIR = Path("outputs/dcs")
TABLE_FILE = "training_table.parquet"

# Maps the short names used in code (settings.require("table")) to their env var.
PATH_VARIABLES = {
    "accepted_dir": ENV_ACCEPTED_DIR,
    "db_path": ENV_DB_PATH,
    "processed_dir": ENV_PROCESSED_DIR,
    "output_dir": ENV_OUTPUT_DIR,
    "table": ENV_TABLE,
}

# What to do when a needed path does not exist, per setting.
MISSING_PATH_HINTS = {
    "accepted_dir": "Run `prepds export-index` on the machine that holds the gold dataset, "
    f"or point {ENV_ACCEPTED_DIR} at it.",
    "db_path": f"Point {ENV_DB_PATH} at the trial workbook.",
    "processed_dir": f"Point {ENV_PROCESSED_DIR} at the prepds output folder.",
    "output_dir": f"Point {ENV_OUTPUT_DIR} at a writable folder.",
    "table": f"Run `python -m dcs featurize` first, or point {ENV_TABLE} at a copied training table.",
}

class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid (message explains the fix)."""


@dataclass(frozen=True)
class Secret:
    """A key that prints as *** (check-config, reprs, tracebacks); `.value` is the real text."""

    value: str = field(repr=False)

    def __str__(self) -> str:
        return "***"


class ReadOnlyMapping(Mapping[str, Any]):
    """Read-only, picklable settings section; nested sections and lists are frozen too (lists become tuples)."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = {key: _freeze(value) for key, value in data.items()}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"


@dataclass(frozen=True)
class Paths:
    """Machine-specific locations. `None` means "not configured" (only some steps need them)."""

    accepted_dir: Path  # gold dataset; defaults to ./accepted
    db_path: Path | None  # trial workbook, for the NTT columns (D-004)
    processed_dir: Path | None  # prepds processed outputs, for the tracker cross-check (D-003)
    output_dir: Path  # everything dcs writes; defaults to ./outputs/dcs (D-014)
    table: Path  # training table; defaults to <output_dir>/training_table.parquet


@dataclass(frozen=True)
class Settings:
    """Everything a step needs to know about its run environment."""

    paths: Paths
    params: ReadOnlyMapping  # merged YAML parameters
    env_file: Path | None = None  # which .env was read (None = no file, env vars only)
    explicit_paths: frozenset[str] = frozenset()  # path names set by the user rather than defaulted

    @property
    def training(self) -> ReadOnlyMapping:
        """The `training:` section (classifier PRD §7.3)."""
        return self.params["training"]

    @property
    def chat(self) -> ReadOnlyMapping:
        """The `chat:` section (research chat, D-073); `base_url` and `api_key` come from .env (ENV_ONLY_SETTINGS)."""
        return self.params["chat"]

    def require(self, name: str) -> Path:
        """Return a configured path that must exist, or fail with a clear "how to fix" message."""
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
            raise ConfigError(f"{env_name} points to {value}, which does not exist. {MISSING_PATH_HINTS[name]}")
        return value


def load_settings(
    env_file: str | os.PathLike[str] | None = None,
    config_file: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Build the `Settings` for this run.

    Precedence (highest first): `environ` (defaults to `os.environ`), then the
    `.env` file (`env_file`, or `./.env` if it exists), then built-in defaults.
    Parameters are the packaged defaults deep-merged with `config_file` (or
    the DCS_CONFIG file); unknown keys, wrong types and out-of-range values
    are errors (D-023).
    """
    environ = os.environ if environ is None else environ

    env_path = _expand(os.fspath(env_file), "env file") if env_file is not None else Path.cwd() / ".env"
    if env_file is not None and not env_path.is_file():
        # The user asked for a specific file, so its absence is an error, not a silent fallback.
        raise ConfigError(f"Env file {env_path} does not exist or is not a file.")
    file_values = _read_env_file(env_path) if env_path.is_file() else {}

    def lookup(name: str) -> str | None:
        """Environment first, then .env; blank or whitespace-only values count as "not set"."""
        for source in (environ, file_values):
            value = (source.get(name) or "").strip()
            if value:
                return value
        return None

    explicit = {name: _as_path(lookup(env_name), env_name) for name, env_name in PATH_VARIABLES.items()}
    output_dir = explicit["output_dir"] or DEFAULT_OUTPUT_DIR
    paths = Paths(
        accepted_dir=explicit["accepted_dir"] or DEFAULT_ACCEPTED_DIR,
        db_path=explicit["db_path"],
        processed_dir=explicit["processed_dir"],
        output_dir=output_dir,
        table=explicit["table"] or output_dir / TABLE_FILE,
    )

    override_file = config_file if config_file is not None else lookup(ENV_CONFIG)
    params = _load_params(_expand(os.fspath(override_file), "config file") if override_file is not None else None)
    base_url = lookup(ENV_CHAT_BASE_URL)
    if base_url is not None and not is_http_url(base_url):
        raise ConfigError(f"{ENV_CHAT_BASE_URL} must be an http:// or https:// address, got {base_url!r}.")
    params["chat"]["base_url"] = base_url  # None = not set; `dcs ask` and `serve-chat` then say where to set it
    api_key = lookup(ENV_CHAT_API_KEY)
    params["chat"]["api_key"] = Secret(api_key) if api_key is not None else None

    return Settings(
        paths=paths,
        params=ReadOnlyMapping(params),
        env_file=env_path if env_path.is_file() else None,
        explicit_paths=frozenset(name for name, value in explicit.items() if value is not None),
    )


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of `base` with `override` applied, merging nested sections key by key.

    An empty section in the override (`training:` with nothing under it) changes nothing.
    """
    merged = {key: copy.deepcopy(value) for key, value in base.items()}
    for key, value in override.items():
        if isinstance(merged.get(key), Mapping) and value is None:
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)  # recurse into sub-sections
        else:
            merged[key] = copy.deepcopy(value)  # plain values / lists replace the default
    return merged


# --- parameters: read, check, merge ---------------------------------------------------


def _load_params(override_path: Path | None) -> dict[str, Any]:
    """Packaged defaults, merged with the override file when given, then checked against the value rules."""
    if not DEFAULT_CONFIG_FILE.is_file():
        raise ConfigError(
            f"Packaged defaults {DEFAULT_CONFIG_FILE} not found. Install dcs from a source checkout "
            "with `pip install -e .`."
        )
    params = _read_yaml(DEFAULT_CONFIG_FILE)
    source = DEFAULT_CONFIG_FILE
    if override_path is not None:
        if not override_path.is_file():
            raise ConfigError(f"Config file {override_path} does not exist or is not a file.")
        override = _read_yaml(override_path)
        _check_types(params, override, source=override_path)
        params = deep_merge(params, override)
        source = override_path
    _check_values(params, source)
    return params


def _check_types(defaults: Mapping[str, Any], override: Mapping[str, Any], source: Path, prefix: str = "") -> None:
    """Reject keys the defaults do not have and values of the wrong kind, naming the dotted key."""
    for key, value in override.items():
        name = f"{prefix}{key}"
        if name in ENV_ONLY_SETTINGS:
            raise ConfigError(f"{name} in {source} is not read from YAML: set {ENV_ONLY_SETTINGS[name]} in your .env instead.")
        if key not in defaults:
            valid = ", ".join(sorted(str(known) for known in defaults))
            raise ConfigError(f"Unknown setting {name!r} in {source}. Valid keys here: {valid}.")
        default = defaults[key]
        if isinstance(default, Mapping):
            if value is not None and not isinstance(value, Mapping):
                raise ConfigError(f"{name} in {source} must be a section (key: value), got {value!r}.")
            _check_types(default, value or {}, source, prefix=f"{name}.")
        elif not _same_kind(default, value):
            raise ConfigError(f"{name} in {source} must be {_kind_name(default)}, got {value!r}.")


def _same_kind(default: Any, value: Any) -> bool:
    """True when `value` may replace `default`: same type, an int for a float, anything for a null default."""
    if default is None:
        return True  # the value rules check what a null default may hold
    if isinstance(default, bool) or isinstance(value, bool):
        return isinstance(default, bool) and isinstance(value, bool)  # bool is an int subclass
    if isinstance(default, float):
        return isinstance(value, (int, float))
    return isinstance(value, type(default))


def _kind_name(default: Any) -> str:
    """How a value of the default's type is described in an error message."""
    if isinstance(default, bool):
        return "true or false"
    if isinstance(default, int):
        return "a whole number"
    if isinstance(default, float):
        return "a number"
    if isinstance(default, list):
        return "a list"
    return "text"


def _check_values(params: Mapping[str, Any], source: Path) -> None:
    """Apply the value rules of dcs.config_rules to the merged parameters; the first violation is a ConfigError."""
    violation = first_violation(params)
    if violation is not None:
        key, requirement, value = violation
        raise ConfigError(f"{key} must be {requirement}, got {value!r} (check {source}).")


# --- small private helpers ------------------------------------------------------


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ReadOnlyMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _read_env_file(path: Path) -> dict[str, str]:
    """Read KEY=VALUE pairs without writing to os.environ (python-dotenv may still read it for ${VAR})."""
    try:
        values = dotenv_values(path)
    except (UnicodeDecodeError, OSError) as error:
        raise ConfigError(f"Cannot read env file {path}: {_one_line(error)}") from None
    return {key: value for key, value in values.items() if value is not None}


def _as_path(value: str | None, what: str) -> Path | None:
    """Turn a string into a Path, expanding `~`; keep None as None."""
    return _expand(value, what) if value is not None else None


def _expand(value: str, what: str) -> Path:
    """Path with `~` expanded; an empty name or an unknown `~user` is a ConfigError, not a traceback."""
    if not value.strip():
        raise ConfigError(f"The {what} name is empty.")
    try:
        return Path(value).expanduser()
    except RuntimeError as error:  # e.g. "~nosuchuser/...": no home directory to expand to
        raise ConfigError(f"Cannot expand {value!r} ({what}): {error}") from None


def _one_line(error: Exception) -> str:
    """Parser messages span several lines (with position marks); the CLI shows one line."""
    return " ".join(str(error).split())


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping; an empty file counts as no overrides."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (yaml.YAMLError, UnicodeDecodeError, ValueError) as error:  # ValueError: an impossible date such as 2024-13-45
        raise ConfigError(f"{path} is not valid YAML: {_one_line(error)}") from None
    except OSError as error:
        raise ConfigError(f"Cannot read {path}: {error}") from None
    if data is None:
        return {}  # an empty file means "no overrides"; [] / 0 / false are not mappings and fail below
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping (key: value), not {type(data).__name__}.")
    return data
