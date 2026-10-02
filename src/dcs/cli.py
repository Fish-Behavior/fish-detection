"""Command-line entry point: `python -m dcs <command> [options]`.

Each step registers one sub-command here in its own unit (synth, featurize,
audit, train, predict: classifier plan §3.1). This first version only has
`check-config`, which shows what `dcs` will read so a new `.env` or override
YAML can be verified before any data is touched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from dcs import __version__
from dcs.config import PATH_VARIABLES, ConfigError, Settings, load_settings

# Folders dcs creates itself, so their absence is never a problem.
CREATED_OUTPUTS = ("output_dir",)
# Default locations that exist only after an earlier step has run.
PRODUCED_INPUTS = {
    "accepted_dir": "not found yet (written by prepds export-index)",
    "table": "not found yet (written by dcs featurize)",
}
LABEL_WIDTH = 19


def build_parser() -> argparse.ArgumentParser:
    """Define the global options and one sub-parser per command."""
    parser = argparse.ArgumentParser(
        prog="dcs",
        description="Compound and dose classifier: gold dataset -> training table -> models and report.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    # Global options shared by every command: where settings come from.
    parser.add_argument("--env-file", help="path to a .env file (default: ./.env if it exists)")
    parser.add_argument("--config", help="YAML file overriding config/default_training.yaml (default: $DCS_CONFIG)")

    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    check = commands.add_parser("check-config", help="show the resolved paths and training settings, then exit")
    check.set_defaults(handler=run_check_config)  # each command points to the function that runs it
    return parser


def run_check_config(settings: Settings, args: argparse.Namespace) -> int:
    """Print every path and training setting `dcs` would use.

    Returns 0 when every path the user set explicitly exists (output folders
    excepted, they are created on demand), 1 otherwise, so it can be used as a
    quick "is my setup right?" test.
    """
    print(f"{'.env file':<{LABEL_WIDTH}}: {settings.env_file or '(none found - using environment variables only)'}")
    problems = 0
    for name, env_name in PATH_VARIABLES.items():
        value = getattr(settings.paths, name)
        status = _path_status(name, value, explicit=name in settings.explicit_paths)
        problems += status == "MISSING"
        print(f"{env_name:<{LABEL_WIDTH}}: {value if value is not None else '-'}  [{status}]")
    for key, value in _flatten(settings.params):
        print(f"{key}: {value}")
    return 1 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, load settings once, and dispatch to the chosen command."""
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(env_file=args.env_file, config_file=args.config)
        return args.handler(settings, args)
    except ConfigError as error:
        # Config problems are user-fixable (bad .env, missing file), so show a
        # one-line message instead of a traceback.
        print(f"Configuration error: {error}")
        return 2


# --- small private helpers ------------------------------------------------------


def _path_status(name: str, value: Path | None, explicit: bool) -> str:
    """One word or phrase per path: set by the user but absent is the only problem."""
    if value is None:
        return "not set"  # allowed: only the steps that need it will complain
    if value.exists():
        return "ok"
    if name in CREATED_OUTPUTS:
        return "will be created"
    if explicit:
        return "MISSING"
    return PRODUCED_INPUTS.get(name, "MISSING")  # only defaulted paths reach here


def _flatten(params: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Turn nested sections into ("training.mlp.dropout", 0.3) pairs, in file order."""
    pairs: list[tuple[str, Any]] = []
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            pairs.extend(_flatten(value, prefix=f"{name}."))
        else:
            pairs.append((name, value))
    return pairs
