"""Command-line entry point: `python -m fishbehavior <command> [options]`.

Each pipeline step registers one sub-command here in its own PR
(validate, track, label, ...). Available so far:

    check-config   show what the pipeline will read, to verify a new `.env`
    validate       clean the trial workbook and match every trial to its video(s)
    scene          per video: empty-beaker background, waterline and fish ROI (+ QA images)
"""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from fishbehavior import __version__
from fishbehavior.catalog import CatalogError, build_catalog, load_trials, select_subjects
from fishbehavior.config import PATH_VARIABLES, ConfigError, Settings, load_settings
from fishbehavior.roi import FLAG_DURATION, FLAG_FPS, FLAG_LOW_CONFIDENCE, run_scene

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def build_parser() -> argparse.ArgumentParser:
    """Define the global options and one sub-parser per command."""
    parser = argparse.ArgumentParser(
        prog="fishbehavior",
        description="Automated zebrafish behavior-state labeling pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    # Global options shared by every command: where settings come from.
    parser.add_argument("--env-file", help="path to a .env file (default: ./.env if it exists)")
    parser.add_argument("--config", help="YAML file overriding default settings (default: $FISH_CONFIG)")

    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    check = commands.add_parser("check-config", help="show the resolved paths and settings, then exit")
    check.set_defaults(handler=run_check_config)  # each command points to the function that runs it

    validate = commands.add_parser(
        "validate", help="clean the workbook, match trials to videos, write the catalog + report"
    )
    validate.add_argument(
        "--strict", action="store_true", help="exit with code 1 if the report lists any issue"
    )
    validate.set_defaults(handler=run_validate)

    scene = commands.add_parser(
        "scene", help="find each video's background, waterline and fish region; write QA images"
    )
    add_subject_options(scene)
    scene.set_defaults(handler=run_scene_command)

    return parser


def add_subject_options(parser: argparse.ArgumentParser) -> None:
    """--subjects / --force, shared by every per-subject step."""
    parser.add_argument(
        "--subjects", nargs="+", metavar="ID",
        help="only these subjects, e.g. 42 0043 F_0044 (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="redo subjects that already have results")


def run_check_config(settings: Settings, args: argparse.Namespace) -> int:
    """Print every path the pipeline would use and whether it exists.

    Returns 0 when all configured inputs exist, 1 otherwise, so it can be
    used as a quick "is my setup right?" test.
    """
    print(f"{'.env file':<19}: {settings.env_file or '(none found - using environment variables only)'}")
    problems = 0
    for name, env_name in PATH_VARIABLES.items():
        value = getattr(settings.paths, name)
        if value is None:
            status = "not set"  # allowed: only the steps that need it will complain
        elif value.exists():
            status = "ok"
        else:
            status = "MISSING"
            problems += 1
        print(f"{env_name:<19}: {value if value is not None else '-'}  [{status}]")
    # The output folder is created on demand by later steps, so it only needs a note.
    output_note = "exists" if settings.paths.output_dir.exists() else "will be created"
    print(f"{'FISH_OUTPUT_DIR':<19}: {settings.paths.output_dir}  [{output_note}]")
    print(f"{'FISH_WORKERS':<19}: {settings.workers}")
    return 1 if problems else 0


def run_validate(settings: Settings, args: argparse.Namespace) -> int:
    """Build the catalog (see catalog.py) and print a short summary.

    The full details are in the written validation_report.md. Returns 0, or 1
    when --strict is given and the report lists any issue.
    """
    try:
        result = build_catalog(settings)
    except CatalogError as error:
        print(f"Workbook error: {error}")  # e.g. a required column is missing
        return 2

    notes = result.notes
    print(f"Workbook   : {notes['trial_rows']} trial rows -> {notes['subjects']} subjects "
          f"({len(notes['split_list'])} split recordings joined)")
    print(f"NTT        : {notes['untracked_subjects']} subjects untracked (all 8 movement values missing)")
    statuses = ", ".join(f"{name} {count}" for name, count in sorted(notes["status_counts"].items()))
    print(f"Videos     : {notes['video_files']} files found; {statuses}")
    issue_count = sum(len(entries) for entries in result.issues.values())
    print(f"Issues     : {issue_count}")
    print(f"Written to : {result.output_dir}  (trials.csv, videos.csv, validation_report.md)")
    return 1 if args.strict and result.has_issues else 0


def run_scene_command(settings: Settings, args: argparse.Namespace) -> int:
    """Run scene setup for the matched subjects and print a summary.

    Returns 1 if any video could not be processed, else 0. Flags (low-confidence
    waterline, duration, fps) are warnings for a person to check, not failures.
    """
    trials = load_trials(settings)
    result = run_scene(settings, trials, select_subjects(trials, args.subjects), force=args.force)

    records = result.records
    failed = [r for r in records if "error" in r]
    cached = sum(1 for r in records if r.get("cached"))
    print(f"{'Videos':<25}: {len(records) - len(failed)} done ({cached} cached), {len(failed)} failed")
    if result.skipped:
        print(f"{'Skipped':<25}: {len(result.skipped)} subject(s) without a matched video: {', '.join(result.skipped)}")
    for record in failed:
        print(f"{'  failed':<25}: {record['file']}: {record['error']}")

    # Flags for the videos of this run only (video_check.csv has everyone).
    files = {r["file"] for r in records}
    check = result.check[result.check["file"].isin(files)]
    for title, flag in (("Low waterline confidence", FLAG_LOW_CONFIDENCE),
                        ("Duration flags", FLAG_DURATION), ("FPS mismatch", FLAG_FPS)):
        hits = check[check["flags"].str.split(";").map(lambda flags, f=flag: f in flags)]
        details = ", ".join(f"{row.file} ({row.duration_s:.0f}s)" if flag == FLAG_DURATION else row.file
                            for row in hits.itertuples())
        print(f"{title:<25}: {len(hits)}" + (f"  {details}" if details else ""))
    print(f"{'Written to':<25}: {result.scene_dir}  (<video>.json, _background.png, _qa.png, video_check.csv)")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, load settings once, and dispatch to the chosen command."""
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(env_file=args.env_file, config_file=args.config)
        _setup_logging(settings)
        return args.handler(settings, args)
    except ConfigError as error:
        # Config problems are user-fixable (bad .env, missing file), so show a
        # one-line message instead of a traceback. Steps raise ConfigError too,
        # e.g. via settings.require() when an input they need is missing.
        print(f"Configuration error: {error}")
        return 2


def _setup_logging(settings: Settings) -> None:
    """Configure console logging from the `run.log_level` setting."""
    level = str(settings.params.get("run", {}).get("log_level", "INFO")).upper()
    if level not in LOG_LEVELS:
        raise ConfigError(f"run.log_level must be one of {', '.join(LOG_LEVELS)}, got {level!r}.")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
