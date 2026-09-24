"""Command-line entry point: `python -m prepds <command> [options]`.

Each pipeline step registers one sub-command here in its own phase (catalog,
calibrate, run, export-index - PRD §9.3). This first version only has
`check-config`, which shows what the pipeline will read so a new `.env` can
be verified before any data is processed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from prepds import __version__
from prepds.catalog import (
    DEFAULT_DURATION_TOLERANCE_S,
    build_exceptions_report,
    dedupe_rows,
    find_duration_mismatches,
    load_workbook,
    match_videos,
    to_trials,
)
from prepds.config import PATH_VARIABLES, ConfigError, Settings, default_workers, load_settings
from prepds.models import MatchStatus, Trial

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
CATALOG_FILE = "trials_catalog.parquet"
RUN_REPORT_FILE = "run_report.json"

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define the global options and one sub-parser per command."""
    parser = argparse.ArgumentParser(
        prog="prepds",
        description="Preprocessing Dataset System: video + trial-metadata -> reviewed ethogram strip.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    # Global options shared by every command: where settings come from.
    parser.add_argument("--env-file", help="path to a .env file (default: ./.env if it exists)")
    parser.add_argument("--config", help="YAML file overriding default thresholds (default: $PDS_CONFIG)")

    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    check = commands.add_parser("check-config", help="show the resolved paths and settings, then exit")
    check.set_defaults(handler=run_check_config)  # each command points to the function that runs it

    catalog = commands.add_parser(
        "catalog",
        help="match trial rows to local videos; write trials_catalog.parquet + exceptions_report.json",
    )
    catalog.add_argument(
        "--duration-tolerance-s",
        type=float,
        default=DEFAULT_DURATION_TOLERANCE_S,
        help=f"FR-004 duration-mismatch tolerance in seconds (default: {DEFAULT_DURATION_TOLERANCE_S})",
    )
    catalog.set_defaults(handler=run_catalog)

    run = commands.add_parser(
        "run",
        help="process every matched, not-yet-processed video: track -> label -> strip -> export (resumable)",
        description=(
            "Process the videos in the catalog, one video per worker process. Videos already processed, edited "
            "or accepted are skipped unless --force. Throughput: tracking is the cost (~40 s per 20-minute "
            f"video on one core), so wall time is roughly videos x 40 s / workers; the default is "
            f"max(1, cpu_count - 2) = {default_workers()} here (PDS_WORKERS or --workers overrides)."
        ),
    )
    run.add_argument("--workers", type=_positive_int, help="worker processes (default: PDS_WORKERS or cpu_count - 2)")
    run.add_argument("--dry-run", action="store_true", help="list the videos that would be processed, then exit")
    run.add_argument("--force", action="store_true", help="regenerate everything, INCLUDING edited/accepted work")
    run.add_argument("--limit", type=_positive_int, help="process at most this many pending videos")
    run.add_argument(
        "--tracker",
        help="model:<run dir> to track with a fine-tuned model (GPU, single process, ~5x slower than classical per "
        "video but it does not lose the fish); default is the classical background-subtraction tracker",
    )
    run.add_argument("--stride", type=_positive_int, default=5, help="model tracker: run the model every Nth frame (default 5)")
    run.add_argument("--device", default=None, help="model tracker: cuda or cpu (default: cuda if available)")
    run.set_defaults(handler=run_run)

    export_index = commands.add_parser(
        "export-index", help="rebuild accepted_index.parquet from the ACCEPTED videos in PDS_ACCEPTED_DIR"
    )
    export_index.set_defaults(handler=run_export_index)

    review = commands.add_parser("review", help="start the review web app (local, loopback only)")
    review.add_argument("--host", default="127.0.0.1", help="loopback address to bind (default: 127.0.0.1)")
    review.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    review.set_defaults(handler=run_review)

    annotate = commands.add_parser("annotate", help="start the frame-labeling app for the Phase 15 tracker fine-tuning set")
    annotate.add_argument("--host", default="127.0.0.1", help="loopback address to bind (default: 127.0.0.1)")
    annotate.add_argument("--port", type=int, default=8001, help="port (default: 8001)")
    annotate.set_defaults(handler=run_annotate)

    return parser


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _load_catalog_trials(settings: Settings) -> list[Trial] | None:
    path = settings.paths.output_dir / CATALOG_FILE
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    frame = frame.astype(object).where(frame.notna(), None)
    return [Trial.from_dict(row) for row in frame.to_dict("records")]


def run_run(settings: Settings, args: argparse.Namespace) -> int:
    """T080: catalog -> per-video pipeline, parallel and resumable. Exit 1 if any video failed."""
    from prepds import __version__ as pipeline_version
    from prepds.calibration.profile import labeling_thresholds
    from prepds.pipeline import Outcome, RunConfig, dedupe_trials, is_pending, run_batch, video_id

    trials = _load_catalog_trials(settings)
    if trials is None:
        raise ConfigError(f"no catalog at {settings.paths.output_dir / CATALOG_FILE}; run `prepds catalog` first.")
    matched, duplicates = dedupe_trials(
        [t for t in trials if t.match_status is MatchStatus.MATCHED and t.video_path is not None]
    )
    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate video id(s) skipped (only the first trial per id is processed): "
              + ", ".join(sorted({video_id(t) for t in duplicates})))
    config = RunConfig(
        out_dir=settings.paths.output_dir / "processed",
        thresholds=labeling_thresholds(settings.params),
        pipeline_version=pipeline_version,
        calibration_profile_version=str(settings.params["calibration_profile"]["version"]),
        processed_at=dt.datetime.now(dt.timezone.utc),
        force=args.force,
        undetermined_color_hex=str(settings.params.get("rendering", {}).get("undetermined_color_hex", "#FF00FF")),
    )
    pending = [t for t in matched if is_pending(t, config)]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{'matched videos':<22}: {len(matched)}")
    print(f"{'pending':<22}: {len(pending)}")
    if not pending:
        print("Nothing to do: every matched video is already processed (use --force to regenerate).")
        return 0
    if args.dry_run:
        for trial in pending:
            print(f"  would process {video_id(trial)}  {trial.video_path}")
        return 0

    tracker = None
    if args.tracker:
        kind, _, target = args.tracker.partition(":")
        if kind != "model" or not target:
            raise ConfigError(f"--tracker must look like model:<run dir>, got {args.tracker!r}.")
        if not (Path(target) / "model.pt").is_file():
            raise ConfigError(f"no fine-tuned model at {target} (expected model.pt and run.json).")
        import torch

        from prepds.model_tracker import ModelTracker

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tracker = ModelTracker(Path(target), device=device, stride=args.stride)
        print(f"tracker: {tracker.name} on {device} (one process; --workers ignored)")
    workers = 1 if tracker else min(args.workers or settings.workers, len(pending))
    started = time.perf_counter()
    done = 0

    def progress(result) -> None:
        nonlocal done
        done += 1
        print(f"[{done}/{len(pending)}] {result.video_id}: {result.outcome.value} {result.message}".rstrip(), flush=True)

    results = run_batch(pending, config, workers=workers, on_result=progress, tracker=tracker)
    wall_s = time.perf_counter() - started

    counts = {outcome.value: sum(1 for r in results if r.outcome is outcome) for outcome in Outcome}
    report = {
        "processed_at": config.processed_at.isoformat(),
        "pipeline_version": pipeline_version,
        "calibration_profile_version": config.calibration_profile_version,
        "workers": workers,
        "tracker": tracker.name if tracker else "classical",
        "wall_time_s": round(wall_s, 1),
        "counts": counts,
        "videos": [
            {"video_id": r.video_id, "outcome": r.outcome.value, "message": r.message, "seconds": round(r.seconds, 1)}
            for r in results
        ],
    }
    report_path = settings.paths.output_dir / RUN_REPORT_FILE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'processed':<22}: {counts['processed']}")
    print(f"{'skipped':<22}: {counts['skipped']}")
    print(f"{'failed':<22}: {counts['failed']}")
    print(f"{'wall time':<22}: {wall_s:.1f} s with {workers} worker(s)")
    print(f"{'wrote':<22}: {report_path}")
    for r in results:
        if r.outcome is Outcome.FAILED:
            print(f"  FAILED {r.video_id}: {r.message}")
    return 1 if counts["failed"] else 0


def run_export_index(settings: Settings, args: argparse.Namespace) -> int:
    """T081: rebuild accepted_index.parquet from the accepted videos on disk."""
    from prepds.review_store import INDEX_FILE, rebuild_index

    trials = _load_catalog_trials(settings)
    if trials is None:
        print("WARNING: no catalog found; workbook fields (strain, age, date, exposure) are kept from the existing "
              "index only. Run `prepds catalog` first to refresh them.")
        trials = []
    count = rebuild_index(settings.paths.accepted_dir, trials=trials)
    print(f"{'accepted videos':<22}: {count}")
    print(f"{'wrote':<22}: {settings.paths.accepted_dir / INDEX_FILE}")
    return 0


def run_review(settings: Settings, args: argparse.Namespace) -> int:
    """Serve the review UI on a loopback address (it has no authentication)."""
    if args.host not in LOOPBACK_HOSTS:
        raise ConfigError(f"the review app has no authentication: --host must be a loopback address {LOOPBACK_HOSTS}.")
    import uvicorn

    from prepds.webapp.app import create_app

    processed = settings.paths.output_dir / "processed"
    app = create_app(processed, settings.paths.accepted_dir, video_dir=settings.paths.video_dir)
    print(f"Review app: http://{args.host}:{args.port}/  (videos in {processed})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def run_annotate(settings: Settings, args: argparse.Namespace) -> int:
    """Serve the Phase 15 frame-labeling app on a loopback address (no authentication)."""
    if args.host not in LOOPBACK_HOSTS:
        raise ConfigError(f"the labeling app has no authentication: --host must be a loopback address {LOOPBACK_HOSTS}.")
    work_dir = settings.paths.output_dir / "phase15"
    if not (work_dir / "sample.json").is_file():
        raise ConfigError(f"no sampled frames in {work_dir}; run `python scripts/phase15_prepare.py` first.")
    import uvicorn

    from prepds.annotation.app import create_annotation_app

    print(f"Labeling app: http://{args.host}:{args.port}/  (frames in {work_dir})")
    uvicorn.run(create_annotation_app(work_dir), host=args.host, port=args.port, log_level="warning")
    return 0


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
    # The output/accepted folders are created on demand by later steps.
    for label, path in (
        ("PDS_OUTPUT_DIR", settings.paths.output_dir),
        ("PDS_ACCEPTED_DIR", settings.paths.accepted_dir),
    ):
        note = "exists" if path.exists() else "will be created"
        print(f"{label:<19}: {path}  [{note}]")
    print(f"{'PDS_WORKERS':<19}: {settings.workers}")
    return 1 if problems else 0


def run_catalog(settings: Settings, args: argparse.Namespace) -> int:
    """FR-001..004: match trial rows to local videos, write catalog + exceptions report.

    Never halts on a single bad row/video (FR-004) - every mismatch category
    lands in exceptions_report.json instead. Returns 0 whenever the catalog
    itself ran to completion; a non-empty exceptions report is expected
    output, not a CLI failure (see docs/progress.md §0 - ~15 of 352 logical
    trials are known to have no locally synced video yet).
    """
    video_dir = settings.require("video_dir")
    db_path = settings.require("db_path")

    LOGGER.info("Loading workbook from %s", db_path)
    raw = load_workbook(db_path)
    deduped, duplicate_groups = dedupe_rows(raw)
    trials = to_trials(deduped)

    LOGGER.info("Matching %d trial rows against %s", len(trials), video_dir)
    matched, unmatched_videos, ambiguous_matches = match_videos(trials, video_dir)

    LOGGER.info("Probing matched videos for FR-004 duration mismatches")
    duration_mismatches, corrupt_videos = find_duration_mismatches(
        matched, tolerance_s=args.duration_tolerance_s
    )

    report = build_exceptions_report(
        trials=matched,
        duplicate_groups=duplicate_groups,
        unmatched_videos=unmatched_videos,
        duration_mismatches=duration_mismatches,
        corrupt_videos=corrupt_videos,
        ambiguous_matches=ambiguous_matches,
    )

    output_dir = settings.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_df = pd.DataFrame([trial.to_dict() for trial in matched])
    catalog_path = output_dir / "trials_catalog.parquet"
    catalog_df.to_parquet(catalog_path, index=False)

    exceptions_path = output_dir / "exceptions_report.json"
    exceptions_path.write_text(json.dumps(report.to_dict(), indent=2))

    matched_count = sum(1 for trial in matched if trial.match_status.value == "matched")
    print(f"{'trials (deduped)':<22}: {len(matched)}")
    print(f"{'matched to a video':<22}: {matched_count}")
    print(f"{'unmatched trials':<22}: {len(report.unmatched_trials)}")
    print(f"{'unmatched videos':<22}: {len(report.unmatched_videos)}")
    print(f"{'duplicate trial rows':<22}: {len(report.duplicate_trials)}")
    print(f"{'duration mismatches':<22}: {len(report.duration_mismatches)}")
    print(f"{'corrupt videos':<22}: {len(report.corrupt_videos)}")
    print(f"{'ambiguous matches':<22}: {len(report.ambiguous_matches)}")
    print(f"{'wrote':<22}: {catalog_path}")
    print(f"{'wrote':<22}: {exceptions_path}")
    return 0


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
