"""Command-line entry point: `python -m fishbehavior <command> [options]`.

Each pipeline step registers one sub-command here in its own PR
(validate, track, label, ...). Available so far:

    check-config   show what the pipeline will read, to verify a new `.env`
    validate       clean the trial workbook and match every trial to its video(s)
    scene          per video: empty-beaker background, waterline and fish ROI (+ QA images)
    scene-review   browser page to check / correct waterlines and ROIs (writes overrides.yaml)
    track          per subject: fish position, size and tilt in every frame (parts joined)
    features       per subject: movement features per frame and per time bin, plus endpoints
    features-qa    per subject: which bin features are stable (frame rate / smoothing checks)
    label          per subject: ethogram state of every time bin, merged into segments
    reference      digitize the reference ethogram figures into approximate timelines (for tuning)
"""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from fishbehavior import __version__
from fishbehavior.catalog import CatalogError, build_catalog, load_trials, select_subjects
from fishbehavior.config import PATH_VARIABLES, ConfigError, Settings, load_settings
from fishbehavior.features import run_features, run_features_qa
from fishbehavior.labeling import LABELS, STATES, run_labeling
from fishbehavior.reference import MAPPING_FILE, SLIDE18_FILE, run_reference
from fishbehavior.review import IMAGE_KEYS, REVIEW_FILE, collect_items, make_server, render_page, serve_review
from fishbehavior.roi import FLAG_DURATION, FLAG_FPS, FLAG_LOW_CONFIDENCE, load_overrides, run_scene
from fishbehavior.tracking import run_tracking

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

    review = commands.add_parser(
        "scene-review", help="open a page to check and correct each video's waterline and ROI"
    )
    review.add_argument("--subjects", nargs="+", metavar="ID", help="only these subjects (default: all)")
    review.add_argument("--port", type=int, default=8765, help="local port for the page (default: 8765)")
    review.add_argument(
        "--no-serve", action="store_true",
        help="only write scene/review.html (e.g. on Colab); save with its Download button",
    )
    review.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    review.set_defaults(handler=run_scene_review)

    track = commands.add_parser(
        "track", help="find the fish in every frame (position, size, tilt); parts of a subject joined"
    )
    add_subject_options(track)
    track.set_defaults(handler=run_track)

    features = commands.add_parser(
        "features", help="speed, turning, depth and posture per frame and per time bin; per-subject endpoints"
    )
    add_subject_options(features)
    features.set_defaults(handler=run_features_command)

    features_qa = commands.add_parser(
        "features-qa", help="check which bin features stay the same at half the frame rate and double smoothing"
    )
    features_qa.add_argument("--subjects", nargs="+", metavar="ID", help="only these subjects (default: all)")
    features_qa.set_defaults(handler=run_features_qa_command)

    label = commands.add_parser(
        "label", help="give every time bin one of the five behavior states and merge them into segments"
    )
    add_subject_options(label)
    label.set_defaults(handler=run_label_command)

    reference = commands.add_parser(
        "reference", help="digitize the reference ethogram figures into per-subject timelines (for tuning)"
    )
    reference.add_argument("--force", action="store_true", help="re-extract the figures and redo the digitizing")
    reference.set_defaults(handler=run_reference_command)

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


def run_scene_review(settings: Settings, args: argparse.Namespace) -> int:
    """Bring scene results up to date, open the review page, then apply the saved overrides."""
    trials = load_trials(settings)
    selected = select_subjects(trials, args.subjects)
    before = run_scene(settings, trials, selected)  # cached videos are instant; new ones are analysed
    scene_dir = before.scene_dir
    items, missing = collect_items(scene_dir, selected)
    if missing:
        print(f"No scene result (failed) for {len(missing)} video(s): {', '.join(missing)}")
    if not items:
        print("Nothing to review.")
        return 1 if missing else 0

    page = render_page(items, load_overrides(scene_dir), settings.params["scene"],
                       serve=not args.no_serve, scene_dir=scene_dir)
    if args.no_serve:
        path = scene_dir / REVIEW_FILE
        path.write_text(page, encoding="utf-8")
        print(f"Review page written to {path}. Open it in a browser, then put the downloaded "
              f"overrides.yaml in {scene_dir} and run `python -m fishbehavior scene`.")
        return 0

    sizes = {item["file"]: (item["width"], item["height"]) for item in items}
    images = {item[key] for item in items for key in IMAGE_KEYS}
    try:
        server = make_server(scene_dir, page, sizes, args.port, images)
    except OSError as error:
        raise ConfigError(f"cannot start the review page on port {args.port} ({error}); try --port 8766") from None
    serve_review(server, open_browser=not args.no_browser)

    # Apply what was saved: only videos whose waterline/ROI changed are recomputed.
    after = run_scene(settings, trials, selected)
    checked = sum(bool(r.get("checked")) for r in after.records)
    overridden = sum(r.get("method") == "override" for r in after.records)
    print(f"Applied    : {checked} of {len(after.records)} videos checked, {overridden} with corrections")
    print(f"Written to : {scene_dir}  (overrides.yaml, <video>.json, video_check.csv)")
    return 0


def run_track(settings: Settings, args: argparse.Namespace) -> int:
    """Track the matched subjects and print one line per subject plus totals.

    Returns 1 if any subject could not be tracked, else 0. A missing scene result is
    computed on the way (same as running `scene` first).
    """
    trials = load_trials(settings)
    result = run_tracking(settings, trials, select_subjects(trials, args.subjects), force=args.force)

    records = result.records
    failed = [r for r in records if "error" in r]
    cached = sum(1 for r in records if r.get("cached"))
    print(f"{'Subjects':<11}: {len(records) - len(failed)} done ({cached} cached), {len(failed)} failed")
    if result.skipped:
        print(f"{'Skipped':<11}: {len(result.skipped)} subject(s) without a matched video: {', '.join(result.skipped)}")
    for record in failed:
        print(f"{'  failed':<11}: {record['subject_id']}: {record['error']}")
    for record in records:
        if "error" not in record:
            s = record["summary"]
            print(f"  {s['subject_id']}: detected {s['detected_pct']:.1f}%, interpolated {s['interpolated_pct']:.1f}%, "
                  f"untracked {s['untracked_s']:.1f} s, body length {s['median_body_length_px']} px")
    print(f"{'Written to':<11}: {result.tracks_dir}  (<subject>.csv.gz, _track_qa.png, summary.csv)")
    return 1 if failed else 0


def run_features_command(settings: Settings, args: argparse.Namespace) -> int:
    """Compute features for the matched subjects; returns 1 if any subject failed (e.g. no track)."""
    trials = load_trials(settings)
    result = run_features(settings, trials, select_subjects(trials, args.subjects), force=args.force)

    records = result.records
    failed = [r for r in records if "error" in r]
    cached = sum(1 for r in records if r.get("cached"))
    print(f"{'Subjects':<11}: {len(records) - len(failed)} done ({cached} cached), {len(failed)} failed")
    if result.skipped:
        print(f"{'Skipped':<11}: {len(result.skipped)} subject(s) without a matched video: {', '.join(result.skipped)}")
    for record in failed:
        print(f"{'  failed':<11}: {record['subject_id']}: {record['error']}")
    print(f"{'Written to':<11}: {result.features_dir}  (<subject>_frames.csv.gz, _bins.csv.gz, endpoints.csv)")
    return 1 if failed else 0


def run_features_qa_command(settings: Settings, args: argparse.Namespace) -> int:
    """Run the stability checks and print, per feature, how many subjects pass and the median numbers."""
    trials = load_trials(settings)
    result = run_features_qa(settings, select_subjects(trials, args.subjects))
    failed = [r for r in result.records if "error" in r]
    for record in failed:
        print(f"  failed: {record['subject_id']}: {record['error']}")
    table = result.table
    if table.empty:
        print("Nothing checked.")
        return 1 if failed else 0

    n = table["subject_id"].nunique()
    print(f"{'feature':<32} {'stable':>9}  {'r half-rate':>11}  {'change half-rate':>16}  {'change smooth x2':>16}")
    for feature, rows in table.groupby("feature", sort=False):
        print(f"{feature:<32} {int(rows['stable'].sum()):>4}/{n:<4}  {rows['r_half_rate'].median():>11.2f}  "
              f"{rows['change_half_rate'].median():>+16.0%}  {rows['change_smooth_2x'].median():>+16.0%}")
    per_subject = table.groupby("subject_id").first()
    print(f"Position jitter (median over subjects): {per_subject['jitter_bl'].median():.3f} BL; "
          f"body length variation: {per_subject['body_length_cv'].median():.0%}")
    print(f"Written to : {result.path}  (medians over {n} subject(s))")
    return 1 if failed else 0


def run_label_command(settings: Settings, args: argparse.Namespace) -> int:
    """Label the matched subjects and print the time share of each state; returns 1 if any failed."""
    trials = load_trials(settings)
    result = run_labeling(settings, trials, select_subjects(trials, args.subjects), force=args.force)

    if result.calibrated:
        print(f"{'Settings':<11}: calibrated values from {result.calibrated}")
    model = result.model
    print(f"{'Swim model':<11}: {model['method']}, {'fitted now' if result.model_refit else 'reused'} "
          f"({model['n_bins']} swim bins of {model['n_subjects']} subjects)")
    records = result.records
    failed = [r for r in records if "error" in r]
    cached = sum(1 for r in records if r.get("cached"))
    print(f"{'Subjects':<11}: {len(records) - len(failed)} done ({cached} cached), {len(failed)} failed")
    if result.skipped:
        print(f"{'Skipped':<11}: {len(result.skipped)} subject(s) without a matched video: {', '.join(result.skipped)}")
    for record in failed:
        print(f"{'  failed':<11}: {record['subject_id']}: {record['error']}")

    # Time share of each label over this run's subjects (summary.csv has one row per subject).
    done = {r["subject_id"] for r in records if "error" not in r}
    rows = result.summary[result.summary["subject_id"].isin(done)]
    if len(rows):
        total = rows["duration_s"].sum()
        shares = ", ".join(f"{name} {100 * rows[f'{name}_s'].sum() / total:.1f}%" for name in LABELS)
        print(f"{'Time share':<11}: {shares}")
    print(f"{'Written to':<11}: {result.labels_dir}  (<subject>_bins.csv, segments.csv, summary.csv, swim_model.json)")
    return 1 if failed else 0


def run_reference_command(settings: Settings, args: argparse.Namespace) -> int:
    """Digitize the reference figures; returns 1 while mapping.yaml still has to be filled in."""
    result = run_reference(settings, force=args.force)
    out = result.reference_dir
    pages = ", ".join(f"page {page}: {sum(p.page == page for p in result.panels)}"
                      for page in dict.fromkeys(p.page for p in result.panels))
    print(f"{'Panels':<11}: {pages}")
    if result.mapping_created:
        print(f"{'Mapping':<11}: template written to {out / MAPPING_FILE}. Fill in group and subject_ids "
              f"(top -> bottom) for every panel, or skip: true, then run `reference` again (see README).")
        return 1

    timelines = result.timelines
    unknown = 100 * (timelines["label"] == "unknown").mean()
    print(f"{'Digitized':<11}: {timelines['subject_id'].nunique()} subjects in "
          f"{timelines['group'].nunique()} groups{' (cached)' if result.cached else ''}, "
          f"{result.skipped_panels} panel(s) skipped, unknown color {unknown:.1f}% of seconds")
    means = result.group_means
    width = max(9, *(len(group) for group in means["group"]))  # align the columns under long group names
    print(f"{'Mean s':<{width + 2}}: " + "  ".join(f"{state:>15}" for state in STATES) + f"  {'no_data':>8}")
    for row in means.itertuples():
        print(f"  {row.group:<{width}}: " + "  ".join(f"{getattr(row, state):>15.0f}" for state in STATES)
              + f"  {row.no_data:>8.0f}")
    if result.slide18_created:
        print(f"{'Slide 18':<11}: optional template {out / SLIDE18_FILE}: fill in values read from the bar "
              f"charts to check the digitizer")
    elif result.slide18_check is not None:
        check = result.slide18_check
        print(f"{'Slide 18':<11}: {len(check)} values compared, median |digitized - slide| "
              f"{check['difference_s'].abs().median():.0f} s, largest {check['difference_s'].abs().max():.0f} s "
              f"(slide18_check.csv)")
    else:
        print(f"{'Slide 18':<11}: no values in {out / SLIDE18_FILE} yet (optional check of the digitizer)")
    print(f"{'Written to':<11}: {out}  (timelines.csv, group_means.csv, digitize_check.png)")
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
