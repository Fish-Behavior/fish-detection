"""Command-line entry point: `python -m dcs <command> [options]`.

Each step registers one sub-command here in its own unit (synth, featurize,
audit, train, predict: classifier plan §3.1). So far: `check-config`,
which shows what `dcs` will read so a new `.env` or override YAML can be
verified before any data is touched; `synth`, which writes a synthetic
gold dataset for trying the other commands without real data; and
`featurize`, which turns the gold set into the training table; `audit`,
which reports what that table holds before anything is trained; `train`,
which evaluates every model on shared folds and writes a run folder with the report and the saved model; and
`predict`, which scores a table with a saved model; `ask` and `serve-chat`, the research chat in the terminal
and as a loopback JSON API.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from dcs import __version__
from dcs.audit import AUDIT_FILE, build_audit, render
from dcs.config import PATH_VARIABLES, ConfigError, Settings, load_settings
from dcs.config_rules import STAGES, VALUE_RULES
from dcs.featurize import featurize, write_outputs
from dcs.artifact import MODEL_DIR, REFERENCE_FILE, load_model
from dcs.gold import ReadOptions, read_gold, read_videos
from dcs.synthetic import SynthConfig, make_gold_dataset
from dcs.train import run_training
from dcs.trainset import load_table

# Folders dcs creates itself, so their absence is never a problem.
CREATED_OUTPUTS = ("output_dir",)
# Default locations that exist only after an earlier step has run.
PRODUCED_INPUTS = {
    "accepted_dir": "not found yet (written by prepds export-index)",
    "table": "not found yet (written by dcs featurize)",
}
LABEL_WIDTH = 19
VIDEOS_TABLE = "videos_table.parquet"  # featurize --videos default, never the training table
MATCH_TOLERANCE = 1e-9  # predict vs reference_predictions.csv (CSV round trip)
DEVICES = ("auto", "cpu", "cuda")  # PRD §7.2
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")  # the chat API has no authentication (as prepds review)


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

    synth = commands.add_parser(
        "synth",
        help="write a synthetic gold dataset (accepted folder + workbook) with placeholder names, for trying dcs",
    )
    synth.add_argument("--out", required=True, help="new or empty folder to write into")
    synth.add_argument("--seed", type=int, default=0, help="random seed, 0 or more (default: 0)")
    synth.set_defaults(handler=run_synth, needs_settings=False)  # works even with a broken .env

    feat = commands.add_parser("featurize", help="gold set (+ workbook NTT) -> training_table.parquet and its schema json")
    feat.add_argument("--profile", help="calibration profile to keep when the set mixes several (EC-21)")
    feat.add_argument(
        "--videos", help="read per-video folders (<dir>/<video_id>/) without index or catalog, for predict (D-018)"
    )
    feat.add_argument("--out", help=f"table to write (default: DCS_TABLE; with --videos: <DCS_OUTPUT_DIR>/{VIDEOS_TABLE})")
    feat.set_defaults(handler=run_featurize)

    audit = commands.add_parser("audit", help="training table -> audit.md: counts, drops, confounds, G1 verdict (no training)")
    audit.set_defaults(handler=run_audit)

    train = commands.add_parser(
        "train", help="training table -> run folder: every model on shared folds, report.md, metrics, predictions"
    )
    train.add_argument("--stage", help=f"{', '.join(STAGES)} (default: training.stage)")
    train.add_argument("--models", help="comma-separated model names, e.g. logreg,random_forest (default: training.models)")
    train.add_argument("--seed", type=int, help="seed for folds, models and permutation (default: training.seed)")
    train.add_argument("--repeats", type=int, help="cross-validation repeats (default: training.repeats)")
    train.add_argument(
        "--device", choices=DEVICES, default="auto", help="where the MLP trains: auto (GPU if PyTorch sees one), cpu, cuda"
    )
    train.add_argument("--no-ablations", action="store_true", help="skip the compound-stage ablations (faster)")
    train.set_defaults(handler=run_train)

    predict = commands.add_parser("predict", help="saved model + table -> class probabilities per fish (CSV)")
    predict.add_argument("--model", required=True, help="run folder written by `dcs train` (holds model/)")
    predict.add_argument("--input", required=True, help="table from `dcs featurize` (.parquet), or the same columns as .csv")
    predict.add_argument("--out", help="CSV to write (default: <DCS_OUTPUT_DIR>/<input name>_predictions.csv)")
    predict.add_argument("--device", choices=DEVICES, default="cpu", help="for an MLP: cpu (default), cuda or auto")
    predict.set_defaults(handler=run_predict)

    ask = commands.add_parser("ask", help="research chat in the terminal: one question, or a session without one")
    ask.add_argument("question", nargs="?", help="the question (omit for an interactive session)")
    ask.add_argument("--run", help="run folder to talk about (default: the latest under <DCS_OUTPUT_DIR>/training)")
    ask.add_argument("--show-tools", action="store_true", help="print each tool call the model made")
    ask.set_defaults(handler=run_ask)

    serve = commands.add_parser("serve-chat", help="research chat as a JSON API on this machine, for a web frontend")
    serve.add_argument("--host", default="127.0.0.1", help="loopback address only (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8010, help="port (default 8010)")
    serve.add_argument("--run", help="run folder to talk about (default: the latest)")
    serve.set_defaults(handler=run_serve_chat)
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


def run_synth(settings: Settings | None, args: argparse.Namespace) -> int:
    """Write <out>/accepted/... and <out>/synthetic_db.xlsx; never into a file or a folder that has files."""
    try:
        out = Path(args.out).expanduser()
        config = SynthConfig(seed=args.seed)
    except (RuntimeError, ValueError) as error:  # unknown ~user, or an invalid config; nothing written yet
        raise ConfigError(str(error)) from None
    try:
        result = make_gold_dataset(out, config)
    except FileExistsError as error:  # --out is a file or a folder with files in it
        raise ConfigError(str(error)) from None
    except OSError as error:  # e.g. no permission to create --out
        raise ConfigError(f"Cannot write to {out}: {error.strerror or error}") from None
    print(f"Wrote a synthetic gold dataset ({len(result.truth)} fish, placeholder names only) to {out}")
    print(f"  accepted folder: {result.accepted_dir}   (DCS_ACCEPTED_DIR)")
    print(f"  workbook:        {result.workbook_path}   (DCS_DB_PATH)")
    return 0


def run_featurize(settings: Settings, args: argparse.Namespace) -> int:
    """Read the gold source chosen in the settings, featurize every kept fish, write the table and schema."""
    training = settings.training
    if args.videos:
        options = ReadOptions(args.profile, training["model_profile_marker"], training["fps_tolerance"])
        gold = read_videos(Path(args.videos).expanduser(), options)
        default = settings.paths.output_dir / VIDEOS_TABLE
    else:
        gold = read_gold(settings, profile=args.profile)
        default = settings.paths.table
    result = featurize(gold, settings.paths.db_path, training)
    table_path, json_path = write_outputs(result, Path(args.out).expanduser() if args.out else default)
    accepted = int(result.table["reviewed"].sum())
    print(
        f"Featurized {len(result.table)} fish ({accepted} Accepted, the rest UNREVIEWED; profile {gold.profile}), "
        f"{len(result.schema['dropped'])} dropped"
    )
    print(f"  table:  {table_path}")
    print(f"  schema: {json_path}")
    return 0


def run_audit(settings: Settings, args: argparse.Namespace) -> int:
    """Audit the training table and its schema; write audit.md to the output folder and print the verdicts."""
    table, described = load_table(settings.require("table"))
    result = build_audit(table, described, settings.training)
    path = settings.paths.output_dir / AUDIT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(result), encoding="utf-8")
    facts = result.facts
    print(f"Audited {facts['fish in the table']} fish ({facts['source']}): G1 stop rule {facts['G1 stop rule']}")
    for note in result.notes:
        print(f"  - {note}")
    print(f"  report: {path}")
    return 0


def run_train(settings: Settings, args: argparse.Namespace) -> int:
    """Evaluate every model on the training table and write one run folder; print where the report is."""
    training = _with_overrides(settings.training, args)
    run = run_training(settings, training, command=shlex.join(["python", "-m", "dcs", *args.argv]), device=args.device, ablations=not args.no_ablations)
    print(f"Run folder: {run}")
    print(f"  report: {run / 'report.md'}")
    return 0


def run_predict(settings: Settings, args: argparse.Namespace) -> int:
    """Score every fish of the input with the saved model; compare with the saved reference where fish overlap (AC-9)."""
    run, source = Path(args.model).expanduser(), Path(args.input).expanduser()
    if not source.is_file():
        raise ConfigError(f"--input {source} does not exist. Build it with `python -m dcs featurize`.")
    model = load_model(run, device=args.device)
    for warning in model.version_warnings:
        print(f"  warning: {warning}")
    table = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    predictions = model.predict(table)
    out = Path(args.out).expanduser() if args.out else settings.paths.output_dir / f"{source.stem}_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    info = model.info
    print(f"Predicted {len(predictions)} fish with {info['model']} (cross-validation verdict: {info['verdict']}): {out}")
    reference = pd.read_csv(run / MODEL_DIR / REFERENCE_FILE).set_index("video_id")
    shared = predictions.set_index("video_id").reindex(reference.index).dropna(subset=["predicted"])
    if not shared.empty:
        columns = [c for c in reference.columns if c.startswith("p:")]
        gap = float((shared[columns] - reference.loc[shared.index, columns]).abs().to_numpy().max())
        verdict = "matches" if gap <= MATCH_TOLERANCE else f"DIFFERS (largest gap {gap:.3g}) from"
        print(f"  {verdict} the saved reference predictions for {len(shared)} fish")
    return 0


def run_ask(settings: Settings, args: argparse.Namespace) -> int:
    """One answer, or an interactive session, from the research chat."""
    from dcs import chat
    from dcs.chat_tools import ResearchData

    chat.check_engine(settings.chat)
    data = ResearchData.from_settings(settings, run=Path(args.run).expanduser() if args.run else None)
    if args.question:
        chat.print_answer(chat.ask(data, args.question, chat=settings.chat), args.show_tools)
    else:
        chat.session(data, settings.chat, args.show_tools)
    return 0


def run_serve_chat(settings: Settings, args: argparse.Namespace) -> int:
    """Serve the chat API on a loopback address (it has no authentication)."""
    if args.host not in LOOPBACK_HOSTS:
        raise ConfigError(f"the chat API has no authentication: --host must be a loopback address {LOOPBACK_HOSTS}.")
    import uvicorn

    from dcs.chat import check_engine
    from dcs.chat_server import create_app
    from dcs.chat_tools import ResearchData

    check_engine(settings.chat)
    data = ResearchData.from_settings(settings, run=Path(args.run).expanduser() if args.run else None)
    print(f"Research chat API: http://{args.host}:{args.port}/api  (model {settings.chat['model']}, run {data.run.name if data.run else 'none'})")
    uvicorn.run(create_app(data, settings.chat), host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, load settings once (unless the command needs none), and dispatch to it."""
    args = build_parser().parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)  # recorded in run_info.json
    try:
        needs_settings = getattr(args, "needs_settings", True)
        settings = load_settings(env_file=args.env_file, config_file=args.config) if needs_settings else None
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


def _with_overrides(training: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """The training settings with the command-line values, each checked by its settings rule (D-023)."""
    rules = dict(VALUE_RULES)
    models = None if args.models is None else tuple(name.strip() for name in args.models.split(","))
    merged = dict(training)
    for key, value in (("stage", args.stage), ("models", models), ("seed", args.seed), ("repeats", args.repeats)):
        if value is None:
            continue
        rule = rules[f"training.{key}"]
        if not rule.check(value):
            raise ConfigError(f"--{key} must be {rule.requirement}, got {value!r}")
        merged[key] = value
    return merged


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
