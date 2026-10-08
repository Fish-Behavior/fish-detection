"""`dcs train`: training table -> one run folder (plan U12, T2.8; PRD §7.2, §7.4, US-1, US-2, NFR-1).

For each stage: training set -> shared folds -> evaluation of every model. Then the run folder
`<DCS_OUTPUT_DIR>/training/<run_id>/` gets run_info.json, config_used.yaml, audit.md, folds.csv, metrics.csv,
predictions.csv, report.md and confusion_<stage>.png. Nothing is written when no stage can be built. The final
model (`model/`) is saved from U16 on.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from functools import partial
from importlib import metadata
from itertools import count
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dcs.audit import build_audit, render
from dcs.config import ConfigError, Settings
from dcs.evaluate import REFERENCE, evaluate
from dcs.folds import make_folds
from dcs.models import BASELINES
from dcs.report import StageResult, render_report, save_confusion
from dcs.trainset import STAGES, build_trainset, load_table

RUNS_FOLDER = "training"
REPO = Path(__file__).resolve().parents[2]
PACKAGES = ("scikit-learn", "numpy", "pandas", "torch")


# Progress goes out line by line even into a file or `| tee train.log` (Python buffers stdout that is not a terminal).
PROGRESS = partial(print, flush=True)


def run_training(
    settings: Settings, training: Mapping[str, Any], command: str, log: Callable[[str], None] = PROGRESS
) -> Path:
    """Evaluate every stage and model in `training` on the table in the settings; return the new run folder."""
    started, clock = datetime.now(timezone.utc), time.monotonic()
    table_path = settings.require("table")
    table, described = load_table(table_path)
    models = list(dict.fromkeys([*REFERENCE, *(m for m in training["models"] if m in BASELINES)]))
    skipped = [m for m in training["models"] if m not in BASELINES]
    for name in skipped:
        log(f"{name}: skipped, not built yet (the MLP arrives in U13)")

    results: dict[str, StageResult] = {}
    notes = []
    for stage in STAGES if training["stage"] == "both" else (training["stage"],):
        try:
            ts = build_trainset(table, described, training, stage)
        except ConfigError as error:
            notes.append(f"{stage} stage cannot be built: {error}")
            log(notes[-1])
            continue
        folds = make_folds(ts.y, ts.groups, ts.ids, training)
        log(f"{stage}: {len(ts.y)} fish, {len(ts.classes)} classes, {len(ts.features)} features, models {', '.join(models)}")
        results[stage] = StageResult(ts, folds, evaluate(ts, folds, models, training, log))
    if not results:
        raise ConfigError(" ".join(notes))

    commit, dirty = _git()
    run = _new_folder(settings.paths.output_dir / RUNS_FOLDER, f"{started:%Y%m%dT%H%M%SZ}-{commit}")
    info = {
        "run_id": run.name,
        "git_commit": commit,
        "git_dirty": dirty,
        "command": command,
        "seed": training["seed"],
        "folds": training["folds"],
        "repeats": training["repeats"],
        "stages": list(results),
        "models": models,
        "skipped_models": skipped,
        "gold_source": described["gold_source"],
        "table": str(table_path),
        "versions": {"python": platform.python_version(), **{name: _version(name) for name in PACKAGES}, "cuda": None},
        "hardware": {"system": platform.system(), "machine": platform.machine(), "cpus": os.cpu_count()},
        "started_utc": started.isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - clock, 1),
    }
    audit = build_audit(table, described, training)
    (run / "audit.md").write_text(render(audit), encoding="utf-8")
    (run / "config_used.yaml").write_text(yaml.safe_dump({"training": _plain(training)}, sort_keys=False), encoding="utf-8")
    for name, frames in (
        ("folds.csv", {stage: r.folds.table for stage, r in results.items()}),
        ("metrics.csv", {stage: r.evaluation.metrics for stage, r in results.items()}),
        ("predictions.csv", {stage: r.evaluation.predictions for stage, r in results.items()}),
    ):
        pd.concat([frame.assign(stage=stage) for stage, frame in frames.items()]).pipe(_stage_first).to_csv(run / name, index=False)
    for stage, result in results.items():
        save_confusion(result, run / f"confusion_{stage}.png")
    (run / "report.md").write_text(render_report(info, results, audit, notes), encoding="utf-8")
    (run / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return run


def _new_folder(base: Path, name: str) -> Path:
    """`base/name`, or `name-2`, `name-3` ... when a run started in the same second."""
    base.mkdir(parents=True, exist_ok=True)
    for number in count(1):
        path = base / (name if number == 1 else f"{name}-{number}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise AssertionError("unreachable")


def _git() -> tuple[str, bool]:
    """Short commit of the checkout and whether tracked files differ from it; `nogit` outside a git checkout."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True, check=False
        )
    except OSError:
        return "nogit", False
    if head.returncode != 0:
        return "nogit", False
    return head.stdout.strip(), bool(status.stdout.strip())


def _version(package: str) -> str | None:
    """Installed version without importing (torch is imported only by the MLP)."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _plain(value: Any) -> Any:
    """Read-only settings (mappings, tuples) as plain YAML-ready dicts and lists."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _stage_first(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["stage", *(column for column in frame.columns if column != "stage")]]
