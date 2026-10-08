"""Final model -> `<run>/model/` and back (plan U16, T5.2; PRD FR-6, §6.9, §7.4; D-006, D-066; EC-18).

`save_model` refits the chosen model on every fish of its training set (preprocessing included) and writes the
model folder. Its expected performance is the cross-validation estimate in the report, never a score on these
fish (PRD §6.9). `reference_predictions.csv` holds the final model's probabilities for its own training fish: a
reloaded model must reproduce it (AC-9, D-006). `load_model` reads the folder back; a different scikit-learn or
torch version is a warning, not an error.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from dcs.config import ConfigError
from dcs.models import make_model
from dcs.preprocess import Preprocess, fit_preprocess
from dcs.trainset import TrainSet

MODEL_DIR = "model"
PREPROCESS_FILE = "preprocess.json"
CLASSES_FILE = "classes.json"
INFO_FILE = "model_info.json"
SKLEARN_FILE = "sklearn.joblib"
MLP_FILE = "mlp.pt"
REFERENCE_FILE = "reference_predictions.csv"
CHECKED_PACKAGES = ("scikit-learn", "numpy", "pandas", "torch")


@dataclass(frozen=True)
class Loaded:
    info: dict[str, Any]
    preprocess: Preprocess
    classes: tuple[str, ...]
    model: Any
    version_warnings: tuple[str, ...]

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        """One row per fish of `table`: `video_id`, `predicted`, `p:<class>`; uses only the saved preprocessing."""
        missing = [name for name in self.preprocess.features if name not in table.columns]
        if missing:
            raise ConfigError(
                f"The input lacks {len(missing)} feature column(s) the model needs: {', '.join(missing)}. "
                "Build it with `python -m dcs featurize` (same dcs version as the model)."
            )
        dates = table["date"].to_numpy(dtype=object) if "date" in table.columns else np.full(len(table), None, dtype=object)
        proba = self.model.predict_proba(self.preprocess.transform(table), dates)
        return _frame(table["video_id"] if "video_id" in table.columns else pd.Series(range(len(table))), self.classes, proba)


def save_model(run: Path, ts: TrainSet, name: str, training: Mapping[str, Any], device: str, verdict: str) -> dict[str, Any]:
    """Refit `name` on all of `ts` and write `run/model/`; returns the model_info written there."""
    folder = Path(run) / MODEL_DIR
    folder.mkdir()
    preprocess = fit_preprocess(ts.X, ts.kinds)
    X, dates = preprocess.transform(ts.X), ts.groups.to_numpy(dtype=object)
    model = make_model(name, training["seed"], training["mlp"], device).fit(X, ts.y.to_numpy(dtype=object), dates)
    if name == "mlp":
        model.save(folder / MLP_FILE)
    else:
        joblib.dump(model, folder / SKLEARN_FILE)  # majority, logreg, forest, boosting, or date-only
    classes = [str(label) for label in model.classes_]
    info = {
        "model": name,
        "stage": ts.stage,
        "verdict": verdict,
        "fish": len(ts.y),
        "features": len(preprocess.features),
        "seed": training["seed"],
        "device": device,
        "versions": {"python": platform.python_version(), **{p: package_version(p) for p in CHECKED_PACKAGES}},
    }
    _write_json(folder / PREPROCESS_FILE, preprocess.as_json())
    _write_json(folder / CLASSES_FILE, classes)
    _write_json(folder / INFO_FILE, info)
    _frame(ts.ids, classes, model.predict_proba(X, dates)).to_csv(folder / REFERENCE_FILE, index=False)
    return info


def load_model(run: Path, device: str = "cpu") -> Loaded:
    """The model saved in `run/model/`, with a warning per package whose version differs from the saved one."""
    folder = Path(run) / MODEL_DIR
    if not (folder / INFO_FILE).is_file():
        raise ConfigError(f"No saved model in {folder}. Point --model at a run folder written by `python -m dcs train`.")
    info = json.loads((folder / INFO_FILE).read_text(encoding="utf-8"))
    if info["model"] == "mlp":
        from dcs.mlp import MLP  # torch only for an MLP

        model: Any = MLP.load(folder / MLP_FILE, device)
    else:
        model = joblib.load(folder / SKLEARN_FILE)
    warnings = tuple(
        f"saved with {package} {saved}, running {now}: predictions may differ slightly"
        for package, saved in info["versions"].items()
        if package in CHECKED_PACKAGES and saved is not None and (now := package_version(package)) != saved
    )
    return Loaded(
        info=info,
        preprocess=Preprocess.from_json(json.loads((folder / PREPROCESS_FILE).read_text(encoding="utf-8"))),
        classes=tuple(json.loads((folder / CLASSES_FILE).read_text(encoding="utf-8"))),
        model=model,
        version_warnings=warnings,
    )


def _frame(ids: pd.Series, classes: Any, proba: np.ndarray) -> pd.DataFrame:
    names = np.asarray(list(classes), dtype=object)
    out = pd.DataFrame({"video_id": np.asarray(ids), "predicted": names[proba.argmax(axis=1)]})
    return pd.concat([out, pd.DataFrame(proba, columns=[f"p:{c}" for c in names])], axis=1)


def package_version(package: str) -> str | None:
    """Installed version, read from the package metadata (no import, so torch is never loaded for this)."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

