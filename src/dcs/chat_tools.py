"""Research query tools for the chat (plan U19; D-071).

Deterministic functions a language model calls to answer researchers' questions; every number in an answer comes
from one of them, so it can be checked. They read the training table and its schema (behavior per fish), the
per-video folders (frames and segments over time) and a `dcs train` run folder (model results). Nothing is written.
Compound-versus-vehicle comparisons use the vehicle fish recorded on the same dates by default, because dates (and
cameras) differ between compounds (D-061). A tool given a bad argument returns `{"error": ...}` naming the valid
choices, so the model can correct itself.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from dcs import schema
from dcs.audit import build_audit
from dcs.config import Settings
from dcs.config_rules import SOURCE_PROCESSED
from dcs.featurize import BEHAVIOR_STATES, slug
from dcs.folds import SCHEME_A, SCHEME_B
from dcs.gold_rules import compound_label
from dcs.trainset import dose_label, load_table
from dcs.workbook import HAS_NTT

MIN_CONTROL = 3  # same-date vehicle fish needed before falling back to all vehicle fish
MAX_ROWS = 40  # longest list a tool returns (bouts, features), so answers stay readable
MAX_BINS = 2_000  # most time bins one timeline returns (bin_s is chosen by the model or an API caller)
SAME_DATES, ALL_VEHICLE = "same_dates", "all"
KINEMATICS = {
    "velocity_mean": "mean swimming speed, px/s (detected frames)",
    "velocity_median": "median swimming speed, px/s",
    "velocity_cv": "speed variability: standard deviation / mean of speed",
    "abs_acceleration_mean": "mean absolute acceleration, px/s²",
    "abs_angular_velocity_mean": "mean absolute turning rate, degrees/s",
    "meander_mean": "mean meander: heading change per distance travelled",
    "immobile_share": "share of detected frames in which the fish is immobile",
    "detected_share": "share of frames in which the tracker found the fish (tracking quality, not behavior)",
}
NTT = {
    "tdm_full": "novel tank test: total distance moved, whole tank (workbook)",
    "tdm_top": "novel tank test: distance moved in the top half",
    "tdm_bottom": "novel tank test: distance moved in the bottom half",
    "velocity_full": "novel tank test: mean velocity, whole tank",
    "velocity_top": "novel tank test: mean velocity in the top half (empty if never there)",
    "velocity_bottom": "novel tank test: mean velocity in the bottom half",
    "time_top_s": "novel tank test: seconds in the top half (anxiety-related: less = more anxious)",
    "time_bottom_s": "novel tank test: seconds in the bottom half",
    HAS_NTT: "1 when the fish has novel tank test values",
}


class ToolError(ValueError):
    """A bad argument; the message says what is allowed."""


@dataclass(frozen=True)
class ResearchData:
    table: pd.DataFrame  # the training table (one row per fish)
    described: Mapping[str, Any]  # its schema JSON
    training: Mapping[str, Any]  # training settings (vehicle name)
    gold_dir: Path | None  # holds <video_id>/frames.parquet and segments.csv
    run: Path | None  # a `dcs train` run folder

    @classmethod
    def from_settings(cls, settings: Settings, run: Path | None = None) -> ResearchData:
        """The table from DCS_TABLE, the video folders of the gold source, and `run` or the latest run folder."""
        table, described = load_table(settings.require("table"))
        if described["gold_source"] == SOURCE_PROCESSED:
            processed = settings.paths.processed_dir
            gold_dir = None if processed is None else processed / schema.PROCESSED_DIR_NAME
        else:
            gold_dir = settings.paths.accepted_dir
        if run is None:
            runs = sorted((settings.paths.output_dir / "training").glob("*/run_info.json"))
            run = runs[-1].parent if runs else None
        return cls(table, described, dict(settings.training), gold_dir, run)

    @property
    def compound(self) -> pd.Series:
        return self.table["compound"].map(compound_label)

    @property
    def dose(self) -> pd.Series:
        return self.table["concentration_mM"].map(dose_label)

    @property
    def vehicle(self) -> str:
        return compound_label(self.training["vehicle_compound"])


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...]
    run: Callable[..., dict[str, Any]]


TOOLS: dict[str, Tool] = {}


def tool(description: str, required: tuple[str, ...] = (), **properties: Any) -> Callable[..., Any]:
    def register(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        TOOLS[function.__name__] = Tool(function.__name__, description, properties, required, function)
        return function

    return register


def tool_schemas() -> list[dict[str, Any]]:
    """The tools in the OpenAI function-calling format (also understood by Ollama, llama.cpp and vLLM)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {"type": "object", "properties": t.properties, "required": list(t.required)},
            },
        }
        for t in TOOLS.values()
    ]


def call_tool(data: ResearchData, name: str, arguments: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Run one tool; bad names, arguments or JSON come back as `{"error": ...}` for the model to fix."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name!r}; tools: {', '.join(TOOLS)}"}
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        if not isinstance(args, dict):
            raise ToolError("arguments must be a JSON object")
        return _plain(TOOLS[name].run(data, **args))
    except (ToolError, json.JSONDecodeError, TypeError) as error:
        return {"error": str(error)}
    except Exception as error:  # noqa: BLE001 (a tool bug must not end `dcs ask` or turn into a 500)
        return {"error": f"{name} failed: {type(error).__name__}: {error}"}


# --- tools ---------------------------------------------------------------------------------------


@tool("List the compounds in the data: fish, Accepted fish, recording dates and doses per compound, which one is vehicle.")
def list_compounds(data: ResearchData) -> dict[str, Any]:
    rows = [
        {
            "compound": name,
            "fish": len(part),
            "accepted_fish": int(part["reviewed"].astype(bool).sum()),
            "dates": int(part["date"].nunique()),
            "doses": sorted(part["concentration_mM"].map(dose_label).unique()),
        }
        for name, part in data.table.groupby(data.compound)
    ]
    return {"source": _source(data), "vehicle": data.vehicle, "fish": len(data.table), "dates": int(data.table["date"].nunique()), "compounds": rows}


@tool(
    "Find behavior measures (features) and what they mean. Use it to map words like freezing, speed, anxiety, top "
    "of the tank, turning or erratic onto feature names before comparing.",
    search={"type": "string", "description": "word to look for in names and meanings; empty lists all"},
)
def find_features(data: ResearchData, search: str = "") -> dict[str, Any]:
    rows = [{"name": c["name"], "group": c.get("feature_group"), "meaning": meaning(c["name"])} for c in data.described["columns"] if c["role"] == "feature"]
    rows = [r for r in rows if search.lower() in (r["name"] + " " + r["meaning"]).lower()]
    return {"features": rows[:MAX_ROWS], "matches": len(rows)}


@tool(
    "Compare one behavior measure between a compound (optionally one dose) and vehicle: n, mean, median, spread, "
    "Hedges' g effect size and Mann-Whitney p. The control is vehicle fish recorded on the same dates unless "
    "control='all'.",
    required=("feature", "compound"),
    feature={"type": "string"},
    compound={"type": "string"},
    dose={"type": "string", "description": "optional dose as written, e.g. 0.1"},
    control={"type": "string", "enum": [SAME_DATES, ALL_VEHICLE]},
)
def compare_to_vehicle(data: ResearchData, feature: str, compound: str, dose: str | None = None, control: str = SAME_DATES) -> dict[str, Any]:
    _check_feature(data, feature)
    rows = _rows(data, compound, dose)
    control_rows, which, caveats = _control(data, rows, control)
    group, reference = data.table.loc[rows, feature].dropna(), data.table.loc[control_rows, feature].dropna()
    g, p = _effect(group, reference)
    return {
        "feature": feature,
        "meaning": meaning(feature),
        "compound": compound_label(compound),
        "dose": None if dose is None else dose_label(dose),
        "group": _describe(group),
        "control": {"which": which, **_describe(reference)},
        "hedges_g": g,
        "p_value": p,
        "direction": "higher" if g > 0.2 else "lower" if g < -0.2 else "about the same (|g| < 0.2)",
        "caveats": caveats + _caveats(data, feature),
    }


@tool(
    "Rank every behavior measure by how much a compound (or one dose) differs from vehicle: effect size, p and "
    "Benjamini-Hochberg q. Use it for 'what did compound X change?'.",
    required=("compound",),
    compound={"type": "string"},
    dose={"type": "string"},
    n={"type": "integer", "description": "how many to return (default 10)"},
    control={"type": "string", "enum": [SAME_DATES, ALL_VEHICLE]},
)
def top_differences(data: ResearchData, compound: str, dose: str | None = None, n: int = 10, control: str = SAME_DATES) -> dict[str, Any]:
    rows = _rows(data, compound, dose)
    control_rows, which, caveats = _control(data, rows, control)
    found = []
    for feature in _numeric_features(data):
        group, reference = data.table.loc[rows, feature].dropna(), data.table.loc[control_rows, feature].dropna()
        if len(group) >= 2 and len(reference) >= 2 and pd.concat([group, reference]).nunique() > 1:
            g, p = _effect(group, reference)
            found.append({"feature": feature, "meaning": meaning(feature), "hedges_g": g, "p_value": p,
                          "compound_mean": float(group.mean()), "control_mean": float(reference.mean())})  # fmt: skip
    if not found:
        raise ToolError("no feature has enough values on both sides to compare")
    q = stats.false_discovery_control([row["p_value"] for row in found])
    for row, value in zip(found, q):
        row["q_value"] = float(value)
    found.sort(key=lambda row: -abs(row["hedges_g"]))
    return {
        "compound": compound_label(compound),
        "dose": None if dose is None else dose_label(dose),
        "control": which,
        "tested": len(found),
        "features": found[: max(1, int(n))],
        "note": f"Hedges' g = compound minus control in pooled standard deviations; q = Benjamini-Hochberg false "
        f"discovery rate over the {len(found)} measures tested; q < 0.05 is the usual bar.",
        "caveats": caveats + _caveats(data, None),
    }


@tool(
    "One behavior measure across every compound, each against vehicle (n, mean, median, Hedges' g). Use it for "
    "'which compound ... the most?'.",
    required=("feature",),
    feature={"type": "string"},
)
def feature_by_compound(data: ResearchData, feature: str) -> dict[str, Any]:
    _check_feature(data, feature)
    vehicle = data.table.loc[data.compound == data.vehicle, feature].dropna()
    rows = []
    for name, part in data.table.groupby(data.compound):
        values = part[feature].dropna()
        row = {"compound": name, "n": len(values), "mean": _num(values.mean()), "median": _num(values.median())}
        if name != data.vehicle:
            row["hedges_g_vs_all_vehicle"] = _effect(values, vehicle)[0]
        rows.append(row)
    return {"feature": feature, "meaning": meaning(feature), "compounds": rows, "caveats": _caveats(data, feature)}


@tool("Everything about one fish (video id like F_0042): labels, date, flags, its behavior measures with percentiles, and what the models predicted for it out of fold.", required=("video_id",), video_id={"type": "string"})
def fish_profile(data: ResearchData, video_id: str) -> dict[str, Any]:
    row = _fish(data, video_id)
    compound = compound_label(row["compound"])
    same = data.table[data.compound == compound]
    vehicle = data.table[data.compound == data.vehicle]
    features = []
    for feature in _numeric_features(data):
        value = row[feature]
        if pd.isna(value):
            continue
        features.append({"feature": feature, "value": float(value),
                         "percentile_in_compound": _percentile(same[feature], value),
                         "percentile_in_vehicle": _percentile(vehicle[feature], value)})  # fmt: skip
    labels = {k: row.get(k) for k in ("compound", "concentration_mM", "date", "sex", "strain", "age", "reviewed", "review_status")}
    flags = {k: row.get(k) for k in ("flag_low_detected", "flag_odd_duration", "undetermined_share", "recording_s")}
    return {"video_id": video_id, **labels, "compound": compound, "flags": flags, "features": features[:MAX_ROWS * 2],
            "predictions": _fish_predictions(data, video_id), "source": _source(data)}  # fmt: skip


@tool(
    "What one fish did over time: behavior-state shares and mean speed per time bin, the list of bouts, and when "
    "each state first appeared. Use it for 'when did fish X freeze / swim erratically?'.",
    required=("video_id",),
    video_id={"type": "string"},
    bin_s={"type": "number", "description": "bin length in seconds (default 60)"},
)
def fish_timeline(data: ResearchData, video_id: str, bin_s: float = 60) -> dict[str, Any]:
    _fish(data, video_id)
    if data.gold_dir is None:
        raise ToolError("the video folders are not configured (DCS_PROCESSED_DIR or DCS_ACCEPTED_DIR)")
    folder = data.gold_dir / video_id
    try:
        frames = pd.read_parquet(folder / schema.FRAMES_FILE, columns=["t_sec", "state", "detected", "velocity"])
        segments = pd.read_csv(folder / schema.SEGMENTS_FILE)
    except (OSError, ValueError) as error:
        raise ToolError(f"cannot read the files of {video_id} in {folder}: {error}") from None
    step = float(np.median(np.diff(frames["t_sec"]))) if len(frames) > 1 else 0.0
    duration = float(segments["end_s"].max())
    if isinstance(bin_s, bool) or not isinstance(bin_s, (int, float)) or not math.isfinite(bin_s) or bin_s <= 0:
        raise ToolError(f"bin_s must be a number of seconds > 0, got {bin_s!r}")
    if math.ceil(duration / bin_s) > MAX_BINS:
        raise ToolError(f"bin_s {bin_s} gives more than {MAX_BINS} bins for {duration:.0f} s; use at least {duration / MAX_BINS:.3g}")
    det = frames["detected"].astype(bool)
    real = det & det.shift(fill_value=False)  # the first frame after a gap carries a 0.0 sentinel velocity
    frames = frames.assign(bin=(frames["t_sec"] // bin_s).astype(int), state=frames["state"].astype(str),
                           speed=frames["velocity"].where(real))  # fmt: skip
    known = frames[frames["state"] != schema.UNDETERMINED]
    known_frames = known.groupby("bin").size()
    shares = known.groupby("bin")["state"].value_counts(normalize=True)
    speeds = frames.groupby("bin")["speed"].mean()
    bins = []
    for number in range(math.ceil(duration / bin_s)):
        state_shares = shares.loc[number] if number in known_frames.index else pd.Series(dtype=float)
        bins.append({"start_s": number * bin_s, "end_s": min((number + 1) * bin_s, duration),
                     "known_s": int(known_frames.get(number, 0)) * step,
                     "state_shares": {k: float(v) for k, v in state_shares.items()},
                     "mean_speed_px_s": _num(speeds.get(number))})  # fmt: skip
    first = segments[segments["state"] != schema.UNDETERMINED].groupby("state")["start_s"].min()
    return {
        "video_id": video_id,
        "duration_s": duration,
        "bin_s": bin_s,
        "bins": bins,
        "bouts": segments[["start_s", "end_s", "state"]].head(MAX_ROWS).to_dict("records"),
        "bouts_total": len(segments),
        "first_bout_s": {k: float(v) for k, v in first.items()},
        "note": "Undetermined time is unknown, not a behavior; speeds are in pixels and depend on the camera",
    }


@tool(
    "Results of the latest training run for one stage ('compound' or 'dose <compound>'): every model's scores with "
    "the date held out (scheme A, the honest one), random split (B), labels shuffled within dates, the date effect, "
    "its verdict and why, and the saved model.",
    stage={"type": "string", "description": "default compound"},
)
def model_results(data: ResearchData, stage: str = "compound") -> dict[str, Any]:
    run = _run(data)
    metrics = _metrics(run, stage)
    path = run / "decisions.csv"  # absent in runs from before U19
    decisions = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=["stage", "model", "verdict", "reason"])
    decisions = decisions[decisions["stage"] == stage].set_index("model")
    rows = []
    for model, part in metrics.groupby("model", sort=False):
        score = part.groupby("scheme")["balanced_accuracy"].agg(["mean", "std"]).fillna(0.0)
        row = {"model": model, **{f"scheme_{s}": _score(score, s) for s in (SCHEME_A, SCHEME_B, "B_permuted")}}
        if SCHEME_A in score.index and SCHEME_B in score.index:
            row["date_effect_B_minus_A"] = float(score.loc[SCHEME_B, "mean"] - score.loc[SCHEME_A, "mean"])
        if model in decisions.index:
            row.update(verdict=decisions.loc[model, "verdict"], reason=decisions.loc[model, "reason"])
        rows.append(row)
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    return {
        "run_id": run.name,
        "stage": stage,
        "metric": "balanced accuracy (mean ± standard deviation across repeats); chance = 1 / number of classes",
        "models": rows,
        "saved_model": info.get("saved_model"),
        "source": info.get("gold_source"),
        "stages_in_run": info.get("stages"),
        "rule": "useful = scheme A beats majority and date_only by more than the spread and beats its own permuted score",
    }


@tool(
    "Per-class precision, recall and F1 of one model in one stage of the latest run (scheme A when it ran), and "
    "which classes it confuses most.",
    stage={"type": "string"},
    model={"type": "string", "description": "default: the saved model, else the best one"},
)
def class_scores(data: ResearchData, stage: str = "compound", model: str | None = None) -> dict[str, Any]:
    run = _run(data)
    predictions = pd.read_csv(run / "predictions.csv", low_memory=False)
    predictions = predictions[predictions["stage"] == stage]
    if predictions.empty:
        raise ToolError(f"stage {stage!r} is not in run {run.name}; stages: {sorted(_read_metrics(run)['stage'].unique())}")
    model = model or _chosen_model(run, stage)
    scheme = SCHEME_A if (predictions["scheme"] == SCHEME_A).any() else SCHEME_B
    rows = predictions[(predictions["model"] == model) & (predictions["scheme"] == scheme)]
    if rows.empty:
        raise ToolError(f"model {model!r} is not in stage {stage!r}; models: {sorted(predictions['model'].unique())}")
    labels = sorted(set(rows["label"]) | set(rows["predicted"]))
    p, r, f, n = precision_recall_fscore_support(rows["label"], rows["predicted"], labels=labels, zero_division=0)
    matrix = confusion_matrix(rows["label"], rows["predicted"], labels=labels)
    confusions = [{"true": labels[i], "predicted_as": labels[j], "count": int(matrix[i, j])}
                  for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i, j]]  # fmt: skip
    return {
        "stage": stage, "model": model, "scheme": scheme, "pooled_repeats": int(rows["repeat"].nunique()),
        "classes": [{"label": l, "precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)}
                    for l, a, b, c, d in zip(labels, p, r, f, n) if d],
        "top_confusions": sorted(confusions, key=lambda c: -c["count"])[:10],
    }  # fmt: skip


@tool("Ablations of the latest run: each model's scores with one change (NTT off, demographics on, depth on, vehicle-normalized) next to the main run.")
def ablation_results(data: ResearchData) -> dict[str, Any]:
    metrics = _read_metrics(_run(data))
    rows = []
    for (ablation, stage, model), part in metrics.groupby(["ablation", "stage", "model"], sort=False):
        score = part.groupby("scheme")["balanced_accuracy"].agg(["mean", "std"]).fillna(0.0)
        if ablation != "main" or stage == "compound":
            rows.append({"ablation": ablation, "stage": stage, "model": model, "scheme_A": _score(score, SCHEME_A), "scheme_B": _score(score, SCHEME_B)})
    note = "" if any(r["ablation"] != "main" for r in rows) else "this run has no ablations (it ran with --no-ablations)"
    return {"ablations": rows, "note": note}


@tool("Data audit facts: fish, dates, compounds, review status, frame rates, camera framing, the G1 go/wait rule, and the notes to keep in mind.")
def audit_facts(data: ResearchData) -> dict[str, Any]:
    audit = build_audit(data.table, data.described, data.training)
    return {"facts": audit.facts, "notes": list(audit.notes)}


# --- helpers -------------------------------------------------------------------------------------


def meaning(name: str) -> str:
    """Plain-English meaning of a table column, from its name."""
    for state in BEHAVIOR_STATES:
        s = slug(state)
        known = {
            f"state_{s}_share": f"share of the fish's known time spent in {state}",
            f"state_{s}_bouts": f"number of {state} bouts",
            f"state_{s}_mean_bout_s": f"mean length of a {state} bout, seconds",
            f"state_{s}_latency_s": f"seconds until the first {state} bout (the recording length if never)",
        }
        if name in known:
            return known[name]
    if name.startswith("trans_") and "_to_" in name:
        a, b = name[len("trans_"):].split("_to_", 1)
        return f"number of direct changes from {_state_name(a)} to {_state_name(b)}"
    if name.startswith("depth_"):
        return "vertical position in pixels from the frame top (" + name[len("depth_"):] + "); depends on the camera framing"
    return KINEMATICS.get(name) or NTT.get(name) or name.replace("_", " ")


def _state_name(state_slug: str) -> str:
    return next((state for state in BEHAVIOR_STATES if slug(state) == state_slug), state_slug)


def _numeric_features(data: ResearchData) -> list[str]:
    return [c["name"] for c in data.described["columns"] if c["role"] == "feature" and c.get("kind") != "category"
            and c["name"] in data.table.columns and pd.api.types.is_numeric_dtype(data.table[c["name"]])]  # fmt: skip


def _check_feature(data: ResearchData, feature: str) -> None:
    if feature not in _numeric_features(data):
        raise ToolError(f"unknown feature {feature!r}; call find_features to look names up")


def _rows(data: ResearchData, compound: str, dose: str | None) -> pd.Series:
    name = compound_label(compound)
    if name not in set(data.compound):
        raise ToolError(f"unknown compound {compound!r}; compounds: {', '.join(sorted(set(data.compound)))}")
    if name == data.vehicle:
        raise ToolError("that is the vehicle; compare a drug with it instead")
    rows = data.compound == name
    if dose is not None:
        doses = sorted(set(data.dose[rows]))
        if dose_label(dose) not in doses:
            raise ToolError(f"{name} has no dose {dose!r}; doses: {', '.join(doses)}")
        rows &= data.dose == dose_label(dose)
    return rows


def _control(data: ResearchData, rows: pd.Series, control: str) -> tuple[pd.Series, str, list[str]]:
    """Vehicle fish on the group's dates (default), else all vehicle fish; with a caveat when it had to fall back."""
    if control not in (SAME_DATES, ALL_VEHICLE):
        raise ToolError(f"control must be {SAME_DATES} or {ALL_VEHICLE}")
    vehicle = data.compound == data.vehicle
    if not vehicle.any():
        raise ToolError(f"no vehicle fish: the vehicle name {data.vehicle!r} is not in the data (training.vehicle_compound)")
    dates = set(data.table.loc[rows, "date"])
    same = vehicle & data.table["date"].isin(dates)
    if control == SAME_DATES and same.sum() >= MIN_CONTROL:
        return same, f"vehicle fish on the same dates ({len(dates)} dates)", []
    caveats = [] if control == ALL_VEHICLE else [f"only {int(same.sum())} vehicle fish on the same dates, so all vehicle fish are the control: date and camera differences are not controlled"]
    return vehicle, "all vehicle fish", caveats


def _effect(group: pd.Series, reference: pd.Series) -> tuple[float, float | None]:
    """Hedges' g (group minus reference, small-sample corrected) and the two-sided Mann-Whitney p."""
    n1, n2 = len(group), len(reference)
    if n1 < 2 or n2 < 2:
        return 0.0, None
    pooled = math.sqrt(((n1 - 1) * group.var() + (n2 - 1) * reference.var()) / (n1 + n2 - 2))
    g = 0.0 if pooled == 0 else (group.mean() - reference.mean()) / pooled * (1 - 3 / (4 * (n1 + n2) - 9))
    p = float(stats.mannwhitneyu(group, reference).pvalue) if pd.concat([group, reference]).nunique() > 1 else 1.0
    return float(g), p


def _describe(values: pd.Series) -> dict[str, Any]:
    if values.empty:
        return {"n": 0}
    return {"n": len(values), "mean": float(values.mean()), "sd": _num(values.std()), "median": float(values.median()),
            "q1": float(values.quantile(0.25)), "q3": float(values.quantile(0.75)), "min": float(values.min()), "max": float(values.max())}  # fmt: skip


def _source(data: ResearchData) -> str:
    if data.described["gold_source"] == SOURCE_PROCESSED:
        return "UNREVIEWED pipeline output, treated as temporarily accepted (not validated by a reviewer)"
    return "Accepted (reviewed) videos"


def _caveats(data: ResearchData, feature: str | None) -> list[str]:
    out = []
    if data.described["gold_source"] == SOURCE_PROCESSED:
        out.append("UNREVIEWED data (temporarily accepted): states and tracks are not validated by a reviewer")
    group = next((c.get("feature_group") for c in data.described["columns"] if c["name"] == feature), None)
    if feature is None or group in ("kinematics", "depth"):
        out.append("pixel-based measures (speeds, depth) depend on the camera zoom and framing, which changed between recording periods")
    return out


def _fish(data: ResearchData, video_id: str) -> pd.Series:
    match = data.table[data.table["video_id"].astype(str) == video_id]
    if match.empty:
        raise ToolError(f"no fish {video_id!r} in the table; ids look like {data.table['video_id'].iloc[0]}")
    return match.iloc[0]


def _percentile(values: pd.Series, value: float) -> float | None:
    values = values.dropna()
    return None if values.empty else float(stats.percentileofscore(values, value, kind="mean"))


def _run(data: ResearchData) -> Path:
    if data.run is None or not (data.run / "metrics.csv").is_file():
        raise ToolError("no training run found: run `python -m dcs train` first (or pass the run folder)")
    return data.run


def _read_metrics(run: Path) -> pd.DataFrame:
    """metrics.csv; a run from before the ablations (U14) has no `ablation` column: all of it is the main run."""
    metrics = pd.read_csv(run / "metrics.csv")
    return metrics if "ablation" in metrics.columns else metrics.assign(ablation="main")


def _metrics(run: Path, stage: str) -> pd.DataFrame:
    metrics = _read_metrics(run)
    main = metrics[(metrics["ablation"] == "main") & (metrics["stage"] == stage)]
    if main.empty:
        raise ToolError(f"stage {stage!r} is not in run {run.name}; stages: {', '.join(sorted(metrics['stage'].unique()))}")
    return main


def _chosen_model(run: Path, stage: str) -> str:
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8")).get("saved_model") or {}
    if info.get("stage") == stage:
        return info["model"]
    score = _metrics(run, stage).query("scheme == @SCHEME_A or scheme == @SCHEME_B").groupby("model")["balanced_accuracy"].mean()
    return str(score.idxmax())


def _fish_predictions(data: ResearchData, video_id: str) -> dict[str, Any] | None:
    if data.run is None or not (data.run / "predictions.csv").is_file():
        return None
    model = _chosen_model(data.run, "compound")
    predictions = pd.read_csv(data.run / "predictions.csv", low_memory=False)
    rows = predictions[(predictions["stage"] == "compound") & (predictions["model"] == model) & (predictions["video_id"] == video_id)]
    scheme = SCHEME_A if (rows["scheme"] == SCHEME_A).any() else SCHEME_B
    rows = rows[rows["scheme"] == scheme]
    if rows.empty:
        return {"model": model, "repeats": 0, "note": "never scored out of fold (its class was pinned or not kept)"}
    true = f"p:{compound_label(rows['label'].iloc[0])}"
    return {"model": model, "scheme": scheme, "repeats": len(rows), "predicted_counts": rows["predicted"].value_counts().to_dict(),
            "mean_probability_of_true_class": _num(rows[true].mean()) if true in rows else None}  # fmt: skip


def _score(score: pd.DataFrame, scheme: str) -> str | None:
    return None if scheme not in score.index else f"{score.loc[scheme, 'mean']:.3f} ± {score.loc[scheme, 'std']:.3f}"


def _num(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _plain(value: Any) -> Any:
    """JSON-safe: numpy scalars to Python, tuples to lists, NaN to None, dates to text."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
