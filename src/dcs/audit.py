"""Training table -> data audit (plan U8, T1.17; PRD FR-2, AC-1, §2.3; EC-11, EC-24, EC-26, EC-31).

`build_audit` reads only the two featurize files (D-039) and changes nothing. It runs each stage's class filter,
training set and folds as `train` will, and tabulates what is in the data, what was dropped and why, and what could
let a model score by recognizing the day, the labeling process or the camera instead of the compound (PRD §2.3).
`render` turns the result into `audit.md`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from dcs.config import ConfigError
from dcs.config_rules import SOURCE_PROCESSED
from dcs.featurize import BEHAVIOR_STATES, slug
from dcs.folds import SCHEME_A, SCHEME_B, make_folds
from dcs.trainset import STAGES, build_trainset, compound_label, dose_label, filter_classes, stage_labels

AUDIT_FILE = "audit.md"
GO, WAIT = "GO", "WAIT"
STOP_RULE_COMPOUNDS = 3  # G1 (PRD §10): this many compounds besides vehicle, each with
STOP_RULE_DATES = 2  # min_class_size Accepted fish spread over at least this many dates
# ponytail: one fixed share of the median tank height (EC-31); make it a setting if G1 needs to tune it
FRAMING_TOLERANCE = 0.1
KEPT, VERY_SMALL = "kept", "very small"
SECTIONS = (
    ("Filter steps", "steps"),
    ("Compounds", "compounds"),
    ("Classes: compound", "classes_compound"),
    ("Classes: dose", "classes_dose"),
    ("Dates", "dates"),
    ("Compound by date", "compound_by_date"),
    ("Missing values", "missing"),
    ("Flagged fish (EC-11, kept)", "flagged"),
    ("States", "states"),
    ("Camera framing (EC-31)", "framing"),
)


@dataclass(frozen=True)
class Audit:
    facts: dict[str, Any]  # set-level numbers and verdicts, in report order
    tables: dict[str, pd.DataFrame]  # keyed by the names in SECTIONS
    notes: tuple[str, ...]  # what the reader must act on or keep in mind


def build_audit(table: pd.DataFrame, described: Mapping[str, Any], training: Mapping[str, Any]) -> Audit:
    """Every AC-1 table and fact for the training table `table` and its schema `described`."""
    compound = table["compound"].map(compound_label)
    dose = stage_labels(table, "dose")
    vehicle = compound_label(training["vehicle_compound"])
    reviewed = table["reviewed"].astype(bool)
    notes: list[str] = []
    facts = _setup(described, len(table), notes)
    if vehicle not in set(compound):
        notes.append(
            f"vehicle {vehicle} is not in the table: set training.vehicle_compound to the vehicle's name; "
            "until then the stop rule counts every compound"
        )

    dose_dates = table["date"].groupby(dose).nunique()
    facts.update(
        {
            "fish in the table": len(table),
            "Accepted fish": int(reviewed.sum()),
            "dates": table["date"].nunique(),
            "fish per date": len(table) / max(table["date"].nunique(), 1),
            "compounds (with vehicle)": compound.nunique(),
            "vehicle fish": int((compound == vehicle).sum()),
            "vehicle dates": table.loc[compound == vehicle, "date"].nunique(),
            "compound+dose classes": len(dose_dates),
            "compound+dose classes on one date": int((dose_dates == 1).sum()),
            "compound+dose classes below min_class_size": int((dose.value_counts() < training["min_class_size"]).sum()),
            "compounds on one date": sorted(c for c, n in table["date"].groupby(compound).nunique().items() if n == 1),
            "dates where one compound has 2+ doses": int((dose.groupby([compound, table["date"]]).nunique() >= 2).sum()),
            "fish with NTT": int(table["has_ntt"].sum()),
            "video duration range (s)": f"{table['video_duration_s'].min():g}-{table['video_duration_s'].max():g}",
            "fish flagged low detected share": int(table["flag_low_detected"].sum()),
            "fish flagged odd duration": (
                "not checked (training.duration_range_s is null)"
                if described["flags"]["duration_range_s"] is None
                else int(table["flag_odd_duration"].sum())
            ),
        }
    )

    before = sorted(Counter(drop["reason"] for drop in described["dropped"]).items())
    steps = [(f"dropped before the table: {reason}", fish) for reason, fish in before]
    steps.append(("in the training table", len(table)))
    tables: dict[str, pd.DataFrame] = {}
    dropped_states: dict[str, tuple[str, ...]] = {}
    for stage in STAGES:
        stage_steps, classes, dropped_states[stage] = _stage(table, described, training, stage, facts, notes)
        steps += stage_steps
        tables[f"classes_{stage}"] = classes

    min_fish = training["min_class_size"]
    verdict, enough = _stop_rule(compound[reviewed], table["date"][reviewed], vehicle, min_fish)
    facts["G1 stop rule"] = verdict
    facts["G1 compounds with enough Accepted fish on 2+ dates"] = enough
    if not reviewed.all():
        facts["G1 stop rule counting unreviewed fish (Q17)"] = _stop_rule(compound, table["date"], vehicle, min_fish)[0]
    if verdict == WAIT:
        notes.append(
            f"G1 stop rule: WAIT. {len(enough)} compound(s) besides vehicle have {min_fish}+ Accepted fish on "
            f"{STOP_RULE_DATES}+ dates; {STOP_RULE_COMPOUNDS} are needed (PRD §10)"
        )

    framing, differs, tolerance = _framing(table)
    setups = int(framing["setup"].nunique())
    unchecked = framing.loc[framing["fish"] == 0, "date"].astype(str).tolist()
    facts.update(
        {"framing differs between dates": differs, "framing setups": setups, "dates without depth data": len(unchecked)}
    )
    if differs:
        how = (
            f"falls into {setups} setups (dates grouped where top, bottom and height agree within {tolerance:.3g} px)"
            if setups > 1
            else f"drifts across dates by more than {tolerance:.3g} px"
        )
        notes.append(
            f"EC-31: camera framing {how}: keep training.use_depth off and treat pixel speeds as date-dependent "
            "(compare scheme A with and without kinematics)"
        )
    if unchecked:
        notes.append(f"EC-31: no depth data on {len(unchecked)} date(s) ({', '.join(unchecked)}): framing not checked there")

    tables.update(
        steps=pd.DataFrame(steps, columns=["step", "fish"]),
        compounds=_compounds(table, compound, reviewed),
        dates=_dates(table, compound, reviewed, vehicle),
        compound_by_date=pd.crosstab(compound.rename("compound"), table["date"]),
        missing=_missing(table, described),
        flagged=_flagged(table, compound),
        states=pd.DataFrame(
            {
                "state": BEHAVIOR_STATES,
                "fish": [int((table[f"state_{slug(state)}_bouts"] > 0).sum()) for state in BEHAVIOR_STATES],
                "dropped in": [", ".join(s for s in STAGES if state in dropped_states[s]) for state in BEHAVIOR_STATES],
            }
        ),
        framing=framing,
    )
    return Audit(facts, tables, tuple(notes))


def render(audit: Audit) -> str:
    """The audit as markdown: facts, notes, then one table per section."""
    lines = ["# Data audit", "", *(f"- {key}: {_text(value)}" for key, value in audit.facts.items())]
    lines += ["", "## Notes", "", *(f"- {note}" for note in audit.notes or ("none",))]
    for title, name in SECTIONS:
        lines += ["", f"## {title}", "", _markdown(audit.tables[name])]
    return "\n".join(lines) + "\n"


# --- parts -------------------------------------------------------------------------------------


def _setup(described: Mapping[str, Any], fish: int, notes: list[str]) -> dict[str, Any]:
    """Source, tracker and recording setup (EC-21, EC-26), with notes for what was not or could not be checked."""
    unreviewed = described["gold_source"] == SOURCE_PROCESSED
    if unreviewed:
        notes.append("UNREVIEWED: the fish come from the pipeline output, not from Accepted videos (Q17, D-033)")
    checked = described["tracker_checked"]
    if checked < fish:
        notes.append(
            f"tracker not cross-checked for {fish - checked} fish (no prepds processed folder for them, or "
            f"DCS_PROCESSED_DIR not set); the profile name alone says {described['tracker']}"
        )
    low, high = described["fps_range"]
    if not described["fps_uniform"]:
        notes.append(
            f"EC-26: frame rates differ ({low:g}-{high:g} fps): pixel-based features (kinematics, depth) are not "
            "comparable across videos"
        )
    return {
        "source": "UNREVIEWED pipeline output" if unreviewed else "Accepted gold folder",
        "profile": described["profile"],
        "tracker": described["tracker"],
        "tracker cross-checked": f"{checked} of {fish} fish",
        "frame rate uniform": described["fps_uniform"],
        "frame rate range": f"{low:g}-{high:g} fps",
        "resolution": "not recorded upstream",
        "NTT available": described["ntt_available"],
    }


def _stage(
    table: pd.DataFrame,
    described: Mapping[str, Any],
    training: Mapping[str, Any],
    stage: str,
    facts: dict[str, Any],
    notes: list[str],
) -> tuple[list[tuple[str, int]], pd.DataFrame, tuple[str, ...]]:
    """One stage's filter steps, class table and dropped states; a stage that cannot be built becomes a note."""
    labels = stage_labels(table, stage)
    keep, dropped = filter_classes(labels, table["compound"].map(compound_label), stage, training["min_class_size"])
    by_reason: Counter[str] = Counter()
    for drop in dropped:
        by_reason[drop["reason"]] += drop["fish"]
    steps = [(f"{stage}: dropped, {reason}", fish) for reason, fish in by_reason.items()]
    steps.append((f"{stage}: kept", int(keep.sum())))
    try:
        trainset = build_trainset(table, described, training, stage)
        folds = make_folds(trainset.y, trainset.groups, trainset.ids, training)
    except ConfigError as error:
        notes.append(f"{stage} stage cannot be built: {error}")
        trainset = folds = None
    pinned = () if folds is None else folds.date_confounded
    held_out = folds is not None and folds.k[SCHEME_A] > 0
    steps.append((f"{stage}: scored in scheme A", int((~trainset.y.isin(pinned)).sum()) if held_out else 0))

    status = {d["label"]: d["reason"] for d in dropped}
    rows = table.assign(label=labels, reviewed=table["reviewed"].astype(bool)).groupby("label")
    classes = pd.DataFrame({"fish": rows.size(), "accepted": rows["reviewed"].sum(), "dates": rows["date"].nunique()})
    small = training["min_class_size"]
    classes["status"] = [status.get(label, VERY_SMALL if fish == small else KEPT) for label, fish in classes["fish"].items()]
    classes["scheme_a"] = [
        "" if folds is None or label in status else "skipped" if not held_out else "pinned" if label in pinned else "scored"
        for label in classes.index
    ]

    facts[f"{stage}: classes kept"] = 0 if trainset is None else len(trainset.classes)
    if trainset is not None:
        per_date = trainset.y.groupby(trainset.groups).nunique()
        facts[f"{stage}: folds (A, B)"] = (folds.k[SCHEME_A], folds.k[SCHEME_B])
        facts[f"{stage}: date-confounded"] = list(pinned)
        facts[f"{stage}: share of fish on dates with 2+ classes"] = float(trainset.groups.map(per_date >= 2).mean())
        notes += [f"{stage}: {note}" for note in folds.notes]
    return steps, classes.reset_index(), () if trainset is None else trainset.dropped_states


def _stop_rule(compound: pd.Series, dates: pd.Series, vehicle: str, min_fish: int) -> tuple[str, list[str]]:
    """G1: GO when enough compounds besides vehicle have min_fish fish on STOP_RULE_DATES or more dates."""
    counts = dates[compound != vehicle].groupby(compound[compound != vehicle]).agg(["size", "nunique"])
    enough = sorted(counts.index[(counts["size"] >= min_fish) & (counts["nunique"] >= STOP_RULE_DATES)])
    return (GO if len(enough) >= STOP_RULE_COMPOUNDS else WAIT), enough


def _compounds(table: pd.DataFrame, compound: pd.Series, reviewed: pd.Series) -> pd.DataFrame:
    """Per compound: fish, Accepted fish and dates (EC-24), doses, labeling-process shares, flags, protocol."""
    rows = table.assign(compound=compound, reviewed=reviewed, dose=table["concentration_mM"].map(dose_label))
    rows = rows.groupby("compound")
    out = pd.DataFrame(
        {
            "fish": rows.size(),
            "accepted": rows["reviewed"].sum(),
            "dates": rows["date"].nunique(),
            "accepted_dates": table[reviewed].groupby(compound[reviewed])["date"].nunique(),
            "doses": rows["dose"].nunique(),
            "manual_share": rows["manual_share"].mean(),
            "undetermined_share": rows["undetermined_share"].mean(),
            "low_detected": rows["flag_low_detected"].sum(),
            "odd_duration": rows["flag_odd_duration"].sum(),
            "exposure_min": rows["agent_exposure_min"].agg(lambda v: ", ".join(sorted({_text(x) for x in v.dropna()}))),
        }
    )
    out["accepted_dates"] = out["accepted_dates"].fillna(0).astype(int)
    return out.rename_axis("compound").reset_index()


def _dates(table: pd.DataFrame, compound: pd.Series, reviewed: pd.Series, vehicle: str) -> pd.DataFrame:
    rows = table.assign(compound=compound, reviewed=reviewed, vehicle=compound == vehicle).groupby("date")
    return pd.DataFrame(
        {
            "fish": rows.size(),
            "accepted": rows["reviewed"].sum(),
            "compounds": rows["compound"].nunique(),
            "vehicle": rows["vehicle"].sum(),
        }
    ).reset_index()


def _missing(table: pd.DataFrame, described: Mapping[str, Any]) -> pd.DataFrame:
    role = {column["name"]: column["role"] for column in described["columns"]}
    counts = table.isna().sum()
    counts = counts[counts > 0]
    roles = [role.get(name, "") for name in counts.index]
    return pd.DataFrame({"column": counts.index, "role": roles, "missing": counts.to_numpy()})


def _flagged(table: pd.DataFrame, compound: pd.Series) -> pd.DataFrame:
    """EC-11: fish with a low detected share or an odd duration; kept, listed here."""
    hit = table["flag_low_detected"].astype(bool) | table["flag_odd_duration"].astype(bool)
    out = table.loc[hit, ["video_id", "date", "detected_share", "video_duration_s"]]
    out.insert(1, "compound", compound[hit])
    out["low_detected"], out["odd_duration"] = table["flag_low_detected"][hit], table["flag_odd_duration"][hit]
    return out.reset_index(drop=True)


def _framing(table: pd.DataFrame) -> tuple[pd.DataFrame, bool, float]:
    """EC-31: per date, where the typical fish's top and bottom sit in the frame (median of the per-fish 1st and 99th
    depth percentiles), the height between them, and the extremes beside. Medians, so one fish or a crowded date
    cannot move a date. Dates whose top, bottom and height agree within the tolerance (FRAMING_TOLERANCE of the
    median height), directly or through other dates, share a `setup`. Returns the table (every date, a date
    without depth data has 0 fish and no setup), whether any two dates differ by more than the tolerance (so slow
    drift counts too), and the tolerance in pixels."""
    rows = table[["date", "depth_p01", "depth_p99"]].dropna().groupby("date")
    out = pd.DataFrame({"fish": rows.size(), "top": rows["depth_p01"].median(), "bottom": rows["depth_p99"].median()})
    out = out.reindex(pd.Index(sorted(table["date"].unique(), key=str), name="date"))
    out["fish"] = out["fish"].fillna(0).astype(int)
    out["height"] = out["bottom"] - out["top"]
    tolerance = FRAMING_TOLERANCE * float(out["height"].median())
    measured = out[["top", "bottom", "height"]].dropna()
    out["setup"] = _setups(measured, tolerance).reindex(out.index)
    out["top_min"], out["bottom_max"] = rows["depth_p01"].min(), rows["depth_p99"].max()
    differs = len(measured) >= 2 and bool(((measured.max() - measured.min()) > tolerance).any())
    return out.reset_index(), differs, tolerance


def _setups(measured: pd.DataFrame, tolerance: float) -> pd.Series:
    """Single-linkage groups (largest of the top, bottom and height gaps <= tolerance), numbered in date order."""
    if len(measured) < 2:
        return pd.Series(1, index=measured.index, dtype="Int64")
    groups = fcluster(linkage(measured.to_numpy(), method="single", metric="chebyshev"), tolerance, criterion="distance")
    number = {group: n for n, group in enumerate(dict.fromkeys(groups), start=1)}
    return pd.Series([number[group] for group in groups], index=measured.index, dtype="Int64")


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "none"
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(_text(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(map(_text, value)) or "none"
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)
