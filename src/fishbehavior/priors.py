"""Sanity checks of the video labels against the workbook and the reference figures.

None of these is ground truth; they point at subjects and groups worth a look:

1. NTT hints: Spearman correlation between video measures (freeze_drift seconds, distance
   in BL, surface_breach seconds) and the workbook's tdm_full / velocity_full / time_top_s.
   The workbook comes from the Novel Tank Test, a DIFFERENT 10-min session, so only a
   loose relation is expected. Subjects far off the rank trend of a matching pair
   (|z| > `priors.flag_z`) are flagged: a possible tracking failure or unusual behavior.
2. Group checks: the video-derived mean seconds per state per group next to the
   reference `group_means.csv` (digitized from the figures).
3. Direction checks: relations listed in the LOCAL file `reference/expectations.yaml`
   (e.g. a group shows more LORR than the vehicle group), tested on both the reference
   and the video means. The same file links each reference group to its workbook subjects
   (compound + concentration), so group names never have to be in Git.

Output: ``<FISH_OUTPUT_DIR>/calibration/priors_report.md`` (git-ignored).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, rankdata, spearmanr

from fishbehavior.config import ConfigError, Settings
from fishbehavior.features import features_dir
from fishbehavior.labeling import STATES, SUMMARY_FILE, labels_dir
from fishbehavior.reference import reference_dir

VIDEO_MEASURES = ("freeze_drift_s", "distance_bl", "surface_breach_s")
WORKBOOK_MEASURES = ("tdm_full", "velocity_full", "time_top_s")
# Pairs that measure roughly the same thing in both sessions; residual flags use these.
PAIRS = (("distance_bl", "tdm_full"), ("freeze_drift_s", "velocity_full"), ("surface_breach_s", "time_top_s"))
RELATIONS = {"greater_than": operator.gt, "less_than": operator.lt}
EXPECTATIONS_FILE, REPORT_FILE = "expectations.yaml", "priors_report.md"
ENDPOINTS_FILE, GROUP_MEANS_FILE, TIMELINES_FILE = "endpoints.csv", "group_means.csv", "timelines.csv"
HINT_NOTE = ("The workbook values come from the Novel Tank Test, a different 10-minute session, not from "
             "these exposure videos. Treat every number here as a hint, never as per-second truth.")

EXPECTATIONS_HEADER = """\
# Local file (git-ignored): what the priors step checks. Edit, then run `priors` again.
#
# groups: which workbook subjects form each reference group (the groups of
#   reference/group_means.csv): `compound` and `concentration` exactly as in the columns
#   compound / concentration_raw of catalog/trials.csv. Filled in automatically where the
#   subjects in mapping.yaml tell; fill in the empty ones by hand (e.g. the vehicle group).
# expectations: relations the video labels should reproduce, one per line, e.g.
#   - {group: <name>, state: lorr, relation: greater_than, than: <vehicle group>}
#   relation: greater_than or less_than; state: one of STATES_LIST
"""


def calibration_dir(settings: Settings) -> Path:
    """Folder shared by priors and calibrate."""
    return settings.paths.output_dir / "calibration"


def markdown_table(table: pd.DataFrame, digits: int = 2) -> str:
    """A DataFrame as a GitHub markdown table (numbers rounded, NaN blank)."""
    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.{digits}f}"
        return str(value)
    lines = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. NTT hints
# ---------------------------------------------------------------------------


def joined_table(settings: Settings, trials: pd.DataFrame) -> pd.DataFrame:
    """Video measures (labels summary + endpoints) joined with the workbook, NTT-tracked subjects only."""
    summary_path, endpoints_path = labels_dir(settings) / SUMMARY_FILE, features_dir(settings) / ENDPOINTS_FILE
    for path, step in ((summary_path, "label"), (endpoints_path, "features")):
        if not path.is_file():
            raise ConfigError(f"{path} not found: run `python -m fishbehavior {step}` first")
    summary = pd.read_csv(summary_path, dtype={"subject_id": str})
    endpoints = pd.read_csv(endpoints_path, dtype={"subject_id": str})[["subject_id", "distance_bl"]]
    workbook = trials[trials["ntt_tracked"].astype(str) == "True"][["subject_id", *WORKBOOK_MEASURES]]
    return summary.merge(endpoints, on="subject_id").merge(workbook, on="subject_id")


def spearman_table(table: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho (and p, n) of every video measure against every workbook measure."""
    rows = []
    for video in VIDEO_MEASURES:
        for workbook in WORKBOOK_MEASURES:
            pair = table[[video, workbook]].apply(pd.to_numeric, errors="coerce").dropna()
            rho, p = spearmanr(pair[video], pair[workbook]) if len(pair) >= 3 else (np.nan, np.nan)
            rows.append({"video": video, "workbook": workbook, "n": len(pair), "rho": float(rho), "p": float(p)})
    return pd.DataFrame(rows)


def residual_flags(table: pd.DataFrame, z_limit: float) -> pd.DataFrame:
    """Subjects far from the rank trend of a PAIRS pair: |z of the residual| > z_limit.

    Normal scores of the ranks (not raw values) keep one extreme subject from bending the line,
    matching Spearman, while the residuals stay roughly normal, so |z| > 2 keeps its usual meaning
    (plain ranks are uniform and almost never reach |z| = 2).
    """
    rows = []
    for video, workbook in PAIRS:
        pair = table[["subject_id", video, workbook]].copy()
        pair[[video, workbook]] = pair[[video, workbook]].apply(pd.to_numeric, errors="coerce")
        pair = pair.dropna()
        if len(pair) < 5:
            continue  # too few for a trend
        x, y = (norm.ppf((rankdata(pair[c]) - 0.5) / len(pair)) for c in (workbook, video))
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        z = (residual - residual.mean()) / (residual.std() or 1.0)
        for row, value in zip(pair.itertuples(index=False), z):
            if abs(value) > z_limit:
                rows.append({"subject_id": row[0], "video": video, "video_value": row[1],
                             "workbook": workbook, "workbook_value": row[2], "z": round(float(value), 2)})
    return pd.DataFrame(rows, columns=["subject_id", "video", "video_value", "workbook", "workbook_value", "z"])


# ---------------------------------------------------------------------------
# 2. Groups: reference group -> workbook subjects (expectations.yaml)
# ---------------------------------------------------------------------------


def expectations_template(ref_means: pd.DataFrame, timelines: pd.DataFrame, trials: pd.DataFrame) -> str:
    """expectations.yaml text: every reference group linked to the compound/concentration of its
    mapped subjects (most common one), or left blank when none of them is in the workbook."""
    known = trials.set_index("subject_id")[["compound", "concentration_raw"]].astype(str)
    members = timelines.drop_duplicates("subject_id").set_index("subject_id")["group"]
    groups = {}
    for group in ref_means["group"]:
        ids = [s for s in members.index[members == group] if s in known.index]
        if ids:
            compound, concentration = known.loc[ids].value_counts().idxmax()
            groups[group] = {"compound": compound, "concentration": concentration}
        else:
            groups[group] = {"compound": "", "concentration": ""}
    body = yaml.safe_dump({"groups": groups, "expectations": []}, sort_keys=False, allow_unicode=True)
    return EXPECTATIONS_HEADER.replace("STATES_LIST", ", ".join(STATES)) + "\n" + body


def load_expectations(path: Path, groups: list[str]) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    """(groups links, expectations) from expectations.yaml, with every name checked."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    links = {str(k): {"compound": str(v.get("compound") or ""), "concentration": str(v.get("concentration") or "")}
             for k, v in (data.get("groups") or {}).items()}
    expectations = data.get("expectations") or []
    for item in expectations:
        where = f"{path}: expectation {item}"
        if not isinstance(item, dict) or set(item) != {"group", "state", "relation", "than"}:
            raise ConfigError(f"{where} needs exactly group, state, relation, than")
        if item["relation"] not in RELATIONS:
            raise ConfigError(f"{where}: relation must be one of {', '.join(RELATIONS)}")
        if item["state"] not in STATES:
            raise ConfigError(f"{where}: state must be one of {', '.join(STATES)}")
        for key in ("group", "than"):
            if str(item[key]) not in groups:
                raise ConfigError(f"{where}: {item[key]!r} is not a reference group ({', '.join(groups)})")
    return links, expectations


def video_group_means(summary: pd.DataFrame, trials: pd.DataFrame,
                      links: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Per linked reference group: number of labeled subjects and mean seconds per state."""
    info = trials[["subject_id", "compound", "concentration_raw"]].astype(str).merge(summary, on="subject_id")
    rows = []
    for group, link in links.items():
        if not link["compound"]:
            continue  # not linked yet
        members = info[(info["compound"] == link["compound"]) & (info["concentration_raw"] == link["concentration"])]
        rows.append({"group": group, "n_subjects": len(members),
                     **{state: members[f"{state}_s"].mean() for state in STATES}})
    return pd.DataFrame(rows, columns=["group", "n_subjects", *STATES])


def group_table(ref_means: pd.DataFrame, video_means: pd.DataFrame) -> pd.DataFrame:
    """Reference vs video mean seconds, one row per group and state."""
    video = video_means.set_index("group")
    rows = []
    for ref in ref_means.itertuples(index=False):
        linked = ref.group in video.index
        for state in STATES:
            ours = float(video.loc[ref.group, state]) if linked else np.nan
            rows.append({"group": ref.group, "state": state, "n_reference": int(ref.n_subjects),
                         "n_video": int(video.loc[ref.group, "n_subjects"]) if linked else 0,
                         "reference_s": float(getattr(ref, state)), "video_s": ours,
                         "difference_s": ours - float(getattr(ref, state))})
    return pd.DataFrame(rows)


def direction_checks(expectations: list[dict[str, Any]], ref_means: pd.DataFrame,
                     video_means: pd.DataFrame) -> pd.DataFrame:
    """Each expectation tested on the reference means and on the video means (None: group not linked)."""
    def holds(means: pd.DataFrame, item: dict[str, Any]) -> bool | None:
        table = means.set_index("group")
        if item["group"] not in table.index or item["than"] not in table.index:
            return None
        a, b = float(table.loc[item["group"], item["state"]]), float(table.loc[item["than"], item["state"]])
        return bool(RELATIONS[item["relation"]](a, b)) if np.isfinite(a) and np.isfinite(b) else None
    return pd.DataFrame([{"expectation": f"{e['group']} {e['state']} {e['relation']} {e['than']}",
                          "reference": holds(ref_means, e), "video": holds(video_means, e)} for e in expectations],
                        columns=["expectation", "reference", "video"])


# ---------------------------------------------------------------------------
# 3. Whole run
# ---------------------------------------------------------------------------


@dataclass
class PriorsResult:
    """What `run_priors` found, for the CLI summary."""

    n_subjects: int  # subjects in the NTT join
    spearman: pd.DataFrame
    flags: pd.DataFrame
    groups: pd.DataFrame | None  # None: no reference digitized yet
    checks: pd.DataFrame | None
    expectations_created: bool
    unlinked: list[str]  # reference groups without workbook subjects in expectations.yaml
    report: Path


def run_priors(settings: Settings, trials: pd.DataFrame) -> PriorsResult:
    """Compute all checks and write priors_report.md."""
    table = joined_table(settings, trials)
    spearman = spearman_table(table)
    flags = residual_flags(table, float(settings.params["priors"]["flag_z"]))

    ref_dir = reference_dir(settings)
    means_path, expectations_path = ref_dir / GROUP_MEANS_FILE, ref_dir / EXPECTATIONS_FILE
    groups = checks = None
    created, unlinked = False, []
    if means_path.is_file():
        ref_means = pd.read_csv(means_path)
        if not expectations_path.is_file():
            timelines = pd.read_csv(ref_dir / TIMELINES_FILE, dtype={"subject_id": str}, usecols=["subject_id", "group"])
            expectations_path.write_text(expectations_template(ref_means, timelines, trials), encoding="utf-8")
            created = True
        links, expectations = load_expectations(expectations_path, list(ref_means["group"]))
        unlinked = [g for g in ref_means["group"] if not links.get(g, {}).get("compound")]
        summary = pd.read_csv(labels_dir(settings) / SUMMARY_FILE, dtype={"subject_id": str})
        video_means = video_group_means(summary, trials, links)
        groups, checks = group_table(ref_means, video_means), direction_checks(expectations, ref_means, video_means)

    out = calibration_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    result = PriorsResult(len(table), spearman, flags, groups, checks, created, unlinked, out / REPORT_FILE)
    result.report.write_text(report_text(result, float(settings.params["priors"]["flag_z"])), encoding="utf-8")
    return result


def report_text(result: PriorsResult, z_limit: float) -> str:
    """priors_report.md."""
    parts = ["# Priors report", "", f"> {HINT_NOTE}", "",
             f"## 1. Video vs workbook (NTT), {result.n_subjects} subjects", "",
             "Spearman rank correlation (rho), with p-value and number of subjects:", "",
             markdown_table(result.spearman, 3), "",
             f"Subjects off the rank trend by |z| > {z_limit:g} on a matching pair "
             f"({', '.join(f'{a} ~ {b}' for a, b in PAIRS)}). Check their track QA image first:", ""]
    parts.append(markdown_table(result.flags) if len(result.flags) else "_none_")
    parts += ["", "## 2. Group means: video vs reference (seconds)", ""]
    if result.groups is None:
        parts.append("_No reference yet: run `python -m fishbehavior reference`._")
    else:
        if result.unlinked:
            parts += [f"Not linked to workbook subjects yet (fill `groups` in reference/{EXPECTATIONS_FILE}): "
                      + ", ".join(result.unlinked), ""]
        parts.append(markdown_table(result.groups, 1))
        parts += ["", "## 3. Direction checks", ""]
        parts.append(markdown_table(result.checks) if len(result.checks)
                     else f"_No expectations listed in reference/{EXPECTATIONS_FILE}._")
    return "\n".join(parts) + "\n"
