"""Privacy (plan U18, T5.5; PRD NFR-2, AC-8; D-007): git holds no data, output, model or .env file.

Runs inside the existing CI job (no extra workflow). Skipped where the tests do not run from a git checkout.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FORBIDDEN = re.compile(
    r"\.(parquet|csv|xlsx|xls|pt|pth|joblib|pkl|pickle|npy|npz|h5|mp4|avi|mov)$|(^|/)\.env$", re.IGNORECASE
)
# Synthetic files made for the prepds tests; no real fish, names or dates.
ALLOWED = {"tests/fixtures/synth_db.xlsx", "tests/fixtures/synth_tiny.mp4"}
# What dcs and prepds write, and where the restricted inputs live: all must be ignored.
MUST_BE_IGNORED = (
    ".env",
    "outputs/dcs/training_table.parquet",
    "outputs/dcs/training/20000101T000000Z-abc1234/model/mlp.pt",
    "outputs/dcs/training/20000101T000000Z-abc1234/model/sklearn.joblib",
    "outputs/dcs/training/20000101T000000Z-abc1234/predictions.csv",
    "outputs/processed/F_0001/frames.parquet",
    "accepted/F_0001/frames.parquet",
    "data/00_NTT_DataBase.xlsx",
    "anywhere/model.pt",
    "anywhere/model.joblib",
)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)


def test_no_data_model_or_env_file_is_tracked() -> None:
    tracked = git("ls-files").stdout.splitlines()
    assert tracked, "git ls-files returned nothing"
    offenders = [name for name in tracked if FORBIDDEN.search(name) and name not in ALLOWED]
    assert offenders == [], f"tracked data/output files (remove with `git rm --cached`): {offenders}"


@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_outputs_and_restricted_inputs_are_ignored(path: str) -> None:
    assert git("check-ignore", "--quiet", "--no-index", path).returncode == 0, f"{path} is not git-ignored"


@pytest.mark.parametrize(
    ("name", "caught"),
    [
        ("outputs/x.parquet", True), ("a/b/predictions.CSV", True), ("model.pt", True), ("m.joblib", True),
        (".env", True), ("sub/.env", True), (".env.example", False), ("src/dcs/train.py", False),
        ("docs/notes.md", False), ("config/default_training.yaml", False),
    ],
)  # fmt: skip
def test_the_pattern_catches_what_it_should(name: str, caught: bool) -> None:
    """The guard itself: a pattern that matched nothing would let every leak through."""
    assert bool(FORBIDDEN.search(name)) is caught
