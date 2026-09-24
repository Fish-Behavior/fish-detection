"""End-to-end test of `all` on a tiny synthetic project: a workbook, one 20 s video, and a
synthetic reference PDF (the helpers of test_catalog / test_reference / synthetic)."""

import numpy as np
import pymupdf
import pytest
import yaml

from fishbehavior.cli import main
from synthetic import make_video
from test_catalog import TRACKED, make_workbook, row
from test_reference import draw_figure, random_rows

OUTPUTS = [  # every file `all` must produce for subject 0042 (relative to the output folder)
    "catalog/trials.csv", "scene/F_0042.json", "scene/F_0042_qa.png", "tracks/0042.csv.gz",
    "features/0042_bins.csv.gz", "features/endpoints.csv", "labels/0042_bins.csv", "labels/segments.csv",
    "labels/summary.csv", "datasets/segments.csv", "datasets/per_second_labels.csv",
    "datasets/behavior_dataset.csv", "datasets/behavior_dataset.xlsx", "datasets/behavior_windows.csv",
    "plots/ethogram_Drug-A_0.03.png", "plots/bars_freeze_drift.png",
]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Workbook with subject 42 (and one without video), its 20 s video, and a .env; returns the CLI options."""
    for name in ("FISH_VIDEO_DIR", "FISH_DB_PATH", "FISH_REFERENCE_PDF", "FISH_OUTPUT_DIR", "FISH_WORKERS", "FISH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    make_workbook(tmp_path / "db.xlsx", [row("0042", "F", "Drug-A", TRACKED), row("0043", "M", "Drug-B", TRACKED)])
    (tmp_path / "videos").mkdir()
    make_video(tmp_path / "videos" / "F_0042.avi", duration_s=20.0)
    (tmp_path / ".env").write_text("FISH_DB_PATH=db.xlsx\nFISH_VIDEO_DIR=videos\nFISH_OUTPUT_DIR=out\nFISH_WORKERS=1\n")
    return tmp_path, ["--env-file", str(tmp_path / ".env")]


def test_all_produces_every_output_and_a_second_run_skips_finished_work(project, capsys):
    root, cli = project

    assert main([*cli, "all"]) == 0

    out = capsys.readouterr().out
    missing = [path for path in OUTPUTS if not (root / "out" / path).is_file()]
    assert not missing, missing
    assert "skipped (FISH_REFERENCE_PDF not set)" in out and "total" in out
    for name in ("validate", "scene", "track", "features", "label", "export", "plot"):
        assert f"  {name:<11}" in out  # one timing line per step

    assert main([*cli, "all"]) == 0
    assert "1 done (1 cached)" in capsys.readouterr().out  # tracking etc. not redone


def test_all_stops_at_the_first_failing_step(project, capsys):
    root, cli = project
    (root / ".env").write_text("FISH_DB_PATH=missing.xlsx\nFISH_VIDEO_DIR=videos\nFISH_OUTPUT_DIR=out\n")

    assert main([*cli, "all"]) == 1

    out = capsys.readouterr().out
    assert "Stopped at `validate`" in out and "=== scene ===" not in out
    assert not (root / "out" / "tracks").exists()


def test_tuning_steps_wait_for_the_mapping_then_run(project, capsys):
    root, cli = project
    rng = np.random.default_rng(5)
    doc = pymupdf.open()
    for _ in range(2):  # pages 1 and 2: one ethogram each (config below), page 3: the bar charts
        jpeg, _ = draw_figure([random_rows(5, rng), random_rows(4, rng)], rng)
        doc.new_page(width=960, height=540).insert_image(pymupdf.Rect(40, 40, 840, 497), stream=jpeg)
    doc.new_page(width=960, height=540)
    doc.save(root / "reference.pdf")
    (root / "config.yaml").write_text("reference:\n  ethogram_pages: [1, 2]\n  barchart_page: 3\n"
                                      "calibration:\n  n_trials: 5\n  folds: 2\n")
    with (root / ".env").open("a") as env:
        env.write("FISH_REFERENCE_PDF=reference.pdf\n")
    cli = [*cli, "--config", str(root / "config.yaml")]

    assert main([*cli, "all"]) == 0  # first run: template written, tuning skipped, rest done
    assert "fill in mapping.yaml" in capsys.readouterr().out
    assert not (root / "out" / "calibration" / "calibrated.yaml").exists()
    assert main([*cli, "all"]) == 0  # template still empty: skipped again, not a failure
    assert "mapping.yaml not filled in" in capsys.readouterr().out

    mapping_path = root / "out" / "reference" / "mapping.yaml"
    mapping = yaml.safe_load(mapping_path.read_text())
    next_id = 9001
    for entry in mapping["panels"]:  # made-up ids, with subject 42 in the first panel
        entry["group"] = f"G{entry['page']}{entry['panel']}"
        entry["subject_ids"] = [f"{next_id + k}" for k in range(entry["rows"])]
        next_id += entry["rows"]
    mapping["panels"][0]["subject_ids"][0] = "42"
    mapping_path.write_text(yaml.safe_dump(mapping))

    assert main([*cli, "all"]) == 0

    out = capsys.readouterr().out
    for name in ("priors", "calibrate", "label again"):
        assert f"  {name:<11}" in out
    assert (root / "out" / "calibration" / "calibrated.yaml").is_file()
    assert (root / "out" / "calibration" / "priors_report.md").is_file()
    assert main([*cli, "all"]) == 0
    assert "skipped (calibrated.yaml is up to date)" in capsys.readouterr().out
