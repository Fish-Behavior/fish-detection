"""Tests for fishbehavior.reference: a synthetic ethogram figure (numpy + JPEG) in a PDF built with pymupdf."""

import cv2
import numpy as np
import pandas as pd
import pymupdf
import pytest
import yaml

from fishbehavior.cli import main
from fishbehavior.config import load_settings
from fishbehavior.labeling import STATES

PALETTE = load_settings(environ={}).params["reference"]["palette"]
SECONDS = 1200
PAGES = {1: [13, 10, 7], 2: [13, 4]}  # rows per panel; page 2 panel 1 repeats page 1 panel 1 (skip: true)
# Like the real figures: every panel equally tall, so rows are 3.8-12.5 px (not whole pixels); ~0.75 px per second.
PANEL_H, X0, X1 = 50, 120, 1015


def random_rows(n, rng):
    """n subject timelines: runs of 8-80 s of random states; the last subject's data ends at 900 s."""
    rows = np.empty((n, SECONDS), object)
    for r in range(n):
        s = 0
        while s < SECONDS:
            length = int(rng.integers(8, 80))
            rows[r, s:s + length] = rng.choice(STATES)
            s += length
    rows[-1, 900:] = "no_data"
    return rows


def draw_figure(panels, rng):
    """A ggplot-like ethogram image (JPEG bytes) and the true label rows of each panel."""
    image = np.full((600, 1050, 3), 255, np.uint8)
    second_of_column = ((np.arange(X0, X1) + 0.5 - X0) / (X1 - X0) * SECONDS).astype(int)
    y, truth = 60, []
    for index, rows in enumerate(panels):
        cv2.putText(image, f"GROUP {index}", (480, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        edges = y + np.round(np.arange(len(rows) + 1) * PANEL_H / len(rows)).astype(int)
        for r, row in enumerate(rows):
            colors = np.array([PALETTE[name] for name in row], np.uint8)
            image[edges[r]:edges[r + 1], X0:X1] = colors[second_of_column]
            # Overlapping subject labels on the left, like the real figure.
            cv2.putText(image, f"F_{r:03d}", (20, edges[r] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        truth.append(rows)
        y += PANEL_H + 45
    for k, name in enumerate(STATES):  # legend boxes
        cv2.rectangle(image, (200 + 150 * k, 560), (215 + 150 * k, 575), PALETTE[name], -1)
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg.tobytes(), truth


@pytest.fixture(scope="module")
def figure(tmp_path_factory):
    """reference.pdf: pages 1-2 ethograms (page 1 also has a small logo image), page 3 bar charts."""
    folder = tmp_path_factory.mktemp("reference")
    rng = np.random.default_rng(7)
    page1 = [random_rows(n, rng) for n in PAGES[1]]
    page2 = [page1[0]] + [random_rows(n, rng) for n in PAGES[2][1:]]  # repeated control panel on top
    doc = pymupdf.open()
    truth = {}
    for number, panels in ((1, page1), (2, page2)):
        jpeg, truth[number] = draw_figure(panels, rng)
        page = doc.new_page(width=960, height=540)
        page.insert_image(pymupdf.Rect(40, 40, 840, 497), stream=jpeg)
        if number == 1:  # a smaller image on the same page must not be taken for the figure
            ok, logo = cv2.imencode(".png", np.zeros((40, 40, 3), np.uint8))
            page.insert_image(pymupdf.Rect(860, 20, 900, 60), stream=logo.tobytes())
    doc.new_page(width=960, height=540).draw_rect(pymupdf.Rect(100, 100, 200, 400), fill=(1, 0, 0))
    doc.save(folder / "reference.pdf")
    return folder / "reference.pdf", truth


@pytest.fixture
def env(figure, tmp_path, monkeypatch):
    """Global CLI options for a run on the synthetic PDF, from an empty folder."""
    for name in ("FISH_VIDEO_DIR", "FISH_DB_PATH", "FISH_REFERENCE_PDF", "FISH_OUTPUT_DIR", "FISH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"FISH_REFERENCE_PDF={figure[0]}\nFISH_OUTPUT_DIR={tmp_path / 'out'}\n",
                                   encoding="utf-8")
    (tmp_path / "config.yaml").write_text("reference:\n  ethogram_pages: [1, 2]\n  barchart_page: 3\n",
                                          encoding="utf-8")
    return ["--env-file", str(tmp_path / ".env"), "--config", str(tmp_path / "config.yaml")]


def fill_mapping(path, drop_one=False):
    """Fill the template like a user would: made-up ids per panel, page 2 panel 1 skipped. Returns {id: (page, panel, row)}."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    where, next_id = {}, 101
    for entry in data["panels"]:
        if (entry["page"], entry["panel"]) == (2, 1):
            entry["skip"] = True
            continue
        entry["group"] = f"G{entry['page']}{entry['panel']}"
        ids = [f"F_{next_id + k:04d}" for k in range(entry["rows"])]
        for row in range(len(ids)):
            where[f"{next_id + row:04d}"] = (entry["page"], entry["panel"], row)
        next_id += entry["rows"]
        entry["subject_ids"] = ids[:-1] if drop_one and entry["panel"] == 2 else ids
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return where


def test_first_run_writes_a_mapping_template_with_estimated_rows(env, tmp_path, capsys):
    assert main([*env, "reference"]) == 1  # nothing digitized until the mapping is filled in

    out = tmp_path / "out" / "reference"
    template = yaml.safe_load((out / "mapping.yaml").read_text(encoding="utf-8"))["panels"]
    expected = [(page, index + 1, n) for page, rows in PAGES.items() for index, n in enumerate(rows)]
    assert [(e["page"], e["panel"], e["rows"]) for e in template] == expected
    assert all(e["group"] == "" and e["subject_ids"] == [] and e["skip"] is False for e in template)
    # The largest image of each page is the figure (not the logo); the bar-chart page is rendered whole.
    assert cv2.imread(str(out / "page1.png")).shape[:2] == (600, 1050)
    assert (out / "page3.png").is_file()
    assert "template written" in capsys.readouterr().out

    # Running again before filling it in gives one short line, not one sentence per panel.
    assert main([*env, "reference"]) == 2
    assert "not filled in yet: page 1 panels 1, 2, 3; page 2 panels 1, 2." in capsys.readouterr().out


def test_digitized_timelines_match_the_figure(env, figure, tmp_path, capsys):
    main([*env, "reference"])
    out = tmp_path / "out" / "reference"
    where = fill_mapping(out / "mapping.yaml")

    assert main([*env, "reference"]) == 0
    timelines = pd.read_csv(out / "timelines.csv", dtype={"subject_id": str})
    assert set(timelines["subject_id"]) == set(where)  # the skipped panel adds no rows
    truth, scores = figure[1], {}
    for subject, rows in timelines.groupby("subject_id"):
        page, panel, row = where[subject]
        scores[subject] = (rows.sort_values("second")["label"].to_numpy() == truth[page][panel - 1][row]).mean()
    assert np.mean(list(scores.values())) >= 0.95  # share of all seconds right
    assert min(scores.values()) >= 0.9, scores  # and no subject row misplaced

    means = pd.read_csv(out / "group_means.csv")
    assert list(means["group"]) == ["G11", "G12", "G13", "G22"]
    assert list(means["n_subjects"]) == [13, 10, 7, 4]
    assert cv2.imread(str(out / "digitize_check.png")) is not None
    slide18 = yaml.safe_load((out / "slide18_values.yaml").read_text(encoding="utf-8"))
    assert list(slide18) == ["G11", "G12", "G13", "G22"] and slide18["G11"]["erratic"] is None

    # Filled-in slide-18 values are compared with the digitized means; the digitizing is cached.
    slide18["G12"]["erratic"] = 100
    (out / "slide18_values.yaml").write_text(yaml.safe_dump(slide18), encoding="utf-8")
    capsys.readouterr()
    assert main([*env, "reference"]) == 0
    assert "(cached)" in capsys.readouterr().out
    check = pd.read_csv(out / "slide18_check.csv")
    assert check[["group", "state", "slide18_s"]].values.tolist() == [["G12", "erratic", 100.0]]
    assert check["digitized_s"].iloc[0] == means.set_index("group").loc["G12", "erratic"]


def test_row_count_mismatch_is_reported(env, tmp_path, capsys):
    main([*env, "reference"])
    fill_mapping(tmp_path / "out" / "reference" / "mapping.yaml", drop_one=True)
    capsys.readouterr()

    assert main([*env, "reference"]) == 2
    message = capsys.readouterr().out
    assert "page 1 panel 2: 9 subject_ids but rows: 10" in message
    assert message.count("\n") == 1  # one line, like every configuration error
