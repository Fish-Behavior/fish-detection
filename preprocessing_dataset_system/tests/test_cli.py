"""check-config exit codes for prepds.cli (T007)."""

from __future__ import annotations

from pathlib import Path

from prepds.cli import main


def test_check_config_ok_when_no_paths_required(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here, so no path is configured/required
    exit_code = main(["check-config"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PDS_VIDEO_DIR" in captured.out


def test_check_config_reports_missing_path(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PDS_VIDEO_DIR", str(tmp_path / "does_not_exist"))
    exit_code = main(["check-config"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "MISSING" in captured.out


def test_missing_env_file_reports_config_error(capsys) -> None:
    exit_code = main(["--env-file", "/no/such/file.env", "check-config"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration error" in captured.out


def test_version_flag(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--version should exit")


# --- Phase 12: run / export-index / review -----------------------------------------------

import datetime as dt

import pandas as pd
import pytest

from prepds.models import MatchStatus, Trial

FIXTURE_VIDEO = Path(__file__).parent / "fixtures" / "synth_tiny.mp4"


def _write_catalog(out_dir: Path, subjects=("0001", "0002")) -> None:
    trials = [
        Trial(s, "F", "Casper", None, "Veh", "1% DMSO", dt.date(2026, 3, 1), 20.0, FIXTURE_VIDEO, MatchStatus.MATCHED)
        for s in subjects
    ]
    trials.append(Trial("0099", "M", "Casper", None, "Veh", "1% DMSO", dt.date(2026, 3, 1), 20.0, None, MatchStatus.NO_VIDEO))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([t.to_dict() for t in trials]).to_parquet(out_dir / "trials_catalog.parquet", index=False)


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PDS_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("PDS_ACCEPTED_DIR", str(tmp_path / "gold"))
    _write_catalog(tmp_path / "out")
    return tmp_path


def test_run_command_dry_run_lists_pending_videos(env, capsys) -> None:
    assert main(["run", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "F_0001" in out and "F_0002" in out and "F_0099" not in out
    assert not (env / "out" / "processed").exists()


def test_run_processes_matched_videos_then_resumes_without_rework(env, capsys) -> None:
    assert main(["run", "--workers", "1"]) == 0
    assert (env / "out" / "processed" / "F_0001" / "manifest.json").is_file()
    assert (env / "out" / "run_report.json").is_file()
    capsys.readouterr()
    assert main(["run", "--dry-run"]) == 0
    assert "nothing to do" in capsys.readouterr().out.lower()


def test_run_limit_and_report_counts(env, capsys) -> None:
    assert main(["run", "--workers", "1", "--limit", "1"]) == 0
    assert (env / "out" / "processed" / "F_0001").is_dir() and not (env / "out" / "processed" / "F_0002").exists()


def test_run_without_a_catalog_explains_what_to_do(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PDS_OUTPUT_DIR", str(tmp_path / "out"))
    assert main(["run"]) == 2
    assert "catalog" in capsys.readouterr().out


def test_run_exit_code_is_1_when_a_video_fails(env, capsys) -> None:
    pd.DataFrame([Trial("0005", "F", "C", None, "Veh", "1% DMSO", dt.date(2026, 3, 1), 20.0, env / "missing.mp4", MatchStatus.MATCHED).to_dict()]).to_parquet(
        env / "out" / "trials_catalog.parquet", index=False)
    assert main(["run", "--workers", "1"]) == 1
    assert "F_0005" in capsys.readouterr().out


def test_run_rejects_a_non_positive_workers_value(env, capsys) -> None:
    with pytest.raises(SystemExit):
        main(["run", "--workers", "0"])


def test_export_index_rebuilds_from_accepted_videos(env, capsys) -> None:
    from prepds.review_store import accept, INDEX_FILE

    main(["run", "--workers", "1"])
    video = env / "out" / "processed" / "F_0001"
    # the synthetic video has undetermined frames; force a fully determined strip by editing
    from prepds.review_store import save_edit
    from prepds.models import BehaviorState as B
    from prepds.review_store import load_manifest
    duration = load_manifest(video).video_duration_s
    save_edit(video, [(0.0, duration, B.CONTROLLED_SWIM)], reviewer="lk", now=dt.datetime(2026, 9, 23, tzinfo=dt.timezone.utc))
    accept(video, accepted_dir=env / "gold", reviewer="lk", now=dt.datetime(2026, 9, 23, tzinfo=dt.timezone.utc))
    (env / "gold" / INDEX_FILE).unlink()
    assert main(["export-index"]) == 0
    index = pd.read_parquet(env / "gold" / INDEX_FILE)
    assert list(index["video_id"]) == ["F_0001"] and index.iloc[0]["strain"] == "Casper"


def test_review_refuses_a_non_loopback_host(env, capsys) -> None:
    assert main(["review", "--host", "0.0.0.0"]) == 2
    assert "loopback" in capsys.readouterr().out.lower()


def test_duplicate_video_ids_are_reported_but_do_not_keep_failing_every_run(env, capsys) -> None:
    _write_catalog(env / "out", subjects=("0001", "0001"))
    assert main(["run", "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "duplicate" in out.lower()
    assert main(["run", "--dry-run"]) == 0
    assert "nothing to do" in capsys.readouterr().out.lower()


def test_export_index_without_a_catalog_keeps_previously_indexed_trial_fields(env, capsys) -> None:
    from prepds.models import BehaviorState as B
    from prepds.review_store import INDEX_FILE, accept, load_manifest, save_edit

    main(["run", "--workers", "1"])
    video = env / "out" / "processed" / "F_0001"
    now = dt.datetime(2026, 9, 23, tzinfo=dt.timezone.utc)
    save_edit(video, [(0.0, load_manifest(video).video_duration_s, B.CONTROLLED_SWIM)], reviewer="lk", now=now)
    trial = Trial("0001", "F", "Casper", 5.0, "Veh", "1% DMSO", dt.date(2026, 3, 1), 20.0, FIXTURE_VIDEO, MatchStatus.MATCHED)
    accept(video, accepted_dir=env / "gold", reviewer="lk", now=now, trial=trial)
    (env / "out" / "trials_catalog.parquet").unlink()
    assert main(["export-index"]) == 0
    out = capsys.readouterr().out
    assert "catalog" in out.lower()  # warns that workbook fields could not be refreshed
    assert pd.read_parquet(env / "gold" / INDEX_FILE).iloc[0]["strain"] == "Casper"


def test_annotate_refuses_a_non_loopback_host(env, capsys) -> None:
    assert main(["annotate", "--host", "0.0.0.0"]) == 2
    assert "loopback" in capsys.readouterr().out.lower()


def test_annotate_without_prepared_frames_explains_what_to_run(env, capsys) -> None:
    assert main(["annotate"]) == 2
    assert "phase15_prepare" in capsys.readouterr().out


# --- Phase 15: --tracker model:<run> ---------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_run(tmp_path_factory) -> Path:
    pytest.importorskip("torch")
    from PIL import Image

    from prepds.annotation.examples import Example
    from prepds.annotation.train import train

    tmp = tmp_path_factory.mktemp("cli_model")
    image = tmp / "f.png"
    Image.new("RGB", (128, 96), (100, 120, 90)).save(image)
    example = Example("f", image, 128, 96, (10.0, 20.0, 100.0, 80.0), {"snout": (20.0, 50.0, 2)}, "no")
    train([example], tmp / "run", epochs=1, pretrained=False, device="cpu", min_size=128, max_size=170, log=lambda *_: None)
    return tmp / "run"


def test_run_with_a_model_tracker_processes_videos_and_records_it(env, tiny_model_run, capsys) -> None:
    import json

    assert main(["run", "--tracker", f"model:{tiny_model_run}", "--stride", "3", "--device", "cpu"]) == 0
    assert (env / "out" / "processed" / "F_0001" / "manifest.json").is_file()
    report = json.loads((env / "out" / "run_report.json").read_text())
    assert report["tracker"] == "model:run@stride3" and report["workers"] == 1


def test_an_unknown_tracker_kind_is_a_config_error(env, capsys) -> None:
    assert main(["run", "--tracker", "magic:x"]) == 2
    assert "tracker" in capsys.readouterr().out.lower()


def test_a_missing_model_run_is_a_config_error(env, capsys) -> None:
    assert main(["run", "--tracker", "model:/no/such/run", "--device", "cpu"]) == 2
