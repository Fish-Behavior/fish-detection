# Plan: Compound and Dose Classifier (`dcs`)

| | |
|---|---|
| **Source of truth** | [../classifier_PRD.md](../classifier_PRD.md) v0.3. Where this plan and the PRD disagree, the PRD wins until the owner approves the change list in §2 |
| **Status and decisions** | [../classifier_progress.md](../classifier_progress.md): live task status, the decision log (D-nnn) and the answers to the open questions |
| **Created** | 2026-10-02, after reading the PRD, [../process/HARNESS_PROCESS.md](../process/HARNESS_PROCESS.md), [../PRD.md](../PRD.md) and the `prepds` code it describes |
| **Privacy** | Placeholders only (`COMPOUND_A`, `F_0042`, `<date>`). No data, outputs, real names, dates or paths |
| **Revisions** | r1 2026-10-02: external review findings R1-R7 (verdicts in the progress file §8); changes C16-C21 below |

Phases come from PRD §10, tests from PRD §9 (every edge-case row is a task, §6 of this plan), and "done" from PRD §1.6 and §9.3 (§8 of this plan).

**What this phase is.** Training **infrastructure**, built and tested on **synthetic data only**. Labels are not validated with the pharmacy department and no video is Accepted yet, so no result produced in this phase is a claim about drug identity.

---

## 1. Inputs checked against the code

Confirmed in `src/prepds` (2026-10-02):

| Item | Fact |
|---|---|
| `frames.parquet` | `frame_idx` int32; `t_sec`, `x`, `y`, `orientation_deg`, `depth_from_surface`, `velocity`, `acceleration`, `angular_velocity`, `meander`, `confidence` float32; `detected`, `is_immobile` bool; `state` category (7 values), `source` category (`auto`, `manual`) |
| State names | `Controlled Swim`, `Erratic Movement`, `Freezing/Drift`, `Listing/LORR`, `Surface Breach`, `Dead`, `Undetermined` |
| Undetected frames | `x`, `y` and the four kinematic columns hold a `0.0` sentinel; `orientation_deg`, `depth_from_surface` are null. Kinematic features must use `detected == True` rows only. A few detected frames right after a gap also carry the `0.0` sentinel (too little history); first accepted as a small bias, then left out of the averages after review (D-048), because the bias grew with patchy tracking |
| `depth_from_surface` | Pixel row from the **top of the frame**, not from the waterline |
| `segments.csv` | `start_s, end_s, duration_s, state, source` |
| Gold folder per video | `<accepted>/<video_id>/` holds `frames.parquet`, `segments.csv`, `strip.png`, `manifest.json`, `provenance.json`. `video_id` = `<sex>_<subject_id>` (e.g. `F_0042`) |
| `accepted_index.parquet` | `video_id, subject_id, sex, compound, concentration_mM, strain, age, date, agent_exposure_min, video_path, video_duration_s, video_fps, pipeline_version, calibration_profile_version, reviewer, reviewed_at, provenance, edit_count, frames_path, segments_path, strip_path, manifest_path` |
| Accept rules | `Undetermined` hard-blocks Accept, so EC-22 is a defensive check |
| Packaging / CI | Python ≥3.11; scikit-learn, pyarrow, matplotlib, openpyxl, PyYAML, python-dotenv are core deps; CI runs `pytest tests/` recursively on Linux 3.11 with `--cov=prepds` and an 85 % floor; no `tests/__init__.py`, so test base names must be unique |

## 2. PRD change list (proposed v0.4; not applied until the owner approves)

| # | PRD text | Finding in code / review | Plan resolution |
|---|---|---|---|
| C1 | §5.2: tracker read from `run_report.json` or a `detections.parquet` next to the frames | `run_report.json` is one batch-level file in the output folder, overwritten every run; `detections.parquet` stays in the processed folder; `accept()` copies neither into the gold folder | D-003: infer the tracker from `calibration_profile_version` (model-tracker profiles contain `-model`, pattern configurable); if `DCS_PROCESSED_DIR` is set, a `detections.parquet` there must agree. Upstream request (T5.6): `prepds` records the tracker (and resolution) in `manifest.json` |
| C2 | §5.3 / EC-26: audit checks resolution | Resolution is not stored in the manifest, index or catalog; only `video_fps` is | Audit checks frame rate; resolution reported as "not recorded upstream" until the upstream request lands |
| C3 | §5.3: NTT columns from the index or `trials_catalog.parquet` | Neither carries them; they exist only in the workbook | D-004: `featurize` reads the workbook (`DCS_DB_PATH`) for the 8 NTT columns |
| C4 | §2.2: `date` is in the index | Accepting in the review app writes `date`, `strain`, `age`, `agent_exposure_min` as null until `prepds export-index` runs with a catalog | D-005: `featurize` stops with a `ConfigError` naming the fix. New EC-27 |
| C5 | §5.2: `prepds` fails duplicate `sex_subject` ids | It warns and skips; the index is unique per `video_id` by construction | EC-10 kept as a defensive check |
| C6 | AC-9 / Appendix B step 9: `predict` must equal `predictions.csv` | `predictions.csv` is out-of-fold (FR-5); the final model refit on all fish cannot reproduce it | D-006: save `model/reference_predictions.csv` (final model on the training rows) and compare against that |
| C7 | §1.5 S5: "CI privacy job green" | No such job exists | D-007: `tests/dcs/test_dcs_privacy.py` checks `git ls-files`; it runs inside the existing backend job (CI file unchanged, Q8) |
| C8 | §6.3: PyTorch MLP | Scope doc §7 names TensorFlow as the modeling framework | Owner decided: PyTorch for now; TensorFlow only if PyTorch proves less efficient (D-027). No PRD change needed |
| C9 | FR-9: subtract the same-date vehicle median | In evaluation this uses the **labels** of vehicle test fish, which is label leakage when vehicle is also a predicted class | D-008: in the FR-9 ablation, vehicle fish are the per-date reference only and are removed from the evaluated classes; the report says so |
| C10 | §7.1: four modules (`featurize`, `trainset`, `models`, `train`) | `train.py` would exceed the harness constitution's 200-400 lines per file (CONST-ARCH-003) | D-009: split into the modules of §3 (one step per module, still NFR-6) |
| C11 | §5.3: state "Erratic" | Code: `Erratic Movement` | Use the code's names |
| C12 | §10 Phase 0: read the real schema and count fish before Phase 1 | Counting needs `dcs audit` (Phase 1 code) and Accepted videos | D-010: real-data audit moves to gate G1, run with `dcs audit` |
| C13 | §6.2: date-only baseline | Undefined for scheme A, where test dates are unseen | D-011: nearest-training-date majority label (scheme B: same-date majority; scheme A: captures experiment campaigns in time) |
| C14 | NFR-2: ignore rules cover outputs | `*.pt` and `*.joblib` are not ignored outside `outputs/` | T1.1 adds them to `.gitignore` |
| C15 | §12 Q11 "open, default no" | `harness init` already ran and is on `origin/master` (commit "harness: initial project scaffold") | Q11 answered: keep governance on (§4) |
| C16 | §5.3: depth features (mean, min, percentiles of `depth_from_surface`) are default features | Pixel position from the frame top encodes camera framing, which can change by date (review R1); pixel speeds likewise depend on zoom | D-015: depth features form an ablation group, **off by default**; audit framing check per date (EC-31) decides at G1 whether they may be switched on and whether pixel speeds are flagged |
| C17 | FR-9 compared with the N-class baseline | Excluding vehicle (C9) changes the class set, so the metrics are not comparable (R2) | D-008 revised: FR-9 is reported as a pair, raw vs normalized, both on the same non-vehicle classes and folds |
| C18 | §6.6 within-date permutation | Dates holding a single class contribute nothing to it (R3) | Kept as the primary null; the audit and report give the share of fish on dates with ≥2 classes (D-016) |
| C19 | §6.4 / §6.7: how fold scores combine | Unstated (R4) | D-017: metrics are computed on the pooled out-of-fold predictions of each repeat; the spread is across repeats |
| C20 | §6.11 1D-CNN input | Recording lengths differ (including the shorter-exposure videos) (R5) | EC-30 in Phase 6: pad to the longest recording with a mask; masked global pooling |
| C21 | FR-8 `predict` input | `featurize` needs the whole index, so a new fish cannot be featurized alone (R6) | D-018: `featurize --videos <dir>` builds the table from per-video folders without an index; labels optional |
| C22 | §8.3 / Appendix B step 6: copy only the training table to the GB10 | Copying onto the box may be blocked (least privileges, owner 2026-10-08) | D-057: the box clones the repository, re-runs `prepds` on the raw videos and trains on that output as temporarily accepted (`gold_source: processed`); runbook in instructions.md |

New edge cases proposed for §9.2: **EC-27** index row without `date` (C4); **EC-28** workbook row for a fish missing, duplicated or disagreeing with the index; **EC-29** tracker evidence disagrees (profile name vs `detections.parquet`); **EC-30** sequences of different lengths (C20, Phase 6); **EC-31** camera framing differs between dates (C16).

## 3. Architecture

### 3.1 Files (all new unless marked)

```
pyproject.toml                    (edit) [project.scripts] dcs = "dcs.cli:main"; extra train = ["torch>=2.4"]
.gitignore                        (edit) add *.pt, *.joblib
.env.example                      (edit) add the DCS_* variables
config/default_training.yaml      training: defaults (no data-derived numbers)
src/dcs/__init__.py               version
src/dcs/__main__.py               python -m dcs
src/dcs/cli.py                    subcommands: synth, featurize, audit, train, predict
src/dcs/config.py                 Settings, DCS_* env vars, ConfigError (same pattern as prepds, no import of it)
src/dcs/synthetic.py              synthetic gold dataset + workbook with planted signal and edge-case knobs
src/dcs/gold.py                   read and validate accepted_index + per-video files (schema, tracker/profile, contract)
src/dcs/workbook.py               NTT columns from the workbook, keyed by video_id
src/dcs/featurize.py              per-fish features (pure functions) -> training_table.parquet + schema json
src/dcs/trainset.py               load table, labels, class filter, state support, forbidden columns, feature matrix
src/dcs/folds.py                  scheme A / B folds, pinning, folds.csv
src/dcs/audit.py                  audit tables -> audit.md
src/dcs/preprocess.py             fold-fitted imputer, log1p, scaler, vehicle normalization; to/from preprocess.json
src/dcs/models.py                 majority, date-only, logreg, RF, HGB behind one interface (no torch)
src/dcs/mlp.py                    PyTorch MLP, imported lazily
src/dcs/evaluate.py               CV loop on shared folds, metrics, diagnostics, decision rule
src/dcs/report.py                 report.md, metrics.csv, predictions.csv, confusion_*.png
src/dcs/train.py                  orchestration: stages, models, ablations, run folder
src/dcs/artifact.py               save/load model folder, run_info.json, predict
tests/dcs/conftest.py             shared fixtures (synthetic gold folder, synthetic table)
tests/dcs/test_dcs_*.py           one file per module (§6)
docs/classifier_progress.md       live status and decisions
```

`dcs` never imports `prepds`; constants it needs (state names, frame columns, workbook headers) are restated in `dcs` and checked against synthetic files. Torch is imported only inside `mlp.py`.

### 3.2 Settings

| Variable | Default | Used by |
|---|---|---|
| `DCS_ACCEPTED_DIR` | `accepted` | `featurize` (gold folder; per-video paths are rebuilt as `<DCS_ACCEPTED_DIR>/<video_id>/`, the index's stored paths are ignored, D-012) |
| `DCS_DB_PATH` | unset | `featurize` (NTT columns; unset → NTT absent, `has_ntt = 0`, ablation reported as unavailable) |
| `DCS_PROCESSED_DIR` | unset | `featurize` (optional tracker cross-check, D-003) |
| `DCS_OUTPUT_DIR` | `outputs/dcs` | everything written (git-ignored by default) |
| `DCS_TABLE` | `<DCS_OUTPUT_DIR>/training_table.parquet` | `audit`, `train` |
| `DCS_CONFIG` | unset | YAML overriding `config/default_training.yaml` |

Precedence as in `prepds`: environment > `.env` > defaults.

### 3.3 Data flow

```
accepted_index.parquet + <video_id>/{frames.parquet, segments.csv, manifest.json}   [+ workbook]
        │  gold.py (validate)  workbook.py (NTT)
        ▼
featurize.py ──► training_table.parquet + training_table_schema.json     (machine with the data)
        │  copy two files only
        ▼
trainset.py ─► folds.py ─► evaluate.py (preprocess.py + models.py / mlp.py per fold) ─► report.py
                                                                         └► artifact.py (final refit)
```

The schema JSON gives every column a role (`feature`, `label`, `group`, `id`, `meta`), a source and a kind (`count`, `duration`, `fraction`, `continuous`, `flag`); `log1p` applies to `count` and `duration` kinds.

## 4. Way of working

1. **Harness governance is on** (Q11 = a). Each unit (U1-U18 below) runs the `new-feature` workflow: `create_spec` → `validate_spec` → `assess_risk` → write the tests → run `pytest tests/dcs -q` and let the gate observe **RED** → implement → observe **GREEN** → review (`python-reviewer`, plus `mle-reviewer` for U9-U17) → `record_decision` → `trace_artifact`. Spec prefix `FISH`. Markdown is exempt from the gate.
2. **Tests first**, synthetic data only, after every change: `pytest tests/dcs -q`. Before asking for review: `pytest tests/ -q`.
3. **Keep `pytest tests/dcs -q` under 60 s**: the gate re-runs the test command with a 120 s limit and records a timeout as RED. Small synthetic sizes; MLP tests use few epochs.
4. Torch tests use `pytest.importorskip("torch")` (CI stays torch-free, Q8).
5. **Progress file** updated after every task; every choice the PRD leaves open gets a D-nnn entry.
6. No commit, push or PR unless the owner asks. No data or real names anywhere in the repo.
7. Playwright is not used: there is no UI.
8. **After every change that adds or alters a tool, command, setting, output column or error message** (a unit, or a fix between units), update the `dcs` part of [../instructions.md](../instructions.md) in the same branch: what it does, how to run it, what its tests check, what to do when it stops. A unit is not done while the instructions lag behind the code (D-041).
9. **After every change, sweep `src/dcs` for unused, legacy or duplicated code** (a function, constant or setting nothing reads; a second copy of a constant that `schema.py` already holds) and delete it in the same branch. Note what was removed in the progress file (D-041).
10. **One branch per unit**, `feature/dcs-u<n>-<name>`, stacked on the previous one. After a unit the owner verifies; the next unit starts only when the owner says so.

## 5. Phases and tasks

`[O]` = owner task (hardware, real data, review). Everything else is agent work on synthetic data.

### Phase 0. Environment (partly owner)
- T0.1 Baseline: `pytest tests/ -q` on `master` before any change; record result.
- T0.2 `[O]` GB10: Appendix B steps 1-5 (aarch64, DGX OS 7, `nvidia-smi`, torch CUDA build, GPU smoke test); record outputs in the progress file.
- T0.3 `[O]` GB10: `pip install -e ".[dev,train]"` succeeds on aarch64 (OpenCV headless, FastAPI, ...). Fallback if not: split `dcs` into its own `pyproject.toml` (decision then).
- T0.4 `[O]` Windows PC: same checks with the CUDA wheel for the 3060 Ti.
- Exit: hardware recorded; `torch.cuda.is_available()` true on at least one machine.

### Phase 0b. Upstream readiness `[O]` (outside this project)
`prepds catalog` → `prepds run` (classical tracker) → review flagged videos → **`prepds export-index` with the catalog present** (fills `date`, C4).

### Gate G1 `[O]` (real data; after Phase 1 code exists)
`dcs featurize` + `dcs audit` on the real accepted set (D-010). Stop rule (PRD §10): fewer than 3 non-vehicle compounds with ≥6 Accepted fish on ≥2 dates → wait for more videos. Q17 (unreviewed output) is decided here only if review lags. **Phases 1-5 proceed on synthetic data regardless of G1.**

### Phase 1. Featurize, training set and folds
- **U1 Scaffold and config.** T1.1 edit `pyproject.toml`, `.gitignore`, `.env.example`. T1.2 tests `test_dcs_config.py` (precedence, YAML override, unknown key, missing path → `ConfigError` with hint). T1.3 `config.py`, `cli.py`, `__main__.py`, `__init__.py`, `config/default_training.yaml`.
- **U2 Synthetic data.** T1.4 tests `test_dcs_synthetic.py`: output matches the frozen `prepds` schema (§1), deterministic per seed, planted compound signal plus a date effect, knobs for each edge case (missing file, `Undetermined`, mixed profiles, null date, missing column, wrong dtype, mixed fps, single-date compound, two-date compound, tiny class, constant feature). T1.5 `synthetic.py` + `dcs synth --out <dir>` (writes clearly synthetic names `COMPOUND_A`, ...; refuses a non-empty folder). Also writes a synthetic workbook with the 8 NTT headers.
- **U3 Gold reader.** T1.6 tests `test_dcs_gold.py`: EC-1, EC-10, EC-21, EC-22, EC-25, EC-27, EC-29, fps part of EC-26. T1.7 `gold.py` (`--profile <version>` filters a mixed set and the audit counts what was filtered out).
- **U4 Workbook.** T1.8 tests `test_dcs_workbook.py`: header whitespace stripped, key `<Sex>_<Subject:04d>`, exact duplicate rows collapsed, conflicting duplicates → error, compound/date disagreeing with the index → error, fish absent → `has_ntt = 0` (EC-28, EC-2 flag part). T1.9 `workbook.py`.
- **U5 Featurize.** T1.10 tests `test_dcs_featurize.py`: per-state share, bouts, mean bout, latency on hand-built segments; transition counts; kinematics on detected frames only; EC-3; whole fish with no detected frames dropped and counted; EC-11 flags; schema JSON roles, with depth features tagged as group `depth` (D-015). T1.11 `featurize.py` + `dcs featurize`.
- **U6 Training set.** T1.12 tests `test_dcs_trainset.py`: EC-4, EC-7, EC-8, EC-12 (forbidden list = PRD §5.3 plus `video_id`, `provenance`, `edited`, `review_flags`, `processed_at`, `*_path`), EC-17, EC-20, EC-23, "very small" flag for classes of exactly `min_class_size`, depth group excluded unless `use_depth` (D-015). T1.13 `trainset.py`.
- **U7 Folds.** T1.14 tests `test_dcs_folds.py`: EC-5, EC-6, EC-13 (200 seeds), EC-14 (folds part), fewer dates than K → K reduced with a note. T1.15 `folds.py`.
- **U8 Audit.** T1.16 tests `test_dcs_audit.py`: AC-1 tables (counts per filter step, per class, per date, compound × date, missing values, duration/detected outliers, manual-frame share per compound, confound statistics of PRD §2.3, evaluable classes, dropped classes and states, fps uniformity, G1 stop-rule verdict, share of fish on dates with ≥2 classes (D-016)); EC-24; EC-31 framing check (per date: distribution of each fish's 1st and 99th depth percentiles; a shift or a change in range is flagged and recommends keeping depth off and flagging pixel speeds). T1.17 `audit.py` + `dcs audit`.
- Exit: tests green; `dcs synth` → `dcs featurize` → `dcs audit` works end to end on synthetic data.

### Phase 2. Baselines
- **U9 Preprocessing.** T2.1 tests `test_dcs_preprocess.py`: EC-9 (extreme test fold does not move the fitted statistics), EC-2 (training-fold median), `log1p` only on count/duration kinds, JSON round trip. T2.2 `preprocess.py`.
- **U10 Baselines.** T2.3 tests `test_dcs_models.py`: majority, date-only (D-011), logreg (L2, class-weighted), RF (balanced), HGB; one interface, deterministic per seed. T2.4 `models.py`.
- **U11 Evaluation.** T2.5 tests `test_dcs_evaluate.py`: every model sees identical folds; scheme-A score excludes pinned fish; metrics of PRD §6.7 (no plain accuracy) on the pooled out-of-fold predictions of each repeat (D-017); within-date permutation; repeat spread; decision rule; vehicle-vs-drug AUROC; **leakage canary**: on a synthetic set where the label depends only on the date, scheme A is near chance and the permutation diagnostic catches it. T2.6 `evaluate.py`.
- **U12 Report and run folder.** T2.7 tests `test_dcs_report.py`, `test_dcs_cli.py` (small end-to-end `dcs train --stage compound --models majority,logreg`): run folder layout of PRD §7.4, `run_info.json` fields, baselines next to every model, date effect = B − A. T2.8 `report.py`, `train.py`.
- Exit: AC-3, AC-5 on synthetic data.

### Gate G2 `[O]`
Owner reviews the synthetic baseline report (layout, wording, caveats). Q9 revisited with the advisor.

### Phase 3. MLP and ablations
- **U13 MLP.** T3.1 tests `test_dcs_mlp.py` (torch only): architecture and config from `training.mlp`, early stopping on an inner split of the training fold, class-weighted loss, 5 seeds with mean and spread, EC-14 (identical CPU metrics), EC-15 (`--device cuda` without GPU → clear error; `auto` → CPU). `test_dcs_no_torch.py`: EC-16 (torch hidden → baselines run, MLP skipped with a message). T3.2 `mlp.py`.
- **U14 Ablations.** T3.3 tests: NTT on/off, demographics on/off, depth on/off (D-015), FR-9 as a raw-vs-normalized pair on the same non-vehicle classes (D-008) with the global-median fallback (<2 vehicle fish on a date); NTT keep rule. T3.4 extend `train.py`, `preprocess.py`.
- T3.5 NFR-4: time a full synthetic run (~300 fish, all models, all ablations) on CPU; record; tune permutation count if needed (D-entry).
- Exit: AC-4, AC-6, AC-7.

### Phase 4. Stage 2 (exploratory)
- **U15 Dose models.** T4.1 tests `test_dcs_stage2.py`: one model per compound with ≥2 eligible doses; vehicle skipped; scheme B + date-only + FR-9 variant; every result carries the "date-confounded" caveat. T4.2 extend `train.py`.

### Phase 5. Save/load and runbook
- **U16 Artifact.** T5.1 tests `test_dcs_artifact.py`: `model/` contents (PRD §7.4 plus `reference_predictions.csv`, D-006); EC-18 reload in a **fresh process** (subprocess) reproduces it; version mismatch → warning. T5.2 `artifact.py`.
- **U17 Predict.** T5.3 tests: `dcs featurize --videos <dir>` builds rows from per-video folders without an index (labels optional; rows from non-Accepted videos marked unreviewed, D-018); `dcs predict --model <run> --input <table>` uses only saved preprocessing; missing or extra columns → error naming them. T5.4 extend `cli.py`, `featurize.py`, `artifact.py`.
- **U18 Privacy.** T5.5 `test_dcs_privacy.py` (D-007): `git ls-files` has no `.parquet`, `.csv`, `.xlsx`, `.pt`, `.joblib`, `.env` outside the named fixtures.
- T5.6 Draft the upstream request for `prepds` (record tracker and resolution in `manifest.json` and the index) for the owner to file.
- T5.7 `[O]` AC-9 across machines (GB10 → PC) and AC-10 runbook walkthrough, first with `dcs synth` data; EC-19 on Windows and aarch64.
- Exit: AC-9 (synthetic, locally), AC-10.

### Gate G4 `[O]`
First real-data run review with the owner and advisor (needs Accepted videos): stop, improve, or add FR-10.

### Phase 6. Optional 1D-CNN (FR-10, only after G4)
`featurize --sequences`, CNN with time-shuffled control, same folds; kept only if it beats the best tabular model on scheme A. EC-30: sequences padded to the longest recording with a mask, masked global pooling.

### Phase 7. Wrap-up
Plan and progress reflect what was built; README pointer to `dcs`; AC-11.

## 6. Edge-case coverage (every PRD §9.2 row is a task)

| EC | Task | Test file |
|---|---|---|
| EC-1 missing/unreadable files | T1.6 | `test_dcs_gold.py` |
| EC-2 NTT missing | T1.8 (flag), T2.1 (fold median) | `test_dcs_workbook.py`, `test_dcs_preprocess.py` |
| EC-3 state never occurred | T1.10 | `test_dcs_featurize.py` |
| EC-4 small class | T1.12 | `test_dcs_trainset.py` |
| EC-5 single-date compound | T1.14 | `test_dcs_folds.py` |
| EC-6 two-date compound | T1.14 | `test_dcs_folds.py` |
| EC-7 dose string variants | T1.12 | `test_dcs_trainset.py` |
| EC-8 compound case/spaces | T1.12 | `test_dcs_trainset.py` |
| EC-9 fit on training fold only | T2.1 | `test_dcs_preprocess.py` |
| EC-10 duplicate subject | T1.6 | `test_dcs_gold.py` |
| EC-11 low detected share / odd duration | T1.10, T1.16 | `test_dcs_featurize.py`, `test_dcs_audit.py` |
| EC-12 forbidden columns | T1.12 | `test_dcs_trainset.py` |
| EC-13 fish/date in two folds | T1.14 | `test_dcs_folds.py` |
| EC-14 same seed | T1.14, T3.1 | `test_dcs_folds.py`, `test_dcs_mlp.py` |
| EC-15 no GPU | T3.1 | `test_dcs_mlp.py` |
| EC-16 no torch | T3.1 | `test_dcs_no_torch.py` |
| EC-17 too few rows / one class | T1.12, T2.3 | `test_dcs_trainset.py`, `test_dcs_models.py` |
| EC-18 reload in fresh process | T5.1 | `test_dcs_artifact.py` |
| EC-19 Windows / aarch64 | T0.3, T5.7 `[O]` | manual, recorded in progress |
| EC-20 constant / all-NaN feature | T1.12 | `test_dcs_trainset.py` |
| EC-21 mixed trackers/profiles | T1.6 | `test_dcs_gold.py` |
| EC-22 `Undetermined` in gold | T1.6 | `test_dcs_gold.py` |
| EC-23 rare state | T1.12 | `test_dcs_trainset.py` |
| EC-24 compound with 0 fish / few dates | T1.16 | `test_dcs_audit.py` |
| EC-25 missing column / dtype | T1.6 | `test_dcs_gold.py` |
| EC-26 mixed fps / resolution | T1.6, T1.16 (fps only, C2) | `test_dcs_gold.py`, `test_dcs_audit.py` |
| EC-27 null `date` (new) | T1.6 | `test_dcs_gold.py` |
| EC-28 workbook mismatch (new) | T1.8 | `test_dcs_workbook.py` |
| EC-29 tracker evidence disagrees (new) | T1.6 | `test_dcs_gold.py` |
| EC-30 sequences of different lengths (new, Phase 6) | Phase 6 | `test_dcs_sequences.py` |
| EC-31 camera framing differs by date (new) | T1.16 | `test_dcs_audit.py` |

## 7. Risks specific to this plan

| Risk | Mitigation |
|---|---|
| Gate timeout (120 s) records a spurious RED | Keep `tests/dcs` fast (§4.3); mark nothing slow in the default run |
| Harness risk keywords (e.g. "session", "token") trigger approval prompts | Read the rationale; approve at the terminal when harmless |
| Synthetic data hides real-data problems | Edge-case knobs in U2; G1 audit on real data before any real training |
| Torch on aarch64 / Blackwell | T0.2 smoke test; PC fallback; baselines are torch-free |
| PRD change list not approved | Items are isolated (one function or one test each); revert to PRD text if rejected |

## 8. Definition of Done

**Infrastructure done (this phase, synthetic data):** U1-U18 green under the harness; every EC row of §6 has a passing test (EC-19 recorded manually); AC-2, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8 pass on synthetic data; AC-9 passes locally; G2 held; progress file current (AC-11).

**Project done (PRD §1.6):** all of PRD §9.3 on real Accepted data, including AC-1 on the real set, AC-9 across machines and AC-10 by the owner; G4 review held; every P0 requirement (FR-1 to FR-6) marked verified in the progress file.

## Appendix A. External review prompt (Q7)

For the owner to paste into another LLM tool together with **this plan only**, or this plan and the PRD with its header table removed (the header names people). Never paste data, outputs, the `prepds` progress log or the scope document.

```
You are reviewing an implementation plan for a small machine-learning project. Two
documents follow: the plan and (optionally) its PRD. All names are placeholders.

Context: zebrafish behavior videos are turned upstream into human-reviewed per-frame
behavior labels. This project reduces them to one row per fish and trains classifiers
for the compound (about 15 classes, a few hundred fish at most) and, exploratorily,
the dose. Fish were recorded on about 40 experiment dates, and dose is fully
confounded with date. Validation is date-held-out group K-fold with a random-split
twin, a date-only baseline and a within-date label permutation test.

Please look for holes, not style:
1. Leakage: any path by which the date, the labeling process or the label itself can
   reach the features or the preprocessing (including the vehicle-normalization
   ablation and the date-only baseline definitions).
2. Validation design: is the fold scheme, the pinning of single-date compounds, the
   decision rule for "useful" and the permutation test statistically sound at this
   sample size? What would you change?
3. Missing edge cases or tests, given the edge-case table.
4. Steps that cannot be verified as written, or tasks whose order is wrong.
5. Anything over-engineered for a few hundred rows.

Answer as a numbered list of findings, each with: section reference, the problem,
why it matters, and a concrete fix. Say "no finding" for a point if you have none.
```
