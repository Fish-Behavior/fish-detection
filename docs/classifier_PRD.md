# PRD: Compound and Dose Classifier (Training Phase)

| | |
|---|---|
| **Status** | Draft v0.3 for verification (nothing implemented). Rebased on the cleaned-up `master`, where `prepds` is the root package and `fishbehavior` is legacy |
| **Created** | 2026-10-02 |
| **Project** | Zebrafish Drug-Response Detection System: drug-classification phase. Consumes the gold dataset of the `prepds` pipeline (repository root, `master`) |
| **Advisor / collaborator** | Dr. Ashish Kharel / Dr. Scott Hall |
| **Template** | Adapted from the reference PRD templates (see [process/PRD_TEMPLATE.md](process/PRD_TEMPLATE.md)); market, pricing and go-to-market sections removed, data, evaluation and environment sections added |
| **Privacy rule** | This file and every file this project adds to `docs/` use placeholders (`COMPOUND_A`, `F_0042`). Never write real compound names, subject IDs, dates, data files or local paths here. The repository's README already forbids committing restricted data; this PRD applies the same rule to names and dates voluntarily. |

Section numbers are stable: other documents (the plan, the progress log) refer to them.

**Change log.** v0.3: file renamed `docs/classifier_PRD.md` (the root `docs/PRD.md` belongs to `prepds`); `fishbehavior` is legacy and gone from `master`; code location decided (§7.1); answers Q12-Q17 recorded; CI facts of the new `master` applied. v0.1: first draft from the earlier export. v0.2: data source rebased on `preprocessing_dataset_system` (read from `master`): §1.1-1.5, §2.2-2.3, §3, FR-1, §5 rewritten; §6.11, §7.1-7.5, §8.3, §9.2-9.3, §10-12 and Appendix B updated; Q12-Q16 added.

---

## 1. Executive summary

### 1.1 Problem
`prepds` turns each zebrafish video and its workbook row into a human-reviewed, per-frame ethogram (the "gold" dataset). Nothing yet learns from that dataset to say which compound, and which dose, a fish received. Without it the project cannot estimate drug identity (scope doc §3, requirement 2).

### 1.2 Proposed solution
Add a new, separate classification package (working name `dcs`, Q12). Its `featurize` command turns the accepted gold dataset into one row per fish; its `train` command reads that table, trains and fairly evaluates several classifiers, and saves a retrievable model. Compound is predicted first (Stage 1), then dose within the compound (Stage 2). Simple models are the baselines, and a small PyTorch MLP has to beat them to be kept. A 1D-CNN on the per-second sequences is a later, optional experiment.

### 1.3 Expected impact
- A first honest measure of how much drug identity the behavior data carries.
- A reproducible, privacy-safe training workflow that runs on the lab Linux box and, as a fallback, on the home GPU.
- A reusable engineering process ([process/HARNESS_PROCESS.md](process/HARNESS_PROCESS.md)) for later projects.

### 1.4 Resources
- **Primary compute:** Dell Pro Max with GB10 (DGX OS 7, 128 GB unified memory, 4 TB SSD), reached over SSH from PuTTY.
- **Fallback compute:** home PC (Windows 11) with an RTX 3060 Ti (8 GB).
- **Data:** the accepted gold dataset of `prepds` (`accepted_index.parquet` plus each video's `frames.parquet`, `segments.csv`, `manifest.json`), reduced by `featurize` to one table with one row per fish (restricted; never leaves approved storage).
- **People:** the project owner implements with Claude Code; advisor and collaborator review.

### 1.5 Success metrics
No numeric accuracy target is promised: with at most a few hundred fish (only human-Accepted videos count, Q14) and a heavy date confound (§2.3) the honest outcome may be modest. Success is defined by process and evidence:

| # | Metric | Target |
|---|---|---|
| S1 | Every model is scored next to the majority-class baseline and a date-only baseline | Always reported |
| S2 | Stage 1 balanced accuracy under date-held-out validation, with its random-split twin | Reported with the gap; "useful" only if it beats both baselines by more than the repeat-to-repeat spread (§6.6) |
| S3 | Runs reproduce from a seed | Same seed, same data, same metrics (§4.3 NFR-1) |
| S4 | A trained model can be copied off the box and reloaded to give identical predictions | Test passes (AC-9) |
| S5 | No restricted data or real names in Git | CI privacy job green |

### 1.6 Definition of Done
All items of the acceptance checklist (§9.3) pass, the first-run report has been reviewed by the owner (Gate G4, §10), and the progress file shows every P0 requirement as verified.

---

## 2. Problem definition

### 2.1 Users
- **Owner (researcher/student):** runs training, reads the report, retrieves models.
- **Advisor and pharmacy collaborator:** read the report and judge whether the evidence is credible.

### 2.2 Context
The project aims to estimate "which compound and dose produced this behavior".

**Upstream system.** `prepds` (the pipeline at the repository root; its own PRD is `docs/PRD.md`, its log `docs/progress.md`) tracks the fish in each of the 353 trial videos, derives per-frame features, labels every frame with one of seven states (the five ethogram states, `Dead`, and the internal `Undetermined`), lets a human reviewer correct the result, and commits only **Accepted** videos to a versioned gold dataset. Its PRD (§1) names this project as the later, separate drug-classification work, and states that `accepted_index.parquet` is the single file that work reads (its §5.4). The two packages are independent (its Clarification C10); this PRD keeps that rule: **the classifier is a read-only consumer and never imports or modifies `prepds`.**

**What `prepds` produces per Accepted video** (its PRD §5.4): `frames.parquet` (one row per frame: `t_sec`, `x`, `y`, `orientation_deg`, `depth_from_surface`, `detected`, `velocity`, `acceleration`, `angular_velocity`, `meander`, `is_immobile`, `state`, `source` = auto or manual, `confidence`), `segments.csv` (run-length encoding of the states), `strip.png`, `manifest.json` (review status, calibration profile version, pipeline version), plus the trial fields (`compound`, `concentration_mM`, `date`, `strain`, `sex`, `age`, `agent_exposure_min`) in the index.

**Status today (Q14, answered).** No video has been processed or Accepted on the owner's side yet. The pipeline will be run on the owner's Windows 11 PC (the `prepds` log reports about 58 minutes for 328 videos with 22 workers; a Mac is not suitable). The pipeline flags videos that need a human look (review flags such as a fish that never moves until the end), and a reviewer overrides labels by hand in its review app. Until videos are Accepted, the gold set is empty, so real training is **blocked on upstream work**; the infrastructure is built and tested on synthetic data meanwhile (§9). The training source is the **classical-tracker** output (Q13, answered). Reading unreviewed output is not preferred (Q17); it would be a clearly labeled fallback decided at gate G1.

**The older pipeline.** The earlier `fishbehavior` package is **legacy**: it is gone from `master` (still on branch `demo-pipeline`). It exported a similar one-row-per-fish workbook, and a sample of it (326 fish, labels not human-reviewed) is the source of the numbers in §2.3. That sample is **reference only** (Q15): it is used to find structural problems such as the date confound, never for reported results. The audit recomputes everything on the real data.

**What this phase is, and is not.** The `prepds` labeling pipeline uses a fine-tuned computer-vision model and has **not yet been validated with the pharmacy department**. Everything built here is therefore **training infrastructure**: it must be ready and tested so that real data can be plugged in later. No statement about drug identity is made from results on the current, unvalidated labels.

### 2.3 What the data looks like (preliminary)
Facts below come from the earlier sample workbook (352 trial rows, 326 with video-derived columns) and are aggregated, with no names. `dcs audit` (FR-2) must reproduce them on the **accepted** set, which can be a smaller subset.

| Fact | Value (earlier sample) | Why it matters |
|---|---|---|
| Fish available | 326 of 352 in the sample; **the real ceiling is the number of ACCEPTED videos (unknown, Q14)** | Training-set size |
| Rows with workbook (NTT) movement columns | 236 of the 326; about 90 lack them | NTT is optional and partly missing |
| Distinct experiment dates | 41 (about 8 fish per date) | Day-to-day batch effects are possible |
| Compounds (incl. vehicle) | 17, two of them combinations | Stage 1 has about 15 classes after the class filter (§5.6) |
| Vehicle fish | 69, present on 39 of 41 dates | Vehicle acts as a per-day control (§6.8) |
| Compound+dose classes | 34; 27 were run on a single date; 6 have fewer than 6 fish | Dose is confounded with date |
| Compounds seen on one date only | 3 (two are dropped by the class filter, §5.6) | The remaining one cannot be validated on an unseen date |
| Doses of one compound sharing a date | None | Dose cannot be separated from date (§6.5) |
| LORR | 6 fish, 83 s in the earlier auto labels | In `prepds` LORR and Dead come only from reviewers, so the accepted set may differ; state support is re-checked (§5.3) |
| Recordings outside 1140-1260 s | 18 | Audit; kept and flagged |
| Protocol columns (`agent_exposure_min`, `uv_min`, `ntt_min`) | Vary with compound | Possible label proxy; excluded (§5.3) |

**Facts specific to the gold dataset** (from the `prepds` README and progress log; the audit confirms them):

| Fact | Why it matters |
|---|---|
| Frame labels come from speed thresholds calibrated on **7 compound/dose groups** (5 of 17 compounds); other compounds are extrapolated, then reviewer-corrected | Label quality can differ by compound |
| Only human-Accepted videos enter the gold set | Which videos get reviewed first can correlate with compound or date (selection bias) |
| Two trackers exist (classical, and a fine-tuned detector with its own calibration profile); speeds are in **pixels per second**, not body lengths, and differ systematically between trackers | Never mix trackers or profiles in one training set (§5.2) |
| Surface breach and Dead thresholds were placeholders; Listing/LORR is a manual label | State features can be rare or reviewer-dependent |

**Consequence.** A model can score well by recognizing the experiment day, or the labeling process, rather than the drug. Every design choice in §6 exists to measure and limit that.

### 2.4 Cost of not solving
No quantitative evidence for drug-identity estimation, so the later anomaly-detection and reporting modules (scope doc §4.3-4.4) would have nothing to build on.

---

## 3. Scope

### 3.1 In scope
| ID | Item | Priority |
|---|---|---|
| FR-1 | `featurize`: gold dataset → one row per fish, plus a schema file | P0 |
| FR-2 | `train --audit`: data audit report, no training | P0 |
| FR-3 | Stage 1: compound classifier (baselines + MLP) | P0 |
| FR-4 | Leakage-aware validation and diagnostics (§6) | P0 |
| FR-5 | Report: metrics, confusion matrix, per-compound results, ablations | P0 |
| FR-6 | Save and reload a model artifact; retrieval runbook | P0 |
| FR-7 | Stage 2: dose within compound, **exploratory** | P1 |
| FR-8 | `predict`: score new rows with a saved model | P1 |
| FR-9 | Per-date vehicle normalization ablation | P1 |
| FR-10 | 1D-CNN on per-second sequences | P2 (optional, after G4) |
| FR-11 | Image branch / video models | Out of scope |

### 3.2 Out of scope
- Anomaly detection, natural-language reporting, frontend/backend, live camera (scope doc §4.3-4.5, Phase 2).
- Detecting LORR or Dead from video (reviewers label them in `prepds`).
- Pre-trained models (decision in §6.10).
- Any change to `prepds` or its outputs (read-only consumer).
- Hyperparameter search on test folds.

### 3.3 MVP
FR-1 to FR-6 on one machine, producing one reviewed report and one loadable model.

### 3.4 Learning goals
1. How much compound identity survives once the date effect is controlled?
2. Does the MLP beat logistic regression and tree models on this table size?
3. Do the NTT columns add signal?

---

## 4. Requirements

### 4.1 User stories
**US-1.** As the owner, I want to run one command on the GB10 box and get a report, so that I can judge the evidence without writing code.
- [ ] `python -m dcs train` runs end to end from the training table.
- [ ] The report states the baselines next to every model.

**US-2.** As the advisor, I want to see whether the score depends on the experiment date, so that I can trust or reject it.
- [ ] The report shows date-held-out and random-split scores and their gap.
- [ ] A date-only baseline is shown.

**US-3.** As the owner, I want to copy a trained model off the box and reload it elsewhere, so that I can use it later.
- [ ] The run folder is self-contained (§7.4).
- [ ] A documented command reloads it and reproduces the saved predictions.

### 4.2 Functional requirements (details)
- **FR-1** `featurize` reads `accepted_index.parquet` and each accepted video's `segments.csv` and `frames.parquet` (read-only), computes the per-fish features of §5.3, attaches the labels of §5.6, and writes `training_table.parquet` plus `training_table_schema.json` (every column, its source, and whether it is a feature, a label, a group or an identifier). `train` reads only that table.
- **FR-2** prints and saves: row counts per filter step, per-class and per-date tables, missing-value counts, duration/tracked outliers, the confound statistics of §2.3, and which classes are evaluable (§6.4).
- **FR-3** trains: majority, date-only, logistic regression, random forest, histogram gradient boosting, MLP (§6.3), each under both validation schemes.
- **FR-4** implements §6.4-6.6 and the leakage diagnostics of §6.6.
- **FR-5** writes `report.md`, `metrics.csv`, `predictions.csv` (out-of-fold), confusion matrices and plots.
- **FR-6** saves the final model fitted on all eligible fish plus everything needed to reload it (§7.4).
- **FR-7** trains one small dose model per compound with at least two eligible doses; every result carries a "date-confounded" caveat (§6.5).
- **FR-8** `predict --model <run folder> --input <csv>` writes class probabilities per row, using only the saved preprocessing.
- **FR-9** subtracts the same-date vehicle median (robust scaling) before modeling; falls back to the global vehicle median when a date has fewer than 2 vehicle fish.

### 4.3 Non-functional requirements
- **NFR-1 Reproducibility.** One seed controls splits, initialization and shuffling. Same seed and data give identical splits, and identical metrics for CPU runs (GPU runs may differ in the last digits; the report says so). Versions of Python, torch, CUDA, scikit-learn and the git commit are saved in `run_info.json`.
- **NFR-2 Privacy.** No data, outputs, real names or paths in Git or in `docs/`. The repository's ignore rules already exclude `*.parquet`, `*.csv`, `*.xlsx`, `data/`, `outputs/` and `accepted/`; nothing the project writes may be force-added. Models and reports are written only under `DCS_OUTPUT_DIR`.
- **NFR-3 Portability.** Python 3.11 or newer (the `prepds` baseline); CI runs Linux with Python 3.11. Code must also run on Windows 11 and on Linux aarch64 (the GB10), checked in Phase 0. Torch is an optional extra; everything except the MLP must run without it. The GB10 is ARM64 (aarch64), so no dependency may be x86-only.
- **NFR-4 Performance.** The full baseline + MLP comparison on a few hundred rows finishes in minutes on one GPU and acceptable time on CPU. No GPU is required for baselines.
- **NFR-5 Robustness.** Clear `ConfigError` messages (as the other steps do) for missing files, missing columns, and too few rows.
- **NFR-6 Code style.** Match `prepds`: own package under `src/`, one module per step, plain functions, settings in a default YAML under `config/`, env vars with their own prefix (`DCS_`), fitted values only in the output folder. `dcs` never imports `prepds`; it reads only the files `prepds` writes.

---

## 5. Data specification

### 5.1 Input and the two-step flow
- **Source (read-only):** `<DCS_ACCEPTED_DIR>/accepted_index.parquet` and, per Accepted video, `frames.parquet`, `segments.csv`, `manifest.json` (`prepds` PRD §5.4).
- **Step 1, `featurize`:** runs on the machine that holds the accepted data; writes `<DCS_OUTPUT_DIR>/training_table.parquet` (one row per fish) and `training_table_schema.json`.
- **Step 2, `train`:** runs anywhere (the GB10); reads only the training table.

Reasons: only one small file has to be moved to the GPU box; fewer restricted files leave their home; the GB10 needs neither OpenCV nor `prepds`.

### 5.2 Eligible rows
1. Review status ACCEPTED (by construction of the index).
2. **One tracker and one calibration profile per training set.** `manifest.json` records `calibration_profile_version` but **no tracker field**; the tracker is read from `run_report.json` (a `detections.parquet` next to the frames also marks a model-tracker run). The audit checks both. A set that mixes them is an error with a clear message (EC-21); which set to use is Q13.
3. One row per subject (`prepds` already fails duplicate `sex_subject` ids; an index with a repeated subject is an error, EC-10).
4. Fish with a low fraction of detected frames or an odd video duration are **kept but flagged** in the audit (decided, Q6).
5. Small classes are removed afterwards by the class filter (§5.6).

### 5.3 Features (computed by `featurize`; exact list frozen from the real schema in Phase 0)
**From `segments.csv`** (states: Controlled Swim, Erratic, Freezing/Drift, Listing/LORR, Surface Breach, Dead; `Undetermined` cannot occur in an Accepted video and is an error if found, EC-22):
- per state: share of the recording, number of bouts, mean bout length, latency to the first bout (state never shown → rules of §5.4);
- transition counts between states.

**From `frames.parquet`** (detected frames only): mean, median and coefficient of variation of `velocity`; mean absolute `acceleration` and `angular_velocity`; mean `meander`; share of `is_immobile`; mean, minimum and percentiles of `depth_from_surface`; share of detected frames. Note: in the code `depth_from_surface` is the fish's vertical distance from the **top of the frame** in pixels (the waterline is advisory and not used), so it is comparable only if the camera framing is the same for all videos; the audit checks this.

**Units caution:** these kinematic values are in pixels and seconds, not body lengths, and the two trackers measure speed differently (box centre vs blob centroid). They are comparable only within one tracker/profile and across recordings of the same camera setup. The audit reports whether resolution and frame rate are uniform; if not, pixel-based features are flagged.

**State support rule (decided, "drop everything that lacks data"):** a state's features are dropped when fewer than `training.min_state_fish` fish (default 10, Q16) show that state at all. In the earlier sample this removes LORR.

**Optional (ablation, default on):** the eight NTT columns (`tdm_*`, `velocity_*`, `time_top_s`, `time_bottom_s`) plus `has_ntt`, if present in the index or the catalog (`trials_catalog.parquet`); confirmed in Phase 0. Kept only if they help (§6.8).

**Excluded, with reasons:**

| Columns | Reason |
|---|---|
| `compound`, `concentration_mM` | They are the label |
| `subject_id`, `date`, paths, reviewer, review timestamps, `edit_count`, share of manual frames, `calibration_profile_version`, `pipeline_version` | Identifiers or labeling-process fields. `date` is used only to build groups. The edit and manual share can differ by compound because of reviewer attention, so it is reported in the audit but never used as a feature |
| `agent_exposure_min`, `ntt_min`, `uv_min`, `h2o_*`, `brain_tissue` | Protocol or measurement fields that differ by compound or batch and could leak the treatment |
| `video_duration_s`, `video_fps`, resolution | Camera and recording artifacts |
| `age`, `sex`, `strain` | Off by default; `--with-demographics` is an ablation to measure bias |

A unit test asserts that none of the excluded columns reaches the feature matrix (EC-12).

### 5.4 Missing values
| Case | Rule |
|---|---|
| State never occurred (no bouts) | Bout count 0; mean bout → 0; latency → recording length (a "never" is the latest possible) |
| No detected frames for a kinematic feature | Row flagged; if the whole fish has none it is dropped and counted |
| NTT columns missing | Median of the training fold plus `has_ntt = 0` |
| Anything else missing | Row dropped and counted in the audit |

### 5.5 Transforms
`log1p` on counts and duration-like features; standardize with statistics **fitted on the training fold only**. Transform parameters are saved with the model.

### 5.6 Labels and filtering
- **Compound label:** `compound`, trimmed and case-normalized (combination treatments are their own classes).
- **Dose label:** `concentration_mM` kept as the string it is (combination doses such as two values joined by a plus sign are not numbers), spaces removed so spacing variants match.
- **Class filter (decided, Q4):** drop every class that lacks data: `training.min_class_size = 6` (configurable). Stage 1 drops compounds with fewer than 6 Accepted fish; Stage 2 drops compound+dose classes under 6. In the earlier sample that was two compounds and six dose classes. The audit lists everything dropped. Vehicle has a single dose and is skipped in Stage 2.
- Classes of exactly 6 fish are kept and flagged "very small" in every report.

---

## 6. Modeling and evaluation

### 6.1 Stages
- **Stage 1:** compound (about 15 classes including vehicle).
- **Stage 2:** dose within compound, only for compounds with at least two eligible doses (about 11 in the sample). Exploratory.

### 6.2 Baselines (all P0)
1. **Majority class** (vehicle: about 21%).
2. **Date-only:** predict the label from the experiment date alone. It measures how much the day explains and acts as the leakage ceiling.
3. **Logistic regression** (L2, class-weighted).
4. **Random forest** (class-balanced).
5. **Histogram gradient boosting** (scikit-learn; no extra dependency, so it also runs on ARM64 and in CI).

### 6.3 PyTorch MLP
Input = the feature vector; 2 hidden layers (128, 64), ReLU, dropout 0.3, AdamW with weight decay, class-weighted cross-entropy, up to 300 epochs with early stopping on an inner validation split of the training fold, batch size 32, 5 seeds (the report shows mean and spread). All values live in `training.mlp` in the config. The MLP is kept only if it matches or beats the best simple baseline.

### 6.4 Validation design (the key decision)
Two schemes, always run together:

| Scheme | Purpose | How |
|---|---|---|
| **A. Date-held-out** (primary) | Does it generalize to an unseen day? | Stratified group K-fold (K=5) with the experiment `date` as the group, repeated with several seeds |
| **B. Random stratified** (reference) | Leakage twin | Stratified K-fold (K=5), same repeats |

Rules for scheme A:
- A compound seen on **one date only** cannot be held out. Its fish are pinned to the training side in scheme A, so they still count as possible predictions; they are excluded from the scheme-A *score* and flagged "date-confounded". They appear only in scheme B.
- Folds are recorded in `folds.csv` and reused by every model, so models are compared on identical splits.
- A fish is never in two folds; a date is never in two folds in scheme A.
- The final scaler/imputer is refit inside each training fold (§9.2 EC-9).

**The gap A vs B is reported as the "date effect".**

### 6.5 Stage 2 and the dose-date confound
No compound has two doses that share an experiment date (§2.3). Holding out a date therefore removes a whole dose class, so scheme A is impossible and scheme B can be won by recognizing the date. Consequences, written into the report:
- Stage 2 is **exploratory (P1)**. No claim about dose recognition may be drawn from scheme-B results.
- The date-only baseline is shown for every stage-2 model.
- The only informative test is FR-9 (vehicle-normalized features): if the model still separates doses after the date's own vehicle level is removed, there is some evidence.

### 6.6 Diagnostics and decision rule
1. **Date-only baseline** score (leakage ceiling).
2. **Within-date label permutation:** shuffle compound labels among fish of the same date and re-run scheme B. If the real score is not clearly above the permuted score, the model has learned nothing beyond which date a label came from.
3. **Repeat spread:** the standard deviation across CV repeats and seeds; "better" means better by more than this.
4. **Decision rule for "useful":** scheme-A balanced accuracy of a model exceeds both the majority baseline and the date-only baseline by more than the repeat spread, and exceeds the permuted-label result.
5. **Vehicle vs drug** (binary) reported separately with AUROC and balanced accuracy, scheme A.

### 6.7 Metrics
Balanced accuracy and macro-F1 (primary); per-compound precision, recall and F1; confusion matrix; log-loss (probability quality); top-3 accuracy. Accuracy alone is never reported.

### 6.8 Ablations (each reported, same folds)
- NTT on/off.
- Demographics on/off.
- Vehicle-normalized features on/off (FR-9).
- Rule: NTT is kept only if scheme-A balanced accuracy improves by more than the repeat spread.

### 6.9 Final model
After evaluation, the chosen model is refit on all eligible fish and saved. Its expected performance is the cross-validation estimate, not a score on the training data.

### 6.10 Pre-trained models (Hugging Face)
Decision: **not used** in this phase. No pre-trained model fits tabular behavior features. `prepds` already fine-tunes a detector for tracking (its Phase 15; its optional `ml` extra); that is upstream and unrelated to this classifier. Time-series foundation models are a possible later experiment and would need an offline download plan for the Linux box.

### 6.11 Optional 1D-CNN (FR-10, after Gate G4)
- **Input:** each Accepted video's `frames.parquet`, resampled to 1 s (about 1200 steps × about 8 channels per fish: `velocity`, `acceleration`, `angular_velocity`, `meander`, `depth_from_surface`, `is_immobile`, `detected`, plus the reviewed state as a one-hot); `log1p` on heavy-tailed channels; undetected seconds masked. Same tracker/profile rule as §5.2.
- **Model:** small dilated 1D-CNN with global pooling and a compound head.
- **Evaluation:** same folds and metrics as §6.4, plus a time-shuffled control. It is added only if it beats the best tabular model on scheme A.
- **Data:** the frame files stay on the machine that has them; a compact per-second array file is built by `featurize --sequences` and copied instead.

---

## 7. Technical specification

### 7.1 Code layout (decided default, Q12)
`fishbehavior` is legacy, so the classifier is a **second package inside the same repository and the same `pyproject.toml`**, importing nothing from `prepds`:
```
src/dcs/__main__.py, cli.py, config.py
src/dcs/featurize.py   gold dataset -> training table (reads prepds output files, pure functions)
src/dcs/trainset.py    load/filter table, labels, folds, audit
src/dcs/models.py      baselines + PyTorch MLP (torch imported lazily)
src/dcs/train.py       both stages, diagnostics, report, save/load, predict
config/default_training.yaml        the `training:` defaults (next to the `prepds` thresholds)
tests/dcs/                          synthetic tables only (a subfolder: no clash with prepds test names)
docs/classifier_PRD.md, docs/classifier_progress.md, docs/plans/classifier_plan.md
.env.example                        gains the DCS_* variables
```
`pyproject.toml` gains `[project.scripts] dcs = "dcs.cli:main"` and the extra `train = ["torch>=2.4"]`; `packages.find` already scans `src/` and finds both packages. `train` does not need `featurize`'s inputs. Installing the root project on the GB10 also installs the `prepds` dependencies (OpenCV headless, FastAPI, ...); Phase 0 checks that this works on aarch64, and splitting `dcs` into its own `pyproject.toml` is the fallback.

### 7.2 CLI
```
python -m dcs featurize                      # gold dataset -> training_table.parquet  (where the data lives)
python -m dcs featurize --sequences          # also the compact per-second arrays (for FR-10)
python -m dcs audit                          # data audit only, no training
python -m dcs train                          # both stages, all models, all ablations
python -m dcs train --stage compound         # Stage 1 only
python -m dcs train --models logreg,mlp      # subset
python -m dcs train --device cuda|cpu|auto   # default auto
python -m dcs train --seed 0 --repeats 5
python -m dcs predict --model <run folder> --input <table>
```

### 7.3 Configuration (`training:`)
`seed`, `folds`, `repeats`, `min_class_size`, `min_state_fish`, `stage`, `models`, `use_ntt`, `use_demographics`, `normalize_by_date_vehicle`, `duration_range_s`, `min_detected_fraction`, and `mlp:` (hidden sizes, dropout, weight decay, learning rate, batch size, max epochs, patience, seeds). Defaults carry no data-derived numbers.

### 7.4 Outputs and the model artifact
```
<DCS_OUTPUT_DIR>/training/<run_id>/
    run_info.json     git commit, versions (python, torch, CUDA, scikit-learn), hardware, seed, command
    config_used.yaml
    audit.md
    folds.csv
    metrics.csv  report.md  predictions.csv  confusion_*.png
    model/
        mlp.pt            PyTorch state_dict
        sklearn.joblib    chosen baseline (versions recorded)
        preprocess.json   feature order, imputation values, scaler mean/std, log1p list
        classes.json      class names in output order
```
`run_id` = UTC timestamp + short git hash. The folder is self-contained so it can be copied as one unit.

### 7.5 Dependencies
No new core dependency is needed: the root `pyproject.toml` already lists pandas, numpy, pyarrow, scikit-learn, matplotlib, PyYAML and python-dotenv. Only the extra `train = ["torch>=2.4"]` is added (the existing `ml` extra is for the tracker and stays separate). LightGBM is **not** used (histogram gradient boosting covers the need). On the GB10, install torch from the CUDA index first, then `pip install -e ".[dev,train]"` (runbook, Appendix B). Whether CI should also measure coverage of `src/dcs` is decided in the plan. The CI (Linux, Python 3.11) installs `.[dev]` only, and its coverage floor (85 %) counts `src/prepds` only: tests that need torch use `pytest.importorskip("torch")`, so CI covers the scikit-learn parts and the MLP tests run on the training machine (Q8).

---

## 8. Execution environment

### 8.1 Primary: Dell Pro Max with GB10 (Linux, SSH from PuTTY)
Treated as a fresh machine; the owner has sudo. Vendor specification (supplied by the owner): NVIDIA GB10 Grace Blackwell Superchip (CPU and GPU in one package, ARM64), 128 GB LPDDR5x unified memory, 4 TB NVMe SSD, NVIDIA DGX OS 7. Facts not yet checked on the machine itself: the exact CUDA/driver versions and whether PyTorch's GPU build runs on it; community reports say the CUDA 13 build for aarch64 does. Phase 0 runs the check commands in Appendix B and records the real values in `run_info.json`.

### 8.2 Fallback: home PC, RTX 3060 Ti (8 GB)
Enough for this table and for the 1D-CNN. Windows 11 (decided, Q3): native Windows with PowerShell, or WSL2 if preferred. Instructions follow the same pattern with the standard torch CUDA wheel.

### 8.3 Data movement
Only `training_table.parquet` (and `training_table_schema.json`) is copied to the box, with `pscp`/WinSCP/`scp`, via USB, or through UToledo OneDrive (approved for this data, Q2). Data never goes through GitHub.

### 8.4 Getting the model back
`pscp`, WinSCP or `scp` copy the run folder (or a `tar.gz` of it) to the owner's computer; the reload and verification command is in Appendix B.

---

## 9. Testing and quality assurance

### 9.1 Test approach
Tests use synthetic tables only (never real data), like the rest of the repo. Test first, then implement: a test for each pure function before the function exists. Run after every significant change:
```
pytest tests/dcs -q                # fast loop for this project
pytest tests/ -q                   # everything, before a pull request
python -m dcs audit                # on the synthetic table / real training table
```

### 9.2 Edge-case catalog (every row is a required task)
| ID | Case | Expected behavior |
|---|---|---|
| EC-1 | Subject in the index whose `frames.parquet` or `segments.csv` is missing or unreadable | Dropped, counted and listed in the audit; never silent |
| EC-2 | NTT columns missing for a row | Median imputed from the training fold, `has_ntt = 0`, no crash |
| EC-3 | State never occurred (no bouts) | 0 bouts, mean bout 0, latency = recording length, as in §5.4 |
| EC-4 | Class with fewer than `min_class_size` fish | Dropped and listed in the audit |
| EC-5 | Compound on a single date | Pinned to training in scheme A, excluded from the A score, flagged |
| EC-6 | Compound on two dates | Each date lands in a different fold in scheme A (or the audit says why not) |
| EC-7 | Dose strings `0.03 + 0.01`, `0.03+0.01` | Normalize to one label |
| EC-8 | Compound name with spaces/case differences | One class |
| EC-9 | Scaler/imputer fit | Fit on training fold only (test with a deliberately extreme test fold) |
| EC-10 | Same subject twice in the index | Error with a message (no silent choice) |
| EC-11 | Fish with a low detected-frame share or odd video duration | Kept, flagged in the audit |
| EC-12 | Forbidden columns (§5.3) | Feature matrix never contains them (assertion test) |
| EC-13 | A fish in two folds, or a date in two scheme-A folds | Impossible (test over many seeds) |
| EC-14 | Same seed twice | Same folds; CPU metrics identical |
| EC-15 | No GPU, `--device cuda` | Clear error; `auto` falls back to CPU |
| EC-16 | Torch not installed | Baselines run; MLP step skipped with a message |
| EC-17 | Too few rows or one class | `ConfigError` with a fix hint |
| EC-18 | Saved model reloaded in a fresh process | Reproduces saved predictions (AC-9) |
| EC-19 | Windows paths / Linux aarch64 | Tests pass on each (CI covers Linux x86 only; Phase 0 checks the GB10 and the Windows PC) |
| EC-20 | Constant or all-NaN feature | Dropped with a note, no division by zero |
| EC-21 | Accepted set mixes trackers or calibration profiles | Error naming the versions found |
| EC-22 | An Accepted video's frames or segments contain `Undetermined` | Error: gold-data contract violated |
| EC-23 | A state shown by fewer than `min_state_fish` fish | Its features dropped and listed (§5.3) |
| EC-24 | A compound with zero Accepted fish, or accepted fish concentrated on few dates | Audit table shows it; class filter and §6.4 rules apply |
| EC-25 | `frames.parquet` lacks an expected column or has a different dtype | Error naming the file and column |
| EC-26 | Mixed frame rates or resolutions among accepted videos | Audit flag; pixel-based features marked (§5.3) |

### 9.3 Acceptance checklist
- [ ] AC-1 `audit` reports the §2.3 statistics for the accepted set, including per-compound and per-date counts of Accepted fish and the share of manual frames.
- [ ] AC-2 Features contain none of the excluded columns (EC-12).
- [ ] AC-3 Splits obey §6.4 on the synthetic and real data (EC-5, EC-6, EC-13).
- [ ] AC-4 All baselines and the MLP run under both schemes on identical folds.
- [ ] AC-5 The report shows date-only, majority and permuted-label baselines next to every result.
- [ ] AC-6 NTT, demographics and vehicle-normalization ablations are reported.
- [ ] AC-7 Same seed gives the same result (EC-14).
- [ ] AC-8 No restricted data, names or paths in Git (`git ls-files` shows no data, output or `.env` file).
- [ ] AC-9 A model copied off the GB10 reloads elsewhere and reproduces saved predictions.
- [ ] AC-10 The runbook (Appendix B) was followed from scratch by the owner without help.
- [ ] AC-11 Progress file and design decisions are up to date.

---

## 10. Phases, gates and exit criteria

| Phase | Work | Exit criteria |
|---|---|---|
| **0. Environment and data audit** | Verify the GB10 (Appendix B); install torch; GPU smoke test. On the PC with the gold data: read the real `accepted_index.parquet` schema, count Accepted fish per compound and per date, check trackers and profiles, resolution and frame rate | Hardware recorded; `torch.cuda.is_available()` true; schema and counts written to the progress file |
| **0b. Upstream readiness (outside this project)** | On the owner's PC: `prepds catalog`, `prepds run` (classical tracker) for all videos, review of flagged videos in the review app, `prepds export-index` | At least the number of Accepted fish needed by the G1 stop rule exist (or Q17 approves unreviewed output) |
| **G1** | Owner reviews the audit and the Q-list. **Stop rule:** fewer than 3 compounds besides vehicle with at least 6 Accepted fish on at least 2 dates means training is not meaningful yet; report and wait for more accepted videos | Open questions answered; go / wait decision |
| **1. Featurize, training set and folds** | `featurize.py`, `trainset.py`, config, tests for EC-1 to EC-14 and EC-21 to EC-26 | Tests green; `featurize` and `audit` work on synthetic gold data |
| **2. Baselines** | Majority, date-only, logreg, RF, HGB, both schemes, diagnostics | Report draft; AC-3, AC-5 |
| **G2** | Review baseline report | Agree to proceed |
| **3. MLP and ablations** | `models.py` MLP, ablations, FR-9 | AC-4, AC-6, AC-7 |
| **4. Stage 2 (exploratory)** | Dose models with the confound caveats | Report section with caveats |
| **5. Save/load and runbook** | Artifact, `predict`, reload test, runbook walkthrough | AC-9, AC-10 |
| **G4** | First-run review with the owner (and advisor) | Decide: stop, improve, or add FR-10 |
| **6. Optional 1D-CNN** | FR-10 as in §6.11 | Beats best tabular model on scheme A, or documented as not helpful |
| **7. Wrap-up** | Final plan/progress docs, README pointer | AC-11 |

Each phase ends with the verification loop of §9.1 and a progress-file update. No calendar dates are promised; the schedule depends on Phase 0 and on review turnaround.

---

## 11. Risks and mitigations

| Risk | Prob. | Impact | Mitigation |
|---|---|---|---|
| Model learns the experiment date, not the drug | High | High | Scheme A, date-only baseline, permutation test, vehicle normalization (§6) |
| Dose cannot be validated (dose ≡ date) | Certain | High | Stage 2 labeled exploratory; no dose claims from scheme B |
| Small classes give noisy per-compound results | High | Medium | Class filter, repeats, spread reported, "very small" flags |
| Protocol columns leak the treatment | Medium | High | Excluded by default (§5.3) |
| No Accepted videos yet; later too few, or acceptance order correlated with compound/date | Certain now | High | Phase 0b; stop rule at G1; Q17 (unreviewed output, labeled as such); audit shows acceptance by compound and date |
| Label quality differs by compound (7 calibrated groups, rest extrapolated) | Medium | High | Report the manual-frame share per compound; ablation restricted to calibrated compounds if the audit shows a skew |
| Mixing trackers or calibration profiles | Medium | High | One tracker and profile per set, enforced (EC-21) |
| Pixel-unit features not comparable across setups | Medium | Medium | Audit resolution and frame rate; flag; no body-length data available |
| `prepds` output format changes | Low | Medium | Schema checks (EC-25); read-only consumer; the `prepds` version is recorded in `manifest.json` |
| ARM64/Blackwell torch install problems | Medium | Medium | Phase 0 smoke test; fallback to the 3060 Ti; CPU baselines unaffected |
| Restricted data leaves approved storage | Low | High | Data stays on owner-controlled machines; no GitHub, no external LLM review of data |
| Scope creep into 1D-CNN or images | Medium | Medium | Gates G2 and G4; FR-10 is P2 |
| CI breaks because of torch | Low | Medium | Optional extra; `importorskip` |

---

## 12. Open questions (answer before or during Phase 0)

| # | Question | Status / default |
|---|---|---|
| Q1 | Does the advisor accept "tabular first, 1D-CNN second"? | Open; default yes |
| Q2 | Is OneDrive approved for this data? | **Answered:** UToledo OneDrive is approved |
| Q3 | Home PC operating system? | **Answered:** Windows 11 |
| Q4 | The compound with exactly 5 fish: keep or drop? | **Answered:** drop every class lacking data (threshold 6, §5.6) |
| Q5 | Stage 2 (dose) exploratory only, given the dose-date confound? | **Answered:** yes |
| Q6 | Fish with odd duration or low tracked fraction? | **Answered:** keep and flag |
| Q7 | External LLM tools for cross-model review? | Open; default no (placeholder-only documents at most, never data) |
| Q8 | Add CPU torch to CI so MLP tests also run there? | Open; default no |
| Q9 | Numeric target for "useful" agreed with the advisor? | Open; default none (decision rule of §6.6) |
| Q10 | Deadline? | Open; none promised |
| Q12 | Where does the classifier code live? | **Answered:** `fishbehavior` is legacy. Default adopted: `src/dcs/` inside the same repository and `pyproject.toml` (§7.1); change only if you prefer a separate project |
| Q13 | Which output set is the training source? | **Answered:** classical tracker (the pipeline's fine-tuned computer vision; not yet validated with the pharmacy department, so this phase builds infrastructure) |
| Q14 | How many videos are ACCEPTED today? | **Answered:** none; the pipeline has not been run on the owner's side yet |
| Q15 | May the earlier-pipeline sample (the `behavior_dataset` workbook) be used only for structure checks, never for reported results? | **Answered:** yes, for reference only |
| Q16 | Cut-off for dropping a state's columns when too few fish show it (`min_state_fish`) | **Answered:** "maybe, if it works": default 10, kept configurable and re-examined after the first audit; all six labels stay defined |
| Q17 | May the classifier read unreviewed (`PROCESSED_AUTO`) output? | **Answered:** "depends, but not preferred". Gold-only (Accepted) is the primary source. Unreviewed output is a clearly labeled fallback, decided at gate G1 only if review lags |
| Q11 | Run `harness init` on this repository? It writes `.claude/` hooks and **creates a git commit**, and then gates every `.py` write (spec and a failing test first); see [process/HARNESS_PROCESS.md](process/HARNESS_PROCESS.md) §4 | Open; default no until reviewed |

---

## Appendix A. Glossary
- **NTT:** Novel Tank Test, the workbook's per-trial movement summary (a different session from the video).
- **Scheme A / B:** date-held-out vs random validation (§6.4).
- **Date effect:** score of B minus score of A.
- **Vehicle:** control fish exposed to the carrier only.
- **Baseline:** a simple model a fancier one must beat.

## Appendix B. Runbook: training on the GB10 over SSH, then retrieving the model

> Verify every step on the real machine in Phase 0; versions and URLs change. Replace `<user>`, `<host>`, `<run_id>` with your own values. Do not paste real data or names into chat or Git.

**1. Connect (PuTTY).** Host Name = box address, Port 22, Connection type SSH, save the session, Open, log in. (Windows 10/11 also has `ssh <user>@<host>` in PowerShell.)

**2. Check the machine** (record the output in the progress file):
```bash
uname -m                      # expect aarch64 (ARM64)
cat /etc/os-release | head -3   # expect NVIDIA DGX OS 7
nvidia-smi                    # GPU, driver, CUDA version
python3 --version
df -h ~ ; free -h
```

**3. Install tools (sudo available):**
```bash
sudo apt update
sudo apt install -y git tmux python3-venv python3-pip
```

**4. Get the code, create an environment** (Python 3.11 or newer is required; if `python3 --version` is older, install 3.11 first, for example with `uv`):
```bash
git clone <repository url> fish-detection && cd fish-detection
git checkout <branch to run>
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
```

**5. Install PyTorch with CUDA first** (community reports say the `cu130` index has aarch64 wheels that run on this GPU; confirm against the current official PyTorch install selector):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev,train]"       # torch is already satisfied, so it is not replaced (pyarrow and scikit-learn come from the core dependencies)
python - <<'EOF'
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda"); print((x @ x).sum().item())
EOF
```
If `is_available()` is `False` or the matrix multiply fails, stop and report the full message; do not work around it silently.

**6. Build the table, then copy only the table in.**

On the machine that holds the accepted gold dataset (your PC; no GPU needed):
```bat
python -m dcs featurize
```
That writes `training_table.parquet` and `training_table_schema.json` under `DCS_OUTPUT_DIR`. Copy just these two files to the box. Option 1, from your Windows 11 PC in PowerShell or Command Prompt:
```bat
pscp training_table.parquet training_table_schema.json <user>@<host>:/home/<user>/fish-detection/data/
```
(`pscp` ships with PuTTY. WinSCP works too.) Option 2, USB: `lsblk`, `sudo mount /dev/sdX1 /mnt/usb`, `cp`, `sudo umount /mnt/usb`. Option 3, UToledo OneDrive: only if the box has a desktop session and browser; otherwise use Option 1. On the box set `DCS_TABLE` (or `DCS_OUTPUT_DIR`) in `.env` so the table is found.

**7. Run, so that it survives a dropped connection:**
```bash
tmux new -s train
source .venv/bin/activate
python -m dcs audit
python -m dcs train 2>&1 | tee train.log
# detach: Ctrl-b then d      reattach later: tmux attach -t train
```
Check progress with `tail -f train.log` and `nvidia-smi`. Without tmux: `nohup python -m dcs train > train.log 2>&1 &`.

**8. Retrieve the model** (from your own computer):
```bat
pscp -r <user>@<host>:/home/<user>/fish-detection/outputs/training/<run_id> .
```
(or on the box `tar czf run.tgz outputs/training/<run_id>` (from the repository folder), then copy `run.tgz` with WinSCP/`pscp`).

**9. Reload and verify elsewhere** (same Python packages needed, including the torch/scikit-learn versions recorded in `run_info.json`):
```bash
python -m dcs predict --model <run_id folder> --input <training table>
```
The result must equal the saved `predictions.csv` for the same rows (AC-9).

**10. Home PC fallback.** Same steps with the CUDA wheel from the PyTorch selector for the 3060 Ti (for example `pip install torch --index-url https://download.pytorch.org/whl/cu126` on Windows/Linux x86; confirm the current index name). WSL2 or native Windows both work; the commands in step 7 are identical.

**11. Cleanup and safety.** Keep data and output folders off Git (add them to the new package's `.gitignore`, as `prepds` does). Delete the data from shared/USB media when finished according to your data agreement.
