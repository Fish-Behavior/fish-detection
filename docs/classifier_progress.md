# Classifier (`dcs`): progress and decisions

**PRD:** [classifier_PRD.md](classifier_PRD.md) v0.3 · **Plan:** [plans/classifier_plan.md](plans/classifier_plan.md)
**Privacy:** placeholders only (`COMPOUND_A`, `F_0042`, `<date>`); no data, outputs, real names, dates or paths.
**Status legend:** `NOT_STARTED` / `IN_PROGRESS` / `DONE` / `BLOCKED: <reason>` / `OWNER`. Updated after every task.

**Last updated:** 2026-10-02. U1 started: T1.1 and T1.2 done, T1.3 waits on a harness blocker (§4: the hook scripts are not executable, so the gate never observes RED/GREEN). `which pytest` now resolves to `.venv` (old blocker closed).

---

## 1. Current status

| Phase | Status | Note |
|---|---|---|
| 0. Environment | IN_PROGRESS | T0.1 done; T0.2-T0.4 owner |
| 0b. Upstream readiness | OWNER | No video Accepted yet (Q14) |
| G1 | BLOCKED: no Accepted videos | Real-data audit with `dcs audit` (D-010) |
| 1. Featurize, training set, folds | IN_PROGRESS | U1 started; blocked on the hook permissions (§4) |
| 2. Baselines | NOT_STARTED | U9-U12 |
| G2 | NOT_STARTED | Synthetic baseline report review |
| 3. MLP and ablations | NOT_STARTED | U13-U14 |
| 4. Stage 2 | NOT_STARTED | U15 |
| 5. Save/load, runbook | NOT_STARTED | U16-U18 |
| G4 | BLOCKED: needs real data | |
| 6. Optional 1D-CNN | NOT_STARTED | Only after G4 |
| 7. Wrap-up | NOT_STARTED | |

## 2. Tasks

| Task | Unit | Status | Spec | Note |
|---|---|---|---|---|
| T0.1 baseline `pytest tests/ -q` | - | DONE | | 535 passed, 4 skipped (local `.venv`, Python 3.14, no torch) |
| T0.2 GB10 checks | - | OWNER | | Record in §6 |
| T0.3 GB10 install of root project | - | OWNER | | |
| T0.4 Windows PC checks | - | OWNER | | |
| T1.1-T1.3 scaffold, config | U1 | BLOCKED: gate hooks not executable (§4) | FISH-PROD-001 (workflow run 2) | T1.1 done (`pyproject.toml`, `.gitignore`, `.env.example`, `config/default_training.yaml`); T1.2 done (`tests/dcs/test_dcs_config.py`, 40 tests, fail at import as expected); T1.3 not started: RED not yet observed by the gate |
| T1.4-T1.5 synthetic data | U2 | NOT_STARTED | | |
| T1.6-T1.7 gold reader | U3 | NOT_STARTED | | |
| T1.8-T1.9 workbook | U4 | NOT_STARTED | | |
| T1.10-T1.11 featurize | U5 | NOT_STARTED | | |
| T1.12-T1.13 training set | U6 | NOT_STARTED | | |
| T1.14-T1.15 folds | U7 | NOT_STARTED | | |
| T1.16-T1.17 audit | U8 | NOT_STARTED | | |
| T2.1-T2.2 preprocessing | U9 | NOT_STARTED | | |
| T2.3-T2.4 baselines | U10 | NOT_STARTED | | |
| T2.5-T2.6 evaluation | U11 | NOT_STARTED | | |
| T2.7-T2.8 report, train | U12 | NOT_STARTED | | |
| T3.1-T3.2 MLP | U13 | NOT_STARTED | | |
| T3.3-T3.4 ablations | U14 | NOT_STARTED | | |
| T3.5 runtime check | - | NOT_STARTED | | |
| T4.1-T4.2 stage 2 | U15 | NOT_STARTED | | |
| T5.1-T5.2 artifact | U16 | NOT_STARTED | | |
| T5.3-T5.4 predict | U17 | NOT_STARTED | | |
| T5.5 privacy test | U18 | NOT_STARTED | | |
| T5.6 upstream request draft | - | NOT_STARTED | | |
| T5.7 cross-machine, runbook | - | OWNER | | |

## 3. Requirements and acceptance

| Item | Status | Evidence |
|---|---|---|
| FR-1 featurize (P0) | NOT_STARTED | |
| FR-2 audit (P0) | NOT_STARTED | |
| FR-3 Stage 1 (P0) | NOT_STARTED | |
| FR-4 validation, diagnostics (P0) | NOT_STARTED | |
| FR-5 report (P0) | NOT_STARTED | |
| FR-6 save/reload (P0) | NOT_STARTED | |
| FR-7 Stage 2 (P1) | NOT_STARTED | |
| FR-8 predict (P1) | NOT_STARTED | |
| FR-9 vehicle normalization (P1) | NOT_STARTED | |
| FR-10 1D-CNN (P2) | NOT_STARTED | |
| AC-1 … AC-11 | NOT_STARTED | See plan §8 for which need real data |

Edge cases EC-1 to EC-29: all NOT_STARTED; task and test file per row in plan §6.

## 4. Answers to the open questions (owner, 2026-10-02)

| # | Answer | Effect |
|---|---|---|
| Q1 | Yes: tabular first, 1D-CNN second | PRD order kept |
| Q7 | The agent does not call external LLM tools; the owner pastes a prompt into them | Prompt in plan Appendix A; placeholder material only |
| Q8 | Default: no torch in CI | CI file unchanged; MLP tests run on the training machine (D-002) |
| Q9 | "Maybe a yes": tentatively no numeric target | PRD §6.6 decision rule; revisit with the advisor at G2 |
| Q10 | No deadline for this project; see the project scope | Scope §9 targets the end of the current year for all of its Phase 1 (seven items, the classifier is one) |
| Q11 | (a) keep harness-os governance on | Every unit follows `new-feature` (D-001) |
| C1 | Follow the recommendation | D-003; upstream request T5.6 |
| C3 / C4 | Read the workbook; stop with a hint on a missing date | D-004, D-005 |

**Blocker for D-001 (closed 2026-10-02):** bare `pytest` was not on the PATH of the Claude Code process. Fixed by launching Claude Code with `.venv` active; `which pytest` now points into `.venv/bin/`.

**Blocker for D-001 (open, 2026-10-02):** the four files in `.claude/hooks/` were committed by the harness scaffold commit with mode `100644` (not executable), and `.claude/settings.json` runs them directly. Every hook call therefore fails with "permission denied", which Claude Code treats as a non-blocking error: `enforce-gate.sh` never runs, the `test_runs` table stays empty (checked), the workflow cannot leave `establish_red_phase`, and the write gate does not enforce anything. Fixing it touches the integrity-checked harness files (CONST-CORE-004), so it waits for the owner (options in the session reply).

**Open for the owner:** approve the PRD change list (plan §2) and decide whether it becomes PRD v0.4; confirm with the advisor that PyTorch is fine although the scope doc names TensorFlow (C8); confirm D-008 at G2; file the upstream request once drafted (T5.6).

## 5. Decision log

| ID | Date | Decision | Why |
|---|---|---|---|
| D-001 | 2026-10-02 | Harness-os governance stays on (it was already initialized on `master`); every `.py` write follows spec → RED → GREEN → review | Q11 = a; matches the test-first plan |
| D-002 | 2026-10-02 | CI workflow unchanged: no torch, coverage floor still counts `prepds` only; `tests/dcs` runs in the existing backend job | Q8 default |
| D-003 | 2026-10-02 | Tracker inferred from `calibration_profile_version` (pattern `-model`, configurable); optional cross-check with `detections.parquet` under `DCS_PROCESSED_DIR`; a mixed set is an error unless `--profile` selects one | Gold folder carries no tracker field (PRD change C1) |
| D-004 | 2026-10-02 | `featurize` reads the 8 NTT columns from the workbook (`DCS_DB_PATH`), keyed by `<Sex>_<Subject:04d>`; unset → NTT unavailable, reported | NTT is in neither the index nor the catalog (C3) |
| D-005 | 2026-10-02 | Index rows without `date` stop `featurize` with a `ConfigError` telling the user to run `prepds catalog` then `prepds export-index` | Review-app accepts leave workbook fields null (C4) |
| D-006 | 2026-10-02 | The model folder holds `reference_predictions.csv` (final model on the training rows); AC-9 compares against it | `predictions.csv` is out-of-fold (C6) |
| D-007 | 2026-10-02 | Privacy check is a pytest file (`git ls-files` scan), not a new CI job | S5 names a job that does not exist; CI stays unchanged (C7) |
| D-008 | 2026-10-02 (rev. r1) | **Proposed, confirm at G2:** in the FR-9 ablation, vehicle fish are the per-date reference only; FR-9 is reported as a pair, raw vs normalized, both on the same non-vehicle classes and folds. The main N-class results stay unnormalized | Using vehicle labels of test fish to normalize is label leakage (C9); with under 2 vehicle fish per date on average, a fish's own value sets about half of its reference. The pair keeps the comparison on identical class sets (R2, C17) |
| D-009 | 2026-10-02 | Module layout of plan §3.1 instead of PRD §7.1's four modules | Harness rule of 200-400 lines per file (C10) |
| D-010 | 2026-10-02 | Real-data schema check and counts happen at G1 with `dcs audit`; Phases 1-5 proceed on synthetic data regardless | Counting needs Phase 1 code and Accepted videos (C12) |
| D-011 | 2026-10-02 | Date-only baseline = majority label of the nearest training date (same date in scheme B) | PRD leaves scheme A undefined; nearest date also measures campaign effects (C13) |
| D-012 | 2026-10-02 | Per-video files are found at `<DCS_ACCEPTED_DIR>/<video_id>/`; the index's stored paths are ignored | Stored paths are machine-specific and break when the folder moves |
| D-013 | 2026-10-02 | `dcs` restates the `prepds` constants it needs (state names, frame columns, workbook headers) instead of importing them | Read-only consumer, never imports `prepds` (PRD §2.2, NFR-6) |
| D-014 | 2026-10-02 | `DCS_OUTPUT_DIR` defaults to `outputs/dcs`; `.gitignore` gains `*.pt`, `*.joblib` | Outputs stay git-ignored even if moved (NFR-2, C14) |
| D-015 | 2026-10-02 | Depth features are an ablation group, off by default (`training.use_depth: false`); the audit's framing check (EC-31) decides at G1 whether to switch them on and whether pixel speeds get flagged | Pixel depth from the frame top and pixel speeds encode camera framing and zoom, which can change by date (R1, C16) |
| D-016 | 2026-10-02 | Within-date permutation stays the primary null; audit and report give the share of fish on dates with ≥2 classes. No dataset-wide permutation: the majority baseline already plays that role | A single-class date adds nothing to the within-date test (R3, C18) |
| D-017 | 2026-10-02 | Metrics are computed on the pooled out-of-fold predictions of each repeat; the spread is the standard deviation across repeats | No per-fold metric is undefined when a fold lacks a class (R4, C19) |
| D-018 | 2026-10-02 | `featurize --videos <dir>` builds rows from per-video folders without an index, labels optional; rows from non-Accepted videos are marked unreviewed | Gives `predict` an input for a new fish (R6, C21) |
| D-019 | 2026-10-02 | Scheme A stays stratified group K-fold, K=5, repeated (PRD §6.4); no leave-one-date-out | Owner, review R4: keeps split randomness for the §6.6 spread rule |
| D-020 | 2026-10-02 | 1D-CNN (FR-10) stays optional, after G4 only | Owner, review R7; consistent with Q1 |
| D-021 | 2026-10-02 | EC-11 flag thresholds: `training.duration_range_s` defaults to `null` (no range check until set at G1); `training.min_detected_fraction` defaults to 0.8 (flag only, never drop) | PRD gives no numbers; a duration range would be read off the real recordings ("no data-derived numbers", §7.3), whereas 0.8 is a judgment threshold for a flag |
| D-022 | 2026-10-02 | Harness spec ids: unit U*n* is `FISH-PROD-0nn` (product spec type; the server only accepts `^[A-Z]{2,10}-PROD-[0-9]{3,}$` and the keys `id, title, summary, goals, nonGoals, stakeholders`). Workflow run 1 (id `FISH-DCS-U1`, never had a spec) is abandoned; run 2 is U1 | Schema found by `validate_spec`; one spec per unit keeps traceability per unit |
| D-023 | 2026-10-02 | Config overrides are validated: a key absent from `config/default_training.yaml` or a value of the wrong type (bool/int/float/str/list/section) is a `ConfigError` naming the dotted key; `null` only where the default is `null`; an int is accepted for a float. Value ranges are checked by the step that uses them | A typo in an override must not be silently ignored (plan T1.2 "unknown key") |
| D-024 | 2026-10-02 | MLP defaults not given by the PRD: learning rate 0.001, weight decay 0.01 (the AdamW default), patience 20 epochs, inner validation share 0.2 | Standard starting values, not data-derived; all in `training.mlp` |

## 6. Hardware record (Phase 0, owner)

| Machine | `uname -m` | OS | GPU / driver / CUDA | Python | torch | `cuda.is_available()` | Install of root project |
|---|---|---|---|---|---|---|---|
| GB10 | | | | | | | |
| Windows PC | | | | | | | |

## 7. Verification log

| Date | Command | Result |
|---|---|---|
| 2026-10-02 | `.venv/bin/pytest tests/ -q` (baseline, before any `dcs` code) | 535 passed, 4 skipped, 31.7 s |
| 2026-10-02 | Step 0: `which pytest`; harness MCP `get_constitution`; `docker ps` | `.venv/bin/pytest`; constitution 1.0.0 returned; `harness_postgres` (healthy) and `harness_gate_daemon` up |
| 2026-10-02 | U1: `pytest tests/dcs -q` (intended RED) | Exit 2, `ModuleNotFoundError: dcs` (expected). **Not recorded by the gate**: hooks not executable (§4) |

## 8. External plan review r1 (2026-10-02, another LLM, plan + PRD with placeholders only)

| # | Finding | Verdict | Change |
|---|---|---|---|
| R1 | `depth_from_surface` in pixels encodes camera framing (date proxy) | Accepted, broadened to pixel speeds | D-015, C16, EC-31 |
| R2 | FR-9 on N−1 classes is not comparable to N-class baselines; proposed keeping vehicle and normalizing all fish | Problem accepted; proposed fix rejected (self-inclusion leak with <2 vehicle fish per date) | D-008 revised, C17 |
| R3 | Within-date permutation is degenerate; proposed a dataset-wide permutation | Rejected: keeping date is the point of the test; dataset-wide null is covered by the majority baseline. Added an informativeness report | D-016, C18 |
| R4 | K=5 group folds drop rare classes from training; proposed leave-one-date-out | Claim mostly wrong (EC-6 keeps a 2-date class in training; metrics pooled). Owner: keep PRD K=5 group folds (leave-one-date-out removes split randomness, so the §6.6 spread would need bootstrap intervals) | D-017, D-019, C19 |
| R5 | 1D-CNN needs uniform sequence lengths | Accepted for Phase 6 | EC-30, C20 |
| R6 | `predict` has no way to featurize a single new fish | Accepted, scaled down | D-018, C21 |
| R7 | Drop the 1D-CNN | Owner: keep it as P2, only after G4, kept only if it beats tabular on scheme A (Q1) | D-020 |
