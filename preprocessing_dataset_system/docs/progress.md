# Preprocessing Dataset System — Implementation Progress

**PRD:** [`preprocessing_dataset_system/docs/PRD.md`](PRD.md) (1141 lines, read in full before this
plan was written). This plan's phase structure is PRD §12 verbatim — no phases invented, none
reordered. Its exit criteria are PRD §13 (Review & Acceptance Checklist) + §8 (Success Criteria)
read together with §7 (NFRs). Its test coverage is the `[P]`-marked test tasks already embedded in
each phase of §12, cross-referenced against every bullet of the §3.3 edge-case catalog (table at the
bottom of this file).

**Status legend:** `NOT_STARTED` / `IN_PROGRESS` / `DONE` / `BLOCKED: <reason>`. Updated after each
task, not at phase end — this file is the resumable source of truth if a session ends mid-phase.

**Last updated:** 2026-09-22 (plan created; Phase 0 not yet started)

---

## 0. Real-data inventory (grounds every "real 353-video set" exit criterion below)

Gathered directly from the files under `fish-detection/` before writing this plan, so phase exits
below say what's actually checkable now vs. `BLOCKED (awaiting full video sync)`, per the PRD's own
C15 (owner syncs OneDrive → local; not yet complete for all 353).

- **Workbook** (`sample_data/00_NTT_DataBase.xlsx`, `Sheet1`, 22 columns incl. trailing
  whitespace/newlines in some headers e.g. `"NTT \nTime (min):"` — code must strip header
  whitespace, not assume exact literal match): 720 rows total, **353 with non-blank `Compund:`**
  (matches PRD exactly). `Body Tissue:` confirmed 100% empty. `Agent Exposure Time (min):` is 20.0
  for 350 rows, 10.0 for exactly 3 rows (matches §3.3's "3 of 353"). Exact full-row duplicate
  confirmed: rows 319/320, both `Subject # 320`, `FD-2-66`, sex M, same date — matches C13 exactly.
  17 distinct `Compund:` values found, **including `"HS-1-51"` (1 row) as well as `"HS-1-151"`
  (7 rows)** — these are almost certainly a data-entry variant of the same family, but the PRD does
  not resolve this as a known alias, and no video folder named `HS-1-51` exists. Per FR-004 this is
  **not** hand-patched; it surfaces as a genuine `unmatched_trial` in the exceptions report, same as
  any other mismatch, and the owner can decide from the report whether it's the same compound.
  Strain counts: Casper 276, Wild-type (AB) 72, ABSL 5 — Wild-type/ABSL rows exist for `DOB`,
  `Fentanyl`, `Methylone`, `Veh` (Wild-type) and `FD-2-67`, `FD-2-95`, `Veh` (ABSL), all of which have
  synced videos locally (see below) — satisfies T029's "spot-check ≥1 video per strain" without
  waiting on more of the sync.
- **Videos** (`videos/Phase_1/`, 16 compound folders, all carrying the C12 parenthetical suffix,
  4 of them additionally marked `Missing` in the folder name itself e.g.
  `MDMA(DONE Compression Only, Missing)`): **337 real `.mp4` files** currently synced (excludes
  `:Zone.Identifier` sidecar files, which are Windows-download metadata, not video data, and must be
  filtered out by extension+exact-suffix match in `catalog.py`/video discovery, not just glob
  `*.mp4`). `videos/Phase_2/` exists but is **empty** — no Phase 2 videos synced yet.
  352 logical trials (353 rows − 1 dedup) vs. 337 synced videos ⇒ **at least 15 trials currently
  have no local video**, concentrated in the four folders whose names already say `Missing`
  (`Veh`: 77 rows/75 files, `FD-2-96`: 20/17, `FD-2-95`: 16/9, `MDMA`: 10/5) plus `HS-1-51`. This is
  expected and exactly what FR-004's exceptions report is for — **not** a bug to fix before Phase 2.
  File naming confirmed genuinely inconsistent per C11: **322 files use `F_`, 15 use lowercase
  `f_`**; zero-padding is **223 files at 4 digits, 106 at 3 digits** — both must be normalized, not
  assumed uniform.
- **Sample video** (`sample_data/826e5e05-....mp4`, a standalone demo file, not part of the
  353-trial set / not named `{sex}_{subject}`): measured **304×240, fps = 29.832876…, frame_count =
  35880, duration = 1202.7 s** — matches PRD's own worked manifest example (`video_fps: 29.83`,
  `video_duration_s: 1202.7`) almost exactly and empirically confirms FR-005/edge-case-catalog's
  "read actual fps, never assume 30/1200" requirement is real, not theoretical.
- **Reference images** (`sample_data/sample_labeled_1.png` 1599×1117, `sample_labeled_2.png`
  1794×1141, both RGBA): present locally, used only by Phase 6 calibration, never committed (C14).
- **Toolchain**: no system `pip`/`venv` available; `uv` is installed and used instead.
  `preprocessing_dataset_system/.venv` created with `uv venv --python 3.11` (matches §9.1's
  Python 3.11 target — `cpython-3.11.15` was already available locally via `uv python list`).
  `opencv-python-headless` used, not full `opencv-python` — this is WSL2 with no display server;
  the full wheel's `libGL.so.1` dependency would fail at import.

**Net effect on this plan:** Phase 0–5 and the catalog/tracking/feature work in Phase 2–5 can run
against real data now. Phase 6 calibration's 5 reference compound/concentration groups (VEH, MDMA,
Methylone, DOB, Fentanyl) all have local videos. Any exit criterion below that says "all 353" is
explicitly qualified as "all 352 logical trials, of which 337 have a locally synced video as of
2026-09-22; the remaining ~15 are expected exceptions-report entries, not blocked work" — full
closure of Phase 13's "all 353 real videos" pass is `BLOCKED: awaiting remaining OneDrive sync`
until the owner syncs the rest, including whatever lands in the now-empty `Phase_2/`.

---

## 1. Sequencing notes (deviations from a naive reading of §12, decided before coding started)

- **Phase 6 ⇄ Phase 7 is not a hard wall.** T045 (calibration search) runs "the (in-progress)
  tracking→features→labeling stack" over real videos — so `labeling.py`'s *rules* must exist,
  parameterized entirely from `config/default_thresholds.yaml`, before calibration can search
  anything. What Phase 7 actually waits on is the frozen **values**, not the rule code. Sequencing
  used in this plan: Phase 7's rule implementation (T052) is written and unit-tested against
  placeholder thresholds from T004 as part of Phase 7's own task list (unchanged from §12), but a
  first draft of the classification functions is available for Phase 6 to import and search over.
  Phase 7's real exit (frozen-threshold validation) still waits on Phase 6 finishing. This split is
  recorded here so a later session doesn't re-litigate the dependency graph.
- **Restricted-data fixtures need a scoped `.gitignore` override.** The root `.gitignore` has
  blanket `*.mp4`, `*.xlsx`, `*.csv`, `*.parquet`, `*.pdf` — but Phase 1 commits
  `tests/fixtures/synth_tiny.mp4` (synthetic, not restricted) and a synthetic `.xlsx`, and Phase 6
  commits `config/default_thresholds.yaml`. A child `.gitignore` inside
  `preprocessing_dataset_system/tests/fixtures/` with narrow `!`-negations (git resolves the
  *nearest* `.gitignore` per path) is added in T001 and verified with
  `git check-ignore -v <path>` before any fixture is committed — never assumed. `data/`, `outputs/`,
  `accepted/` stay hard-excluded (no negation) per FR-018/NFR-004.
- **`PRD.md` actually lives at `preprocessing_dataset_system/docs/PRD.md`**, not at the package
  root as §9.3's tree diagram draws it. Noted here so the difference isn't later mistaken for a
  drift from spec — it's just where it was placed when written.
- **harness-os pipeline stages run at phase granularity, not per-task.** Constitution → Spec → Test
  Suite → Generate Code → Code Review runs once per PRD §12 phase (14 runs total, not 90). Stage
  artifacts/outcomes are logged under each phase's entry in §3 below.
- **Missing deps not listed in PRD §9.1, added to `requirements.txt` in T001**: `scikit-learn`
  (Controlled-vs-Erratic GMM, §9.5.6), `pytesseract` (best-effort per-subject OCR, §9.5.5 — optional,
  its absence/failure must not block calibration, only degrade it to group-level-only), `openpyxl`
  (pandas' `.xlsx` engine), `Pillow` (image I/O for calibration digitization).
- **NFR-002 vs. `sklearn.mixture.GaussianMixture`**: unseeded GMM fitting is non-deterministic.
  `random_state` is fixed and the seed value is itself stored as a field inside
  `CalibrationProfile`, so "same video + same calibration profile version ⇒ identical
  `frames.parquet`" is actually enforced, not just assumed.

---

## 2. Design decisions log

Filled in as implementation proceeds, per phase. PRD-flagged open items to resolve here:
threshold values (§9.5.6, deferred to Phase 6 calibration) and the `Undetermined` placeholder
color/styling (§5.2, "finalized during `rendering.py` implementation" → Phase 8).

*(none yet — plan just created)*

---

## 3. Phase-by-phase task tracking

Task IDs/descriptions/file targets are copied verbatim from PRD §12; only `Status` and `Notes`
columns are added here. `[P]` tasks are parallelizable within their phase (noted from PRD).

### Phase 0 — Setup & Repo Scaffolding

harness-os stage: Constitution ✅ (`get_constitution`), Spec ✅ (`PDS-PROD-001` created, workflow run 167),
Risk ✅ (`assess_risk` → `high`, a false-positive keyword match on the word "secret" appearing in the
*request description itself* — required gates spec/tests/review:security, all satisfied below), Test Suite
✅ (manual RED confirmed for T006/T007), Code ✅, Review ✅ (code-reviewer + security-reviewer agents run,
findings folded in below). `harness init` is TTY-gated and unavailable in this non-interactive session, so
the gate-daemon's automatic RED/GREEN observation and config checksumming (CONST-CORE-002/004) could not be
activated — recorded as decision id 1863 (medium risk, approved): proceeded with manually-observed TDD
(write test → run → see it fail → implement → run → see it pass, real pytest exit codes, not self-reported)
instead. Owner can run `harness init /home/lehoa/projects/fish-detection --prefix PDS` interactively at any
time to enable full automated gating retroactively; nothing here is blocked on that.

| ID | Description | Status | Notes |
|---|---|---|---|
| T001 | Create top-level folder + `pyproject.toml`/`requirements.txt`/`README.md`/`.env.example`/`.gitignore` | DONE | Scoped fixtures `.gitignore` negation added and verified with `git check-ignore`/`git add` dry-runs — fixtures are committable, `data/`/`outputs/`/`accepted/`/`.venv/` correctly blocked |
| T002 [P] | Scaffold `src/prepds/__init__.py`, `__main__.py` | DONE | |
| T003 [P] | `src/prepds/config.py` — Settings/env loader | DONE | Mirrors `fishbehavior.config` exactly (env > .env > defaults); own `PDS_*` vars (`PDS_VIDEO_DIR`/`PDS_DB_PATH`/`PDS_REFERENCE_DIR`/`PDS_OUTPUT_DIR`/`PDS_ACCEPTED_DIR`/`PDS_WORKERS`/`PDS_CONFIG`) |
| T004 [P] | `config/default_thresholds.yaml` placeholder | DONE | All numeric threshold values `null` — structure only, per §9.5.5's "don't hand-pick reasonable-looking numbers" instruction; `random_state: 0` field added for NFR-002 GMM determinism (§1 sequencing note) |
| T005 | `src/prepds/cli.py` skeleton, `check-config` | DONE | |
| T006 [P] | `tests/test_config.py` | DONE | 9 tests; confirmed RED (`ModuleNotFoundError`) before T002-T005 existed, then GREEN after |
| T007 [P] | `tests/test_cli.py` | DONE | 4 tests; same RED→GREEN cycle. Bug found and fixed during RED: `/dev/null` is not `Path.is_file()`-true on Linux (character device), so the original test would have failed for the wrong reason — switched to `monkeypatch.chdir(tmp_path)` with no `--env-file` flag |
| T008 | Verify pytest green, commit scaffold | IN_PROGRESS | pytest green (13/13, real run, exit 0). **Not yet committed** — per this session's standing instruction to only commit when the user explicitly asks, git add/commit is held pending that go-ahead rather than auto-committed per the PRD task text |

**Exit (PRD):** `python -m prepds check-config` runs (verified: exit 0, prints all `PDS_*` settings) ✅;
13-test suite passes ✅ (Phase 0's actual scope is config+CLI, so "empty test suite" from the PRD's own
exit wording is read as "the test suite that exists for this phase's scope", not literally zero tests).
**Phase 0 functionally complete**, pending only the explicit-commit approval noted in T008.

### Phase 1 — Data Contracts & Synthetic Fixtures

harness-os stage: covered under the same workflow run 167 / decision 1863 as Phase 0 (spec `PDS-PROD-001`
covers the whole system, not re-split per phase for this low-risk, no-new-external-surface phase). Manual
RED→GREEN TDD as in Phase 0. Code review: folded into the same code-reviewer pass as Phase 0 (ran after
models.py existed; see Phase 0 section — findings applied). Security review: in progress (background agent,
covers `models.py` too).

| ID | Description | Status | Notes |
|---|---|---|---|
| T009 [P] | `Trial`, `VideoAsset` schemas — `models.py` | DONE | Scoped to exactly T009/T010's 5 named types (not all of §5.1's 9 entities) — the rest (`Track`, `FeatureFrame`, `StateFrame`, `StripImage`, `CalibrationProfile`, `ReviewRecord`, `CatalogExceptionsReport`) are added in the phase that first needs them (Phase 2/4/5/6/8/9), per YAGNI — noted as a deliberate scoping decision, not an oversight |
| T010 [P] | `FrameRow`, `StateSegment`, `ManifestRecord` schemas | DONE | `FrameRow` matches the `frames.parquet` field list in PRD §5.4 exactly (7-value `BehaviorState` enum, `FrameSource` auto/manual) |
| T011 [P] | Synthetic ~5s 304×240 30fps test video | DONE | `tests/fixtures/make_synth_video.py` → `synth_tiny.mp4` (150 frames, verified via `cv2.VideoCapture` probe: fps=30.0, frames=150, 304×240 exactly). Uses the real measured 304×240 (§0), not the scope doc's approximate 320×240. Deterministic (pure function of t, no RNG) |
| T012 [P] | Synthetic 10-row mini workbook | DONE | `tests/fixtures/make_synth_db.py` → `synth_db.xlsx`. Headers copied verbatim from the real workbook incl. embedded-newline quirks (§0). Covers all 5 required cases plus a strain variant and the real 10-min-exposure edge case; verified with pandas read-back (8/10 real rows, dup pair present) |
| T013 [P] | Legend swatch crops from real reference PNGs | DONE | `tests/fixtures/make_legend_swatches.py` → `legend_swatches.png` (146×30, 5×26px solid patches). Swatch x-ranges found by scanning `sample_labeled_1.png` for near-exact PRD §5.2 palette matches restricted to the legend's y-band, keeping only the widest contiguous column run per color. Verified: Controlled Swim and Surface Breach are exact hex matches (`BEBEBE`, `00FF00`); Erratic/Freezing/Listing are within a few hex units (`FC0005` vs `FF0000` etc. — PNG scaling/compression, not a wrong crop). No subject-timeline data included, only the legend row — safe to commit (confirmed via `git add` + `git check-ignore` dry run) |
| T014 | `tests/test_models.py` round-trip | DONE | 16 tests incl. all 7 `BehaviorState` values parametrized; confirmed RED (`ModuleNotFoundError`) before `models.py` existed, GREEN after |

**Exit:** all 5 fixture/schema artifacts exist and round-trip/probe-verify correctly against real-data-derived
expectations (measured video specs from §0, real workbook headers, real reference-image colors) — not just
internally self-consistent synthetic data. Phase 1 complete.

### Phase 2 — Catalog (`catalog.py`) — FR-001..004

harness-os stage: covered under workflow run 167 / spec `PDS-PROD-001` / decision 1863 (same as Phase 0-1).
Risk assessed separately (id 328): `medium`, gates spec/tests/review:code (no security gate this time - the
Phase 0 "high" was a keyword false-positive on "secret" in the request text, not a real signal). Code review:
DONE - code-reviewer agent found 1 HIGH (case-sensitive `glob("*.mp4")` vs. the already-case-insensitive
`VIDEO_FILENAME_RE` - real inconsistency, though empirically 0 uppercase `.MP4` files exist in the current
337-video mirror) + 2 MEDIUM (match-key uniqueness asserted only in a docstring, not enforced in code; NaN
`agent_exposure_min` would silently read as "within tolerance" since NaN comparisons are always False in
Python) + 1 LOW (manual `Trial` field-by-field reconstruction instead of `dataclasses.replace()`). All 4
fixed: case-insensitive file discovery via `iterdir()` + `suffix.lower()`; a new `ambiguous_matches`
exceptions-report category (`CatalogExceptionsReport` field) catches both video-side and trial-side key
collisions instead of silently overwriting/double-assigning; `find_duration_mismatches()` now explicitly
guards `math.isnan()`; `match_videos()` uses `dataclasses.replace()`. 4 new tests added (50/50 suite green);
real catalog run re-verified identical (328 matched, 0 ambiguous) - fixes are neutral on current data, close
real gaps. Security review not required by the risk gate for this phase; not separately run.
Manual RED→GREEN TDD throughout, plus a **real-data verification pass (T023)** that caught and fixed a
genuine matching bug before it could have silently corrupted the catalog - see T023 notes below. This is
exactly the "run a verification loop, confirm output matches expectations, debug and iterate" step the
task instructions require, and it mattered: synthetic-fixture tests alone (T015-T019) all passed against a
**wrong** implementation.

| ID | Description | Status | Notes |
|---|---|---|---|
| T015 [P] | `test_excludes_blank_rows` | DONE | |
| T016 [P] | `test_excludes_body_tissue_column` | DONE | |
| T017 [P] | `test_normalized_matching` | DONE | Covers real observed variants: `F_`/`f_`, 3-digit vs 4-digit zero-pad, `(status suffix)` incl. `", Missing"`, `+` combo-treatment spacing |
| T018 [P] | `test_duplicate_row_detected_and_flagged` | DONE | |
| T019 [P] | `test_unmatched_trial_and_unmatched_video_reported` | DONE | |
| T020 | `load_workbook()` | DONE | Strips header whitespace (real headers have trailing `\n`, §0) |
| T021 | `match_videos()` | DONE | See T023 - key was revised after the real-data run |
| T022 | `build_exceptions_report()` + `cli.py catalog` | DONE | Also pulled forward `video_io.py::probe()` (T024's actual scope only, not the rest of Phase 3) since FR-004's duration-mismatch check needs it - see §1 |
| T023 | Run against real local video mirror | DONE | **Found and fixed a real matching bug** - see below |

**T023 finding (important):** the first real run matched only 141/352 trials (should have been ~328-337).
Root cause: `match_videos()`'s key originally included the video filename's leading letter (`F_`/`f_`)
treated as sex, matched against the workbook's `Sex (M/F):` column. Checked directly against real data:
**all 337 real video files use `F`/`f` as the leading letter** (322 `F_`, 15 `f_` - zero `M_` files exist),
while the workbook's `Sex (M/F):` column is a real, roughly-balanced 198 M / 155 F. The filename letter does
not encode fish sex (most likely short for "Fish", not "Female") - PRD C11 never claims it does, only that
casing/zero-padding are inconsistent; the sex-encoding assumption was mine, not the PRD's, and it was wrong.
Fix: matching key changed to `(compound, subject_number)`, dropping sex entirely. Verified safe before
committing to the fix: in the real workbook, max 2 rows share a `(compound, subject_number)` pair (exactly
the known C13 duplicate, already collapsed pre-matching) and zero subject numbers span more than one
compound - so `(compound, subject_number)` is a genuinely unique, safe key. Re-ran: **328/352 matched**
(9 remaining real videos have ambiguous filenames - see below - correctly left unmatched rather than
guessed). All 7 synthetic-fixture tests (T015-T019 plus 2 extra) still pass unchanged, since none of them
happened to use a sex/filename-letter mismatch that would have exposed the bug - a reminder that synthetic
fixtures alone would not have caught this; the real-data run was load-bearing, not a formality.

**T023 real-run results** (`PDS_VIDEO_DIR=videos/Phase_1`, `PDS_DB_PATH=sample_data/00_NTT_DataBase.xlsx`):
352 deduped trials, **328 matched**, 24 unmatched trials, 9 unmatched videos, 1 duplicate-row group (Subject
#320, as expected), 20 duration mismatches (tolerance 30s - see below), 0 corrupt videos. Manually verified
every category is a genuine, correctly-categorized case, not a bug:
- Unmatched trials by compound: `Veh` 7, `FD-2-95` 7, `MDMA` 5, `FD-2-96` 3, `FD-2-67` 1, `HS-1-51` 1 -
  matches the folders already marked `Missing` in their real OneDrive names (§0) plus the `HS-1-51` vs
  `HS-1-151` compound-name mismatch found during inventory. Not a matching bug.
- Unmatched videos: `F_0179A.mp4`/`F_0179B.mp4`, `F_310 (2).mp4`, `F_0021(P1).mp4`/`F_0021(P2).mp4`,
  `F_0176A.mp4`/`F_0176B.mp4`, `F_289.mp4` - all genuinely ambiguous real filenames (split-recording parts,
  retake/duplicate takes) that `VIDEO_FILENAME_RE` correctly rejects as not `{letter}_{subject}.mp4`-shaped,
  rather than silently guessing which one is canonical (matches the "must be reported, not silently dropped
  or silently guessed" requirement, §3.3).
- Duration mismatches: verified the tolerance choice against real data rather than guessing a round number -
  across all 328 matched videos, deviation from expected duration has median 2.7s, p95 17.7s, p99 109.7s;
  the smallest of the 20 flagged deviations is 30.6s, giving a clean separation from the tight normal
  cluster. `DEFAULT_DURATION_TOLERANCE_S = 30.0` (in `catalog.py`, overridable via `--duration-tolerance-s`)
  is kept as the default on this basis, not arbitrarily. Also specifically verified the real 3 short-duration
  trials (Subjects 0161/0162/0163, `MTA-5-62`, `Agent Exposure Time (min): 10`) are read correctly per-row
  (expected 600s each, not a hardcoded 1200s) and measured at 600.8-601.3s - correctly NOT flagged, exactly
  the FR-004/§3.3 behavior required. None of the 20 flagged mismatches are these 3 rows.
- Duplicate: Subject #320, row indices matching the real workbook rows found in §0. Correct.

**Exit (PRD, qualified per §0):** catalog produced against the **352 logical / 337 currently-synced**
video set. 328/337 videos matched (the other 9 are genuinely ambiguous filenames, correctly unmatched, not
a bug); the ~24 unmatched trials + `HS-1-51` compound-name mismatch appear as expected, correctly-categorized
exceptions-report entries — not treated as bugs. **Phase 2 DONE** (code review complete, findings fixed).

### Phase 3 — Video I/O & Scene Setup — supports FR-005

harness-os stage: covered under workflow run 167 / spec `PDS-PROD-001`. Code/security review: not yet
dispatched for this phase specifically (queued next, alongside Phase 4 since tracking.py will touch roi.py's
output directly). This phase produced a significant real finding (see T029) that changes how Phase 4/6/7
must be designed - documented in `docs/strain_tracking_notes.md` rather than silently absorbed into code.

| ID | Description | Status | Notes |
|---|---|---|---|
| T024 [P] | `test_probe_reads_actual_fps_and_duration_not_hardcoded` | DONE | Pulled forward into Phase 2 (see there) |
| T025 | `video_io.py::probe()` and `frames()` generator | DONE | `probe()` done in Phase 2; `frames(video_path, stride=1)` generator added here, yields `(true_frame_idx, frame_bgr)` - stride preserves true indices, not a re-numbered count, so `t_sec = frame_idx / fps` stays correct for callers |
| T026 [P] | `test_waterline_detection_on_synthetic_frame` | DONE | Plus an edge-margin test (`test_waterline_detection_ignores_frame_edges`) |
| T027 [P] | `test_background_model_from_sparse_sample` | DONE | Verifies the synthetic moving dot is removed by per-pixel median (>99% of pixels within 3 gray levels of the true background) |
| T028 | `roi.py::estimate_background()` and `detect_waterline()` | DONE | Median-over-sparse-sample background model; row-wise grayscale gradient peak for the edge detector |
| T029 | Spot-check 1 real video per strain | DONE | **Major finding - see below and `docs/strain_tracking_notes.md`** |
| T030 | Adjust `roi.py` thresholds if strain check reveals contrast failures | DONE | No threshold change made - pigmented (Wild-type/ABSL) fish are visually distinguishable in raw frames just like Casper; the real open issue found here was edge *semantics*, not contrast (see T029) |

**T029 finding (important, changes Phase 4/6/7 design):** initial 3-video (1-per-strain) spot-check showed
`detect_waterline()` landing on the beaker's bottom rim, not a visible water-air interface, in all 3 cases -
but advisor review correctly flagged that all 3 were from the same compound folder (`Veh`), too narrow to
generalize. Broadened to 48 videos across all 16 matched compounds (first/middle/last by date per compound).
Findings, in order of importance:
1. **Resolution is not uniform**: 47/48 sampled videos are 320x240, 1 is 192x240 - and the `sample_data/`
   demo video (not part of the 353-trial set) is a *third* value, 304x240, and not representative of the
   main mirror's dominant resolution. Every pixel-unit threshold in Phase 4+ must be resolution-normalized
   (fraction of frame width/height), never a fixed pixel count - added as an explicit action item for
   Phase 4.
2. **The detected edge is consistently strong** (SNR 9.6x-110.5x the frame's own median row-gradient across
   all 48 videos) - not a low-confidence/noise problem.
3. **But it is bimodal in *where* it lands**: ~12/48 videos peak very near the top of frame (1-9% down -
   possibly the true water surface), ~36/48 peak in the lower half to bottom (42-90% down - beaker-bottom
   rim, confirmed by direct visual inspection). `detect_waterline()`'s return value does not reliably
   correspond to the same physical feature across videos.
4. **Design response (deliberately deferred, not resolved here)**: `roi.py::detect_waterline()`'s docstring
   was corrected to stop claiming its output is a validated water surface (it previously overclaimed this,
   caught before the 48-video survey backed it up - or rather, didn't). Phase 4's `tracking.py` is planned to
   record *two* candidate depth signals per frame (top-of-frame-relative and detected-edge-relative) rather
   than commit to one now; the real resolution is Phase 4's own T035 task (debug-overlay video, visually
   reviewed across a larger strain/compound-balanced sample) - full reasoning and raw survey numbers in
   `docs/strain_tracking_notes.md`.

**Exit (PRD):** all Phase 3 tasks complete; `video_io.py`/`roi.py` implemented and tested (full package suite:
50/50 passing). The PRD's own exit bar for this phase ("adjust thresholds if strain spot-check reveals
contrast failures") is met (no contrast failure found), but T029's deeper finding (edge-semantics ambiguity,
not contrast) is carried forward as an explicit, documented open design question for Phase 4/6/7 rather than
papered over - **Phase 3 DONE** on that basis.

| ID | Description | Status | Notes |
|---|---|---|---|
| T024 [P] | `test_probe_reads_actual_fps_and_duration` | NOT_STARTED | |
| T025 | `video_io.py::probe()`/`frames()` | NOT_STARTED | Validate against sample video's measured 29.8329 fps / 35880 frames / 1202.7s (§0) as a real-data sanity check, not just synthetic |
| T026 [P] | `test_waterline_detection_on_synthetic_frame` | NOT_STARTED | |
| T027 [P] | `test_background_model_from_sparse_sample` | NOT_STARTED | |
| T028 | `roi.py::estimate_background()`/`detect_waterline()` | NOT_STARTED | |
| T029 | Spot-check 1 real video per strain | NOT_STARTED | Candidates already identified (§0): Wild-type → `DOB`/`Fentanyl`/`Methylone`/`Veh`; ABSL → `FD-2-67`/`FD-2-95`/`Veh`; Casper → any other folder. All have synced videos, no sync blocker. |
| T030 | Adjust thresholds if strain check reveals contrast failures | NOT_STARTED | |

### Phase 4 — Tracking (`tracking.py`) — FR-005

harness-os stage: covered under workflow run 167 / spec `PDS-PROD-001` / decision 1863 (same as Phase
0-3). Code review: first pass via code-reviewer agent found 3 HIGH findings (fabricated
`y_from_frame_top`, jump-rejection anchor staleness, unaddressed resolution-normalization); first two
fixed. A determinism bug (NFR-002 violation) was found independently during this pass's own
investigation, not by the code reviewer, and fixed the same way. Follow-up review pass on the fix delta:
**APPROVE** (0 CRITICAL/HIGH, 3 MEDIUM/3 LOW, none blocking) — addressed: `pyproject.toml` missing
`pythonpath = ["."]` (fixed, plain `pytest`/`uv run pytest` now works without a `PYTHONPATH` workaround);
`y_from_frame_top`/`y` docstring rationale corrected (it had claimed `y` was "unambiguous" when it in fact
carries the identical fabrication risk `y_from_frame_top` was just fixed for). Left as noted, not fixed:
`track_video_with_context`'s untyped `**kwargs` (LOW), reacquisition-threshold not being fps-normalized
(LOW, same class of issue as the already-deferred resolution-normalization item).

**Important correction to the HIGH #2 regression tests themselves** (found while addressing the
reviewer's "test only exercises the no-contour path" MEDIUM finding, so worth recording — the review
process caught a problem one level up from what it was looking for): both `test_reacquires_after_*` tests
originally built their "establish a stale anchor" phase from a dot held at a **fixed** pixel position from
frame 0. Verified directly: KNN treats a spatially-constant object as background from the very first
frames (not gradually absorbed like a moving one) — the dot was **never detected at all** in that phase,
so `last_detected_xy` stayed `None` throughout and the tests were passing trivially (no real anchor
existed to test the reset against), despite the earlier code-reviewer pass approving the fix based on
these tests. Fixed by rebuilding the anchor-establishing phase around a genuinely *moving* dot
(empirically confirmed to produce real `detected=True` frames), and adding an explicit assertion
(`assert any(row.detected for row in track[:40])`) so this failure mode can't silently recur. Also found,
while building the second test's "gap filled with rejected jumps not missing contours" variant: a
**stationary** dot at the far position gets absorbed into the background within well under 30 frames too
(same underlying KNN behavior), so it stops producing a contour before the 30-miss reset could ever fire
on the rejected-jump path specifically — fixed by making the far-position phase drift locally as well.
Both fixes verified with a real RED check: manually replaying the pre-fix (no-reset) tracking logic
against the corrected test video reproduces 10/10 tail frames `detected=False`, confirming the rebuilt
tests do catch the original bug, not just look like they do.

| ID | Description | Status | Notes |
|---|---|---|---|
| T031 [P] | `test_tracks_synthetic_moving_dot` | DONE | Split into 3 tests after the KNN rewrite: trajectory spot-check (frames 30/75/120/149), explicit 4-frame-warmup-window assertion, and a >=95% overall detection-rate floor on the synthetic fixture |
| T032 [P] | `test_detected_false_on_blank_frames` | DONE | Also asserts `y_from_frame_top is None` on every undetected frame (not just `x`/`y == 0.0`) after the HIGH #1 fix below |
| T033 | `track_video()` | DONE | Rewritten mid-phase from static-median absdiff to `cv2.createBackgroundSubtractorKNN`; real-footage validation is a much bigger open question than "done" implies — see finding below. `y_from_frame_top` fabrication (HIGH #1) fixed: RED-verified via `test_track_round_trip_undetected`, now `None` when `detected=False`, mirroring `orientation_deg` |
| T034 [P] | `detected` confidence handling | DONE | Area-band contour filter + frame-to-frame jump rejection. Anchor-staleness bug (HIGH #2, code review) fixed: `last_detected_xy` now resets after `DEFAULT_MAX_CONSECUTIVE_MISSES_BEFORE_REACQUIRE` (30) consecutive misses. RED-verified: `test_reacquires_after_long_gap_instead_of_permanently_rejecting_distant_position` — manually confirmed the old (no-reset) logic leaves the tail of that test's video permanently `detected=False` (10/10 tail frames), the new logic detects the reappeared position correctly. Thresholds remain provisional/uncalibrated (Phase 6/T045) and not resolution-normalized (open risk below) |
| T035 | Run over 10-15 real videos, debug-overlay export | DONE | Final, authoritative 12-video numbers: mean 76.1% detected (51.6%-93.9%), essentially tied with the abandoned median-diff approach (74.7% mean) but fixing its catastrophic case (Subject 57: 45.1%→73.1%) and now NFR-002-deterministic. **Decision point raised to the user 2026-09-23** (no PRD-stated numeric tracking-quality bar exists; NFR-005's "typical (mostly-correct) video reviewable well under 20 minutes" was offered as the closest inferred proxy, not a stated threshold): proceed to Phase 5 now, treating FR-015's hard-block-until-reviewer-resolves-Undetermined workflow as the designed absorber for the current detection quality, rather than building an appearance-based detector first or routing low-detection videos to the exceptions report. Recorded here as a known, accepted limitation to revisit if reviewer time on low-detection videos (~52% floor, e.g. Subject 58) turns out to violate NFR-005 in practice — not a silently-dropped concern |

**T033/T035 finding (the real story of this phase — required a mid-phase rewrite, and two corrections to
this doc):** the first `track_video()` implementation used `cv2.absdiff(frame, background)` against
`roi.py`'s static per-video median background (`estimate_background()`, sampled every 30th frame). That
failed badly whenever a fish rested in roughly the same spot across enough of the sparse 30-frame-stride
samples — the median absorbed the fish's own silhouette into the "background." Measured on Subject 0057
(`Veh`, Wild-type): only **44.9%** of 35865 frames detected, confirmed by inspecting the extracted
background PNG (a faint fish-shaped smudge). A full 12-video sample under this implementation
(`/tmp/pds_t035/summary.csv`) showed detection ranging **42.9%-96.8%**, mean **74.7%**.

Switched to `cv2.createBackgroundSubtractorKNN` on the theory that a running per-pixel model (vs. one
static snapshot) would stop a still fish from being absorbed. **Correction #1:** this document and the
`tracking.py` docstring briefly recorded **97.9%** detected on Subject 0057 under KNN. That number was
never reproduced in this session and does not match direct re-measurement (**56.4%** on the full video) —
retracted as a bad transcription from an interrupted prior session, not a real result. **Correction #2:**
the first 12-video KNN comparison (before the HIGH #2 fix) showed KNN performing *worse* than the original
approach on most videos:

| Subject | Old (median-diff) % | New (KNN, pre-fix) % | Δ |
|---|---|---|---|
| 1 | 84.2 | 85.7 | +1.5 |
| 57 | 45.1 | 56.6 | +11.5 |
| 307 | 83.4 | 85.7 | +2.3 |
| 2 | 88.4 | 57.6 | -30.8 |
| 22 | 42.9 | 45.8 | +2.9 |
| 42 | 81.7 | 59.1 | -22.6 |
| 58 | 59.1 | 31.8 | -27.3 |
| 82 | 68.4 | 42.9 | -25.5 |
| 106 | 84.2 | 79.3 | -4.9 |
| 130 | 69.4 | 22.5 | -46.9 |
| 165 | 92.5 | 75.5 | -17.0 |
| 181 | 96.8 | 69.5 | -27.3 |

Mean dropped from 74.7% (old) to 59.3% (new, pre-fix) — KNN was worse on 8/12 videos. Two hypotheses were
checked, not assumed, before trusting this table:

1. *Is the old approach's high rate inflated by false-positive glare/reflection, not real fish
   detections?* Checked by extracting frames from the old run's rendered debug-overlay video (the marker
   position is baked into that video, so the original absdiff code didn't need to still exist) at frame
   indices where the old run reported `detected=True`, for the two largest regressions (Subjects 130 and
   181), 3 spot-checks each. **All 6 markers land on the fish**, not glare. The old approach's rate is real
   detections, not an artifact — this hypothesis was wrong and is recorded here so it isn't re-litigated.
2. *Does a larger KNN `history` window fix the absorption-of-a-still-fish problem?* Swept
   `history` ∈ {500, 2000, 5000, 10000} on Subject 0057: 56.4% / 60.4% / 56.2% / 63.0%. Not monotone in
   `history` (5000 is worse than 2000), so this is measurement noise, not a trend — `history` tuning alone
   is not the fix, and no default was changed off this sweep.

The real remaining confound going into the fixed-code re-run: code review separately found the
jump-rejection anchor never reset across a gap (HIGH #2, now fixed above), and KNN's observed undetected
runs on real footage reach up to **2214 frames** (Subject 0057 run-length analysis: 1198 total gaps, mostly
short, but 2 runs of 871 and 2214 frames) — so KNN's numbers in the table above were penalized by a bug
that median-diff's shorter gap structure triggered less. **This confound must be resolved (HIGH #2 fix +
12-video re-run) before concluding anything about which subtractor is actually better.**

Also investigated directly (not assumed): visually inspected a raw frame from inside a long undetected
run on Subject 0057 (frame 14344) — the fish is plainly visible, resting motionless near the tank bottom,
no occlusion. This rules out "the footage is genuinely unrecoverable" and confirms these are real
detection-pipeline gaps. This matters beyond the raw percentage: sustained stillness is exactly what
Freezing/Drift, Listing/LORR, and Dead need to measure (FR-006/FR-007) — if either subtractor's own
background model swallows a genuinely still fish, that corrupts the states this pipeline most needs to
get right, and if the fixed 12-video number still lands both approaches around 60-75%, that's a real
open problem for Phase 4/5, not a resolved one. Neither approach should be assumed adequate on the
strength of one video's number again — that assumption is what produced both retractions above.

**HIGH #2 fix re-validated with the 12-video sample:**

| Subject | Old (median-diff) % | KNN + HIGH#2-fix % | Δ |
|---|---|---|---|
| 1 | 84.2 | 93.7 | +9.5 |
| 57 | 45.1 | 74.2 | +29.1 |
| 307 | 83.4 | 87.5 | +4.1 |
| 2 | 88.4 | 82.0 | -6.4 |
| 22 | 42.9 | 74.3 | +31.4 |
| 42 | 81.7 | 73.9 | -7.8 |
| 58 | 59.1 | 52.0 | -7.1 |
| 82 | 68.4 | 63.8 | -4.6 |
| 106 | 84.2 | 83.9 | -0.3 |
| 130 | 69.4 | 74.9 | +5.5 |
| 165 | 92.5 | 79.0 | -13.5 |
| 181 | 96.8 | 73.9 | -22.9 |

Mean: old 74.7%, KNN+fix 76.1% — roughly at parity (7 videos worse, 5 better), a much smaller and more
even spread than the pre-fix table (8 worse, mean 59.3%), confirming the anchor-staleness bug explains
most of the earlier regression. The worst-case floor also improved: min old 42.9% (Subject 22) vs.
min KNN+fix 52.0% (Subject 58) — the videos that most benefited (57: +29.1, 22: +31.4) are exactly the
ones where the original median-absorption bug hit hardest.

Re-acquisition correctness check (per the caveat above — the fix accepts the next contour unconditionally
after a 30-miss gap, so a rate gain could be glare, not fish): extracted frames at re-acquisition points
(first `detected=True` after a ≥30-frame gap) for Subject 1 (3 spot-checks) and Subject 130 (3
spot-checks, the largest pre-fix regression). **5/6 markers land on the fish.** One (Subject 130, first
attempt) landed above the beaker rim — investigated, and traced to the determinism finding below (the
verification script's second `track_video()` call produced a different track than the one used to render
the overlay it was checking against, not a real false-positive); re-verified with a single consistent
call and the same frame index correctly shows the marker on the fish. No genuine false positive found at
a re-acquisition point in this sample.

**Determinism finding (NFR-002 violation, found while investigating the mismatch above):** two
`track_video()` calls on the identical Subject 130 video, no seeding between them, produced different
`Track` lists — 1469-1565/35860 frames (4.10%-4.36%) differed in `detected`, though the *aggregate*
`pct_detected` was stable across 3 runs (75.5%/75.4%/75.1%, within ~0.4pp). Root cause:
`cv2.createBackgroundSubtractorKNN`'s internal sample-replacement history update uses OpenCV's global
RNG, which is not exposed per-subtractor and is not seeded anywhere in this codebase — a real violation
of NFR-002 ("deterministic tracking/classification, no unseeded randomness"), independent of which
subtractor turns out to have the better detection rate. Verified the fix before applying it: called
`cv2.setRNGSeed(42)` before each of two `track_video()` calls (on both the synthetic fixture and the
real Subject 130 video) and got byte-identical `Track` lists both times. Fixed: `track_video()` now
calls `cv2.setRNGSeed()` with a fixed default (`DEFAULT_TRACKING_RANDOM_SEED`) at the top of every call.
RED-verified via `test_track_video_is_deterministic_across_repeated_calls` (asserts full equality of two
back-to-back runs on the synthetic fixture, not just a similar rate) — the unseeded pre-fix behavior was
confirmed non-deterministic by direct measurement above, not just inferred. Caveat carried into Phase 12:
`cv2.setRNGSeed()` is process-global state, so `track_video()` is not thread-safe with respect to other
concurrent OpenCV RNG use in the same process — fine for sequential per-video batch processing, a real
constraint if that model changes.

The 12-video table above and its overlay-derived re-acquisition checks were run **before** this fix
(unseeded). **Final, authoritative re-run under the now-deterministic implementation**
(`/tmp/pds_t035_final/summary.csv`):

| Subject | Old (median-diff) % | Final (deterministic KNN+fix) % | Δ |
|---|---|---|---|
| 1 | 84.2 | 93.9 | +9.7 |
| 57 | 45.1 | 73.1 | +28.0 |
| 307 | 83.4 | 87.5 | +4.1 |
| 2 | 88.4 | 82.1 | -6.3 |
| 22 | 42.9 | 74.5 | +31.6 |
| 42 | 81.7 | 74.1 | -7.6 |
| 58 | 59.1 | 51.6 | -7.5 |
| 82 | 68.4 | 64.4 | -4.0 |
| 106 | 84.2 | 84.5 | +0.3 |
| 130 | 69.4 | 75.4 | +6.0 |
| 165 | 92.5 | 78.4 | -14.1 |
| 181 | 96.8 | 74.3 | -22.5 |

Mean: old 74.7%, final 76.1% (6 worse, 6 better) — confirms the pre-fix/post-fix numbers were within the
measured aggregate noise band (~0.4pp), not meaningfully different from each other. **This is the number
Phase 4's exit decision is based on.**

**Is the ~24% miss concentrated during stillness (the states this pipeline most needs to measure), or
spread evenly?** Checked directly rather than assumed, since this materially changes how serious the gap
is. The frame 14344 visual finding (above) showed one clear case: a resting, motionless, unoccluded fish
went undetected. But a quantitative check on Subject 57 — median frame-to-frame velocity among consecutive
detected-frame pairs (2.48 px/frame, n=24847) vs. velocity in the last detected frame immediately before
each undetected run of ≥10 frames (median 4.67 px/frame, n=229; fraction in the bottom quartile of
velocity: 16% pre-gap vs. 25% baseline) — showed pre-gap frames were *faster*-moving than baseline, not
slower. **Mixed evidence, not a clean confirmation either way**: stillness can clearly cause a miss (frame
14344), but misses are not obviously concentrated in low-motion periods in aggregate on this video. Given
this ambiguity plus the NFR-005 framing below, the decision was raised to the user rather than resolved
unilaterally on either reading of the evidence.

**Decision (2026-09-23, per user input):** proceed to Phase 5 with the current tracking implementation
and quality (76.1% mean / 51.6% floor across the 12-video sample), treating FR-015's hard block on
Accept-while-Undetermined as the designed mechanism for a reviewer to resolve the gap, rather than
building an appearance-based detector first or routing low-detection videos to the exceptions report.
This was presented as a judgment call, not a PRD-mandated threshold — the PRD states no numeric
tracking-quality bar; NFR-005 ("a reviewer must be able to review and accept a typical (mostly-correct)
video in well under the video's own 20-minute runtime") was offered as the closest inferred proxy, and a
~52% floor (Subject 58) plausibly strains it, but this is an inference, not a stated requirement being
violated. Recorded here as a known, explicitly-accepted limitation - revisit if reviewer time on
low-detection videos turns out to violate NFR-005 in practice during Phase 10/11 (review UI) or Phase 13
(full integration pass).

One real, expected consequence of the KNN switch, unrelated to the detection-rate/determinism questions
above: KNN's first `apply()` calls have no prior history to compare against, so **the first few frames of
every video are structurally never detected** (not an occlusion/glare edge case — a warmup property of
the algorithm). Initially assumed (incorrectly) to be a single frame 0 — measuring it directly showed the
leading undetected run is exactly **4 frames**, confirmed identically on both the synthetic fixture (150
frames, stable across 5 repeated runs) and the full 35865-frame Subject 0057 video, so it is a fixed
algorithmic property independent of video length. `tests/test_tracking.py` asserts this explicitly
(`test_leading_warmup_window_is_exactly_four_frames_knn_has_no_prior_history`). Downstream consumers
(Phase 5 features, Phase 7 labeling) must not assume the start of a trial video is a valid data point —
this is a natural fit with the existing `detected=False` → Undetermined handling already required by
§3.3, but combined with the much larger detection gaps above, it means Undetermined segments may be
common and non-trivial in length across a trial video, which Phase 7's design must account for
deliberately (FR-015 hard-blocks Accept on Undetermined) rather than discover later as an "most videos
need manual relabeling" surprise.

**Open risk flagged for Phase 6 (not yet addressed in Phase 4):** the pixel-unit constants in
`tracking.py` (`DEFAULT_MAX_JUMP_PX`, `DEFAULT_MIN/MAX_CONTOUR_AREA_PX2`) are not resolution-normalized,
despite T029's explicit finding that the real-video sample spans two resolutions (320×240 for ~98% of
videos, one outlier at 192×240 — docs/strain_tracking_notes.md §2). Left as-is deliberately since these
are still provisional defaults pending Phase 6's calibration search (T045), not because the risk is
resolved — T045 must either search per-resolution or normalize by frame size, so the search doesn't
silently fit only the dominant resolution and mis-threshold the outlier video. Documented as a caveat in
`tracking.py`'s module docstring and added as an explicit note on T045 below.

**Also renamed during this pass:** `Track.depth_from_surface` → `Track.y_from_frame_top`. The old name
invited a Phase 5/7 author to assume it was already surface-relative depth (it's raw top-of-frame pixel
y — T029 found the detected background edge is not reliably the true water surface). `FrameRow`'s
later-stage `depth_from_surface` field keeps its name unchanged, since by that point in the pipeline it
is expected to carry the calibration-aware value.

### Phase 5 — Features (`features.py`) — FR-006

harness-os stage: covered under workflow run 167 / spec `PDS-PROD-001` / decision 1863 (same as Phase
0-4). Code review: **BLOCK** on first pass — 1 CRITICAL, 1 MEDIUM, 2 LOW. Both correctness findings
fixed, RED-verified, and re-checked against real footage; see finding below.

| ID | Description | Status | Notes |
|---|---|---|---|
| T036 [P] | `test_velocity_matches_finite_difference` | DONE | Plus a suite of gating tests (undetected-gap, frame_idx-gap, non-positive-`dt` defensive guard) — see design decisions below |
| T037 [P] | `test_immobility_flag_on_sustained_low_velocity` | DONE | Caught a real off-by-one in its own first implementation — see finding below |
| T038 [P] | `test_meander_definition_matches_thesis_metric` | DONE | Checked `docs/zebrafish_drug_detection_scope.md` §4.1 before starting: it has **no explicit "Meander" formula** — only "movement smoothness and turning-angle variance" as prose, no named metric. The cross-reference in this task's original note was based on a false premise. Implemented directly against PRD §9.5.4's own definition instead: "a rolling meander measure (path curvature normalized by distance, matching the thesis's own 'Meander' definition)" — PRD asserts the thesis has one, but the scope doc available in this repo doesn't state it explicitly, so "path curvature normalized by distance" is the operative definition implemented and tested against. Implemented as *trajectory* tortuosity (position-derived heading), not body-orientation curvature — see design decisions below |
| T039 | `derive_features()` | DONE | `src/prepds/features.py`, 100% test coverage (17 unit tests + 4 gap-coverage tests = 21 total). Verified against real tracked footage (Subject 0057, Subject 0001, a Fentanyl video) — no NaN/inf on any of ~36k-frame real videos; see real-data finding below |

**Design decisions (PRD §9.5.4 leaves several genuinely open, consistent with Phase 3/4's pattern):**

1. **`smoothness` field dropped, `meander` kept — a real PRD inconsistency, not a resolved ambiguity.**
   §5.1's `FeatureFrame` entity table lists `frame_idx, velocity, acceleration, angular_velocity,
   meander, smoothness, is_immobile`; §5.4's actual `frames.parquet` schema (the persisted, tested
   contract) has no `smoothness` column; FR-006 phrases it as "a path-smoothness **or** meander
   measure" — alternatives, not two features. Implemented only `meander`, matching the schema that's
   actually tested against (`models.py::FeatureFrame` has 6 fields, no `smoothness`). Flagged here
   explicitly in case a later phase (Phase 6's calibration YAML, Phase 9's export) expects a distinct
   `smoothness` column — if so, this is where the gap was first found and the omission was deliberate.
2. **`orientation_deg` (from `cv2.fitEllipse`) is periodic with period 180, not 360** — verified
   empirically (not assumed): fit an elongated synthetic ellipse contour at known rotations 0-359deg
   and confirmed the returned angle repeats every 180deg (an ellipse's major axis has no head/tail
   distinction — 0deg and 180deg rotations fit to the same angle). `angular_velocity`'s circular
   difference uses `period=180`; getting this wrong would silently produce angular velocities wrong by
   up to 2x at wraparound boundaries, with no test failure to catch it (the bug is only visible right at
   the 180/0 seam).
3. **`meander` is trajectory tortuosity (position-derived heading), not body-orientation curvature.**
   PRD's phrase "path curvature normalized by distance" is ambiguous between the two. Chose trajectory
   heading deliberately: it keeps `meander` independent from `angular_velocity` (which already carries
   the body-orientation signal) — the labeling GMM needs three genuinely distinct input dimensions
   `[velocity, angular_velocity_variance, meander]` per §9.5.6, and conflating meander with body
   orientation would make two of the three inputs near-redundant. Heading (`atan2(dy, dx)`) is a genuine
   360-periodic direction (unlike body orientation — "moving left" and "moving right" are different
   headings, not the same axis), so its circular diff uses `period=360`.
4. **Every kinematic field uses the `0.0`-sentinel-on-not-computable convention**, matching
   `Track.x`/`Track.y` (never `None` — `frames.parquet` types these non-nullable). Each field has its
   own minimum consecutive-detected-history requirement before a value counts as real, not fabricated:
   velocity needs 2 consecutive detected+frame_idx-adjacent frames, acceleration needs 3 (so it can
   difference two *real* velocities, not difference against another sentinel), angular_velocity needs 2
   consecutive detected frames that both *also* have a non-`None` `orientation_deg` (a detected frame
   can still fail `cv2.fitEllipse` — too small a contour — a real Phase 4 output shape, tested
   explicitly). `is_immobile` requires a full rolling window of REAL (non-sentinel) velocities — fails
   open to `False`, not True, if the window includes an undetected frame or the window's own start
   (never infers sustained stillness across an unconfirmed stretch). Downstream (Phase 7 labeling.py)
   must gate on the frame's own `detected` flag for the Undetermined case, never infer it from a `0.0`
   feature value — this is the same principle established in Phase 4, just extended to derived features.

**T037 finding (an off-by-one caught by the test suite, not by inspection):** `_compute_immobility`'s
first implementation used `run_length[i] < window_frames` as its readiness gate. This is wrong by one:
the earliest velocity sample inside a `window_frames`-wide window is *always* a run-start sentinel
(`0.0`, not a real measurement) unless the run is at least `window_frames + 1` frames long. With the
off-by-one, a stationary synthetic test video (`test_immobility_flag_on_sustained_low_velocity`, all
frames at velocity truly `0.0`) failed to catch this — the sentinel and the real value are numerically
identical when the fish is genuinely motionless, so the bug was invisible until the specific frame-count
boundary assertion (`features[window - 1].is_immobile is False`) caught the actual RED failure
(`True` returned instead of `False`). Fixed to `run_length[i] < window_frames + 1`; documented in the
function's own docstring so the reasoning isn't lost.

**Real-data finding (Fentanyl video, Subject 0106 — expected to show LORR-driven stillness):** verified
`derive_features()` produces finite, sane output on real ~36k-frame tracked footage, but
`is_immobile` was `False` for **every single frame** at the provisional `DEFAULT_IMMOBILITY_VELOCITY_
FLOOR_PX_PER_S = 5.0`. Investigated rather than assumed correct: confirmed the mechanism itself isn't
broken (a floor sweep — 5/15/30/50/75 px/s — shows the flag starts firing at 75 px/s, and there are
10,455 frames with enough consecutive-detected history to evaluate a window at all), so this is a
threshold-calibration gap, not a logic bug. Root cause read from the data: 484 frames is the longest
single consecutive-detected run in this video, and even within long detected runs, real frame-to-frame
centroid jitter from Phase 4's still-uncalibrated tracking rarely drops below ~5 px/s for a full 30-frame
window — the *actual* sustained-stillness periods more likely fall inside the long undetected gaps
Phase 4 already found (up to 2214 frames), where `detected=False` correctly forces `Undetermined`
instead. This is the same tension Phase 4's investigation surfaced (a genuinely still fish is also the
case most likely to lose tracking) showing up again downstream, not a new problem - and it's additional,
concrete evidence for Phase 6's calibration search needing a much less strict default than this
placeholder. Separately: real velocity distributions are heavy-tailed (Subject 0057: median 74.9 px/s,
p99 1632 px/s, with 0.52% of frames sitting right at the ~1800 px/s ceiling implied by `tracking.py`'s
`DEFAULT_MAX_JUMP_PX=60` at ~30fps) — consistent with detection centroid jitter near the jump-rejection
boundary, not literal fish speed, and another concrete argument for Phase 6 to calibrate
`tracking.py`'s and `features.py`'s provisional constants together rather than independently.

**Open risk carried into Phase 6/T045 (same class as Phase 4's already-flagged items):**
`DEFAULT_IMMOBILITY_WINDOW_FRAMES` and `DEFAULT_MEANDER_WINDOW_FRAMES` are frame counts, not durations
(fps-normalization gap), and `velocity`'s pixel units (and therefore the immobility floor) are not
resolution-normalized — same open risk as Phase 4's contour-area/jump-distance constants. All deferred
to the same calibration pass, documented in `features.py`'s module docstring.

**Code review finding (CRITICAL, fixed): `meander` leaked stale values across detection gaps.**
`_compute_meanders`'s rolling-window sum was indexed purely by list position
(`range(max(0, i-window_frames+1), i+1)`), with no check that the window actually stayed within one
continuous detected run. On real footage this fires routinely, not as an edge case: detection gaps under
30 frames (the default `meander_window_frames`) are common (Phase 4's Subject 0057 data: 1198 total
gaps, most under 50 frames). Two concrete failures this caused, both reproduced and fixed: (1) an
undetected frame right after a real zigzag reported the zigzag's leftover turn/distance ratio instead of
the documented `0.0` sentinel — directly violating this module's own "never fabricate from missing data"
contract, worse than every other field's convention (which all correctly gate to `0.0` on `detected`);
(2) a frame soon after reacquisition silently blended pre-gap and post-gap trajectory segments into one
ratio, misrepresenting a genuinely straight post-reacquisition path as still-tortuous. Fixed with a
one-line change: clamp the window's lower bound to `i - run_length[i] + 1` in addition to the existing
`window_frames` bound — this makes the window naturally empty (and `meanders[i] == 0.0` via the existing
zero-distance guard) for any frame with `run_length[i] < 2`, with no separate special case needed. Two
regression tests added (`test_meander_is_sentinel_zero_on_undetected_frames_not_a_stale_value`,
`test_meander_does_not_blend_pre_gap_and_post_gap_trajectory`), both RED-verified against the pre-fix
code, then GREEN after the fix. Re-verified against real footage (Subject 0057): all 9657 undetected
frames now report the `0.0` sentinel (previously contaminated), all values finite.

**Code review finding (MEDIUM, fixed): a `dt<=0` "sentinel" velocity was silently treated as a real
measurement by acceleration and immobility.** `run_length[i] >= 2` is necessary but not sufficient for
"`velocities[i]` is a real measurement" — two consecutive detected, frame_idx-adjacent frames can still
share a non-increasing `t_sec` (a defensive case, not expected from `video_io` in practice), which also
yields the `0.0` sentinel via a separate guard `_compute_velocities` already had. Both
`_compute_accelerations` and `_compute_immobility` gated only on `run_length`, so a `dt<=0` sentinel could
be differenced against as if it were a real prior velocity (acceleration), or silently count as a real
below-floor sample inside an immobility window (contradicting that function's own documented "fails open
to `False`" contract). Fixed by having `_compute_velocities` return a `velocity_valid` mask alongside the
values, and requiring it directly in both consumers instead of re-deriving validity from `run_length`.
Two regression tests added and RED-verified
(`test_acceleration_does_not_treat_a_dt_sentinel_velocity_as_real_history`,
`test_immobility_flag_does_not_count_a_dt_sentinel_velocity_as_a_real_below_floor_sample`). One
consequence: the old separate `dt<=0` guard inside `_compute_accelerations` itself became dead code
(`velocity_valid[i]` already guarantees that same `dt>0`, since it's the identical computation) — removed,
confirmed by coverage (`features.py` back to 100%).

**Code review findings (LOW, fixed): `_circular_diff`'s docstring stated the wrong output range**
(`(-period/2, period/2]` vs. the actual `[-period/2, period/2)` — Python's `%` on floats returns
`[0, period)`, so the seam lands on the negative side, not the positive side; the ambiguity at exactly
half a period is genuine, only the docstring's claimed range was wrong) — corrected. **Also flagged, not
fixed (non-blocking):** `_compute_meanders` is O(n × window_frames); measured directly at 36,000
synthetic frames with the default window, ~0.12s — negligible next to video decode/tracking cost even
across hundreds of videos, so left as-is with a note for a future prefix-sum optimization if Phase 12's
actual batch timing ever shows otherwise.

### Phase 6 — Calibration — FR-008

harness-os stage: *not yet run*

| ID | Description | Status | Notes |
|---|---|---|---|
| T040 [P] | `test_axis_pixel_to_seconds_mapping` | NOT_STARTED | |
| T041 [P] | `test_color_nearest_match_to_palette` (incl. white→Dead) | NOT_STARTED | |
| T042 | `digitize_reference.py` | NOT_STARTED | Both reference PNGs present locally (§0); 5 groups (VEH/MDMA/Methylone/DOB/Fentanyl) all have synced videos — confirm 100µM Fentanyl panel presence at implementation time per PRD note |
| T043 | Confirm digitized targets sane → `reference_targets.json` | NOT_STARTED | |
| T044 [P] | `test_search_reduces_distance_to_target` | NOT_STARTED | |
| T045 | `calibrate.py` search | NOT_STARTED | Depends on Phase 4/5 validated on real data (per §12 Dependencies) and on Phase 7's rule *code* (not values) existing — see §1 sequencing note. Must also resolve open risks from both Phase 4 and Phase 5: (1) resolution-normalization (320×240 vs 192×240 in the real sample, docs/strain_tracking_notes.md §2) for `tracking.py`'s `DEFAULT_MAX_JUMP_PX`/`DEFAULT_MIN/MAX_CONTOUR_AREA_PX2` and `features.py`'s velocity-derived immobility floor — search per-resolution or normalize by frame size, not just fit the dominant resolution; (2) fps-normalization for `tracking.py`'s `DEFAULT_MAX_CONSECUTIVE_MISSES_BEFORE_REACQUIRE` and `features.py`'s `DEFAULT_IMMOBILITY_WINDOW_FRAMES`/`DEFAULT_MEANDER_WINDOW_FRAMES` (frame counts, not durations); (3) `tracking.py` and `features.py` constants likely need calibrating *together*, not independently — Phase 5's real-data check found `DEFAULT_IMMOBILITY_VELOCITY_FLOOR_PX_PER_S=5.0` never fires on real footage because Phase 4's still-uncalibrated tracking noise floor sits above it, and the ~1800 px/s velocity ceiling some frames hit is a direct artifact of `tracking.py`'s `DEFAULT_MAX_JUMP_PX=60`, not real fish speed |
| T046 | Full calibration run, freeze `default_thresholds.yaml` | NOT_STARTED | Record `random_state` seed for GMM inside the CalibrationProfile (NFR-002, §1) |

**Exit (PRD):** versioned frozen threshold set checked in (YAML only, never the reference PNGs —
C14). Not yet reached.

### Phase 6 — running notes (T040–T045 implemented; T046 full run pending)

**Digitizer (T040–T043) — DONE, real targets written.** `palette.py`, `calibration/digitize_reference.py`,
`tests/test_digitize_reference.py` (20 tests), output `calibration/reference_targets.json` (9 group targets).
Findings that changed the design, each verified against the real images rather than assumed:
- **Plot geometry is hardcoded** (advisor was right): a first auto-detector missed panels and picked up
  the row-label text. Bounds were measured once from colour-band structure; 5 panels/image, shared x-extent
  per image (332–1553 and 345–1705 px for 0–1200 s).
- **Nearest-RGB matching was abandoned for HSV rules.** The figures are heavily JPEG-blurred/darkened
  (plot-body blue ≈ (17,7,176), not (0,0,255)); nearest-RGB snapped red/blue blends onto the gray between
  them, giving VEH gray 54% (truth ≈ 30%). HSV rules reject blends instead of guessing.
- **A synthetic ground-truth check exposed two more biases** (blur+JPEG figure with known proportions):
  gray was under-counted ~5–6pp (chroma leaking from neighbouring rows past the S≤0.10 cutoff → widened to
  0.25, error → <2pp), and a 1–2px "light reddish" overshoot band at red↔gray boundaries manufactured ~4.5%
  phantom Listing/LORR on vehicle panels (reference ≈ 0%). A vertical-stability filter fixed the phantom but
  biased toward frequent states (4.6pp), so it was dropped; instead pink is now S≤0.32, V≥0.94 (real pink
  measured S 0.19–0.31, V≥0.94), cutting phantom pink to ≤0.6% at the cost of ~7–8% of genuine pink pixels
  (a small, known low bias on pink). Top/bottom 3 rows are skipped (white margin bleed inflated Dead 2.0%→0.6%).
- **Advisor sanity checks all pass:** proportions sum to 1; VEH Dead ≤0.7%, VEH pink ≤0.6%; Fentanyl 100µM =
  29.5% pink + 40.9% Dead; Methylone 100µM 61% blue; the two VEH panels agree to ~2.8pp TV (averaged, with a
  TV-agreement guard that raises if duplicates disagree). Rejected-pixel fraction is reported per panel
  (14–43%; expected for ~9px rows under blur).
- **Reference states are Freezing/Drift-heavy** (blue ≈ 30–60% in most groups, even ~32% for VEH) while the
  pipeline produced 0% Freezing — the freeze floor's search range matters enormously.
- **MDMA 100µM has no local videos** (0 matched) — that target cannot be used; 8 of 9 groups are calibratable.
  OCR skipped (pytesseract not installed; group-level aggregate is the primary target).

**Phase 7 bug found by the calibration toy test (fixed): Freezing/Listing bouts were truncated.** The rule
tested "time elapsed since onset ≥ minimum" per frame, so a qualifying bout lost its first `min_bout` seconds
(a 3s bout with a 2s minimum only had its last 1s labeled) — systematically undercounting Freezing/Listing.
Earlier tests only asserted the *last* frame; two of them (`states[5] != FREEZING`) encoded the bug and were
corrected. Fix: `_sustained_run_flags` labels every frame of a maximal contiguous run whose span ≥ minimum.
Regression tests RED-verified.

**Refactor for calibration speed:** `labeling.prepare_video` (validity + frozen-GMM predictions, once per
video) and `classify_prepared[_states]` (cheap rule pass, per search sample); `classify_video(gmm=...)`
delegates to them (equivalence tested).

**Search (T044–T045) — implemented:** `calibration/calibrate.py` (`search`, `evaluate_params`, seeded, TV over
*determined* frames with coverage reported, GMM confidence margin held fixed), `track_cache.py`
(parallel one-time tracking of a calibration subset, cached as gzip JSON under git-ignored `outputs/`).
Tracking parameters are NOT searched (cost: ~43s/video/sample) — a documented deviation from T045.

**First real calibration run (T045/T046 attempt) — result: NOT frozen; needs a decision.** Set: 47 videos
(≤6/group, 8 groups), tracks cached (904s, 10 workers), one shared pooled GMM, 0.45s/evaluation, seeded
search (seed 20260923, 1568 evaluations, 832s). Mean TV fell 0.526 (placeholders) → 0.282. `default_thresholds.yaml`
was deliberately left untouched. Reasons the optimum is not trustworthy:
- **Boundary-hugging:** `velocity_freeze_floor` = 150 (its cap), `min_freeze_bout_s` = 0.33 (its floor),
  `surface_breach_depth_threshold` ≈ 3.1 (floor). A 150 px/s "near-zero velocity" floor is not a stillness
  threshold (median real velocity is 22–66 px/s); the search is relabeling tracker-jitter-level motion as
  Freezing to match the blue-heavy reference. It fits DOB (TV 0.03–0.07) and over-fits VEH (Freezing 57% vs 32%).
- **Listing/LORR is not identifiable from orientation.** Share of detected frames with |orientation−90°| ≥45°
  is 23–27% in *every* group, Fentanyl (22.8–25.5%) included, vs Veh 27.4% — no signal. Pipeline Listing is
  ≤0.5% everywhere while Fentanyl targets are 44%/29.5%, so Fentanyl TV stays 0.44/0.70. Likely the
  `fitEllipse` blob (waterline/reflection/limb noise) does not track body posture. Not a threshold problem.
- **Dead never fires** (tracking dropouts + no-bridging suffix rule); Fentanyl-100 Dead target is 40.9%.
- The velocity distribution *is* informative (median 29–35 px/s at 100µM vs 40–66 at 30µM/veh), so
  velocity-based states are the calibratable part.
- Also noted: the pooled GMM's split is essentially velocity-driven (means: v≈325 vs 26, meander ≈13 vs 104),
  so "Controlled" ≈ slow/jittery, "Erratic" ≈ fast — semantically shaky.
Decision needed before T046 freeze (options: fix/replace the Listing signal e.g. tighter ROI or body-axis from
the contour; re-run tracking-noise-aware velocity; accept a Listing/Dead-blind classifier and lean on FR-015
review; widen ranges — not recommended). Phase 7 part 2 (validate frozen thresholds on held-out videos) blocked
on this.

**Signal-fix attempt (user instruction: tighter ROI / contour body axis) — result: orientation is NOT the LORR
signal; do not adopt a new detector.** Measured on the reference-matched 2026 Fentanyl 30µM cohort (F_331–341):
- Contour body axis (close 7x7 + PCA of largest blob) vs raw `fitEllipse`, early vs late windows, F_332/F_333
  and 2024 fentanyl vs Veh F_0057: no separation with either method (median deviation from horizontal 10–25°
  everywhere; PCA is no better, sometimes noisier). Reason is geometric: in a side view a fish that rolls onto
  its side keeps a horizontal long axis, so axis-angle cannot see the roll. Tighter ROI could only reduce
  false Listing, not create true Listing.
- Cohort time-course test (reference Fentanyl 30 pink fraction per 30 s bin vs cohort-mean feature per bin):
  `dev>=45°` correlates **−0.46** with pink (wrong sign); velocity −0.72, detection rate −0.66, slow-frame
  fraction +0.49, y +0.42. I.e. reference LORR coincides with a still, often *undetected* fish (tracker loses
  it near the reflective bottom), which looks like Freezing/Undetermined, not a distinct posture.
- Correction: the three `_detect_fish` synthetic tests added to `tests/test_tracking.py` passed on unchanged code
  (never RED); they are characterization tests, not evidence of a fix. `tracking.py` was NOT changed.
- Cohort fix still pending: Fentanyl 30 must be subjects 331–341 only (10 tracks now cached); Fentanyl 100µM,
  MDMA 100µM and one VEH cohort have no local videos and should be excluded from the objective.
- Consequence: no Listing threshold can be frozen. Decision for the user: report/exclude Listing from the
  calibration objective (Controlled/Freezing/Surface calibrated on corrected cohorts) and rely on human review,
  or pursue a non-orientation LORR proxy (spec change).

**Non-orientation Listing/LORR proxy (user approved a spec change) — result: no automatic proxy is reliable; Listing goes to manual review.**
Tried, all on real tracks, against the digitized reference:
- Stillness + bottom position (velocity floor 10/25 px/s, y >= 170/190, bout 3/10 s): fires ~0% in every group
  (frame velocity is centroid-jitter dominated; sustained low-velocity bouts do not exist).
- Absolute y is a setup confound, not behaviour: 2026 cohorts (Fentanyl 30, MDMA 30) sit at y~195, 2024 groups at y~157.
  MDMA 30 (2026, ~0% pink) is as low and as still as Fentanyl 30 (44% pink).
- Blob shape (closed-mask largest blob; 15 videos Fentanyl 30 vs MDMA 30, later 36 others): late-window area is larger in
  Fentanyl 30 (521+-167 vs 284+-30 px2) and the lowest blob pixel is ~10 px lower; area/bottom track the reference pink
  time course at r=0.41 / 0.39; rule "10 s window, median area >= 450 and bottom >= 212" gives 48% in Fentanyl 30 (target 44%,
  cohort-level time-course r=0.58) but 14% in MDMA 30 and 26% in the 2026-style vehicle videos (targets ~0.2-0.5%); with a
  2024-geometry floor (bottom >= 170) it fires 42-62% in DOB/methylone/vehicle/Fentanyl-100. Area alone (>=450) fires ~50% everywhere.
  The signal is partial, cohort-specific and geometry-dependent; false positives are not controllable with the local data.
- No per-frame ground truth exists (reference rows are blurred/unlabelled by subject), so per-frame accuracy of any proxy cannot be measured.
Decision: do not freeze any automatic Listing rule. The orientation rule stays disabled-by-calibration (threshold not frozen);
Listing/LORR is labelled manually via the review app (FR-015/FR-015a path). Blob-shape features (area, bbox, lowest pixel) are
a candidate *review aid* only. Scratch scripts: scratchpad/proxy.py, shape_extract*.py, shape_tc.py.

**Corrected-cohort calibration with Listing excluded (7 groups: DOB 30/100, Fentanyl 30 = 331-341, MDMA 30, methylone 30/100, Veh) — result: NOT frozen.**
Added `exclude_states` to `calibrate.evaluate_params` (target renormalised without Listing; RED->GREEN, 196 tests pass).
Search (1249 evals, seed 20260923): mean TV 0.457 (placeholders) -> 0.185, but `velocity_freeze_floor` = 144 px/s (cap 150) and
`min_freeze_bout_s` = 0.33 s (floor 0.3) again hug their bounds, and MDMA 30 (TV 0.36: Freezing 60/46, Controlled 12.5/48) and
methylone 100 (0.27) do not fit. Cause: raw frame-to-frame velocity is centroid-jitter dominated, so the GMM only ever sees the fastest frames.
**Retune of the Controlled/Erratic split (exploration on cached tracks, not yet in the pipeline):**
- A 3-band rule on speed = displacement over a 0.5-1 s lag (Freezing < a <= Controlled < b <= Erratic) reaches in-sample mean TV 0.183 (0.5 s: a=20.5, b=44.9;
  1 s: a=17.8, b=33.7) with interior thresholds that are stable (TV within 0.01 for a in [11,29], b in [39,60] at 0.5 s), i.e. the same fit as the GMM path without boundary-hugging.
- Leave-one-group-out held-out TV is 0.24-0.25 (worst: MDMA 30 0.36, Fentanyl 30 0.29, methylone 100 0.29-0.35), so ~0.06 of the in-sample fit is over-fit.
- Adding path shape (4 s tortuosity, step-length CV) does not help: group medians are indistinguishable (tortuosity ~2.9-3.6, CV ~0.5-0.58 in every group) and LOGO gets worse (0.24-0.27).
- Structural limit: the rule cannot produce the high Controlled share of MDMA 30 (48%) / methylone 100 (33%) with ~6% Erratic; those groups have the same speed distribution as the others with this tracker.
Conclusion: retuning cannot beat ~0.18 in-sample / ~0.24 held-out; the gain from replacing the raw-velocity GMM by a smoothed-speed rule is interpretability and non-degenerate thresholds, not accuracy.

**Adopted the speed-band rule and froze a profile (T046) — `config/calibration_profiles/cal-2026-09-23.yaml`, also copied into `default_thresholds.yaml`.**
- Code: `features.compute_smoothed_speed` (displacement over 1 s, forward fallback at run start, never bridges a detection gap that leaves no detected partner);
  `labeling.py` rewritten track-only (no GMM): Freezing = sustained speed <= floor, Erratic = speed >= threshold, Controlled = between, Undetermined when no computable speed,
  frames carry no confidence; Listing off unless `listing_orientation_deviation_deg` is set (manual review); `erratic_speed_threshold > freeze_speed_floor` enforced (ValueError).
  `calibrate.py`: `exclude_states`, `constrained_objective`, `prepare_groups(videos)`; `calibration/profile.py`: immutable profile writer + `labeling_thresholds`. 213 tests pass.
- Two search results were rejected on the way: (1) unordered thresholds let the search put erratic (~17) below the freeze floor (~60) so Controlled became brief slow blips
  (TV 0.188 but meaningless) -> ordering now enforced; (2) with a 0.3 s bout floor the search squeezed the Controlled band to ~1 px/s -> bout lower bound widened to 0.03 s.
- Surface Breach and Dead thresholds are NOT identifiable (reference mass ~0%: the search turns Surface off / lets the Dead bout wander) -> removed from the search space and
  frozen at placeholders (20 px, 60 s), flagged UNCALIBRATED in the profile.
- Frozen values (seed 1; 3 seeds agree): freeze floor 20.6 px/s, erratic threshold 32.8 px/s (seeds 32.7-35.3), min freeze bout 0.03 s (at search lower bound: gate not needed on smoothed speed), lag 1.0 s.
- Agreement with the reference (Listing excluded, Undetermined excluded): mean TV 0.193 in-sample (placeholders 0.416), **0.266 leave-one-group-out**. Per group in-sample:
  DOB30 0.05, DOB100 0.08, methylone30 0.15, Veh 0.19, Fentanyl30 0.25, MDMA30 0.32, methylone100 0.33. Coverage (determined fraction) 0.51-0.77; on the 45 calibration videos 37.5% of frames are Undetermined.
- Real-track validation: the frozen rules ran over all 45 calibration videos and every result passed `segments.frames_to_segments` (Dead monotonicity re-validated; Dead never fires, Surface ~0%).
- Known limits, not fixed: MDMA30/methylone100 stay poor (same speed distribution as other groups); Dead never fires (tracking dropouts break the terminal run); Listing/LORR manual.

**Profile revision 2 — `cal-2026-09-23-r2` (now the packaged default; r1 stays as the immutable record).** Re-calibrated against the 1 s-consolidated output the pipeline actually emits (`evaluate_params(consolidate=True)`), 2 seeds + leave-one-group-out:
freeze floor 22.7 px/s (seeds 17.9-22.7), erratic threshold 41.2 px/s (40.6-41.2), min freeze bout 0.03 s (lower bound), lag 1.0 s; Surface/Dead still uncalibrated placeholders.
Mean TV 0.198 in-sample, **0.246 leave-one-group-out** (r1 thresholds scored on the same consolidated output: 0.230 in-sample; r1's own per-frame numbers 0.193 / 0.266 are not comparable). Per group: DOB30 0.05, DOB100 0.07, methylone30 0.17, Veh 0.21, MDMA30 0.29, methylone100 0.30, Fentanyl30 0.30 (held-out 0.35). Coverage now 0.67-0.88.
On the 45 calibration videos after consolidation: median 432 segments per video; 22.9% Undetermined, 35.5% Freezing, 28.9% Erratic, 12.7% Controlled; all pass `frames_to_segments` (Dead monotonicity).

**Dead investigation (user asked to fix it) — result: not separable from LORR/long stillness with this tracker; left as is, Dead is effectively manual.**
- Why the rule never fires: a motionless fish is absorbed by the background subtractor, so the terminal still stretch is *undetected*, not detected-and-slow, and the rule needs detected frames to the end.
- Measured on the 45 cached tracks (seconds since last movement > 20.6 px/s; detection rate in the last 120 s): long still tails exist in 7 videos - Fentanyl 30 #340 (359 s, det 0.00), #333 (271 s), Fentanyl 100 #124 (341 s, det 0.00), DOB 30 #62 (568 s; video is 1773 s long), DOB 100 #74 (90 s), methylone 30 #84/#86 (76/102 s). Everyone else moves within seconds of the end.
- Frames from the tails (5 min, 2 min, 5 s before the end): F_340 and F_124 are visually identical across 300 s; F_333 changes slightly; F_120 (alive) moves. So "nothing moves" exists as a signal, but
  Fentanyl 30 (#340 is in the reference cohort 331-341) has a reference Dead share of 0% and ~44% pink (LORR): a fish lying motionless for 6 min is labeled alive/LORR there. A gap-tolerant "no movement to the end" rule would label F_340 Dead against the reference.
  Dead and LORR are the same observation (a still fish absent from the foreground) with this camera and tracker. The calibration set has no identifiable Dead (reference ~0% in all 7 groups; the Fentanyl 100 uM cohort with 41% Dead has no local videos), so no threshold can be validated.
- Decision: keep the conservative rule (fires only for detected-and-slow-to-the-end; monotonic, segments-validated) and treat Dead and Listing/LORR together as a manual decision in review. Proposed review aid for Phases 9-11 (not built): flag a terminal "no action" suffix >= ~120 s per video ("Dead or LORR?") so the reviewer can confirm; the per-video suffix length is computed by scratchpad/dead_explore.py.
- **Review flag built (data side of the review-aid).** `models.ReviewFlag(kind, start_s, end_s, message)`; `ManifestRecord.review_flags` (tuple, default empty, serialised in `manifest.json`, older manifests without the key still load);
  `review_flags.terminal_no_action_flag(tracks, freeze_speed_floor_px_per_s=..., min_no_action_s=120)` flags "no movement (detected frame with a real smoothed speed above the floor) from t to the end" for >= 120 s; undetected frames are neither movement nor stillness evidence.
  13 new tests (226 pass). On the 51 cached calibration videos it flags exactly four: Fentanyl 30 #340 (from 842 s), #333 (930 s), Fentanyl 100 #124 (863 s) and DOB 30 #62 (1204 s of a 1773 s recording - the footage past the 1200 s session, so that one is post-session, not a Dead candidate).
  Not built yet: computing the flag inside `export_video()` (T060) and showing it in the review UI (T070/T073) - both are Phase 9-11 items that do not exist yet; the accept endpoint should surface the flag next to the FR-015a Dead confirmation. Open question for T060: clamp the flag to the 1200 s session for recordings longer than that.

### Phase 7 — Labeling (`labeling.py`, `segments.py`) — FR-007

harness-os stage: *not yet run*

| ID | Description | Status | Notes |
|---|---|---|---|
| T047 [P] | `test_freezing_rule_sustained_low_velocity` | DONE | Plus resets-across-gap and doesn't-fire-early regression tests |
| T048 [P] | `test_surface_breach_rule_depth_spike` | DONE | Plus pure-per-frame-no-duration-gate test |
| T049 [P] | `test_listing_rule_sustained_orientation` | DONE | Plus orientation-`None`-abstains-not-Undetermined and resets-across-gap tests |
| T050 [P] | `test_controlled_vs_erratic_gmm_separates_synthetic_clusters` | DONE | Plus low-confidence→Undetermined-with-confidence-recorded and too-few-eligible-frames-skips-gracefully tests |
| T051 [P] | `test_undetected_frames_are_undetermined_not_guessed` | DONE | Plus detected=False-overrides-every-other-rule-even-with-extreme-features test |
| T052 | `classify_frame()`/`classify_video()` (as `classify_video()`; no separate `classify_frame()` — see narrative) | DONE | Implemented against config-driven placeholder thresholds (`DEFAULT_*` constants in `labeling.py`, values pending T046) |
| T053 [P] | `test_run_length_encoding_round_trips_to_frames` | DONE | `tests/test_segments.py::test_segments_round_trip_preserves_state_and_source` |
| T054 | `frames_to_segments()`/`segments_to_frames()` | DONE | `Dead` monotonicity (FR-007a) enforced as `DeadMonotonicityError`, explicitly tested for both an ordinary non-Dead-after-Dead violation and the Undetermined-after-Dead edge case |

**Design decisions (locked in via `advisor()` before implementation, see full text in `labeling.py`'s
module docstring — summarized here):**

- **Rule precedence, fixed and documented** (PRD §9.5.6 lists the six rule families but never orders
  them): `detected=False` → Undetermined (absolute) > Dead (whole-video terminal-suffix pass) >
  Surface Breach (per-frame spike) > Listing/LORR (sustained) > Freezing/Drift (sustained) > GMM
  Controlled/Erratic > Undetermined (GMM posterior below `gmm_confidence_margin`).
- **No gap-bridging.** Every "sustained for ≥ N seconds" rule measures elapsed `t_sec` over a run of
  contiguous *detected* frames only; an undetected frame or a `frame_idx` gap resets the bout clock to
  zero rather than bridging across it. Deliberate, not an oversight — fabricating continuity across a
  tracking dropout would silently convert a Phase 4 tracking deficiency into a biological claim nobody
  observed. Explicitly RED/GREEN-tested for Freezing/Drift and Listing/LORR (`..._resets_across_an_undetected_gap_no_bridging`)
  and for Dead (`test_dead_does_not_fire_if_the_video_tail_is_undetected`).
- **Dead (FR-007a) is a whole-video terminal-suffix pass**, not a sliding window: `_compute_dead_suffix`
  walks backward from the *last* frame while the near-zero-velocity condition holds contiguously: if
  that terminal run's `t_sec` span reaches `min_dead_bout_s`, every frame from its onset to the true end
  is Dead — monotonic by construction. If the video's tail is undetected (common on real footage — see
  below) or recovers before the end, no Dead is ever emitted, only Undetermined. `segments.py` then
  re-validates this invariant independently (`DeadMonotonicityError`, carrying both the offending frame's
  index/state and the Dead-onset frame's index) as defense in depth against any `StateFrame` sequence
  assembled some other way (e.g. post-manual-review edits) — "reject or flag," never silently repaired.
- **The sentinel-vs-real problem, again.** `FeatureFrame.velocity`/`.angular_velocity` are `0.0` both when
  genuinely near-zero *and* when not computable from history (Phase 5's established convention). Since
  `FeatureFrame` doesn't expose a public validity mask, `labeling.py` re-derives the same "two
  consecutive detected, `frame_idx`-adjacent frames" validity rule locally (`_real_below_floor`,
  `_angular_velocity_valid_mask`) rather than trusting a raw `0.0` — the same bug class Phase 5's review
  caught and fixed in `features.py` itself. `orientation_deg`, by contrast, is a direct per-frame
  measurement with no such history dependency, so the Listing/LORR rule just checks `is not None`
  directly — a genuine asymmetry between the two families of rule, documented rather than papered over.
- **`orientation_deg is None` abstains, doesn't force Undetermined.** Only Listing/LORR, which needs it,
  falls through to the next rule in precedence order; velocity-based rules and the GMM remain evaluable
  on that frame.
- **Threshold reuse/omission is schema-driven, not accidental.** `config/default_thresholds.yaml`'s
  `labeling:` block has no separate Dead-velocity-floor key, so Dead reuses `velocity_freeze_floor_px_per_s`
  (PRD's own wording — "sustained near-zero velocity" — is the same near-zero concept as Freezing/Drift,
  just gated by a much longer bout). It also has no Surface-Breach `min_bout_s` key, read as a deliberate
  signal that Surface Breach is a pure per-frame threshold crossing ("a short, spike-like event"),
  explicitly tested (`test_surface_breach_is_a_pure_per_frame_threshold_no_sustained_duration_required`).
- **GMM determinism (NFR-002).** Seeded via `gmm_random_state` (default `0`, mirroring
  `calibration_profile.random_state` in the YAML), explicitly tested (`test_classify_video_is_deterministic`).
  Falls back to all-Undetermined with no confidence value (nothing was fit) when fewer than
  `2 * n_components` frames remain eligible after the hard rules — never raises on a short/quiet video.
- **Placeholder constants preserve intended structure, not just plausible-looking numbers.**
  `DEFAULT_MIN_DEAD_BOUT_S = 60.0` is deliberately an order of magnitude above
  `DEFAULT_MIN_FREEZE_BOUT_S = DEFAULT_MIN_LISTING_BOUT_S = 2.0`, preserving PRD's intended
  "Dead requires much longer sustained stillness than Freezing" relationship even before Phase 6 supplies
  real numbers.
- **`segments.py`'s `end_s`/`duration_s` convention** (not specified by PRD's terse `segments.csv`
  schema): a segment's `end_s` is the next segment's `start_s` (segments tile the video with no gaps);
  the final segment extrapolates using the *median* inter-frame spacing across the whole sequence
  (median, not the last observed gap, so one anomalous final interval can't skew it).
- **`segments_to_frames` is the practical "round trip"**, not a literal segments→frames without any
  input: `StateSegment` (matching `segments.csv`) carries no `frame_idx` grid, so reconstruction takes
  the original `(frame_idx, t_sec)` pairs back in and looks up which segment's `[start_s, end_s)` window
  each falls in — `confidence` always comes back `None` since segments.csv has no confidence column, a
  genuine, documented loss crossing that boundary, never fabricated back. Assumes non-decreasing `t_sec`
  input order (how it will actually be called — re-reading `frames.parquet` in frame order).

**Real-data findings, pre-fix** (same three videos used in Phase 4/5: `Veh/F_0057`, `Veh/f_0001`,
`Fentanyl/F_0106`; full pipeline tracking → features → labeling → segments, placeholder thresholds).
**Superseded by the code review's CRITICAL finding below — these numbers include fabricated
Controlled/Erratic labels on sentinel frames and should not be treated as trustworthy; kept here only
for before/after comparison against the corrected run further down:**

| Video | Detected | Controlled Swim | Erratic Movement | Listing/LORR | Undetermined | Freezing/Drift | Dead |
|---|---|---|---|---|---|---|---|
| Veh/F_0057 | 73.1% | 32.0% | 33.4% | 1.6% | 33.0% | 0% | 0% |
| Veh/f_0001 | 93.9% | 18.5% | 39.1% | 27.1% | 15.3% | 0% | 0% |
| Fentanyl/F_0106 | 84.5% | 43.7% | 34.1% | 4.5% | 17.7% | 0% | 0% |

**Freezing/Drift and Dead never fired on any of the three real videos.** This is the predicted,
accepted consequence flagged before implementation (advisor: "report back... how often each rule
actually fires on real footage"), not a bug: at 73–94% detection with frequent short gaps (Phase 4/5's
own measured tracking quality), a contiguous detected run long enough to accumulate 2s (Freezing) or
60s (Dead) of continuous below-floor velocity essentially never occurs under the no-bridging rule.
This is exactly the tradeoff flagged before implementation — bridging the gaps would make these rules
fire, but only by fabricating continuity the tracker never actually observed; the absorber for this
gap is FR-015's reviewer-correction workflow, by design, not a reason to add bridging.
Listing/LORR's firing rate varies widely across subjects (1.6%–27.1%) with the placeholder 45°
deviation threshold — plausibly too permissive, a concrete data point for Phase 6's calibration search
rather than a reason to hand-adjust the placeholder now. The GMM produces two non-degenerate,
differently-sized clusters on all three real videos (unlike the near-duplicate-point
`ConvergenceWarning` seen on some synthetic test fixtures with tightly clustered placeholder feature
values) — functions correctly, real balance is Phase 6's to judge, not this phase's. Dead monotonicity
held on all three (no `DeadMonotonicityError` raised) — expected, since Phase 7's own construction
guarantees it, but confirms `segments.py`'s independent validator doesn't false-positive on real
tracker output either.

**Code review (`code-reviewer` agent) — findings and fixes.** Verdict: BLOCK (1 CRITICAL, 2 HIGH,
2 MEDIUM, 2 LOW), verified against the installed `sklearn==1.9.1` and direct repros, not just static
reading. All findings addressed:

- **[CRITICAL, fixed]** Sentinel `0.0` feature rows (frame 0 of every video; every tracking-gap
  reacquisition frame) were fed into the GMM with no validity gating — `eligible_idx` only checked
  "not already claimed by a rule," never whether `velocity`/`angular_velocity`/`meander` were real
  measurements. Repro against real `derive_features()` output: frame 0 and the first frame after a
  synthetic gap both landed `ERRATIC_MOVEMENT, confidence=1.0` — fabricated labels presented with
  maximum confidence, hitting a routine fraction of frames given Phase 4/5's own measured ~76%/~52%
  detection. **Fix:** `features.py` gained a public `compute_feature_validity()` (returning a
  `FeatureValidity` mask for velocity/angular_velocity/meander) and a public
  `consecutive_detected_run_length()`; `labeling.py` now computes `gmm_input_valid` from these and
  routes any GMM-eligible frame lacking all three real inputs straight to Undetermined with
  `confidence=None`, never fit or predicted. Regression test:
  `test_real_derive_features_output_never_confidently_labels_a_sentinel_frame` (piped real
  `derive_features()` output through `classify_video`, asserting frame 0 and the post-gap frame are
  never Controlled/Erratic and always carry `confidence=None`).
- **[HIGH, fixed]** `labeling.py`'s own re-derivation of velocity/angular-velocity validity (before this
  fix) checked frame_idx adjacency but not `dt > 0`, reintroducing the exact `dt<=0`-sentinel-treated-
  as-real bug Phase 5's review already caught once in `features.py`. **Fix:** eliminated the local
  re-derivation entirely — `labeling.py` now consumes `features.compute_feature_validity()` directly
  (which already has the `dt > 0` guard from Phase 5), a single source of truth instead of a second,
  drifting copy. Regression tests added directly to `test_features.py`
  (`test_compute_feature_validity_flags_dt_sentinel_velocity_as_invalid` and the angular-velocity
  equivalent) lock in the guard at its source.
- **[HIGH, fixed]** `_rolling_angular_velocity_variance`'s window wasn't clamped to the current
  contiguous detected run (unlike `features.py`'s `_compute_meanders`, which was fixed for exactly this
  in Phase 5's own review), so pre-gap turning samples could pollute a post-gap frame's variance —
  repro showed a variance of 332448.98 immediately after a gap where the true post-gap value was ~0.
  **Fix:** `_rolling_angular_velocity_variance` now takes `run_length` and clamps its window the same
  way `_compute_meanders` does (`lo = max(0, i - window_frames + 1, i - run_length[i] + 1)`), and
  returns a `variance_valid` companion list feeding directly into the CRITICAL fix's `gmm_input_valid`.
  Regression test: `test_rolling_angular_velocity_variance_does_not_blend_across_a_detection_gap`
  (mirrors the review's exact repro).
- **[MEDIUM, fixed]** `segments_to_frames` had no validation of its documented "non-decreasing `t_sec`"
  precondition, so a violation would silently misattribute frames rather than error. **Fix:** added an
  explicit `ValueError` check (same "reject or flag" philosophy as `DeadMonotonicityError`). The
  forward-only pointer's `>=` comparator was investigated as a possible second fix (switching to strict
  `>`) but reverted after it broke the *normal* (non-tied) boundary case — `end_s` is the next segment's
  `start_s` by construction, so the first frame of every segment legitimately has `t_sec == previous
  segment's end_s` and must advance there; `>=` already handles that correctly. An exact-`t_sec`-tie of
  two *different* frames straddling a real state transition remains genuinely unresolvable from
  segments.csv's schema alone (no `frame_idx` grid is stored) — documented as an explicit, accepted
  limitation rather than something the comparator can fix, and not reachable from the current sole
  producer's strictly-increasing timestamps.
- **[MEDIUM, fixed]** The GMM's minimum-eligible-frames guard used a module-level constant frozen from
  the *default* `n_components`, so a caller passing a different `gmm_n_components` got a floor that
  didn't scale with it. **Fix:** the floor (`max(2 * n_components, n_components)`) is now computed
  inside `_classify_controlled_vs_erratic` from the actual parameter. Regression test:
  `test_gmm_skipped_when_fittable_frames_are_below_n_components_scaled_floor`.
- **[LOW, fixed]** `frames_to_segments` merged consecutive same-state/same-source frames without
  checking `frame_idx` adjacency — not triggerable by the current sole producer (`track_video` emits no
  gaps), but inconsistent with `labeling.py`'s own no-bridging philosophy for any `StateFrame` sequence
  assembled another way. **Fix:** added a `frame_idx` adjacency check as a third boundary condition.
  Regression test: `test_frames_to_segments_splits_on_frame_idx_gap_even_if_state_and_source_match`.
- **[LOW, fixed]** Test coverage gaps: the cluster-separation test only checked "some different label,"
  not the specific Controlled↔low-variance / Erratic↔high-variance mapping; the too-few-frames test
  accepted any state as passing instead of the specific documented Undetermined-with-`None` fallback;
  and no test exercised `classify_video` against real `derive_features()` output (exactly why the
  CRITICAL bug had no failing test). All three tightened/added (see fixes above).

Full suite after all fixes: 142 passed (was 131 before this pass — 11 new/tightened tests across
`test_features.py`, `test_labeling.py`, `test_segments.py`).

**Real-data findings, corrected** (same three videos, same pipeline, after the CRITICAL/HIGH fixes):

| Video | Detected | Controlled Swim | Erratic Movement | Listing/LORR | Undetermined | Freezing/Drift | Dead |
|---|---|---|---|---|---|---|---|
| Veh/F_0057 | 73.1% | 30.0% | 30.2% | 1.6% | 38.2% | 0% | 0% |
| Veh/f_0001 | 93.9% | 18.0% | 39.1% | 27.1% | 15.8% | 0% | 0% |
| Fentanyl/F_0106 | 84.5% | 35.9% | 30.9% | 4.5% | 28.7% | 0% | 0% |

Undetermined's share rose in all three videos relative to the pre-fix table — most sharply on
`Fentanyl/F_0106` (17.7% → 28.7%, +11.0pp) and `Veh/F_0057` (33.0% → 38.2%, +5.2pp), both markedly
lower-detection videos (84.5%/73.1%) with correspondingly more reacquisition frames; smallest on
`Veh/f_0001` (15.3% → 15.8%, +0.5pp), the highest-detection video (93.9%) with the fewest gaps to
reacquire from. Total segment counts dropped in every video too (e.g. `Veh/F_0057`: 8902 → 7038) -
previously-fabricated confident flip-flops between Controlled/Erratic on sentinel frames were creating
spurious extra segments; folding those correctly into Undetermined reduced segment churn as a direct,
visible side effect, not just a statistics shift. Listing/LORR and Controlled/Erratic's relative split
are otherwise close to the pre-fix numbers (as expected - the fix only affects which frames the GMM is
allowed to see, not the rule-based states upstream of it). Freezing/Drift and Dead still never fire on
any of the three videos, unaffected by this fix (that finding was never about the GMM path) - the
no-gap-bridging analysis above stands unchanged.

## Phase 6 prep — advisor consultation before writing `calibrate.py`

Before starting Phase 6, consulted `advisor()` with the Phase 7 code (post-review-fixes) and the
real-data findings above. It surfaced three blockers that needed resolving before a calibration search
could produce anything meaningful — two turned out to be real, confirmed defects, not calibration
questions. All three addressed below.

**1. [CONFIRMED BUG, fixed] The Listing/LORR orientation-deviation formula was inverted.**
`_listing_condition` computed `deviation = min(orientation_deg, 180.0 - orientation_deg)`, built on the
Phase 5 assumption "`cv2.fitEllipse`'s angle: `0°` == horizontal." Verified directly against the actual
OpenCV behavior (not assumed): a synthetic wide/short (horizontal) blob fits to `orientation_deg=90.0`;
a tall/narrow (vertical) blob fits to `0.0`; a blob rotated 45° off horizontal fits to `135.0`. OpenCV's
convention is `90° == horizontal`, `0°`/`~180° == vertical` — the **opposite** of what Phase 5 assumed
and what Phase 7's formula encoded. The bug: a normally-horizontal-swimming fish (`orientation_deg~90`)
scored *maximal* deviation under the old formula, while an actually-vertical/rolled fish
(`orientation_deg~0`) scored *zero* deviation — backwards. This is exactly what produced the 27.1%
Listing/LORR rate on `Veh/f_0001` (a vehicle-control, undrugged fish) flagged as suspicious pre-Phase-6:
no reference panel except Fentanyl shows meaningful pink, so a control fish reading 27% Listing was a
red flag, not noise. **Fix:** `_listing_condition` now computes `deviation = abs(orientation_deg - 90.0)`
(correct given `fitEllipse`'s `[0, 180)` output range). Real-data re-run after the fix: Listing/LORR
dropped to 0.4% (`Veh/F_0057`), 0.1% (`Veh/f_0001`), 0% (`Fentanyl/F_0106`, none in the top states at
all) — consistent with the reference figures' near-zero Listing on non-Fentanyl groups. (Fentanyl's own
0% here is itself a further data point for Phase 6 — the placeholder 45° threshold is likely too strict
now that the formula is correct, not evidence the rule is broken; Phase 6's search range must span low
enough to let it fire on the actual Fentanyl LORR pattern.)

**2. [ROOT CAUSE CORRECTED] Freezing/Drift's 0% firing rate is NOT primarily a no-bridging/run-scarcity
issue, as this doc previously claimed — retracted, same pattern as the Phase 4 97.9%-detection
retraction.** Measured directly on `Veh/f_0001` (93.9% detected): 226 distinct contiguous detected
runs, 106 of them ≥60 frames (the 2s-at-30fps bout minimum), longest run 2438 frames — long enough
contiguous stretches are abundant, not scarce. Real (non-sentinel) velocity distribution: p5=3.81,
p25=16.59, p50=46.50 px/s; `DEFAULT_VELOCITY_FREEZE_FLOOR_PX_PER_S=10.0`; 15.4% of individual real
velocity samples are ≤ floor. So individual low-velocity frames are common, and long contiguous runs
are common, but a run of 60 *consecutive* frames *all* ≤ floor essentially never happens — velocity
oscillates in and out of the floor within otherwise-long detected runs, consistent with the advisor's
hypothesis that the floor sits inside the tracker's own positional jitter band rather than below it (a
genuinely still fish's *measured* velocity is noisy around a value near/above the floor, not settled
comfortably under it). No-bridging is still real and still matters (confirmed separately by regression
tests), but it is not the dominant explanation for 0% on this video — the floor/bout-duration
relationship to tracking noise is. This reframes Phase 6's job for `velocity_freeze_floor`: search
relative to the *measured* real-velocity distribution (e.g. anchored near p25 rather than picked
independently), not treated as a free parameter with no data-grounded starting range.

**3. [ARCHITECTURAL FIX] The Controlled-Swim-vs-Erratic-Movement GMM must be fit once on pooled data and
frozen, never per video — this was the most consequential finding.** `classify_video` previously fit a
fresh 2-component GMM on each video's own eligible frames. That decision boundary is *relative to that
one video*: every fish gets split into a "calmer" and a "less calm" half of *itself*, regardless of
which drug it received. Confirmed empirically: three videos from three different exposure groups showed
near-identical Controlled:Erratic splits (~30/30, ~18/39 pre-fix, ~36/31) that track each video's own
internal spread, not the actual pharmacology — and critically, **no parameter in Phase 6's search space
can move that ratio**, since the GMM always re-splits whatever frames reach it roughly in half by
construction. Since the reference groups differ most on exactly this ratio (MDMA 30µM mostly gray/
Controlled, DOB 30/100µM heavily red/Erratic), a per-video-fit GMM would make the aggregate-proportion
calibration search meaningless regardless of how long it ran.
**Fix:** `labeling.py` gained `fit_controlled_erratic_gmm(rows, ...)` (fits once, returns a frozen
`sklearn.mixture.GaussianMixture`) and `gmm_input_rows(tracks, features, ...)` (gathers one video's
validity-gated `[velocity, angular_velocity_variance, meander]` rows for pooling across a calibration
group). `classify_video` gained an optional `gmm=` parameter: predict-only against a supplied frozen
model when given; falls back to the old per-video fit only when `gmm` is omitted, now explicitly
documented as a standalone/exploratory convenience, never used for calibration or cross-video
comparison. Regression test `test_classify_video_with_frozen_gmm_never_refits_and_is_shared_across_videos`
fits the pooled model on one synthetic video and classifies a second, independently-built video against
it, confirming consistent labels come from a shared boundary, not two independent relative splits.
`gmm_input_rows` deliberately pools on feature validity alone, not also filtering by which frames the
hard rules would claim first — filtering by rule-eligibility would make the GMM's own fit depend on the
rule thresholds Phase 6 is simultaneously searching over (a circular dependency judged not worth solving
for PRD §9.6's "one-time, small-scale" calibration); documented as a deliberate simplification in
`labeling.py`'s module docstring, revisit only if it visibly skews results.

Full suite after all three fixes: 147 passed (was 142 — 5 new tests for the pooled-GMM API, plus the
existing Listing tests' fixtures corrected to use real OpenCV orientation semantics, not new tests).

### Phase 8 — Rendering (`rendering.py`) — FR-009

harness-os stage: *not yet run*

Order note: Phase 8 was done before Phase 9 (export writes the strip PNG, so export depends on rendering), i.e. 8 -> 9 -> 10 -> 11.

| ID | Description | Status | Notes |
|---|---|---|---|
| T055 [P] | `test_strip_uses_exact_reference_palette` | DONE | Pixel-samples each segment centre against `palette.PALETTE_RGB` (exact RGB). |
| T056 [P] | `test_time_axis_matches_actual_video_duration` | DONE | `seconds_to_x` is linear over the real duration (a 1773 s video is not squeezed into 1200 s); boundary and last-column tests. |
| T057 | `render_strip()` | DONE | PIL (no antialiasing) so a sampled pixel is exactly the legend colour. Ticks at a step chosen for <= 8 labels, subject label left, "seconds" caption. **Design decision:** Undetermined renders magenta `#FF00FF` (PRD §5.2's own placeholder), distinct from all six legend colours incl. white (Dead) and pink (LORR); configurable as `rendering.undetermined_color_hex`. |

**New stage found while rendering real data: temporal consolidation (`consolidate.py`).** Per-frame labels flicker (F_332: ~5,000 detection on/off transitions, ~10,000 segments, median segment 2 frames), which is unreviewable (FR-015 forces every Undetermined segment to be resolved) and finer than the ~1 s reference figures.
Each 1 s bin now takes the majority of its *determined* frames (Undetermined abstains), is Undetermined only if < 20% of its frames are determined, keeps a Surface Breach spike unless Dead holds >= half the determined frames, ties break Dead > Listing > Freezing > Erratic > Controlled; Dead stays a monotonic suffix. Never crosses a bin boundary, so it does not fabricate long bouts across dropouts.
On the 45 calibration videos: median segments 5,976 -> 366 per video (max 559); Undetermined 36.7% -> 22.3%. It changed the agreement (0.193 -> 0.230 with thresholds tuned on per-frame output), so the calibration objective now scores the consolidated output (`evaluate_params(..., consolidate=True)`).

### Phase 9 — Export & Review State — FR-010, FR-014, FR-016, FR-017

harness-os stage: *not yet run*

| ID | Description | Status | Notes |
|---|---|---|---|
| T058 [P] | `test_writes_three_consistent_artifacts` | DONE | segments.csv is the exact RLE of the frames.parquet on disk (float32 rounding of `t_sec` included); PRD dtypes (int32/float32/category[7]). |
| T059 [P] | `test_manifest_records_pipeline_and_calibration_version` | DONE | Also carries `review_flags` (Dead-or-LORR candidate). |
| T060 | `export_video()` | DONE | Never overwrites EDITED/ACCEPTED output (`ReviewedWorkExists`) unless `overwrite=True` (FR-016). |
| T061 [P] | `test_status_transitions_valid_only` | DONE | Table-driven; ACCEPTED is terminal; REJECTED -> PROCESSED_AUTO only. |
| T062 [P] | `test_reject_clears_edits_and_requeues` | DONE | Removes frames/segments/strip, edit_count 0, status REJECTED, `should_process` True. |
| T063 [P] | `test_accept_blocks_on_undetermined_without_force` | DONE | FR-015 hard block, `force` ignored (also with Dead present). |
| T064 | `review_store.py` | DONE | FR-015/FR-015a genuinely asymmetric: Undetermined -> `AcceptBlocked`; Dead -> `ConfirmationRequired` unless `force=True`. `save_edit` (frame-accurate, frame_idx/fps float64 boundaries, manual source, one edit per call, Dead monotonicity re-validated), `accept` (gold copy, provenance.json, accepted_index.parquet upsert), `reject`. |
| T065 [P] | `test_batch_rerun_skips_accepted_videos` | DONE | `should_process(status, force)`. |

Code review (two rounds): no CRITICAL. HIGH fixed: `accept` used to flip the source manifest to ACCEPTED before the gold copy/index were written (an index failure stranded the video); now gold artifacts and the atomic-replaced index are written first and the source manifest last, and gold `segments.csv` is re-derived from the frames rather than copied. MEDIUM/LOW fixed: `save_edit` builds everything in a temp dir and `os.replace`s with the manifest last; edit boundaries use float64 `frame_idx/fps` (float32 `t_sec` is wrong past ~2048 s); empty/NaN/zero-frame edits rejected; gold directory name validated; a lone Dead frame no longer turns a Surface bin Dead; `export_video` overwrite guard.
Not done: the accepted index only carries Trial fields when `accept(..., trial=...)` is given (Phase 12's orchestrator has the catalog); nothing calls `export_video`/`should_process` yet (Phase 12).

### Phase 10 — Review Web App Backend — FR-011..016

harness-os stage: *not yet run*

| ID | Description | Status | Notes |
|---|---|---|---|
| T066 [P] | `webapp/schemas.py` | DONE | Pydantic models; `EditsRequest` takes a list of `{start_s, end_s, new_state}` (a superset of the PRD's single relabel: one Save = one edit count), non-finite numbers and unknown states rejected (422). |
| T067 | `GET /videos`, `GET /videos/{id}` | DONE | List (sorted, `?status=` filter, empty for a missing dir); detail = manifest incl. review flags + segments + seconds-per-state summary + `video_url`/`strip_url` (None once rejected). Ids are validated before any filesystem access. |
| T068 | `GET /videos/{id}/video` range support | DONE | Starlette `FileResponse`; verified 206 + `Content-Range` in tests and over real HTTP against a real mp4 (100,000-byte range). Extra `GET /videos/{id}/strip` serves the PNG for the UI. |
| T069 | `POST /videos/{id}/edits` | DONE | -> `review_store.save_edit`; invalid edits (empty/outside range, Dead monotonicity) 422 and change nothing; edit after accept 409. |
| T070 | `POST /videos/{id}/accept` | DONE | Undetermined -> 409 `undetermined_present` always (`force` ignored, also over real HTTP); Dead without `force` -> 409 `dead_confirmation_required`; `force=true`+Dead -> accepted. |
| T071 | `POST /videos/{id}/reject` | DONE | Clears edits, status REJECTED; rejecting an ACCEPTED video is 409. |
| T072 [P] | `test_webapp_api.py` | DONE | (now more after hardening; suite 343 passing) 26 initial tests (TestClient) incl. range requests, traversal ids, the full accept matrix, concurrent edits (serialised by a write lock). |

Smoke test on real data (scratchpad/smoke_export.py, serve_smoke.py): F_332/F_333/F_340 exported from cached tracks with the frozen r2 profile + consolidation, served with uvicorn on 127.0.0.1 and driven with curl. F_340 shows its terminal-no-action flag and 1058 s of Undetermined (the tracker lost the fish), which is why Accept is blocked until a reviewer resolves it.
Code-review hardening (all fixed, each with a RED test first):
- DNS-rebinding/CSRF: `create_app(..., allowed_hosts=...)` adds `TrustedHostMiddleware` (foreign Host -> 400) and refuses state-changing requests with a foreign `Origin` (403 `foreign_origin`).
- `manifest.json` writes are now atomic (temp file + `os.replace`); a failed replace leaves the old manifest and no `.tmp`.
- One corrupt manifest no longer breaks the list (skipped); detail/stream return 500 `corrupt_manifest` with no absolute path in the body.
- `InvalidEdit` (ValueError subclass) separates malformed edits (422) from other errors; missing artifacts -> 409 `missing_artifacts`; other failures stay generic 500.
- Edits capped at 1000 per request; reviewer names are stripped, non-blank, control-character-free.
- `/video` serves only video suffixes and, when `video_dir` is set, only files resolving under it (traversal/absolute escapes -> 404); ids like `a..b` are allowed, `.`/`..` rejected, and the list applies the same id check.
`create_app(processed_dir, accepted_dir, *, video_dir=None, clock=...)`; the CLI/`prepds review` entry point that starts it belongs to Phase 12. Local single-user tool: bind to 127.0.0.1.

### Phase 11 — Review Web App Frontend — FR-011, FR-012

harness-os stage: *not yet run*

Vanilla HTML/CSS/JS (no build step) in `src/prepds/webapp/static/`, served by the FastAPI app at `/` and `/static/*`; packaged via `package-data`. All server text is inserted with `textContent` (no `innerHTML`).

| ID | Description | Status | Notes |
|---|---|---|---|
| T073 | `static/index.html` | DONE | Video list + status filter, reviewer field, `<video>`, timeline, state picker, staged-edit list, per-state seconds legend, Accept/Save/Reject/Discard. |
| T074 | `app.js` timeupdate → playhead | DONE | Playhead follows `timeupdate`/`seeked`; a click on the timeline seeks; arrow keys seek. |
| T075 | `app.js` drag-select → state picker | DONE | Pointer drag selects a range (4 px edge snap so the last frame is reachable), pick a state, "Stage relabel"; staged edits are previewed on the timeline and sent in one Save. `Undetermined` is not assignable. `[`/`]` keys set selection bounds at the playhead. |
| T076 | `app.js` wire Accept/Save/Reject | DONE | Accept: `undetermined_present` shown as a hard message (no override offered); `dead_confirmation_required` opens a confirm and retries with `force`. Save/Reject/Accept refuse a blank reviewer. |
| T077 | `app.css` | DONE | Paper/ink "lab notebook" palette, light+dark, focus rings, reduced-motion, no horizontal overflow at 375 px. |
| T078 | Manual UX pass, 10 real videos | PARTIAL | Playwright-driven against 3 real exported videos (F_0332/F_0333/F_0340 with the real mp4s), not 10: list, open, flags, playback/playhead, drag-select, stage, save, Undetermined-blocked accept, Dead-confirm accept -> gold copy + `accepted_index.parquet`, reject, mobile width. |

Bugs found while driving the real UI (each fixed):
- Boundaries copied from `segments.csv` (float32) missed the segment's first frame by up to ~2e-6 s because the edit tolerance was 1e-9 s, leaving one-frame Undetermined slivers that blocked Accept. RED test over every frame, then `_BOUNDARY_EPS_S` raised to 5e-4 s (far below half a frame).
- Timeline drag could not reach the final pixel, leaving a tail unlabeled (and then Dead monotonicity correctly rejected the save): added edge snap.
- Topbar overflowed at 375 px.
- `/videos/..` is now normalised by clients to `/` (serves the page); the id test no longer uses `..` and uses `.hidden` instead.

Code-review fixes (frontend): stale responses are dropped (open/list sequence tokens); Save/Accept/Reject go through one `mutate()` that allows a single in-flight request, is bound to the video it started on, and never reports a committed change as failed when only the list refresh fails; the detail is re-fetched after a 409/422; `aria-valuetext` names the state under the playhead. Verified in Playwright: double-click Save posts once (edit_count 1). Not done: announcing selection/staged edits to screen readers, focusable segments.

Known UX limits: a video with many short Undetermined slivers takes many drags to resolve (no "relabel all Undetermined as X" bulk action yet); not tested in Firefox/Safari; the frontend has no automated JS tests (verified by Playwright only).

### Phase 12 — Batch Orchestration & CLI

harness-os stage: *not yet run*

| ID | Description | Status | Notes |
|---|---|---|---|
| T079 [P] | `test_run_command_dry_run_lists_pending_videos` | DONE | Plus resume, `--limit`, missing-catalog, failure exit code, `--workers 0`, export-index, loopback-only review tests. |
| T080 | `cli.py run` subcommand | DONE | `pipeline.py`: probe -> track -> features -> classify -> consolidate -> review flag -> export, one video per worker (spawn context: a fork-based pool deadlocked once OpenCV/threads had run in the parent, found when the full suite hung). Skips PROCESSED_AUTO/EDITED/ACCEPTED (resumable; only NOT_PROCESSED/REJECTED are (re)generated) unless `--force` (which also overwrites reviewed work). A failing video is reported, not raised; duplicate `sex_subject` ids are failed, not overwritten. Writes `run_report.json` (counts, workers, wall time, per-video outcome). Exit 1 if any video failed. |
| T081 | `cli.py export-index` | DONE | `review_store.rebuild_index`: rebuilds `accepted_index.parquet` from ACCEPTED manifests (skips unreadable/non-accepted dirs, drops stale rows), fills strain/age/date/exposure from the catalog. |
| T082 | `--workers N` override + docs | PARTIAL | `--workers`, default `max(1, cpu_count-2)` (config default changed from 1; PDS_WORKERS still overrides), throughput note in `run --help`. README section belongs to Phase 14. Also added `prepds review [--host --port]` (loopback only) to start the web app. |
| T083 | Time full 353-video run | DONE (328 videos, see Phase 13) | Measured only a 3-video real run through the CLI: 35.1 s wall with 3 workers (about 35 s per 20-min video). Linear extrapolation on this 24-core machine (22 workers): ~337 videos x 35 s / 22 = ~9 min, NOT yet measured; disk/decoder contention may change it. |

Real-data check: F_0332/F_0333/F_0340 processed through `prepds run --workers 3`; F_0333 and F_0340 carry the terminal-no-action flag, F_0332 does not; a second run reports nothing to do.
Code-review fixes (each with a RED test first): a natively crashing worker (segfault/OOM) no longer kills the batch, since unfinished videos are retried one per fresh pool so only the crasher is FAILED; a failing progress callback cannot abort the batch; a forced re-export deletes the old manifest first and writes the new one atomically (a mid-export failure leaves no manifest, never new frames under an old EDITED status); duplicate `sex_subject` ids are skipped with a warning instead of failing every run; `export-index` keeps previously indexed workbook fields when the catalog is missing and says so.
Known limits: no cross-process locking, so two simultaneous `prepds run` invocations, or a run alongside accepts in the review app, can collide and the accepted index update is an unlocked read-modify-write (`export-index` repairs it) - run one at a time; an integer `age` column with missing values round-trips through the catalog as float; a run killed mid-video leaves the first-time output without a manifest (regenerated next run). Not covered: catalog-to-run on the real workbook (Phase 13).

### Phase 13 — Full Integration Pass

harness-os stage: *not yet run*

| ID | Description | Status | Notes |
|---|---|---|---|
| T084 | Full pipeline run, all real videos | DONE (on the 337 synced videos) | `prepds catalog` then `prepds run` on the real workbook + `videos/Phase_1`/`Phase_2`: 352 trials = **328 processed + 24 unmatched trials** (all in `exceptions_report.json`); 0 failed; 9 local videos matched no trial (e.g. `F_0179A/B`, `F_0021(P1/P2)`, `F_310 (2)`); 20 duration mismatches (videos longer than 20 min); 0 corrupt. |
| T085 | Stratified review sample through UI | PARTIAL | `outputs/review_sample.csv` lists 29 candidates (median-Undetermined video per compound family (16), 3 flagged, the 2 highest-Undetermined, the 9 short videos). The machinery was exercised on 3 real processed videos (copied to a scratch dir, all Undetermined bulk-relabelled through the HTTP API, accepted). The actual human review of the sample is still to do. |
| T086 | Confirm `accepted_index.parquet` usable | DONE | Scratch check: accept x3 -> `rebuild_index` with the real catalog -> one row per video with strain/date/provenance/paths; the referenced `frames.parquet` reads back. Only tested on 3 scratch copies (the real `accepted/` untouched). |

Catalog fix found by running on real data: `match_videos` only looked one level under `PDS_VIDEO_DIR`, so the real `videos/Phase_N/<compound>/` layout matched 0 trials. It now finds compound folders at any depth (RED test first).

**T083 measured** (replaces the estimate): 328 videos, 108.8 h of video, **3001.6 s (50 min) wall with 22 workers** (~6.6 videos/min). My earlier ~9 min extrapolation was wrong by ~5x: the machine's load average sat around 37 on 24 cores, so the workers oversubscribe (likely BLAS/numpy threads on top of OpenCV, and video decode I/O); not yet investigated. Trying fewer workers or limiting BLAS threads may be faster.

Results (from `outputs/processed`, `scratchpad/phase13_stats.py`):
- Fish detected in a median 79% of frames (p10 49%).
- Undetermined fraction: median 14%, p75 26%, p90 47%, max 88%. **70 videos > 30%, 29 > 50%, 9 > 70%.** Worst groups are FD-2-* compounds outside the calibration groups (FD-2-67 0.2: 73%, FD-2-97: 57%, FD-2-66 0.2: 51%).
- Mean state share across all videos: Freezing/Drift 46%, Erratic 25%, Controlled 10%, Undetermined 19%, Listing 0%, Surface 0%. Dead: 0 videos (rule never fired); 13 videos carry the terminal-no-action review flag.
- Segments per video: median 408, p90 537, max 740.
Interpretation: 29/328 (9%) mostly-Undetermined videos is a review burden, not the "large share" that was the trigger for moving Phase 15 ahead of review. Listing/LORR, Surface Breach and Dead will only ever come from reviewers; the calibrated speed thresholds were fitted on 5 reference groups and Controlled looks low in FD-2-* compounds, so treat their auto labels as less trustworthy. These numbers are the Phase 15 baseline.

### Phase 14 — Documentation & Polish

harness-os stage: *not yet run*

| ID | Description | Status | Notes |
|---|---|---|---|
| T087 [P] | `README.md` | DONE | Quick start, commands (`catalog`/`run`/`review`/`export-index`), per-video outputs, review UI use, how labels are produced. Fixed stale entries: the old README listed a `calibrate` command and a `uvicorn prepds.webapp.server:app` launch that never existed (now `prepds review`). |
| T088 [P] | Document calibration methodology + limitation | DONE | README "Calibration" section: method, in-sample 0.198 / LOGO 0.246 vs placeholder 0.418, 7 of the reference groups calibratable, 5 of 17 families / 2 of up to 4 concentrations, weak groups, placeholders for Surface/Dead, Listing manual. |
| T089 [P] | Update root `fish-detection/README.md` | DONE | New "Preprocessing Dataset System" section (independent package, no shared runtime code with `fishbehavior`, non-binding). |
| T090 | Final clarification-marker pass | DONE | `grep` finds no `[NEEDS CLARIFICATION]` markers in the PRD (the two hits are the sentences saying none remain / describing this task). |

Gap found while writing the README: there was **no in-repo way to reproduce the calibration** (it ran from scratch scripts and no `prepds calibrate` command exists, despite the old README). Added `scripts/build_track_cache.py` and `scripts/calibrate_thresholds.py` (cohort selection, search, per-group and leave-one-group-out; they do not write a profile - freezing goes through `write_calibration_profile`). Only the first stage of the calibration script was re-run (placeholder TV 0.4187 vs 0.4184 recorded at the time; the small difference is unexplained, possibly a changed video set); the full search + LOGO was not re-run to confirm 0.198/0.246. The scripts are untested by the suite. A proper `prepds calibrate` command remains undone.

### Phase 15 — Better Tracker / Pose Model (IN PROGRESS, started 2026-09-23 in parallel with review)

harness-os stage: *not yet run*

Decisions (user, 2026-09-23): GPU available (RTX 4070 Ti, 12 GB, visible in WSL); build a small labeling page like the review app; Listing/LORR is in scope (all labels covered), fewer Undetermined frames is a second benefit; start now, in parallel with review.

Motivation and evidence: Undetermined share per video correlates 0.97 with the undetected-frame share; of undetected frames 27% are in gaps < 1 s, 39% in 1-10 s (most recoverable), 34% in gaps >= 10 s (still fish; 37 videos have a >= 60 s gap, 8 have >= 300 s). The contour body angle is not a Listing signal on this side-view footage, and higher fps would not help.

Design (from advisor review):
- **One annotation pass, one model:** box + keypoints. Keypoint schema chosen for Listing: **snout, dorsal-fin base, ventral/belly, tail base, tail tip**, each with visibility. A head-tail axis measures only pitch (the Phase 7 failure); roll/inversion needs a dorsal and a ventral point. Per-frame tag `listing: yes/no/unsure` plus `no fish visible`.
- **Hold-out split frozen before any frame is sampled:** by subject, stratified by compound group, written to a file; frames never split across it.
- **Sampler** weights: 1-10 s undetected gaps, the 13 flagged still-to-end videos, Fentanyl (likely LORR), and deliberate mining for LORR candidates (rare class). Frames are restricted data: written under `outputs/` only (gitignored).
- **Pre-labels:** classical-tracker centroid where `detected`, open-vocabulary detector ("fish") elsewhere, so annotators correct rather than draw. Labeling page reuses the webapp hardening (host/Origin checks, `textContent`, atomic per-frame JSON with annotator + timestamp), exports COCO-keypoints.
- **Environment:** torch etc. as an optional `[ml]` extra (not in core deps); CUDA wheel installed into the 3.11 venv; gate = `torch.cuda.is_available()` under WSL.
- **Licences to verify, not assume** (recorded below when checked): Ultralytics YOLO (believed AGPL-3.0), YOLOX/MMDetection/MMPose/torchvision (believed Apache/BSD), DeepLabCut (LGPL), SLEAP (BSD).
- **Pilot:** open-vocabulary detector (OWLv2 / Grounding DINO / YOLO-World) on ~20 frames weighted to classical misses; COCO has no fish class, so a COCO-pretrained detector is not zero-shot usable. Tells how much can be pre-labelled; not the final model.
- **Later-stage constraints:** GPU tracker must not be spawned per CPU worker (22 x model in 12 GB): single inference process fed by decode workers, or workers 1-2 when the GPU tracker is selected. The re-run writes to a **separate `PDS_OUTPUT_DIR` (e.g. `outputs_r3/`)** - a normal run skips EDITED videos (stale labels) and `--force` would destroy reviewer work; a separate dir also gives the side-by-side baseline. New tracker changes speed noise, so **recalibrate to profile r3** first (`scripts/build_track_cache.py` needs a tracker argument).

Fixed success metrics (on the frozen held-out split):
1. detection recall on frames the classical tracker missed;
2. keypoint error (px);
3. drop in Undetermined share vs the Phase 13 baseline (median 14%, 29/328 videos > 50%);
4. agreement with classical labels where both trackers detect;
5. Listing precision/recall.

| ID | Description | Status | Notes |
|---|---|---|---|
| T091a | Plan + metrics recorded here | DONE | This section. |
| T091b | Environment gate: `[ml]` extra, CUDA torch in venv, `torch.cuda.is_available()` | DONE | `ml` extra in pyproject (install the CUDA wheel first: `--index-url .../whl/cu124`); venv has torch 2.6.0+cu124, `cuda.is_available()` True on the RTX 4070 Ti under WSL, 4096x4096 matmul OK; transformers 5.17. Core suite unaffected. |
| T091c | Licence + weights verification of candidates | DONE | Checked via the GitHub/HuggingFace APIs (2026-09-23): OWLv2 `google/owlv2-base-patch16-ensemble` apache-2.0; Grounding DINO (tiny/base) apache-2.0; **Ultralytics YOLO AGPL-3.0**; YOLO-World GPL-3.0; YOLOX Apache-2.0; MMDetection/MMPose Apache-2.0; detectron2 Apache-2.0; DeepLabCut LGPL-3.0; SLEAP BSD-3-Clause-Clear. Plan: avoid AGPL/GPL code; fine-tune with torchvision (BSD) Keypoint R-CNN (box + keypoints in one model; COCO person weights as the starting point, K=5 keypoints) or MMPose/RTMPose. Not yet checked: torchvision pretrained-weight terms, and whether the project would ever ship as a service (matters only for AGPL). |
| T091d | Freeze subject-level train/held-out split file | DONE | `annotation/split.py` (deterministic, stratified by compound, groups < 5 videos stay in train, write-once file). `outputs/phase15/split.json` frozen from the 328 processed videos (seed 15, 20% held out). |
| T091e | Frame sampler (TDD) | DONE | `annotation/sampler.py` + `inputs.py`; `scripts/phase15_prepare.py` sampled 250 train frames (114 videos: gap_1_10s 88, detected 74, lorr_candidate 38, gap_long 25, flagged_tail 25) and 100 held-out frames (35 videos), extracted to `outputs/phase15/frames/` (gitignored). LORR candidates are only 'late-session slow Fentanyl frames': no ground truth exists, so the Listing class may still be very sparse. |
| T091f | Open-vocabulary pilot on ~20 frames | DONE | OWLv2 (fp16, prompt "a fish"/"a zebrafish") on 15 classical-missed + 5 classical-detected train frames: a box on 17/20, score > 0.2 on 12/15 of the classical misses (visually a plausible fish box on most of them; boxes sometimes also enclose the fish's reflection). Findings: (1) frames are 320x240 landscape (not 192x240) and show the fish clearly, so misses are tracker failures, not invisibility; (2) the beaker gives a mirror-image reflection of the fish that a blob tracker and a detector can both latch onto; (3) classical centroids differ from the detector box centre by 20-45 px on frames where both fire (3 of 5 pairs measured), so the classical centroid is a poor prefill: prefill with the detector box. |
| T091g | Labeling page (TDD, code review) | DONE (review fixes pending, see below) | `prepds annotate` (loopback, port 8001) serves `annotation/app.py` + static page over `outputs/phase15/`: frame list (status/split filters), canvas at 3x with drag-to-draw box, keypoint tools (keys B, 1-5; shift+click = occluded), Listing yes/no/unsure (required when a fish is visible), "no fish visible", suggested OWLv2 box (322/350 frames prefilled by `scripts/phase15_prefill.py`, drawn dashed until edited), live progress incl. Listing counts, `GET /api/export/coco/{train,heldout}`. Atomic per-frame JSON (`annotations/<frame>.json`, server timestamp, annotator), validation in `annotation/store.py` (22 rules tested), shared loopback/Origin guards extracted to `webapp/security.py`. Verified with Playwright on a scratch copy of the 350 frames (box suggestion, 5 keypoints, save -> next todo, listing-required message, no-fish path, COCO export); the real `outputs/phase15/annotations/` is still empty. Suite: 449 passed. Code-review fixes (RED test first): export refuses (409 `corrupt_annotations`, ids listed) instead of silently dropping unreadable annotations, and `status=corrupt` lists them so they can be relabelled; COCO width/height per image; `load_video_infos` refuses non-contiguous `frame_idx` (verified contiguous for all 328 real videos) and `cv2.imwrite` failure is checked; unique temp file per save (two tabs); frontend: Enter/keys ignored on focused buttons and with modifiers, sequence token in `openFrame`, dirty flag cleared only for the frame that was saved. Not fixed (noted): annotation is pointer-only (no keyboard placement); keypoints are not required to lie inside the box (fins/tails may legitimately fall outside); the Origin guard compares hostname only (no port) and allows requests with no Origin; no fsync on save. |
| T092 | Annotate; fine-tune box+keypoint model; evaluate on held-out | DONE for a first model (r1); more labels/epochs likely to help | **Run r1** (144 train labels, 60 epochs, batch 4, lr 0.005, 437 s on the RTX 4070 Ti; run dir `outputs/phase15/models/r1/`), evaluated once on the **45 held-out labeled frames (41 with a fish, 4 without)**: detection rate 41/41, recall@IoU0.5 0.93 (38/41), mean IoU 0.73, box-centre error median 7.4 px (p90 13.1), 0/4 false positives on no-fish frames; keypoint mean error snout 12.7, dorsal 11.0, ventral 18.8, tail base 5.3, **tail tip 41.7 px**. **Vs the classical tracker on the same held-out frames:** where the classical tracker had MISSED the fish (20 frames) the model detects all 20 (IoU>=0.5 on 19); where the classical tracker had fired (21 frames) its centroid was inside the true box on 20/21 and the model has IoU>=0.5 on 19. Caveats: held-out sample is small (n=41) and biased toward classical misses by design; a single split; single frames only (no temporal-flicker/reflection-swap check, which matters because speed is computed from successive positions); loss was still falling at epoch 60 (1.50), so it is not converged; **Listing tilt is not usable yet**: predicted dorsal->ventral tilt is 40.7 deg vs 10.9 deg ground truth on 'no' frames (n=39), and the 'yes' frames (n=2) are too few to say anything. Earlier notes (smoke run, needed label counts): Code done: `annotation/{examples,metrics,train}.py` + `scripts/phase15_finetune.py` (torchvision Keypoint R-CNN, COCO-pretrained backbone, new heads: fish + 5 keypoints; horizontal-flip and brightness/contrast augmentation only, never a vertical flip; loads only the train split for training; write-once run dirs under `outputs/phase15/models/`; metrics: detection rate, recall@IoU0.5, mean IoU, per-keypoint px error, false positives on no-fish frames, tilt by Listing tag). 18 tests (torch tests skip without the `ml` extra). **Plumbing smoke run on the first 10 labels** (all train; 4 Listing yes / 6 no; 0 held-out labeled), 60 epochs, GPU, 41 s: loss 8.6 -> 3.3, but still under-fit on its own training frames (recall@0.5 0.7, mean IoU 0.61, keypoint errors 3-67 px), so this proves the pipeline runs, NOT that the model works; no held-out metric exists yet. One early signal: in the human labels the dorsal->ventral tilt is 88 deg for Listing=yes vs 24 deg for no (n=4 vs 5), consistent with the keypoint-geometry approach to Listing (T094). Needed before a real evaluation: about 100+ labeled train frames and at least ~30 labeled held-out frames. |
| T093 | GPU tracker adapter behind `Tracker`; recalibrate r3; re-run into `outputs_r3/`; compare with baseline | DONE (2026-09-24; results unreviewed by a human) | **Done:** `strided_tracks.py` (detections every Nth frame, linear interpolation only between two detected samples, gaps stay undetected; 11 tests), `model_tracker.py` (`ModelTracker`, same `path -> list[Track]` contract, box centre as x/y, snout->tail_base angle as orientation; single process, 5 tests), `prepds run --tracker model:<run> [--stride 5] [--device]` (forces one worker, records the tracker in `run_report.json`). **Speed:** ~490 frames/s at stride 5 and min_size 400 (~73 s per 20-min video; batch benchmark 69/115/133 inferences/s at min_size 600/400/320, held-out recall@0.5 0.93/0.93/0.90) => a full 328-video re-run is ~6.6 h on the GPU (the GPU is the bottleneck, so more processes would not help). **First look on 4 real videos vs the Phase 13 classical tracks** (r1, stride 5): undetected frames 1% vs 88% (F_0238), 14% vs 89% (F_0340), 5% vs 53% (F_0124), 5% vs 21% (F_0047); identity jumps > 60 px per 5-frame step: 0/7065, 0/6114, 7/6846, 42/6801 (the last two suggest occasional swaps, probably to the reflection); a visual check of 12 random frames where only the model detected showed the marker on the fish each time (fish resting near the surface or on the beaker floor, some apparently tilted/inverted on F_0340). Median centroid offset vs the classical tracker where both detect is 38-60 px (box centre vs blob centroid; the classical blob often covers the reflection). **PAUSED 2026-09-24 by the user, then RESUMED the same day (see the resume note at the end of this row).** The calibration re-track (`scripts/build_model_track_cache.py --run outputs/phase15/models/r1`) was stopped cleanly at **48 of 54 videos** in `outputs/calibration/tracks_r1/` (resumable: it skips finished names; the run in progress was lost). Observed: most videos took 77-163 s, but two took 2401 s (methylone_30_0086) and 1341 s (methylone_30_0092), and a third was still running after ~7 min when stopped. Those videos are ordinary (about 20 min, 320x240, 5-7 MB), so the cause is environmental (GPU driver/WSL/Windows scheduling suspected, NOT confirmed); the tracker process held ~11.9 of 12 GB of VRAM (0.7 GB after it was stopped) and one CPU core. At ~150 s/video the full 328-video re-run would be ~14 h, not the 6.6 h estimated from the 73 s benchmark. **To resume:** (1) with the GPU otherwise idle, time ONE video (twice) to see whether ~73 s reproduces, and consider `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` / a Windows power plan set to High performance; (2) finish the cache (same command); (3) `python scripts/calibrate_thresholds.py --cache outputs/calibration/tracks_r1` and freeze profile r3 with `write_calibration_profile`; (4) `python -m prepds run --tracker model:outputs/phase15/models/r1 --config <r3 profile>` into a separate `PDS_OUTPUT_DIR` (never over `outputs/processed`), then compare with the Phase 13 baseline. **RESUME NOTE (2026-09-24):** with the GPU otherwise idle a clean video takes **74-75 s (485 frames/s, 2.1 GB peak VRAM)** twice in a row, so the earlier slowdowns were environmental, not the tracker; the calibration cache was finished (54/54, ~75 s each, no stalls). **Profile r3-model** (`config/calibration_profiles/cal-2026-09-24-r3-model.yaml`, model tracker only; r2 stays the classical profile): floor 13.34 px/s, min_freeze_bout 0.57 s, erratic 14.74 px/s; in-sample TV 0.177 (r2 on classical tracks 0.198), LOGO mean 0.253 (r2 0.246); the r2 thresholds on model tracks score 0.437 (~85% of frames Freezing) because box-centre speeds are lower than blob-centroid speeds. Determined-frame coverage on the calibration videos rises to 0.90-0.99 (classical 0.67-0.88). Same weak groups (MDMA 30: Freezing 79% vs 46% reference; methylone 100: Erratic 41% vs 6%); Controlled band is thin (1.4 px/s) and the floor is weakly identified (seed 2: 8.2 px/s, TV 0.181). Post-processing (features, labeling, export) is 0.5 s/video, so the run is GPU-bound. **Full re-run launched** (`--config <r3> run --tracker model:outputs/phase15/models/r1`, `PDS_OUTPUT_DIR=outputs_r3`, `PDS_ACCEPTED_DIR=accepted_r3`, both gitignored; log in the session scratchpad): 328 videos, ~6.8 h expected if no stalls; resumable (re-run the same command). Comparison with the Phase 13 baseline still to do when it finishes. |
| T094 | Listing rule/classifier from keypoints + `listing` tags; evaluate | DONE, outcome negative: no automatic Listing rule (tilt AUC 0.75, crop classifier AUC 0.82 with precision ~0.3); shipped as review-app hints only | **Findings 2026-09-24:** (1) In the human labels Listing=yes means an **inverted (belly-up) fish**: dorsal->ventral tilt median 172 deg (p10 158) vs 7 deg (p90 21) for 'no' (48 yes / 127 no labeled with both points); a tilt threshold of 35-65 deg gives precision 1.00 / recall 0.94 with human keypoints, i.e. the signal is strong if the keypoints are accurate. (2) **r1's predicted keypoints do not generalize to unseen videos**: on the 41 held-out fish frames 14/39 upright fish get a predicted tilt > 60 deg and only 1 of the 2 held-out Listing-yes frames is caught (on the training frames the same rule is precision 1.00 / recall 0.93, i.e. memorised). (3) **Run r2 (150 epochs, loss 0.92 vs 1.50) is NOT better on held-out**: recall@0.5 0.90 (r1 0.93), keypoint error snout 16.7 / dorsal 12.7 / ventral 21.9 / tail base 15.3 / tail tip 43.1 px (all equal or worse), tilt on upright 43 deg (r1 41), on the 2 yes frames 55 deg: it is over-fitting, so more epochs do not help. (4) The held-out set has only 2 Listing-yes frames, so no Listing rule or classifier can be evaluated yet. Needed: more labeled Listing-yes frames (especially in held-out videos) and probably a more direct approach (e.g. a crop classifier for 'inverted') than dorsal/ventral geometry. Also saved: the model tracker now writes `detections.parquet` per video (score, box, 5 keypoints with scores for every sampled frame, even below the score threshold) so Listing rules can be tried later without re-tracking; the full r3 re-run was stopped and NOT restarted pending a decision (r1 gives good detection but unreliable Listing). |

---

## 4. Dependencies (copied from PRD §12, unchanged)

```text
Phase 0 → Phase 1 → Phase 2 (catalog, needs real video mirror + workbook)
Phase 2 → Phase 3 → Phase 4 → Phase 5
Phase 4/5 (validated on real data) → Phase 6 (calibration)
Phase 6 → Phase 7 (labeling needs frozen thresholds — see §1 for the rule-code-vs-values split)
Phase 7 → Phase 8 → Phase 9
Phase 9 → Phase 10 → Phase 11 (webapp needs export + review_store first)
Phase 9 → Phase 12 (batch CLI needs export; independent of webapp)
Phase 10/11 + Phase 12 → Phase 13 (full integration needs both the batch pipeline and the review UI)
Phase 13 → Phase 14
```

---

## 5. §3.3 Edge-case catalog cross-reference (exit criteria for "complete")

Every bullet is a required task, not optional polish. Mapped to the task(s) that cover it.

| Edge case (PRD §3.3) | Covered by | Status |
|---|---|---|
| Video duration ≠ expected `Agent Exposure Time` (3 of 353 at 10 min, not 20) | T004(FR-004 tolerance config), T024/T025 (real fps/duration read), confirmed real in §0 | NOT_STARTED |
| Fish undetectable for a stretch → `Undetermined`, never forced | T032, T034, T051 | NOT_STARTED |
| Subject immobile/LORR for entire session — valid single-state strip | T049, T052 (Listing rule + no forced diversity) | NOT_STARTED |
| Subject dies partway → white `Dead`, flagged for reviewer confirmation | T052 (FR-007a rule), T054 (monotonicity), T063/T070 (FR-015a soft-warn Accept) | NOT_STARTED |
| Combo-treatment compound with folder status suffix (`FD-2-66 + methylone(DONE Compressed Only)`) | T021, confirmed real in §0 | NOT_STARTED |
| Trial with no video / video with no trial row | T019, T021, T022 | NOT_STARTED |
| Exact-duplicate workbook row (`Subject # 320`) | T018, T020, confirmed real in §0 | NOT_STARTED |
| Corrupted or zero-byte video file | T025 (`probe()` must raise/flag, not crash the whole batch) | NOT_STARTED |
| Measured fps deviates from nominal 30 (~29.83 observed) | T024/T025, confirmed real in §0 (29.8329) | NOT_STARTED |
| Accept hard-blocked while `Undetermined` segments remain | T063, T070, T076 | NOT_STARTED |
| Accept soft-warn+override when `Dead` segments present | T063 (asymmetric test), T070, T076 | NOT_STARTED |

---

## 6. Success Criteria (PRD §8) — exit tracking

- [ ] All 353 trial rows matched+processed or explicitly exception-listed with reason. *(352 logical
      after dedup; 337/352 have a local video as of 2026-09-22 — see §0)*
- [ ] Auto-classifier's group-level state proportions agree reasonably with the 2 reference figures
      post-calibration.
- [ ] Reviewer can take a video open → corrected → Accepted using only the review UI, no code/raw
      file access.
- [ ] `accepted/` gold dataset directly consumable (documented schema, no further transform) once a
      meaningful fraction is Accepted.

## 7. Acceptance Checklist (PRD §13) — exit tracking

- [x] Requirements testable/unambiguous where information was available. *(PRD-level, pre-existing)*
- [x] Scope bounded via explicit Non-Goals. *(PRD-level, pre-existing)*
- [x] Ground-truth assumptions stated explicitly. *(PRD-level, pre-existing)*
- [x] Technical approach/architecture/data model/tasks unified with consistent cross-references.
      *(PRD-level, pre-existing)*
- [x] OD-1..OD-4 resolved. *(PRD-level, pre-existing)*
- [x] `Dead` vs `Undetermined` distinction threaded consistently. *(PRD-level, pre-existing — to be
      re-verified against actual code once Phase 7-9 land)*
- [ ] Reviewed and approved by project owner before implementation begins. *(owner-only step, not
      blocking Step 2 per the task instructions — plan proceeds directly to implementation)*

**Phase 15 note, 2026-09-24 - Listing candidate batch (`batch_002`).** Because random sampling yields few Listing-yes frames (2 in held-out), `scripts/phase15_mine_listing.py` scanned every 180th frame of all 328 processed videos with r1 (cache `outputs/phase15/scan_r1.parquet`, about 35 min, not the 10 min estimated) and selected frames by predicted dorsal->ventral tilt (`annotation/mining.py`, 8 tests): 140 frames from 91 videos in 16 compounds (train 60 likely-inverted + 20 ambiguous 60-100 deg; held-out 45 + 15), at most 4 per video, at least 20 s apart, not within 2 s of an existing sample. The prefill is the box only (no keypoints, to avoid anchoring the annotator). The label is the annotator's answer, not the model's tilt. The frames are biased toward what r1 thinks is inverted, so the labeled batch must not be used to estimate Listing prevalence. Listing-yes was found mostly in DOB / FD-2 series, not Fentanyl (the earlier `lorr_candidate` bucket yielded none). Full suite: 510 passed. Labeling-page Batch / "Why sampled" filters are not yet browser-verified.

**Phase 15 note, 2026-09-24 - run m3 (330 train labels incl. 75 Listing-yes, 60 epochs, loss 1.46, 971 s) on 160 held-out frames (148 with fish, 12 without).** Detection: rate 1.0, recall@IoU0.5 0.95 (r1 0.93 on a smaller set), mean IoU 0.74. Keypoint error px: snout 23.1, dorsal 14.8, ventral 19.4, tail base 11.5, tail tip 39.5. **Listing from predicted dorsal->ventral tilt is still NOT usable:** on 23 yes / 100 no held-out frames AUC 0.75; tilt >= 100 gives recall 0.48 / precision 0.44, tilt >= 60 recall 0.70 / precision 0.33 (yes-frame tilt median 92 vs 172 in human labels). Caveat: the held-out "no" set contains hard negatives that r1 scored as inverted (batch_002 selection), so precision here is pessimistic, but recall is not affected by that and is also too low. Conclusion: keypoint geometry does not give a reliable Listing rule; next candidate is a crop-level inverted classifier trained on the human listing labels (not started).

**Phase 15 note, 2026-09-24 - T094 crop classifier c1 (ImageNet ResNet-18, 271 train crops of which 75 yes, 15 epochs, box jitter, hflip + brightness/contrast only).** Code: `annotation/crop_classifier.py` (6 tests; an image-edge crash was found on real data and fixed with a RED test first), `scripts/phase15_listing_classifier.py`. Held-out (23 yes / 100 no), no threshold tuned on it: on the detector m3's predicted boxes AUC 0.82 (tilt rule: 0.75), recall 0.96 and precision 0.31 at p>=0.5 (48 false positives), precision 0.36 at p>=0.9 with recall 0.91; on human boxes AUC 0.80. Scores are saturated (many normal frames at 1.00). The false positives I inspected are side-on fish with a visible reflection, i.e. visually ambiguous at this resolution. Held-out prevalence here is about 19% Listing (enriched by the mining), so real-video precision would be lower still. **Conclusion: not reliable enough to label Listing automatically; possibly useful as a high-recall "possible Listing" flag for human review.** Not wired into the pipeline.

**Phase 15 note, 2026-09-24 - "possible Listing" hints in the review app.** `listing_flags.py` (merge scores into ranges, `listing_flags.json` sidecar, atomic write; a corrupt sidecar is reported in the API as `listing_flags_error` and never blocks reviewing), API fields `listing_flags` / `listing_flags_error`, review page: hint list with "Go to" buttons plus hatched timeline markers (pointer-events off, so drag-selection is unaffected). `scripts/phase15_flag_listing.py --detector m3 --classifier c1` scans each processed video every 3 s and writes the sidecar (skips videos that have one unless --force). Hints are advisory only: they never touch states, manifests or the accept rules. Verified in a browser on F_0006 (84 markers, list rendered). **Noise:** at p>=0.9 with single-sample flags F_0006 had 84 flags (37% of sampled frames) and F_0007 16 (5%); requiring >=2 consecutive samples (default `--min-samples 2`, about 6 s) leaves 37 and 3. That default is a guess, not tuned: we have no labels for how long real Listing bouts last. The full scan of all 328 videos has NOT been run (est. 1 h+ GPU). Suite: 525 passed.

**Phase 15 note, 2026-09-24 - hint scan, recalibration r4, full m3 re-run started.** (1) Hint scan (`phase15_flag_listing.py`, m3 + c1, p>=0.9, >=2 consecutive samples) finished on all 328 processed videos (about 45 min): median 25 hints per video (p90 48, max 63; 17 videos with none). That is a lot of hints for a reviewer; the threshold/min-samples defaults are untuned and may need tightening. (2) `m3` track cache for the 54 calibration videos built (about 69 min). (3) Recalibration on the m3 tracks (3 seeds, consolidated 1 s output): placeholder TV 0.443 -> best 0.1768 in-sample, LOGO 0.203 (r3/r1 model: 0.177 / 0.253; classical r2: 0.198 / 0.246). Seeds agree: freeze floor ~6-7 px/s, erratic ~12-13 px/s, min freeze bout at the search lower bound (0.03 s). Coverage 0.93-1.00. Weak groups unchanged: MDMA 30 (TV 0.311), methylone 100 (0.335). Frozen as `config/calibration_profiles/cal-2026-09-24-r4-model-m3.yaml`. (4) Full re-run started: `PDS_OUTPUT_DIR=outputs_r3 PDS_ACCEPTED_DIR=accepted_r3 python -m prepds --config <r4 profile> run --tracker model:outputs/phase15/models/m3`, single process on the GPU, resumable, log /tmp/rerun.log. Comparison with `scripts/phase15_compare.py` pending.

**Phase 15 / T093 result, 2026-09-24 - full m3 re-run (`outputs_r3`, profile r4) vs classical run (`outputs`, profile r2).** Run: 328 processed, 0 failed, 24,683 s (6.9 h), single GPU process. `scripts/phase15_compare.py` over the 328 shared videos: undetected frames median 21.5% -> 0.4% (mean 25.2% -> 2.1%); Undetermined frames median 13.7% -> 0.0% (mean 19.4% -> 1.1%); videos with Undetermined > 30%: 70 -> 4; > 50%: 29 -> 0; videos with any review flag 13 -> 7. State mix (mean share) Controlled 9.7 -> 16.1%, Erratic 25.2 -> 31.3%, Freezing 45.7 -> 51.3%, Dead 0 -> 0.2%. **Label agreement with the classical run on frames both determined is only median 63% (mean >80% overall)**, which is expected because the tracker (box centre vs blob centroid) and the thresholds (r4 vs r2) both changed; it is NOT a measure of correctness: there is no frame-level ground truth for behaviour states, and the only external check is the group-level TV against the digitized reference (in-sample on 54 videos: 0.177 vs 0.198 for r2; LOGO 0.203 vs 0.246). The single Dead video (M_0317, 60% Dead) is the fish found motionless from 482 s to the end, previously Undetermined (74%); it keeps its terminal_no_action review flag and needs human confirmation (Dead is a soft confirm at accept). Remaining Undetermined > 30% videos: F_0037, M_0062, M_0294, M_0319. Not verified: the Listing/LORR state is still never produced by the rule (hints only); MDMA 30 and methylone 100 remain poorly fit. Human review (T085) of a sample from the new run has not been done. `outputs_r3` is a separate, unreviewed tree; the review app still points at `outputs/processed` unless PDS_OUTPUT_DIR is set.

**Review page layout fix, 2026-09-24.** With ~25 Listing hints per video the hint list pushed the player below the fold, so "Go to" seemed to do nothing (seeking itself worked). The player, the hint list (reviewer flags plus Listing hints, scrolling) and the timeline now form a sticky block at the top of the review pane, video on the left and hints on the right (stacked below 800 px), so the video is always next to the clicked "Go to". Verified in a browser at 1200x650 (video and hints visible together after scrolling; click seeks). Not checked: touch/mobile widths beyond the CSS rule, other browsers.

**Review corrections, 2026-09-24 (code-reviewer + mle-reviewer; model output, each claim below was checked against the data).**
*Retractions / corrections to earlier notes:*
1. The T093 note said label agreement was "median 63% (mean >80% overall)". Wrong: the mean agreement is 63.6% (median 63.4%); ">80%" was the share of frames both runs determined. The two runs disagree on about 37% of the frames both label, because tracker and thresholds both changed.
2. **m3 no-fish false positives were omitted:** 6 of 12 held-out no-fish frames get a detection (r1: 0 of 4). "Detection rate 1.0" only covers frames that have a fish. For a tracker this means reflections / empty-tank detections are possible; only the terminal_no_action flag covers it today.
3. **Annotation anchoring in batch_002:** the box prefill came from r1's own prediction, and 69% of the 137 human boxes in that batch are exactly the prefill (median IoU 1.0). So the boxes of the 60 mined held-out frames are correlated with r1 (and m3 is the same architecture). The first batch was prefilled by OWLv2 (322/350), also an anchor, but independent of our model. Clean numbers are therefore the random-batch frames: recall@0.5 0.978 at min_size 600 and **0.901 at min_size 400 (91 frames)**; the mined batch gives 0.912 / 0.965 and is optimistic. The headline 0.95 was measured at min_size 600 in fp32; production (`ModelTracker`) uses min_size 400 with fp16 autocast, so use ~0.90-0.92, not 0.95 (fp16 itself not re-measured).
4. Calibration/comparison are largely in-sample for the tracker: 44 of the 54 calibration videos and 264 of the 328 compared videos were in m3's train split (10 / 64 held-out). Checked: the detection gain holds on held-out videos alone (64 videos: undetected median 19.0% -> 0.4%, mean 23.0% -> 2.7%; train videos 22.7% -> 0.4%), so it is not a train-set artefact. But fewer undetected frames is not track accuracy: there is no accuracy metric for m3, and reflection swaps at stride 5 are not quantified.
5. The r4 fit is weakly identified: min_freeze_bout sits on the search lower bound (0.03 s), the freeze floor moved 13.3 (r3) -> 6 (r4) px/s (about 1 px of box-centre jitter per step), LOGO folds used 300 random starts vs 600 for the main fit, and the TV gain (0.177 vs 0.198 in-sample) is a fit-to-target on 3 parameters and 54 videos, not a validation of behaviour labels. Read the state-mix shifts (Controlled 9.7 -> 16.1%, etc.) as calibration effects, not as detections of more behaviour.
6. c1: "recall 0.96 / precision 0.31-0.36" mixes thresholds: p>=0.5 gives recall 0.96 / precision 0.31; p>=0.9 gives recall 0.91 / precision 0.36. No confidence interval was computed: with 23 positives (correlated frames from the same videos) the AUC 0.82 is roughly +-0.1. The 0.9 hint threshold was chosen after seeing the c1 threshold table, so "no threshold tuned on held-out" should be read as: the classifier was not tuned, the hint threshold was picked by looking.
7. Held-out was also used for small choices: r1 vs r2 (60 vs 150 epochs) and min_size 400 vs 600. Held-out is therefore not untouched for the detector.
*Code fixes made (each with a failing test first where testable):* stale `detections.parquet` no longer survives a failed forced export (pipeline unlinks it before export); zero-area/degenerate boxes get a minimum 8 px crop window; sidecar reader rejects NaN/inf; timeline markers clamp to the timeline; classifier weights saved atomically; hint scan reads/scores 16 frames at a time, refuses unusable fps/frame-count metadata and unreadable frames (no sidecar is written, the failure is listed and the exit code is 1); mining cache name now includes the stride, the batch file and cache are written atomically, frame rows are looked up by frame_idx (not by position). Annotation hashes and the split hash as of now: `outputs/phase15/snapshots/annotations_2026-09-24.json` (reproducibility: training did not record them).
*Not fixed / accepted:* variable-frame-rate drift between the hint time (index/fps) and the review timeline and seek-vs-decode off-by-a-frame in the mined PNGs (LOW); training is not bit-reproducible (AMP, no cudnn.deterministic); run.json lacks code SHA and package versions; reflection-swap and single-missed-sample (9-frame gap) effects of stride-5 tracking are not quantified; `run_report.json` does not stop someone pairing profile r4 with a different tracker.

**Status reconciliation, 2026-09-24.** The task table has stale duplicate rows: T025, T028, T042-T046 near lines 300-690 are old "planned" rows, and the work exists (`video_io.py`, `roi.py`, `digitize_reference.py`, `calibrate.py`, `reference_targets.json`, frozen profiles r1-r4, tests); T029/T030 are DONE in the Phase 3 table (`docs/strain_tracking_notes.md`). **Genuinely open:** T008 (scaffold commit: done by the branch commits made on 2026-09-24), T078 (UX pass covered 3 real videos, not 10), T082 (done: `--workers` is in `run --help` and the README), **T085 (human review of a stratified sample: waiting on the owner)**. Phase 15 known limitations are in "Review corrections" above; Listing/LORR remains a manual label.

**Experiments with ideas from PR #7 (`demo-pipeline`, `fishbehavior` package), 2026-09-24. Nothing adopted; scratch scripts, results in `outputs/experiments/` (gitignored).** Protocol fixed before any run (advisor): label-side ideas scored on the 54-video m3 track cache with consolidated 1 s output vs group targets (Listing excluded), main fit 3 seeds x (300 random starts + 5 refine rounds), leave-one-group-out (LOGO) with the SAME 300 starts (seed 1), GMM fitted on training groups only; tracker-side ideas scored on the 148 held-out fish frames at the production setting (min_size 400, fp16); workbook check = Spearman of video speed vs the NTT workbook (a DIFFERENT session; 236 videos with values; paired bootstrap CI).
| Idea | Metric | Baseline | Variant | Change | Noise floor | Verdict |
|---|---|---|---|---|---|---|
| Overlap/continuity box choice (their reflection rule), 6-sample chain | held-out recall@0.5 | 0.932 | 0.885 | -0.047 (13 frames differ) | about +-0.02 (n=148; top-1 itself moved 0.932/0.926 between runs) | worse |
| Same, 1 previous sample | held-out recall@0.5 | 0.926 | 0.905 | -0.021 (7 frames differ) | about +-0.02 | no gain (slightly worse); no-fish false positives unchanged (7/12) |
| Body-length normalised speed | in-sample TV / LOGO TV | 0.181 / 0.2034 | 0.171 / 0.2149 | -0.010 / +0.011 | seed spread 0.011 / LOGO paired SE 0.013 | within noise (in-sample better, LOGO worse) |
| same | workbook rho (TDM) | 0.347 | 0.381 | +0.034 (95% CI -0.028..+0.095) | | within noise |
| Savitzky-Golay speed (1 s window) | in-sample TV / LOGO TV | 0.181 / 0.2034 | 0.177 / 0.2066 | -0.004 / +0.003 | 0.003 / SE 0.008 | within noise |
| same | workbook rho (TDM) | 0.347 | 0.360 | +0.013 (CI -0.001..+0.027) | | tiny, at the edge of noise |
| Pooled 2-component mixture for Controlled/Erratic (speed + turning rate; removes the erratic threshold) | in-sample TV / LOGO TV | 0.181 / 0.2034 | 0.184 / 0.1983 | +0.003 / -0.005 | 0.001 / SE 0.018 | within noise; 5 of 7 groups better but MDMA 30 much worse (0.322 -> 0.418); one parameter fewer |
| Workbook check, tracker level (1 s displacement speed, same measure for both) | rho (TDM) classical vs m3 | 0.389 | 0.347 | -0.041 (CI -0.082..-0.002) | | no improvement; m3 slightly worse (within-compound 0.36 vs 0.26) |
| Per-subject, per-second reference timelines | per-second F1 | - | - | - | - | not measured: needs a hand-filled subject-to-row mapping (row labels illegible) |
| Head (eye) based surface breach | - | - | - | - | - | not measured: reference has ~0% Surface Breach, nothing to score |
| Their LORR rule, CI/privacy job, window export | - | - | - | - | - | not applicable (Listing here is a belly-up fish; CI collides with PR #7's ci.yml; no effect on results) |
**Retraction:** a first workbook comparison in this session used the frames.parquet `velocity` column (frame-to-frame, jitter-dominated for the classical tracker) and suggested m3 improved the correlation (0.25 -> 0.35). With the same 1 s-displacement speed for both trackers the sign flips (0.389 classical vs 0.347 m3). The earlier reading was an artefact. **Conclusion:** none of these ideas gave a measurable gain on our metrics, and the workbook check gives no evidence that m3 improves speed measurement (its clear gain is detection coverage). Our metrics have limited resolution (7 groups, 54 videos; differences under about 0.01-0.02 are undetectable), so "no evidence of gain" is not "no gain". The per-subject timeline comparison would be the missing accurate check.

**Follow-up on PR #7 (2026-09-24): per-subject reference, eye-based surface breach, their thresholds and UI. Nothing adopted; uncommitted note.**
1. **Per-subject reference timelines: not measurable.** Rows read from both reference PNGs (about 10 rows per panel; workbook groups also have 10 subjects; 6 panels usable: MDMA 30 has only 5 local videos, MDMA 100 / Fentanyl 100 / vehicle have no usable subject list). The hypothesis "rows are the group's workbook subjects in numeric order, highest on top" was tested and NOT supported: per-second macro-F1 is 0.356 (highest-on-top) / 0.399 (lowest-on-top) vs 0.377 mean and 0.399 p95 for random subject-to-row assignments (classical run; m3: 0.367 / 0.422 vs 0.393 / 0.420); subject-level state-share rank correlation 0.02-0.19 vs null p95 0.24-0.26. The subject-to-row mapping has to come from the slide author.
2. **Surface breach.** Our rule flags a frame only when the fish y is <= 20 px from the top of the FRAME; in the videos checked the waterline is at y=131-170 and fish positions start around y=100-160, so it cannot fire (0.0% in both full runs, structural, not calibration). Their rule (head/eye within 0.25 body lengths of the waterline, pitch >= 25 deg nose-up, seen side-on) does fire (0.3% on their F_0029). A quick untuned port on our m3 keypoints (their waterline, snout as head, 14 of the calibration videos) over-fires: 0.8-12% of seconds vs reference 0.0-0.45%; the reference mass is too small to calibrate it reliably.
3. **Their thresholds (screenshot of their live page, video F_0029, calibrated values):** surface breach >= 0.435 of frames nose-up at the surface; lorr tilt fraction >= 0.4711 with median speed < 0.9735 BL/s for >= 5 s; freeze median speed < 0.2658 BL/s for >= 5 s; slow swim < 0.5 BL/s; Controlled/Erratic split by an unsupervised pooled mixture (no threshold). Units are body lengths per second from a 0.2 s Savitzky-Golay speed per 1 s bin, so they do not transfer to our 1 s-displacement px/s thresholds. Ran their pipeline as shipped on the same 54 videos (6 min) and scored it with our measure on the same 38-video cohort: mean TV 0.290 with their defaults, 0.250 with the screenshot thresholds (coverage 1.0) vs ours 0.176-0.184 in-sample / 0.198-0.215 leave-one-group-out (classical profile r2: 0.198 / 0.246). Theirs is much better on MDMA 30 (0.10-0.16 vs our 0.30-0.42) and worse on DOB / fentanyl (0.29-0.38 vs ours 0.07-0.17): it over-labels Controlled (32-51% vs reference 6-15% in active groups) and under-labels Erratic. Caveats: the thresholds come from one screenshot (other calibrated keys, e.g. min bout, unknown), their calibration used the same reference figure (partly in-sample for them), the MDMA 30 group has 5 videos.
4. **UI and time per state:** see the answer given to the owner; ours lists seconds only for states with time and has no percent or bouts; theirs has seconds, percent, bouts and bars for all six states, a per-second rule checklist ("why this label"), a scene editor (waterline click, ROI drag, eye/ellipse overlays) and per-second click-to-review. Ours has the review workflow (statuses, edit/accept/reject, Undetermined hard block, Dead confirm, listing hints, accepted index) which theirs lacks.
Leftovers outside the repo: git worktree /tmp/theirs (remove with `git worktree remove --force /tmp/theirs`), their outputs in /tmp/theirs_out, remote-tracking ref origin/demo-pipeline.

**Review page: "Why this label" and fuller "Time per state" (borrowed from PR #7's live page), 2026-09-24. Uncommitted.** New `prepds/explain.py` (`VideoExplainer`: replays the labeling rules on a video's stored track with its calibration profile's thresholds, then the 1 s consolidation; reports per second the ordered rules with the numbers next to their thresholds, the frame votes, and whether a reviewer changed the automatic label), API `GET /videos/{id}/explain?second=N` (profile read by name only from `config/calibration_profiles/`, small in-memory cache, 404 unknown second, 422 bad input; when the profile is missing or unreadable the measurements are still returned with the reason), `frame_summary` gained `bouts_by_state` and `total_s`. Page: the panel follows the video's playhead (fetch on second change / seek, stale responses dropped); "Time per state" now lists all seven states with bar, seconds, percent and bouts. 12 tests added (test_explain, test_explain_api). Fidelity of the replay against the stored labels on real videos: model run 4 of 5,377 seconds differ (4 videos), classical run 8 of 9,882 (0.08%): the differing seconds sit on a threshold and the stored track is float32-rounded; the panel says so. Verified in a browser on M_0317 (profile r4). Not done: click-to-review on the ethogram, waterline/ROI editor, mobile layout beyond a CSS rule. Suite: 541 passed.

**Review page: fish markers on the video, 2026-09-24 (uncommitted).** API `GET /videos/{id}/overlay?start_s=&end_s=` (window at most 30 s): per-frame track point in video pixels (null, never 0, for undetected frames), the coded frame size (`width`/`height`, read once per video with OpenCV, null if the source cannot be read), and for model-tracker videos the detector samples (box + 5 keypoints, only above the tracker's 0.3 score floor). A corrupt `detections.parquet` is reported in `detections_error` and does not hide the track. Page: a canvas over the player with toggles for the track point (red), the last 2 s path (orange), the detector box (yellow) and the keypoints (snout magenta, tail green, dorsal/ventral cyan; joined by thin lines). Classical-tracker videos show only the track point and path (no box or keypoints exist for them). **Bug found only by a real-data check:** F_0029 decodes as 320x240 but the browser reports 320x257 (non-square pixel aspect), so one uniform scale left the box 16 px short of the frame bottom; each axis is now scaled from the coded size (verified: painted rows reach the video bottom). Marker alignment was also checked by drawing the same data on raw OpenCV frames at 3 times of F_0029 (box encloses the fish, snout at the head, track point at the box centre). Click-to-review needed no new code: a click on the timeline already seeks the video and the "Why this label" panel follows. A waterline/ROI editor was deliberately NOT built: nothing in our labeling uses a waterline (Surface Breach uses the frame top), so an editor would have no consumer; decide the surface-breach rule first. Tests: 7 in test_overlay_api.

### Review page: bigger video, external controls, waterline (2026-09-24)

- The video is now about 470 px wide (`min(58%, 52vh)`) and the native controls are removed; play/pause, seek bar, clock and speed sit below the picture, and a click on the video toggles play, so nothing covers the box or keypoints.
- Box and keypoints exist only for model-run output (`detections.parquet`, e.g. `outputs_r3`). Classical output has no box, so the page now says so (`has_detector` in `/overlay`) instead of silently drawing nothing.
- Waterline: manual only. "Set waterline" then a click on the video stores `waterline.json` beside the artifacts (`GET/PUT/DELETE /videos/{id}/waterline`, validated to lie inside the frame). It is drawn as a dashed blue line. It is advisory: no labeling rule reads it. An automatic waterline was not added because `roi.detect_waterline` finds the beaker bottom in most videos (see its docstring).
- Fixed: a NaN in a request body made the default 422 response fail to serialize (HTTP 500); the app now returns a plain 422 without echoing the input.
- Tests: `tests/test_waterline_api.py` (13). Full suite: 560 passed. Browser-checked on F_0029 at 100.2 s (marker alignment, waterline click, play/pause).
