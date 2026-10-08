# PRD: Zebrafish Drug-Response Detection System

This is the single product requirements document. **Part I** is the preprocessing dataset system (`prepds`: tracking,
features, behavior-state labels, review app, gold dataset). **Part II** is the compound/dose classifier and research
chat (`dcs`), merged in on 2026-10-08 from the former `classifier_PRD.md`. **Part III** collects the watch list and the
model roadmap (lightweight reasoning model now, a better vision model once there is more data).

| Part | Package | Status |
|---|---|---|
| I | `src/prepds/` | Built; 328 of 352 trials processed; human review still to do |
| II | `src/dcs/` | Built and tested on synthetic data; first real run on the GPU box pending |
| III | - | Living list; update it when an item closes |

# Part I: Preprocessing Dataset System

**Project:** Zebrafish Drug-Response Detection System — early-stage labeling pipeline
**Location:** `fish-detection/src/prepds/`
**Status:** Built (Phases 0-15 in [progress.md](progress.md)); see Part III for open items
**Created:** 2026-09-22
**Origin:** Owner request — build an early-stage, semi-autonomous pipeline that turns
each raw drug-exposure video plus its trial-metadata row into a reviewed, per-subject
behavior-state "strip" diagram (and machine-readable labels), because no frame-level
ground truth currently exists for any of the 353 trials.

---

## 1. Overview

Logan Kountz's dissertation work classifies zebrafish behavior during acute drug
exposure into five ethogram states — **Controlled Swim, Erratic Movement,
Freezing/Drift, Listing/LORR, Surface Breach** — and visualizes each subject as a
colored horizontal timeline ("strip"). Two example composite figures (multiple
subjects, stacked, grouped by compound/concentration) exist as slides in the defense
deck; no frame-level labels back them, and no documented, reproducible thresholds
exist for how a frame is assigned to a state.

The broader lab project (tracked in `docs/zebrafish_drug_detection_scope.md`, in the
existing `fish-detection` repo) intends to eventually train a supervised model that
predicts drug identity from behavior. That model needs labeled (video → per-frame
state) training data, which does not exist yet.

**This project is the system that creates that labeled data.** For each of the 353
trial videos, it must:

1. locate and track the fish using computer vision (no human watches the raw video
   frame-by-frame to decide behavior — the CV pipeline does),
2. derive kinematic features per frame from that tracking (velocity, turning,
   posture, depth),
3. auto-classify every frame into one of the five behavioral states, the terminal
   `Dead / No Action Recorded` state, or the internal `Undetermined` marker, purely
   from those computer-vision-derived features — this classification **is** what
   determines each color band in the output diagram (see the worked example in
   [§9.4.3](#943-worked-example-tracking--diagram)),
4. render a single-subject strip diagram in the reference visual style,
5. let a human reviewer correct it against the source video, and
6. commit only human-**Accepted** results into a growing, versioned "gold" dataset.

This project explicitly stops at that gold dataset. Drug-identity classification,
anomaly detection, natural-language reporting, multi-subject composite figures, and
live/real-time operation are handled by later phases of the broader project and are
**not** built here (see [§4 Non-Goals](#4-non-goals)).

---

## 2. Clarifications

Two rounds of clarifying questions were resolved with the project owner before this
PRD was written. Key resolutions:

| # | Topic | Resolution |
|---|---|---|
| C1 | Video phase | All 353 videos are the **20-minute acute drug-exposure phase** (beaker, side view), one video per trial. No Novel Tank Test (NTT) videos are available. |
| C2 | Excel movement columns | `TDM`, `Velocity`, and `Time Spent (Top/Bot)` in `00_NTT_DataBase.xlsx` describe the **10-minute NTT phase** (confirmed: `Time Spent Top + Bot` sums to ≈600s, not ≈1200s), a **different recording session** than the exposure video. These columns are carried as reference metadata only and **must not** be used as ground truth or as a calibration signal for per-frame state classification. |
| C3 | Excel usable fields | `Subject #`, `Sex`, `Strain`, `Compund`, `Conc. (mM)`, `Date of EXP`, `Agent Exposure Time (min)` are the fields this project actually consumes. |
| C4 | Output unit | Per video, the system outputs **one subject's strip** (potentially many color segments across the timeline). It does **not** assemble the multi-subject, multi-panel composite figure — that is a trivial downstream step, out of scope here. |
| C5 | Classification resolution | Raw 30 fps, per-frame (matching each video's actual measured fps, not an assumed constant). |
| C6 | Ground truth / calibration | No frame-level ground truth exists anywhere except the two reference images supplied by the owner. The system performs a one-time **supervised calibration pass** (digitize the reference images → tune thresholds → freeze them), then runs unattended per video, with human review as the actual accuracy gate. |
| C7 | Review workflow | Local, single-user, browser-based review tool. Reviewer can click-drag a time range on the strip and reassign its state (frame-accurate boundaries, not literal per-frame clicking). Reviewer sees the source video **playing back in sync** with the strip. **Accept**, **Save/Submit** (after manual edits, not itself an accept), and **Reject** (discard and regenerate from scratch) are distinct actions. |
| C8 | Export format | All three of: per-frame table, run-length segment list, and rendered PNG strip — decision on which is authoritative deferred to implementation / later use. |
| C9 | Environment | Local single-user desktop tool on the owner's GPU workstation (RTX 4070 Ti, 12GB VRAM). No auth, no multi-user concerns. |
| C10 | Repo placement | New top-level folder at the repo root: `fish-detection/preprocessing_dataset_system/`, independent of `src/fishbehavior/`. |
| C11 | Video source | 353 videos exist on OneDrive, organized as `{Compound}/{Sex}_{SubjectNumber}.mp4`, e.g. `FD-2-45/f_0020.mp4`, `FD-2-97/F_282.mp4`. Casing (`f_`/`F_`) and zero-padding width (`0020` vs `282`) are **inconsistent** and must be normalized during matching, not assumed uniform. |
| C12 | Real OneDrive folder naming (resolves OD-1) | Folders carry a literal, directly-appended status suffix, e.g. `FD-2-66 + methylone(DONE Compressed Only)`, `FD-2-67(DONE Compressed Only)`, `Fentanyl(DONE Compressed Only)`. This suffix is **not** part of the compound name and must be stripped before matching against the workbook's `Compund:` values. |
| C13 | Duplicate workbook row (resolves OD-2) | Confirmed: if two workbook rows are fully identical, it **is** a data-entry duplicate. Deduplicate automatically (no manual disambiguation step needed) — applies to `Subject # 0320` and any future case matching the same pattern. |
| C14 | Reference materials handling (resolves OD-3) | The two reference PNGs and the thesis PDF **are restricted data** — never committed to git, handled exactly like the videos/workbook (local, gitignored). |
| C15 | Video access (resolves OD-4) | Confirmed: videos currently live on OneDrive; the owner will download/sync them to local disk to support development, matching the local-mirror approach already recommended in [§9.5.2](#952-video-access--resolves-od-4). |
| C16 | Meaning of white (`#FFFFFF`) in the reference figures | Confirmed: white specifically means **the fish is dead — no action recorded**, not a generic "unknown/low-confidence" marker. This is a real terminal biological event with its own reference-palette color, distinct from the pipeline's own internal `Undetermined` (tracking-confidence) marker. See [§5.2](#52-reference-color-legend) and [§9.5.6](#956-labeling-labelingpy--fr-007). |

### 2.1 Open items — resolution log

All four items raised in the first draft are now resolved (see C12–C15 above);
kept here as an audit trail rather than deleted, per standard clarification-log
practice.

- ~~**OD-1** Exact OneDrive folder naming for combo treatments.~~ **RESOLVED — see C12.**
- ~~**OD-2** `Subject # 0320` duplicate-row handling.~~ **RESOLVED — see C13.**
- ~~**OD-3** Whether reference materials may be committed to git.~~ **RESOLVED — see C14.**
- ~~**OD-4** Local video access strategy.~~ **RESOLVED — see C15.**

No outstanding `[NEEDS CLARIFICATION]` markers remain at this time.

---

## 3. User Scenarios & Testing

### 3.1 Primary user story

As the researcher running this pipeline, I point it at a local folder of trial videos
and the trial workbook. I run a one-time calibration against the two known reference
figures. I then run the batch pipeline unattended over all 353 videos. For each video,
I get a candidate single-subject strip. I open the review tool, watch the source video
next to its generated strip, and either Accept it as-is, drag-select and relabel the
parts that are wrong and Save, then Accept, or Reject it outright if it's unusable and
send it back for reprocessing. Every Accepted video's data (per-frame table, segment
list, PNG) lands in a growing dataset directory that later becomes training data for
the drug-classification model.

### 3.2 Acceptance scenarios

1. **Given** a matched (trial row, video file) pair with a normally trackable fish,
   **when** the batch pipeline runs, **then** the system produces a per-frame state
   table, a segment list, and a PNG strip, all mutually consistent, and marks the
   video's review status `PROCESSED_AUTO`.
2. **Given** a `PROCESSED_AUTO` video, **when** the reviewer opens it in the review
   tool, **then** the strip renders with the same 5-color legend as the reference
   figures, and the source video plays back in sync with the strip's time axis.
3. **Given** a reviewer drag-selects a time range on the strip and reassigns its
   state, **when** they click Save/Submit, **then** the per-frame table, segment
   list, and PNG all update consistently and the status becomes `EDITED`; the
   original auto-generated version is not lost (recoverable via Reject).
4. **Given** a `PROCESSED_AUTO` or `EDITED` video, **when** the reviewer clicks
   Accept, **then** its three artifacts (plus a provenance record: reviewer,
   timestamp, source=`auto`|`manual`) are written into the accepted/gold dataset
   directory and the master accepted-index is updated.
5. **Given** any video in `PROCESSED_AUTO` or `EDITED` status, **when** the reviewer
   clicks Reject, **then** any manual edits are discarded, the status returns to a
   re-queued state, and the next batch run regenerates a fresh candidate from the
   current (possibly re-tuned) thresholds.
6. **Given** a video already `ACCEPTED`, **when** the batch pipeline is re-run,
   **then** that video is skipped (not reprocessed or overwritten) unless explicitly
   forced.
7. **Given** the trial workbook and the OneDrive video listing, **when** the catalog
   step runs, **then** it reports (without crashing) any trial with no matching
   video, any video with no matching trial, and the known duplicate `Subject #
   0320`, as a machine-readable exceptions report.

### 3.3 Edge cases

- Video duration does not match the expected `Agent Exposure Time (min)` (3 of 353
  trials are logged as 10-minute exposure, not 20 — their videos are expected to be
  ~600s, not ~1200s; the pipeline must read actual duration/fps from the file, never
  assume constants).
- Fish is undetectable for a stretch of frames (occluded by bubbles/reflection/glare,
  resting against the beaker wall out of the camera's easy contrast range). These
  frames must be labeled `Undetermined`, never silently forced into one of the five
  named behavioral states or into `Dead`.
- A subject is immobile/LORR for the entire session (e.g., high-dose fentanyl) —
  single-state strips are valid output, not an error.
- A subject dies partway through the session (e.g., opioid overdose) — the
  remaining time must render as white `Dead / No Action Recorded`, not `Undetermined`
  and not an extended `Freezing/Drift`/`Listing/LORR` bout, but the system must also
  flag this for reviewer confirmation, since "died and never moved again" and
  "extremely deeply sedated and the recording simply ended first" can look identical
  from video alone (FR-015a).
- A trial row's compound is a combination treatment (`FD-2-67 + methylone`) — its
  OneDrive folder also carries a status suffix (e.g.
  `FD-2-66 + methylone(DONE Compressed Only)`, confirmed real example) that is not
  part of the compound name; matching and grouping logic must strip that suffix and
  handle the `+` and any spacing variants.
- A trial has no video (video missing from OneDrive) or a video has no trial row —
  both must be reported, not silently dropped or silently guessed.
- A workbook row is an exact duplicate of another (confirmed real example: `Subject #
  0320`) — deduplicated automatically per Clarification C13, still logged for
  visibility.
- Corrupted or zero-byte video file.
- Measured fps deviates slightly from nominal 30 (observed ~29.83 fps average in the
  sample video) — must be read from container metadata.
- Reviewer clicks Accept on a strip that still contains `Undetermined` segments — the
  system must hard-block the Accept until every such segment is resolved to a real
  label (FR-015; not silently accept ambiguous data into the gold dataset). A strip
  containing `Dead` segments, by contrast, may be Accepted after one explicit
  confirmation click (FR-015a) — `Dead` is a valid label, `Undetermined` is not.

---

## 4. Non-Goals

Explicitly out of scope for this project (belongs to later phases of the broader
project, per `docs/zebrafish_drug_detection_scope.md`):

- Drug-compound / dose classification model (built separately: Part II).
- Anomaly / novelty detection.
- Natural-language reporting or any LLM-based narration (a research chat over the classifier exists: Part II, II.8).
- Assembling the multi-subject, multi-panel composite comparison figure (the "full"
  reference-style figure with all compounds/doses side by side) — this project only
  produces the single-subject building block.
- Processing Novel Tank Test (10-minute, rectangular-tank) footage — no such videos
  are available to this project.
- Live camera / real-time / streaming inference.
- Multi-user access, authentication, or networked deployment.
- Any modification of `src/fishbehavior/` — this is a new, independent top-level
  package that may later be integrated, but integration is not part of this PRD.

---

## 5. Data Model

### 5.1 Entities

| Entity | Description | Key fields |
|---|---|---|
| **Trial** | One cleaned row from `00_NTT_DataBase.xlsx` (of the 353 real trials) | `subject_id`, `sex`, `strain`, `age`, `compound`, `concentration_mM`, `date`, `agent_exposure_min`, `video_path`, `match_status` |
| **VideoAsset** | The located, validated video file for a trial | `path`, `duration_s`, `fps`, `frame_count`, `resolution` |
| **Track** | Per-frame raw localization of the fish | `frame_idx`, `t_sec`, `x`, `y`, `orientation_deg`, `depth_from_surface`, `detected` (bool) |
| **FeatureFrame** | Per-frame derived kinematics | `frame_idx`, `velocity`, `acceleration`, `angular_velocity`, `meander`, `smoothness`, `is_immobile` |
| **StateFrame** | Per-frame classification result | `frame_idx`, `t_sec`, `state` (one of the 5 behavioral states, `Dead`, or `Undetermined` — 7 total), `source` (`auto`/`manual`), `confidence` |
| **StateSegment** | Run-length-encoded contiguous block of one state | `start_s`, `end_s`, `state`, `source` |
| **StripImage** | Rendered PNG for one subject | `subject_id`, `image_path`, `palette_version` |
| **CalibrationProfile** | Versioned, frozen set of classification thresholds | `version`, `created_at`, `parameters`, `calibration_metric_summary` |
| **ReviewRecord** | Human review outcome for one video | `subject_id`, `status` (`NOT_PROCESSED`/`PROCESSED_AUTO`/`EDITED`/`ACCEPTED`/`REJECTED`), `reviewer`, `timestamp`, `edit_log` |
| **CatalogExceptionsReport** | Output of the matching step | `unmatched_trials`, `unmatched_videos`, `duplicate_trials`, `duration_mismatches` |

### 5.2 Reference color legend

Sampled directly from the provided reference images (not guessed); matches R's base
named colors, suggesting the originals were produced in R:

| State | Hex | Source |
|---|---|---|
| Controlled Swim | `#BEBEBE` | R `"gray"` |
| Erratic Movement | `#FF0000` | R `"red"` |
| Freezing/Drift | `#0000FF` | R `"blue"` |
| Listing/LORR | `#FFC0CB` | R `"pink"` |
| Surface Breach | `#00FF00` | R `"green"` |
| **Dead / No Action Recorded** *(confirmed, Clarification C16 — real reference-legend state, not invented)* | `#FFFFFF` | Owner-confirmed meaning of white in the original figures |
| Undetermined *(pipeline-internal only; not part of the original 6-state legend)* | a color clearly distinct from all six above (e.g. `#FF00FF` magenta, or a hatched pattern) — exact styling finalized during `rendering.py` implementation | — |

Two different things can both look like "no color band" in casual reading, so they
are kept deliberately distinct:

- **`Dead / No Action Recorded` (white)** is a real, calibrated ethogram outcome —
  the animal died and the remainder of the session has nothing to classify. It is
  detectable (with caveats — see §9.5.6) from the video itself and is one of the
  six labels a reviewer or the auto-classifier can assign.
- **`Undetermined` (magenta placeholder)** is *only* a bookkeeping marker meaning
  "the pipeline couldn't confidently classify this span" (occlusion, glare, a
  borderline feature value) — it is never a claim about the fish's biological
  state, and per FR-015 it must be resolved (by the reviewer, to one of the six real
  labels) before a video can be Accepted. It intentionally does **not** reuse white,
  so it can never be silently confused with a genuine death.

### 5.3 Trial schema (from `catalog.py`, one row per real trial)

| field | type | notes |
|---|---|---|
| `subject_id` | str, 4-digit zero-padded | normalized from workbook `Subject #:` |
| `sex` | `"M"` \| `"F"` | |
| `strain` | str | Casper / Wild-type (AB) / ABSL — affects tracking contrast, see [§10 Risks](#10-risks--mitigations) |
| `age` | float | |
| `compound` | str | raw `Compund:` value, incl. combo treatments |
| `concentration_mM` | str | kept as string; combo doses like `"0.03 + 0.01"` are not numeric |
| `date` | date | parsed from `YYMMDD` int |
| `agent_exposure_min` | float | expected video duration driver (20 for 350 rows, 10 for 3) |
| `video_path` | str \| null | resolved local path after matching; null if unmatched |
| `match_status` | enum | `matched` / `no_video` / `no_trial_row` / `duplicate_row` |

*(`ntt_time_min`, `TDM`/`Velocity`/`Time Spent` columns, `H2O (Before/After)` are
retained verbatim as passthrough metadata columns but are explicitly excluded from
every downstream computation — enforced by not passing them into feature extraction
or labeling code at all, not just by convention. See Clarification C2.)*

### 5.4 Per-video output schemas

**`frames.parquet`** (one row per frame):
`frame_idx:int32, t_sec:float32, x:float32, y:float32, orientation_deg:float32|null,
depth_from_surface:float32|null, detected:bool, velocity:float32,
acceleration:float32, angular_velocity:float32, meander:float32, is_immobile:bool,
state:category[7], source:category["auto","manual"], confidence:float32|null`
(the 7 categories: `Controlled Swim`, `Erratic Movement`, `Freezing/Drift`,
`Listing/LORR`, `Surface Breach`, `Dead`, `Undetermined`)

**`segments.csv`** (run-length encoded from `frames.parquet`):
`start_s, end_s, duration_s, state, source`

**`manifest.json`**:

```json
{
  "subject_id": "0068", "sex": "F", "compound": "FD-2-45", "concentration_mM": "0.03",
  "video_path": "...", "video_duration_s": 1202.7, "video_fps": 29.83,
  "pipeline_version": "0.1.0", "calibration_profile_version": "cal-2026-09-30",
  "processed_at": "...", "review_status": "ACCEPTED",
  "reviewer": "...", "reviewed_at": "...", "edited": true, "edit_count": 2
}
```

**`accepted_index.parquet`** (master index, one row per Accepted video): join of the
Trial fields + `manifest.json` summary fields + paths to that video's three accepted
artifacts. This is the single file the future Phase 2 classifier work reads.

---

## 6. Functional Requirements

- **FR-001**: System MUST ingest `00_NTT_DataBase.xlsx` (`Sheet1`), keep only rows
  with a non-blank `Compund:` (the 353 real trials), and exclude the always-empty
  `Body Tissue` column entirely.
- **FR-002**: System MUST treat the 8 movement/zone columns (`TDM`/`Velocity`/`Time
  Spent` × Full/Top/Bot) as **reference metadata only**; they MUST NOT feed into
  tracking, feature extraction, or state classification (see Clarification C2).
- **FR-003**: System MUST resolve each trial's video file using a normalized match
  key: compound folder name — with any trailing parenthetical status suffix (e.g.
  `(DONE Compressed Only)`, confirmed real-world example per Clarification C12)
  stripped before comparison — normalized further for case/whitespace/`+`, sex
  letter (case-insensitive), subject number (compared as integer, independent of
  zero-padding width).
- **FR-004**: System MUST produce a catalog exceptions report listing, without
  halting the run: trials with no matching video, videos with no matching trial,
  duplicate trial rows for the same subject number (auto-deduplicated per
  Clarification C13, still logged for visibility), and videos whose measured
  duration deviates from the trial's expected `Agent Exposure Time (min)` by more
  than a configurable tolerance.
- **FR-005**: System MUST track the fish across every frame of each matched video at
  the video's native (measured, not assumed) frame rate, producing per-frame
  position, an orientation/tilt estimate, and vertical position relative to the
  water surface, with an explicit `detected=false` flag for frames where tracking
  fails.
- **FR-006**: System MUST derive, per frame, at minimum: instantaneous velocity,
  acceleration, turning/angular velocity, a path-smoothness or meander measure, and
  an immobility indicator — sufficient to distinguish all five target states.
- **FR-007**: System MUST classify every frame into exactly one of {Controlled Swim,
  Erratic Movement, Freezing/Drift, Listing/LORR, Surface Breach, **Dead / No Action
  Recorded**, Undetermined} using a versioned, config-driven threshold/rule set (no
  hardcoded magic numbers in code). Classification is derived **only** from that
  video's own computer-vision tracking/feature output (§9.4.3) — never from the
  reference images or the Excel movement columns (FR-002).
- **FR-007a**: System MUST detect sustained terminal immobility (zero/near-zero
  velocity with no recovery through the end of the video, for longer than a
  calibrated minimum duration clearly exceeding a plausible Freezing/Drift or
  Listing/LORR bout) and classify it as `Dead / No Action Recorded`. Once a frame is
  classified `Dead`, every subsequent frame in that video MUST also be `Dead`
  (monotonic/terminal — death has no "recovery" state); the system MUST reject or
  flag any candidate classification that violates this as a data-quality error.
- **FR-008**: System MUST support a one-time calibration procedure that: (a)
  digitizes the two reference images into approximate group-level state-proportion
  targets (and per-subject-row timelines on a best-effort basis, where subject IDs
  are legible and matchable to a video), (b) runs the current auto-classifier over
  the corresponding compound/concentration groups, (c) reports an agreement/error
  metric per candidate threshold setting, and (d) freezes the chosen thresholds as a
  versioned `CalibrationProfile` used by all subsequent unattended runs.
- **FR-009**: System MUST render, per video, a single-subject strip PNG using the
  exact reference color legend (§5.2), with a time axis in seconds and a
  subject-ID label, visually consistent with the two reference figures' style.
- **FR-010**: System MUST export, per video, three mutually consistent artifacts:
  a per-frame state table, a run-length-encoded segment list, and the rendered PNG
  strip.
- **FR-011**: System MUST provide a local, single-user, browser-based review UI that
  displays, per video: the rendered strip, the source video with playback
  synchronized to the strip's time axis, and the current review status.
- **FR-012**: Reviewer MUST be able to click-drag a time range directly on the strip
  and reassign it to any of the seven labels (five behavioral states, `Dead`, or
  `Undetermined`), with frame-accurate boundaries; the change MUST be reflected
  live in the strip, segment list, and per-frame table before saving.
- **FR-013**: System MUST provide three distinct reviewer actions: **Accept**
  (commit current state as-is), **Save/Submit** (persist manual edits without
  accepting — status becomes `EDITED`), and **Reject** (discard current auto/manual
  state and re-queue the video for regeneration from scratch).
- **FR-014**: On Accept, system MUST write the video's three artifacts plus a
  provenance record (reviewer, timestamp, `auto` or `manual`, calibration profile
  version used) into a versioned accepted/gold dataset directory, and update a
  master accepted-index joining subject metadata for downstream consumption.
- **FR-015**: System MUST **block** Accept (hard block, no override) while a strip
  still contains any `Undetermined` segment — it is not a real ethogram label and
  must be resolved by the reviewer to one of the six real labels (five behavioral
  states or `Dead`) before that video's data can enter the gold dataset.
- **FR-015a**: System MUST **warn, but allow** Accept (soft warning, overridable
  with one explicit confirmation click) when a strip contains any `Dead` segment,
  since distinguishing genuine death from prolonged, unrecovered Freezing/Drift or
  Listing/LORR near the very end of a session is inherently uncertain from video
  alone (§9.5.6) — unlike `Undetermined`, `Dead` is a real, acceptable label; the
  warning exists only because it's a higher-stakes claim worth a second look, not
  because the label itself is invalid.
- **FR-016**: Batch pipeline runs MUST be idempotent and resumable: videos already
  `ACCEPTED` are skipped on re-run unless a `--force` style override is given; every
  video's status is one of `NOT_PROCESSED` / `PROCESSED_AUTO` / `EDITED` /
  `ACCEPTED` / `REJECTED` at all times.
- **FR-017**: System MUST record processing provenance (pipeline version,
  calibration profile version, timestamp) per video so that outputs from different
  threshold generations are distinguishable and reproducible.
- **FR-018**: System MUST NOT commit restricted research data (videos, workbook,
  reference thesis materials, generated outputs) into version control, consistent
  with the existing repository's data-handling policy.

## 7. Non-Functional Requirements

- **NFR-001 (Performance)**: The full 353-video batch run must complete in a bounded,
  predictable time on a single local workstation (target: comfortably within a few
  hours; concrete batching/parallelism strategy in [§9.5.9](#959-batch-orchestration--performance--resolves-the-batch-size-question)).
  No hard runtime SLA was set by the owner.
- **NFR-002 (Reproducibility)**: Given the same video, trial metadata, and
  calibration profile version, re-running the pipeline must produce identical
  per-frame classifications (deterministic tracking/classification, no unseeded
  randomness).
- **NFR-003 (Auditability)**: Every accepted record must be traceable to exactly
  which calibration profile and pipeline version produced it, and whether/how it was
  manually edited.
- **NFR-004 (Data hygiene)**: All restricted inputs and generated outputs live under
  gitignored local paths, following the existing repo convention.
- **NFR-005 (Usability)**: A reviewer must be able to review and accept a typical
  (mostly-correct) video in well under the video's own 20-minute runtime — the
  synced video/strip view and drag-select correction are the primary levers for
  this.

## 8. Success Criteria

- All 353 trial rows are either matched to a video and processed, or explicitly
  listed in the exceptions report with a reason.
- The auto-classifier's group-level state proportions (per compound/concentration)
  are in visually/statistically reasonable agreement with the two reference figures
  after calibration (exact agreement threshold to be defined during calibration —
  no ground truth exists to score against beyond these two images).
- A reviewer can take a freshly auto-processed video from open → corrected →
  Accepted using only the review UI, without touching code or raw data files.
- The accepted/gold dataset directory, once a meaningful fraction of the 353 videos
  are Accepted, is directly consumable (documented schema, no further transformation
  needed) as training data for the future Phase 2 classifier.

---

## 9. Technical Approach

### 9.1 Technical Context

| | |
|---|---|
| Language/version | Python 3.11 (matches `fishbehavior`'s `>=3.10` floor) |
| Primary dependencies | `opencv-python`, `numpy`, `pandas`, `pyarrow` (Parquet), `PyYAML`, `python-dotenv`, `matplotlib`, `fastapi`, `uvicorn`, `pydantic` |
| Optional/deferred dependency | `ultralytics` (YOLOv8n) — only if classical CV tracking proves insufficient on low-contrast frames (see §10 Risks) |
| Storage | Local filesystem only: Parquet (per-frame tables), CSV/JSON (segments, manifests), PNG (strips). No database in v1 — a `review_status.json` per video is the source of truth for review state. |
| Testing | `pytest`, synthetic fixtures only (no real trial data in the test suite — matches existing repo policy) |
| Target platform | Local Linux/Windows/macOS workstation; developed against the owner's RTX 4070 Ti (12GB VRAM) box |
| Project type | Offline batch CLI pipeline + a small local single-user web review tool |
| Performance goal | Full 353-video batch pass completes in well under a working day on the target workstation (see §9.5.9) |
| Constraints | No frame-level ground truth exists anywhere except two reference images; videos live on OneDrive, not locally, until synced; strains vary (translucent Casper vs. pigmented Wild-type/ABSL) which affects contrast-based tracking |
| Scale/scope | 353 videos, ~30fps, ~1200s (36k frames) each, ~2GB total; single subject per video |

### 9.2 Guiding principles

1. **No restricted data in git.** Videos, the workbook, reference thesis materials,
   and all generated outputs stay under gitignored local paths (`data/`, `outputs/`),
   mirroring the existing repo's `.gitignore`. Confirmed policy (Clarification C14 /
   resolved OD-3): reference materials are restricted, same as videos/workbook.
2. **Determinism over cleverness.** Given a fixed calibration profile version, the
   same video must always produce the same per-frame classification. No unseeded
   randomness anywhere in the tracking/classification path.
3. **Simplicity first.** Classical, interpretable computer vision (background
   subtraction + rule thresholds + light unsupervised clustering) is the default.
   A deep-learning detector is an escape hatch for specific failure modes, not the
   starting point — it adds GPU dependency, training data needs, and opacity that
   aren't justified yet for a plain-background, single-object, side-view video.
4. **Every requirement traces to an FR.** Each module below is annotated with the
   requirement(s) it satisfies.
5. **The human review step is a first-class deliverable, not an afterthought.**
   Since there is no ground truth, review UI correctness and reviewer efficiency are
   as important as pipeline accuracy — most effort should go toward making
   incorrect auto-classifications *cheap to fix*, not toward chasing a perfect
   classifier with no way to validate it.

### 9.3 Project structure

```text
fish-detection/
└── preprocessing_dataset_system/          # NEW top-level package (independent of src/fishbehavior/)
    ├── PRD.md                             # this document
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── .env.example
    ├── .gitignore                         # extends root .gitignore locally for this folder's own data/outputs
    ├── config/
    │   └── default_thresholds.yaml        # versioned, calibrated classification parameters
    ├── src/
    │   └── prepds/
    │       ├── __init__.py
    │       ├── config.py                  # Settings/.env loader, same pattern as fishbehavior.config, own PDS_* env vars
    │       ├── catalog.py                 # FR-001..004: workbook cleaning + video matching + exceptions report
    │       ├── video_io.py                # video open/probe (fps, duration, frame_count) — no assumed constants
    │       ├── roi.py                     # waterline / beaker-edge detection, background model
    │       ├── tracking.py                # FR-005: per-frame fish localization
    │       ├── features.py                # FR-006: per-frame kinematic feature derivation
    │       ├── calibration/
    │       │   ├── digitize_reference.py  # FR-008a: reference PNG -> approximate timelines/targets
    │       │   └── calibrate.py           # FR-008b-d: threshold search against digitized targets
    │       ├── labeling.py                # FR-007: per-frame state classification from features + thresholds
    │       ├── segments.py                # frame-level states -> run-length segments
    │       ├── rendering.py               # FR-009: strip PNG rendering (reference palette)
    │       ├── export.py                  # FR-010, FR-014: writes frames/segments/PNG + manifest
    │       ├── review_store.py            # FR-016, FR-017: status state machine, accepted-index maintenance
    │       ├── cli.py                     # subcommands: check-config, catalog, calibrate, run, export-index
    │       └── webapp/
    │           ├── server.py              # FastAPI app: serves video (range requests), REST for edits
    │           ├── static/
    │           │   ├── index.html
    │           │   ├── app.js             # video<->strip sync, drag-select, Accept/Save/Reject calls
    │           │   └── app.css
    │           └── schemas.py             # pydantic request/response models
    ├── data/                              # gitignored: local video mirror + workbook copy
    ├── outputs/                           # gitignored: per-video working artifacts (PROCESSED_AUTO/EDITED)
    ├── accepted/                          # gitignored: gold dataset (ACCEPTED only) + accepted_index.parquet
    └── tests/
        ├── fixtures/                      # synthetic tiny videos + synthetic workbook rows only
        ├── test_catalog.py
        ├── test_tracking.py
        ├── test_features.py
        ├── test_labeling.py
        ├── test_rendering.py
        ├── test_export.py
        └── test_review_store.py
```

This mirrors `src/fishbehavior`'s config/CLI conventions closely enough to make a
future merge straightforward, without creating a hard dependency between the two
packages now (per Clarification C10).

### 9.4 Architecture / pipeline flow

The pipeline has **two distinct flows that run at different times and for different
reasons**. Splitting them out explicitly here (the combined diagram in the previous
draft blurred this and caused legitimate confusion):

#### 9.4.1 One-time calibration flow — the only place the 2 reference PNGs are used

**Why do we need the two reference PNGs at all?** Because there is no frame-level
ground truth anywhere for any of the 353 trial videos — nobody has ever labeled,
frame by frame, what a real fish was doing. The two reference figures (30 µM and
100 µM panels from the defense slides) are the *only* place where a human-produced,
"this is what correct looks like" signal exists, even though it only covers 5 of 17
compound families and is itself just a picture, not underlying data. This flow's
entire job is to **extract an approximate target from those two images once**, use
it to tune the classifier's numeric thresholds, and then freeze those thresholds. It
does not touch, and is not repeated for, any of the 353 actual trial videos.

```text
two reference PNGs (restricted, local only)
        │
        ▼
calibration/digitize_reference.py   — read pixel colors along each labeled subject
        │                              row, nearest-match against the confirmed
        │                              palette (§5.2), produce approximate
        │                              group-level "% time in each state" targets
        ▼
reference_targets.json   (e.g. "DOB 30µM ≈ 40% Erratic, 35% Controlled Swim, ...")
        │
        ▼
calibration/calibrate.py   — run the (in-progress) tracking→features→labeling
        │                     stack over the real videos belonging to those same
        │                     5 compound/concentration groups, search threshold
        │                     values that make the pipeline's own output match
        │                     reference_targets.json as closely as possible
        ▼
config/default_thresholds.yaml  (versioned, e.g. "cal-2026-09-30")   — FROZEN
```

This runs **once** (or occasionally, if thresholds are deliberately re-tuned later —
each re-tune produces a new version, old ones are kept). Its only output that the
rest of the system depends on is the frozen threshold file.

#### 9.4.2 Per-video runtime flow — runs for all 353 videos, never touches the reference PNGs

This is the flow that actually produces each video's strip. It reads the *already
frozen* `config/default_thresholds.yaml` from §9.4.1 — it does not re-derive
thresholds and does not reference the two images at all:

```text
00_NTT_DataBase.xlsx ──┐
                       ├─▶ catalog.py ──▶ trials_catalog.parquet + exceptions_report.json
OneDrive video mirror ─┘        (FR-001..004)

trials_catalog.parquet ──▶ video_io.py + roi.py ──▶ tracking.py ──▶ features.py
        (per matched trial)     (probe, waterline)     (FR-005)        (FR-006)
                                                                          │
                                                                          ▼
                              config/default_thresholds.yaml ──▶ labeling.py (FR-007)
                              (frozen in §9.4.1, read-only here)          │
                                                                          ▼
                                                    segments.py ──▶ rendering.py (FR-009)
                                                          │               │
                                                          ▼               ▼
                                                       export.py writes frames.parquet,
                                                       segments.csv, strip.png, manifest.json
                                                       (status: PROCESSED_AUTO)   (FR-010)
                                                                          │
                                                                          ▼
                                              webapp (review UI) ◀── review_store.py
                                          Accept / Save-Submit / Reject   (FR-011..016)
                                                          │
                                                          ▼
                                        accepted/ (gold dataset) + accepted_index.parquet
                                                       (FR-014)
```

This flow runs independently once per video, 353 times, entirely driven by that
video's own computer-vision output (tracking + features) evaluated against the
frozen thresholds — never by re-reading the reference images.

#### 9.4.3 Worked example: tracking → diagram

To make §9.4.2 concrete — yes, this is exactly the expected behavior: **the CV
pipeline itself decides what the fish is doing, frame by frame, and that decision
is what becomes the colored diagram.** No step in this pipeline looks at the
reference images, the Excel sheet's movement columns, or anything else to decide a
given video's coloring — only that video's own tracked motion.

1. `tracking.py` outputs, for every frame `f`, a position `(x, y)` and an
   orientation. Say frames 900–2700 (t = 30.0s–90.0s at 30fps) show almost no
   change in `(x, y)` from one frame to the next.
2. `features.py` turns that into `velocity ≈ 0` for every frame in 900–2700.
3. `labeling.py` applies the frozen Freezing/Drift rule from §9.4.1
   (`velocity < v_freeze_threshold` sustained for `≥ min_bout_duration`): frames
   900–2700 all get `state = "Freezing/Drift"`.
4. `segments.py` collapses that run of identical per-frame states into one segment:
   `{start_s: 30.0, end_s: 90.0, state: "Freezing/Drift"}`.
5. `rendering.py` looks up `"Freezing/Drift"` in the fixed palette (§5.2) →
   `#0000FF` (blue) — and paints a solid blue band from t=30.0s to t=90.0s on that
   subject's strip.

So concretely: **"the fish froze from a to b" → `velocity≈0` for that span →
`Freezing/Drift` label for that span → a blue band from a to b in the output PNG.**
The same mechanism, with different feature rules, produces every other color: a
sustained abnormal body-tilt span becomes a pink `Listing/LORR` band, a brief
crossing of the near-surface depth threshold becomes a short green `Surface Breach`
mark, sustained terminal immobility through the end of the video becomes a white
`Dead / No Action Recorded` band (§9.5.6), and everything else (not claimed by any
rule, not low-confidence) is split between gray `Controlled Swim` and red `Erratic
Movement` by the smoothness/turning-variance clustering step.

### 9.5 Component notes & key design decisions

#### 9.5.1 Catalog & matching (`catalog.py`) — FR-001..004

- Normalize match key in two steps:
  1. Strip any trailing parenthetical status suffix from the folder name — confirmed
     real examples (Clarification C12): `FD-2-66 + methylone(DONE Compressed
     Only)` → `FD-2-66 + methylone`; `FD-2-67(DONE Compressed Only)` → `FD-2-67`;
     `Fentanyl(DONE Compressed Only)` → `Fentanyl`. Implemented as a general
     `re.sub(r"\s*\([^)]*\)\s*$", "", folder_name)` rather than a hardcoded string
     match, so other suffix variants (if any exist) are handled the same way.
  2. Normalize the result plus the workbook's `Compund:` value the same way:
     `(compound.strip().lower().replace(" ", ""), sex.upper(), int(subject_number))`.
     Combo-treatment names are matched by normalizing `+` and surrounding whitespace
     identically on both sides.
- Exact-duplicate workbook rows (same subject/sex/compound/date, confirmed real
  example: `Subject # 0320`) are **deduplicated automatically to one logical trial**
  per Clarification C13 (no manual confirmation step needed) — both original row
  indices are still recorded in the exceptions report for visibility/audit only.
- Output: `trials_catalog.parquet` (all 353, with `match_status`) and
  `exceptions_report.json` (human-readable summary of every mismatch category from
  FR-004 / §3.3 Edge Cases).

#### 9.5.2 Video access — resolves OD-4

Confirmed approach (Clarification C15): videos currently live on OneDrive; the
owner will sync/download the 353-video folder to local disk to support
development, rather than the pipeline talking to OneDrive live. Point
`PDS_VIDEO_DIR` at that local mirror, exactly like `fishbehavior`'s existing
`FISH_VIDEO_DIR` pattern. Reasoning (unchanged from the original recommendation,
now confirmed rather than proposed): a one-time ~2GB transfer is simpler and more
reliable than adding OAuth, throttling, and retry logic for a local, single-user
batch job with no live-data requirement. A live-sync mode remains a reasonable
future enhancement but is not required to satisfy any functional requirement in
this PRD.

#### 9.5.3 Tracking (`roi.py`, `tracking.py`) — FR-005

Starting approach (classical, GPU-free):

1. **Background model**: static camera, plain background ⇒ per-video background
   estimated via median/mode of a sparse frame sample (e.g., every 30th frame),
   refreshed if the beaker/lighting shifts mid-video (rare, but check for it).
2. **Waterline / ROI**: detect the beaker's water surface as a stable horizontal
   edge in the background frame (Hough line or simple row-wise gradient peak) once
   per video; store `y_waterline` used by `depth_from_surface` and Surface-Breach
   detection.
3. **Foreground extraction**: `cv2.absdiff` against the background model (or
   `BackgroundSubtractorKNN` for videos with subtler background drift) →
   threshold → largest connected component = fish.
4. **Orientation**: fit an ellipse (`cv2.fitEllipse`) to the fish contour; major-axis
   angle approximates body tilt, used for Listing/LORR.
5. **`detected=false`** frames are wherever no plausible-sized contour is found
   (bubble/glare/occlusion) — never guessed or interpolated silently through to a
   named state.

**Known risk**: this approach was validated only by eye against one Casper
(translucent) video. Wild-type (AB) and ABSL fish are pigmented, which changes
contrast against the background. Early implementation work must explicitly sample
≥1 video per strain before committing to final thresholds. If classical background
subtraction proves unreliable for pigmented strains, the escape hatch is a
lightweight detector (YOLOv8n or similar) fine-tuned on a small hand-labeled frame
set — deferred unless actually needed, per the "simplicity first" principle.

#### 9.5.4 Features (`features.py`) — FR-006

Per frame, from the raw track: velocity (`Δposition/Δt`), acceleration, angular
velocity (`Δorientation/Δt`), a rolling meander measure (path curvature normalized
by distance, matching the thesis's own "Meander" definition), and an immobility
flag (velocity below a floor threshold sustained over a short rolling window, to
avoid single-frame noise flipping the flag).

#### 9.5.5 Calibration (`calibration/`) — FR-008

1. `digitize_reference.py`: read the two reference PNGs; map pixel-x to seconds
   using the known axis range (0–1200s) and the panel's plotting-area bounds; for
   each subject row, sample color at ~1s intervals and nearest-match against the
   confirmed reference palette **including white** (§5.2 — colors were extracted
   directly from the provided images, not guessed; white pixels within a subject's
   row are `Dead / No Action Recorded` per Clarification C16, not "no data" to be
   skipped).
   Produces, per compound/concentration panel (VEH, MDMA 30/100µM, Methylone
   30/100µM, DOB 30/100µM, Fentanyl 30µM — 100µM Fentanyl panel wasn't in the second
   sample image's subject list, confirm at implementation time), a **group-level
   target**: % of total time in each state (including % `Dead`).
   Per-subject-row timelines are also extracted where the row's tiny ID label is
   legible (best-effort OCR, e.g. `pytesseract`), for a secondary, stricter
   per-subject check — but the **group-level aggregate is the primary calibration
   target**, since exact subject-to-video matching from a low-resolution label
   crop is not guaranteed.
2. `calibrate.py`: grid/coarse-random search over the threshold parameters in
   `config/default_thresholds.yaml` (velocity cutoffs, angular-velocity/meander
   cutoffs for the Controlled-vs-Erratic GMM, depth cutoffs for Surface
   Breach/Listing, minimum bout durations) minimizing total-variation distance
   between the pipeline's own aggregated state proportions (run over the 353
   videos' matching compound/concentration groups) and the digitized targets from
   step 1. The winning parameter set is frozen and versioned (`cal-YYYY-MM-DD`),
   never edited in place — a new calibration run produces a new version, old ones
   are kept for reproducibility (NFR-002/003).

#### 9.5.6 Labeling (`labeling.py`) — FR-007

Rule + light unsupervised approach, matching the broader project's own documented
plan in `docs/zebrafish_drug_detection_scope.md` §4.1:

- **Freezing/Drift**: sustained velocity below the calibrated floor for ≥ a minimum
  bout duration.
- **Surface Breach**: `depth_from_surface` crosses the near-surface threshold — a
  short, spike-like event (matches the sparse green marks in the reference images).
- **Listing/LORR**: sustained abnormal body orientation (large deviation from
  horizontal) for ≥ a minimum bout duration — expected to dominate in
  high-dose-opioid trials, consistent with the reference Fentanyl panels.
- **Controlled Swim vs. Erratic Movement**: for frames not already claimed by the
  three rules above, a 2-component Gaussian Mixture over
  `[velocity, angular_velocity_variance, meander]` separates smooth directional
  swimming from erratic, high-turning-variance movement.
- **Dead / No Action Recorded** (Clarification C16, FR-007a): sustained
  near-zero velocity with **no recovery for the remainder of the video**, once that
  sustained span exceeds a calibrated minimum that is deliberately set well above
  any plausible Freezing/Drift or Listing/LORR bout length. Because death is
  terminal, once a frame is labeled `Dead` every later frame in that video is
  forced to `Dead` too (a hard post-processing invariant in `segments.py`, not just
  a soft preference). Pharmacologically, `Dead` is expected to typically follow a
  `Listing/LORR` (pink) run in high-dose-opioid trials — LORR progressing to
  respiratory depression and death is a documented overdose pattern — which is a
  useful sanity check on calibration, but not a hard rule (death is detected from
  the immobility itself, not from having seen LORR first).
  **Important caveat**: whether a terminal immobile stretch is "definitely dead" or
  "definitely still alive but extremely sedated, and the video just ended first" is
  not fully resolvable from a low-resolution side-view video alone (no reliable
  gill-movement signal at this resolution). The rule above is therefore a
  best-effort heuristic, and FR-015a requires reviewer confirmation before any
  `Dead`-containing strip is Accepted — the classifier proposes, the reviewer
  decides.
- **Undetermined**: any frame with `detected=false`, or with feature values that
  don't confidently fall into any of the above (near cluster/threshold boundaries,
  below a confidence margin). Never used for a genuine death — that is always
  `Dead`, even if the classifier's confidence in the exact death *onset* frame is
  low.

#### 9.5.7 Rendering (`rendering.py`) — FR-009

`matplotlib` single-row `imshow`/`broken_barh` style strip: x-axis seconds (0 to
actual video duration, not a hardcoded 1200), one color band per segment using the
exact hex palette, subject ID label, legend matching the reference figures.

#### 9.5.8 Review UI (`webapp/`) — FR-011..016

**Recommendation: a small FastAPI backend + a single-page vanilla HTML/CSS/JS
frontend**, not Streamlit. Reasoning: the core UX requirement (native `<video>`
playback whose position stays visually synced with a click-draggable strip
timeline, bidirectionally) needs a real `timeupdate` event loop and direct canvas/DOM
control that Streamlit's rerun-per-interaction model makes awkward and laggy to get
right. A ~200-line vanilla JS page (`<video>` + `<canvas>` timeline + `fetch()` calls
to a few REST endpoints) is not more implementation effort than fighting Streamlit
for this specific interaction, and keeps the dependency footprint tiny (no Node
build step). FastAPI serves the video file with HTTP range-request support (needed
for scrubbing) and exposes:

- `GET /videos` — list with status
- `GET /videos/{id}` — manifest + segments + frame table summary
- `GET /videos/{id}/video` — range-enabled video stream
- `POST /videos/{id}/edits` — apply a `{start_s, end_s, new_state}` relabel (Save/Submit)
- `POST /videos/{id}/accept` — FR-014 (rejects if `Undetermined` present and
  `force != true`, per FR-015; warns but allows with `force == true` if `Dead` is
  present, per FR-015a)
- `POST /videos/{id}/reject` — FR-013, re-queues for the batch pipeline

#### 9.5.9 Batch orchestration & performance — resolves the batch-size question

The tracking/feature/labeling stages (§9.5.3–9.5.6) are classical CV: CPU- and
I/O-bound, not GPU-bound. Recommendation:

- **Video-level parallelism, not GPU batching, is the lever that matters.** Run
  `N_workers = max(1, os.cpu_count() - 2)` videos concurrently via
  `multiprocessing`/`concurrent.futures.ProcessPoolExecutor` (each video is fully
  independent). On a typical desktop paired with an RTX 4070 Ti (commonly 12–16
  logical cores), that's **10–14 workers**.
- **Rough throughput estimate** (to validate empirically once `tracking.py`
  exists, not a guarantee): plain background-subtraction + contour tracking on a
  304×240 frame typically runs in the low single-digit milliseconds per frame on one
  core → a ~36,000-frame, 1200s video finishes in roughly 1–3 minutes per core. At
  12 parallel workers, the full 353-video batch is estimated at **well under 2
  hours**, comfortably inside the "within a week" target with room for re-runs
  after calibration changes.
- **If/when** a supplementary GPU detector is added (§9.5.3 escape hatch): batch
  **128 frames per forward pass** for a small model like YOLOv8n at this resolution
  (12GB VRAM has generous headroom at 304×240 even at that batch size) and limit
  concurrent GPU-resident videos to **1–2 at a time**, letting CPU-side decode and
  classical-stage work for other videos continue in parallel — avoids VRAM
  contention without leaving the GPU idle between batches.
- The calibration search (§9.5.5) is a one-time, small-scale (few reference groups
  × parameter grid) job — seconds to low minutes, no special resourcing needed.

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No frame-level ground truth exists beyond 2 reference images | Cannot claim measured accuracy | Calibrate to group-level distribution match (§9.5.5); treat human review as the real accuracy gate (FR-011..016), not the classifier alone |
| Strain-dependent contrast (translucent Casper vs. pigmented Wild-type/ABSL) | Tracking may silently degrade on non-Casper trials (77 of 353) | Explicit early spot-check across strains before freezing thresholds; `detected=false` fallback is mandatory, never a silent guess |
| `Dead` vs. prolonged `Freezing/Drift`/`Listing/LORR` is inherently ambiguous from video alone near the end of a session (no gill-movement signal at this resolution) | An auto-classified `Dead` label could be wrong (fish was alive, just deeply sedated at recording end) | FR-015a requires explicit reviewer confirmation before Accept whenever `Dead` is present; monotonicity check (§9.5.6) catches internally inconsistent classifications as a sanity backstop |
| Reviewer fatigue over 353 videos | Slower path to a usable gold dataset | UI prioritizes speed-to-correct (drag-select, synced playback) over completeness of automated pre-labeling |
| Threshold overfitting to the 5 groups in the 2 reference images, generalizing poorly to the other 12 compound families with zero reference coverage | Silent miscalibration on most of the dataset | Documented as an explicit, accepted limitation in this PRD's Success Criteria (§8); review UI is the backstop for exactly this gap |

---

## 11. Future Integration Path (non-binding)

Once a meaningful fraction of `accepted/` exists, merging this package's concepts
back into `src/fishbehavior/`'s originally planned module layout
(`catalog.py`/`tracking.py`/`features.py`/`labeling.py`/`reference.py`/
`priors.py`/`calibrate.py`/`export.py`/`plots.py`) is straightforward, since the
module boundaries here were deliberately named to correspond 1:1. That merge is
explicitly **not** part of this PRD.

---

## 12. Task Breakdown

Format: `[ ] ID [P?] Description — file(s)`. `[P]` = safe to do in parallel with
other `[P]` tasks in the same phase (different files, no shared dependency). Tasks
without `[P]` are sequential within their phase, or depend on a prior phase. Every
task references the requirement(s) it satisfies where applicable.

### Phase 0 — Setup & Repo Scaffolding

- [ ] **T001** Create `fish-detection/preprocessing_dataset_system/` top-level folder
  with `pyproject.toml`, `requirements.txt`, `README.md`, `.env.example`,
  `.gitignore` (extends root: `data/`, `outputs/`, `accepted/`, `*.mp4`, `*.xlsx`,
  `*.pdf`)
- [ ] **T002** [P] Scaffold `src/prepds/__init__.py`, `src/prepds/__main__.py`
  (`python -m prepds <command>`)
- [ ] **T003** [P] `src/prepds/config.py` — `Settings`/`load_settings()`,
  `PDS_VIDEO_DIR`, `PDS_DB_PATH`, `PDS_REFERENCE_DIR`, `PDS_OUTPUT_DIR`,
  `PDS_ACCEPTED_DIR`, `PDS_WORKERS`, `PDS_CONFIG` env vars, same precedence rules
  as `fishbehavior.config` (env > `.env` > defaults)
- [ ] **T004** [P] `config/default_thresholds.yaml` — placeholder structure for
  tracking/feature/labeling parameters (values TBD until Phase 6 calibration)
- [ ] **T005** `src/prepds/cli.py` skeleton — `check-config` subcommand (mirrors
  `fishbehavior.cli.run_check_config`), argument parser with `--env-file`/`--config`
- [ ] **T006** [P] `tests/test_config.py` — env precedence, missing/invalid path
  errors (synthetic paths only)
- [ ] **T007** [P] `tests/test_cli.py` — `check-config` exit codes
- [ ] **T008** Verify `pytest` runs green with only Phase 0 code; commit as the
  initial scaffold PR

**Exit:** `python -m prepds check-config` runs; empty test suite passes.

### Phase 1 — Data Contracts & Synthetic Fixtures

- [ ] **T009** [P] Define `Trial`, `VideoAsset` dataclasses/schemas —
  `src/prepds/models.py` (fields per §5)
- [ ] **T010** [P] Define `FrameRow`, `StateSegment`, `ManifestRecord` schemas —
  `src/prepds/models.py`
- [ ] **T011** [P] Generate a synthetic ~5s, 304×240, 30fps test video (moving dot on
  plain background, a few direction/speed changes) — `tests/fixtures/make_synth_video.py`
  + committed small output `tests/fixtures/synth_tiny.mp4`
- [ ] **T012** [P] Generate a synthetic 10-row mini "workbook" (`.xlsx`) covering:
  a normal row, a combo-treatment row, a blank-template row, a duplicate-subject
  pair, an all-8-movement-columns-blank row — `tests/fixtures/make_synth_db.py`
- [ ] **T013** [P] Crop and save small test crops of the two real reference PNGs'
  legend swatches only (5 solid-color patches, no subject data) as
  `tests/fixtures/legend_swatches.png`, for palette-matching unit tests without
  depending on restricted source images
- [ ] **T014** `tests/test_models.py` — schema round-trip (serialize/deserialize) for
  all dataclasses in T009/T010

### Phase 2 — Catalog (`catalog.py`) — FR-001..004

- [ ] **T015** [P] `tests/test_catalog.py::test_excludes_blank_rows` (against T012 fixture)
- [ ] **T016** [P] `tests/test_catalog.py::test_excludes_body_tissue_column`
- [ ] **T017** [P] `tests/test_catalog.py::test_normalized_matching` (case/zero-pad/`+` variants)
- [ ] **T018** [P] `tests/test_catalog.py::test_duplicate_row_detected_and_flagged`
- [ ] **T019** [P] `tests/test_catalog.py::test_unmatched_trial_and_unmatched_video_reported`
- [ ] **T020** Implement `src/prepds/catalog.py::load_workbook()` — reads `Sheet1`,
  filters to 353 real rows, drops `Body Tissue`, parses `Date of EXP`
- [ ] **T021** Implement `src/prepds/catalog.py::match_videos()` — normalized key
  matching against `PDS_VIDEO_DIR`'s `{compound}/{sex}_{subject}.mp4` layout,
  including the folder-suffix stripping confirmed in Clarification C12 (e.g.
  `(DONE Compressed Only)`)
- [ ] **T022** Implement `src/prepds/catalog.py::build_exceptions_report()` +
  `cli.py` `catalog` subcommand writing `trials_catalog.parquet` +
  `exceptions_report.json`
- [ ] **T023** Run `catalog` subcommand against the **real** local video mirror once
  the owner's download/sync (Clarification C15) is available; confirm the
  suffix-stripping and duplicate-row dedup (Clarification C13) both behave
  correctly against the actual OneDrive folder names and the real `Subject # 0320`
  rows — this is a validation pass now, not an open resolution

**Exit:** `python -m prepds catalog` produces a complete, human-reviewed catalog +
exceptions report against the real 353-video set.

### Phase 3 — Video I/O & Scene Setup — supports FR-005

- [ ] **T024** [P] `tests/test_video_io.py::test_probe_reads_actual_fps_and_duration`
  (against synthetic video, asserts no hardcoded 30/1200 assumption)
- [ ] **T025** Implement `src/prepds/video_io.py::probe()` and `frames()` generator
  (reads real fps/frame_count/duration from container, not assumed constants)
- [ ] **T026** [P] `tests/test_roi.py::test_waterline_detection_on_synthetic_frame`
- [ ] **T027** [P] `tests/test_roi.py::test_background_model_from_sparse_sample`
- [ ] **T028** Implement `src/prepds/roi.py::estimate_background()` and
  `detect_waterline()`
- [ ] **T029** Spot-check background/waterline detection against **1 real video per
  strain** (Casper, Wild-type AB, ABSL) — informal validation, log findings/screenshots
  as `docs/strain_tracking_notes.md`, feeding Phase 4 risk mitigation
- [ ] **T030** Adjust `roi.py` thresholds if strain spot-check (T029) reveals
  pigmented-fish contrast failures, before proceeding to full tracking rollout

### Phase 4 — Tracking (`tracking.py`) — FR-005

- [ ] **T031** [P] `tests/test_tracking.py::test_tracks_synthetic_moving_dot`
  (position accuracy against known synthetic ground-truth trajectory)
- [ ] **T032** [P] `tests/test_tracking.py::test_detected_false_on_blank_frames`
- [ ] **T033** Implement `src/prepds/tracking.py::track_video()` — foreground
  extraction (`absdiff`/`BackgroundSubtractorKNN`), largest-contour selection,
  `cv2.fitEllipse` orientation, `depth_from_surface` from T028's waterline
- [ ] **T034** [P] Add per-frame `detected` confidence handling (contour size sanity
  bounds, reject implausible jumps) — same file
- [ ] **T035** Run `tracking.py` over 10–15 real videos spanning multiple
  compounds/strains; visually spot-check with an overlay-annotated debug video
  export (`--debug-overlay` CLI flag) before scaling to all 353

### Phase 5 — Features (`features.py`) — FR-006

- [ ] **T036** [P] `tests/test_features.py::test_velocity_matches_finite_difference`
- [ ] **T037** [P] `tests/test_features.py::test_immobility_flag_on_sustained_low_velocity`
- [ ] **T038** [P] `tests/test_features.py::test_meander_definition_matches_thesis_metric`
- [ ] **T039** Implement `src/prepds/features.py::derive_features()` — velocity,
  acceleration, angular velocity, meander, rolling immobility flag

### Phase 6 — Calibration — FR-008

- [ ] **T040** [P] `tests/test_digitize_reference.py::test_axis_pixel_to_seconds_mapping`
  (against T013's legend fixture + known panel geometry constants)
- [ ] **T041** [P] `tests/test_digitize_reference.py::test_color_nearest_match_to_palette`
  (must include white → `Dead`, not treated as "no data")
- [ ] **T042** Implement `src/prepds/calibration/digitize_reference.py` — reads the
  two (restricted, locally-stored) reference PNGs, produces per-panel group-level
  state-proportion targets; best-effort per-subject-row OCR (`pytesseract`) where
  legible
- [ ] **T043** Manually confirm digitized group targets look sane (spot check
  against a visual re-read of the reference images) — record as
  `calibration/reference_targets.json`
- [ ] **T044** [P] `tests/test_calibrate.py::test_search_reduces_distance_to_target`
  (synthetic toy case with a known-optimal threshold)
- [ ] **T045** Implement `src/prepds/calibration/calibrate.py` — parameter search
  minimizing total-variation distance between pipeline output (run over real videos
  in the matching compound/concentration groups) and T043's targets
- [ ] **T046** Run full calibration once tracking/features are validated across
  strains (depends on Phase 4/5 completion on real data); freeze result as
  `config/default_thresholds.yaml` version `cal-<date>` + record the achieved
  agreement metric in a calibration report doc

**Exit:** a versioned, frozen threshold set exists and is checked in (the YAML
only — not the reference images themselves, which are confirmed restricted per
Clarification C14).

### Phase 7 — Labeling (`labeling.py`, `segments.py`) — FR-007

- [ ] **T047** [P] `tests/test_labeling.py::test_freezing_rule_sustained_low_velocity`
- [ ] **T048** [P] `tests/test_labeling.py::test_surface_breach_rule_depth_spike`
- [ ] **T049** [P] `tests/test_labeling.py::test_listing_rule_sustained_orientation`
- [ ] **T050** [P] `tests/test_labeling.py::test_controlled_vs_erratic_gmm_separates_synthetic_clusters`
- [ ] **T051** [P] `tests/test_labeling.py::test_undetected_frames_are_undetermined_not_guessed`
- [ ] **T052** Implement `src/prepds/labeling.py::classify_frame()` /
  `classify_video()` using T046's frozen thresholds
- [ ] **T053** [P] `tests/test_segments.py::test_run_length_encoding_round_trips_to_frames`
- [ ] **T054** Implement `src/prepds/segments.py::frames_to_segments()` and
  `segments_to_frames()`

### Phase 8 — Rendering (`rendering.py`) — FR-009

- [ ] **T055** [P] `tests/test_rendering.py::test_strip_uses_exact_reference_palette`
  (pixel-sample the output PNG, assert hex matches `#BEBEBE/#FF0000/#0000FF/#FFC0CB/#00FF00`)
- [ ] **T056** [P] `tests/test_rendering.py::test_time_axis_matches_actual_video_duration`
  (not a hardcoded 1200s)
- [ ] **T057** Implement `src/prepds/rendering.py::render_strip()`

### Phase 9 — Export & Review State — FR-010, FR-014, FR-016, FR-017

- [ ] **T058** [P] `tests/test_export.py::test_writes_three_consistent_artifacts`
- [ ] **T059** [P] `tests/test_export.py::test_manifest_records_pipeline_and_calibration_version`
- [ ] **T060** Implement `src/prepds/export.py::export_video()` — writes
  `frames.parquet`, `segments.csv`, `strip.png`, `manifest.json` to `outputs/<compound>/<sex>_<subject>/`
- [ ] **T061** [P] `tests/test_review_store.py::test_status_transitions_valid_only`
  (`NOT_PROCESSED → PROCESSED_AUTO → {EDITED|ACCEPTED|REJECTED}`, etc.)
- [ ] **T062** [P] `tests/test_review_store.py::test_reject_clears_edits_and_requeues`
- [ ] **T063** [P] `tests/test_review_store.py::test_accept_blocks_on_undetermined_without_force`
  (FR-015)
- [ ] **T064** Implement `src/prepds/review_store.py` — status state machine,
  `accept()`, `save_edit()`, `reject()`, `accepted_index` read/append
- [ ] **T065** [P] `tests/test_review_store.py::test_batch_rerun_skips_accepted_videos`
  (FR-016 idempotency)

### Phase 10 — Review Web App Backend — FR-011..016

- [ ] **T066** [P] `src/prepds/webapp/schemas.py` — pydantic request/response models
  for edits, accept, reject
- [ ] **T067** Implement `GET /videos`, `GET /videos/{id}` — `src/prepds/webapp/server.py`
- [ ] **T068** Implement `GET /videos/{id}/video` with HTTP range-request support
  (required for smooth scrubbing) — same file
- [ ] **T069** Implement `POST /videos/{id}/edits` → `review_store.save_edit()`
- [ ] **T070** Implement `POST /videos/{id}/accept` → `review_store.accept()`
  (respects FR-015 confirmation-required behavior via a `force` flag)
- [ ] **T071** Implement `POST /videos/{id}/reject` → `review_store.reject()`
- [ ] **T072** [P] `tests/test_webapp_api.py` — exercise all endpoints against a
  small in-memory/synthetic dataset (`fastapi.testclient`)

### Phase 11 — Review Web App Frontend — FR-011, FR-012

- [ ] **T073** `static/index.html` — layout: video player, strip canvas, status
  badge, Accept/Save/Reject buttons
- [ ] **T074** `static/app.js` — `<video>` `timeupdate` → move playhead marker on
  strip canvas
- [ ] **T075** `static/app.js` — click-drag range selection on strip canvas →
  highlight + state picker (7 labels: 5 behavioral states, `Dead`, `Undetermined`)
- [ ] **T076** `static/app.js` — wire Accept/Save/Reject buttons to backend
  endpoints (T067–T071); **block** Accept with a visible warning when any
  `Undetermined` segment remains (FR-015, hard block, no override); **warn but
  allow** Accept with an explicit confirmation click when any `Dead` segment is
  present (FR-015a, soft warning, overridable)
- [ ] **T077** `static/app.css` — basic layout/legend styling matching reference
  palette
- [ ] **T078** Manual UX pass: review 10 real auto-processed videos end-to-end
  through the actual UI, fix rough edges before scaling to all 353

### Phase 12 — Batch Orchestration & CLI — resolves batch/perf recommendation

- [ ] **T079** [P] `tests/test_cli.py::test_run_command_dry_run_lists_pending_videos`
- [ ] **T080** Implement `cli.py` `run` subcommand: catalog → track → features →
  label → segments → render → export, parallelized across
  `max(1, cpu_count()-2)` worker processes (one video per worker), skipping
  `ACCEPTED` videos unless `--force`
- [ ] **T081** Implement `cli.py` `export-index` subcommand — rebuilds
  `accepted_index.parquet` from all `ACCEPTED` manifests
- [ ] **T082** Add `--workers N` override flag; document the throughput
  recommendation from §9.5.9 in `--help` text and `README.md`
- [ ] **T083** Time an actual full 353-video run once Phases 2–9 are validated on
  real data; record wall-clock time and worker count used in `README.md` (replaces
  the estimate in §9.5.9 with measured numbers)

### Phase 13 — Full Integration Pass

- [ ] **T084** Run the full pipeline (`catalog` → `run`) against all 353 real videos
  end-to-end; confirm `exceptions_report.json` accounts for every non-processed
  trial with a reason (FR-004)
- [ ] **T085** Review a stratified sample (≥1 per compound family, ≥1 per strain,
  the 3 short-duration MTA-5-62 trials, at least one heavily-LORR fentanyl trial)
  through the review UI; confirm Accept/Save/Reject all behave correctly on real
  data, not just synthetic fixtures
- [ ] **T086** Confirm `accepted_index.parquet` schema is directly usable (spot
  load with `pandas`/`pyarrow`, no further transformation needed) — validates the
  Success Criteria in §8

### Phase 14 — Documentation & Polish

- [ ] **T087** [P] `README.md` — setup, `.env` config, how to run calibration once,
  how to run the batch pipeline, how to launch the review UI
- [ ] **T088** [P] Document the calibration methodology and its known limitation
  (reference coverage: only 5 of 17 compound families, 2 of up to 4 concentration
  levels) prominently, matching the risk noted in §10
- [ ] **T089** [P] Update root `fish-detection/README.md` to mention the new
  top-level `preprocessing_dataset_system/` folder and its relationship to the
  planned `src/fishbehavior/` modules (non-binding integration note only)
- [ ] **T090** Final pass: resolve or explicitly re-confirm any remaining
  `[NEEDS CLARIFICATION: OD-*]` markers from §2.1; remove markers that are now
  resolved

### Dependencies (high level)

```text
Phase 0 → Phase 1 → Phase 2 (catalog, needs real video mirror + workbook)
Phase 2 → Phase 3 → Phase 4 → Phase 5
Phase 4/5 (validated on real data) → Phase 6 (calibration)
Phase 6 → Phase 7 (labeling needs frozen thresholds)
Phase 7 → Phase 8 → Phase 9
Phase 9 → Phase 10 → Phase 11 (webapp needs export + review_store first)
Phase 9 → Phase 12 (batch CLI needs export; independent of webapp)
Phase 10/11 + Phase 12 → Phase 13 (full integration needs both the batch pipeline
   and the review UI working)
Phase 13 → Phase 14
```

### Parallel execution examples

Within a phase, `[P]`-marked tasks touch different files and can be handed out
independently. For example, all of Phase 1's fixture tasks:

```text
T009, T010, T011, T012, T013  — all [P], no shared files, can run concurrently
```

Or Phase 7's rule-specific test tasks before the single implementation task:

```text
T047, T048, T049, T050, T051  — all [P] (different test functions/fixtures)
T052                          — sequential, depends on all five above being written first
```

---

## 13. Review & Acceptance Checklist

- [x] Requirements are testable and unambiguous where information was available.
- [x] Scope is bounded; explicit Non-Goals section (§4) prevents drift into
      later-phase work (classification, anomaly detection, reporting, live
      operation).
- [x] All assumptions made in place of unavailable ground truth are stated
      explicitly (§2) rather than silently baked into requirements.
- [x] Technical approach, architecture, data model, and task breakdown are unified
      in a single document with consistent cross-references.
- [x] Open items OD-1 through OD-4 resolved (see §2, Clarifications C12–C15).
- [x] `Dead / No Action Recorded` vs. `Undetermined` distinction (Clarification C16)
      threaded consistently through the data model, requirements, labeling design,
      review UI, and tasks.
- [ ] Reviewed and approved by project owner before implementation begins.

---

# Part II: Compound and Dose Classifier (`dcs`)

| | |
|---|---|
| **Status** | Built (U1-U20, 2026-10-02 to 2026-10-08) and tested on synthetic data; run once on the unreviewed real output. Not yet run on the lab GPU box; no result is a claim about drug identity |
| **Merged into this file** | 2026-10-08, from the former `classifier_PRD.md` v0.3 (now removed; it is in git history). Facts that changed during the build are applied here (the former plan's change list C1-C24; the decisions D-nnn are in [classifier_progress.md](classifier_progress.md)) |
| **Advisor / collaborator** | Dr. Ashish Kharel / Dr. Scott Hall |
| **Privacy rule** | This file and every file added to `docs/` use placeholders (`COMPOUND_A`, `F_0042`). Never write real compound names, subject IDs, dates, data files or local paths here. The README forbids committing restricted data; this PRD applies the same rule to names and dates voluntarily |

**Numbering.** Part II sections are written `II.n`. The code (`src/dcs`, `tests/dcs`), `config/default_training.yaml` and [instructions.md](instructions.md) cite them as "PRD §n.m" without the `II.` (for example "PRD §6.4" = **II.6.4**). Requirement ids (FR-1..FR-11, NFR-1..6, AC-1..11, EC-1..31, Q1..Q17, D-nnn) belong to Part II and do not collide with Part I's three-digit ids (FR-001...).

## II.1 Executive summary

### II.1.1 Problem
Part I turns each zebrafish video and workbook row into a human-reviewable, per-frame ethogram (the "gold" dataset). Nothing in Part I says which compound, and which dose, a fish received. Without that the project cannot estimate drug identity (scope doc §3, requirement 2).

### II.1.2 Solution
A second, separate package, `dcs`. `featurize` turns the gold dataset into one row per fish; `train` evaluates several classifiers on that table with validation that guards against date and camera effects, and saves a model; `predict` scores new fish; a **research chat** lets researchers ask, in English, about behavior per compound and dose, single fish and model results. Compound is predicted first (Stage 1), then dose within a compound (Stage 2, exploratory). Simple models are the baselines; a small PyTorch MLP is kept only if it matches or beats them.

### II.1.3 Model strategy (decided 2026-10-08)
| Layer | Now | Plan |
|---|---|---|
| **Reasoning / language** | A **lightweight, pre-trained, open-weights language model** run locally (any OpenAI-compatible server with tool calling: Ollama, llama.cpp, vLLM). It is **not trained or fine-tuned**; it chooses query tools and writes the answer, while every number comes from deterministic code (II.8). Chosen so data summaries stay on the lab machine | Pick and test the concrete model on the GB10 (first real run still to do). A larger or hosted model only if the data agreement allows (`chat.allow_remote`) |
| **Vision** | A **fine-tuned computer-vision tracker** (Part I, Phase 15: torchvision Keypoint R-CNN, box plus 5 keypoints, starting from COCO weights) and a small ResNet-18 crop classifier for Listing hints. The drug classifier itself never sees pixels; it reads features computed from the reviewed tracks | **A better vision model once more data exists**: more labeled frames for the detector, and, once enough reviewed state labels are Accepted, a clip-level behavior model trained on them (this is the route left open for Listing/LORR). Trigger and conditions: III.2 |
| **Drug classifier** | Tabular models (logistic regression, random forest, gradient boosting, small MLP) on one row per fish | Re-run on Accepted data; 1D-CNN on per-second sequences only after gate G4 (II.6.11) |

### II.1.4 Resources
- **Primary compute:** Dell Pro Max with GB10 (DGX OS 7, aarch64, 128 GB unified memory), reached over SSH; **fallback:** home PC (Windows 11) with an RTX 3060 Ti. The Part I detector was trained on an RTX 4070 Ti (r1: 437 s). The tabular classifier needs no GPU; only the MLP and the chat model benefit from one.
- **Data:** the gold dataset of Part I, or, until videos are Accepted, its unreviewed output (II.5.1).
- **People:** the project owner implements with Claude Code; advisor and collaborator review.

### II.1.5 Success metrics
No numeric accuracy target is promised: with at most a few hundred fish and a heavy date and camera confound (II.2.3) the honest outcome may be modest (Q9: tentatively no numeric target; revisit with the advisor at G2).

| # | Metric | Target | State |
|---|---|---|---|
| S1 | Every model scored next to the majority-class and a date-only baseline | Always reported | Met (report) |
| S2 | Stage 1 balanced accuracy under date-held-out validation, with its random-split twin | "Useful" only if it beats both baselines by more than the repeat spread and the permuted-label score (II.6.6) | **Not met on the real unreviewed set** (II.2.3) |
| S3 | Runs reproduce from a seed | Same seed, same data, same metrics | Met on CPU |
| S4 | A trained model copied off the box reloads with identical predictions | AC-9 | Met locally; across machines is the owner's step |
| S5 | No restricted data or real names in Git | `tests/dcs/test_dcs_privacy.py` | Met |

### II.1.6 Definition of Done
The acceptance checklist (II.9.3) passes, the first real-data report on the GB10 has been reviewed by the owner (G2, G4), and the progress file shows every P0 requirement verified.

## II.2 Problem definition

### II.2.1 Users
**Owner (researcher/student)** runs training and reads the report; **advisor and pharmacy collaborator** judge whether the evidence is credible; **researchers** use the chat.

### II.2.2 Context
**Upstream.** Part I tracks the fish in each trial video, derives per-frame features, labels every frame with one of seven states (the five ethogram states, `Dead`, internal `Undetermined`), lets a reviewer correct them, and commits only **Accepted** videos to the gold dataset (`accepted_index.parquet` is the single file the classifier reads). **`dcs` is a read-only consumer: it never imports or modifies `prepds`** and restates the file contract once in `src/dcs/schema.py` (D-013, D-030).

Per Accepted video: `frames.parquet` (`t_sec`, `x`, `y`, `orientation_deg`, `depth_from_surface`, `detected`, `velocity`, `acceleration`, `angular_velocity`, `meander`, `is_immobile`, `state`, `source`, `confidence`), `segments.csv`, `manifest.json`, plus trial fields in the index (`compound`, `concentration_mM`, `date`, `strain`, `sex`, `age`, `agent_exposure_min`).

**Status today.** No video is Accepted yet (Q14). By the owner's answer to Q17 (D-033), `dcs` reads the **unreviewed** Part I output (`training.gold_source: processed`, 328 fish on 41 dates, classical tracker) as **temporarily accepted**; every report says UNREVIEWED. `gold_source: accepted` is switched on when enough videos are Accepted. The older `fishbehavior` package is legacy; its sample workbook was used for structure checks only (Q15).

**What this phase is, and is not.** Part I's labeling uses a fine-tuned vision model and thresholds that have **not yet been validated with the pharmacy department**. Everything in Part II is therefore working, tested **training infrastructure**; no statement about drug identity is made from results on the current labels.

### II.2.3 What the data looks like
Facts below are aggregates only. `dcs audit` (FR-2) recomputes them on whatever set is loaded.

| Fact | Real unreviewed set | Why it matters |
|---|---|---|
| Fish; experiment dates | 328 fish, 41 dates (about 8 per date); 323 fish enter Stage 1 | Training-set size |
| Compounds | 15 classes after the class filter (323 fish; combination treatments are their own classes) | Stage 1 classes |
| Vehicle | 27 dates have 2+ vehicle fish, 13 have 1 (vehicle taken as the compound recorded on most dates) | Per-date control, but only 1-3 vehicle fish per date |
| Doses of one compound sharing a date | None found by the audit | Dose cannot be separated from date (II.6.5) |
| Camera framing | 3 zoom **epochs**, 18 framing setups, lining up with compounds | A model can recognize the camera instead of the drug (EC-31, D-050, D-061) |
| NTT columns (workbook) | `has_ntt` is 93 / 71 / 39 % by epoch | An epoch proxy, not a feature to trust |
| LORR, Dead, Surface Breach | Each shown by fewer than 10 fish (143-fish dry run: 0, 0 and 7) | Their features are dropped by the state-support rule (II.5.3) |
| Protocol columns | Vary with compound | Possible label proxy; excluded (II.5.3) |

**First real-data result (unreviewed output, 5 folds x 5 repeats).** Stage 1: scheme-A balanced accuracy about 0.17-0.23 for logistic regression, forest, boosting and MLP, against **0.47 for the date-only baseline** (majority 0.07): every behavior model is *not useful*. Vehicle versus drug (scheme A) AUROC is 0.68-0.70 for the behavior models against 0.555 for date-only: the one signal beyond the date. Per-epoch vehicle normalization helps a little (+0.03), per-date normalization leaks the date. NTT adds nothing. Stage 2 gives the same verdict and is date-confounded.

**Consequence.** A model can score well by recognizing the day, the camera or the labeling process rather than the drug. Every design choice in II.6 exists to measure and limit that.

### II.2.4 Cost of not solving
No quantitative evidence for drug-identity estimation, so later anomaly detection and reporting (scope doc §4.3-4.4) would have nothing to build on.

## II.3 Scope

### II.3.1 In scope
| ID | Item | Priority | State |
|---|---|---|---|
| FR-1 | `featurize`: gold dataset to one row per fish, plus schema file | P0 | Built (U5) |
| FR-2 | `audit`: data audit report, no training | P0 | Built (U8) |
| FR-3 | Stage 1 compound classifier (baselines + MLP) | P0 | Built (U10-U13) |
| FR-4 | Leakage-aware validation and diagnostics | P0 | Built (U7, U11) |
| FR-5 | Report: metrics, confusion matrices, per-compound results, ablations | P0 | Built (U12, U14) |
| FR-6 | Save and reload a model artifact | P0 | Built (U16) |
| FR-7 | Stage 2 dose within compound, exploratory | P1 | Built (U15) |
| FR-8 | `predict` for new rows or videos | P1 | Built (U17) |
| FR-9 | Vehicle-normalization ablation | P1 | Built (U14), reference rule changed (D-061) |
| FR-10 | 1D-CNN on per-second sequences | P2 | Not started; only after G4 |
| FR-11 | Image branch / video models | Out of scope now | See III.2 |
| FR-12 | Research chat: English questions answered from query tools by a local language model (terminal and loopback API) | P1 (owner addition 2026-10-08) | Built (U19-U20); first real model run pending |

### II.3.2 Out of scope
- Anomaly detection and report generation (scope doc §4.3-4.4); a **web frontend** (the chat API is ready for one; III.2); **live camera and real-time inference** (Phase 2). The classifier's unit is a whole recording, so it cannot run on a live stream without a windowed redesign.
- Detecting LORR or Dead from video (reviewers label them in Part I).
- Training or fine-tuning a language model (a few hundred fish cannot teach it facts; tool answers can be checked, D-071).
- Any change to `prepds` or its outputs.
- Hyperparameter search on test folds.

### II.3.3 MVP
FR-1 to FR-6 on one machine, producing one reviewed report and one loadable model.

### II.3.4 Learning goals
1. How much compound identity survives once the date effect is controlled?
2. Does the MLP beat logistic regression and trees at this table size?
3. Do the NTT columns add signal?

## II.4 Requirements

### II.4.1 User stories
- **US-1.** The owner runs one command and gets a report that states the baselines next to every model.
- **US-2.** The advisor sees whether a score depends on the date: date-held-out and random-split scores, their gap, and a date-only baseline.
- **US-3.** The owner copies a trained model off the box and reloads it elsewhere: the run folder is self-contained and a documented command reproduces the saved predictions.
- **US-4.** A researcher asks the chat "how does COMPOUND_A at 0.1 differ from vehicle?" and gets numbers, effect size, p and the caveats, with every number traceable to a tool call.

### II.4.2 Functional requirements (details)
- **FR-1** reads the index and each video's `segments.csv` and `frames.parquet` (read-only), computes the features of II.5.3, attaches labels (II.5.6) and writes `training_table.parquet` plus `training_table_schema.json` (every column: source, role feature/label/group/id/meta). `train` reads only these two files.
- **FR-2** prints and saves row counts per filter step, per-class and per-date tables, missing counts, outliers, the confound statistics of II.2.3, framing setups, the G1 verdict and which classes are evaluable.
- **FR-3** trains majority, date-only, logistic regression, random forest, histogram gradient boosting and the MLP under both validation schemes.
- **FR-4** implements II.6.4-II.6.6.
- **FR-5** writes `report.md`, `metrics.csv`, `predictions.csv` (out-of-fold), `decisions.csv`, `folds.csv`, confusion matrices.
- **FR-6** saves the best compound model refit on all eligible fish with everything needed to reload it (II.7.4).
- **FR-7** one small dose model set per compound with at least two eligible doses; every result carries a "date-confounded" caveat.
- **FR-8** `predict --model <run> --input <table>` writes class probabilities using only the saved preprocessing; `featurize --videos <dir>` builds a table for new fish without an index.
- **FR-9** raw versus vehicle-normalized features as a pair on identical classes and folds (II.6.8).
- **FR-12** see II.8.

### II.4.3 Non-functional requirements
- **NFR-1 Reproducibility.** One seed controls splits, initialization and shuffling; same seed and data give identical splits and, on CPU, identical metrics (GPU runs may differ in the last digits; the report says so). Versions and the git commit are in `run_info.json`.
- **NFR-2 Privacy.** No data, outputs, real names or paths in Git or `docs/`; outputs only under `DCS_OUTPUT_DIR`. `*.pt` and `*.joblib` are git-ignored.
- **NFR-3 Portability.** Python 3.11+; Linux (CI), Windows 11 and Linux aarch64 (GB10). Torch is an optional extra; everything except the MLP runs without it.
- **NFR-4 Performance.** The full comparison on a few hundred rows takes minutes on a GPU box and acceptable time on CPU (measured: 32.6 min for everything incl. ablations on a shared 10-core laptop; 291 s for the baselines).
- **NFR-5 Robustness.** Clear `ConfigError` messages for missing files, columns and too few rows.
- **NFR-6 Code style.** Own package under `src/`, one module per step, settings in `config/default_training.yaml`, env vars prefixed `DCS_`, no import of `prepds`.

## II.5 Data specification

### II.5.1 Input and the two-step flow
- **Source (read-only), two kinds, chosen by `training.gold_source`:** `accepted` reads `<DCS_ACCEPTED_DIR>` (the reviewed gold folder); `processed` (current default) reads the unreviewed Part I output (`DCS_PROCESSED_DIR`: `trials_catalog.parquet` plus `processed/<video_id>/`), allows `Undetermined` (measured per fish, never a behavior), drops `REJECTED`/`NOT_PROCESSED` fish with a reason, and marks every row unreviewed.
- **Step 1, `featurize`** runs where the data lives; **step 2, `train`** reads only the table. The original plan was to copy the table to the GB10; because copying onto the box may be blocked by least-privilege rules, the box instead **clones the repository, re-runs `prepds` on the raw videos and trains on that output** (D-057). Runbook: [instructions.md](instructions.md), "Training on the GB10".

### II.5.2 Eligible rows
1. Review status Accepted (source `accepted`), or any processed non-rejected fish (source `processed`, flagged).
2. **One tracker and one calibration profile per training set.** The gold folder carries no tracker field; it is inferred from the profile name (`-model` marker, configurable) and cross-checked against a `detections.parquet` where the processed folder exists. A mixed set is an error unless `--profile` selects one (D-003, D-034, EC-21). An upstream request to record the tracker, resolution and camera setup in the manifest is drafted: [upstream_request_prepds.md](upstream_request_prepds.md).
3. One row per subject (EC-10).
4. Low detected-frame share or odd video duration: **kept but flagged** (Q6, D-021).
5. Small classes are removed by the class filter (II.5.6).

### II.5.3 Features (computed by `featurize`)
**From `segments.csv`** (states Controlled Swim, Erratic Movement, Freezing/Drift, Listing/LORR, Surface Breach, Dead): per state, share of the known time, number of bouts, mean bout length, latency to first bout; transition counts. `Undetermined` is unknown time, never a state; shares are of known time; transitions are not bridged across it (D-039).

**From `frames.parquet`** (detected frames only, leaving out the frames where `prepds` writes 0.0 for lack of history, D-048): mean, median and coefficient of variation of speed, mean absolute acceleration and turning rate, mean meander, immobile share, detected share. **Depth features are an ablation group, off by default** (`use_depth`): depth is pixels from the frame top and encodes camera framing (D-015).

**Units caution.** Kinematics are in pixels and seconds, not body lengths, and the two trackers measure speed differently. They are comparable only within one tracker/profile and one camera epoch. The audit reports resolution, frame rate and framing.

**State-support rule.** A state's features are dropped when fewer than `training.min_state_fish` (10) fish show it (Q16, EC-23); on the real set this drops Listing/LORR, Dead and Surface Breach.

**Optional (ablation, on):** the eight NTT workbook columns plus `has_ntt`, read from the workbook (`DCS_DB_PATH`, D-004). A `-` cell means the fish never entered that half (TDM 0, velocity missing; D-037). The separate EthoVision exports in `data/new_reference/` are **not trusted and not used** (owner ruling 2026-10-08: human-derived numbers).

**Excluded:** `compound`, `concentration_mM` (the label); `subject_id`, `date`, paths, reviewer fields, edit counts, manual share, profile and pipeline versions (identifiers or labeling-process fields; `date` is used only to build groups); `agent_exposure_min`, `ntt_min`, `uv_min`, `h2o_*`, tissue columns (protocol fields that can leak the treatment); video duration, fps, resolution; `age`, `sex`, `strain` (off unless `--with-demographics`, an ablation). A test asserts none of them reaches the feature matrix (EC-12).

### II.5.4 Missing values
State never shown: bout count 0, mean bout 0, latency = recording length. No detected frames: fish dropped and counted. NTT missing: median of the training fold plus `has_ntt = 0`. Anything else missing: row dropped and counted in the audit.

### II.5.5 Transforms
Per fold: `log1p` on counts and durations, fill with the training-fold median, standardize with the fold's mean and spread; category features one-hot. Parameters are saved with the model (D-051, D-062).

### II.5.6 Labels and filtering
- **Compound label:** trimmed, whitespace collapsed, upper case; combinations are their own classes. **Dose label:** the written text without whitespace (`0.03 + 0.01` equals `0.03+0.01`); dose-stage label `<compound> @ <dose>`.
- **Class filter (Q4):** classes with fewer than `training.min_class_size` (6) fish are dropped and listed in the audit; exactly 6 is kept and flagged "very small". Stage 2 drops small dose classes first, then compounds left with fewer than 2 doses (this removes vehicle by rule).

## II.6 Modeling and evaluation

### II.6.1 Stages
Stage 1: compound (about 15 classes). Stage 2: dose within compound for compounds with at least two kept doses. Exploratory.

### II.6.2 Baselines (P0)
1. Majority class. 2. **Date-only**: the label mix of the test fish's date, else of the nearest training date (ties to the earlier), else of all training fish; it measures how much the day explains and is the leakage ceiling (D-011). 3. Logistic regression (L2, balanced). 4. Random forest (500 trees, balanced). 5. Histogram gradient boosting (balanced). Fixed settings, no tuning (D-053).

### II.6.3 PyTorch MLP
Two hidden layers (128, 64), ReLU, dropout 0.3, AdamW, class-weighted cross-entropy, up to 300 epochs with early stopping on an inner validation split, batch 32, 5 seeds averaged; values in `training.mlp`. Kept only if it matches or beats the best simple baseline. PyTorch rather than TensorFlow (owner, D-027).

### II.6.4 Validation design (the key decision)
Two schemes, always run together, 5 folds x 5 repeats with different seeds:

| Scheme | Purpose | How |
|---|---|---|
| **A. Date-held-out** (primary) | Does it generalize to an unseen day? | Stratified group K-fold with the experiment date as group |
| **B. Random stratified** (reference) | Leakage twin | Stratified K-fold |

A compound seen on one date only cannot be held out: it is pinned to the training side in scheme A, excluded from the A score and flagged "date-confounded" (EC-5). A scheme that is impossible (every class on one date) is skipped with a note. Folds are written to `folds.csv` and shared by every model. Metrics are computed on the pooled out-of-fold predictions of each repeat; the spread is the standard deviation across repeats (D-017). **The gap A versus B is reported as the "date effect".**

### II.6.5 Stage 2 and the dose-date confound
No compound has two doses that share a date, so scheme A is impossible and scheme B can be won by recognizing the date. Stage 2 is exploratory; no dose claim may rest on scheme B; the date-only baseline is shown; the only informative test is the vehicle-normalized variant.

### II.6.6 Diagnostics and decision rule
1. Date-only score (leakage ceiling). 2. **Within-date label permutation** (scheme B): if the real score is not clearly above the permuted one, the model learned only which date a label came from. 3. Repeat spread. 4. **"Useful"** = scheme-A balanced accuracy beats both the majority and the date-only baselines by more than the larger spread and exceeds the permuted result. 5. Vehicle versus drug reported separately (AUROC, balanced accuracy).

### II.6.7 Metrics
Balanced accuracy and macro-F1 (primary); per-class precision, recall, F1; confusion matrix; log-loss; top-3 accuracy. Accuracy alone is never reported.

### II.6.8 Ablations (same folds)
NTT on/off; demographics on/off; **vehicle-normalized features** (FR-9). FR-9 is a pair on the non-vehicle classes (vehicle fish are the reference only, so their labels are not used to normalize their own test rows, D-008). Reference rule (D-061): the fish's own date when it has 2+ vehicle fish, else its framing setup, else all vehicle fish; or, with `training.camera_epochs` set from the audit, the camera epoch. On the real set the per-date reference leaks the date (1-3 vehicle fish per date is a date fingerprint); the epoch reference helps slightly. NTT is kept only if scheme-A balanced accuracy improves by more than the spread; it did not.

### II.6.9 Final model
The best compound model by scheme-A balanced accuracy (references only if nothing else ran) is refit on all compound-stage fish and saved whatever its verdict; the verdict is stored with it. Dose models are not saved.

### II.6.10 Pre-trained models
No pre-trained model fits tabular behavior features, so the drug classifier uses none. Pre-trained models are used where they fit: the **language model of the chat** (II.8), the **COCO/ImageNet backbones** of the Part I detector and crop classifier. Time-series foundation models are a possible later experiment and would need an offline download plan for the Linux box.

### II.6.11 Optional 1D-CNN (FR-10, only after G4)
Input: each video's frames resampled to 1 s (about 1200 steps x 8 channels, undetected seconds masked, shorter recordings padded with a mask, EC-30); small dilated 1D-CNN with masked global pooling; same folds and metrics plus a time-shuffled control; added only if it beats the best tabular model on scheme A. Built by `featurize --sequences` (not implemented yet).

## II.7 Technical specification

### II.7.1 Code layout (as built)
`src/dcs/`, a second package in the same repository and `pyproject.toml`: `schema`, `config`, `config_rules`, `gold`/`gold_checks`/`gold_rules` (readers), `workbook`, `featurize`, `trainset`, `folds`, `audit`, `preprocess`, `models`, `mlp`, `evaluate`, `ablations`, `stage2`, `report`, `train`, `artifact`, `chat_tools`, `chat`, `chat_server`, `synthetic*` (synthetic gold data), `cli`. One step per module, under the harness limit of 400 lines per file (D-009). Defaults: `config/default_training.yaml`. Tests: `tests/dcs/` with synthetic data only (975 tests in about 54 s). Per-module documentation: [instructions.md](instructions.md), part "Classifier (`dcs`)".

### II.7.2 CLI
`check-config`, `synth`, `featurize [--videos] [--profile]`, `audit`, `train [--stage --models --seed --repeats --device --no-ablations]`, `predict`, `ask`, `serve-chat`. Full table: [instructions.md](instructions.md), "Commands".

### II.7.3 Configuration
`training:` (seed, folds, repeats, `min_class_size`, `min_state_fish`, `vehicle_compound`, `camera_epochs`, `use_ntt`/`use_depth`/`use_demographics`, `gold_source`, flags, and `mlp:`) and `chat:` (`model`, `temperature`, `max_tool_rounds`, `timeout_s`, `allow_remote`); the chat server's address is machine-specific and comes only from `DCS_CHAT_BASE_URL` in `.env` or the environment (D-075). Values read from the real data (vehicle name, camera epochs, duration range) go in the owner's override file, never in the defaults. Unknown keys and wrong types are a `ConfigError` (D-023).

### II.7.4 Outputs and the model artifact
```
<DCS_OUTPUT_DIR>/training/<run_id>/
    run_info.json  config_used.yaml  folds.csv  metrics.csv  predictions.csv  decisions.csv
    report.md  confusion_<stage>.png
    model/   sklearn.joblib or mlp.pt, preprocess.json, classes.json, model_info.json, reference_predictions.csv
```
`run_id` = UTC timestamp plus short git hash. `reference_predictions.csv` (final model on its own training fish) is what `predict` is compared against (AC-9); `predictions.csv` is out-of-fold and cannot be reproduced by a refit (D-006).

### II.7.5 Dependencies
No new core dependency; the extra `train = ["torch>=2.4"]`. CI (Linux, Python 3.11) installs `.[dev]` only and runs `tests/dcs` in the existing backend job; torch tests skip without torch (Q8, D-002).

## II.8 Research chat (FR-12)

**What it is.** A local open-weights language model with tool calling reads a question, calls deterministic **query tools** for every number, and writes the answer in plain English. Nothing is trained (D-071, D-073).

**Pieces.** `chat_tools.py`: 11 read-only tools (`list_compounds`, `find_features`, `compare_to_vehicle`, `top_differences`, `feature_by_compound`, `fish_profile`, `fish_timeline`, `model_results`, `class_scores`, `ablation_results`, `audit_facts`). `chat.py`: the engine (OpenAI chat-completions protocol over the standard library, so Ollama, llama.cpp, vLLM and most hosts work by changing `DCS_CHAT_BASE_URL`) and the system prompt. `dcs ask` is the terminal client; `dcs serve-chat` is a loopback JSON API (`/api/health`, `/api/tools`, `/api/ask`, `/api/tool`) for a later web page.

**Rules the prompt enforces:** numbers only from tools; look up a measure's name before comparing; give n, effect size (Hedges' g) and p or q, and say how strong the evidence is; repeat the tools' caveats (UNREVIEWED data, different dates and cameras, pixel measures depend on the camera); scheme A is the honest score, date-only is the baseline.

**Controls.** Compound-versus-vehicle uses same-date vehicle fish (fewer than 3: all vehicle fish, with a caveat). A `DCS_CHAT_BASE_URL` that is not this machine is refused unless `chat.allow_remote` is true (data summaries would leave the box); an API key goes only in `DCS_CHAT_API_KEY` in `.env` or the environment (masked as `***` when settings are printed). The API has no authentication and listens on loopback only, like the review app, with the same guard: a non-loopback `Host` (DNS rebinding) and a foreign `Origin` on POST are refused. `/api/ask` accepts only `user`/`assistant` history turns and caps the question and history size (D-075).

**State.** Tested against a stub server speaking the protocol; no language model is installed on the development laptop. The first real model run, and the choice of model, happen on the GB10.

## II.9 Validation of the data and quality assurance

### II.9.1 How the data is validated (layers)
| Layer | Question | How | State |
|---|---|---|---|
| 1. File contract | Do the files have the expected columns, types, states? | Gold readers: missing or unreadable files drop the fish and are listed (EC-1); a file that breaks the contract stops the run (EC-22, EC-25); one tracker and profile (EC-21); one row per subject (EC-10) | Built, tested |
| 2. Data audit (gate G1) | Is the set big and balanced enough to train on, and what confounds it? | `dcs audit`: counts per compound and date, stop rule (3+ compounds besides vehicle with 6+ Accepted fish on 2+ dates), vehicle coverage, dose-date confound, framing setups (EC-31), fps, tracker check, flags | Built; real run: WAIT on Accepted fish, GO counting unreviewed |
| 3. Label validity | Are the per-frame states right? | Part I: human review and the Accept rules; the reviewed sample (T085) is **still to do**; pharmacy-department validation of the labeling is **not done**; EthoVision numbers are not trusted (owner ruling) | **Open (the main gap)** |
| 4. Model validity | Does the score reflect drug, not day or camera? | Scheme A versus B, date-only, within-date permutation, repeat spread, epoch-normalized ablation, decision rule II.6.6 | Built; verdict on real unreviewed data: not useful |
| 5. Reproducibility | Same seed, same result; saved model reloads? | Seeded splits, `run_info.json`, `reference_predictions.csv`, fresh-process reload test (AC-9) | Built; cross-machine step is the owner's (T5.7) |
| 6. Privacy | Is anything restricted in Git? | `tests/dcs/test_dcs_privacy.py` | Built |

**What is still missing for an independent test.** Everything above is retrospective. The strongest check is prospective: record a small number of new fish of every compound (and vehicle) on the same new dates, with one camera setup, do not look at them while tuning, and score them once with the saved model. That also lifts the dose-date confound (all doses of a compound share dates). It needs the owner's recording plan; see III.1.

### II.9.2 Test approach
Synthetic tables and synthetic gold data only; test first for every pure function. `pytest tests/dcs -q` for the fast loop, `pytest tests/ -q` before a pull request. Every implemented edge case has a test (map: the Tests table in [instructions.md](instructions.md)); EC-30 belongs to the not-started 1D-CNN. Edge cases added during the build: EC-27 index row without `date`; EC-28 workbook row missing, duplicated or disagreeing; EC-29 tracker evidence disagrees; EC-30 sequences of different lengths; EC-31 camera framing differs between dates.

### II.9.3 Acceptance checklist
- [x] AC-1 `audit` reports the II.2.3 statistics (synthetic and real unreviewed set)
- [x] AC-2 features contain none of the excluded columns (EC-12)
- [x] AC-3 splits obey II.6.4 (EC-5, EC-6, EC-13)
- [x] AC-4 all baselines and the MLP run under both schemes on identical folds
- [x] AC-5 the report shows date-only, majority and permuted baselines next to every result
- [x] AC-6 NTT, demographics and vehicle-normalization ablations reported
- [x] AC-7 same seed gives the same result (EC-14)
- [x] AC-8 no restricted data, names or paths in Git
- [ ] AC-9 a model copied off the GB10 reloads elsewhere and reproduces its saved predictions (verified locally in a fresh process; across machines is the owner's step)
- [ ] AC-10 the runbook was followed from scratch by the owner (owner)
- [x] AC-11 progress file and decisions up to date

## II.10 Phases, gates and status
| Phase | Work | Status |
|---|---|---|
| 0 Environment | GB10 and Windows checks, hardware record | Owner (T0.2-T0.4) |
| 0b Upstream readiness | Run `prepds`, review flagged videos, export the index | Owner; no video Accepted yet |
| **G1** | Owner reads the audit; stop rule | Audit built; real run WAIT on Accepted fish |
| 1 Featurize, training set, folds | U1-U8 | Done |
| 2 Baselines | U9-U12 | Done |
| **G2** | Owner reviews a baseline report | Pending (the GB10 run) |
| 3 MLP and ablations | U13-U14 | Done |
| 4 Stage 2 | U15 | Done |
| 5 Save/load, predict, privacy | U16-U18 | Done; T5.7 owner |
| Research chat | U19-U20 | Done |
| **G4** | First real-data review with owner and advisor: stop, improve, or add FR-10 | Blocked on real data |
| 6 Optional 1D-CNN | FR-10 | Not started |
| 7 Wrap-up | Docs reflect what was built | Done (this merge) |

## II.11 Risks and mitigations
| Risk | Prob. | Impact | Mitigation |
|---|---|---|---|
| Model learns the day or the camera, not the drug | **Realized** | High | Scheme A, date-only baseline, permutation, epoch-normalized ablation; new recordings balanced across compounds per date and camera (III.1) |
| Dose cannot be validated (dose = date) | Certain | High | Stage 2 exploratory; no dose claim from scheme B |
| Labels unreviewed and not validated with the pharmacy department | Certain now | High | UNREVIEWED in every report; switch to Accepted data; G1 stop rule |
| Small classes give noisy per-compound results | High | Medium | Class filter, repeats, spread, "very small" flags |
| Protocol columns leak the treatment | Medium | High | Excluded by default |
| Label quality differs by compound (7 calibrated groups, rest extrapolated) | Medium | High | Manual-frame share per compound in the audit |
| Mixing trackers or profiles | Medium | High | One tracker and profile per set (EC-21) |
| Pixel features not comparable across setups | Medium | Medium | Audit resolution, fps and framing; upstream request for the camera setup |
| `prepds` output format changes | Low | Medium | Schema checks (EC-25); one restated contract file |
| ARM64/Blackwell torch install problems | Medium | Medium | Smoke test; fallback to the 3060 Ti; baselines unaffected |
| Restricted data leaves approved storage | Low | High | Data stays on owner machines; chat refuses a non-local server unless allowed |
| The chat model invents numbers or picks wrong tools | Medium | Medium | Numbers only from tools, errors list valid choices, tool calls shown (`--show-tools`); model chosen after testing on the GB10 |
| Scope creep into video models, frontend or live use | Medium | Medium | Gates G2/G4; III.2 roadmap |

## II.12 Questions and decisions (owner answers, 2026-10-02 to 2026-10-08)
Q1 tabular first, 1D-CNN second: yes. Q2 UToledo OneDrive approved. Q3 home PC Windows 11. Q4 drop every class lacking data (threshold 6). Q5 Stage 2 exploratory only. Q6 odd duration or low tracked share: keep and flag. Q7 no external LLM tools called by the agent. Q8 no torch in CI. Q9 tentatively no numeric target (revisit at G2). Q10 no deadline for this project (scope doc §9 targets the end of the current year for Phase 1). Q11 harness governance stays on. Q12 code in `src/dcs`. Q13 classical tracker output is the training source. Q14 none Accepted yet. Q15 earlier sample for structure checks only. Q16 `min_state_fish` 10, configurable. Q17 until videos are Accepted, read the unreviewed output (D-033). C8: PyTorch (D-027). Every other decision is D-001..D-074 in [classifier_progress.md](classifier_progress.md).

## II.13 Glossary and runbook
- **NTT:** Novel Tank Test, the workbook's per-trial movement summary (a different session from the video). **Scheme A / B:** date-held-out versus random validation. **Date effect:** score of B minus score of A. **Vehicle:** control fish exposed to the carrier only. **Baseline:** a simple model a fancier one must beat. **Camera epoch:** a period with one zoom/framing; the real set has three. **Temporarily accepted:** unreviewed Part I output used as training data until reviewers Accept videos.
- **Runbook** (GB10 over SSH, retrieving and reloading a model, home-PC fallback): [instructions.md](instructions.md), "Training on the GB10" and "Predicting new fish". It replaces the former Appendix B.

---

# Part III: Watch list and roadmap

Everything deferred, accepted as a known limit, or waiting on someone, collected from the progress logs and the owner's rulings. Nothing here is silently dropped; each item says what would close it.

## III.1 Watch list

### Classifier and chat
| # | Item | Waiting on / closes when |
|---|---|---|
| W1 | **First real run on the GB10**: hardware record (T0.2-T0.4), `pytest tests/dcs`, `dcs audit`, `dcs train`; compare with the laptop results in II.2.3 (a difference means the box's preprocessing differs); cross-machine reload (T5.7, AC-9) and runbook walkthrough (AC-10) | Owner; gates G2, G4 |
| W2 | **No video is Accepted.** Switch `gold_source: accepted` and re-run audit and train when enough are; re-read the G1 stop rule | Part I review (T085) |
| W3 | **Real result is "not useful"**: behavior models do not beat the date-only baseline in scheme A; only vehicle-vs-drug carries signal (AUROC 0.68-0.70). Do not state drug identity from these results | New data (W5) and Accepted labels |
| W4 | **Camera epochs**: set `training.camera_epochs` from the audit's framing table in the override file; file the upstream request so `prepds` records tracker, resolution and `camera_setup` | Owner ([upstream_request_prepds.md](upstream_request_prepds.md)) |
| W5 | **Dose is confounded with date.** Next recordings should put all doses of a compound, plus vehicle, on the same dates and one camera setup (also the prospective test of II.9.1) | Owner's recording plan |
| W6 | Features of Listing/LORR, Dead and Surface Breach are dropped (fewer than 10 fish show them); revisit when reviewers label them | Reviewed labels |
| W7 | `hist_gb` keeps scikit-learn's `min_samples_leaf = 20` and cannot split a training fold under 40 fish (D-053); real folds hold 180+ | Only matters on small sets |
| W8 | **Chat**: choose and test the language model on the GB10 (tool-calling quality, speed; a 7-30B model is slow on CPU); the API has no authentication (loopback only); hosting decided later by the lab; a remote server needs `allow_remote` and approval under the data agreement | Owner |
| W9 | The pre-trained language model stays untrained; reconsider only if tool-use quality is poor and a larger or hosted model is allowed | After W8 |
| W10 | 1D-CNN (FR-10) and `featurize --sequences` | Only after G4 |
| W11 | Open questions: Q9 numeric target (advisor, at G2); Q10 deadline | Advisor |
| W12 | Harness database lacks the `test_runs` migration, so RED/GREEN were observed in the terminal for U4-U20 (D-036..D-074) | Owner (harness setup) |
| W13 | `uv.lock` not refreshed for the `train` extra (D-028); the `dcs` suite takes about 54 s of a 60 s budget | Next dependency change |
| W15 | **No `/api/predict`.** Class probabilities exist only as a CSV from `dcs predict`; a Predict screen needs a loopback endpoint that wraps `load_model(...).predict` and returns verdict, reason and the UNREVIEWED flag with the probabilities (request/response sketched in the pipeline-map page). No frontend exists yet apart from the `prepds review` app; frontend work starts from the chat API (II.8) | Frontend work (owner) |
| W14 | EthoVision exports (`data/new_reference/`) are not trusted and unused; do not propose reference-based auto-accept or NTT correction again | Owner ruling 2026-10-08 |

### Preprocessing pipeline and vision (Part I)
| # | Item | Waiting on / closes when |
|---|---|---|
| P1 | **Human review of a stratified sample (T085)** is waiting on the owner; the output has no frame-level ground truth, so measured accuracy cannot be claimed | Owner |
| P2 | **Listing/LORR has no automatic rule.** Tilt rule AUC 0.75; ResNet-18 crop classifier AUC about 0.82 with precision about 0.3 (about 19% prevalence in the held-out set); shipped as review hints only (median 25 hints per video, thresholds untuned). LORR is about 0% of labeled time; the fish is slow with a horizontal axis from the side. Do not retry a hand rule without new evidence; leave it to a clip-based classifier trained on reference LORR seconds (III.2) | More labeled Listing/LORR |
| P3 | **Dead** rarely fires (tracking dropouts break the terminal run) and always needs reviewer confirmation | Review |
| P4 | **Surface Breach** cannot fire with the frame-top rule (0.0% in both runs, structural); an eye/head-based rule over-fires untuned and the reference has about 0% to calibrate against. Decide the rule first | Owner decision |
| P5 | **Detector (m3)**: no-fish frames give detections on 6 of 12 held-out frames (reflections); recall@0.5 about 0.90-0.92 at the production setting (fp16, min_size 400), not 0.95; tail-tip error about 42 px; reflection swaps at stride 5 and fp16 accuracy not quantified; held-out frames were used for small choices; training is not bit-reproducible and `run.json` lacks the code SHA and package versions | More labels (III.2) |
| P6 | Calibration: MDMA 30 and methylone 100 stay poorly fit; the r4 fit is weakly identified (minimum freeze bout at the search bound); gains are fit-to-target on 54 videos, not validation | More reference groups |
| P7 | Body length is orientation-biased (a fish facing the camera looks short, about 14% median); kept on purpose because changing it rescales every body-length threshold | Only with re-tuning on real data |
| P8 | Strain-dependent contrast (translucent Casper versus pigmented fish, 77 of 353 trials): spot-check tracking quality per strain | Review |
| P9 | No cross-process locking: run one `prepds run` at a time, and not alongside accepts in the review app (`export-index` repairs the index) | Only if parallel use is needed |
| P10 | Review app gaps: click-to-review on the ethogram, waterline/ROI editor (deliberately not built: no rule reads a waterline), mobile layout, UX pass covered 3 videos (T078) | Owner priority |
| P11 | Ideas tested and not adopted (continuity box choice, body-length-normalised speed, Savitzky-Golay, pooled mixture, their thresholds): no measurable gain at our metrics' resolution (differences under about 0.01-0.02 are undetectable); "no evidence of gain" is not "no gain" | Per-subject reference timelines from the slide author |

## III.2 Roadmap: lightweight reasoning now, a better vision model once there is more data
1. **Now.** Tabular classifier plus a **lightweight, pre-trained, local language model** for reasoning and wording (II.8). No GPU is needed to train anything in `dcs`; the GPU helps the MLP and the chat model.
2. **Next, on the GB10.** Real run (W1), pick the language model (W8), look at the first honest report with the owner at G2.
3. **When more data exists, a better vision model.** Trigger: enough reviewed (Accepted) videos and labeled frames, and recordings balanced across compounds, dates and cameras (W2, W5). Then, in this order: (a) more labeled frames for the fish detector, with no-fish and reflection frames, evaluated on a held-out set that no choice touched; (b) a **clip-level behavior model** trained on reviewed state seconds, the route left open for Listing/LORR and Surface Breach; (c) optionally a small vision-language model as one more chat tool ("what is happening in this frame?"), advisory only, since it would do worse than the dedicated classifier on ambiguous frames. The drug classifier keeps reading features computed from the reviewed tracks; a video model replaces the labeling stage, not the drug classifier, until G4 says otherwise.
4. **After the chat is validated.** A thin web page over `dcs serve-chat` (the API exists), hosting decided by the lab. Live, real-time use is Phase 2 and needs a windowed redesign of the classifier plus a measured detector speed.
