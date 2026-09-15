# Zebrafish Drug-Response Detection System — Project Scope

**Collaboration:** University of Toledo research team and Pharmacy Department
**Status:** Scoping complete, Phase 1 ready to begin

---

## 1. Project Overview

This project develops an AI system that observes zebrafish behavior on video —
recorded initially, live in a later phase — and estimates the probability that the
fish was exposed to a given compound and dose (e.g. "60% MDMA, 30% DOB, 10% Veh").
The system combines supervised learning (trained on labeled trial data) with
unsupervised learning (behavior-state discovery and novelty detection), and
produces its output as a natural-language explanation, a timestamped behavior log,
and a flag for unusual or novel behavioral patterns.

## 2. Background and Data Sources

### 2.1 Trial database (`00_NTT_DataBase.xlsx`)

Per-trial summary metrics from Novel Tank Test (NTT) tracking software: distance
moved, velocity, and time-in-zone, each split by full arena, top half, and bottom
half of the tank.

- The workbook contains 720 rows; only the first 353 hold real trial data. The
  remaining rows are blank template rows and are excluded from the pipeline.
- The compounds tested span 17 distinct families across 35 compound+dose
  combinations, including two combination treatments (an experimental compound
  paired with methylone). Class sizes range from 77 trials (vehicle control) down
  to single-digit counts for several experimental compounds.
- One column ("Body Tissue") is empty across all 353 trials and is excluded from
  the dataset.

### 2.2 Video dataset

353 videos, one per real trial in the database, giving complete coverage — every
compound+dose class has the same trial count in the video set as in the tabular
data. Each file is named by sex and subject number (e.g. `F_0068.mp4` for subject
#68), matching the corresponding database row directly.

Reference specs: 320×240 resolution, approximately 30 fps, approximately 1200
seconds (20 minutes) per session, single fish per tank, side-view camera, low
contrast lighting. At roughly 6 MB per file, the full set totals approximately 2 GB
of raw footage.

No frame-level behavior annotations exist for these videos. A reference diagram
showing a five-state ethogram (Controlled Swim, Erratic Movement, Freezing/Drift,
Listing/LORR, Surface Breach) illustrates the target reporting style rather than
existing training labels.

## 3. Functional Requirements

The system must produce, for each observed session:

1. **Live behavior narration** — a running, timestamped classification of the
   fish's current behavior state while a session is in progress.
2. **Drug-identity probability** — a probability distribution over known compounds
   (and doses) once the observation window ends or the model reaches sufficient
   confidence.
3. **Reasoning** — a natural-language explanation of the behavioral evidence
   supporting the probability estimate.
4. **Anomaly flag** — an indication when session behavior does not match any known
   compound's typical profile.
5. **Reporting and query interface** — a shareable report in the ethogram-diagram
   style, plus a conversational interface for querying past sessions.

## 4. System Architecture

### 4.1 Behavior-State Extraction (unsupervised / heuristic)

Because no frame-level ground truth exists, this module is not a supervised video
classifier in Phase 1. It instead combines computer-vision tracking with rule-based
and unsupervised methods:

- Computer-vision tracking extracts the fish's position, velocity, acceleration,
  and orientation per frame using background subtraction, given the static camera
  and plain background.
- Rule-based thresholds detect the unambiguous states: Freezing/Drift (sustained
  near-zero velocity), Surface Breach (vertical position crossing the waterline),
  and Listing/LORR (persistently abnormal body orientation).
- Unsupervised clustering (e.g. Gaussian Mixture or Hidden Markov Model) over the
  kinematic feature stream separates Controlled Swim from Erratic Movement, a
  distinction based on movement smoothness and turning-angle variance rather than a
  fixed threshold.
- Pharmacy Department review of the discovered clusters and thresholds validates
  and refines the state definitions. The validated output becomes training data for
  a future supervised state classifier.

### 4.2 Drug-Identity Classification (supervised)

- Phase 1 features draw from the database's per-trial summary metrics (distance
  moved, velocity, time-in-zone) — a small, clean, already-labeled dataset that
  supports a fast initial baseline.
- Phase 2 enriches these features with video-derived signals once 4.1 is in place:
  percentage of time in each behavior state, transition frequency between states,
  and related measures. This is also where the video and tabular pipelines merge.
- **Label strategy:** compound and dose are treated as separate classes. The four
  smallest classes are pooled into an adjacent dose of the same compound to reach a
  workable sample size:
  - FD-2-95 @ 0.1 (n=3) pooled into FD-2-95 @ 0.03 (n=7 → 10)
  - FD-2-67 @ 0.2 (n=3) pooled into FD-2-67 @ 0.1 (n=16 → 19)
  - FD-2-66 @ 0.2 (n=3) pooled into FD-2-66 @ 0.1 (n=6 → 9)
  - FD-2-97 @ 0.1 (n=1) pooled into FD-2-97 @ 0.03 (n=6 → 7)
  - HS-1-51 @ 0.03 (n=1) remains unresolved: it is the only dose tested for that
    compound, so no adjacent dose exists to pool into (see Section 8).
  - This produces a working label set of 31 classes, pending resolution of HS-1-51.
- Given the resulting class sizes, cross-validation is used in place of a single
  held-out test split, and a hierarchical model — predicting compound family first,
  then dose within family — is planned to make the smaller classes more tractable.

### 4.3 Anomaly / Novelty Detection (unsupervised)

Outlier detection on the learned feature embedding (e.g. isolation forest, or
distance from the nearest known-compound cluster) flags sessions that do not match
any known compound's profile, supporting detection of new compounds, dosing
errors, or tracking failures.

### 4.4 Natural-Language Reporting (LLM layer)

This module consumes the timestamped behavior log (4.1), the classifier's
probability output and feature importances (4.2), and the anomaly flag (4.3), and
produces live narration during a session, a final reasoning summary, an
ethogram-style report, and responses to researcher queries about past sessions in a
chat interface. This is the highest-priority and most technically demanding
component of the system. Whether this module calls a hosted API or runs a
self-hosted open-weight model is an open decision (Section 8).

### 4.5 Frontend and Backend

Treated as lower priority relative to 4.1–4.4:

- **Backend:** an API that accepts a video (and, in Phase 2, a live camera stream),
  runs the modules above, and returns the structured result.
- **Frontend:** an interface to upload a session, view live narration, and review
  the final report.

## 5. Data Preparation Rules

- Rows with a blank compound field (the unused template rows) are excluded from all
  processing.
- A literal "-" placeholder in a top-half or bottom-half column indicates the fish
  spent zero time in that zone and is imputed as 0.
- A trial missing all eight movement columns simultaneously (104 such trials,
  occurring in blocks by subject number) indicates the trial was not tracked; these
  values are treated as missing and masked out of training rather than imputed.
- The "Body Tissue" column is excluded entirely (no data across any trial).

## 6. Development Phases

**Phase 1 — Offline analysis of recorded video** (primary scope for the current
timeline)

1. Database cleaning and feature-extraction pipeline (missing-data rules, class
   pooling)
2. Computer-vision tracking pipeline across the 353 videos, producing kinematic
   features
3. Heuristic and unsupervised behavior-state module, validated with the Pharmacy
   Department
4. Supervised compound classifier — Phase 1 on tabular features, Phase 2 enriched
   with video-derived features
5. Anomaly detection module
6. Natural-language reporting module for offline report generation and query
7. Minimal frontend and backend to integrate the above

**Phase 2 — Real-time operation**

8. Live camera integration and streaming inference
9. Real-time natural-language narration

## 7. Technology Stack and Infrastructure

- **Modeling framework:** TensorFlow, for the classifier and other learned
  components. Computer-vision tracking uses OpenCV, which operates independently of
  the modeling framework choice.
- **Compute:** given the video resolution (320×240), dataset size (353 sessions),
  and tabular dataset size (353 rows), this is a modest training workload. A single
  mid-range GPU (16–24 GB VRAM — e.g. RTX 4060 Ti or RTX 4090 class, or an
  equivalent cloud instance) is sufficient for Sections 4.1 through 4.3. A larger
  GPU (24 GB+) would only be required if the natural-language reporting module
  (4.4) self-hosts an open-weight language model rather than calling a hosted API.

## 8. Outstanding Decisions

- **HS-1-51 @ 0.03 (n=1):** the only dose tested for this compound, so it cannot be
  pooled by dose as the other small classes were. Pending decision: exclude it from
  Phase 1 training, or merge it with the related HS-1-151 compound (a distinct
  compound, not simply a different dose).
- **LLM sourcing:** whether the natural-language reporting module (4.4) calls a
  hosted API or runs a self-hosted local model. A hosted API requires no additional
  GPU budget and is the faster path to a working version, but sends data outside
  local infrastructure and depends on network access — a consideration against any
  applicable data-handling requirements for this dataset. Self-hosting keeps
  processing local at a fixed cost after GPU purchase, but adds setup and
  maintenance work and requires the larger GPU noted in Section 7.

## 9. Timeline

The project is open-ended with a target completion by the end of the current year
(approximately 3.5 months from project scoping). Phase 1 (Section 6, items 1–7)
constitutes the realistic scope for this period; Phase 2 (real-time operation) is a
stretch goal or a subsequent milestone.
