# Zebrafish Drug-Response Detection System

Advisor: Dr. Ashish Kharel

Collaborator: Dr. Scott Hall

## Project Overview

This project develops an AI system that analyzes zebrafish behavior to estimate
which psychoactive compound and dose produced an observed response. The work is a
collaboration between the University of Toledo research team and Pharmacy
Department.

The project addresses the overdose and drug-abuse crisis, where current methods
cannot rapidly classify newly emerging psychoactive compounds. Behavioral
fingerprinting focuses on measurable features including distance, velocity,
anxiety, exploration, and mobility.

## Scientific Objective

The system will combine computer-vision tracking, supervised learning, and
unsupervised behavior-state discovery. For each session, it is intended to:

- extract movement and behavioral features from recorded or live video;
- estimate probabilities across known compound-and-dose classes;
- identify behavior that does not match known compound profiles;
- produce a timestamped behavior log and natural-language explanation; and
- support researcher queries about previous sessions.

The initial ethogram contains five target states: Controlled Swim, Erratic
Movement, Freezing/Drift, Listing/LORR, and Surface Breach.

## Current Scope and Status

Scoping is complete and Phase 1 is ready to begin. The committed near-term scope
is offline analysis of recorded video. Phase 2 is planned for live camera
integration, streaming inference, and real-time narration.

See the complete scientific and technical scope in
[docs/zebrafish_drug_detection_scope.md](docs/zebrafish_drug_detection_scope.md).

The repository currently contains project documentation and workflow configuration;
the analysis pipeline, dependency manifest, datasets, tests, backend, and frontend
are still to be implemented.

## Data Sources and Access

### Trial database

The planned tabular source is `00_NTT_DataBase.xlsx`, which contains 720 rows:

- 353 rows contain real trial data; the remaining rows are blank templates;
- metrics include distance moved, velocity, and time-in-zone measurements;
- the empty `Body Tissue` column is excluded; and
- blank compound rows are excluded from processing.

### Video dataset

There are 353 videos, one for each real trial. Filenames identify sex and subject
number, such as `F_0068.mp4`, and match the corresponding database row.

The reference recordings are approximately 320x240 pixels, 30 fps, and 20 minutes
long, with a single fish in a side-view tank. The complete collection is
approximately 2 GB.

The database and videos are restricted research data and must be obtained through
the appropriate project channels. Do not commit them to this repository.

## Prerequisites

Implementation prerequisites are pending because the runtime and dependency
manifests have not yet been created. The planned stack includes:

- Python and a project-managed virtual environment;
- TensorFlow for learned components;
- OpenCV for computer-vision tracking;
- a mid-range GPU for the classifier and tracking experiments; and
- either a hosted language-model API or a self-hosted model for reporting.

Supported Python, TensorFlow, OpenCV, GPU, language-model, and network requirements
will be recorded here when implementation begins. Do not install or run commands
from this section yet; no executable application entry point exists.

## Quick Start

The project is not yet runnable. The intended Phase 1 workflow is:

```text
data acquisition
	-> dataset validation and database/video matching
	-> database cleaning
	-> video tracking
	-> behavior-state extraction
	-> classifier training and evaluation
	-> anomaly detection
	-> report generation
```

### Environment Setup

Create the project environment and install dependencies once the implementation
adds a supported Python version and dependency manifest.

### Data Placement

Place the approved workbook and videos in the project data location defined by the
future pipeline configuration. Keep restricted data outside Git and document the
local path through environment or configuration settings rather than hard-coding
it.

### Run the Phase 1 Pipeline

The future pipeline will clean and validate the trial database, match each trial to
its video, extract kinematic features, discover behavior states, train and evaluate
the classifier, and run anomaly detection. Script and module names will be added
when those components exist.

### Generate a Report

The future reporting step will combine the behavior log, classifier probabilities,
feature evidence, and anomaly result into an offline ethogram-style report. The
report schema and language-model provider are still open decisions.

## Expected Outputs

The planned Phase 1 outputs are:

- cleaned tabular data and a database/video validation report;
- per-frame and per-session tracking features;
- a timestamped behavior log and behavior-state summaries;
- classifier probabilities over compound-and-dose labels;
- an anomaly or novelty flag;
- feature-based evidence and a natural-language reasoning summary; and
- an offline, shareable report with researcher-query support.

Live narration during an active session is a Phase 2 requirement. Phase 1 may
produce timestamped narration from recorded sessions after the reporting design is
implemented.

## Development Phases

### Phase 1: Offline analysis of recorded video

1. Clean the database, apply missing-data rules, and pool small classes.
2. Build computer-vision tracking across the 353 videos.
3. Discover and validate behavior states with the Pharmacy Department.
4. Train and evaluate the supervised compound classifier, starting with tabular
	 features and later adding video-derived features.
5. Build anomaly and novelty detection.
6. Build natural-language reporting for offline reports and queries.
7. Add the minimal backend and frontend needed to integrate these modules.

### Phase 2: Real-time operation

8. Integrate a live camera and streaming inference.
9. Add real-time natural-language narration.

Phase 1 is the realistic scope for the current project timeline. Phase 2 is a
stretch goal or subsequent milestone.

## Known Decisions and Limitations

### Data preparation

- Rows with a blank compound field are excluded.
- A `-` in a top-half or bottom-half field is interpreted as zero time in that
	zone.
- Trials missing all eight movement columns are treated as untracked and masked
	out of training rather than imputed. The scope identifies 104 such trials.
- `Body Tissue` is excluded because it is empty for all real trials.

### Modeling

- No frame-level behavior annotations exist, so Phase 1 behavior-state extraction
	uses tracking, rules, and unsupervised methods rather than a supervised video
	classifier.
- Compound and dose are separate label dimensions; each compound-and-dose pair is
	treated as a classification class.
- Four very small classes are pooled into adjacent doses of the same compound,
	producing a 31-class working label set pending the decision for HS-1-51 @ 0.03.
- Cross-validation is planned instead of a single held-out test split because
	several classes remain small.

### Open decisions

- HS-1-51 @ 0.03 has one trial and cannot be pooled with an adjacent dose. It may
	be excluded, retained as a singleton, or handled through another approved rule.
- The meaning of probability examples such as `60% MDMA, 30% DOB, 10% Veh` must
	be confirmed as mutually exclusive class probabilities or a mixture/multi-label
	result.
- Observation windows, confidence thresholds, calibration, stopping rules, report
	schema, and video/database mismatch behavior remain to be defined.
- The reporting layer may use a hosted API or a self-hosted language model. This
	choice depends on privacy requirements, network access, maintenance, and GPU
	availability.

## Contribution Workflow

Use a feature branch for every change and submit work through a pull request. Do
not commit or push directly to `master`.

The complete workflow, including branch creation, review, commits, pushes, PRs,
branch synchronization, and cleanup, is documented in
[docs/how-to-start.md](docs/how-to-start.md).

## Data Handling and Privacy

Never commit the trial workbook, video recordings, generated reports, model
artifacts, credentials, API keys, or other restricted research data. Keep local
data paths and secrets in ignored configuration or environment variables.

A hosted language model may simplify deployment but sends report inputs outside
the local infrastructure and requires approval under the project's data-handling
requirements. A self-hosted model keeps processing local but adds GPU, setup, and
maintenance costs. This decision must be resolved before production reporting is
enabled.