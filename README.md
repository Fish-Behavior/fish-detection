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

The system will combine computer-vision tracking, supervised learning, and
unsupervised behavior-state discovery. The ethogram has five target states
(Controlled Swim, Erratic Movement, Freezing/Drift, Listing/LORR, Surface Breach)
plus Undetermined and Dead. Full scope:
[docs/zebrafish_drug_detection_scope.md](docs/zebrafish_drug_detection_scope.md).

## Status

| Part | State |
|---|---|
| **Preprocessing dataset system** (`prepds`): catalog, tracking, features, labeling, review app, gold-dataset export | Built; 328 of 352 trials processed from the 337 synced videos. Human review of the output is still to do. |
| Model tracker (fine-tuned fish detector) | First model trained; detection is much better than the classical tracker, Listing is still unreliable. Full re-run not done. |
| Listing/LORR and Dead | Not detected automatically; they come from reviewers. |
| Drug classifier, anomaly detection, reporting, live (Phase 2) camera | Not started. |

Details and the task-by-task log: [docs/progress.md](docs/progress.md). Requirements:
[docs/PRD.md](docs/PRD.md).

## Quick Start

Everyone uses the same two commands. Only Docker (with Compose) is needed.

```sh
cp .env.example .env    # once: set PDS_VIDEO_DIR, PDS_DB_PATH, FISHLAB_HOST, FISHLAB_PORT
./start.sh              # build, prepare, start the app, print the local link
./stop.sh               # back up saved work and shut down
```

If `.env` is not set up, `./start.sh` says what is missing and starts nothing. When it runs it prints
this workflow and stops at the first failed step:

1. **Build** the Docker image (frontend and Python): a few minutes the first time.
2. **Check** the configuration and **catalog** trials against videos.
3. **Track** and label every video not yet processed (`prepds run`): about 35 s per 20-minute video per
   worker; 328 videos took about 50 minutes on 22 workers. Finished videos are skipped on reruns.
4. **Verify** every video has its outputs.
5. **DCS** feature extraction, audit, model (trains only if none is saved) and predictions.
6. Optional research **chat**.
7. **Serve** the frontend and API and print one local link.

Review and accept videos in the browser, then `./stop.sh` stops the app, verifies a dated backup under
`backups/`, and removes the containers. Details, the review UI, outputs, calibration and what to trust
in the labels: **[docs/instructions.md](docs/instructions.md)**. Contributors who need to run the code or
tests without Docker: see "Development without Docker" there.

## Repository Layout

```text
fish-detection/
├── src/prepds/        # the pipeline package (catalog, tracking, features, labeling, webapp, ...)
├── scripts/           # calibration and detector fine-tuning scripts
├── config/            # default thresholds and frozen calibration profiles
├── tests/             # automated tests (synthetic fixtures only)
├── docs/              # instructions, PRD, progress log, scope, contribution workflow
├── start.sh, stop.sh  # the way everyone runs the project (Docker)
├── Dockerfile, compose.yaml, scripts/docker_app.py   # launcher internals
├── .env.example       # template for your local .env; never commit .env
├── pyproject.toml, requirements.txt, uv.lock
└── data/, outputs/, accepted/   # local only, git-ignored (restricted research data)
```

## Data Sources and Access

- **Trial database** `00_NTT_DataBase.xlsx`: 720 rows, 353 with real trial data; blank
  compound rows and the empty `Body Tissue` column are excluded.
- **Videos**: one per real trial, named by sex and subject number (e.g. `F_0068.mp4`),
  about 320x240, 30 fps, 20 minutes, single fish in a side-view tank.

The database, videos and reference figures are restricted research data. Obtain them
through the project channels and never commit them. Keep local paths and secrets in
`.env`. A hosted language model for future reporting would send data off site and
needs approval first.

## Development Phases

1. **Phase 1, offline analysis of recorded video:** clean the database; track the
   fish; discover and validate behavior states; train and evaluate the compound
   classifier; anomaly detection; natural-language reporting; minimal backend and
   frontend.
2. **Phase 2, real-time operation:** live camera, streaming inference and narration
   (stretch goal).

### Open decisions

- HS-1-51 @ 0.03 has one trial and cannot be pooled with an adjacent dose.
- Whether probabilities such as `60% MDMA, 30% DOB, 10% Veh` are mutually exclusive
  class probabilities or a mixture/multi-label result.
- Observation windows, confidence thresholds, report schema and the language-model
  provider (hosted vs self-hosted).

## Contribution Workflow

Use a feature branch for every change and submit work through a pull request. Do
not commit or push directly to `master`. See [docs/how-to-start.md](docs/how-to-start.md).
