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
| **Compound/dose classifier** (`dcs`): training table, audit, baselines and MLP with date-held-out validation, ablations, dose models, saved model, `predict`, research chat (`dcs ask`) | Built and tested on synthetic data. On the unreviewed real output no model yet names the compound better than the recording date does; behavior does separate vehicle from drug. Status and decisions: [backend/dcs/docs/classifier_progress.md](backend/dcs/docs/classifier_progress.md). |
| **Research chat** in the app: local `qwen3:30b-a3b` in Docker, answers built from query tools | Working on the GB10 on unreviewed data; answers follow a lab-editable layout ([backend/dcs/config/chat_guidance.yaml](backend/dcs/config/chat_guidance.yaml)) and show the data lookups behind them. About 15-60 s per answer. |
| Anomaly detection, reporting, live (Phase 2) camera | Not started. |

Model strategy: a **lightweight, pre-trained, local language model** does the reasoning and wording for the research
chat (nothing is trained; every number comes from deterministic query tools), and a **better vision model** is planned
once more reviewed data exists (more labeled frames for the detector, then a clip-level behavior model).

Details and the task-by-task log: [backend/prepds/docs/progress.md](backend/prepds/docs/progress.md). Requirements, watch list and roadmap, all in one
file: [docs/PRD.md](docs/PRD.md) (Part I preprocessing, Part II classifier and chat, Part III open items).

## Quick Start

Everyone uses the same commands. Only Docker (with Compose) is needed.

```sh
cp .env.example .env    # once: set PDS_VIDEO_DIR, PDS_DB_PATH, FISHLAB_HOST, FISHLAB_PORT
./start.sh --setup      # once per machine, and again when inputs change: build + prepare the data, then stop
./start.sh              # start the app and print the local link (fast; asks for --setup when it is needed)
./start.sh --without-model  # start without Ollama and disable the research chat for this launch
./stop.sh               # shut down
```

If `.env` is not set up, `./start.sh` says what is missing and starts nothing. Use `--without-model` when Ollama
is unavailable or the research chat is not needed; this does not change `.env`. It prints this workflow and stops
at the first failed step; `--setup` runs steps 1-5, plain `./start.sh` builds if needed and runs 6-7 (with chat
on, both start the model server first, since preparation checks it), `--without-model` skips the model and chat,
and `--full` redoes everything:

1. **Build** the Docker image (frontend and Python): a few minutes the first time.
2. **Check** the configuration and **catalog** trials against videos.
3. **Track** and label every video not yet processed (`prepds run`): about 35 s per 20-minute video per
   worker; 328 videos took about 50 minutes on 22 workers. Finished videos are skipped on reruns.
4. **Verify** every video has its outputs.
5. **DCS** feature extraction, audit, model (trains only if none is saved) and predictions.
6. Optional research **chat** (`FISHLAB_DCS_CHAT=1`): starts the local model server container (Ollama on the GPU,
   models kept in `.ollama/models`). Download the model once. Pass `./start.sh --without-model` to skip this step
   and run the app without the research chat:
   `docker compose --profile chat up -d model && docker compose exec model ollama pull qwen3:30b-a3b`.
7. **Serve** the frontend and API and print one local link.

Review and accept videos in the browser, then `./stop.sh` stops the app and removes the containers. Every command, why it
exists and how it works (prepds, dcs, the research chat, the GB10 runbook, what to trust, error fixes), plus the reviewer steps:
**[docs/RUNBOOK.md](docs/RUNBOOK.md)**. Running without Docker: section 4 there.

```bash
pip install -e ".[dev,train]"   # adds PyTorch for the classifier's MLP (optional)
python -m dcs check-config
```

## Repository Layout

```text
fish-detection/
├── backend/
│   ├── prepds/        # preprocessing pipeline: src/, tests/, config/ (thresholds, calibration profiles), scripts/, docs/
│   ├── dcs/           # compound/dose classifier and research chat (reads prepds output, never imports it): src/, tests/, config/, docs/
│   └── shared/        # used by both: tests/fixtures (synthetic data), tests/deployment, scripts/ (docker_app.py, changed_tests.py)
├── frontend/          # React review UI
├── docs/              # RUNBOOK, PRD, scope, contribution workflow (cross-unit docs; unit docs live in backend/<unit>/docs)
├── start.sh, stop.sh  # the way everyone runs the project (Docker)
├── Dockerfile, compose.yaml   # launcher internals
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
