# Backend integration

Implemented on `feature/frontend-backend`. Docker remains the next separate phase.

The React UI uses the existing prepds FastAPI service. The application is a local, single reviewer tool with one Uvicorn worker. PostgreSQL is unnecessary for this workflow: each save atomically replaces a per-video JSON file. A revision check under a filesystem lock rejects stale saves from other tabs.

## Run

From the repository root, configure `.env` using `.env.example`, including `PDS_VIDEO_DIR`, `PDS_DB_PATH`, `PDS_OUTPUT_DIR`, and `PDS_ACCEPTED_DIR`. Source videos stay in the configured folder; no uploads are required.

```sh
.venv/bin/python -m prepds catalog
cd frontend
npm ci
npm start
```

Open `http://<FISHLAB_HOST>:<FISHLAB_PORT>/` (host and ports come only from `FISHLAB_HOST`, `FISHLAB_PORT`, `FISHLAB_API_PORT` in the root `.env`; there are no built-in defaults). `npm start` builds and starts the complete app. `npm run preview` starts the same Python server with the existing build. `npm run dev` starts a private Python API on `FISHLAB_API_PORT`, waits for it, and starts Vite on `FISHLAB_PORT`; `/api`, `/docs` and `/openapi.json` are proxied. All three commands present the same browser address, read `.env` from the repository root, check for occupied ports and stop their owned processes on Ctrl+C. Run only one of them at a time.

The advanced `.venv/bin/python -m prepds review` command still serves `frontend/dist` when present, otherwise the original prepds review page. It can be used independently of the npm launcher. `--frontend-dir` selects another build folder. The launcher accepts the optional `--chat-url`, `--predictions`, `--classifier-root` and `--model-run` review arguments, but fixes its own host, ports and frontend directory.

Manifest source paths may be absolute, relative to `PDS_VIDEO_DIR`, or repository-relative as written by the pipeline. Every resolved video must remain inside the configured source folder. Repository-relative paths no longer receive a duplicate video-folder prefix. Missing files are reported for the selected recording while the reload button stays available.

The sidebar recording list includes processed, unprocessed, unmatched, corrupt, failed, and interrupted recordings. `Run analysis` uses the existing classical tracker, calibrated labeling, and export pipeline for a uniquely matched catalog trial. Fix unmatched or duplicate workbook/catalog identities before processing. Existing output and correction records cannot be overwritten by this action. Processing is serialized within the single server worker, continues after the request, and can be polled or retried after a failure. If the server restarts mid-job, the API reports interrupted status.

## API

Swagger/OpenAPI: `/docs` on the app link.

| Method and path | Contract |
| --- | --- |
| `GET /api/health` | Backend status and configured capabilities |
| `GET /api/sessions` | Inventory and processing/review summaries |
| `GET /api/sessions/{id}` | Normalized session, baseline fingerprint, saved review, stale flags and warnings |
| `GET /api/sessions/{id}/video` | Source stream with byte-range seeking |
| `GET /api/sessions/{id}/overlay?start_s=…&end_s=…` | At most 30 seconds, with absolute frame indices and source dimensions |
| `GET /api/sessions/{id}/review-tools?baseline=…` | Existing prepds manifest metadata (without source paths), pipeline review flags, and advisory Listing hints/errors; rejects a changed baseline |
| `GET /api/sessions/{id}/explain?second=…&baseline=…` | Existing prepds per-second explanation of the original stored track and its exact calibration profile; rejects a changed baseline |
| `PUT /api/sessions/{id}/review` | `{revision, baseline, reviewer, edits, decision}`; complete replacement of the correction document |
| `POST /api/sessions/{id}/process` | `{reviewer}`; queue analysis, return 202 |
| `GET /api/sessions/{id}/processing` | Current job status/message |
| `POST /api/sessions/{id}/rerun` | `{revision, baseline, reviewer}`; recompute reviewed measurements and labels |
| `POST /api/sessions/{id}/predict` | `{revision, baseline, reviewer}`; DCS feature extraction and saved-model inference |
| `POST /api/ask` | DCS `{question, history}` → `{answer, tool_calls, history}` |

`edits` matches `src/model.ts`: nullable scene, frame corrections keyed by absolute frame index, behavior label intervals, and a nullable final compound/dose override. Scene/tracking changes mark measurements, automatic labels and predictions stale; label changes mark derived features/predictions stale. A manual final result stays separate from automatic model probabilities. All corrections and decisions retain reviewer names, server timestamps and revisions. Server validation enforces finite coordinates, source dimensions/frame count, time ranges, reviewers, keypoint scores, known states, and terminal Dead labels.

Errors use HTTP status and `detail`, generally `{code, message}`. Invalid requests return 422, revision or baseline conflicts return 409, unavailable capabilities return 503, and unexpected failures return a sanitized 500 with details only in the backend log. The UI provides request timeouts, reload, inline errors, processing polling, and save acknowledgment. On a timeout, reload before retrying because the server may have committed the save. If another tab saved, reload to obtain its revision.

## Persistence and recalculation

`<PDS_OUTPUT_DIR>/processed/<video_id>/review.json` holds the current corrections, revision, reviewer, timestamp, independent review decision, recalculated results and model result/provenance. `processing.json` holds the latest job status. Writes use a same-filesystem temporary file, flush/fsync and atomic replacement. `.workspace.lock` protects the revision check plus write. There is no database, migration, version-history system, or per-edit append-only log.

Automatic `frames.parquet`, `segments.csv`, detector data and manifests remain the original baseline; manual changes replace `review.json`. Reset restores the original result while keeping a durable reset revision. Fingerprints include source and pipeline artifact sizes/modification times; changed inputs are refused rather than silently reusing corrections. To review a replacement recording, explicitly archive the incompatible `review.json` first. Fingerprints detect changes cheaply and are not content hashes.

`Rerun affected stages` applies manual points, missing-fish flags and ROI membership to the stored track, uses the exact calibration profile for new labels, applies the behavior overrides, and recomputes kinematics using prepds helpers. Measurements that cannot be computed are null and break the chart lines. Waterline depth is displayed only when a waterline is known; calibrated behavior rules retain their existing frame-top coordinate contract. Box/keypoint edits are durable review annotations; they do not invent a calibrated body angle or rerun detector training. Rerun does not decode/retrack the original video or clear compound staleness. `Run compound model` handles that separately.

A FishLab accept/reject decision is saved separately from prepds gold acceptance; it does not publish a gold dataset, erase corrections, validate model performance, or clear stale outputs. The legacy `/videos/.../accept` endpoint retains its existing scientific acceptance rules and gold export behavior.

## Classifier and research chat

DCS currently lives on `feature/dcs-wrapup`. Keep a separate DCS checkout and its environment/settings; the backend does not merge branches, train a model, or fabricate outputs. The Python interpreter running prepds must also have the dependencies that checkout requires. `--classifier-root` supplies DCS source via `PYTHONPATH` to its existing CLI. Exported `DCS_*` paths are resolved against the app's working directory before the subprocess changes to the DCS checkout; the checkout's own `.env` and defaults still supply settings that were not exported. The npm launcher exports the root `.env` automatically; when launching Python directly, export needed DCS variables or configure the DCS checkout's `.env`.

With a trained run and DCS checkout:

```sh
.venv/bin/python -m prepds review --classifier-root /path/to/dcs-checkout --model-run /path/to/trained-run
```

Inference builds a temporary per-video snapshot of corrected frames and segments, preserving DCS dtypes and pipeline provenance, then runs native `dcs featurize --videos` and `dcs predict --model`. The original pipeline files are not modified. The temporary snapshot is removed afterward; the probabilities and run name are persisted with the review revision. Required upstream recalculation must finish first. An existing DCS prediction CSV/parquet can also be displayed with `--predictions /path/to/predictions.csv`; invalid probabilities or missing video results are shown as unavailable.

### Terminal workflow

Playback is independent of analysis. Watching a recording to the end does not start inference. `Run analysis` computes tracking and behavior; `Run compound model` runs inference after analysis or correction recalculation. Enter a reviewer name in Review before running either action.

For batch operation, run these commands from the prepds repository root, with DCS settings in the root `.env`. `DCS_PROCESSED_DIR` must point at the prepds output folder and `training.gold_source` must select `processed` for unreviewed outputs; use `accepted` for gold data. The model is trained separately and reused for predictions:

```sh
.venv/bin/python -m prepds run --skip-existing
PYTHONPATH=/path/to/dcs-checkout/src .venv/bin/python -m dcs --env-file .env featurize
PYTHONPATH=/path/to/dcs-checkout/src .venv/bin/python -m dcs --env-file .env predict \
  --model /path/to/trained-run --input /path/to/training_table.parquet \
  --out outputs/dcs/compound_predictions.csv
npm --prefix frontend start -- --classifier-root /path/to/dcs-checkout \
  --model-run /path/to/trained-run --predictions outputs/dcs/compound_predictions.csv
```

Stop an existing app before starting another. With `--predictions`, matching results appear when a recording loads; reload after regenerating the CSV. New recordings need inference before a result appears. Saved corrections require per-video recalculation/inference so predictions reflect the reviewed data. The backend does not automatically chain analysis into inference.

Start the existing DCS chat service from its checkout with its configured local model, using `python -m dcs serve-chat --port <port>`. Connect it with:

```sh
.venv/bin/python -m prepds review --chat-url <FISHLAB_CHAT_URL>   # or FISHLAB_CHAT_URL in .env
```

`DCS_CHAT_BASE_URL` points DCS at its language-model server (for example Ollama's `/v1` endpoint). `FISHLAB_CHAT_URL` points FishLab at the DCS chat service. They are separate services.

Options can be combined. The chat bubble is always available in a session. Without a configured chat service, the panel shows a not-connected message and disables questions; configured services enable sending. Service/model failures remain visible inside chat. The adapter accepts only a loopback HTTP URL and refuses redirects and environment HTTP proxies; remote chat is not enabled by this connection. Questions/history live in the frontend and are not saved in the correction file. The DCS service researches its configured dataset, not unsaved FishLab drafts.

## Verification and limits

- Python contract/regression tests: `pytest tests/review tests/ingest/test_cli.py tests/ingest/test_config.py`; the local optional tracker test needs `torchvision`.
- Frontend: `npm test` and `npm run build`.
- Native DCS CLI inference was exercised with a trained synthetic logistic regression artifact and synthetic recording, separately from real research data.
- Real per-recording inference through `/api/sessions/{id}/predict` was verified on the running app: 15-class probabilities persisted across reload, in about 22 seconds, with the original frames, segments and manifest unchanged.
- On 2026-10-08, the separate DCS checkout passed 1000 tests. An exploratory CPU logistic regression run trained on 323 retained fish from the existing 328-row unreviewed table, then scored all 328 rows. Its verdict was **not useful**, so these probabilities verify integration and do not establish reliable compound identification. One cross-validation repeat was used and ablations were skipped. Regenerate older artifacts when the DCS model format changes; current artifacts include a model-file checksum.
- Browser interaction could not be exercised in this environment: Playwright's profile was already in use and no computer-use browser was available. Verify video codec playback, responsive overlays, correction previews, and the full UI on owner hardware.
- Run one Uvicorn worker, on loopback. This phase has no authentication, multi-user deployment, GPU/model tracker selection in the UI, full retracking after scene edits, validated research model, backup/history policy, transcoding, or Docker. Those are explicit boundaries; Docker is the user's next call.
