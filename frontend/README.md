# FishLab frontend

React/TypeScript interface for zebrafish behavior analysis: a **Dashboard** for exploring a session and a **Review** page for correcting tracking, scene, and behavior results. The interface connects to the existing prepds FastAPI service with file-backed correction persistence.

> The backend inventories the configured video folder. No bundled research data exists. `npm start` starts the whole app. See [BACKEND_HANDOFF.md](BACKEND_HANDOFF.md) for setup, API contracts, persistence, analysis, DCS inference and chat.

## Run

Requires Node.js 22.18 or newer, npm, and the project's `.venv` with Python dependencies installed. Configure the repository-root `.env` first. From the repository root:

```sh
npm --prefix frontend ci       # first-time frontend setup
npm --prefix frontend start    # build, then start frontend + API together
```

Open the address printed on startup (default **http://127.0.0.1:8000**; set `FISHLAB_HOST`/`FISHLAB_PORT` in `.env`). The scrollable sidebar list includes the entire source video inventory, including unprocessed videos; “Needs review” filters individual review decisions, while “With corrections” shows saved edits. Use one launch command at a time; Ctrl+C stops the services it started. The launcher always reads `.env` from the repository root, even when run from `frontend/`.

From `frontend/`:

| Command | Behavior |
| --- | --- |
| `npm start` | Build and serve the complete app on port 8000 |
| `npm run dev` | Start the API automatically and serve the same app at port 8000 with live updates |
| `npm run preview` | Serve the existing `dist/` build plus API at port 8000; requires a successful build |
| `npm run build` | Type-check and write static files to `dist/`, without starting a server |
| `npm test` | Frontend model, API and rendered-component tests |
| `npm run test:startup` | Launcher startup, port-conflict, failure and shutdown tests using temporary data |

Normal use and preview use one Python server. Development uses an internal API at loopback port 8008; Vite proxies API and documentation requests so the browser still uses only port 8000. Never start `prepds review` separately alongside these npm commands. Occupied ports cause a clear startup error, and unrelated processes are left alone.

Optional backend settings can be forwarded, for example `npm start -- --classifier-root /path/to/dcs --model-run /path/to/run --chat-url http://127.0.0.1:8010` (or set `FISHLAB_CHAT_URL` in `.env`). This points to the DCS research-chat service; DCS's `DCS_CHAT_BASE_URL` points to its language-model server. The npm launcher accepts `--chat-url`, `--predictions`, `--classifier-root`, and `--model-run`; use the Python CLI directly for other server options.

## Pages

Both pages render only what the session contains: no measurements, labels, predictions, tracking, or detector output means that card (or pipeline stage) shows as unavailable or is hidden.

**Dashboard** — simple video playback with saved tracking overlays (track point, 2 s trail, detector box, keypoints, waterline), behavior timeline (ethogram), movement traces, behavior segments, compound prediction, time per state, and session details. Playback, the seek slider, charts, and segment table share one playhead. Unsaved review drafts stay out of this view.

**Review** — the same video and timeline plus correction tools; video and time stay shared with the Dashboard.

- *Tracking review workspace*: previous/next frame, numerical time, layer toggles including ROI, range-start/end buttons, and keyboard shortcuts. Focus the player: arrows seek 1 second (Shift: 10 seconds); `[` and `]` mark the range. Frame steps use stored timestamps when available; browser seeking is approximate.
- *Prepds review flags*: subject/exposure and calibration metadata, pipeline flags, and possible Listing/LORR hints. Clicking a flag selects its range and seeks there. Hints remain advisory and reflect the original output.
- *Why this label?*: per-second stored label, detection counts, measurements, frame votes, and calibration rule explanations from the existing prepds explainer. It explicitly explains the original stored track, rather than saved or staged corrections. Missing profiles or hints show actionable errors.
- *Review controls*: reviewer name (needed to save, not to preview) and the video interaction tool.
- *Scene settings*: waterline and ROI. *Frame correction*: tracking point, missing-fish flag, detector box, keypoints (only where the detector produced them). *Relabel a time range*: behavior override, under the timeline.
- *Stage range*: collect multiple relabels, remove individual staged edits, then save or discard the whole batch. Staged labels preview only in Review. Failed saves retain the batch; switching recordings or reloading warns before discarding it. Latest overlapping edit wins; ranges are `[start, end)`. Review decisions and reruns require saving or discarding staged labels first.
- Review's *Time per state* includes percentages and bout counts.
- Edits **preview instantly in purple** on the video and timeline while you type, click, drag a rectangle, or move the draft with the arrow keys (focus the video; Shift = 10 px). Nothing is saved until you apply it. Changing frames discards unsaved frame drafts.
- *Review decision*: accept/reject, separate from the scientific result. Scene/tracking corrections mark measurements, labels, and predictions stale; behavior corrections mark features and predictions stale. "Rerun affected stages" recomputes reviewed measurements and labels. Compound probabilities remain stale until "Run compound model" succeeds with a configured DCS artifact.
- A manual final compound/dose override never changes the automatic probabilities. Model dose estimates are unavailable in the current DCS artifact.

Applied edits, reviewer attribution and review decisions survive refresh in the per-video `review.json`. Each apply/reset waits for server acknowledgment. **Reset edits** restores the automatic baseline. Unsaved drafts are discarded on reload; chat remains in memory.

## Data contract

`src/model.ts` holds the types (`SessionData`, `OverlayWindow`, `DetectionOut`, corrections, chat) and the validation/derivation helpers. `OverlayWindow`/`DetectionOut` preserve prepds's `fps`, coded dimensions, timestamps, nullable x/y, detected flags, detector boxes, and named keypoint arrays. Coordinates are source-video pixels, independent of display scaling; missing detections hide the marker and split trails. A frame without a detector record has `box: null` and no keypoints; nothing is generated in their place. `predictions` is `null` when no model result exists, and `measurements`/`segments` may be empty. Predictions keep DCS's `video_id`, `predicted`, and `p:<class>` columns. Chat matches DCS `/api/ask` (`question`/`history` → `answer`/`tool_calls`/`history`).

## Tests

`npm test` covers correction validation, overlapping labels, duration totals, stale dependencies, immutable automatic predictions, missing-frame trails, absent detector/tracking data, letterbox coordinate mapping, windowed absolute frame alignment and stepping, HTTP/network/timeout error handling, and rendered Review/Dashboard controls and draft separation. The tests build their own small session; no shared sample data exists. `npm run build` type-checks the app.
