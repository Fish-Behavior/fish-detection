# FishLab frontend

React/TypeScript interface for zebrafish behavior analysis: a **Dashboard** for exploring a session and a **Review** page for correcting tracking, scene, and behavior results. The interface connects to the existing prepds FastAPI service with file-backed correction persistence.

> The backend inventories the configured video folder. No bundled research data exists. `npm start` starts the whole app. Setup, persistence, analysis, DCS inference and the research chat: [docs/RUNBOOK.md](../docs/RUNBOOK.md). API contracts: `/docs` (Swagger) on the running app.

## Run

To use the whole app, run `./start.sh` (or `./start.sh --with-model` when Ollama/chat is needed) and
`./stop.sh` from the repository root; see the root README. The commands below are for frontend development only.

Requires Node.js 22.18 or newer, npm, and the project's `.venv` with Python dependencies installed. Configure the repository-root `.env` first. From the repository root:

```sh
npm --prefix frontend ci       # first-time frontend setup
npm --prefix frontend start    # build, then start frontend + API together
```

Open the address printed on startup (`FISHLAB_HOST`:`FISHLAB_PORT` from `.env`; both are required). The scrollable sidebar list includes the entire source video inventory, including unprocessed videos; “Needs review” filters individual review decisions, while “With corrections” shows saved edits. Use one launch command at a time; Ctrl+C stops the services it started. The launcher always reads `.env` from the repository root, even when run from `frontend/`.

From `frontend/`:

| Command | Behavior |
| --- | --- |
| `npm start` | Build and serve the complete app on `FISHLAB_PORT` |
| `npm run dev` | Start the API automatically and serve the same app at `FISHLAB_PORT` with live updates |
| `npm run preview` | Serve the existing `dist/` build plus API at `FISHLAB_PORT`; requires a successful build |
| `npm run build` | Type-check and write static files to `dist/`, without starting a server |
| `npm test` | Frontend model, API and rendered-component tests |
| `npm run test:startup` | Launcher startup, port-conflict, failure and shutdown tests using temporary data |

Normal use and preview use one Python server. Development uses an internal API at `FISHLAB_API_PORT`; Vite proxies API and documentation requests so the browser still uses only `FISHLAB_PORT`. Never start `prepds review` separately alongside these npm commands. Occupied ports cause a clear startup error, and unrelated processes are left alone.

Optional backend settings can be forwarded, for example `npm start -- --classifier-root /path/to/dcs --model-run /path/to/run --chat-url <FISHLAB_CHAT_URL>` (or set `FISHLAB_CHAT_URL` in `.env`). This points to the DCS research-chat service; DCS's `DCS_CHAT_BASE_URL` points to its language-model server. The npm launcher accepts `--chat-url`, `--predictions`, `--classifier-root`, and `--model-run`; use the Python CLI directly for other server options.

## Pages

Both pages render only what the session contains: no measurements, labels, predictions, tracking, or detector output means that card (or pipeline stage) shows as unavailable or is hidden.

**Look and wording.** The page is written for the client, not the pipeline: stored codes are shown in plain words (`PROCESSED_AUTO` reads "Not reviewed yet", `processed` reads "Analyzed"), the pipeline steps are Setup, Fish tracking, Measurements, Behaviors and Drug prediction, and "stale" reads "Out of date". The header carries The University of Toledo logo (`src/assets/ut-logo.png`, bundled so the server's `/assets` route serves it) over a gold rule, with **Refresh** to reload recordings and results. Colours are tokens in `src/styles.css` `:root`, built on UT blue `#003e7e` and gold `#ffd200` sampled from that logo: gold marks the current view, light red marks unsaved work and errors, amber marks out-of-date results, green marks finished steps.

**Dashboard** — read-only: video playback with saved tracking overlays and a legend for them (fish position, 2-second path, detection box, body points, water surface), the behavior timeline, movement over time, the behavior log, the drug prediction (the top three compounds, with every compound one click away under *All compounds*), time in each behavior, and recording details (length, frame rate, picture size, saved corrections). Playback, the seek slider, charts, and the log share one playhead. Editing actions (undo all corrections, the manual final result, updating results) live on the Review page, and unsaved review drafts stay out of this view.

**Review** — the same video and timeline plus correction tools; video and time stay shared with the Dashboard.

- *Correction toolbar* above the video: Inspect (`V`), Re-point fish (`P`), Box (`B`) and Keypoint (`K`) where the detector produced them, ROI (`R`), Waterline (`W`); `Esc` returns to Inspect. A one-line hint under the toolbar says what the selected tool does, and *Keyboard shortcuts* below the player lists the keys. Picking a tool moves keyboard focus to the player. Re-point fish: click the fish, drag to fine-tune. ROI and Box: drag a new rectangle, drag inside it to move it, or drag a corner handle to resize; the preview follows the drag.
- *Tracking review workspace*: previous/next frame, numerical time, layer toggles including ROI, range-start/end buttons, and keyboard shortcuts. Focus the player: arrows seek 1 second (Shift: 10 seconds) in Inspect and nudge the draft 1 px (Shift: 10 px) with a tool; `,` and `.` step one frame; `[` and `]` mark the range. Frame steps use stored timestamps when available; browser seeking is approximate.
- *Flagged moments*: subject/exposure and calibration metadata, pipeline flags, and possible Listing/LORR hints. Clicking a flag selects its range and seeks there. Hints remain advisory and reflect the original output.
- *Why this label?*: per-second stored label, detection counts, measurements, frame votes, and calibration rule explanations from the existing prepds explainer. It explicitly explains the original stored track, rather than saved or staged corrections. Missing profiles or hints show actionable errors.
- *Reviewer*: reviewer name (needed to save, not to preview).
- *Scene settings*: waterline and ROI. *Frame correction*: tracking point, missing-fish flag, detector box, keypoints (only where the detector produced them). *Relabel a time range*: behavior override, under the timeline.
- *Stage range*: collect multiple relabels, remove individual staged edits, then save or discard the whole batch. Staged labels preview only in Review. Failed saves retain the batch. Latest overlapping edit wins; ranges are `[start, end)`. Review decisions and reruns require saving or discarding all unsaved edits first.
- Review's *Time in each behavior* adds seconds and bout counts to the percentages.
- Edits **preview instantly in purple** on the video and timeline while you type, click, drag a rectangle, or move the draft with the arrow keys. Nothing reaches the backend until you apply it. Frame drafts stay staged per frame when you change frames, so you can re-point a run of frames and save them together.
- *Unsaved edits bar* (above the video): counts the staged frames, scene draft and labels, with **Save all** (one backend save for everything staged) and **Discard** (confirmed). Unsaved work **autosaves in this browser** (`localStorage`, key `fishlab:review-draft:<video_id>`) on every change and is restored, with a notice, when the recording is opened again after a reload, crash or switching recordings. A draft made against a different pipeline baseline is dropped. If the browser blocks storage, the bar says so and leaving or reloading warns first. Autosave is per browser: it does not sync between machines.
- *Review decision*: accept/reject, separate from the scientific result. Scene/tracking corrections mark measurements, labels, and predictions out of date; behavior corrections mark measurements and predictions out of date. "Update measurements and behaviors" recomputes reviewed measurements and labels. Compound probabilities stay out of date until "Rerun drug prediction" succeeds with a configured DCS artifact.
- A manual final compound/dose override never changes the automatic probabilities. Model dose estimates are unavailable in the current DCS artifact.

Applied edits, reviewer attribution and review decisions survive refresh in the per-video `review.json`. Each apply/reset waits for server acknowledgment. **Undo all corrections** (Review page, confirmed) restores the automatic baseline and clears staged drafts. Unsaved drafts autosave in this browser until saved or discarded; chat remains in memory.

## Data contract

`src/model.ts` holds the types (`SessionData`, `OverlayWindow`, `DetectionOut`, corrections, chat) and the validation/derivation helpers. `OverlayWindow`/`DetectionOut` preserve prepds's `fps`, coded dimensions, timestamps, nullable x/y, detected flags, detector boxes, and named keypoint arrays. Coordinates are source-video pixels, independent of display scaling; missing detections hide the marker and split trails. A frame without a detector record has `box: null` and no keypoints; nothing is generated in their place. `predictions` is `null` when no model result exists, and `measurements`/`segments` may be empty. Predictions keep DCS's `video_id`, `predicted`, and `p:<class>` columns. Chat matches DCS `/api/ask` (`question`/`history` → `answer`/`tool_calls`/`history`).

## Tests

`npm test` covers correction validation, overlapping labels, duration totals, stale dependencies, immutable automatic predictions, missing-frame trails, absent detector/tracking data, letterbox coordinate mapping, windowed absolute frame alignment and stepping, HTTP/network/timeout error handling, and rendered Review/Dashboard controls and draft separation. The tests build their own small session; no shared sample data exists. `npm run build` type-checks the app.
