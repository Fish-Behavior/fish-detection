# FishLab frontend

React/TypeScript interface for zebrafish behavior analysis: a **Dashboard** for exploring a session and a **Review** page for correcting tracking, scene, and behavior results. Every file lives inside `frontend/`; the Python source and configuration are untouched.

> **The page is empty until the backend provides data.** There is no bundled data. `src/api.ts` is the only integration point: `loadSession()` and `askChat` currently return nothing, so the app shows "No data yet" and hides the chat. Implement those two functions; see [BACKEND_HANDOFF.md](BACKEND_HANDOFF.md).

## Run

Requires Node.js 22.18 or newer and npm. From the repository root:

```sh
cd frontend
npm ci
npm run dev      # http://127.0.0.1:5173
npm test         # model/validation tests (Node's built-in runner)
npm run build    # type-check + static bundle in dist/
npm run preview  # serve the built bundle
```

## Pages

Both pages render only what the session contains: no measurements, labels, predictions, tracking, or detector output means that card (or pipeline stage) shows as unavailable or is hidden.

**Dashboard** — video with tracking overlays (track point, 2 s trail, detector box, keypoints, waterline, ROI), behavior timeline (ethogram), movement traces, behavior segments, compound prediction, time per state, and session details. Playback, the seek slider, numeric time, charts, and segment table share one playhead.

**Review** — the same video and timeline plus correction tools; video and time stay shared with the Dashboard.

- *Review controls*: reviewer name (needed to save, not to preview) and the video interaction tool.
- *Scene settings*: waterline and ROI. *Frame correction*: tracking point, missing-fish flag, detector box, keypoints (only where the detector produced them). *Relabel a time range*: behavior override, under the timeline.
- Edits **preview instantly in purple** on the video and timeline while you type, click, drag a rectangle, or move the draft with the arrow keys (focus the video; Shift = 10 px). Nothing is saved until you apply it. Changing frames discards unsaved frame drafts.
- *Review decision*: accept/reject, separate from the scientific result. Scene/tracking corrections mark measurements, labels, and predictions stale; behavior corrections mark features and predictions stale. "Rerun affected stages" needs the backend and never clears stale flags.
- A manual final compound/dose override never changes the automatic probabilities. Model dose estimates are unavailable in the current DCS artifact.

Edits and reviewer name live in memory only; refresh clears them. **Reset edits** restores the automatic baseline.

## Data contract

`src/model.ts` holds the types (`SessionData`, `OverlayWindow`, `DetectionOut`, corrections, chat) and the validation/derivation helpers. `OverlayWindow`/`DetectionOut` preserve prepds's `fps`, coded dimensions, timestamps, nullable x/y, detected flags, detector boxes, and named keypoint arrays. Coordinates are source-video pixels, independent of display scaling; missing detections hide the marker and split trails. A frame without a detector record has `box: null` and no keypoints; nothing is generated in their place. `predictions` is `null` when no model result exists, and `measurements`/`segments` may be empty. Predictions keep DCS's `video_id`, `predicted`, and `p:<class>` columns. Chat matches DCS `/api/ask` (`question`/`history` → `answer`/`tool_calls`/`history`).

## Tests

`npm test` covers correction validation, overlapping labels, duration totals, stale dependencies, immutable automatic predictions, missing-frame trails, absent detector/tracking data, letterbox coordinate mapping, and the empty backend seam. The tests build their own small session; no shared sample data exists. `npm run build` type-checks the app.
