# FishLab frontend demo

Interactive React/TypeScript scaffold based on `demo-pipeline`'s live and scene-review pages. This branch is based on local `master`; every addition is inside `frontend/`. Existing Python source and configuration are untouched.

## Start the preview

Requires Node.js 22.18 or newer and npm. From the repository root:

```sh
cd frontend
npm ci
npm run dev
```

Open the localhost URL printed by Vite (normally `http://127.0.0.1:5173`).

```sh
npm test
npm run build
npm run preview
```

## Try the workflow

1. Play the synthetic clip, change playback speed, or seek using the slider, numeric time, charts, or segment table.
2. Switch to **Review & corrections**. The selected video and time remain shared.
3. Enter a reviewer name. Choose a video interaction to draw a point, keypoint, waterline, ROI, or detector box. Numeric fields provide keyboard alternatives. Apply the draft in its corresponding panel.
4. Correct a behavior time range. The reviewed timeline and time totals update. Later overlapping edits win.
5. Expand **Override final drug / dose** to record a manual decision. It does not alter the automatic compound-probability bars. Model dose prediction is unavailable in the current DCS artifact.
6. Accept/reject the review separately from the scientific result. Scene/tracking corrections mark automatic labels, measurements, and predictions stale. Behavior corrections mark features/predictions stale. The demo's rerun button explains that a backend is required and never clears stale results.
7. Open the fixed bottom-right chat bubble. Enter opens it and Escape closes the popup. Replies are explicitly mocked and cannot perform actions.
8. Choose a local video. It plays via a browser object URL; no file is uploaded or persisted. Synthetic analysis is hidden for picked files. Returning to the demo/replacing the file/unmounting revokes the object URL.

Edits, reviewer name, and chat exist in memory only. Refresh clears them. **Reset demo** restores baseline corrections; **Clear chat** resets chat separately. Changing frames discards unsaved frame drafts. Automatic values stay available for comparison, and restore controls remove the corresponding manual corrections.

## Future integration

The [backend handoff](BACKEND_HANDOFF.md) maps the preview data to existing prepds/DCS services, identifies missing APIs and normalization rules, and lists acceptance checks for connecting real videos.

`src/model.ts` contains the typed demo provider and correction model. `OverlayWindow`/`DetectionOut` preserve prepds's `fps`, coded dimensions, timestamps, nullable x/y, detected flags, detector boxes, and named keypoint arrays. Missing detections hide the marker and split trails. Coordinates are source-video pixels, independent of display scaling.

Prediction fixtures preserve DCS's `video_id`, `predicted`, and `p:<class>` columns. Chat input/output matches the existing DCS `/api/ask` question/history and answer/tool_calls/history shape. This preview performs no API calls, real tracking, feature extraction, training, or inference. All figures are illustrative and no accuracy is claimed.

After UI review, implement the API adapter/backend on a separate branch from `master`, importing prepds/DCS after their merges and using PostgreSQL for durable corrections/results. Docker Compose and the single shell launcher belong to a separate integration branch. No database or Docker dependency is needed for this preview.

The preferred video-library workflow is to configure the existing video folder once and have the backend recursively inventory supported videos, rather than require one-by-one uploads. Reuse prepds's catalog and processed-video manifests; distinguish available recordings from processed results. The browser-only preview cannot scan that folder and currently offers one synthetic session or one locally picked file. Folder discovery and batch analysis require the later backend integration.

## Synthetic asset

`public/demo-tank.mp4` is a generated fish illustration, not a lab recording. It is H.264, 640 × 360, 30 fps, 12 seconds. Its position and orientation match the deterministic sample overlay trajectory in `src/model.ts`. A short deliberate detection gap demonstrates missing-point handling. Sample boxes/keypoint confidences, ethogram, turning values, and probabilities are illustrative.

To regenerate with the existing repository Python environment (OpenCV/NumPy and an H.264 encoder), run from the repository root:

```sh
.venv/bin/python frontend/scripts/generate_demo.py
```

## Checks

`npm test` uses Node's built-in test runner for correction validation, overlapping labels, duration totals, stale dependencies, immutable automatic predictions, missing-frame trails, letterbox coordinate mapping, and chat shape. `npm run build` type-checks the application and creates the static bundle. Browser checks cover playback, review interaction, local-file cleanup, fixed chat positioning/focus, and responsive layouts.
