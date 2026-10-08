# Backend connection handoff

The React frontend is complete as a UI but has no API, database, or inference connection, and it ships with no data: until the backend supplies a session it shows only "No data yet" and hides the chat. The next implementation belongs on a separate backend branch from local `master`, followed by Docker integration on its own branch, as agreed for the first delivery.

## Frontend boundary

`src/api.ts` is the only integration point and has two stubs to replace:

- `loadSession(): Promise<SessionData | null>` — resolve `null` when there is no session, reject on failure (the UI shows the error and never substitutes other data).
- `askChat: ((q: ChatQuestion) => Promise<ChatAnswer>) | null` — leave `null` to hide the chat bubble; set it to the DCS `/api/ask` adapter to show it. Rejections appear inside the chat.

`src/model.ts` owns the normalized `SessionData`, `OverlayWindow`, `DetectionOut`, correction, prediction, and chat types. `App.tsx` calls `loadSession()` once on startup; a video list/selection, refresh, and persistence calls still need to be added (the sidebar currently shows the one loaded session).

The current UI expects a session with a browser-playable `video_url`, duration, positive coded frame width/height, scene, time-aligned overlay arrays, behavior segments, measurement samples, and predictions. Source-frame pixel coordinates are used for overlays and correction validation. Absent data is explicit: a frame with no detector record has `box: null` and no keypoints (`frameAt()` never generates geometry); `predictions` is `null` when no model result exists; `measurements`, `segments`, and the overlay arrays may be empty, and the UI then hides or marks the affected cards and pipeline stages as unavailable. Validate probabilities as finite values from 0 to 1; do not manufacture probabilities, confidence scores, measurements, or dose estimates.

`OverlayWindow` in the existing prepds API permits `width` and `height` to be null. Its `/overlay` endpoint is limited to windows of at most 30 seconds. The adapter must load bounded windows around playback, keep their timestamps and frame indices aligned, and obtain/verify coded dimensions before allowing coordinate edits. Video playback needs a browser-reachable URL with range requests. Preserve the automatic baseline separately from manual corrections and identify stale derived outputs after upstream edits.

## Existing services and gaps

| Need | Available source | Backend work needed |
| --- | --- | --- |
| Processed video list and detail | prepds `GET /videos`, `GET /videos/{video_id}` on local `master` | Build frontend session/list responses; `GET /videos` only includes processed manifests. |
| Source video and tracking | prepds `GET /videos/{video_id}/video`, `GET /videos/{video_id}/overlay` | Map nullable dimensions and windowed overlay data; expose absence and processing errors. |
| Waterline | prepds `GET/PUT/DELETE /videos/{video_id}/waterline` | Integrate with the durable correction record and downstream stale state. |
| Behavior review | prepds `POST /videos/{video_id}/edits`, `/accept`, `/reject` | Reconcile validation and review transitions with this UI. Existing edit API accepts behavior ranges only. |
| ROI, frame point, missing-fish flag, detector box, keypoints, manual final result | In-memory in the frontend only (no endpoints yet) | Design and persist correction endpoints and original/manual provenance in PostgreSQL. Recompute or mark dependent outputs stale. |
| Existing video folder | prepds catalog scans MP4 files; current `/videos` lists processed manifests | Configure/mount the existing folder once, inventory available recordings, identify unmatched/unprocessed files, and provide processing status. No per-video upload should be required. |
| Compound probabilities | DCS `predict` on local `feature/dcs-wrapup` writes `video_id`, `predicted`, and `p:<class>` columns | Expose predictions and model/run provenance after an artifact is available. Show unavailable when no valid result exists. Current DCS does not supply a usable model dose estimate. |
| Research chat | DCS `POST /api/ask` on local `feature/dcs-wrapup` accepts `{question, history}` and returns `{answer, tool_calls, history}` | Connect with loading/error handling and preserve the answer-only UI. The chat is hidden until `askChat` is set. |

## Backend acceptance checks

1. A configured video folder is inventoried without one-by-one upload. Processed and unprocessed recordings are distinguished; selecting a video keeps playback and charts synchronized.
2. For an actual video, source dimensions and overlay timestamps align at multiple display sizes. Gaps hide points and split trails; absent detector data never becomes generated geometry.
3. Server validation rejects out-of-frame coordinates, invalid time ranges, and blank reviewers. Saved corrections survive refresh and retain the automatic baseline and reviewer provenance.
4. Scene/tracking edits invalidate affected measurements, automatic labels, and predictions. Behavior relabeling updates the reviewed ethogram/totals and invalidates affected features/predictions. Manual final compound/dose remains separate from model probabilities.
5. Missing artifacts, processing failures, unavailable predictions, and chat errors are visible as such; the UI renders only what the API returns.

No API path is specified here for the missing capabilities. Define those paths and PostgreSQL schema with the backend work, then replace the provider calls and add contract tests against the implemented responses.
