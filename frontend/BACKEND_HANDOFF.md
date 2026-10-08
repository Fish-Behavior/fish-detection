# Backend connection handoff

The React scaffold is a browser-only preview on `feature/demo-frontend`. It is ready for UI review and has no API, database, or inference connection. The next implementation belongs on a separate backend branch from local `master`, followed by Docker integration on its own branch, as agreed for the first delivery.

## Frontend boundary

`src/model.ts` owns the demo provider and the normalized `DemoSession`, `OverlayWindow`, `DetectionOut`, correction, prediction, and chat shapes. `App.tsx` currently reads one synchronous synthetic session; `Chat.tsx` calls the same provider for a synchronous demo reply. A real adapter needs asynchronous loading, error and unavailable states, a video list/selection, and persistence calls. Keep synthetic responses explicitly labeled, and never substitute them when an API request fails.

The current UI expects a session with a browser-playable `video_url`, duration, positive coded frame width/height, scene, time-aligned overlay arrays, behavior segments, measurement samples, and predictions. Add an unavailable-result state before loading real videos without a model artifact. Source-frame pixel coordinates are used for overlays and correction validation. `frameAt()` currently uses synthetic boxes/keypoints when the demo has no detector record; a real adapter must not use that fallback for actual videos. Missing detections and missing detector/keypoint data need explicit absent states. Validate probabilities as finite values from 0 to 1; do not manufacture probabilities, confidence scores, measurements, or dose estimates.

`OverlayWindow` in the existing prepds API permits `width` and `height` to be null. Its `/overlay` endpoint is limited to windows of at most 30 seconds. The adapter must load bounded windows around playback, keep their timestamps and frame indices aligned, and obtain/verify coded dimensions before allowing coordinate edits. Video playback needs a browser-reachable URL with range requests. Preserve the automatic baseline separately from manual corrections and identify stale derived outputs after upstream edits.

## Existing services and gaps

| Need | Available source | Backend work needed |
| --- | --- | --- |
| Processed video list and detail | prepds `GET /videos`, `GET /videos/{video_id}` on local `master` | Build frontend session/list responses; `GET /videos` only includes processed manifests. |
| Source video and tracking | prepds `GET /videos/{video_id}/video`, `GET /videos/{video_id}/overlay` | Map nullable dimensions and windowed overlay data; expose absence and processing errors. |
| Waterline | prepds `GET/PUT/DELETE /videos/{video_id}/waterline` | Integrate with the durable correction record and downstream stale state. |
| Behavior review | prepds `POST /videos/{video_id}/edits`, `/accept`, `/reject` | Reconcile validation and review transitions with this UI. Existing edit API accepts behavior ranges only. |
| ROI, frame point, missing-fish flag, detector box, keypoints, manual final result | In-memory frontend demo only | Design and persist correction endpoints and original/manual provenance in PostgreSQL. Recompute or mark dependent outputs stale. |
| Existing video folder | prepds catalog scans MP4 files; current `/videos` lists processed manifests | Configure/mount the existing folder once, inventory available recordings, identify unmatched/unprocessed files, and provide processing status. No per-video upload should be required. |
| Compound probabilities | DCS `predict` on local `feature/dcs-wrapup` writes `video_id`, `predicted`, and `p:<class>` columns | Expose predictions and model/run provenance after an artifact is available. Show unavailable when no valid result exists. Current DCS does not supply a usable model dose estimate. |
| Research chat | DCS `POST /api/ask` on local `feature/dcs-wrapup` accepts `{question, history}` and returns `{answer, tool_calls, history}` | Connect with loading/error handling and preserve the answer-only UI. The preview replies are illustrative. |

## Backend acceptance checks

1. A configured video folder is inventoried without one-by-one upload. Processed and unprocessed recordings are distinguished; selecting a video keeps playback and charts synchronized.
2. For an actual video, source dimensions and overlay timestamps align at multiple display sizes. Gaps hide points and split trails; absent detector data never becomes synthetic geometry.
3. Server validation rejects out-of-frame coordinates, invalid time ranges, and blank reviewers. Saved corrections survive refresh and retain the automatic baseline and reviewer provenance.
4. Scene/tracking edits invalidate affected measurements, automatic labels, and predictions. Behavior relabeling updates the reviewed ethogram/totals and invalidates affected features/predictions. Manual final compound/dose remains separate from model probabilities.
5. Missing artifacts, processing failures, unavailable predictions, and chat errors are visible as such; the UI never presents demo values as real results.

No API path is specified here for the missing capabilities. Define those paths and PostgreSQL schema with the backend work, then replace the provider calls and add contract tests against the implemented responses.
