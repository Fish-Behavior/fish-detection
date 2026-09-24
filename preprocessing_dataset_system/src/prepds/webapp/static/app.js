"use strict";
// Review UI: all server data is inserted with textContent / style properties, never innerHTML.
const STATES = [
  ["Controlled Swim", "--s-controlled"], ["Erratic Movement", "--s-erratic"], ["Freezing/Drift", "--s-freezing"],
  ["Listing/LORR", "--s-listing"], ["Surface Breach", "--s-surface"], ["Dead", "--s-dead"],
  ["Undetermined", "--s-undetermined"],
];
// Undetermined is a pipeline marker: reviewers resolve it, they cannot assign it.
const ASSIGNABLE = STATES.filter(([name]) => name !== "Undetermined");
const $ = (id) => document.getElementById(id);
const colorOf = (state) => `var(${(STATES.find(([n]) => n === state) || [, "--muted"])[1]})`;
const fmt = (s) => `${s.toFixed(2)} s`;

const EDGE_SNAP_PX = 4;
const app = { busy: false, listSeq: 0, openSeq: 0, detail: null, videoId: null, selection: null, pending: [], chosen: null, drag: null };

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  node.append(...children);
  return node;
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (response.ok) return response.json();
  let body = null;
  try { body = await response.json(); } catch (_) { /* non-JSON error */ }
  const detail = body && body.detail;
  const error = new Error(typeof detail === "string" ? detail : (detail && detail.message) || response.statusText);
  error.code = detail && detail.code;
  error.status = response.status;
  if (Array.isArray(detail)) error.message = detail.map((d) => d.msg).join("; ");
  throw error;
}

function say(text, isError = false) {
  const node = $("message");
  node.textContent = text;
  node.classList.toggle("error", isError);
}

function duration() { return app.detail ? app.detail.manifest.video_duration_s : 0; }
function reviewer() { return $("reviewer").value.trim(); }
function locked() { return !app.detail || ["ACCEPTED", "REJECTED"].includes(app.detail.manifest.review_status); }

// ---------- list ----------
async function loadList() {
  const status = $("status-filter").value;
  const seq = ++app.listSeq;
  const rows = await api(`/videos${status ? `?status=${status}` : ""}`);
  if (seq !== app.listSeq) return;  // a newer request superseded this one
  const list = $("video-list");
  list.replaceChildren();
  $("list-empty").hidden = rows.length > 0;
  for (const row of rows) {
    const button = el("button", { type: "button" },
      row.subject_id, el("span", { className: "badge", textContent: row.review_status }),
      el("span", { className: "sub", textContent: `${row.compound} ${row.concentration_mM} mM · ${row.sex}${row.flag_count ? ` · ${row.flag_count} flag(s)` : ""}` }));
    button.dataset.id = row.video_id;
    if (row.video_id === app.videoId) button.setAttribute("aria-current", "true");
    button.addEventListener("click", () => openVideo(row.video_id));
    list.append(el("li", {}, button));
  }
}

async function openVideo(id) {
  if (app.busy) return;
  const seq = ++app.openSeq;
  const detail = await api(`/videos/${encodeURIComponent(id)}`).catch((e) => { say(`Cannot open: ${e.message}`, true); return null; });
  if (!detail || seq !== app.openSeq) return;  // failed, or a later click won
  app.videoId = id;
  app.detail = detail;
  app.pending = [];
  app.selection = null;
  say("");
  render();
  const player = $("player");
  player.src = app.detail.video_url;
  resetOverlay();
  player.load();
  await loadList();
}

async function refresh() {
  app.detail = await api(`/videos/${encodeURIComponent(app.videoId)}`);
  render();
}

// ---------- render ----------
function render() {
  const d = app.detail, m = d.manifest;
  $("review").hidden = false;
  $("placeholder").hidden = true;
  $("title").textContent = m.subject_id;
  $("subtitle").textContent = `${m.compound} ${m.concentration_mM} mM · ${m.sex} · ${m.review_status}` +
    (m.reviewer ? ` by ${m.reviewer}` : "") + ` · ${m.edit_count} edit(s) · profile ${m.calibration_profile_version}`;
  $("axis-end").textContent = `${duration().toFixed(1)} s`;
  $("flags").replaceChildren(...m.review_flags.map((f) =>
    el("li", { textContent: `Check ${f.start_s.toFixed(0)}–${f.end_s.toFixed(0)} s: ${f.message}` })));
  renderListingFlags(d);
  requestWhy(true);
  renderSegments();
  renderSummary();
  renderPicker();
  renderPending();
  renderSelection();
  const isLocked = locked();
  $("accept").disabled = isLocked;
  $("reject").disabled = m.review_status === "REJECTED" || m.review_status === "ACCEPTED";
  $("timeline").style.cursor = isLocked ? "default" : "crosshair";
}

function renderListingFlags(d) {
  const seek = (t) => { $("player").currentTime = t; };
  const rows = d.listing_flags.map((f) => el("li", { textContent: `Possible Listing ${f.start_s.toFixed(0)}–${f.end_s.toFixed(0)} s (model score ${f.max_score.toFixed(2)}; a hint, please check)` },
    el("button", { type: "button", textContent: "Go to", onclick: () => seek(f.start_s) })));
  if (d.listing_flags_error) rows.push(el("li", { textContent: `Listing hints unavailable: ${d.listing_flags_error}` }));
  $("listing-flags").replaceChildren(...rows);
  $("listing-layer").replaceChildren(...d.listing_flags.map((f) => {
    const node = el("div", { className: "lst" });
    node.style.left = xPct(f.start_s);
    node.style.width = `${Math.min(100 - parseFloat(node.style.left), ((f.end_s - f.start_s) / (duration() || 1)) * 100)}%`;
    node.title = `Possible Listing ${fmt(f.start_s)} – ${fmt(f.end_s)}`;
    return node;
  }));
}

const xPct = (s) => `${Math.min(100, Math.max(0, (s / (duration() || 1)) * 100))}%`;
function placeBlock(node, start, end, state) {
  node.style.left = xPct(start);
  node.style.width = `${((end - start) / duration()) * 100}%`;
  node.style.background = colorOf(state);
  node.title = `${state}: ${fmt(start)} – ${fmt(end)}`;
  return node;
}

function renderSegments() {
  $("segments").replaceChildren(...app.detail.segments.map((s) =>
    placeBlock(el("div", { className: "seg" }), s.start_s, s.end_s, s.state)));
  $("pending-layer").replaceChildren(...app.pending.map((p) =>
    placeBlock(el("div", { className: "pend" }), p.start_s, p.end_s, p.new_state)));
}

function renderSummary() {
  const s = app.detail.frame_summary;
  const total = s.total_s || 0;
  $("summary-total").textContent = total ? `of ${total.toFixed(0)} s in this video` : "";
  $("summary").replaceChildren(...STATES.map(([name]) => {
    const sec = s.seconds_by_state[name] || 0, bouts = s.bouts_by_state[name] || 0;
    const pct = total ? (sec / total) * 100 : 0;
    const bar = el("span", {});
    bar.style.width = `${pct}%`;
    bar.style.background = colorOf(name);
    return el("li", { className: sec ? "" : "none" },
      el("span", { className: "swatch", style: `background:${colorOf(name)}` }),
      el("span", { textContent: name }),
      el("span", { className: "bar" }, bar),
      el("span", { className: "num", textContent: `${sec.toFixed(1)} s` }),
      el("span", { className: "num pct", textContent: `${pct.toFixed(1)}%` }),
      el("span", { className: "num bouts", textContent: `${bouts} bout${bouts === 1 ? "" : "s"}` }));
  }));
}

// ---------- why this label ----------
const why = { second: null, video: null, seq: 0, timer: null };

function requestWhy(force = false) {
  if (!app.videoId) return;
  const second = Math.max(0, Math.floor($("player").currentTime || 0));
  if (!force && second === why.second && why.video === app.videoId) return;
  why.second = second; why.video = app.videoId;
  clearTimeout(why.timer);
  why.timer = setTimeout(() => loadWhy(app.videoId, second), 150);
}

async function loadWhy(videoId, second) {
  const seq = ++why.seq;
  try {
    const data = await api(`/videos/${encodeURIComponent(videoId)}/explain?second=${second}`);
    if (seq === why.seq) renderWhy(data);
  } catch (error) {
    if (seq !== why.seq) return;
    $("why-second").textContent = `${second}–${second + 1} s`;
    $("why-summary").textContent = "Not available for this second.";
    $("why-summary").className = "why-summary";
    $("why-rules").replaceChildren();
  }
}

function renderWhy(d) {
  $("why-second").textContent = `${d.start_s.toFixed(0)}–${d.end_s.toFixed(0)} s`;
  const parts = [`Label: ${d.stored_state}${d.stored_source === "manual" ? " (set by a reviewer)" : ""}.`,
    `${d.n_detected} of ${d.n_frames} frames have the fish detected.`];
  if (d.speed_median_px_per_s !== null) parts.push(`Median speed ${d.speed_median_px_per_s.toFixed(1)} px/s.`);
  const votes = Object.entries(d.votes).sort((a, b) => b[1] - a[1]).map(([s, n]) => `${n} ${s}`).join(", ");
  if (votes) parts.push(`Frame votes: ${votes}.`);
  let changed = false;
  if (d.auto_state && d.stored_state !== d.auto_state) {
    changed = true;
    parts.push(d.stored_source === "manual"
      ? `The automatic label was ${d.auto_state}.`
      : `Recomputed with the profile thresholds this second would be ${d.auto_state}, not ${d.stored_state}: the profile file may have changed since the run, or the second sits right on a threshold (the stored track is rounded).`);
  }
  if (!d.thresholds_available) parts.push(`Rules unavailable: ${d.thresholds_error || "no thresholds"}.`);
  $("why-summary").textContent = parts.join(" ");
  $("why-summary").className = changed ? "why-summary changed" : "why-summary";
  $("why-rules").replaceChildren(...d.rules.map((r) => el("li", { className: r.wins ? "wins" : "" },
    el("span", { className: "mark", textContent: r.wins ? "✓" : "–" }),
    el("span", { className: "frames", textContent: r.frames ? String(r.frames) : "" }),
    el("span", { textContent: `${r.state}: ${r.detail}` }))));
}

function renderPicker() {
  $("state-buttons").replaceChildren(...ASSIGNABLE.map(([name]) => {
    const b = el("button", { type: "button" }, el("span", { className: "swatch", style: `background:${colorOf(name)}` }), name);
    b.setAttribute("aria-pressed", String(app.chosen === name));
    b.addEventListener("click", () => { app.chosen = name; renderPicker(); updateButtons(); });
    return b;
  }));
  updateButtons();
}

function renderPending() {
  $("pending-list").replaceChildren(...app.pending.map((p, i) => {
    const remove = el("button", { type: "button", textContent: "remove" });
    remove.addEventListener("click", () => { app.pending.splice(i, 1); renderSegments(); renderPending(); });
    return el("li", {}, el("span", { className: "swatch", style: `background:${colorOf(p.new_state)}` }),
      `${fmt(p.start_s)} – ${fmt(p.end_s)} → ${p.new_state}`, remove);
  }));
  updateButtons();
}

function renderSelection() {
  const node = $("selection");
  if (!app.selection) {
    node.hidden = true;
    $("selection-info").textContent = "Drag on the timeline to select a time range.";
  } else {
    node.hidden = false;
    node.style.left = xPct(app.selection.start_s);
    node.style.width = `${((app.selection.end_s - app.selection.start_s) / duration()) * 100}%`;
    $("selection-info").textContent = `Selected ${fmt(app.selection.start_s)} – ${fmt(app.selection.end_s)}`;
  }
  updateButtons();
}

function updateButtons() {
  const open = !locked();
  $("stage").disabled = !(open && app.selection && app.chosen);
  $("save").disabled = !(open && app.pending.length);
  $("discard").disabled = !app.pending.length;
}

// ---------- timeline interaction ----------
function timeAt(event) {
  const rect = $("timeline").getBoundingClientRect();
  const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
  const snap = EDGE_SNAP_PX / rect.width;  // the last pixel must be reachable, or a tail stays unlabeled
  return (fraction < snap ? 0 : fraction > 1 - snap ? 1 : fraction) * duration();
}

function setSelection(a, b) {
  const [start_s, end_s] = a <= b ? [a, b] : [b, a];
  app.selection = end_s - start_s > 0.01 ? { start_s, end_s } : null;
  renderSelection();
}

const timeline = $("timeline");
timeline.addEventListener("pointerdown", (event) => {
  if (!app.detail) return;
  timeline.setPointerCapture(event.pointerId);
  const t = timeAt(event);
  app.drag = { origin: t, moved: false };
});
timeline.addEventListener("pointermove", (event) => {
  if (!app.drag || locked()) return;
  const t = timeAt(event);
  if (Math.abs(t - app.drag.origin) > 0.01 * duration() / 10) app.drag.moved = true;
  if (app.drag.moved) setSelection(app.drag.origin, t);
});
timeline.addEventListener("pointerup", (event) => {
  if (!app.drag) return;
  if (!app.drag.moved) { $("player").currentTime = timeAt(event); }  // a plain click seeks
  app.drag = null;
});
timeline.addEventListener("pointercancel", () => { app.drag = null; });
timeline.addEventListener("keydown", (event) => {
  const player = $("player");
  const step = event.shiftKey ? 10 : 1;
  if (event.key === "ArrowRight") player.currentTime = Math.min(duration(), player.currentTime + step);
  else if (event.key === "ArrowLeft") player.currentTime = Math.max(0, player.currentTime - step);
  else if (event.key === "[") app.selection = { start_s: player.currentTime, end_s: Math.max(player.currentTime + 0.5, app.selection ? app.selection.end_s : 0) };
  else if (event.key === "]") app.selection = { start_s: Math.min(app.selection ? app.selection.start_s : 0, player.currentTime - 0.5), end_s: player.currentTime };
  else return;
  if (app.selection && app.selection.start_s < 0) app.selection.start_s = 0;
  if (app.selection && app.selection.end_s > duration()) app.selection.end_s = duration();
  event.preventDefault();
  renderSelection();
});

function movePlayhead() {
  if (!app.detail) return;
  const t = $("player").currentTime || 0;
  $("playhead").style.left = xPct(t);
  timeline.setAttribute("aria-valuenow", t.toFixed(1));
  const seg = app.detail.segments.find((x) => t >= x.start_s && t < x.end_s);
  timeline.setAttribute("aria-valuetext", `${t.toFixed(1)} s${seg ? `, ${seg.state}` : ""}`);
  timeline.setAttribute("aria-valuemax", String(duration()));
}
$("player").addEventListener("timeupdate", () => { movePlayhead(); requestWhy(); });
$("player").addEventListener("seeked", () => { movePlayhead(); requestWhy(); });

// ---------- actions ----------
$("stage").addEventListener("click", () => {
  app.pending.push({ ...app.selection, new_state: app.chosen });
  app.selection = null;
  renderSegments(); renderPending(); renderSelection();
});
$("discard").addEventListener("click", () => { app.pending = []; renderSegments(); renderPending(); say("Staged edits discarded."); });

function needReviewer() {
  if (reviewer()) return true;
  say("Enter your name first.", true);
  $("reviewer").focus();
  return false;
}

async function post(action, body) {
  return api(`/videos/${encodeURIComponent(app.videoId)}/${action}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
}

// One mutating request at a time, bound to the video it was started for.
async function mutate(action, body, done, failed) {
  if (app.busy) return;
  const id = app.videoId;
  app.busy = true;
  setBusy(true);
  let result = null;
  try {
    result = await post(action, body);
  } catch (error) {
    if (id === app.videoId) {
      say(failed(error), true);
      if (error.status === 409 || error.status === 422) await refresh().catch(() => {});  // the server's view wins
    }
    return null;
  } finally {
    app.busy = false;
    setBusy(false);
  }
  if (id === app.videoId) { app.detail = result; app.pending = []; render(); say(done); }
  try { await loadList(); } catch (error) { say(`${done} (list refresh failed: ${error.message})`, true); }
  return result;
}

function setBusy(on) {
  for (const id of ["save", "accept", "reject", "stage", "discard"]) if (on) $(id).disabled = true;
  if (!on && app.detail) render();
}

$("save").addEventListener("click", () => {
  if (!needReviewer()) return;
  return mutate("edits", { reviewer: reviewer(), edits: app.pending }, "Edits saved.", (e) => `Not saved: ${e.message}`);
});

$("accept").addEventListener("click", async () => {
  if (!needReviewer()) return;
  if (app.pending.length) { say("Save or discard staged edits before accepting.", true); return; }
  let needsConfirm = false;
  const first = await mutate("accept", { reviewer: reviewer(), force: false }, "Accepted.", (e) => {
    if (e.code === "dead_confirmation_required") { needsConfirm = true; return "Dead segments present: confirmation needed."; }
    if (e.code === "undetermined_present") return "Cannot accept: Undetermined segments remain. Relabel them first.";
    return `Not accepted: ${e.message}`;
  });
  if (first || !needsConfirm) return;
  if (window.confirm("This video contains Dead segments. Accept anyway?")) {
    await mutate("accept", { reviewer: reviewer(), force: true }, "Accepted.", (e) => `Not accepted: ${e.message}`);
  } else say("Accept cancelled.");
});

$("reject").addEventListener("click", () => {
  if (!needReviewer() || !window.confirm("Reject this video? Its edits are cleared.")) return;
  return mutate("reject", { reviewer: reviewer() }, "Rejected.", (e) => `Not rejected: ${e.message}`);
});

$("status-filter").addEventListener("change", () => loadList().catch((e) => say(e.message, true)));
try { $("reviewer").value = localStorage.getItem("prepds.reviewer") || ""; } catch (_) { /* storage blocked */ }
$("reviewer").addEventListener("change", () => { try { localStorage.setItem("prepds.reviewer", reviewer()); } catch (_) { /* ignore */ } });
loadList().catch((e) => say(e.message, true));


// ---------- fish markers on the video ----------
const OVERLAY_CHUNK_S = 20;
const OVERLAY_PATH_S = 2;
const KEYPOINT_COLORS = { snout: "#e0e", dorsal_fin_base: "#0bb", ventral: "#0bb", tail_base: "#0a0", tail_tip: "#0a0" };
const overlay = { video: null, chunks: new Map(), inflight: new Set(), message: "", waterline: null, picking: false };

function resetOverlay() {
  overlay.video = app.videoId;
  overlay.chunks.clear();
  overlay.inflight.clear();
  overlay.message = "";
  overlay.waterline = null;
  setPicking(false);
  $("water-clear").hidden = true;
  $("overlay-note").textContent = "";
  drawOverlay();
  loadWaterline();
}

async function ensureOverlayChunk(index) {
  const key = `${overlay.video}:${index}`;
  if (index < 0 || overlay.chunks.has(key) || overlay.inflight.has(key)) return;
  overlay.inflight.add(key);
  const videoId = overlay.video;
  try {
    const data = await api(`/videos/${encodeURIComponent(videoId)}/overlay?start_s=${index * OVERLAY_CHUNK_S}&end_s=${(index + 1) * OVERLAY_CHUNK_S}`);
    if (videoId === overlay.video) {
      overlay.chunks.set(key, data);
      if (data.detections_error) overlay.message = data.detections_error;
      $("overlay-note").textContent = overlay.message || (data.has_detector ? "" :
        "This run has no detector output (classical tracker): only the track point and path exist. Box and keypoints need a model run (outputs_r3).");
      drawOverlay();
    }
  } catch (error) {
    overlay.message = "Fish markers unavailable.";
  } finally {
    overlay.inflight.delete(key);
  }
}

function nearestIndex(times, t) {
  let lo = 0, hi = times.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (times[mid] < t) lo = mid + 1; else hi = mid; }
  if (lo > 0 && Math.abs(times[lo - 1] - t) < Math.abs(times[lo] - t)) lo -= 1;
  return lo;
}

function drawOverlay() {
  const video = $("player"), canvas = $("overlay");
  if (!video || !overlay.video || !video.videoWidth) return;
  const w = video.clientWidth, h = video.clientHeight, dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  // marker coordinates are in the decoded frame's pixels; the browser may show a non-square pixel aspect, so
  // each axis is scaled from the coded size when the server knows it
  const first = overlay.chunks.get(`${overlay.video}:0`) || [...overlay.chunks.values()][0];
  const codedW = (first && first.width) || video.videoWidth, codedH = (first && first.height) || video.videoHeight;
  const sx = w / codedW, sy = h / codedH, t = video.currentTime || 0;
  const index = Math.floor(t / OVERLAY_CHUNK_S);
  ensureOverlayChunk(index);
  if (t - index * OVERLAY_CHUNK_S > OVERLAY_CHUNK_S - 4) ensureOverlayChunk(index + 1);
  if (overlay.waterline !== null && $("ov-water").checked) {
    const y = overlay.waterline * sy;
    ctx.strokeStyle = "#2a7fff"; ctx.lineWidth = 2; ctx.setLineDash([8, 4]);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#2a7fff"; ctx.font = "11px monospace"; ctx.fillText(`waterline y=${Math.round(overlay.waterline)}`, 6, Math.max(12, y - 4));
  }
  const chunk = overlay.chunks.get(`${overlay.video}:${index}`);
  if (!chunk || !chunk.t.length) return;
  const tolerance = 1.5 / (chunk.fps || 30);
  const i = nearestIndex(chunk.t, t);
  const near = Math.abs(chunk.t[i] - t) <= tolerance;
  if ($("ov-path").checked) {
    ctx.strokeStyle = "#e80"; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    for (let k = i; k >= 0 && chunk.t[k] >= t - OVERLAY_PATH_S; k--) {
      if (chunk.x[k] === null) { started = false; continue; }
      const px = chunk.x[k] * sx, py = chunk.y[k] * sy;
      if (started) ctx.lineTo(px, py); else { ctx.moveTo(px, py); started = true; }
    }
    ctx.stroke();
  }
  if (near && $("ov-track").checked && chunk.x[i] !== null) {
    ctx.fillStyle = "#e11"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(chunk.x[i] * sx, chunk.y[i] * sy, 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }
  if (near && chunk.x[i] === null) {
    ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(4, 4, 96, 16);
    ctx.fillStyle = "#fff"; ctx.font = "11px monospace"; ctx.fillText("fish not found", 8, 16);
  }
  const samples = chunk.detections;
  if (samples.length && near) {
    const times = samples.map((d) => d.t), j = nearestIndex(times, t);
    if (Math.abs(times[j] - t) <= 3 / (chunk.fps || 30)) {
      const d = samples[j];
      if ($("ov-box").checked) {
        ctx.strokeStyle = "#e8b800"; ctx.lineWidth = 1.5;
        ctx.strokeRect(d.box[0] * sx, d.box[1] * sy, (d.box[2] - d.box[0]) * sx, (d.box[3] - d.box[1]) * sy);
      }
      if ($("ov-kp").checked) {
        const kp = d.keypoints, line = (a, b) => { ctx.beginPath(); ctx.moveTo(kp[a][0] * sx, kp[a][1] * sy); ctx.lineTo(kp[b][0] * sx, kp[b][1] * sy); ctx.stroke(); };
        ctx.strokeStyle = "rgba(0,170,170,.8)"; ctx.lineWidth = 1.5; line("snout", "tail_base"); line("tail_base", "tail_tip"); line("dorsal_fin_base", "ventral");
        for (const [name, [x, y]] of Object.entries(kp)) {
          ctx.fillStyle = KEYPOINT_COLORS[name] || "#fff"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(x * sx, y * sy, name === "snout" ? 4 : 3, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        }
      }
    }
  }
}

(function wireOverlay() {
  const video = $("player");
  const schedule = () => { if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(loop); else requestAnimationFrame(loop); };
  const loop = () => { drawOverlay(); if (!video.paused && !video.ended) schedule(); };
  video.addEventListener("play", schedule);
  for (const type of ["pause", "seeked", "timeupdate", "loadedmetadata", "loadeddata"]) video.addEventListener(type, drawOverlay);
  window.addEventListener("resize", drawOverlay);
  for (const id of ["ov-track", "ov-path", "ov-box", "ov-kp", "ov-water"]) $(id).addEventListener("change", drawOverlay);
})();


// ---------- waterline (manual, advisory: no labeling rule reads it) ----------
function codedSize() {
  const first = [...overlay.chunks.values()][0], video = $("player");
  return { w: (first && first.width) || video.videoWidth, h: (first && first.height) || video.videoHeight };
}

function setPicking(on) {
  overlay.picking = on;
  document.querySelector(".video-wrap").classList.toggle("picking", on);
  $("water-set").textContent = on ? "Click the water surface..." : "Set waterline";
}

async function loadWaterline() {
  const videoId = overlay.video;
  try {
    const data = await api(`/videos/${encodeURIComponent(videoId)}/waterline`);
    if (videoId !== overlay.video) return;
    overlay.waterline = data.y_px;
    $("water-clear").hidden = data.y_px === null;
    drawOverlay();
  } catch (error) { /* the line is optional; the review works without it */ }
}

async function saveWaterline(y) {
  const videoId = overlay.video;
  try {
    const data = await api(`/videos/${encodeURIComponent(videoId)}/waterline`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ y_px: y }),
    });
    if (videoId !== overlay.video) return;
    overlay.waterline = data.y_px; $("water-clear").hidden = false; $("ov-water").checked = true; drawOverlay();
  } catch (error) { say(`Waterline not saved: ${error.message}`, true); }
}

$("water-set").addEventListener("click", () => { setPicking(!overlay.picking); });
$("water-clear").addEventListener("click", async () => {
  const videoId = overlay.video;
  try {
    await api(`/videos/${encodeURIComponent(videoId)}/waterline`, { method: "DELETE" });
    if (videoId === overlay.video) { overlay.waterline = null; $("water-clear").hidden = true; drawOverlay(); }
  } catch (error) { say(`Waterline not cleared: ${error.message}`, true); }
});

// ---------- playback controls (kept outside the picture so nothing covers the fish) ----------
const playerEl = $("player");
const fmtClock = (t) => (Number.isFinite(t) ? t : 0).toFixed(1);
function syncControls() {
  const d = Number.isFinite(playerEl.duration) ? playerEl.duration : 0;
  $("seek").max = String(d);
  if (document.activeElement !== $("seek")) $("seek").value = String(playerEl.currentTime || 0);
  $("clock").textContent = `${fmtClock(playerEl.currentTime)} / ${fmtClock(d)} s`;
  $("play").textContent = playerEl.paused ? "Play" : "Pause";
  $("play").setAttribute("aria-label", playerEl.paused ? "Play" : "Pause");
}
function togglePlay() { if (playerEl.paused) playerEl.play().catch(() => {}); else playerEl.pause(); }
$("play").addEventListener("click", togglePlay);
$("seek").addEventListener("input", () => { playerEl.currentTime = Number($("seek").value); });
$("speed").addEventListener("change", () => { playerEl.playbackRate = Number($("speed").value); });
for (const type of ["timeupdate", "play", "pause", "loadedmetadata", "durationchange", "seeked", "ended"]) playerEl.addEventListener(type, syncControls);
playerEl.addEventListener("loadedmetadata", () => { playerEl.playbackRate = Number($("speed").value); });
playerEl.addEventListener("click", (event) => {
  if (overlay.picking) {
    const rect = playerEl.getBoundingClientRect(), size = codedSize();
    if (rect.height > 0 && size.h) { setPicking(false); saveWaterline(Math.round(((event.clientY - rect.top) / rect.height) * size.h * 10) / 10); }
    return;
  }
  togglePlay();
});
