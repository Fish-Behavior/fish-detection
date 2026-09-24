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
  const seconds = app.detail.frame_summary.seconds_by_state;
  $("summary").replaceChildren(...STATES.filter(([n]) => seconds[n]).map(([n]) =>
    el("li", {}, el("span", { className: "swatch", style: `background:${colorOf(n)}` }), `${n} ${seconds[n].toFixed(1)} s`)));
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
$("player").addEventListener("timeupdate", movePlayhead);
$("player").addEventListener("seeked", movePlayhead);

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
