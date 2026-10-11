"use strict";
// Labeling UI. Server strings go in via textContent only. Coordinates are image pixels; the canvas is drawn at ZOOM x.
const ZOOM = 3;
const COLORS = { snout: "#ff5a36", dorsal_fin_base: "#ffd23f", ventral: "#3bceac", tail_base: "#4ea8de", tail_tip: "#b388eb" };
const $ = (id) => document.getElementById(id);
let openSeq = 0;
const st = { frames: [], id: null, detail: null, work: null, tool: "box", drag: null, img: null, dirty: false, suggested: false };

const el = (tag, props = {}, ...kids) => { const n = document.createElement(tag); Object.assign(n, props); n.append(...kids); return n; };
const say = (t, err = false) => { $("message").textContent = t; $("message").classList.toggle("error", err); };
const annotator = () => $("annotator").value.trim();

async function api(path, options) {
  const r = await fetch(path, options);
  if (r.ok) return r.json();
  let body = null; try { body = await r.json(); } catch (_) { /* not JSON */ }
  const d = body && body.detail;
  const e = new Error(typeof d === "string" ? d : (d && d.message) || (Array.isArray(d) ? d.map((x) => x.msg).join("; ") : r.statusText));
  e.status = r.status; throw e;
}

// ---------- list / progress ----------
let listSeq = 0;
async function loadList() {
  const seq = ++listSeq;
  const q = new URLSearchParams();
  if ($("f-status").value) q.set("status", $("f-status").value);
  if ($("f-split").value) q.set("split", $("f-split").value);
  if ($("f-batch").value) q.set("batch", $("f-batch").value);
  if ($("f-reason").value) q.set("reason", $("f-reason").value);
  const rows = await api(`/api/frames?${q}`);
  if (seq !== listSeq) return;
  st.frames = rows;
  await fillFilters();
  const list = $("frame-list"); list.replaceChildren();
  for (const r of rows) {
    const b = el("button", { type: "button" }, r.id, el("span", { className: "st", textContent: r.status }),
      el("span", { className: "sub", textContent: `${r.split} · ${r.reason} · ${r.compound} · ${r.batch}` }));
    if (r.id === st.id) b.setAttribute("aria-current", "true");
    b.addEventListener("click", () => openFrame(r.id));
    list.append(el("li", {}, b));
  }
  await loadProgress();
}
let filtersFilled = false;
async function fillFilters() {  // options come from the unfiltered list, once
  if (filtersFilled) return;
  filtersFilled = true;
  const all = await api("/api/frames");
  for (const [id, key] of [["f-batch", "batch"], ["f-reason", "reason"]]) {
    for (const v of [...new Set(all.map((r) => r[key]))].sort()) $(id).append(el("option", { value: v, textContent: v }));
  }
}
async function loadProgress() {
  const p = await api("/api/progress");
  const s = Object.entries(p.by_split).map(([k, v]) => `${k} ${v.done}/${v.total}`).join(" · ");
  const b = Object.entries(p.by_batch).map(([k, v]) => `${k} ${v.done}/${v.total}`).join(" · ");
  $("progress").textContent = `${p.done}/${p.total} labeled (${s}; ${b}) · listing yes ${p.listing.yes}, unsure ${p.listing.unsure}, no ${p.listing.no}`;
}

// ---------- open / working copy ----------
async function openFrame(id) {
  if (st.dirty && !window.confirm("Discard unsaved changes to this frame?")) return;
  const seq = ++openSeq;
  const detail = await api(`/api/frames/${encodeURIComponent(id)}`).catch((e) => { say(e.message, true); return null; });
  if (!detail) return;
  const img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = detail.image_url; }).catch(() => null);
  if (!img) { say("Could not load the image.", true); return; }
  if (seq !== openSeq) return;  // a later click superseded this one
  st.id = id; st.detail = detail; st.img = img; st.dirty = false; st.suggested = false;
  const a = detail.annotation;
  if (a) st.work = { fish_visible: a.fish_visible, box: a.box, keypoints: { ...a.keypoints }, listing: a.listing };
  else if (detail.prefill) { st.work = { fish_visible: true, box: detail.prefill.box, keypoints: {}, listing: null }; st.suggested = true; }
  else st.work = { fish_visible: true, box: null, keypoints: {}, listing: null };
  st.tool = "box";
  $("work").hidden = false; $("placeholder").hidden = true;
  $("title").textContent = id;
  $("subtitle").textContent = `${detail.split} · ${detail.reason} · ${detail.compound} · t=${detail.t_sec.toFixed(1)} s` + (st.suggested ? " · box is a suggestion: check it" : "");
  const c = $("canvas"); c.width = detail.width * ZOOM; c.height = detail.height * ZOOM;
  say(""); syncControls(); draw();
  for (const b of $("frame-list").querySelectorAll("button")) b.toggleAttribute("aria-current", false);
  const cur = [...$("frame-list").querySelectorAll("button")].find((b) => b.firstChild.textContent === id);
  if (cur) cur.setAttribute("aria-current", "true");
}

// ---------- controls ----------
function tools() { return [["box", "Box"], ...st.detail.keypoint_names.map((n) => [n, n.replace(/_/g, " ")])]; }
function syncControls() {
  $("tools").replaceChildren(...tools().map(([key, label], i) => {
    const b = el("button", { type: "button" }, el("span", { className: "swatch", style: `background:${key === "box" ? "#23c552" : COLORS[key]}` }), `${i === 0 ? "B" : i} ${label}`);
    const has = key === "box" ? !!st.work.box : !!st.work.keypoints[key];
    if (has) b.append(el("span", { className: "done", textContent: key === "box" ? "set" : (st.work.keypoints[key][2] === 1 ? "occl" : "set") }));
    b.setAttribute("aria-pressed", String(st.tool === key));
    b.addEventListener("click", () => { st.tool = key; syncControls(); $("canvas").focus(); });
    return b;
  }));
  $("nofish").checked = !st.work.fish_visible;
  for (const r of document.querySelectorAll("input[name=listing]")) { r.checked = st.work.listing === r.value; r.disabled = !st.work.fish_visible; }
  $("suggest").disabled = !(st.detail && st.detail.prefill);
}
function touch() { st.dirty = true; st.suggested = false; syncControls(); draw(); }

$("nofish").addEventListener("change", () => { st.work.fish_visible = !$("nofish").checked; if (!st.work.fish_visible) { st.work.box = null; st.work.keypoints = {}; st.work.listing = null; } touch(); });
for (const r of document.querySelectorAll("input[name=listing]")) r.addEventListener("change", () => { st.work.listing = r.value; st.dirty = true; });
$("suggest").addEventListener("click", () => { st.work.fish_visible = true; st.work.box = st.detail.prefill.box.slice(); touch(); });

// ---------- canvas ----------
function pt(e) {
  const r = $("canvas").getBoundingClientRect();
  const x = Math.min(st.detail.width, Math.max(0, ((e.clientX - r.left) / r.width) * st.detail.width));
  const y = Math.min(st.detail.height, Math.max(0, ((e.clientY - r.top) / r.height) * st.detail.height));
  return [x, y];
}
const canvas = $("canvas");
canvas.addEventListener("pointerdown", (e) => {
  if (!st.detail || !st.work.fish_visible) return;
  canvas.setPointerCapture(e.pointerId);
  const p = pt(e);
  if (st.tool === "box") st.drag = { origin: p, current: p };
  else { st.work.keypoints[st.tool] = [p[0], p[1], e.shiftKey ? 1 : 2]; touch(); }
});
canvas.addEventListener("pointermove", (e) => { if (st.drag) { st.drag.current = pt(e); draw(); } });
canvas.addEventListener("pointerup", () => {
  if (!st.drag) return;
  const [a, b] = [st.drag.origin, st.drag.current]; st.drag = null;
  const box = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  if (box[2] - box[0] >= 3 && box[3] - box[1] >= 3) { st.work.box = box; touch(); } else draw();
});
canvas.addEventListener("pointercancel", () => { st.drag = null; draw(); });
window.addEventListener("keydown", (e) => {
  if (!st.detail || e.ctrlKey || e.metaKey || e.altKey) return;
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(document.activeElement.tagName)) return;
  const names = st.detail.keypoint_names;
  if (e.key === "b" || e.key === "B") st.tool = "box";
  else if (/^[1-9]$/.test(e.key) && names[+e.key - 1]) st.tool = names[+e.key - 1];
  else if (e.key === "Delete" || e.key === "Backspace") { if (st.tool === "box") st.work.box = null; else delete st.work.keypoints[st.tool]; st.dirty = true; }
  else if (e.key === "Enter") { save(); return; }
  else return;
  e.preventDefault(); syncControls(); draw();
});

function draw() {
  const ctx = canvas.getContext("2d");
  ctx.drawImage(st.img, 0, 0, canvas.width, canvas.height);
  const box = st.drag ? [Math.min(st.drag.origin[0], st.drag.current[0]), Math.min(st.drag.origin[1], st.drag.current[1]), Math.max(st.drag.origin[0], st.drag.current[0]), Math.max(st.drag.origin[1], st.drag.current[1])] : st.work.box;
  if (box) {
    ctx.strokeStyle = st.suggested ? "#ffb703" : "#23c552"; ctx.lineWidth = 2; ctx.setLineDash(st.suggested ? [6, 4] : []);
    ctx.strokeRect(box[0] * ZOOM, box[1] * ZOOM, (box[2] - box[0]) * ZOOM, (box[3] - box[1]) * ZOOM); ctx.setLineDash([]);
  }
  const k = st.work.keypoints;
  const line = (a, b, color) => { if (k[a] && k[b]) { ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(k[a][0] * ZOOM, k[a][1] * ZOOM); ctx.lineTo(k[b][0] * ZOOM, k[b][1] * ZOOM); ctx.stroke(); } };
  line("snout", "tail_base", "rgba(255,255,255,.6)"); line("tail_base", "tail_tip", "rgba(255,255,255,.6)"); line("dorsal_fin_base", "ventral", "rgba(255,80,80,.8)");
  for (const [name, [x, y, v]] of Object.entries(k)) {
    ctx.fillStyle = COLORS[name]; ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x * ZOOM, y * ZOOM, v === 1 ? 5 : 7, 0, 2 * Math.PI);
    if (v === 1) { ctx.stroke(); } else { ctx.fill(); ctx.stroke(); }
    ctx.font = "12px monospace"; ctx.fillStyle = "#fff"; ctx.fillText(name.replace(/_/g, " "), x * ZOOM + 9, y * ZOOM - 6);
  }
}

// ---------- save / skip ----------
let saving = false;
async function save() {
  if (saving || !st.detail) return;
  if (!annotator()) { say("Enter your name first.", true); $("annotator").focus(); return; }
  if (st.work.fish_visible && !st.work.box) { say("Draw (or accept) the box first.", true); return; }
  if (st.work.fish_visible && !st.work.listing) { say("Choose Listing: yes / no / unsure.", true); return; }
  const id = st.id; saving = true; $("save").disabled = true;
  try {
    await api(`/api/frames/${encodeURIComponent(id)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...st.work, annotator: annotator() }),
    });
  } catch (e) { say(`Not saved: ${e.message}`, true); saving = false; $("save").disabled = false; return; }
  saving = false; $("save").disabled = false;
  if (st.id === id) st.dirty = false;  // the annotator may have moved on to (and edited) another frame meanwhile
  const idx = st.frames.findIndex((f) => f.id === id);
  try { await loadList(); } catch (e) { say(`Saved, but the list refresh failed: ${e.message}`, true); }
  const next = st.frames.slice(Math.max(0, idx)).find((f) => f.id !== id && f.status === "todo") || st.frames.find((f) => f.status === "todo");
  if (st.id !== id && st.dirty) return;  // do not yank the annotator away from a frame they are editing
  if (next) await openFrame(next.id); else say("Saved. Nothing left to do in this view.");
}
$("save").addEventListener("click", save);
$("skip").addEventListener("click", async () => {
  const idx = st.frames.findIndex((f) => f.id === st.id);
  const next = st.frames.slice(idx + 1).find((f) => f.status === "todo") || st.frames.find((f) => f.status === "todo" && f.id !== st.id);
  if (next) await openFrame(next.id); else say("No other frame to do in this view.");
});
$("f-status").addEventListener("change", () => loadList().catch((e) => say(e.message, true)));
for (const id of ["f-split", "f-batch", "f-reason"]) $(id).addEventListener("change", () => loadList().catch((e) => say(e.message, true)));
try { $("annotator").value = localStorage.getItem("prepds.annotator") || ""; } catch (_) { /* blocked */ }
$("annotator").addEventListener("change", () => { try { localStorage.setItem("prepds.annotator", annotator()); } catch (_) { /* ignore */ } });
window.addEventListener("beforeunload", (e) => { if (st.dirty) { e.preventDefault(); e.returnValue = ""; } });
loadList().catch((e) => say(e.message, true));
