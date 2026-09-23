// HelpCo inspector — a live view of the office plus everything the game view hides:
// memories, beliefs (and who they came from), relationships, identity history, private thoughts,
// and every prompt sent to a model. Renders with the same server-generated art as the Godot client.
"use strict";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const S = {
  ws: null, connected: false, office: null, art: null, clock: null, clockAt: 0,
  employees: new Map(), items: new Map(), questions: new Map(), decisions: new Map(), bubbles: new Map(),
  images: new Map(), selected: null, feed: [], cmdSeq: 0, pending: new Map(),
};

// ------------------------------------------------------------------ connection
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  S.ws = ws;
  ws.onopen = () => { S.connected = true; setStatus("live", true); };
  ws.onclose = () => { S.connected = false; setStatus("offline — reconnecting…", false); setTimeout(connect, 1500); };
  ws.onmessage = (m) => handle(JSON.parse(m.data));
}
function setStatus(text, ok) { const el = $("#status"); el.textContent = text; el.classList.toggle("ok", ok); }

function send(command) {
  return new Promise((resolve) => {
    const id = `c${++S.cmdSeq}`;
    S.pending.set(id, resolve);
    S.ws?.send(JSON.stringify({ type: "command", id, command }));
  });
}

function handle(env) {
  const p = env.payload;
  switch (env.type) {
    case "snapshot": return onSnapshot(p);
    case "clock": S.clock = p; S.clockAt = performance.now(); return renderClock();
    case "employee": S.employees.set(p.id, p); loadSheet(p); return refreshPeople();
    case "item": S.items.set(p.id, p); return;
    case "question": S.questions.set(p.id, p); return renderQuestions();
    case "event": return onEvent(p);
    case "decision": S.decisions.set(p.employee, p.decision); if (S.selected === p.employee) renderPerson(false); return;
    case "whiteboard": return;
    case "ack": { const r = S.pending.get(p.id); S.pending.delete(p.id); r?.(p.result); return; }
  }
}

function onSnapshot(p) {
  S.office = p.office; S.art = p.art; S.clock = p.clock; S.clockAt = performance.now();
  S.employees = new Map(p.employees.map((e) => [e.id, e]));
  S.items = new Map(p.items.map((i) => [i.id, i]));
  S.questions = new Map(p.questions.map((q) => [q.id, q]));
  $("#brain").textContent = `brain: ${p.llm.provider}${p.llm.provider === "openrouter" ? ` · $${p.llm.spent_usd.toFixed(3)}` : ""}`;
  loadImage("bg", "/art/office_bg.png"); loadImage("atlas", "/art/office_atlas.png");
  S.employees.forEach(loadSheet);
  renderClock(); refreshPeople(); renderQuestions();
  if (!S.feed.length) loadFeedHistory();
}

// ------------------------------------------------------------------ art
function loadImage(key, url) {
  const cur = S.images.get(key);
  if (cur && cur.src.endsWith(url)) return cur;
  const img = new Image(); img.src = url; S.images.set(key, img); return img;
}
function loadSheet(e) { if (e.look_key) loadImage(`sheet:${e.id}`, `/art/employee/${e.id}.png?v=${e.look_key}`); }

// ------------------------------------------------------------------ clock
function simNow() {
  if (!S.clock) return 0;
  if (S.clock.paused) return S.clock.sim_ms;
  return S.clock.sim_ms + (performance.now() - S.clockAt) * S.clock.effective_scale;
}
function fmtTime(ms) {
  if (!S.clock) return "";
  return new Date(ms).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", timeZone: tzOrUndef() });
}
function tzOrUndef() { try { new Intl.DateTimeFormat([], { timeZone: S.clock.tz }); return S.clock.tz; } catch { return undefined; } }
function renderClock() {
  if (!S.clock) return;
  $("#clock-date").textContent = `${S.clock.date} · day ${S.clock.day} · ${S.clock.phase}`;
  $("#btn-pause").textContent = S.clock.paused ? "Resume" : "Pause";
  document.querySelectorAll("#speeds button").forEach((b) => b.classList.toggle("active", Number(b.dataset.scale) === S.clock.scale));
}
setInterval(() => { if (S.clock) $("#clock-time").textContent = fmtTime(simNow()); }, 250);

// ------------------------------------------------------------------ office canvas
const canvas = $("#office"), ctx = canvas.getContext("2d");
const SCALE = 2;

function empPixel(e, now) {
  let x = e.pos[0], y = e.pos[1], dir = e.facing, walking = false;
  if (e.path && e.path.length > 1 && e.ms_per_tile) {
    const f = Math.max(0, (now - e.path_t0) / e.ms_per_tile);
    const i = Math.min(Math.floor(f), e.path.length - 1);
    const a = e.path[i], b = e.path[Math.min(i + 1, e.path.length - 1)];
    const t = i >= e.path.length - 1 ? 0 : f - i;
    x = a[0] + (b[0] - a[0]) * t; y = a[1] + (b[1] - a[1]) * t;
    if (b[0] !== a[0] || b[1] !== a[1]) { dir = b[0] > a[0] ? "right" : b[0] < a[0] ? "left" : b[1] > a[1] ? "down" : "up"; walking = true; }
  }
  return { x: x * 16 + 8, y: (y + 1) * 16, dir, walking };
}

function animFor(e, p) {
  const pose = e.activity?.pose;
  if (p.walking) return `walk_${p.dir}`;
  if (pose === "type") return "type";
  if (pose === "sit") return p.dir === "up" ? "sit_up" : "sit_down";
  const held = e.holding && S.items.get(e.holding);
  const talking = S.bubbles.get(e.id) && S.bubbles.get(e.id).until > performance.now();
  if (p.dir === "down") {
    if (talking) return "talk_down";
    if (held && held.kind === "coffee") return (Math.floor(performance.now() / 4000) + e.hire_no) % 3 === 0 ? "sip" : "hold_down";
    return "idle_down";
  }
  return `idle_${p.dir}`;
}

function draw() {
  requestAnimationFrame(draw);
  const bg = S.images.get("bg"), atlas = S.images.get("atlas");
  if (!S.art || !bg?.complete || !atlas?.complete) return;
  const W = 384, H = 224;
  if (canvas.width !== W * SCALE) { canvas.width = W * SCALE; canvas.height = H * SCALE; }
  ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(bg, 0, 0);
  const om = S.art.office, cm = S.art.character, now = simNow(), t = performance.now() / 1000;
  const list = [];
  for (const pl of om.placements) list.push({ y: pl.sort_y, o: 0, fn: () => sprite(atlas, om.sprites[pl.sprite], pl.x, pl.y) });
  const slotUse = {};
  for (const it of S.items.values()) {
    const loc = it.location || {}, name = om.items[it.kind];
    if (!name) continue;
    const r = om.sprites[name];
    if (loc.on && om.surfaces[loc.on]) {
      const slots = om.surfaces[loc.on]; const i = slotUse[loc.on] = (slotUse[loc.on] ?? -1) + 1;
      const [sx, sy] = slots[i % slots.length];
      list.push({ y: sy + 40, o: 2, fn: () => sprite(atlas, r, sx, sy - r.h + 4) });
    } else if (loc.at) {
      list.push({ y: loc.at[1] * 16 + 15, o: 2, fn: () => sprite(atlas, r, loc.at[0] * 16 + 4, loc.at[1] * 16 + 8) });
    }
  }
  const overlays = [];
  for (const e of S.employees.values()) {
    if (e.status !== "present") continue;
    const sheet = S.images.get(`sheet:${e.id}`);
    const p = empPixel(e, now);
    const anim = cm.animations[animFor(e, p)] || cm.animations.idle_down;
    const frame = Math.floor(t * anim.fps + e.hire_no * 3) % anim.frames;
    const seat = ["sit", "type"].includes(e.activity?.pose) && !p.walking ? 2 : 0;
    const dx = Math.round(p.x - cm.anchor[0]), dy = Math.round(p.y - cm.anchor[1] + seat);
    list.push({ y: p.y, o: 1, fn: () => {
      if (sheet?.complete && sheet.naturalWidth) ctx.drawImage(sheet, frame * cm.frame_w, anim.row * cm.frame_h, cm.frame_w, cm.frame_h, dx, dy, cm.frame_w, cm.frame_h);
      if (S.selected === e.id) { ctx.strokeStyle = "#f06fa4"; ctx.lineWidth = 1; ctx.beginPath(); ctx.ellipse(p.x, p.y - 1, 8, 3, 0, 0, Math.PI * 2); ctx.stroke(); }
    } });
    overlays.push({ e, p });
  }
  list.sort((a, b) => a.y - b.y || a.o - b.o).forEach((d) => d.fn());
  // labels, thinking dots and speech bubbles at native resolution
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  for (const { e, p } of overlays) {
    const sx = p.x * SCALE, sy = (p.y - 34) * SCALE;
    label(e.name, sx, sy + 4);
    const b = S.bubbles.get(e.id);
    if (b && b.until > performance.now()) bubble(b.text, sx, sy - 6);
    else if (e.thinking) bubble("…", sx, sy - 6, true);
  }
}
function sprite(img, r, x, y) { ctx.drawImage(img, r.x, r.y, r.w, r.h, Math.round(x), Math.round(y), r.w, r.h); }
function label(text, x, y) {
  ctx.font = "600 11px ui-rounded, system-ui, sans-serif";
  const w = ctx.measureText(text).width + 10;
  ctx.fillStyle = "rgba(42,31,51,.78)"; roundRect(x - w / 2, y - 12, w, 16, 8); ctx.fill();
  ctx.fillStyle = "#fff7e8"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, x, y - 4);
}
function bubble(text, x, y, small) {
  ctx.font = small ? "700 14px system-ui, sans-serif" : "13px ui-rounded, system-ui, sans-serif";
  const lines = wrap(text, small ? 40 : 180).slice(0, 3);
  const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 16, h = lines.length * 16 + 10;
  const bx = Math.max(4, Math.min(canvas.width - w - 4, x - w / 2)), by = Math.max(4, y - h - 14);
  ctx.fillStyle = "#fffaf2"; ctx.strokeStyle = "#2a1f33"; ctx.lineWidth = 2;
  roundRect(bx, by, w, h, 9); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x - 5, by + h - 1); ctx.lineTo(x, by + h + 7); ctx.lineTo(x + 5, by + h - 1); ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.moveTo(x - 5, by + h); ctx.lineTo(x, by + h + 7); ctx.lineTo(x + 5, by + h); ctx.stroke();
  ctx.fillStyle = "#2a1f33"; ctx.textAlign = "left"; ctx.textBaseline = "top";
  lines.forEach((l, i) => ctx.fillText(l, bx + 8, by + 6 + i * 16));
}
function wrap(text, maxW) {
  const words = String(text).split(/\s+/), lines = []; let cur = "";
  for (const w of words) { const t = cur ? cur + " " + w : w; if (ctx.measureText(t).width > maxW && cur) { lines.push(cur); cur = w; } else cur = t; }
  if (cur) lines.push(cur);
  if (lines.length > 3) lines[2] = lines[2].replace(/.{0,2}$/, "…");
  return lines;
}
function roundRect(x, y, w, h, r) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); }
canvas.addEventListener("click", (ev) => {
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * (canvas.width / rect.width) / SCALE, y = (ev.clientY - rect.top) * (canvas.height / rect.height) / SCALE;
  let best = null, bd = 1e9;
  for (const e of S.employees.values()) {
    if (e.status !== "present") continue;
    const p = empPixel(e, simNow()); const d = Math.hypot(p.x - x, p.y - 14 - y);
    if (d < bd) { bd = d; best = e; }
  }
  if (best && bd < 24) { selectPerson(best.id); showTab("people"); }
});
requestAnimationFrame(draw);

// ------------------------------------------------------------------ feed
const QUIET = new Set(["action.started", "employee.entered", "activity.completed", "conversation.ended"]);
function onEvent(ev) {
  if (["speech.said", "employee.introduced", "owner.reply"].includes(ev.type) && ev.actor) {
    const txt = ev.payload.text || "";
    S.bubbles.set(ev.actor, { text: txt, until: performance.now() + 2500 + 60 * txt.length });
  }
  if (ev.type === "llm.fallback" || ev.type === "reflection.completed") refreshBrain();
  addFeed(ev);
  if (S.selected && (ev.actor === S.selected || ev.witnesses?.[S.selected])) schedulePersonRefresh();
  if (["note.written", "file.written"].includes(ev.type)) renderFiles();
}
function feedClass(ev) {
  if (["speech.said", "employee.introduced", "owner.reply", "owner.message"].includes(ev.type)) return "speech";
  if (ev.type === "day.started") return "day";
  if (ev.type === "action.denied") return "denied";
  if (ev.type === "question.submitted") return "question";
  if (ev.type === "question.answered") return "answer";
  if (ev.type === "reflection.completed" || ev.type === "identity.changed") return "reflect";
  if (ev.visibility === "system") return "system";
  return "";
}
function addFeed(ev, bulk) {
  S.feed.push(ev); if (S.feed.length > 600) S.feed.shift();
  if (!bulk) renderFeedItem(ev);
}
function renderFeedItem(ev) {
  const verbose = $("#feed-verbose").checked;
  if (!verbose && QUIET.has(ev.type)) return;
  const li = document.createElement("li"); li.className = feedClass(ev);
  li.innerHTML = `<span class="t">${esc(fmtTime(ev.sim_ms))}</span><span class="m">${esc(ev.summary)}</span>`;
  const ol = $("#feed"); const stick = ol.parentElement.scrollTop + ol.parentElement.clientHeight >= ol.parentElement.scrollHeight - 30;
  ol.appendChild(li); while (ol.children.length > 400) ol.firstChild.remove();
  if (stick) ol.parentElement.scrollTop = ol.parentElement.scrollHeight;
}
function rerenderFeed() { $("#feed").innerHTML = ""; S.feed.forEach(renderFeedItem); }
async function loadFeedHistory() {
  const res = await fetch("/api/events?limit=2000&after=" + Math.max(0, (S.lastSeqHint || 0)));
  const rows = await res.json();
  rows.slice(-400).forEach((r) => addFeed(r, true)); rerenderFeed();
}
$("#feed-verbose").addEventListener("change", rerenderFeed);

// ------------------------------------------------------------------ people
function refreshPeople() {
  const list = $("#people-list"); list.innerHTML = "";
  const opts = [];
  for (const e of [...S.employees.values()].sort((a, b) => a.hire_no - b.hire_no)) {
    const chip = document.createElement("div"); chip.className = "chip" + (S.selected === e.id ? " active" : "");
    chip.innerHTML = `${e.look_key ? `<img src="/art/portrait/${esc(e.id)}.png?v=${esc(e.look_key)}" alt="">` : "<span>🆕</span>"}
      <div><b>${esc(e.name)}</b><small>${esc(e.status === "present" ? e.activity.label : e.status)}</small></div>`;
    chip.onclick = () => selectPerson(e.id);
    list.appendChild(chip);
    if (e.status !== "hired") opts.push(`<option value="${esc(e.id)}">${esc(e.name)}</option>`);
  }
  for (const sel of ["#msg-to", "#give-to"]) { const el = $(sel), v = el.value; el.innerHTML = opts.join(""); if (v) el.value = v; }
  if (S.selected) renderPerson(false);
}
let personTimer = null;
function schedulePersonRefresh() { clearTimeout(personTimer); personTimer = setTimeout(() => renderPerson(true), 600); }
function selectPerson(id) { S.selected = id; S.personTab = S.personTab || "memories"; refreshPeople(); renderPerson(true); }

async function renderPerson(fetchDetail) {
  const e = S.employees.get(S.selected); if (!e) return;
  if (fetchDetail || !S.detail || S.detail.employee.id !== e.id) {
    const res = await fetch(`/api/employees/${encodeURIComponent(e.id)}`); S.detail = await res.json();
  }
  const d = S.detail, dec = S.decisions.get(e.id) || d.last_decision || {};
  const needs = e.needs || {};
  const bar = (v, color) => `<div class="bar"><span style="width:${Math.round((v || 0) * 100)}%;background:${color}"></span></div>`;
  const tabs = ["memories", "today", "beliefs", "people", "identity", "goals", "prompts"];
  const el = $("#person");
  el.innerHTML = `
    ${e.look_key ? `<img class="portrait" src="/art/portrait/${esc(e.id)}.png?v=${esc(e.look_key)}" alt="">` : ""}
    <h2>${esc(e.name)} <span class="muted" style="font-size:14px">${esc(e.pronouns)}</span></h2>
    <div class="meta">${esc(e.status === "present" ? e.activity.label : e.status)} · ${esc(d.days_worked)} day(s) worked · desk ${esc(e.desk_id || "–")}</div>
    <div class="meta">“${esc(d.intro)}”</div>
    <div class="thought"><div class="k">last decision ${dec.source ? `· ${esc(dec.source)} ${esc(dec.model || "")}` : ""}</div>
      ${dec.thought ? `<div><i>${esc(dec.thought)}</i></div><div class="muted">→ ${esc(dec.action)} ${esc(dec.target || "")} ${dec.text ? `“${esc(dec.text)}”` : ""}</div>` : "<span class='muted'>—</span>"}</div>
    <div class="needs"><span>energy</span>${bar(needs.energy, "#5aa36a")}<span>hunger</span>${bar(needs.hunger, "#e8893a")}<span>social</span>${bar(needs.social, "#8e6cc9")}</div>
    <div class="sub-tabs">${tabs.map((t) => `<button data-sub="${t}" class="${S.personTab === t ? "active" : ""}">${t}</button>`).join("")}</div>
    <div id="person-body"></div>`;
  el.querySelectorAll("[data-sub]").forEach((b) => b.onclick = () => { S.personTab = b.dataset.sub; renderPerson(false); });
  $("#person-body").innerHTML = personBody(S.personTab, d);
}

function when(ms) { return `${new Date(ms).toLocaleDateString([], { month: "short", day: "numeric", timeZone: tzOrUndef() })} ${fmtTime(ms)}`; }
function personBody(tab, d) {
  const li = (inner, cls = "") => `<li class="${cls}">${inner}</li>`;
  if (tab === "memories") {
    if (!d.memories.length) return "<p class='muted'>No long-term memories yet — they form at night, during reflection.</p>";
    return `<ul class="list">${d.memories.map((m) => li(`<span class="imp" title="importance">${Math.round(m.importance)}</span>${esc(m.text)}
      <div class="when">${esc(m.kind)} · ${when(m.sim_ms)} · recalled ${m.access_count}×</div>`)).join("")}</ul>`;
  }
  if (tab === "today") {
    if (!d.today.length) return "<p class='muted'>Nothing yet today.</p>";
    return `<ul class="list">${d.today.map((m) => li(`${esc(m.text)}<span class="prov">${esc(m.provenance?.channel || "")}</span>
      <div class="when">${fmtTime(m.sim_ms)}</div>`)).join("")}</ul>`;
  }
  if (tab === "beliefs") {
    if (!d.beliefs.length) return "<p class='muted'>No beliefs yet.</p>";
    return `<ul class="list">${d.beliefs.map((b) => li(`<b>${esc(b.about_name)}</b>: ${esc(b.statement)}
      <span class="prov">${esc(b.source?.channel || "")}${b.source?.person ? " · from " + esc(b.source.person) : ""} · confidence ${Number(b.confidence).toFixed(2)}</span>
      <div class="when">since ${when(b.adopted_ms)}${b.abandoned_ms ? " · abandoned " + when(b.abandoned_ms) : ""}</div>`, b.abandoned_ms ? "old" : "")).join("")}</ul>`;
  }
  if (tab === "people") {
    if (!d.relationships.length) return `<p class='muted'>No relationships yet. Met: ${esc(d.met.join(", ") || "nobody")}</p>`;
    const bar = (v, c) => `<div class="bar"><span style="width:${Math.round(v * 100)}%;background:${c}"></span></div>`;
    return d.relationships.map((r) => `<div class="rel"><b>${esc(r.name)}</b><div class="bars">
      <span>familiarity</span>${bar(r.familiarity, "#6cb4e4")}<span>warmth</span>${bar((r.warmth + 1) / 2, "#f06fa4")}<span>trust</span>${bar(r.trust, "#5aa36a")}</div>
      <ul class="list">${r.notes.slice().reverse().map((n) => li(`${esc(n.text)} <span class="when">${when(n.ms)}</span>`)).join("")}</ul></div>`).join("");
  }
  if (tab === "identity") {
    return `<ul class="list">${d.identity.slice().reverse().map((v) => li(`<b>v${v.version}</b> ${esc(v.name)} (${esc(v.pronouns)}) — ${esc(v.change_reason)}
      <div>${v.self_concept ? esc(v.self_concept) : "<span class='muted'>(no self-concept yet)</span>"}</div>
      <div class="when">${when(v.valid_from_ms)}${v.valid_to_ms ? " → " + when(v.valid_to_ms) : " → now"}</div>`, v.valid_to_ms ? "muted" : "")).join("")}</ul>`;
  }
  if (tab === "goals") {
    if (!d.goals.length) return "<p class='muted'>No goals yet.</p>";
    return `<ul class="list">${d.goals.map((g) => li(`${esc(g.text)} <span class="prov">${esc(g.status)}</span>`, g.status !== "active" ? "old" : "")).join("")}</ul>`;
  }
  if (tab === "prompts") {
    return d.llm_calls.map((c) => {
      let req = {}; try { req = JSON.parse(c.request || "{}"); } catch {}
      const user = req.user ?? req.messages?.find((m) => m.role === "user")?.content ?? "";
      return `<details><summary><b>${esc(c.task)}</b> · ${esc(c.model)} · ${fmtTime(c.sim_ms)} ${c.cost ? "· $" + Number(c.cost).toFixed(4) : ""} ${c.ok ? "" : "· ⚠ " + esc(c.error || "")}</summary>
        <div class="muted">prompt</div><pre>${esc(typeof user === "string" ? user : JSON.stringify(user, null, 1))}</pre>
        <div class="muted">reply</div><pre>${esc(c.content)}</pre></details>`;
    }).join("") || "<p class='muted'>No model calls yet.</p>";
  }
  return "";
}

// ------------------------------------------------------------------ questions, files, models
function renderQuestions() {
  const qs = [...S.questions.values()].sort((a, b) => b.submitted_ms - a.submitted_ms);
  $("#questions").innerHTML = qs.length ? qs.map((q) => `<div class="q"><div class="status">${esc(q.id)} · ${esc(q.status)}
    ${q.claimed_by ? " · " + esc(S.employees.get(q.claimed_by)?.name || q.claimed_by) : ""}</div>
    <div><b>${esc(q.text)}</b></div>
    ${q.answer ? `<div class="answer">${esc(q.answer)}</div><div class="muted" style="font-size:12px">answered by ${esc(S.employees.get(q.answered_by)?.name || "")} at ${fmtTime(q.answered_ms)}</div>` : ""}</div>`).join("")
    : "<p class='muted'>No questions yet. Send one below the office.</p>";
}
async function renderFiles() {
  const files = await (await fetch("/api/files")).json();
  $("#files").innerHTML = files.length ? files.map((f) => `<details><summary><b>${esc(f.path)}</b> <span class="muted">by ${esc(S.employees.get(f.author)?.name || f.author)}</span></summary><pre>${esc(f.text)}</pre></details>`).join("")
    : "<p class='muted'>No files yet. Employees can write to /shared/… or their own /private/… folder.</p>";
}
async function renderModels() {
  const d = await (await fetch("/api/llm")).json();
  const s = d.status;
  $("#models").innerHTML = `<p><b>${esc(s.provider)}</b> · spent $${Number(s.spent_usd).toFixed(4)} of $${esc(s.budget_usd)} ${s.exhausted ? "· <b>budget reached</b>" : ""} · cassette: ${esc(s.cassette)}</p>
    <pre>${esc(JSON.stringify(s.routes, null, 2))}</pre>
    ${d.recent.map((c) => `<details><summary>${esc(c.task)} · ${esc(S.employees.get(c.emp_id)?.name || c.emp_id || "")} · ${esc(c.model)} · ${fmtTime(c.sim_ms)} ${c.cost ? "· $" + Number(c.cost).toFixed(4) : ""}</summary><pre>${esc(c.content)}</pre></details>`).join("")}`;
}
async function refreshBrain() {
  const d = await (await fetch("/api/llm")).json();
  $("#brain").textContent = `brain: ${d.status.provider}${d.status.provider === "openrouter" ? ` · $${d.status.spent_usd.toFixed(3)}` : ""}${d.status.exhausted ? " · budget hit" : ""}`;
}

// ------------------------------------------------------------------ tabs & controls
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== `tab-${name}`));
  if (name === "files") renderFiles();
  if (name === "models") renderModels();
  if (name === "people" && !S.selected && S.employees.size) selectPerson([...S.employees.keys()][0]);
}
document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => showTab(b.dataset.tab));
function toast(t) { $("#toast").textContent = t; clearTimeout(toast.t); toast.t = setTimeout(() => $("#toast").textContent = "", 5000); }

$("#form-question").onsubmit = async (ev) => {
  ev.preventDefault(); const text = $("#q-text").value.trim(); if (!text) return;
  const r = await send({ type: "submit_question", text }); $("#q-text").value = "";
  toast(r.ok ? `Sent as ${r.id}. Whoever's in the work area will see it.` : r.error);
};
$("#form-message").onsubmit = async (ev) => {
  ev.preventDefault(); const text = $("#msg-text").value.trim(); if (!text) return;
  const r = await send({ type: "message", to: $("#msg-to").value, text }); $("#msg-text").value = "";
  toast(r.ok ? (r.delivered ? "They heard you." : "They're not in right now — it'll be waiting in their memory.") : r.error);
};
$("#form-give").onsubmit = async (ev) => {
  ev.preventDefault();
  const r = await send({ type: "give", to: $("#give-to").value, kind: $("#give-kind").value });
  toast(r.ok ? "Delivered." : r.error);
};
$("#btn-hire").onclick = async () => { const r = await send({ type: "hire" }); toast(r.ok ? "Hired! They'll choose who to be before they walk in." : r.error); };
$("#btn-pause").onclick = () => send({ type: "clock", action: S.clock?.paused ? "resume" : "pause" });
document.querySelectorAll("#speeds button").forEach((b) => b.onclick = () => send({ type: "clock", action: "scale", value: Number(b.dataset.scale) }));

connect();
