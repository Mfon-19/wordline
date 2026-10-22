// Worldline viewer: a clip drawn as a solid block (image width x image height x time)
// with one tube per tracked animal. Time runs left to right along the block's long axis.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const LEN = 3.0;   // block length along time
const WID = 1.6;   // block width (image x)
const COLORS = ["#ffb454", "#5ccfe6", "#bae67e", "#f28779", "#d4bfff", "#ffd580", "#73d0ff", "#95e6cb"];

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color("#0b0d10");
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 50);
camera.position.set(-3.3, 1.5, 3.4);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

// The block is drawn by ray marching: for every pixel, step along the line of sight through
// the block, adding a little of each frame's colour (the haze). Where the line crosses the
// "now" sheet, the frame is drawn sharp and nearly opaque.
const volumeMaterial = new THREE.ShaderMaterial({
  glslVersion: THREE.GLSL3,
  side: THREE.BackSide,
  transparent: true,
  depthWrite: false,
  uniforms: {
    vol: { value: null }, size: { value: new THREE.Vector3(LEN, 1, WID) },
    camPos: { value: new THREE.Vector3() }, now: { value: 0 }, haze: { value: 0.35 },
  },
  vertexShader: /* glsl */ `
    uniform vec3 size; out vec3 vPos;
    // the mesh is a unit cube scaled to the block's size; the ray maths wants world units
    void main() { vPos = position * size; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    precision highp float; precision highp sampler3D;
    uniform sampler3D vol; uniform vec3 size; uniform vec3 camPos; uniform float now; uniform float haze;
    in vec3 vPos; out vec4 outColor;
    const int STEPS = 160;
    // rgb is the frame; alpha is how far the pixel is from its average over the clip, so the
    // haze shows what changed (animals, mostly) rather than what is merely bright (snow, sunlit ground)
    // block position -> texture coordinate: image x along z, image y down along y, time along x
    vec3 uvw(vec3 p) { return vec3(p.z / size.z + 0.5, 0.5 - p.y / size.y, p.x / size.x + 0.5); }
    void main() {
      vec3 dir = normalize(vPos - camPos);
      vec3 inv = 1.0 / dir;
      vec3 t0 = (-0.5 * size - camPos) * inv, t1 = (0.5 * size - camPos) * inv;
      vec3 tmin = min(t0, t1), tmax = max(t0, t1);
      float tn = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
      float tf = min(min(tmax.x, tmax.y), tmax.z);
      if (tf <= tn) discard;
      float xNow = (now - 0.5) * size.x;
      float tSlice = (xNow - camPos.x) * inv.x;
      float dt = (tf - tn) / float(STEPS);
      vec4 acc = vec4(0.0);
      float t = tn;
      for (int i = 0; i < STEPS; i++) {
        float tNext = t + dt;
        if (tSlice > t && tSlice <= tNext) {
          vec3 c = texture(vol, uvw(camPos + dir * tSlice)).rgb;
          acc.rgb += (1.0 - acc.a) * 0.96 * c; acc.a += (1.0 - acc.a) * 0.96;
        }
        vec4 s = texture(vol, uvw(camPos + dir * (t + 0.5 * dt)));
        vec3 c = s.rgb;
        float lum = dot(c, vec3(0.299, 0.587, 0.114));
        float a = haze * dt * (0.8 * smoothstep(0.05, 0.9, lum) + 16.0 * smoothstep(0.06, 0.4, s.a));
        acc.rgb += (1.0 - acc.a) * a * c; acc.a += (1.0 - acc.a) * a;
        if (acc.a > 0.98) break;
        t = tNext;
      }
      outColor = acc;
    }`,
});
const block = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), volumeMaterial);
scene.add(block);
const frame = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
  new THREE.LineBasicMaterial({ color: "#3a414c" }));
scene.add(frame);
const tubes = new THREE.Group();
const rings = new THREE.Group();
scene.add(tubes, rings);
// draw order: block first, then tubes and the current outline on top of it
tubes.renderOrder = 1;
rings.renderOrder = 2;

const axis = new THREE.Group();
scene.add(axis);

// Text drawn on a canvas and shown as a sprite that always faces the camera.
function textSprite(text, worldHeight, color = "#c9ced6") {
  const c = document.createElement("canvas"), ctx = c.getContext("2d");
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, transparent: true }));
  sprite.renderOrder = 3;
  sprite.userData.draw = (t) => {
    ctx.font = "500 44px system-ui, sans-serif";
    c.width = Math.ceil(ctx.measureText(t).width) + 16;
    c.height = 60;
    ctx.font = "500 44px system-ui, sans-serif";
    ctx.fillStyle = color;
    ctx.textBaseline = "middle";
    ctx.fillText(t, 8, 31);
    sprite.material.map.dispose();
    sprite.material.map = new THREE.CanvasTexture(c);
    sprite.scale.set(worldHeight * c.width / c.height, worldHeight, 1);
  };
  sprite.userData.draw(text);
  return sprite;
}

let nowLabel = null;

function buildAxis() {
  axis.clear();
  const dur = data.times[data.frames - 1] - data.times[0] || 1;
  const step = [0.5, 1, 2, 5, 10, 20, 30, 60].find((s) => dur / s <= 7) || 120;
  const y = -height / 2, z = WID / 2, ticks = [];
  for (let t = Math.ceil(data.times[0] / step) * step; t <= data.times[data.frames - 1] + 1e-6; t += step) {
    const x = ((t - data.times[0]) / dur - 0.5) * LEN;
    ticks.push(x, y, z, x, y - 0.05, z);
    const label = textSprite(`${+t.toFixed(1)} s`, 0.07, "#8b929c");
    label.position.set(x, y - 0.11, z);
    axis.add(label);
  }
  const lines = new THREE.BufferGeometry();
  lines.setAttribute("position", new THREE.Float32BufferAttribute(ticks, 3));
  axis.add(new THREE.LineSegments(lines, new THREE.LineBasicMaterial({ color: "#5a626e" })));
  const arrow = textSprite("time →", 0.07, "#8b929c");
  arrow.position.set(LEN / 2 + 0.22, y - 0.11, z);
  axis.add(arrow);
  nowLabel = textSprite("0.0 s", 0.09, "#e6e8eb");
  axis.add(nowLabel);
}

let data = null, height = 1, selected = null, hovered = null, playing = false, lastTick = 0;
let search = null; // { query, results: [{clip, tube, score, ...}], lo, hi }

async function loadVolume(meta, base) {
  // decode off the main thread; Image.decode() can stall in background tabs
  const res = await fetch(`${base}/${meta.atlas.file}`);
  if (!res.ok) throw new Error(`atlas ${res.status}`);
  const img = await createImageBitmap(await res.blob());
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const { frame_w: w, frame_h: h, frames: T } = meta;
  // read the atlas once, then copy each tile's rows into its layer of the block
  const px = ctx.getImageData(0, 0, c.width, c.height).data;
  const out = new Uint8Array(w * h * T * 4);
  for (let k = 0; k < T; k++) {
    const x0 = (k % meta.atlas.cols) * w, y0 = Math.floor(k / meta.atlas.cols) * h;
    for (let y = 0; y < h; y++) {
      const src = ((y0 + y) * c.width + x0) * 4;
      out.set(px.subarray(src, src + w * 4), (k * h + y) * w * 4);
    }
  }
  // alpha channel: each pixel's distance from its own average brightness over the clip
  const n = w * h, mean = new Uint32Array(n);
  for (let k = 0, o = 0; k < T; k++) for (let i = 0; i < n; i++, o += 4) mean[i] += (out[o] * 77 + out[o + 1] * 150 + out[o + 2] * 29) >> 8;
  for (let i = 0; i < n; i++) mean[i] = mean[i] / T;
  for (let k = 0, o = 0; k < T; k++) for (let i = 0; i < n; i++, o += 4) {
    const d = ((out[o] * 77 + out[o + 1] * 150 + out[o + 2] * 29) >> 8) - mean[i];
    out[o + 3] = Math.min(255, (d < 0 ? -d : d) * 4);
  }
  const tex = new THREE.Data3DTexture(out, w, h, T);
  tex.format = THREE.RGBAFormat;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

// image-normalised point at frame f -> position in the block
const place = (f, u, v) => new THREE.Vector3(
  (f / (data.frames - 1) - 0.5) * LEN, (0.5 - v) * height, (u - 0.5) * WID);

function ringPoints(tr, k) {
  const pts = [], n = data.ring, aspect = data.frame_w / data.frame_h;
  for (let j = 0; j < n; j++) {
    const a = (j / n) * 2 * Math.PI, r = tr.r[k][j];
    pts.push(place(tr.start + k, tr.cx[k] + r * Math.cos(a), tr.cy[k] + r * Math.sin(a) * aspect));
  }
  return pts;
}

function buildTubes() {
  tubes.clear();
  const n = data.ring;
  data.tracks.forEach((tr, ti) => {
    const pos = [], idx = [], line = [];
    tr.cx.forEach((_, k) => {
      const pts = ringPoints(tr, k);
      pts.forEach((p, j) => {
        pos.push(p.x, p.y, p.z);
        const q = pts[(j + 1) % n];
        line.push(p.x, p.y, p.z, q.x, q.y, q.z);
        if (k > 0) {
          const a = (k - 1) * n + j, b = (k - 1) * n + (j + 1) % n, c = k * n + j, d = k * n + (j + 1) % n;
          idx.push(a, c, b, b, c, d);
        }
      });
    });
    const color = new THREE.Color(COLORS[ti % COLORS.length]);
    const skinGeo = new THREE.BufferGeometry();
    skinGeo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    skinGeo.setIndex(idx);
    const skin = new THREE.Mesh(skinGeo, new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: 0.22, side: THREE.DoubleSide,
      depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending }));
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(line, 3));
    const ribs = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({
      color, transparent: true, opacity: 0.35, depthTest: false, blending: THREE.AdditiveBlending }));
    const g = new THREE.Group();
    skin.renderOrder = ribs.renderOrder = 1;
    g.add(skin, ribs);
    g.userData = { skin, ribs };
    tubes.add(g);
  });
}

function updateRings(f) {
  rings.clear();
  data.tracks.forEach((tr, ti) => {
    if (f < tr.start || f > tr.end || (selected !== null && selected !== ti)) return;
    const pts = ringPoints(tr, f - tr.start);
    pts.push(pts[0]);
    const ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: COLORS[ti % COLORS.length], depthTest: false }));
    ring.renderOrder = 2;
    rings.add(ring);
  });
}

function setFrame(f) {
  f = Math.max(0, Math.min(data.frames - 1, f));
  $("scrub").value = f;
  // sample at the centre of frame f's layer in the texture
  volumeMaterial.uniforms.now.value = (Math.round(f) + 0.5) / data.frames;
  $("clock").textContent = `${data.times[Math.round(f)].toFixed(1)} s`;
  if (nowLabel) {
    nowLabel.userData.draw($("clock").textContent);
    nowLabel.position.set((Math.round(f) / Math.max(data.frames - 1, 1) - 0.5) * LEN, height / 2 + 0.1, WID / 2);
  }
  updateRings(Math.round(f));
}

// How well a tube matches the current search, from 0 to 1 (unconfident matches are capped low), or null.
function match(tr) {
  const r = search?.results.find((x) => x.clip === $("clip").value && x.tube === tr.id);
  if (!r) return null;
  const rel = (r.score - search.lo) / (search.hi - search.lo || 1);
  return r.confident ? 0.4 + 0.6 * rel : 0.3 * rel;
}

function restyle() {
  tubes.children.forEach((g, i) => {
    const m = match(data.tracks[i]);
    let w = selected === null ? (m === null ? 0.5 : 0.08 + 0.92 * m * m) : (selected === i ? 1 : 0.08);
    if (hovered === i) w = Math.max(w, 0.85);
    g.userData.skin.material.opacity = 0.03 + 0.4 * w;
    g.userData.ribs.material.opacity = 0.05 + 0.65 * w;
  });
  [...$("tracks").children].forEach((li) => {
    const i = +li.dataset.ti;
    li.classList.toggle("on", selected === i);
    li.classList.toggle("off", selected !== null && selected !== i);
  });
}

function select(ti) {
  selected = selected === ti ? null : ti;
  restyle();
  if (selected !== null) setFrame(data.tracks[ti].start);
  else updateRings(+$("scrub").value);
}

function buildPanels() {
  const list = $("tracks"), bars = $("bars");
  list.replaceChildren();
  bars.replaceChildren();
  $("empty").hidden = data.tracks.length > 0;
  const anyConfident = search?.results.some((r) => r.confident);
  $("tracks-title").textContent = !search ? "Animals found"
    : anyConfident ? `Matches for “${search.query}”` : `No confident match for “${search.query}”`;
  $("search-note").hidden = !search || anyConfident;
  $("nothing").hidden = data.tracks.length > 0;
  const order = data.tracks.map((tr, ti) => ti);
  if (search) order.sort((a, b) => match(data.tracks[b]) - match(data.tracks[a]));
  for (const ti of order) {
    const tr = data.tracks[ti], color = COLORS[ti % COLORS.length];
    const li = document.createElement("li");
    li.dataset.ti = ti;
    li.style.setProperty("--c", color);
    const title = document.createElement("b");
    title.textContent = `Animal ${tr.id}`;
    const when = document.createElement("small");
    when.textContent = `In at ${tr.enter_s.toFixed(1)} s, stayed ${tr.dwell_s.toFixed(1)} s`;
    const more = document.createElement("small");
    more.textContent = `Moved ${(tr.path_len * 100).toFixed(0)}% of frame width` +
      (tr.found_by === "motion" ? " · found by motion only" : "");
    li.append(title, when, more);
    const m = match(tr);
    if (m !== null) {
      const bar = document.createElement("span"), fill = document.createElement("span");
      bar.className = "match";
      bar.title = `Match ${Math.round(m * 100)}%`;
      fill.style.width = `${Math.max(3, m * 100)}%`;
      bar.append(fill);
      li.append(bar);
    }
    li.onclick = () => select(ti);
    list.append(li);
    const bar = document.createElement("i");
    bar.style.setProperty("--c", color);
    bar.style.left = `${(tr.start / (data.frames - 1)) * 100}%`;
    bar.style.width = `${((tr.end - tr.start) / (data.frames - 1)) * 100}%`;
    bar.style.top = `${4 + (ti % 6) * 6}px`;
    bars.append(bar);
  }
  buildOthers();
  restyle();
}

// The strongest matches in clips other than the one on screen.
function buildOthers() {
  const others = $("others");
  others.replaceChildren();
  const rest = (search?.results || []).filter((r) => r.clip !== $("clip").value);
  const sure = rest.filter((r) => r.confident);
  const rows = (sure.length ? sure : rest).slice(0, 5);
  $("elsewhere").hidden = rows.length === 0;
  $("elsewhere-title").textContent = sure.length ? "Matches in other clips" : "Closest in other clips (not confident)";
  for (const r of rows) {
    const li = document.createElement("li"), b = document.createElement("button");
    const m = (r.score - search.lo) / (search.hi - search.lo || 1);
    b.innerHTML = `<b></b><small></small><span class="match" style="--c:var(--accent)"><span></span></span>`;
    b.querySelector("b").textContent = `${r.clip.replaceAll("_", " ")} · Animal ${r.tube}`;
    b.querySelector("small").textContent = ` at ${r.enter_s.toFixed(1)} s`;
    b.querySelector(".match > span").style.width = `${Math.max(3, m * 100)}%`;
    b.onclick = async () => {
      $("clip").value = r.clip;
      await load(r.clip);
      select(data.tracks.findIndex((t) => t.id === r.tube));
    };
    li.append(b);
    others.append(li);
  }
}

async function runSearch(q) {
  q = q.trim();
  if (!q) { search = null; buildPanels(); $("status").textContent = clipStatus(); return; }
  $("status").textContent = "Searching…";
  try {
    const extra = myUpload ? `&include=${myUpload}` : "";
    const r = await fetch(`api/search?q=${encodeURIComponent(q)}${extra}`);
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || r.statusText);
    const scores = body.results.map((x) => x.score);
    search = { query: q, results: body.results, lo: Math.min(...scores), hi: Math.max(...scores) };
    selected = null;
    buildPanels();
    const n = body.results.filter((x) => x.confident).length;
    $("status").textContent = `${n} confident match${n === 1 ? "" : "es"} · ${body.ms} ms`;
  } catch (e) {
    $("status").textContent = /loading/.test(e.message) ? "Search is starting up; try again in a few seconds."
      : `Search unavailable: ${e.message}`;
  }
}

// ---- Agent: ask a question, watch its steps, approve reprocessing ----
let agentRun = null, agentTimer = null;

function linkify(el, text) {
  // clip#tube references become links that open the clip and select the tube
  const re = /([a-z0-9_]+(?:\/[a-z0-9_]+)*)#(\d+)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    el.append(text.slice(last, m.index));
    const a = document.createElement("a"), [ref, clip, tube] = m;
    a.textContent = ref;
    a.onclick = async () => {
      if ([...$("clip").options].some((o) => o.value === clip)) $("clip").value = clip;
      await load(clip);
      const ti = data.tracks.findIndex((t) => t.id === +tube);
      if (ti >= 0) select(ti);
    };
    el.append(a);
    last = re.lastIndex;
  }
  el.append(text.slice(last));
}

function step(cls, text, extra) {
  const li = document.createElement("li");
  if (cls) li.className = cls;
  linkify(li, text);
  if (extra) li.append(extra);
  $("steps").append(li);
  return li;
}

function renderRun(run) {
  $("steps").replaceChildren();
  for (const e of run.events) {
    if (e.kind === "model") {
      if (e.thinking) {
        const d = document.createElement("details"), s = document.createElement("summary");
        s.textContent = "reasoning";
        d.append(s, e.thinking);
        step("", "", d);
      }
      if (e.text && e.calls.length) step("", e.text);
      for (const c of e.calls) step("tool", `→ ${c.tool}(${Object.values(c.input).filter((v) => v !== null).join(", ")})`);
    } else if (e.kind === "tool_result") {
      step(e.error ? "err" : "", `${e.error ? "✗" : "✓"} ${e.tool}: ${e.text.slice(0, 160)}${e.images ? ` [${e.images} image${e.images > 1 ? "s" : ""}]` : ""}`);
    } else if (e.kind === "approval") {
      step("", e.approved ? "Approved." : `Declined: ${e.note}`);
    } else if (e.kind === "final") {
      step("answer", e.answer);
    } else if (e.kind === "error" || e.kind === "stopped") {
      step("err", run.answer || e.message || e.reason);
    }
  }
  if (run.pending) {
    const row = document.createElement("div"), yes = document.createElement("button"), no = document.createElement("button");
    row.className = "row";
    yes.textContent = "Approve";
    no.textContent = "Decline";
    const send = async (approve) => {
      yes.disabled = no.disabled = true;
      await fetch(`api/agent/${run.id}/decision`, { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ approve, note: approve ? "" : "declined in the viewer" }) });
      pollRun();
    };
    yes.onclick = () => send(true);
    no.onclick = () => send(false);
    row.append(yes, no);
    const w = run.pending.window_s.some((v) => v !== null) ? ` from ${run.pending.window_s[0] ?? 0} s to ${run.pending.window_s[1] ?? "end"}` : "";
    step("approve", `The agent wants to re-run the OpenCV pipeline on ${run.pending.clip}${w} at ${run.pending.sensitivity} sensitivity (up to a minute of CPU). Reason: ${run.pending.reason}`, row);
  } else if (run.status === "running") {
    step("", "…");
  }
}

async function pollRun() {
  clearTimeout(agentTimer);
  const r = await fetch(`api/agent/${agentRun}`);
  if (!r.ok) return;
  const run = await r.json();
  renderRun(run);
  $("ask-go").disabled = run.status === "running" || run.status === "awaiting_approval";
  if (run.status === "running" || run.status === "awaiting_approval") agentTimer = setTimeout(pollRun, 1200);
}

$("ask").onsubmit = async (e) => {
  e.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("steps").replaceChildren();
  $("ask-go").disabled = true;
  const r = await fetch("api/agent", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ question }) });
  const body = await r.json();
  if (!r.ok) { step("err", body.detail || "The agent is unavailable."); $("ask-go").disabled = false; return; }
  agentRun = body.id;
  pollRun();
};

const clipStatus = () =>
  `${data.duration_s.toFixed(1)} s clip · ${data.frames} frames · ${data.tracks.length} tubes · OpenCV ${data.opencv}`;

let loadSeq = 0;  // only the most recent load() call may update the view

async function load(name) {
  const mine = ++loadSeq;
  $("status").textContent = "Loading…";
  $("loading").hidden = false;
  const t0 = performance.now();
  playing = false; $("play").textContent = "▶"; selected = null; hovered = null;
  const base = `data/${name}`;
  let meta, vol;
  try {
    meta = await (await fetch(`${base}/worldline.json`)).json();
    vol = await loadVolume(meta, base);
  } catch (e) {
    if (mine !== loadSeq) return;
    $("loading").hidden = true;
    $("status").textContent = "Couldn't load this clip. Check your connection and try again.";
    return;
  }
  if (mine !== loadSeq) { vol.dispose(); return; }
  data = meta;
  height = WID * data.frame_h / data.frame_w;
  const old = volumeMaterial.uniforms.vol.value;
  volumeMaterial.uniforms.vol.value = vol;
  old?.dispose();
  volumeMaterial.uniforms.size.value.set(LEN, height, WID);
  block.scale.set(LEN, height, WID);
  frame.scale.set(LEN, height, WID);
  $("scrub").max = data.frames - 1;
  buildTubes();
  buildAxis();
  buildPanels();
  setFrame(0);
  $("loading").hidden = true;
  console.info(`loaded ${name} in ${Math.round(performance.now() - t0)} ms`);
  $("status").textContent = clipStatus();
}

// ---- Uploading your own clip ----
let myUpload = null;

function setUploadStatus(text) { $("upload-status").textContent = text; }

async function openUpload(id) {
  myUpload = id;
  const name = `_uploads/${id}`;
  if (![...$("clip").options].some((o) => o.value === name)) $("clip").add(new Option("Your clip (private)", name), 0);
  $("clip").value = name;
  history.replaceState(null, "", `#u=${id}`);
  await load(name);
}

async function watchUpload(id) {
  const r = await fetch(`api/upload/${id}`);
  const job = await r.json();
  if (!r.ok) { setUploadStatus(job.detail || "That upload is no longer available."); return; }
  if (job.status === "queued") setUploadStatus(`Waiting in line${job.ahead ? ` (${job.ahead} ahead)` : ""}…`);
  else if (job.status === "processing") setUploadStatus("OpenCV is finding and tracing animals. This takes about a minute…");
  else if (job.status === "error") { setUploadStatus(job.error); $("upload-go").disabled = false; return; }
  if (job.status !== "done") { setTimeout(() => watchUpload(id), 2000); return; }
  setUploadStatus(`Done in ${job.seconds} s: ${job.tubes} animal visit${job.tubes === 1 ? "" : "s"} found.` +
    (job.people_blurred ? " Anything that looked like a person was pixelated (this sometimes catches animals too)." : "") + " Bookmark this page to come back to it today.");
  $("upload-go").disabled = false;
  await openUpload(id);
}

$("upload-open").onclick = () => { $("upload").hidden = !$("upload").hidden; };
$("upload-cancel").onclick = () => { $("upload").hidden = true; };
$("upload").onsubmit = (ev) => {
  ev.preventDefault();
  const file = $("upload-file").files[0];
  if (!file) return;
  if (file.size > 50 * 2 ** 20) { setUploadStatus("That file is over 50 MB."); return; }
  $("upload-go").disabled = true;
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "api/upload?agree=1");
  xhr.setRequestHeader("X-Filename", file.name);
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) setUploadStatus(`Uploading… ${Math.round(100 * e.loaded / e.total)}%`); };
  xhr.onerror = () => { setUploadStatus("The upload failed. Check your connection and try again."); $("upload-go").disabled = false; };
  xhr.onload = () => {
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch {}
    if (xhr.status !== 200) { setUploadStatus(body.detail || "The upload was refused."); $("upload-go").disabled = false; return; }
    watchUpload(body.id);
  };
  xhr.send(file);
};

// ---- Pointing at tubes in the 3D view ----
const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
let downAt = null;

function tubeAt(ev) {
  const r = canvas.getBoundingClientRect();
  pointer.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(tubes.children.map((g) => g.userData.skin))[0];
  return hit ? { ti: tubes.children.findIndex((g) => g.userData.skin === hit.object), point: hit.point } : null;
}

canvas.addEventListener("pointermove", (ev) => {
  if (!data || ev.buttons) return;
  const hit = tubeAt(ev), tip = $("tip");
  const ti = hit ? hit.ti : null;
  if (ti !== hovered) { hovered = ti; restyle(); }
  canvas.style.cursor = hit ? "pointer" : "";
  if (!hit) { tip.hidden = true; return; }
  const tr = data.tracks[hit.ti], r = canvas.getBoundingClientRect();
  tip.textContent = `Animal ${tr.id} · in at ${tr.enter_s.toFixed(1)} s · stayed ${tr.dwell_s.toFixed(1)} s`;
  tip.style.left = `${ev.clientX - r.left + 14}px`;
  tip.style.top = `${ev.clientY - r.top + 14}px`;
  tip.hidden = false;
});
canvas.addEventListener("pointerleave", () => { $("tip").hidden = true; if (hovered !== null) { hovered = null; restyle(); } });
canvas.addEventListener("pointerdown", (ev) => { downAt = [ev.clientX, ev.clientY]; });
canvas.addEventListener("pointerup", (ev) => {
  // a click, not the end of a drag to turn the block
  if (!data || !downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 5) return;
  const hit = tubeAt(ev);
  if (!hit) { if (selected !== null) select(selected); return; }
  if (selected !== hit.ti) select(hit.ti);
  setFrame(Math.round((hit.point.x / LEN + 0.5) * (data.frames - 1)));
});

// ---- Keyboard ----
document.addEventListener("keydown", (ev) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName) && ev.target.type !== "range") return;
  if (!data) return;
  const f = +$("scrub").value;
  if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
    setFrame(f + (ev.key === "ArrowRight" ? 1 : -1) * (ev.shiftKey ? 10 : 1));
  } else if (ev.key === " ") {
    $("play").click();
  } else if (ev.key === "Escape") {
    if (selected !== null) select(selected);
    $("intro").hidden = true;
  } else if (ev.key === "/") {
    $("q").focus();
  } else return;
  ev.preventDefault();
});

// ---- First-visit explainer ----
function showIntro(show) {
  $("intro").hidden = !show;
  if (!show) try { localStorage.setItem("worldline-intro-seen", "1"); } catch {}
}
$("intro-close").onclick = () => showIntro(false);
$("help").onclick = () => showIntro($("intro").hidden);
try { showIntro(!localStorage.getItem("worldline-intro-seen")); } catch { showIntro(true); }

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function tick(ms) {
  if (playing && data && ms - lastTick > 1000 * (data.duration_s / data.frames)) {
    lastTick = ms;
    setFrame((+$("scrub").value + 1) % data.frames);
  }
  controls.update();
  volumeMaterial.uniforms.camPos.value.copy(camera.position);
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

const CLIPS = await (await fetch("data/index.json")).json();
for (const c of CLIPS) {
  const animals = c.tubes === 0 ? "no animals found" : `${c.tubes} animal visit${c.tubes === 1 ? "" : "s"}`;
  $("clip").add(new Option(`${c.clip.replaceAll("_", " ")} · ${Math.round(c.duration_s)} s · ${animals}`, c.clip));
}
$("clip").onchange = (e) => load(e.target.value);
$("scrub").oninput = (e) => setFrame(+e.target.value);
$("haze").oninput = (e) => { volumeMaterial.uniforms.haze.value = +e.target.value; };
$("search").onsubmit = (e) => { e.preventDefault(); runSearch($("q").value); };
$("q").onsearch = (e) => { if (!e.target.value) runSearch(""); };
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "⏸" : "▶"; };
new ResizeObserver(resize).observe(canvas.parentElement);
resize();
requestAnimationFrame(tick);
const shared = location.hash.match(/^#u=([a-z0-9]{8,32})$/);
if (shared) {
  $("upload").hidden = false;
  watchUpload(shared[1]);
}
const first = CLIPS.some((c) => c.clip === "coyote_family") ? "coyote_family" : CLIPS[0].clip;
$("clip").value = first;
load(first);
window.worldline = { get data() { return data; }, setFrame, select, runSearch };
