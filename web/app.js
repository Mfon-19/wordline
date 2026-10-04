// Worldline viewer: a clip drawn as a solid block (image width x image height x time)
// with one tube per tracked animal. Time runs left to right along the block's long axis.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CLIPS = ["coyote_family", "badger_cubs", "roe_deer", "red_fox"];
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
        vec3 c = texture(vol, uvw(camPos + dir * (t + 0.5 * dt))).rgb;
        float lum = dot(c, vec3(0.299, 0.587, 0.114));
        float a = haze * 6.0 * dt * smoothstep(0.05, 0.9, lum);
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

let data = null, height = 1, selected = null, playing = false, lastTick = 0;

async function loadVolume(meta, base) {
  const img = new Image();
  img.src = `${base}/${meta.atlas.file}`;
  await img.decode();
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
  updateRings(Math.round(f));
}

function select(ti) {
  selected = selected === ti ? null : ti;
  tubes.children.forEach((g, i) => {
    const on = selected === null || selected === i;
    g.userData.skin.material.opacity = on ? (selected === null ? 0.22 : 0.4) : 0.03;
    g.userData.ribs.material.opacity = on ? (selected === null ? 0.35 : 0.7) : 0.05;
  });
  [...$("tracks").children].forEach((li, i) => {
    li.classList.toggle("on", selected === i);
    li.classList.toggle("off", selected !== null && selected !== i);
  });
  if (selected !== null) setFrame(data.tracks[ti].start);
  else updateRings(+$("scrub").value);
}

function buildPanels() {
  const list = $("tracks"), bars = $("bars");
  list.replaceChildren();
  bars.replaceChildren();
  $("empty").hidden = data.tracks.length > 0;
  const rows = Math.max(data.tracks.length, 1);
  data.tracks.forEach((tr, ti) => {
    const color = COLORS[ti % COLORS.length];
    const li = document.createElement("li");
    li.style.setProperty("--c", color);
    const guess = Object.keys(tr.guesses)[0];
    const title = document.createElement("b");
    title.textContent = `Animal ${tr.id}`;
    const when = document.createElement("small");
    when.textContent = `In at ${tr.enter_s.toFixed(1)} s, stayed ${tr.dwell_s.toFixed(1)} s`;
    const more = document.createElement("small");
    more.textContent = `Moved ${(tr.path_len * 100).toFixed(0)}% of frame width · model guess: ${guess}`;
    li.append(title, when, more);
    li.onclick = () => select(ti);
    list.append(li);
    const bar = document.createElement("i");
    bar.style.setProperty("--c", color);
    bar.style.left = `${(tr.start / (data.frames - 1)) * 100}%`;
    bar.style.width = `${((tr.end - tr.start) / (data.frames - 1)) * 100}%`;
    bar.style.top = `${4 + (ti % 6) * 6}px`;
    bars.append(bar);
  });
}

async function load(name) {
  $("status").textContent = "Loading…";
  const t0 = performance.now();
  playing = false; $("play").textContent = "▶"; selected = null;
  const base = `data/${name}`;
  data = await (await fetch(`${base}/worldline.json`)).json();
  height = WID * data.frame_h / data.frame_w;
  const old = volumeMaterial.uniforms.vol.value;
  volumeMaterial.uniforms.vol.value = await loadVolume(data, base);
  old?.dispose();
  volumeMaterial.uniforms.size.value.set(LEN, height, WID);
  block.scale.set(LEN, height, WID);
  frame.scale.set(LEN, height, WID);
  $("scrub").max = data.frames - 1;
  buildTubes();
  buildPanels();
  setFrame(0);
  console.info(`loaded ${name} in ${Math.round(performance.now() - t0)} ms`);
  $("status").textContent = `${data.duration_s.toFixed(1)} s clip · ${data.frames} frames · ${data.tracks.length} tubes · OpenCV ${data.opencv}`;
}

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

for (const c of CLIPS) $("clip").add(new Option(c.replace("_", " "), c));
$("clip").onchange = (e) => load(e.target.value);
$("scrub").oninput = (e) => setFrame(+e.target.value);
$("haze").oninput = (e) => { volumeMaterial.uniforms.haze.value = +e.target.value; };
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "⏸" : "▶"; };
new ResizeObserver(resize).observe(canvas.parentElement);
resize();
requestAnimationFrame(tick);
load(CLIPS[0]);
window.worldline = { get data() { return data; }, setFrame, select };
