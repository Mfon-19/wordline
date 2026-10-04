// Review and correct ground-truth labels: a few (time, x, y) points per animal visit.
// The pipeline's tubes are deliberately not shown here, so the labels stay independent of them.
const COLORS = ["#ffb454", "#5ccfe6", "#bae67e", "#f28779", "#d4bfff", "#ffd580", "#73d0ff", "#95e6cb"];
const SCALE = 3;
const $ = (id) => document.getElementById(id);
const canvas = $("frame"), ctx = canvas.getContext("2d");

let data, truth, atlas, name, sel = 0, frame = 0, dirty = false, playing = false, last = 0;

const time = () => data.times[frame];
const span = (a) => a.points.length ? [a.points[0][0], a.points[a.points.length - 1][0]] : null;

function at(a, t) {
  const p = a.points;
  if (!p.length || t < p[0][0] - 1e-6 || t > p[p.length - 1][0] + 1e-6) return null;
  for (let i = 1; i < p.length; i++) {
    if (t <= p[i][0]) {
      const w = (t - p[i - 1][0]) / (p[i][0] - p[i - 1][0] || 1);
      return [p[i - 1][1] + w * (p[i][1] - p[i - 1][1]), p[i - 1][2] + w * (p[i][2] - p[i - 1][2])];
    }
  }
  return [p[p.length - 1][1], p[p.length - 1][2]];
}

const keyIndex = (a) => a.points.findIndex((p) => Math.abs(p[0] - time()) < 0.02);

function draw() {
  const { frame_w: w, frame_h: h } = data, cols = data.atlas.cols;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(atlas, (frame % cols) * w, Math.floor(frame / cols) * h, w, h, 0, 0, canvas.width, canvas.height);
  truth.animals.forEach((a, i) => {
    const xy = at(a, time());
    if (!xy) return;
    const key = keyIndex(a) >= 0;
    ctx.beginPath();
    ctx.arc(xy[0] * canvas.width, xy[1] * canvas.height, i === sel ? 11 : 8, 0, 2 * Math.PI);
    ctx.lineWidth = 3;
    ctx.strokeStyle = COLORS[i % COLORS.length];
    ctx.fillStyle = COLORS[i % COLORS.length];
    key ? ctx.fill() : ctx.stroke();
    ctx.font = "bold 14px system-ui";
    ctx.fillText(String(i + 1), xy[0] * canvas.width + 14, xy[1] * canvas.height + 5);
  });
  $("clock").textContent = `${time().toFixed(1)} s`;
  $("scrub").value = frame;
}

function panels() {
  const list = $("animals"), bars = $("bars"), dur = data.times[data.frames - 1] || 1;
  list.replaceChildren();
  bars.replaceChildren();
  truth.animals.forEach((a, i) => {
    const s = span(a), li = document.createElement("li"), c = COLORS[i % COLORS.length];
    li.style.setProperty("--c", c);
    li.className = i === sel ? "on" : "";
    const b = document.createElement("b"), sm = document.createElement("small"), n = document.createElement("small");
    b.textContent = `Visit ${i + 1}`;
    sm.textContent = s ? `${s[0].toFixed(1)} s to ${s[1].toFixed(1)} s · ${a.points.length} points` : "no points yet";
    n.textContent = a.note || "";
    li.append(b, sm, n);
    li.onclick = () => { sel = i; if (s) frame = nearest(s[0]); refresh(); };
    list.append(li);
    if (s) {
      const bar = document.createElement("i");
      bar.style.setProperty("--c", c);
      bar.style.left = `${(s[0] / dur) * 100}%`;
      bar.style.width = `${((s[1] - s[0]) / dur) * 100}%`;
      bar.style.top = `${4 + (i % 6) * 6}px`;
      bars.append(bar);
    }
  });
  $("anote").value = truth.animals[sel]?.note || "";
  $("cnote").value = truth.note || "";
  $("verified").checked = !!truth.verified;
  $("saved").textContent = dirty ? "Unsaved changes" : "";
}

const nearest = (t) => data.times.reduce((best, v, i) => Math.abs(v - t) < Math.abs(data.times[best] - t) ? i : best, 0);
const refresh = () => { panels(); draw(); };
const touch = () => { dirty = true; refresh(); };

async function load(clip) {
  if (dirty && !confirm("Discard unsaved changes to this clip?")) { $("clip").value = name; return; }
  name = clip;
  data = await (await fetch(`data/${clip}/worldline.json`)).json();
  truth = await (await fetch(`truth/${clip}.json`, { cache: "no-store" })).json();
  truth.animals ??= [];
  atlas = new Image();
  atlas.src = `data/${clip}/${data.atlas.file}`;
  await atlas.decode();
  canvas.width = data.frame_w * SCALE;
  canvas.height = data.frame_h * SCALE;
  $("scrub").max = data.frames - 1;
  frame = 0; sel = 0; dirty = false; playing = false; $("play").textContent = "▶";
  $("status").textContent = `${data.duration_s.toFixed(1)} s · ${data.frames} frames`;
  refresh();
}

async function save() {
  truth.animals.forEach((a) => a.points.sort((p, q) => p[0] - q[0]));
  if (truth.verified) truth.labelled_by = "Drafted by Claude from sampled frames; checked and corrected by the author.";
  const r = await fetch(`truth/${name}.json`, { method: "PUT", body: JSON.stringify(truth) });
  dirty = !r.ok;
  $("saved").textContent = r.ok ? "Saved" : "Save failed";
}

canvas.onclick = (e) => {
  if (!truth.animals.length) truth.animals.push({ note: "", points: [] });
  const r = canvas.getBoundingClientRect();
  const x = +((e.clientX - r.left) / r.width).toFixed(3), y = +((e.clientY - r.top) / r.height).toFixed(3);
  const a = truth.animals[sel], k = keyIndex(a);
  if (k >= 0) a.points[k] = [a.points[k][0], x, y];
  else { a.points.push([+time().toFixed(3), x, y]); a.points.sort((p, q) => p[0] - q[0]); }
  touch();
};
document.onkeydown = (e) => {
  if (["INPUT", "TEXTAREA"].includes(e.target.tagName) && e.target.type !== "range") return;
  const step = e.shiftKey ? 10 : 1;
  if (e.key === "ArrowRight") frame = Math.min(data.frames - 1, frame + step);
  else if (e.key === "ArrowLeft") frame = Math.max(0, frame - step);
  else if (e.key.toLowerCase() === "x" && truth.animals[sel]) {
    const k = keyIndex(truth.animals[sel]);
    if (k >= 0) { truth.animals[sel].points.splice(k, 1); return touch(); }
  } else return;
  e.preventDefault();
  draw();
};
$("scrub").oninput = (e) => { frame = +e.target.value; draw(); };
$("add").onclick = () => { truth.animals.push({ note: "", points: [] }); sel = truth.animals.length - 1; touch(); };
$("del").onclick = () => { if (truth.animals[sel]) { truth.animals.splice(sel, 1); sel = Math.max(0, sel - 1); touch(); } };
$("anote").oninput = (e) => { if (truth.animals[sel]) { truth.animals[sel].note = e.target.value; dirty = true; $("saved").textContent = "Unsaved changes"; } };
$("cnote").oninput = (e) => { truth.note = e.target.value; dirty = true; $("saved").textContent = "Unsaved changes"; };
$("verified").onchange = (e) => { truth.verified = e.target.checked; touch(); };
$("save").onclick = save;
$("clip").onchange = (e) => load(e.target.value);
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "⏸" : "▶"; };
window.onbeforeunload = (e) => { if (dirty) e.preventDefault(); };

function tick(ms) {
  if (playing && data && ms - last > 1000 * (data.duration_s / data.frames)) {
    last = ms;
    frame = (frame + 1) % data.frames;
    draw();
  }
  requestAnimationFrame(tick);
}

const clips = await (await fetch("data/index.json")).json();
for (const c of clips) $("clip").add(new Option(c.replaceAll("_", " "), c));
requestAnimationFrame(tick);
load(clips[0]);
