"""Worldline agent: answers questions about camera-trap footage by driving the OpenCV pipeline.

Perceive: search the tubes, inspect one (measurements plus OpenCV crops the model can look at),
or look at a whole frame. Decide: Claude (on Amazon Bedrock) plans the next step from what it saw.
Act: re-run the OpenCV pipeline on a clip or a time window with more sensitive settings, which
costs CPU time and therefore waits for a person to approve it.

Every step is recorded as a trace event (kept with the run and logged as a JSON line).
"""
import base64
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

import anthropic
import cv2
import numpy as np

from . import embed

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "web" / "data"
CLIPS = ROOT / "clips"
MODEL = os.environ.get("WORLDLINE_AGENT_MODEL", "anthropic.claude-opus-5-5")
REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_TURNS = 14           # model calls per run
MAX_REPROCESS = 2        # re-runs of the OpenCV pipeline per run
APPROVAL_TIMEOUT = 600   # seconds to wait for a person before treating a request as declined
MAX_RUNS_PER_DAY = int(os.environ.get("WORLDLINE_AGENT_DAILY_RUNS", "25"))
NAME = re.compile(r"^[a-z0-9_]+(/[a-z0-9_]+)*$")

log = logging.getLogger("worldline.agent")

SYSTEM = """You help ecologists answer questions about camera-trap footage that has been processed by \
an OpenCV pipeline. Each clip has been turned into "tubes": one tube per animal visit, built from \
detections, background subtraction and optical flow, with an outline per frame. Each tube has an \
embedding of crops of the animal, so it can be searched by typed words.

How to work:
- Start by searching or listing clips; then inspect the most promising tubes. Inspecting returns \
measurements and crops of the animal: look at the crops before trusting a match. Search scores are \
relative, not probabilities; the top result can still be wrong.
- If the tubes don't answer the question (nothing convincing found, or a tube looks broken or \
cut short), look at frames with view_frame. If you can see an animal the pipeline missed, or a \
visit seems split, call reprocess_clip with higher sensitivity and, ideally, a time window. A \
person must approve each reprocessing; give a short, specific reason. If they decline, carry on \
with what you have.
- Never name a species from the detector's own labels; it only knows everyday categories. Name a \
species only if the crops show it, and say how sure you are.
- People are deliberately not tracked by this system. Do not try to find, follow or describe people.

Answer in a few sentences: what you found, when (seconds into the clip), how long it stayed, and \
how sure you are. Refer to tubes as clip#tube (for example coyote_family#3) so the viewer can link \
them. If you could not answer, say what you tried and what would help."""

TOOLS = [
    {
        "name": "list_clips",
        "description": "List the processed clips: duration, number of tubes, and where recordings were joined (cuts).",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "search_footage",
        "description": "Rank tubes by how well their animal crops match a description (e.g. 'badger', 'bird on water'). "
                       "Returns clip, tube id, match score, entry/exit times and how the tube was found.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in a few words."},
                "clip": {"type": ["string", "null"], "description": "Limit to one clip, or null for all clips."},
                "limit": {"type": "integer", "description": "How many results, 1 to 10."},
            },
            "required": ["query", "clip", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "inspect_tube",
        "description": "Measurements for one tube (entry, exit, dwell, distance moved, path through the frame, how much "
                       "of it came from the detector versus motion versus optical flow) plus crops of the animal at its "
                       "start, middle and end.",
        "input_schema": {
            "type": "object",
            "properties": {"clip": {"type": "string"}, "tube": {"type": "integer"}},
            "required": ["clip", "tube"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "view_frame",
        "description": "The whole frame of a clip at a time, and which tubes (if any) cover it. Use it to check for "
                       "animals the pipeline may have missed.",
        "input_schema": {
            "type": "object",
            "properties": {"clip": {"type": "string"}, "time_s": {"type": "number"}},
            "required": ["clip", "time_s"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "reprocess_clip",
        "description": "Re-run the OpenCV pipeline on a clip (optionally only a time window, sampled more densely) with "
                       "more sensitive settings, to recover missed or broken tubes. Takes up to a minute of CPU and "
                       "needs a person's approval. Returns the new tubes, under a new clip name you can inspect.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clip": {"type": "string"},
                "reason": {"type": "string", "description": "Why, in one sentence, for the person approving it."},
                "sensitivity": {"type": "string", "enum": ["normal", "high"]},
                "start_s": {"type": ["number", "null"]},
                "end_s": {"type": ["number", "null"]},
            },
            "required": ["clip", "reason", "sensitivity", "start_s", "end_s"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# Settings for reprocessing: "high" lowers the detector threshold and the motion thresholds so small,
# dim or still animals survive; a time window also samples more densely.
SENSITIVE = {"CONF": 0.15, "NOISE_K": 3.0, "MIN_BLOB": 0.0003, "MIN_DETS": 2, "MIN_TRAVEL": 0.08}


class ToolError(Exception):
    pass


def load_meta(clip):
    if not NAME.match(clip):
        raise ToolError(f"bad clip name {clip!r}")
    f = DATA / clip / "worldline.json"
    if not f.exists():
        raise ToolError(f"no clip called {clip!r}; use list_clips")
    return json.loads(f.read_text())


def tube_of(meta, tube):
    for t in meta["tracks"]:
        if t["id"] == tube:
            return t
    raise ToolError(f"clip has no tube {tube}; it has {[t['id'] for t in meta['tracks']]}")


def atlas_frame(clip, meta, f):
    """One stored frame of the clip (from the atlas the viewer uses), as a BGR image."""
    atlas = cv2.imread(str(DATA / clip / meta["atlas"]["file"]))
    w, h, cols = meta["frame_w"], meta["frame_h"], meta["atlas"]["cols"]
    return atlas[f // cols * h:(f // cols + 1) * h, f % cols * w:(f % cols + 1) * w]


def jpeg_block(img, width=384):
    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.tobytes()).decode()}}


def region(x, y):
    col = "left" if x < 1 / 3 else "centre" if x < 2 / 3 else "right"
    row = "top" if y < 1 / 3 else "middle" if y < 2 / 3 else "bottom"
    return f"{row} {col}" if not (row == "middle" and col == "centre") else "centre"


def list_clips():
    out = []
    for f in sorted(DATA.glob("*/worldline.json")):
        m = json.loads(f.read_text())
        out.append({"clip": f.parent.name, "duration_s": m["duration_s"], "tubes": len(m["tracks"]),
                    "cuts_at_s": [m["times"][c] for c in m.get("cuts", [])]})
    return out


def search_footage(query, clip, limit):
    limit = int(min(max(limit, 1), 10))
    qv = embed.embed_query(query)
    rows = []
    dirs = [DATA / clip] if clip else [f.parent for f in DATA.glob("*/embeddings.json")]
    for d in dirs:
        if not (d / "embeddings.json").exists():
            continue
        meta = json.loads((d / "worldline.json").read_text())
        tubes = {t["id"]: t for t in meta["tracks"]}
        for tid, vec in embed.load(d / "embeddings.json").items():
            t = tubes.get(tid)
            if t:
                rows.append({"clip": d.name if d.parent == DATA else str(d.relative_to(DATA)), "tube": tid,
                             "score": round(float(vec @ qv), 3), "enter_s": t["enter_s"], "exit_s": t["exit_s"],
                             "found_by": t["found_by"]})
    rows.sort(key=lambda r: -r["score"])
    return rows[:limit]


def inspect_tube(clip, tube):
    meta = load_meta(clip)
    t = tube_of(meta, tube)
    n = len(t["cx"])
    marks = sorted({0, n // 2, n - 1})
    path = [{"t_s": meta["times"][t["start"] + k], "where": region(t["cx"][k], t["cy"][k]),
             "x": round(t["cx"][k], 2), "y": round(t["cy"][k], 2)} for k in np.linspace(0, n - 1, min(n, 6)).astype(int)]
    src = t.get("src", "")
    facts = {
        "clip": clip, "tube": tube, "enter_s": t["enter_s"], "exit_s": t["exit_s"], "dwell_s": t["dwell_s"],
        "moved_frame_widths": t["path_len"], "path": path,
        "evidence": {"detector_frames": src.count("d"), "motion_frames": src.count("m"),
                     "optical_flow_frames": src.count("f"), "bridged_gap_frames": src.count("g")},
        "changing_pixels_share": t.get("motion"), "found_by": t["found_by"],
        "note": "Crops follow, at the tube's start, middle and end. Detector class guesses are not species.",
    }
    blocks = [{"type": "text", "text": json.dumps(facts)}]
    W, H = meta["frame_w"], meta["frame_h"]
    for k in marks:
        img = atlas_frame(clip, meta, t["start"] + k)
        half = max(16, 1.4 * max(t["r"][k]) * W)
        cx, cy = t["cx"][k] * W, t["cy"][k] * H
        crop = img[int(max(0, cy - half)):int(min(H, cy + half)), int(max(0, cx - half)):int(min(W, cx + half))]
        if crop.size:
            blocks += [{"type": "text", "text": f"crop at {meta['times'][t['start'] + k]:.1f} s"}, jpeg_block(crop, 256)]
    return blocks


def view_frame(clip, time_s):
    meta = load_meta(clip)
    f = int(np.argmin(np.abs(np.asarray(meta["times"]) - time_s)))
    covering = [t["id"] for t in meta["tracks"] if t["start"] <= f <= t["end"]]
    text = {"clip": clip, "frame_time_s": meta["times"][f], "tubes_covering_this_frame": covering}
    return [{"type": "text", "text": json.dumps(text)}, jpeg_block(atlas_frame(clip, meta, f))]


def reprocess_clip(run_id, n, clip, sensitivity, start_s, end_s):
    from . import pipeline
    base = clip.split("/")[-1].split("__")[0]
    src = next(CLIPS.glob(base + ".*"), None)
    if src is None:
        raise ToolError(f"the source video for {base!r} is not on this server")
    name = f"_agent/{run_id}/{base}__{n}"
    overrides = dict(SENSITIVE) if sensitivity == "high" else {}
    if start_s is not None or end_s is not None:
        overrides["MAX_FPS"] = 15
    t0 = time.time()
    with pipeline.settings(**overrides):
        meta = pipeline.run(src, out_dir=DATA / name, start_s=start_s, end_s=end_s)
    return {"new_clip": name, "seconds": round(time.time() - t0, 1), "settings": overrides or "default",
            "window_s": [start_s, end_s],
            "tubes": [{"tube": t["id"], "enter_s": t["enter_s"], "exit_s": t["exit_s"], "dwell_s": t["dwell_s"],
                       "found_by": t["found_by"]} for t in meta["tracks"]],
            "note": "Inspect these with inspect_tube using the new clip name."}


class Run:
    """One question answered by the agent, in a background thread, with a full trace."""

    def __init__(self, question):
        self.id = uuid.uuid4().hex[:10]
        self.question = question
        self.status = "running"     # running | awaiting_approval | done | stopped | error
        self.events, self.answer, self.pending = [], None, None
        self.usage = {"input": 0, "output": 0, "cache_read": 0}
        self._decision, self._ready = None, threading.Event()
        self.started = time.time()

    def event(self, kind, **data):
        e = {"t": round(time.time() - self.started, 2), "kind": kind, **data}
        self.events.append(e)
        log.info(json.dumps({"run": self.id, **e}, default=str)[:4000])

    def decide(self, approve, note=""):
        if self.status != "awaiting_approval":
            return False
        self._decision = (bool(approve), note[:300])
        self._ready.set()
        return True

    def ask_approval(self, request):
        self.pending, self.status = request, "awaiting_approval"
        self.event("approval_request", **request)
        self._ready.clear()
        got = self._ready.wait(APPROVAL_TIMEOUT)
        approve, note = self._decision if got else (False, "no answer within the time limit")
        self.pending, self.status, self._decision = None, "running", None
        self.event("approval", approved=approve, note=note)
        return approve, note

    def view(self):
        return {"id": self.id, "question": self.question, "status": self.status, "answer": self.answer,
                "pending": self.pending, "events": self.events, "usage": self.usage, "model": MODEL}


def summarise(result):
    """Short text of a tool result for the trace (images are counted, not stored)."""
    if isinstance(result, list):
        texts = [b["text"] for b in result if b["type"] == "text"]
        return {"text": " | ".join(texts)[:1500], "images": sum(b["type"] == "image" for b in result)}
    return {"text": json.dumps(result, default=str)[:1500]}


def client():
    return anthropic.AnthropicBedrockMantle(aws_region=REGION)


def execute(run, block, reprocessed):
    """Run one tool call. Returns (content, is_error)."""
    name, args = block.name, block.input
    try:
        if name == "list_clips":
            return json.dumps(list_clips()), False
        if name == "search_footage":
            return json.dumps(search_footage(args["query"], args.get("clip"), args.get("limit", 5))), False
        if name == "inspect_tube":
            return inspect_tube(args["clip"], int(args["tube"])), False
        if name == "view_frame":
            return view_frame(args["clip"], float(args["time_s"])), False
        if name == "reprocess_clip":
            if reprocessed[0] >= MAX_REPROCESS:
                return f"Limit reached: at most {MAX_REPROCESS} reprocessing runs per question.", True
            load_meta(args["clip"].split("/")[-1].split("__")[0])
            approve, note = run.ask_approval({"tool": name, "clip": args["clip"], "reason": args["reason"],
                                              "sensitivity": args["sensitivity"],
                                              "window_s": [args.get("start_s"), args.get("end_s")]})
            if not approve:
                return f"The person declined this reprocessing ({note}). Continue with what you have.", True
            reprocessed[0] += 1
            return json.dumps(reprocess_clip(run.id, reprocessed[0], args["clip"], args["sensitivity"],
                                             args.get("start_s"), args.get("end_s"))), False
        return f"Unknown tool {name}", True
    except ToolError as e:
        return str(e), True
    except Exception as e:  # a failing tool must not kill the run; the model sees the error and adapts
        log.exception("tool %s failed", name)
        return f"{name} failed: {type(e).__name__}: {e}", True


def agent_loop(run):
    messages = [{"role": "user", "content": run.question}]
    reprocessed = [0]
    api = client()
    try:
        for turn in range(1, MAX_TURNS + 1):
            t0 = time.time()
            response = api.messages.create(
                model=MODEL, max_tokens=16000,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS, messages=messages,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": "medium"},
            )
            u = response.usage
            run.usage["input"] += u.input_tokens + (u.cache_creation_input_tokens or 0)
            run.usage["output"] += u.output_tokens
            run.usage["cache_read"] += u.cache_read_input_tokens or 0
            run.event("model", turn=turn, seconds=round(time.time() - t0, 1), stop=response.stop_reason,
                      thinking=" ".join(b.thinking for b in response.content if b.type == "thinking")[:1500],
                      text=" ".join(b.text for b in response.content if b.type == "text")[:2000],
                      calls=[{"tool": b.name, "input": b.input} for b in response.content if b.type == "tool_use"])
            # keep every block (including thinking) so the next request continues the same conversation
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "refusal":
                run.answer, run.status = "The model declined to answer this question.", "done"
                return
            calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not calls:
                run.answer = " ".join(b.text for b in response.content if b.type == "text").strip()
                run.status = "done"
                run.event("final", answer=run.answer)
                return
            results = []
            for block in calls:
                t1 = time.time()
                content, is_error = execute(run, block, reprocessed)
                run.event("tool_result", tool=block.name, error=is_error, seconds=round(time.time() - t1, 1),
                          **summarise(content))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error})
            messages.append({"role": "user", "content": results})
        run.status = "stopped"
        run.answer = f"Stopped after {MAX_TURNS} steps without a final answer."
        run.event("stopped", reason="step limit")
    except anthropic.APIStatusError as e:
        detail = e.body.get("error", {}).get("message") if isinstance(e.body, dict) else None
        run.status, run.answer = "error", f"The model service refused the request ({e.status_code}): {detail or e.message}"
        run.event("error", status=e.status_code, message=(detail or str(e.message))[:500])
    except anthropic.APIConnectionError as e:
        run.status, run.answer = "error", "Could not reach the model service."
        run.event("error", message=str(e)[:500])


class Agents:
    """Runs one question at a time and caps runs per day, to bound cost on a public endpoint."""

    def __init__(self):
        self.runs, self.lock = {}, threading.Lock()
        self.day, self.count = time.strftime("%Y-%m-%d"), 0

    def busy(self):
        return any(r.status in ("running", "awaiting_approval") for r in self.runs.values())

    def start(self, question):
        with self.lock:
            today = time.strftime("%Y-%m-%d")
            if today != self.day:
                self.day, self.count = today, 0
            if self.busy():
                raise RuntimeError("another question is being answered; try again shortly")
            if self.count >= MAX_RUNS_PER_DAY:
                raise RuntimeError("the daily limit of agent questions has been reached")
            self.count += 1
            run = Run(question)
            self.runs[run.id] = run
            run.event("question", question=question, model=MODEL)
        threading.Thread(target=agent_loop, args=(run,), daemon=True).start()
        return run
