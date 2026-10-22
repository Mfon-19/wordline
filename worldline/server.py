"""Worldline web server: the viewer, search by typed words, and (optionally) label editing.

GET  /api/health                  model and data status
GET  /api/search?q=badger         every tube in every processed clip, best match first
POST /api/agent {question}         ask the agent; returns a run id
GET  /api/agent/{id}               the run's status, trace, pending approval and answer
POST /api/agent/{id}/decision      approve or decline a pending reprocessing request
GET  /truth/{clip}.json, PUT ...  ground-truth labels, only with --labels (local use)
GET  /                            the viewer (web/)

Usage: python -m worldline.server [--port 8765] [--labels]
"""
import argparse
import json
import logging
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import agent, embed

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DATA = WEB / "data"
TRUTH = ROOT / "eval" / "truth"
CLIP = re.compile(r"^[a-z0-9_]+$")
MAX_QUERY = 200

app = FastAPI(title="Worldline", docs_url=None, redoc_url=None)
state = {"ready": False, "error": None, "index": {}, "mtimes": {}}
agents = agent.Agents()


def load_model():
    try:
        embed.model()
        embed.embed_query("warm up")
        state["ready"] = True
    except Exception as e:  # reported by /api/health instead of crashing the viewer
        state["error"] = f"{type(e).__name__}: {e}"


def index():
    """Tube vectors and timings for every processed clip, reloaded when files change."""
    for f in DATA.glob("*/embeddings.json"):
        clip, mtime = f.parent.name, f.stat().st_mtime
        if state["mtimes"].get(clip) == mtime:
            continue
        meta = json.loads((f.parent / "worldline.json").read_text())
        tubes = {t["id"]: t for t in meta["tracks"]}
        state["index"][clip] = [(tid, vec, tubes[tid]) for tid, vec in embed.load(f).items() if tid in tubes]
        state["mtimes"][clip] = mtime
    return state["index"]


@app.get("/api/health")
def health():
    return {"model": embed.MODEL_ID, "ready": state["ready"], "error": state["error"],
            "clips": len(index()), "tubes": sum(len(v) for v in index().values())}


@app.get("/api/search")
def search(q: str = Query(min_length=1, max_length=MAX_QUERY)):
    if not state["ready"]:
        raise HTTPException(503, state["error"] or "search model is still loading")
    t0 = time.time()
    qv = embed.embed_query(q.strip())
    results = [{"clip": clip, "tube": tid, "score": round(float(vec @ qv), 4),
                "enter_s": t["enter_s"], "exit_s": t["exit_s"]}
               for clip, rows in index().items() for tid, vec, t in rows]
    results.sort(key=lambda r: -r["score"])
    return {"query": q, "ms": round(1000 * (time.time() - t0)), "results": results}


@app.post("/api/agent")
async def ask(request: Request):
    body = await request.json()
    question = str(body.get("question", "")).strip()
    if not 3 <= len(question) <= 500:
        raise HTTPException(400, "ask a question of 3 to 500 characters")
    if not state["ready"]:
        raise HTTPException(503, "search model is still loading")
    try:
        run = agents.start(question)
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    return {"id": run.id}


@app.get("/api/agent/{run_id}")
def run_status(run_id: str):
    run = agents.runs.get(run_id)
    if not run:
        raise HTTPException(404, "no such run")
    return JSONResponse(run.view(), headers={"Cache-Control": "no-store"})


@app.post("/api/agent/{run_id}/decision")
async def decision(run_id: str, request: Request):
    run = agents.runs.get(run_id)
    if not run:
        raise HTTPException(404, "no such run")
    body = await request.json()
    if not run.decide(bool(body.get("approve")), str(body.get("note", ""))):
        raise HTTPException(409, "nothing is waiting for approval")
    return {"ok": True}


def enable_labels():
    """Read and save ground-truth labels. For local labelling only: never expose publicly."""
    @app.get("/truth/{clip}.json")
    def get_truth(clip: str):
        if not CLIP.match(clip):
            raise HTTPException(400, "bad clip name")
        f = TRUTH / f"{clip}.json"
        return JSONResponse(json.loads(f.read_text()) if f.exists() else {"clip": clip, "verified": False, "animals": []},
                            headers={"Cache-Control": "no-store"})

    @app.put("/truth/{clip}.json")
    async def put_truth(clip: str, request: Request):
        body = await request.body()
        if not CLIP.match(clip) or not 0 < len(body) < 1_000_000:
            raise HTTPException(400, "bad request")
        try:
            data = json.loads(body)
            assert isinstance(data.get("animals"), list)
        except (ValueError, AssertionError, AttributeError):
            raise HTTPException(400, "bad json")
        TRUTH.mkdir(parents=True, exist_ok=True)
        (TRUTH / f"{clip}.json").write_text(json.dumps(data, indent=1))
        return {"ok": True}


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--labels", action="store_true", help="allow editing ground-truth labels (local only)")
    args = ap.parse_args()
    if args.labels:
        enable_labels()
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    threading.Thread(target=load_model, daemon=True).start()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
