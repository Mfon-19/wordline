"""Worldline web server: the viewer, search by typed words, and (optionally) label editing.

GET  /api/health                  model and data status
GET  /api/search?q=badger         every tube in every processed clip, best match first
POST /api/upload                   upload a clip (raw body, X-Filename header); returns a job id
GET  /api/upload/{id}              processing status of an upload
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

from . import agent, embed, uploads

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DATA = WEB / "data"
TRUTH = ROOT / "eval" / "truth"
CLIP = re.compile(r"^[a-z0-9_]+$")
MAX_QUERY = 200

app = FastAPI(title="Worldline", docs_url=None, redoc_url=None)
state = {"ready": False, "error": None, "index": {}, "mtimes": {}}
agents = agent.Agents()
jobs = uploads.Uploads()
JOB = re.compile(r"^[a-z0-9]{8,32}$")


def load_model():
    try:
        embed.model()
        embed.embed_query("warm up")
        embed.alternatives()
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


def upload_rows(job_id):
    """Tube vectors for one private upload, if it exists and is processed."""
    d = DATA / "_uploads" / job_id
    if not JOB.match(job_id) or not (d / "embeddings.json").exists():
        return {}
    meta = json.loads((d / "worldline.json").read_text())
    tubes = {t["id"]: t for t in meta["tracks"]}
    return {f"_uploads/{job_id}": [(tid, vec, tubes[tid]) for tid, vec in embed.load(d / "embeddings.json").items() if tid in tubes]}


@app.get("/api/search")
def search(q: str = Query(min_length=1, max_length=MAX_QUERY), include: str = ""):
    if not state["ready"]:
        raise HTTPException(503, state["error"] or "search model is still loading")
    t0 = time.time()
    q = q.strip()
    qv = embed.embed_query(q)
    results = []
    for clip, rows in {**index(), **(upload_rows(include) if include else {})}.items():
        for tid, vec, t in rows:
            m = embed.margin(vec, qv, q)
            results.append({"clip": clip, "tube": tid, "score": round(float(vec @ qv), 4), "margin": round(m, 4),
                            "confident": m > 0, "enter_s": t["enter_s"], "exit_s": t["exit_s"]})
    results.sort(key=lambda r: -r["score"])
    return {"query": q, "ms": round(1000 * (time.time() - t0)), "results": results}


def visitor(request: Request):
    """The visitor's address. CloudFront appends the real one as the last X-Forwarded-For entry;
    earlier entries come from the visitor and can be forged."""
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[-1].strip() or (request.client.host if request.client else "unknown")


@app.post("/api/upload")
async def upload(request: Request):
    if request.query_params.get("agree") != "1":
        raise HTTPException(400, "please confirm you have the right to share this footage")
    size = int(request.headers.get("content-length") or 0)
    if size > uploads.MAX_BYTES:
        raise HTTPException(413, f"Clips can be up to {uploads.MAX_BYTES // 2**20} MB.")
    try:
        jobs.admit(visitor(request))
    except uploads.Refused as e:
        raise HTTPException(429, str(e))
    dst, written = jobs.new_path(request.headers.get("x-filename")), 0
    with open(dst, "wb") as f:
        async for chunk in request.stream():
            written += len(chunk)
            if written > uploads.MAX_BYTES:
                f.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"Clips can be up to {uploads.MAX_BYTES // 2**20} MB.")
            f.write(chunk)
    if written == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "the upload was empty")
    try:
        job_id = jobs.add(visitor(request), dst)
    except uploads.Refused as e:
        raise HTTPException(400, str(e))
    return {"id": job_id}


@app.get("/api/upload/{job_id}")
def upload_status(job_id: str):
    job = jobs.view(job_id) if JOB.match(job_id) else None
    if not job:
        raise HTTPException(404, "no such upload (uploads are deleted after 24 hours)")
    return JSONResponse(job, headers={"Cache-Control": "no-store"})


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


@app.middleware("http")
async def cache_headers(request: Request, call_next):
    """Code and clip lists must be fresh after a deploy; frame images can be cached for a while."""
    response = await call_next(request)
    path = request.url.path
    if "cache-control" not in response.headers:
        if path.endswith((".jpg", ".png")):
            response.headers["Cache-Control"] = "public, max-age=3600"
        elif not path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
    return response


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
