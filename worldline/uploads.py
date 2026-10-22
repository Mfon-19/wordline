"""Clips uploaded by visitors: validated, processed one at a time, kept private, deleted after a day.

Each upload gets an unguessable id. Its results live under web/data/_uploads/<id>/ and are never
listed in the shared clip index, so only someone with the link can open them. The source video is
kept (for the agent's reprocessing) until the results expire.
"""
import json
import logging
import queue
import secrets
import shutil
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

from . import pipeline

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUT = ROOT / "web" / "data" / "_uploads"
MAX_BYTES = 50 * 1024 * 1024
MAX_SECONDS = 60
MAX_PIXELS = 3840 * 2160
KEEP_HOURS = 24
QUEUE_LIMIT = 3          # waiting uploads before new ones are refused
PER_VISITOR_HOUR = 3     # uploads per visitor (IP) per hour
PER_DAY = 40             # uploads per day, all visitors
EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".ogv"}

log = logging.getLogger("worldline.uploads")


class Refused(Exception):
    """An upload we won't take; the message is safe to show the visitor."""


class Uploads:
    def __init__(self):
        UPLOADS.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        self.jobs, self.lock = {}, threading.Lock()
        self.todo = queue.Queue()
        self.recent = defaultdict(deque)   # visitor -> upload times in the last hour
        self.day, self.today = time.strftime("%Y-%m-%d"), 0
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._cleaner, daemon=True).start()

    def admit(self, visitor):
        """Check the limits before reading an upload's body."""
        with self.lock:
            now = time.time()
            if time.strftime("%Y-%m-%d") != self.day:
                self.day, self.today = time.strftime("%Y-%m-%d"), 0
            q = self.recent[visitor]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= PER_VISITOR_HOUR:
                raise Refused(f"You can upload {PER_VISITOR_HOUR} clips an hour. Please try again later.")
            if self.today >= PER_DAY:
                raise Refused("The daily upload limit has been reached. Please try again tomorrow.")
            if self.todo.qsize() >= QUEUE_LIMIT:
                raise Refused("Several clips are already being processed. Please try again in a few minutes.")

    def add(self, visitor, src):
        """Validate a saved upload and queue it. Returns the job id."""
        cap = cv2.VideoCapture(str(src))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        ok, _ = cap.read()
        cap.release()
        if not ok or fps <= 0 or w <= 0:
            src.unlink(missing_ok=True)
            raise Refused("That file couldn't be read as a video. Try an MP4 or WebM file.")
        if frames > 0 and frames / fps > MAX_SECONDS + 1:
            src.unlink(missing_ok=True)
            raise Refused(f"Clips can be up to {MAX_SECONDS} seconds long; this one is {frames / fps:.0f} s.")
        if w * h > MAX_PIXELS:
            src.unlink(missing_ok=True)
            raise Refused("Clips can be up to 4K resolution.")
        job_id = src.stem
        with self.lock:
            self.recent[visitor].append(time.time())
            self.today += 1
            self.jobs[job_id] = {"id": job_id, "status": "queued", "created": time.time(), "error": None,
                                 "clip": f"_uploads/{job_id}", "seconds": None}
        self.todo.put((job_id, src))
        log.info(json.dumps({"upload": job_id, "event": "queued", "bytes": src.stat().st_size,
                             "duration_s": round(frames / fps, 1) if frames > 0 else None}))
        return job_id

    def view(self, job_id):
        job = self.jobs.get(job_id)
        if not job:
            return None
        ahead = sum(1 for j in self.jobs.values() if j["status"] == "queued" and j["created"] < job["created"])
        return {**job, "ahead": ahead}

    @staticmethod
    def new_path(filename):
        ext = Path(filename or "").suffix.lower()
        return UPLOADS / f"{secrets.token_urlsafe(12).replace('-', 'x').replace('_', 'y').lower()}{ext if ext in EXTENSIONS else '.mp4'}"

    @staticmethod
    def source(job_id):
        return next(UPLOADS.glob(f"{job_id}.*"), None)

    def _worker(self):
        while True:
            job_id, src = self.todo.get()
            job = self.jobs[job_id]
            job["status"] = "processing"
            t0 = time.time()
            try:
                with pipeline.settings():  # one pipeline run at a time, shared with the agent
                    meta = pipeline.run(src, out_dir=OUT / job_id, hide_people=True)
                job.update(status="done", seconds=round(time.time() - t0, 1), tubes=len(meta["tracks"]),
                           people_blurred=meta["stats"]["people_hidden"])
            except Exception:  # report the failure to the visitor; never crash the worker
                log.exception("upload %s failed", job_id)
                job.update(status="error", error="Processing failed. The video may use an unusual format.")
                shutil.rmtree(OUT / job_id, ignore_errors=True)
            log.info(json.dumps({"upload": job_id, "event": job["status"], "seconds": round(time.time() - t0, 1)}))

    def _cleaner(self):
        while True:
            cutoff = time.time() - KEEP_HOURS * 3600
            for d in list(OUT.iterdir()) + list(UPLOADS.iterdir()):
                try:
                    if d.stat().st_mtime < cutoff:
                        shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink(missing_ok=True)
                        self.jobs.pop(d.stem, None)
                except FileNotFoundError:
                    pass
            time.sleep(900)
