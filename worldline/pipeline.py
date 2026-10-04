"""Clip -> worldline data: a frame atlas for the block and one tube per animal.

Every step is OpenCV 5: decode (VideoCapture), detect (cv2.dnn), fill gaps between
detections (Lucas-Kanade optical flow), outline (GrabCut + findContours), measure.

Usage: python -m worldline.pipeline clips/coyote_family.webm
"""
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .detector import ANIMALS, Detector

ROOT = Path(__file__).resolve().parent.parent
WORK_W = 640      # width frames are analysed at
ATLAS_W = 256     # width of each frame stored for the browser block
MAX_FRAMES = 256  # frames kept along the time axis
MAX_FPS = 10
RING = 48         # points per outline
MAX_MISS = 8      # sampled frames a track survives on optical flow alone
MIN_DETS = 3      # detections a track needs to be kept
CONF = 0.3
IOU_MATCH = 0.2


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def flow_shift(prev, cur, box):
    """Median Lucas-Kanade displacement of corner features inside box, or None."""
    x, y, w, h = (int(v) for v in box)
    mask = np.zeros_like(prev)
    mask[max(y, 0):y + h, max(x, 0):x + w] = 255
    pts = cv2.goodFeaturesToTrack(prev, 60, 0.01, 4, mask=mask)
    if pts is None or len(pts) < 4:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    good = st.ravel() == 1
    if good.sum() < 4:
        return None
    return np.median((nxt - pts)[good].reshape(-1, 2), axis=0)


def outline(frame, box):
    """Centre and RING polar radii of the object in box, in frame pixels."""
    H, W = frame.shape[:2]
    x, y, w, h = box
    px, py = 0.15 * w, 0.15 * h
    x0, y0 = int(max(x - px, 0)), int(max(y - py, 0))
    x1, y1 = int(min(x + w + px, W)), int(min(y + h + py, H))
    crop = frame[y0:y1, x0:x1]
    rect = (int(x - x0), int(y - y0), max(int(w), 2), max(int(h), 2))
    angles = np.linspace(0, 2 * np.pi, RING, endpoint=False)
    a, b = w / 2, h / 2
    centre = np.array([x + a, y + b], float)
    ellipse = a * b / np.sqrt((b * np.cos(angles)) ** 2 + (a * np.sin(angles)) ** 2)
    if crop.shape[0] < 12 or crop.shape[1] < 12:
        return centre, ellipse
    mask = np.zeros(crop.shape[:2], np.uint8)
    try:
        cv2.grabCut(crop, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return centre, ellipse
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return centre, ellipse
    c = max(contours, key=cv2.contourArea)
    share = cv2.contourArea(c) / (w * h + 1e-9)
    if not 0.15 < share < 0.95:
        return centre, ellipse
    m = cv2.moments(c)
    centre = np.array([m["m10"] / m["m00"] + x0, m["m01"] / m["m00"] + y0])
    d = c.reshape(-1, 2) + (x0, y0) - centre
    bins = (np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi) / (2 * np.pi) * RING).astype(int) % RING
    radii = np.zeros(RING)
    np.maximum.at(radii, bins, np.hypot(d[:, 0], d[:, 1]))
    have = radii > 0
    radii = np.interp(np.arange(RING), np.flatnonzero(have), radii[have], period=RING)
    return centre, radii


def smooth(a):
    """3-frame moving average along time."""
    a = np.asarray(a, float)
    if len(a) < 3:
        return a
    p = np.concatenate((a[:1], a, a[-1:]))
    return (p[:-2] + p[1:-1] + p[2:]) / 3


def run(path, out_dir=None):
    path = Path(path)
    out_dir = Path(out_dir) if out_dir else ROOT / "web" / "data" / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        while cap.grab():
            total += 1
        cap = cv2.VideoCapture(str(path))
    step = max(fps / MAX_FPS, total / MAX_FRAMES, 1.0)

    det = Detector()
    live, done, small_frames, times = [], [], [], []
    prev_gray, next_sample, i, next_id = None, 0.0, 0, 1
    det_ms, people_ignored = 0.0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i < next_sample:
            i += 1
            continue
        next_sample += step
        f = len(times)
        times.append(i / fps)
        i += 1

        H0, W0 = frame.shape[:2]
        work = cv2.resize(frame, (WORK_W, round(H0 * WORK_W / W0)))
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        small_frames.append(cv2.resize(frame, (ATLAS_W, round(H0 * ATLAS_W / W0 / 2) * 2), interpolation=cv2.INTER_AREA))

        # 1. carry every live track forward with optical flow
        for tr in live:
            shift = flow_shift(prev_gray, gray, tr["box"])
            if shift is not None:
                tr["box"] = tr["box"] + np.array([shift[0], shift[1], 0, 0])
            tr["miss"] += 1

        # 2. detect, and attach detections to the tracks they overlap
        t0 = time.time()
        dets = det(work, conf=CONF)
        det_ms += (time.time() - t0) * 1000
        people_ignored += sum(1 for d in dets if d[2] == "person")
        dets = [d for d in dets if d[2] in ANIMALS]
        pairs = sorted(((iou(tr["box"], d[0]), ti, di) for ti, tr in enumerate(live) for di, d in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < IOU_MATCH or ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr, (box, conf, label) = live[ti], dets[di]
            tr["box"], tr["miss"] = box.astype(float), 0
            tr["conf"].append(conf)
            tr["guesses"][label] += 1
        for di, (box, conf, label) in enumerate(dets):
            if di not in used_d:
                live.append({"id": next_id, "start": f, "box": box.astype(float), "miss": 0, "conf": [conf],
                             "guesses": Counter({label: 1}), "cx": [], "cy": [], "r": [], "src": ""})
                next_id += 1

        # 3. outline each track in this frame; retire tracks the detector has lost
        still = []
        for tr in live:
            if tr["miss"] > MAX_MISS:
                done.append(tr)
                continue
            centre, radii = outline(work, tr["box"])
            tr["cx"].append(centre[0] / work.shape[1])
            tr["cy"].append(centre[1] / work.shape[0])
            tr["r"].append(radii / work.shape[1])
            tr["src"] += "d" if tr["miss"] == 0 else "f"
            still.append(tr)
        live = still
        prev_gray = gray
    done += live

    # 4. trim flow-only tails, drop flickers, measure
    T = len(times)
    aspect = small_frames[0].shape[1] / small_frames[0].shape[0]
    tracks = []
    for tr in done:
        n = len(tr["src"].rstrip("f"))
        if tr["src"].count("d") < MIN_DETS or n < 2:
            continue
        cx, cy, r = smooth(tr["cx"][:n]), smooth(tr["cy"][:n]), smooth(np.array(tr["r"][:n]))
        end = tr["start"] + n - 1
        path_len = float(np.hypot(np.diff(cx), np.diff(cy) / aspect).sum())
        tracks.append({
            "id": len(tracks) + 1, "label": "animal", "guesses": dict(tr["guesses"].most_common(3)),
            "start": tr["start"], "end": end,
            "enter_s": round(times[tr["start"]], 2), "exit_s": round(times[end], 2),
            "dwell_s": round(times[end] - times[tr["start"]], 2),
            "path_len": round(path_len, 3), "conf": round(float(np.mean(tr["conf"])), 3),
            "det_share": round(tr["src"][:n].count("d") / n, 3),
            "cx": np.round(cx, 4).tolist(), "cy": np.round(cy, 4).tolist(), "r": np.round(r, 4).tolist(),
        })

    # 5. frame atlas for the browser
    fh, fw = small_frames[0].shape[:2]
    cols = 4096 // fw
    rows = math.ceil(T / cols)
    atlas = np.zeros((rows * fh, cols * fw, 3), np.uint8)
    for k, sf in enumerate(small_frames):
        atlas[k // cols * fh:(k // cols + 1) * fh, k % cols * fw:(k % cols + 1) * fw] = sf
    cv2.imwrite(str(out_dir / "atlas.jpg"), atlas, [cv2.IMWRITE_JPEG_QUALITY, 85])

    meta = {
        "clip": path.name, "opencv": cv2.__version__, "source_fps": round(fps, 3), "source_frames": total,
        "duration_s": round(total / fps, 2), "frames": T, "times": [round(t, 3) for t in times],
        "frame_w": fw, "frame_h": fh, "ring": RING,
        "atlas": {"file": "atlas.jpg", "cols": cols, "rows": rows},
        "tracks": tracks,
        "stats": {"detector_ms_per_frame": round(det_ms / max(T, 1), 1), "people_ignored": people_ignored,
                  "total_s": round(time.time() - t_start, 1)},
    }
    (out_dir / "worldline.json").write_text(json.dumps(meta, separators=(",", ":")))
    return meta


if __name__ == "__main__":
    for p in sys.argv[1:]:
        m = run(p)
        print(f"{m['clip']}: {m['frames']} frames, {len(m['tracks'])} tubes, {m['stats']['total_s']} s "
              f"(detector {m['stats']['detector_ms_per_frame']} ms/frame)")
        for t in m["tracks"]:
            print(f"  #{t['id']} {t['enter_s']:6.2f}s -> {t['exit_s']:6.2f}s  dwell {t['dwell_s']:5.2f}s  "
                  f"path {t['path_len']:.2f}  conf {t['conf']:.2f}  detected {t['det_share']:.0%}  guesses {t['guesses']}")
