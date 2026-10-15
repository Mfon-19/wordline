"""Clip -> worldline data: a frame atlas for the block and one tube per animal.

Every step is OpenCV 5: decode (VideoCapture), detect (cv2.dnn), find motion against a
median background (connected components), fill gaps between detections (Lucas-Kanade
optical flow), outline (GrabCut + findContours), measure.

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
WORK_W = 640       # width frames are analysed at
ATLAS_W = 256      # width of each frame stored for the browser block
MAX_FRAMES = 256   # frames kept along the time axis
MAX_FPS = 10
RING = 48          # points per outline
GRAB_MAX = 128     # longest side GrabCut is run at
CONF = 0.3         # detector confidence threshold
IOU_MATCH = 0.2    # overlap needed to attach a detection to a track
MAX_MISS = 8       # sampled frames a track survives on optical flow alone
MIN_DETS = 3       # detections that make a track detector-confirmed
MIN_MOTION = 5     # confirmed frames a motion-only track needs
MIN_TRAVEL = 0.15  # start-to-end distance (frame widths) a motion-only track needs
FG_THRESH = 25     # grey-level difference from the background that counts as change
NOISE_K = 6.0      # ...or this many times the pixel's own usual flicker, if larger
MIN_BLOB = 0.0006  # smallest moving region, as a share of the frame
ALIVE_FG = 0.05    # share of a track's box that must be changing for motion to keep it alive
STALL = 15         # frames a detected track may be kept alive by motion without moving or being re-detected
STALL_MOVE = 0.05  # ...where moving means this far (frame widths) from where it was last detected
MIN_FG = 0.02      # below this mean share of changed pixels, and
MIN_PATH = 0.05    # below this travel (frame widths), a track is scenery (a rock, a log)
MERGE_GAP = 10     # sampled frames a tube may vanish for and still be joined up
CUT_FACTOR = 5     # a frame-wide change this many times the clip's usual one is a cut
CUT_MIN = 4.0      # ...and at least this many grey levels


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def centre(box):
    return np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])


def inside(box, other):
    """True if the centre of box lies inside other."""
    cx, cy = centre(box)
    return other[0] <= cx <= other[0] + other[2] and other[1] <= cy <= other[1] + other[3]


def share(mask, box):
    """Fraction of the box covered by the mask."""
    x, y, w, h = (int(v) for v in box)
    roi = mask[max(y, 0):y + h, max(x, 0):x + w]
    return float(roi.mean()) if roi.size else 0.0


def read_clip(path):
    """Decode and sample the clip. Returns fps, source frame count, times, work frames, atlas frames."""
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
    times, work, small, next_sample, i = [], [], [], 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= next_sample:
            next_sample += step
            times.append(i / fps)
            H0, W0 = frame.shape[:2]
            work.append(cv2.resize(frame, (WORK_W, round(H0 * WORK_W / W0))))
            small.append(cv2.resize(frame, (ATLAS_W, round(H0 * ATLAS_W / W0 / 2) * 2), interpolation=cv2.INTER_AREA))
        i += 1
    return fps, total, times, work, small


def foreground(grays):
    """Two per-frame masks of pixels that differ from the clip's median background.

    loose: a fixed threshold. Used to tell whether a known animal's box is still changing.
    strict: the threshold also scales with each pixel's own flicker over the clip, so water,
    reeds and leaves need a bigger change to count. Used to start tracks from motion alone.
    """
    blur = np.array([cv2.GaussianBlur(g, (5, 5), 0) for g in grays])
    bg = np.median(blur, axis=0).astype(np.uint8)
    diff = np.array([cv2.absdiff(b, bg) for b in blur])
    noise = 1.4826 * np.median(diff, axis=0)  # robust spread of each pixel over time
    kernel = np.ones((3, 3), np.uint8)

    def masks(thresh):
        return [cv2.morphologyEx((d > thresh).astype(np.uint8), cv2.MORPH_OPEN, kernel) for d in diff]

    return masks(FG_THRESH), masks(np.maximum(FG_THRESH, NOISE_K * noise))


def find_cuts(grays):
    """Frame indices where a new recording starts (camera-trap clips are often several joined up)."""
    blur = [cv2.GaussianBlur(g, (5, 5), 0) for g in grays]
    change = np.array([cv2.absdiff(a, b).mean() for a, b in zip(blur, blur[1:])])
    if not len(change):
        return set()
    return set((np.flatnonzero(change > max(CUT_MIN, CUT_FACTOR * np.median(change))) + 1).tolist())


def motion_blobs(mask):
    """Boxes of connected moving regions; none if most of the frame changed (lighting shift)."""
    if mask.mean() > 0.3:
        return []
    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, np.ones((9, 9), np.uint8)))
    return [s[:4].astype(float) for s in stats[1:] if s[4] >= MIN_BLOB * mask.size]


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


def track_clip(work, grays, fg, fg_strict, cuts, det):
    """Detect and follow objects. Returns raw tracks, detector time (ms), people skipped."""
    live, done, det_ms, people = [], [], 0.0, 0

    def new_track(f, box, hit, conf=None, label=None):
        live.append({"start": f, "box": np.asarray(box, float), "miss": 0, "hit": hit, "motion_born": hit == "m", "anchor": centre(box), "stall": 0,
                     "conf": [conf] if conf else [], "guesses": Counter([label] if label else []),
                     "boxes": [], "fg": [], "src": ""})

    for f, frame in enumerate(work):
        # 1. carry every live track forward with optical flow; nothing survives a cut
        if f in cuts:
            done += live
            live = []
        for tr in live:
            shift = flow_shift(grays[f - 1], grays[f], tr["box"])
            if shift is not None:
                tr["box"] = tr["box"] + np.array([shift[0], shift[1], 0, 0])
            tr["miss"] += 1
            tr["hit"] = ""

        # 2. detect, and attach detections to the tracks they overlap
        t0 = time.time()
        dets = det(frame, conf=CONF)
        det_ms += (time.time() - t0) * 1000
        people += sum(1 for d in dets if d[2] == "person")
        dets = [d for d in dets if d[2] in ANIMALS]
        pairs = sorted(((iou(tr["box"], d[0]), ti, di) for ti, tr in enumerate(live) for di, d in enumerate(dets)), reverse=True)
        used = set()
        for score, ti, di in pairs:
            tr = live[ti]
            if score < IOU_MATCH or tr["hit"] or di in used:
                continue
            used.add(di)
            box, conf, label = dets[di]
            tr["box"], tr["miss"], tr["hit"] = box.astype(float), 0, "d"
            tr["anchor"], tr["stall"] = centre(tr["box"]), 0
            tr["conf"].append(conf)
            tr["guesses"][label] += 1
        for di, (box, conf, label) in enumerate(dets):
            if di not in used:
                new_track(f, box, "d", conf, label)

        # 3. motion the detector did not explain keeps a track alive, or starts a candidate one
        for blob in motion_blobs(fg_strict[f]):
            owner = next((tr for tr in live if inside(blob, tr["box"]) or iou(blob, tr["box"]) > 0.1), None)
            if owner is None:
                new_track(f, blob, "m")
            elif owner["motion_born"] and not owner["hit"]:
                owner["box"], owner["miss"], owner["hit"] = blob, 0, "m"
        for tr in live:
            if tr["hit"] or tr["motion_born"] or share(fg[f], tr["box"]) < ALIVE_FG:
                continue
            # a detected animal whose box is still changing stays alive, but not for long if it
            # is going nowhere: a parked box that only flickers is usually leaves, not an animal
            if np.linalg.norm(centre(tr["box"]) - tr["anchor"]) > STALL_MOVE * frame.shape[1]:
                tr["anchor"], tr["stall"] = centre(tr["box"]), 0
            tr["stall"] += 1
            if tr["stall"] <= STALL:
                tr["miss"], tr["hit"] = 0, "m"

        # 4. log this frame; retire tracks nothing has confirmed for a while
        still = []
        for tr in live:
            if tr["miss"] > MAX_MISS:
                done.append(tr)
                continue
            tr["boxes"].append(tr["box"].copy())
            tr["fg"].append(share(fg[f], tr["box"]))
            tr["src"] += tr["hit"] or "f"
            still.append(tr)
        live = still
    return done + live, det_ms, people


def select(raw, width, cuts):
    """Trim flow-only tails, keep real animals, join broken tubes. Distances in frame widths."""
    def travel(a, b):
        return float(np.linalg.norm(centre(a) - centre(b))) / width

    kept = []
    for tr in raw:
        n = len(tr["src"].rstrip("f"))
        src, boxes, fgs = tr["src"][:n], tr["boxes"][:n], tr["fg"][:n]
        if n < 2:
            continue
        dets = src.count("d")
        path = sum(travel(a, b) for a, b in zip(boxes, boxes[1:]))
        if dets >= MIN_DETS:
            # a detection that never moves and never differs from the background is scenery
            if np.mean(fgs) < MIN_FG and path < MIN_PATH:
                continue
        elif n - src.count("f") < MIN_MOTION or (dets < 2 and travel(boxes[0], boxes[-1]) < MIN_TRAVEL):
            continue
        kept.append({"start": tr["start"], "end": tr["start"] + n - 1, "src": src, "boxes": boxes, "fg": fgs,
                     "conf": tr["conf"], "guesses": tr["guesses"]})

    kept.sort(key=lambda t: t["start"])
    merged = []
    for t in kept:
        best = None
        for m in merged:
            gap = t["start"] - m["end"]
            dist = travel(m["boxes"][-1], t["boxes"][0])
            reach = 1.5 * max(m["boxes"][-1][2:].max(), t["boxes"][0][2:].max()) / width
            cut = any(m["end"] < c <= t["start"] for c in cuts)
            if 0 < gap <= MERGE_GAP and not cut and dist < reach and (best is None or dist < best[0]):
                best = (dist, m, gap)
        if best is None:
            merged.append(t)
            continue
        _, m, gap = best
        a, b = m["boxes"][-1], t["boxes"][0]
        m["boxes"] += [a + (b - a) * g / gap for g in range(1, gap)] + t["boxes"]  # bridge in a straight line
        m["fg"] += [0.0] * (gap - 1) + t["fg"]
        m["src"] += "g" * (gap - 1) + t["src"]
        m["end"], m["conf"], m["guesses"] = t["end"], m["conf"] + t["conf"], m["guesses"] + t["guesses"]
    return merged


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
    mid = np.array([x + a, y + b], float)
    ellipse = a * b / np.sqrt((b * np.cos(angles)) ** 2 + (a * np.sin(angles)) ** 2)
    if crop.shape[0] < 12 or crop.shape[1] < 12:
        return mid, ellipse
    k = min(1.0, GRAB_MAX / max(crop.shape[:2]))  # GrabCut cost grows with area; work on a small copy
    if k < 1.0:
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        rect = tuple(max(int(v * k), 1) for v in rect)
    mask = np.zeros(crop.shape[:2], np.uint8)
    try:
        cv2.grabCut(crop, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return mid, ellipse
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mid, ellipse
    c = max(contours, key=cv2.contourArea).reshape(-1, 2) / k
    if not 0.15 < cv2.contourArea(c.astype(np.float32)) / (w * h + 1e-9) < 0.95:
        return mid, ellipse
    m = cv2.moments(c.astype(np.float32))
    mid = np.array([m["m10"] / m["m00"] + x0, m["m01"] / m["m00"] + y0])
    d = c + (x0, y0) - mid
    bins = (np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi) / (2 * np.pi) * RING).astype(int) % RING
    radii = np.zeros(RING)
    np.maximum.at(radii, bins, np.hypot(d[:, 0], d[:, 1]))
    have = radii > 0
    return mid, np.interp(np.arange(RING), np.flatnonzero(have), radii[have], period=RING)


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

    fps, total, times, work, small_frames = read_clip(path)
    grays = [cv2.cvtColor(w, cv2.COLOR_BGR2GRAY) for w in work]
    cuts = find_cuts(grays)
    fg, fg_strict = foreground(grays)
    raw, det_ms, people_ignored = track_clip(work, grays, fg, fg_strict, cuts, Detector())

    T = len(times)
    H, W = work[0].shape[:2]
    tracks = []
    for t in select(raw, W, cuts):
        n = len(t["src"])
        shapes = [outline(work[t["start"] + k], box) for k, box in enumerate(t["boxes"])]
        cx = smooth([c[0] / W for c, _ in shapes])
        cy = smooth([c[1] / H for c, _ in shapes])
        r = smooth(np.array([radii / W for _, radii in shapes]))
        dets = t["src"].count("d")
        tracks.append({
            "id": len(tracks) + 1, "label": "animal", "found_by": "detector" if dets >= MIN_DETS else "motion",
            "guesses": dict(t["guesses"].most_common(3)),
            "start": t["start"], "end": t["end"],
            "enter_s": round(times[t["start"]], 2), "exit_s": round(times[t["end"]], 2),
            "dwell_s": round(times[t["end"]] - times[t["start"]], 2),
            "path_len": round(float(np.hypot(np.diff(cx), np.diff(cy) * H / W).sum()), 3),
            "conf": round(float(np.mean(t["conf"])), 3) if t["conf"] else None,
            "det_share": round(dets / n, 3), "motion": round(float(np.mean(t["fg"])), 3), "src": t["src"],
            "cx": np.round(cx, 4).tolist(), "cy": np.round(cy, 4).tolist(), "r": np.round(r, 4).tolist(),
        })

    # frame atlas for the browser
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
        "cuts": sorted(cuts), "tracks": tracks,
        "stats": {"detector_ms_per_frame": round(det_ms / max(T, 1), 1), "people_ignored": people_ignored,
                  "raw_tracks": len(raw), "total_s": round(time.time() - t_start, 1)},
    }
    (out_dir / "worldline.json").write_text(json.dumps(meta, separators=(",", ":")))
    done = sorted(p.parent.name for p in out_dir.parent.glob("*/worldline.json"))
    (out_dir.parent / "index.json").write_text(json.dumps(done))
    return meta


if __name__ == "__main__":
    for p in sys.argv[1:]:
        m = run(p)
        print(f"{m['clip']}: {m['frames']} frames, {len(m['tracks'])} tubes from {m['stats']['raw_tracks']} candidates, "
              f"{m['stats']['total_s']} s (detector {m['stats']['detector_ms_per_frame']} ms/frame)")
        for t in m["tracks"]:
            print(f"  #{t['id']} {t['enter_s']:6.2f}s -> {t['exit_s']:6.2f}s  path {t['path_len']:.2f}  motion {t['motion']:.2f}  "
                  f"by {t['found_by']:8s} detected {t['det_share']:.0%}  guesses {list(t['guesses'])}")
