"""Probe: run YOLOX (via cv2.dnn) and MOG2 background subtraction over a clip.

Prints what the detector calls each animal and how much of the clip has motion,
and saves an annotated preview frame. Exploratory, not part of the pipeline.

Usage: python scripts/probe.py clips/red_fox.webm
"""
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from worldline.detector import Detector  # noqa: E402


def main(path):
    det = Detector()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    labels, n, det_frames, motion_frames, t_det, best = Counter(), 0, 0, 0, 0.0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        mask = cv2.morphologyEx(bg.apply(small), cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = [c for c in contours if cv2.contourArea(c) > 0.002 * mask.size]
        motion_frames += bool(blobs) and n > 10
        if n % 5 == 0:
            t = time.time()
            dets = det(frame)
            t_det += time.time() - t
            det_frames += 1
            labels.update(d[2] for d in dets)
            if dets and (best is None or len(dets) > len(best[1])):
                best = (frame.copy(), dets, small.shape, blobs)
        n += 1
    print(f"{Path(path).name}: {n} frames @ {fps:.1f} fps, {frame_size(cap)}")
    print(f"  detector ran on {det_frames} frames, {1000 * t_det / max(det_frames, 1):.0f} ms each")
    print(f"  labels: {dict(labels)}")
    print(f"  frames with motion blobs: {motion_frames}/{n}")
    if best:
        img, dets, sshape, blobs = best
        k = img.shape[1] / sshape[1]
        for c in blobs:
            cv2.drawContours(img, [(c * k).astype(np.int32)], -1, (0, 255, 255), 2)
        for (x, y, w, h), s, name in dets:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{name} {s:.2f}", (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out = ROOT / "out" / (Path(path).stem + "_probe.jpg")
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1]))))
        print(f"  preview: {out.relative_to(ROOT)}")


def frame_size(cap):
    return f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
