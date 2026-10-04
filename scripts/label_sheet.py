"""Raw frames (no outlines) with a quarter grid, for drafting ground-truth labels.

Usage: python scripts/label_sheet.py coyote_family [tiles]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main(name, tiles=None):
    d = json.loads((ROOT / "web" / "data" / name / "worldline.json").read_text())
    atlas = cv2.imread(str(ROOT / "web" / "data" / name / "atlas.jpg"))
    w, h, cols = d["frame_w"], d["frame_h"], d["atlas"]["cols"]
    tiles = tiles or int(np.clip(d["duration_s"] / 0.6, 10, 30))
    tw = 400
    th = round(h * tw / w)
    out = []
    for f in np.linspace(0, d["frames"] - 1, tiles).astype(int):
        img = cv2.resize(atlas[f // cols * h:(f // cols + 1) * h, f % cols * w:(f % cols + 1) * w], (tw, th), interpolation=cv2.INTER_CUBIC)
        for q in (0.25, 0.5, 0.75):
            cv2.line(img, (int(q * tw), 0), (int(q * tw), th), (0, 200, 255), 1)
            cv2.line(img, (0, int(q * th)), (tw, int(q * th)), (0, 200, 255), 1)
        cv2.putText(img, f"{d['times'][f]:.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        out.append(img)
    per_row = 5 if th <= tw else 8
    while len(out) % per_row:
        out.append(np.zeros_like(out[0]))
    dst = ROOT / "out" / f"{name}_label.jpg"
    dst.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(dst), np.vstack([np.hstack(out[i:i + per_row]) for i in range(0, len(out), per_row)]))
    print(dst.relative_to(ROOT), tiles, "tiles")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
