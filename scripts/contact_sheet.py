"""Contact sheet: outlines from worldline.json drawn on sampled frames, for checking tubes by eye.

Usage: python scripts/contact_sheet.py coyote_family [tiles]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COLORS = [(84, 180, 255), (230, 207, 92), (126, 230, 186), (121, 135, 242), (255, 191, 212)]


def main(name, tiles=20):
    d = json.loads((ROOT / "web" / "data" / name / "worldline.json").read_text())
    atlas = cv2.imread(str(ROOT / "web" / "data" / name / "atlas.jpg"))
    w, h, cols = d["frame_w"], d["frame_h"], d["atlas"]["cols"]
    out = []
    for f in np.linspace(0, d["frames"] - 1, tiles).astype(int):
        img = cv2.resize(atlas[f // cols * h:(f // cols + 1) * h, f % cols * w:(f % cols + 1) * w], (w * 2, h * 2))
        for t in d["tracks"]:
            if not t["start"] <= f <= t["end"]:
                continue
            k = f - t["start"]
            ang = np.linspace(0, 2 * np.pi, d["ring"], endpoint=False)
            r = np.array(t["r"][k])
            pts = np.stack(((t["cx"][k] + r * np.cos(ang)) * w * 2, (t["cy"][k] * h + r * np.sin(ang) * w) * 2), 1)
            cv2.polylines(img, [pts.astype(np.int32)], True, COLORS[(t["id"] - 1) % len(COLORS)], 2)
            cv2.putText(img, f"#{t['id']}{t['src'][k]}", tuple(pts[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"{d['times'][f]:.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        out.append(img)
    while len(out) % 5:
        out.append(np.zeros_like(out[0]))
    sheet = np.vstack([np.hstack(out[i:i + 5]) for i in range(0, len(out), 5)])
    dst = ROOT / "out" / f"{name}_sheet.jpg"
    dst.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(dst), sheet)
    print(dst.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
