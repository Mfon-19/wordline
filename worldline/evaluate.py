"""Score the pipeline's tubes against ground truth in eval/truth/.

Ground truth is, per animal visit, a few (time, x, y) points; the animal is taken to be
present from its first point to its last, moving in straight lines between them.
A tube "covers" an animal in a frame when the animal's point lies within the tube's outline
(with a 25% margin).

Usage: python -m worldline.evaluate            (reads web/data/*/worldline.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MARGIN = 1.25      # slack on the tube's radius when testing whether it covers a point
FOUND = 0.5        # share of an animal's frames a tube must cover for it to count as found
FALSE_TUBE = 0.3   # a tube covering an animal in fewer of its frames than this is a false tube
MIN_LINK = 3       # frames of cover before a tube counts as one of an animal's fragments


def truth_positions(animal, times):
    """Per frame: (x, y) of the animal, or NaN where it is absent."""
    pts = np.array(sorted(animal["points"]), float)
    t = np.asarray(times)
    xy = np.stack((np.interp(t, pts[:, 0], pts[:, 1]), np.interp(t, pts[:, 0], pts[:, 2])), 1)
    step = np.median(np.diff(t)) if len(t) > 1 else 0
    xy[(t < pts[0, 0] - step / 2) | (t > pts[-1, 0] + step / 2)] = np.nan
    return xy


def covers(tube, f, xy, aspect):
    if not tube["start"] <= f <= tube["end"]:
        return False
    k = f - tube["start"]
    d = np.hypot(xy[0] - tube["cx"][k], (xy[1] - tube["cy"][k]) / aspect)
    return d <= MARGIN * max(tube["r"][k])


def score_clip(data, truth):
    times, tubes = data["times"], data["tracks"]
    aspect = data["frame_w"] / data["frame_h"]
    T = len(times)
    animals = [truth_positions(a, times) for a in truth["animals"]]
    gt_present = np.zeros(T, bool)
    for xy in animals:
        gt_present |= ~np.isnan(xy[:, 0])
    pred_present = np.zeros(T, bool)
    for tb in tubes:
        pred_present[tb["start"]:tb["end"] + 1] = True

    tube_hits = np.zeros(len(tubes), int)
    rows = []
    for xy in animals:
        frames = np.flatnonzero(~np.isnan(xy[:, 0]))
        per_tube = np.array([[covers(tb, f, xy[f], aspect) for f in frames] for tb in tubes], bool).reshape(len(tubes), len(frames))
        tube_hits += per_tube.sum(1)
        linked = np.flatnonzero(per_tube.sum(1) >= MIN_LINK)
        coverage = float(per_tube.any(0).mean()) if len(frames) else 0.0
        row = {"frames": len(frames), "coverage": coverage, "found": coverage >= FOUND, "fragments": len(linked)}
        if row["found"] and len(linked):
            row["enter_err"] = abs(min(times[tubes[i]["start"]] for i in linked) - times[frames[0]])
            row["exit_err"] = abs(max(times[tubes[i]["end"]] for i in linked) - times[frames[-1]])
        rows.append(row)
    lengths = np.array([tb["end"] - tb["start"] + 1 for tb in tubes])
    false_tubes = int((tube_hits / np.maximum(lengths, 1) < FALSE_TUBE).sum()) if len(tubes) else 0
    return {
        "frames": T, "tp": int((gt_present & pred_present).sum()), "fp": int((~gt_present & pred_present).sum()),
        "fn": int((gt_present & ~pred_present).sum()), "tn": int((~gt_present & ~pred_present).sum()),
        "animals": len(animals), "found": sum(r["found"] for r in rows),
        "coverage": [r["coverage"] for r in rows], "fragments": [r["fragments"] for r in rows if r["found"]],
        "enter_err": [r["enter_err"] for r in rows if "enter_err" in r],
        "exit_err": [r["exit_err"] for r in rows if "exit_err" in r],
        "tubes": len(tubes), "false_tubes": false_tubes, "verified": bool(truth.get("verified")),
    }


def ratio(a, b):
    return a / b if b else float("nan")


def summarise(scores):
    s = {k: sum(c[k] for c in scores.values()) for k in ("frames", "tp", "fp", "fn", "tn", "animals", "found", "tubes", "false_tubes")}
    cat = lambda k: [v for c in scores.values() for v in c[k]]
    return {
        "clips": len(scores), "verified_clips": sum(c["verified"] for c in scores.values()),
        "animals": s["animals"], "animals_found": s["found"], "animal_recall": ratio(s["found"], s["animals"]),
        "tubes": s["tubes"], "false_tubes": s["false_tubes"], "tube_precision": ratio(s["tubes"] - s["false_tubes"], s["tubes"]),
        "presence_precision": ratio(s["tp"], s["tp"] + s["fp"]), "presence_recall": ratio(s["tp"], s["tp"] + s["fn"]),
        "empty_correctly_skipped": ratio(s["tn"], s["tn"] + s["fp"]),
        "mean_coverage": float(np.mean(cat("coverage"))) if cat("coverage") else float("nan"),
        "tubes_per_found_animal": float(np.mean(cat("fragments"))) if cat("fragments") else float("nan"),
        "median_enter_err_s": float(np.median(cat("enter_err"))) if cat("enter_err") else float("nan"),
        "median_exit_err_s": float(np.median(cat("exit_err"))) if cat("exit_err") else float("nan"),
    }


def evaluate(data_dir=ROOT / "web" / "data", truth_dir=ROOT / "eval" / "truth"):
    scores = {}
    for tf in sorted(Path(truth_dir).glob("*.json")):
        df = Path(data_dir) / tf.stem / "worldline.json"
        if df.exists():
            scores[tf.stem] = score_clip(json.loads(df.read_text()), json.loads(tf.read_text()))
    return scores, summarise(scores)


def main():
    scores, total = evaluate()
    print(f"{'clip':16s} {'animals':>7s} {'found':>5s} {'tubes':>5s} {'false':>5s} {'pres.P':>6s} {'pres.R':>6s}  checked")
    for name, c in scores.items():
        p, r = ratio(c["tp"], c["tp"] + c["fp"]), ratio(c["tp"], c["tp"] + c["fn"])
        print(f"{name:16s} {c['animals']:7d} {c['found']:5d} {c['tubes']:5d} {c['false_tubes']:5d} {p:6.2f} {r:6.2f}  {'yes' if c['verified'] else 'draft'}")
    print()
    for k, v in total.items():
        print(f"{k:26s} {v:.3f}" if isinstance(v, float) else f"{k:26s} {v}")
    out = ROOT / "eval" / "results.json"
    out.write_text(json.dumps({"summary": total, "clips": scores}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
