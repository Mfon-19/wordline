"""Trial: can EmbeddingGemma 2 name the animal in a tube from a typed word?

For every tube that matches a labelled animal, OpenCV crops the animal from a few of its
frames (using the tube's outline), EmbeddingGemma 2 embeds the crops, and the crops are
compared with typed species names. Also run on the whole frame, to see what cropping adds.

Usage: python scripts/embed_trial.py
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from worldline.evaluate import covers, truth_positions  # noqa: E402

CROPS = 4
DISTRACTORS = ["domestic cat", "dog", "rabbit", "raccoon", "bear", "horse", "hedgehog", "squirrel"]


def frames_at(path, indices):
    want, out, i = set(indices), {}, 0
    cap = cv2.VideoCapture(str(path))
    while want - out.keys():
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = f
        i += 1
    return out


def labelled_tubes():
    """(clip, tube, species) for every tube whose frames mostly cover one labelled animal."""
    rows = []
    for tf in sorted((ROOT / "eval" / "truth").glob("*.json")):
        truth = json.loads(tf.read_text())
        data = json.loads((ROOT / "web" / "data" / tf.stem / "worldline.json").read_text())
        aspect = data["frame_w"] / data["frame_h"]
        animals = [(a.get("species"), truth_positions(a, data["times"])) for a in truth["animals"]]
        for tb in data["tracks"]:
            frames = range(tb["start"], tb["end"] + 1)
            hits = [sum(not np.isnan(xy[f, 0]) and covers(tb, f, xy[f], aspect) for f in frames) for _, xy in animals]
            if hits and max(hits) >= 0.3 * len(frames) and animals[int(np.argmax(hits))][0]:
                rows.append((tf.stem, data, tb, animals[int(np.argmax(hits))][0]))
    return rows


def crops_for(clip, data, tb):
    """Crops around the animal, and the whole frame, at CROPS of the tube's confirmed frames."""
    good = [tb["start"] + k for k, c in enumerate(tb["src"]) if c in "dm"] or list(range(tb["start"], tb["end"] + 1))
    picks = [good[int(i)] for i in np.linspace(0, len(good) - 1, CROPS)]
    fps = data["source_fps"]
    src = frames_at(next((ROOT / "clips").glob(clip + ".*")), [round(data["times"][f] * fps) for f in picks])
    crops, whole = [], []
    for f in picks:
        img = src.get(round(data["times"][f] * fps))
        if img is None:
            continue
        H, W = img.shape[:2]
        k = f - tb["start"]
        cx, cy, half = tb["cx"][k] * W, tb["cy"][k] * H, max(48, 1.3 * max(tb["r"][k]) * W)
        x0, y0, x1, y1 = (int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half)))
        crops.append(Image.fromarray(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)))
        whole.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return crops, whole


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def score(name, tube_vecs, truth, vocab, text):
    sims = unit(tube_vecs) @ text.T                       # tubes x vocabulary
    order = np.argsort(-sims, 1)
    top1 = np.mean([vocab[o[0]] == s for o, s in zip(order, truth)])
    top3 = np.mean([s in [vocab[j] for j in o[:3]] for o, s in zip(order, truth)])
    hits = []  # for each species typed, is the best-matching tube across all clips that species?
    for j, s in enumerate(vocab):
        if s in truth:
            hits.append(truth[int(np.argmax(sims[:, j]))] == s)
    print(f"{name:12s} name the tube: top-1 {top1:.0%}  top-3 {top3:.0%}   search by name: right tube first {np.mean(hits):.0%} ({sum(hits)}/{len(hits)})")
    return {"top1": top1, "top3": top3, "search_hit1": float(np.mean(hits)), "searches": len(hits),
            "predicted": [vocab[o[0]] for o in order]}


def main():
    t0 = time.time()
    model = SentenceTransformer("google/embeddinggemma-2", config_kwargs={"audio_config": None},
                                model_kwargs={"torch_dtype": torch.float32})
    rows = labelled_tubes()
    truth = [s for *_, s in rows]
    vocab = sorted(set(truth)) + DISTRACTORS
    text = unit(model.encode(vocab, prompt_name="SearchQuery"))
    crop_vecs, whole_vecs, t_img, n_img = [], [], 0.0, 0
    for clip, data, tb, s in rows:
        crops, whole = crops_for(clip, data, tb)
        t = time.time()
        crop_vecs.append(unit(model.encode(crops)).mean(0))
        whole_vecs.append(unit(model.encode(whole)).mean(0))
        t_img += time.time() - t
        n_img += len(crops) + len(whole)
    print(f"{len(rows)} labelled tubes from {len({r[0] for r in rows})} clips, {len(set(truth))} species, "
          f"{len(vocab)} names offered ({len(DISTRACTORS)} not present); {1000 * t_img / n_img:.0f} ms per image on CPU")
    res = {"crop": score("cropped", np.array(crop_vecs), truth, vocab, text),
           "whole": score("whole frame", np.array(whole_vecs), truth, vocab, text)}
    for (clip, _, tb, s), p, q in zip(rows, res["crop"]["predicted"], res["whole"]["predicted"]):
        print(f"  {clip:15s} tube {tb['id']}: truth {s:13s} cropped -> {p:13s} whole -> {q}")
    out = {"tubes": [{"clip": c, "tube": tb["id"], "species": s} for c, _, tb, s in rows], "vocab": vocab, **res,
           "ms_per_image": 1000 * t_img / n_img, "total_s": time.time() - t0}
    (ROOT / "eval" / "embed_trial.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
