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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from worldline import embed  # noqa: E402
from worldline.evaluate import covers, truth_positions  # noqa: E402

DISTRACTORS = ["domestic cat", "dog", "rabbit", "raccoon", "bear", "horse", "hedgehog", "squirrel"]


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
    path = next((ROOT / "clips").glob(clip + ".*"))
    return embed.tube_images(path, data, tb), embed.tube_images(path, data, tb, crop=False)


def score(name, tube_vecs, truth, vocab, text):
    sims = embed.unit(tube_vecs) @ text.T                       # tubes x vocabulary
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
    model = embed.model()
    rows = labelled_tubes()
    truth = [s for *_, s in rows]
    vocab = sorted(set(truth)) + DISTRACTORS
    text = embed.unit(model.encode(vocab, prompt_name="SearchQuery"))
    crop_vecs, whole_vecs, t_img, n_img = [], [], 0.0, 0
    for clip, data, tb, s in rows:
        crops, whole = crops_for(clip, data, tb)
        t = time.time()
        crop_vecs.append(embed.unit(model.encode(crops)).mean(0))
        whole_vecs.append(embed.unit(model.encode(whole)).mean(0))
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
