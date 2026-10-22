"""Search tubes by meaning: EmbeddingGemma 2 vectors for OpenCV-cropped tubes and typed queries.

Each tube is represented by the mean of its crops' embeddings: OpenCV cuts the animal out of
CROPS of the tube's confirmed frames, using the tube's outline. A typed query is embedded with
the model's search prompt and compared by cosine similarity.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MODEL_ID = "google/embeddinggemma-2"
CROPS = 4
PAD = 1.3        # crop half-size, as a multiple of the outline's largest radius
MIN_HALF = 48    # smallest crop half-size, in source pixels


@lru_cache(maxsize=1)
def model():
    import torch
    from sentence_transformers import SentenceTransformer
    # WORLDLINE_EMBED_DTYPE: float32 (default), bfloat16, or auto (bfloat16 when the CPU has native
    # bf16, such as Graviton4). float16 must not be used: the model's activations overflow it.
    choice = os.environ.get("WORLDLINE_EMBED_DTYPE", "float32")
    if choice == "auto":
        flags = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
        choice = "bfloat16" if " bf16" in flags else "float32"
    dtype = getattr(torch, choice)
    # images and text only: skipping the audio encoder saves 300M parameters
    return SentenceTransformer(MODEL_ID, config_kwargs={"audio_config": None}, model_kwargs={"torch_dtype": dtype})


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def frames_at(path, indices):
    """Decode only the source frames at the given indices."""
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


def picks(tube, n=CROPS):
    """Sampled-frame indices to crop: spread over frames where the animal was actually seen."""
    seen = [tube["start"] + k for k, c in enumerate(tube["src"]) if c in "dm"]
    seen = seen or list(range(tube["start"], tube["end"] + 1))
    return [seen[int(i)] for i in np.linspace(0, len(seen) - 1, n)]


def tube_images(clip_path, meta, tube, crop=True, frames=None):
    """RGB images of the tube's animal (or of the whole frame) at its picked frames."""
    from PIL import Image
    fps, chosen = meta["source_fps"], picks(tube)
    index = {f: round(meta["times"][f] * fps) for f in chosen}
    frames = frames if frames is not None else frames_at(clip_path, index.values())
    out = []
    for f in chosen:
        img = frames.get(index[f])
        if img is None:
            continue
        if crop:
            H, W = img.shape[:2]
            k = f - tube["start"]
            cx, cy = tube["cx"][k] * W, tube["cy"][k] * H
            half = max(MIN_HALF, PAD * max(tube["r"][k]) * W)
            img = img[int(max(0, cy - half)):int(min(H, cy + half)), int(max(0, cx - half)):int(min(W, cx + half))]
        out.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return out


def embed_tubes(clip_path, meta):
    """{tube id: unit vector} for every tube in the clip, decoding the clip once."""
    if not meta["tracks"]:
        return {}
    fps = meta["source_fps"]
    needed = {round(meta["times"][f] * fps) for t in meta["tracks"] for f in picks(t)}
    frames = frames_at(clip_path, needed)
    vecs = {}
    for t in meta["tracks"]:
        images = tube_images(clip_path, meta, t, frames=frames)
        if images:
            vecs[t["id"]] = unit(unit(model().encode(images).astype("float32")).mean(0))
    return vecs


def embed_query(text):
    return unit(model().encode([text], prompt_name="SearchQuery").astype("float32"))[0]


# Other things a tube could be. A result is "confident" only if the typed query fits the tube
# better than every one of these (names contained in the query, or containing it, are skipped).
ALTERNATIVES = ["bird", "duck", "waterbird", "deer", "fox", "coyote", "dog", "cat", "wild cat", "badger",
                "wild boar", "bear", "raccoon", "rabbit", "squirrel", "rodent", "horse", "cow", "sheep", "moose",
                "mountain lion", "weasel", "hedgehog", "otter", "empty ground", "plants moving"]


@lru_cache(maxsize=1)
def alternatives():
    return unit(model().encode(ALTERNATIVES, prompt_name="SearchQuery").astype("float32"))


def margin(tube_vec, query_vec, query):
    """How much better the query fits the tube than the best alternative name (positive = confident)."""
    q = query.lower()
    keep = [i for i, w in enumerate(ALTERNATIVES) if w not in q and q not in w]
    return float(tube_vec @ query_vec - (alternatives()[keep] @ tube_vec).max())


def save(path, vecs):
    Path(path).write_text(json.dumps({
        "model": MODEL_ID, "dim": 768, "crops": CROPS,
        "tubes": {str(k): np.round(v, 5).tolist() for k, v in vecs.items()},
    }, separators=(",", ":")))


def load(path):
    d = json.loads(Path(path).read_text())
    return {int(k): np.asarray(v, np.float32) for k, v in d["tubes"].items()}
