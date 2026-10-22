"""Shared test helpers. Tests that need the downloaded clips, models or processed data skip
themselves when those files are missing (run scripts/fetch_assets.py to get them)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def textured_square_video(path, frames=40, size=(320, 240), step=6, fps=10):
    """A grey scene with a textured square moving left to right: something the motion stages must find."""
    rng = np.random.default_rng(0)
    tex = rng.integers(40, 255, (50, 50, 3), dtype=np.uint8)
    bg = np.full((size[1], size[0], 3), 90, np.uint8)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        f = bg.copy()
        x = 10 + i * step
        f[100:150, x:x + 50] = tex
        w.write(f)
    w.release()
    return path


@pytest.fixture
def square_video(tmp_path):
    return textured_square_video(tmp_path / "square.mp4")


def need(*paths):
    missing = [p for p in paths if not (ROOT / p).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"missing {missing}; run scripts/fetch_assets.py")
