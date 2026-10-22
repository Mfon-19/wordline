"""The OpenCV stages of the pipeline, on synthetic video so no downloads are needed."""
from collections import Counter

import numpy as np
import pytest

from worldline import pipeline as p
from conftest import need, textured_square_video


def test_box_geometry():
    assert p.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert p.iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0
    assert p.inside((4, 4, 2, 2), (0, 0, 10, 10)) and not p.inside((40, 4, 2, 2), (0, 0, 10, 10))
    mask = np.zeros((10, 10), np.uint8)
    mask[:5] = 1
    assert p.share(mask, (0, 0, 10, 10)) == pytest.approx(0.5)


def test_smooth_keeps_length_and_constants():
    assert p.smooth([3, 3, 3, 3]).tolist() == [3, 3, 3, 3]
    assert len(p.smooth([1, 5, 2, 8, 3])) == 5


def test_read_clip_samples_and_windows(square_video):
    fps, total, times, work, small = p.read_clip(square_video)
    assert total == 40 and len(times) == len(work) == len(small)
    assert work[0].shape[1] == p.WORK_W and small[0].shape[1] == p.ATLAS_W
    _, _, wt, _, _ = p.read_clip(square_video, start_s=0.5, end_s=1.0)
    assert min(wt) >= 0.5 - 1e-6 and max(wt) <= 1.0 + 1e-6


def test_optical_flow_follows_motion(square_video):
    import cv2
    _, _, _, work, _ = p.read_clip(square_video)
    g0, g1 = (cv2.cvtColor(w, cv2.COLOR_BGR2GRAY) for w in work[:2])
    scale = p.WORK_W / 320
    dx, dy = p.flow_shift(g0, g1, np.array([10, 100, 50, 50]) * scale)
    assert dx == pytest.approx(6 * scale, abs=1.0) and abs(dy) < 1.0


def test_foreground_finds_the_moving_square(square_video):
    import cv2
    _, _, _, work, _ = p.read_clip(square_video)
    grays = [cv2.cvtColor(w, cv2.COLOR_BGR2GRAY) for w in work]
    loose, strict = p.foreground(grays)
    blobs = p.motion_blobs(strict[20])
    assert len(blobs) == 1
    x, y, w, h = blobs[0]
    scale = p.WORK_W / 320
    assert x <= (10 + 20 * 6) * scale <= x + w  # the square's left edge at frame 20


def test_cuts_are_found_at_scene_changes():
    frames = [np.full((60, 80), 50, np.uint8)] * 10 + [np.full((60, 80), 200, np.uint8)] * 10
    noisy = [f + np.random.default_rng(i).integers(0, 3, f.shape, dtype=np.uint8) for i, f in enumerate(frames)]
    assert p.find_cuts(noisy) == {10}


def track(start, boxes, src, fg=0.3):
    n = len(boxes)
    return {"start": start, "src": src, "boxes": [np.array(b, float) for b in boxes], "fg": [fg] * n,
            "fg_strict": [fg] * n, "conf": [0.8], "guesses": Counter()}


def test_select_drops_scenery_and_keeps_animals():
    still = track(0, [(100, 100, 20, 20)] * 10, "d" * 10, fg=0.0)        # never moves or changes
    animal = track(0, [(100 + 10 * i, 100, 20, 20) for i in range(10)], "d" * 10)
    kept = p.select([still, animal], 640, set())
    assert len(kept) == 1 and kept[0]["boxes"][-1][0] == 190


def test_select_joins_broken_tubes_but_not_across_cuts():
    a = track(0, [(100 + 5 * i, 100, 40, 40) for i in range(6)], "d" * 6)
    b = track(9, [(140 + 5 * i, 100, 40, 40) for i in range(6)], "d" * 6)
    assert len(p.select([a, b], 640, set())) == 1
    a = track(0, [(100 + 5 * i, 100, 40, 40) for i in range(6)], "d" * 6)
    b = track(9, [(140 + 5 * i, 100, 40, 40) for i in range(6)], "d" * 6)
    assert len(p.select([a, b], 640, {8})) == 2


def test_blur_people_scrambles_only_nearby_frames():
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (144, 256, 3), dtype=np.uint8) for _ in range(9)]
    before = [f.copy() for f in frames]
    p.blur_people(frames, [(4, (100, 60, 80, 100))], 256 / 640)
    changed = [i for i in range(9) if (frames[i] != before[i]).any()]
    assert changed == [2, 3, 4, 5, 6]
    assert frames[4][20:70, 36:76].std() < before[4][20:70, 36:76].std() / 3


def test_settings_restore_constants():
    old = p.CONF
    with p.settings(CONF=0.01):
        assert p.CONF == 0.01
    assert p.CONF == old


@need("models/yolox_s.onnx")
def test_run_end_to_end_on_synthetic_clip(tmp_path, monkeypatch):
    """The whole pipeline: a moving square has no detector class, so motion alone must make one tube."""
    monkeypatch.setenv("WORLDLINE_DETECTOR", "yolox")
    clip = textured_square_video(tmp_path / "clip.mp4", frames=60, step=4)
    meta = p.run(clip, out_dir=tmp_path / "out", embed=False)
    assert (tmp_path / "out" / "atlas.jpg").exists() and (tmp_path / "out" / "worldline.json").exists()
    assert len(meta["tracks"]) == 1
    t = meta["tracks"][0]
    assert t["found_by"] == "motion" and t["path_len"] > 0.3 and t["dwell_s"] > 1.5
    assert not (tmp_path / "index.json").exists()  # private outputs are never listed
