"""The scoring rules, on hand-made tubes and labels."""
import numpy as np
import pytest

from worldline import evaluate


def data(tracks, frames=20):
    return {"times": [i * 0.5 for i in range(frames)], "frame_w": 160, "frame_h": 90, "tracks": tracks}


def tube(tid, start, end, x=0.5, y=0.5, r=0.1):
    n = end - start + 1
    return {"id": tid, "start": start, "end": end, "cx": [x] * n, "cy": [y] * n, "r": [[r] * 8] * n}


def test_truth_positions_interpolate_and_end():
    xy = evaluate.truth_positions({"points": [[1.0, 0.2, 0.5], [3.0, 0.6, 0.5]]}, [0, 1, 2, 3, 4])
    assert np.isnan(xy[0, 0]) and np.isnan(xy[4, 0])
    assert xy[2, 0] == pytest.approx(0.4)


def test_a_matching_tube_finds_the_animal():
    truth = {"animals": [{"points": [[1.0, 0.5, 0.5], [4.0, 0.5, 0.5]]}]}
    s = evaluate.score_clip(data([tube(1, 2, 8)]), truth)
    assert s["found"] == 1 and s["false_tubes"] == 0 and s["fragments"] == [1]
    assert s["enter_err"] == [pytest.approx(0.0)]


def test_a_tube_elsewhere_is_false_and_the_animal_missed():
    truth = {"animals": [{"points": [[1.0, 0.2, 0.2], [4.0, 0.2, 0.2]]}]}
    s = evaluate.score_clip(data([tube(1, 2, 8, x=0.8, y=0.8)]), truth)
    assert s["found"] == 0 and s["false_tubes"] == 1


def test_empty_clip_counts_skipped_frames():
    s = evaluate.score_clip(data([]), {"animals": []})
    assert s["tn"] == 20 and s["fp"] == 0
    s = evaluate.score_clip(data([tube(1, 0, 9)]), {"animals": []})
    assert s["fp"] == 10 and s["tn"] == 10


def test_summary_ratios():
    clip = evaluate.score_clip(data([tube(1, 2, 8)]), {"animals": [{"points": [[1.0, 0.5, 0.5], [4.0, 0.5, 0.5]]}]})
    total = evaluate.summarise({"a": clip})
    assert total["animal_recall"] == 1.0 and total["tube_precision"] == 1.0
