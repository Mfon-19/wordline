"""Both detectors load in OpenCV 5 and find the animal in a real camera-trap frame."""
import pytest

from conftest import ROOT


@pytest.mark.parametrize("which,model", [("yolox", "models/yolox_s.onnx"), ("megadetector", "models/md_v5a_640.onnx")])
def test_detector_finds_the_coyote(which, model, monkeypatch):
    clip = ROOT / "clips/coyote_family.webm"
    if not (ROOT / model).exists() or not clip.exists():
        pytest.skip("needs downloaded clips and models (scripts/fetch_assets.py, scripts/export_megadetector.sh)")
    from worldline import detector
    from worldline.pipeline import read_clip
    monkeypatch.setenv("WORLDLINE_DETECTOR", which)
    det = detector.make_detector()
    assert type(det).__name__ == ("MegaDetector" if which == "megadetector" else "Detector")
    _, _, _, work, _ = read_clip(clip)
    boxes = [d for d in det(work[8], conf=0.3) if d[2] in detector.ANIMALS]
    assert boxes, "no animal found in a frame with a coyote in plain view"
    x, y, w, h = boxes[0][0]
    assert 0.25 < (x + w / 2) / work[8].shape[1] < 0.65   # the coyote is left of centre


def test_falls_back_to_yolox_when_megadetector_missing(monkeypatch, tmp_path):
    from worldline import detector
    if not detector.MODEL.exists():
        pytest.skip("needs models/yolox_s.onnx")
    monkeypatch.setattr(detector, "MEGADETECTOR", tmp_path / "missing.onnx")
    monkeypatch.setenv("WORLDLINE_DETECTOR", "megadetector")
    assert type(detector.make_detector()).__name__ == "Detector"
