#!/bin/bash
# Convert the official MegaDetector v5a weights to ONNX at 640 px, for OpenCV's DNN module.
#
# Uses a throwaway Python 3.11 environment with YOLOv5 v7.0's export tool (AGPL-3.0). The tool
# only runs here, at build time; Worldline itself runs the exported model with OpenCV.
# MegaDetector: https://github.com/agentmorris/MegaDetector (code MIT; see its licence notes).
#
# Usage: scripts/export_megadetector.sh      (needs uv: https://docs.astral.sh/uv/)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=models/md_v5a_640.onnx
[ -f "$OUT" ] && { echo "$OUT already exists"; exit 0; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git clone -q --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git "$WORK/yolov5"
uv venv -q -p 3.11 "$WORK/venv"
uv pip install -q -p "$WORK/venv/bin/python" "torch==2.2.2" "torchvision==0.17.2" "numpy<2" "onnx<1.17" \
  opencv-python-headless pandas pyyaml requests tqdm matplotlib seaborn pillow scipy psutil ipython thop "setuptools<70"
curl -sL -o "$WORK/md_v5a.0.1.pt" https://huggingface.co/agentmorris/megadetector/resolve/main/md_v5a.0.1.pt
echo "fe3e90e4b1955821  (first 16 hex digits of the expected SHA-256 of md_v5a.0.1.pt)"
shasum -a 256 "$WORK/md_v5a.0.1.pt"
(cd "$WORK" && venv/bin/python yolov5/export.py --weights md_v5a.0.1.pt --imgsz 640 --include onnx --opset 17 --device cpu)
mkdir -p models
mv "$WORK/md_v5a.0.1.onnx" "$OUT"
echo "wrote $OUT"
