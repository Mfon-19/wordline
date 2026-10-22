"""YOLOX-S object detector run through OpenCV 5's DNN module (CPU)."""
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "models" / "yolox_s.onnx"
SIZE = 640
COCO = ("person bicycle car motorcycle airplane bus train truck boat trafficlight firehydrant "
        "stopsign parkingmeter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
        "backpack umbrella handbag tie suitcase frisbee skis snowboard sportsball kite baseballbat "
        "baseballglove skateboard surfboard tennisracket bottle wineglass cup fork knife spoon bowl "
        "banana apple sandwich orange broccoli carrot hotdog pizza donut cake chair couch "
        "pottedplant bed diningtable toilet tv laptop mouse remote keyboard cellphone microwave "
        "oven toaster sink refrigerator book clock vase scissors teddybear hairdrier toothbrush").split()
ANIMALS = set(COCO[14:24])


def _grid():
    grids, strides = [], []
    for s in (8, 16, 32):
        n = SIZE // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids), np.concatenate(strides)


class Detector:
    def __init__(self, model=MODEL):
        # The classic engine gives identical detections for this model and runs about twice as
        # fast on Graviton2 (396 vs 845 ms per frame, measured on the deployed t4g.large).
        self.net = cv2.dnn.readNetFromONNX(str(model), cv2.dnn.ENGINE_CLASSIC)
        self.grid, self.stride = _grid()

    def __call__(self, frame, conf=0.4, nms=0.5):
        """Return [((x, y, w, h), score, label)] in frame pixels."""
        h, w = frame.shape[:2]
        r = min(SIZE / h, SIZE / w)
        canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
        canvas[: int(h * r), : int(w * r)] = cv2.resize(frame, (int(w * r), int(h * r)))
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1.0, (SIZE, SIZE), swapRB=True))
        out = self.net.forward()[0]
        xy = (out[:, :2] + self.grid) * self.stride
        wh = np.exp(out[:, 2:4]) * self.stride
        scores = out[:, 4:5] * out[:, 5:]
        cls, sc = scores.argmax(1), scores.max(1)
        boxes = np.concatenate((xy - wh / 2, wh), 1) / r
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), sc.tolist(), conf, nms)
        return [(boxes[i].astype(int), float(sc[i]), COCO[cls[i]]) for i in np.array(keep).flatten()]
