from __future__ import annotations

import time

import numpy as np

from .base import Detector, empty_dets, make_dets


class UltralyticsDetector(Detector):
    def __init__(self, name: str, weights: str, device: str = "cpu", imgsz: int = 640, score_thresh: float = 0.001,
                 max_det: int = 100, iou: float = 0.7, end2end: bool = False):
        super().__init__()
        from ultralytics import YOLO

        self.name, self.weights, self.device = name, weights, device
        self.imgsz, self.score_thresh, self.max_det, self.iou, self.end2end = imgsz, score_thresh, max_det, iou, end2end
        self.model = YOLO(weights)
        self.names = self.model.names

    def predict(self, image: np.ndarray) -> dict:
        t0 = time.perf_counter()
        h, w = image.shape[:2]
        bgr = np.ascontiguousarray(image[..., ::-1])  # ultralytics treats ndarray input as BGR
        kw = dict(imgsz=self.imgsz, conf=self.score_thresh, iou=self.iou, max_det=self.max_det,
                  device=self.device, verbose=False)
        if self.end2end:
            kw["nms"] = False
        r = self.model.predict(bgr, **kw)[0]
        sp = r.speed
        dev_sync = self.device
        if dev_sync.startswith(("mps", "cuda")):
            from ..timing import sync
            sync(dev_sync)
        self.last_timing = {"preprocess_ms": sp["preprocess"], "forward_ms": sp["inference"],
                            "postprocess_ms": sp["postprocess"], "total_ms": (time.perf_counter() - t0) * 1e3}
        if r.boxes is None or len(r.boxes) == 0:
            return empty_dets()
        b = r.boxes
        labels = [self.names[int(c)] for c in b.cls.cpu().numpy()]
        return make_dets(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), labels, self.max_det, (h, w))

    def _config(self) -> dict:
        return {
            "checkpoint": self.weights,
            "input": f"letterbox, long side {self.imgsz}, minimal stride-32 padding (predict-style, rect)",
            "nms_iou": "none (end-to-end one-to-one head, nms=False)" if self.end2end else self.iou,
            "head": "end2end" if self.end2end else "one2many + NMS",
            "timing_note": "stage split from ultralytics Results.speed; total_ms includes RGB->BGR copy",
        }
