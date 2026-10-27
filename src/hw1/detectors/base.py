from __future__ import annotations

import platform

import numpy as np
import torch
import torchvision

# COCO category names in id order; matches instances_val2017.json (HF RT-DETR id2label uses VOC-style spellings)
COCO80 = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]


def empty_dets() -> dict:
    return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32), "labels": []}


def make_dets(boxes, scores, labels: list[str], max_det: int, hw: tuple[int, int]) -> dict:
    """Sort by score, keep top max_det, clip boxes to the (h, w) image."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4).copy()
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, hw[1])
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, hw[0])
    scores = np.asarray(scores, np.float32).reshape(-1)
    order = np.argsort(-scores, kind="stable")[:max_det]
    return {"boxes": boxes[order], "scores": scores[order], "labels": [labels[i] for i in order]}


def lib_versions() -> dict:
    out = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__}
    for mod in ("ultralytics", "transformers"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return out


class Detector:
    name: str = ""
    open_vocab: bool = False
    device: str = "cpu"
    score_thresh: float = 0.001
    max_det: int = 100

    def __init__(self) -> None:
        self.last_timing: dict = {}

    def predict(self, image: np.ndarray) -> dict:
        """image: HxWx3 uint8 RGB. Returns boxes (N,4) xyxy original pixels, scores (N,), labels list[str]."""
        raise NotImplementedError

    def set_vocabulary(self, prompts: list[str]) -> None:
        raise NotImplementedError(f"{self.name} is closed-set")

    def config(self) -> dict:
        return {
            "name": self.name,
            "open_vocab": self.open_vocab,
            "device": self.device,
            "dtype": "float32",
            "score_thresh": self.score_thresh,
            "max_det": self.max_det,
            "versions": lib_versions(),
            **self._config(),
        }

    def _config(self) -> dict:
        return {}
