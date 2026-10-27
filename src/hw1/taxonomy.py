"""KITTI target taxonomy and the mapping of detector outputs onto it.

Ground-truth rules (KITTI label type -> target class or ignore region) and prediction rules
(COCO class name or open-vocab prompt -> target class) live here, so every evaluation variant
is a pure function of raw predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torchvision.ops import nms

KITTI_CLASSES = {1: "car", 2: "truck", 3: "pedestrian", 4: "cyclist", 5: "tram"}
KITTI_IDS = {v: k for k, v in KITTI_CLASSES.items()}
CORE3 = [1, 2, 3]  # classes a COCO model can name directly


@dataclass(frozen=True)
class GTRules:
    """How KITTI label types become COCO-style GT. 'ignore' entries become iscrowd=1 regions."""

    van: str = "merge"  # "merge": Van is car GT; "ignore": Van is an ignore region for car
    cyclist_ignore_for_ped: bool = True  # riders detected as person are not pedestrian FPs
    dontcare_ignore: bool = True  # DontCare is an ignore region for every class

    def targets(self, kitti_type: str) -> tuple[list[int], list[int]]:
        """Return (classes this box is GT for, classes this box is an ignore region for)."""
        all_ids = list(KITTI_CLASSES)
        t = kitti_type
        if t == "Car":
            return [1], []
        if t == "Van":
            return ([1], []) if self.van == "merge" else ([], [1])
        if t == "Truck":
            return [2], []
        if t == "Pedestrian":
            return [3], []
        if t == "Person_sitting":
            return [], [3]
        if t == "Cyclist":
            return [4], [3] if self.cyclist_ignore_for_ped else []
        if t == "Tram":
            return [5], []
        if t == "Misc":
            return [], [1, 2]
        if t == "DontCare":
            return [], all_ids if self.dontcare_ignore else []
        raise ValueError(f"unknown KITTI type {t}")


COCO_TO_KITTI = {"car": 1, "truck": 2, "person": 3, "train": 5}

# Raw KITTI names, the zero-shot baseline vocabulary for open-vocab detectors.
KITTI_NAME_PROMPTS = {"car": 1, "van": 1, "truck": 2, "pedestrian": 3, "cyclist": 4, "tram": 5}

# Human baseline: ngdet's hand-written PROMPT_SYNONYMS (taxonomy.py, driving3/driving5 presets) folded onto
# the KITTI classes; ngdet has no tram class, so the raw KITTI name is added for it.
NGDET_PROMPTS = {"car": 1, "van": 1, "truck": 2, "person": 3, "pedestrian": 3, "bicycle": 4, "cyclist": 4, "tram": 5}


@dataclass
class PredRules:
    """How raw detector labels become KITTI target classes."""

    cyclist: str = "merge"  # "merge" person+bicycle pairs, "naive" bicycle->cyclist, "none"
    prompt_map: dict[str, int] | None = None  # open-vocab: prompt string -> target id
    synonym_nms_iou: float = 0.6  # open-vocab: NMS across synonyms that map to one class
    merge_min_overlap: float = 0.2  # intersection / person area needed to pair a rider with a bike
    extra: dict = field(default_factory=dict)


def _merge_cyclists(boxes: np.ndarray, scores: np.ndarray, labels: list[str], min_overlap: float):
    """Pair each bicycle with the best-overlapping person above it; return union boxes and scores."""
    bi = [i for i, l in enumerate(labels) if l == "bicycle"]
    pi = [i for i, l in enumerate(labels) if l == "person"]
    if not bi or not pi:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    used: set[int] = set()
    out_b, out_s = [], []
    for b in sorted(bi, key=lambda i: -scores[i]):
        bx = boxes[b]
        bcy = (bx[1] + bx[3]) / 2
        best, best_ov = -1, min_overlap
        for p in pi:
            if p in used:
                continue
            px = boxes[p]
            pcx, pcy = (px[0] + px[2]) / 2, (px[1] + px[3]) / 2
            if not (bx[0] <= pcx <= bx[2]) or pcy >= bcy:
                continue
            iw = min(bx[2], px[2]) - max(bx[0], px[0])
            ih = min(bx[3], px[3]) - max(bx[1], px[1])
            if iw <= 0 or ih <= 0:
                continue
            ov = iw * ih / max((px[2] - px[0]) * (px[3] - px[1]), 1e-6)
            if ov > best_ov:
                best, best_ov = p, ov
        if best >= 0:
            used.add(best)
            px = boxes[best]
            out_b.append([min(bx[0], px[0]), min(bx[1], px[1]), max(bx[2], px[2]), max(bx[3], px[3])])
            out_s.append(float(np.sqrt(scores[b] * scores[best])))
    return np.asarray(out_b, np.float32).reshape(-1, 4), np.asarray(out_s, np.float32)


def map_predictions(pred: dict, rules: PredRules, open_vocab: bool, with_source: bool = False):
    """Raw {boxes, scores, labels} -> (boxes xyxy, scores, KITTI class ids[, source prompt per box])."""
    boxes = np.asarray(pred["boxes"], np.float32).reshape(-1, 4)
    scores = np.asarray(pred["scores"], np.float32)
    labels = list(pred["labels"])
    out_b, out_s, out_c, out_src = [], [], [], []

    if open_vocab:
        pm = rules.prompt_map or KITTI_NAME_PROMPTS
        for cid in KITTI_CLASSES:
            idx = [i for i, l in enumerate(labels) if pm.get(l) == cid]
            if not idx:
                continue
            b, s, src = boxes[idx], scores[idx], [labels[i] for i in idx]
            if len(set(src)) > 1:
                keep = nms(torch.from_numpy(b), torch.from_numpy(s), rules.synonym_nms_iou).numpy()
                b, s, src = b[keep], s[keep], [src[i] for i in keep]
            out_b.append(b), out_s.append(s), out_c.append(np.full(len(s), cid)), out_src.extend(src)
    else:
        for name, cid in COCO_TO_KITTI.items():
            idx = [i for i, l in enumerate(labels) if l == name]
            if idx:
                out_b.append(boxes[idx]), out_s.append(scores[idx]), out_c.append(np.full(len(idx), cid))
        if rules.cyclist == "naive":
            idx = [i for i, l in enumerate(labels) if l == "bicycle"]
            if idx:
                out_b.append(boxes[idx]), out_s.append(scores[idx]), out_c.append(np.full(len(idx), 4))
        elif rules.cyclist == "merge":
            b, s = _merge_cyclists(boxes, scores, labels, rules.merge_min_overlap)
            if len(s):
                out_b.append(b), out_s.append(s), out_c.append(np.full(len(s), 4))

    if not out_b:
        empty = np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
        return (*empty, []) if with_source else empty
    res = np.concatenate(out_b), np.concatenate(out_s), np.concatenate(out_c).astype(np.int64)
    return (*res, out_src) if with_source else res
