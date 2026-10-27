from __future__ import annotations

import numpy as np
import torch
import torchvision
from torchvision.models import detection as D

from ..timing import Timer
from .base import Detector, empty_dets, make_dets

_BUILDERS = {
    "frcnn_v2": (D.fasterrcnn_resnet50_fpn_v2, D.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, "box_score_thresh"),
    "fcos": (D.fcos_resnet50_fpn, D.FCOS_ResNet50_FPN_Weights.COCO_V1, "score_thresh"),
}


class TorchvisionDetector(Detector):
    def __init__(self, name: str, device: str = "cpu", score_thresh: float = 0.001, max_det: int = 100,
                 nms_iou: float | None = None):
        super().__init__()
        builder, weights, thr_arg = _BUILDERS[name]
        kw = {thr_arg: score_thresh, "box_detections_per_img": max_det}
        if nms_iou is not None:
            kw["box_nms_thresh" if name == "frcnn_v2" else "nms_thresh"] = nms_iou
        self.name, self.device, self.score_thresh, self.max_det = name, device, score_thresh, max_det
        self.model = builder(weights=weights, **kw).to(device).eval()
        self.categories = weights.meta["categories"]
        self.weights_name = str(weights)
        self.split = name == "frcnn_v2"

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> dict:
        h, w = image.shape[:2]
        tm = Timer(self.device)
        x = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float().div_(255)
        m = self.model
        if self.split:
            images, _ = m.transform([x])
            tm.mark("preprocess")
            feats = m.backbone(images.tensors)
            props, _ = m.rpn(images, feats)
            dets, _ = m.roi_heads(feats, props, images.image_sizes)
            tm.mark("forward")
            out = m.transform.postprocess(dets, images.image_sizes, [(h, w)])[0]
            tm.mark("postprocess")
        else:
            tm.mark("preprocess")
            out = m([x])[0]
            tm.mark("forward")
        self.last_timing = tm.result()
        if len(out["boxes"]) == 0:
            return empty_dets()
        labels = [self.categories[int(i)] for i in out["labels"].cpu()]
        return make_dets(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), labels, self.max_det, (h, w))

    def _config(self) -> dict:
        t = self.model.transform
        return {
            "checkpoint": self.weights_name,
            "input": f"GeneralizedRCNNTransform resize short side {t.min_size[0]}, long side cap {t.max_size}, keep aspect",
            "nms_iou": getattr(getattr(self.model, "roi_heads", self.model), "nms_thresh", None),
            "timing_note": "transform+backbone+rpn+roi_heads split; roi_heads includes NMS" if self.split
            else "preprocess = tensor copy only; forward includes resize and NMS",
        }
