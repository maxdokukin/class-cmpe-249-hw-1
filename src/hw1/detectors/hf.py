from __future__ import annotations

import numpy as np
import torch

from ..timing import Timer
from .base import COCO80, Detector, empty_dets, make_dets


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


class RTDetr(Detector):
    def __init__(self, name: str = "rtdetr_r50", device: str = "cpu", ckpt: str = "PekingU/rtdetr_r50vd",
                 score_thresh: float = 0.0, max_det: int = 100):
        super().__init__()
        from transformers import AutoImageProcessor, RTDetrForObjectDetection

        self.name, self.device, self.ckpt, self.score_thresh, self.max_det = name, device, ckpt, score_thresh, max_det
        self.proc = AutoImageProcessor.from_pretrained(ckpt)
        self.model = RTDetrForObjectDetection.from_pretrained(ckpt).to(device).eval()
        assert len(self.model.config.id2label) == 80

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> dict:
        h, w = image.shape[:2]
        tm = Timer(self.device)
        inputs = self.proc(images=image, return_tensors="pt").to(self.device)
        tm.mark("preprocess")
        out = self.model(**inputs)
        tm.mark("forward")
        r = self.proc.post_process_object_detection(out, threshold=self.score_thresh, target_sizes=[(h, w)])[0]
        scores = _to_np(r["scores"])
        tm.mark("postprocess")
        self.last_timing = tm.result()
        if len(scores) == 0:
            return empty_dets()
        labels = [COCO80[int(i)] for i in r["labels"].cpu()]
        return make_dets(_to_np(r["boxes"]), scores, labels, self.max_det, (h, w))

    def _config(self) -> dict:
        return {"checkpoint": self.ckpt, "input": "fixed 640x640 resize, aspect not kept",
                "nms_iou": "none", "queries": 300, "postprocess": "sigmoid, top-300 over queries x classes, then top-max_det"}


class OWLv2(Detector):
    open_vocab = True

    def __init__(self, name: str = "owlv2", device: str = "cpu", ckpt: str = "google/owlv2-base-patch16-ensemble",
                 score_thresh: float = 0.001, max_det: int = 100):
        super().__init__()
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.name, self.device, self.ckpt, self.score_thresh, self.max_det = name, device, ckpt, score_thresh, max_det
        self.proc = Owlv2Processor.from_pretrained(ckpt)
        self.model = Owlv2ForObjectDetection.from_pretrained(ckpt).to(device).eval()
        self.prompts: list[str] = []
        self._text = None

    def set_vocabulary(self, prompts: list[str]) -> None:
        self.prompts = list(prompts)
        t = self.proc.tokenizer(self.prompts, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
        self._text = {k: v.to(self.device) for k, v in t.items()}

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> dict:
        if self._text is None:
            raise RuntimeError("call set_vocabulary() first")
        h, w = image.shape[:2]
        tm = Timer(self.device)
        pix = self.proc.image_processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        tm.mark("preprocess")
        out = self.model(pixel_values=pix, **self._text)
        tm.mark("forward")
        r = self.proc.image_processor.post_process_object_detection(out, threshold=self.score_thresh, target_sizes=[(h, w)])[0]
        scores = _to_np(r["scores"])
        tm.mark("postprocess")
        self.last_timing = tm.result()
        if len(scores) == 0:
            return empty_dets()
        labels = [self.prompts[int(i)] for i in r["labels"].cpu()]
        return make_dets(_to_np(r["boxes"]), scores, labels, self.max_det, (h, w))

    def _config(self) -> dict:
        return {"checkpoint": self.ckpt, "input": "pad to square (gray), resize 960x960", "nms_iou": "none",
                "prompts": self.prompts, "postprocess": "one box per query (argmax prompt), sigmoid score",
                "timing_note": "text tokenized once in set_vocabulary; text tower still runs every forward"}


class GroundingDino(Detector):
    open_vocab = True
    MAX_TOKENS = 256

    def __init__(self, name: str = "gdino_t", device: str = "cpu", ckpt: str = "IDEA-Research/grounding-dino-tiny",
                 score_thresh: float = 0.001, max_det: int = 100):
        super().__init__()
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.name, self.device, self.ckpt, self.score_thresh, self.max_det = name, device, ckpt, score_thresh, max_det
        self.proc = AutoProcessor.from_pretrained(ckpt)
        self.tok = self.proc.tokenizer
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(ckpt).to(device).eval()
        self.prompts: list[str] = []
        self.chunks: list[dict] = []

    def _ntok(self, text: str) -> int:
        return len(self.tok(text)["input_ids"])

    def set_vocabulary(self, prompts: list[str]) -> None:
        self.prompts = [p.strip().lower() for p in prompts]
        groups: list[list[int]] = []
        for i, p in enumerate(self.prompts):
            if groups and self._ntok(" . ".join(self.prompts[j] for j in groups[-1] + [i]) + " .") <= self.MAX_TOKENS:
                groups[-1].append(i)
            else:
                groups.append([i])
        self.chunks = [self._encode(g) for g in groups]

    def _encode(self, idx: list[int]) -> dict:
        text, spans = "", []
        for j, i in enumerate(idx):
            if j:
                text += " . "
            spans.append((len(text), len(text) + len(self.prompts[i])))
            text += self.prompts[i]
        text += " ."
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True, return_token_type_ids=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        masks = []
        for s, e in spans:
            m = torch.tensor([(b > a and a >= s and b <= e) for a, b in offs])
            masks.append(m)
        return {"idx": idx, "enc": {k: v.to(self.device) for k, v in enc.items()}, "tok_masks": torch.stack(masks).to(self.device)}

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> dict:
        if not self.chunks:
            raise RuntimeError("call set_vocabulary() first")
        h, w = image.shape[:2]
        tm = Timer(self.device)
        px = self.proc.image_processor(images=image, return_tensors="pt").to(self.device)
        tm.mark("preprocess")
        outs = [self.model(**px, **c["enc"]) for c in self.chunks]
        tm.mark("forward")
        boxes, scores, labels = [], [], []
        for c, o in zip(self.chunks, outs):
            probs = torch.sigmoid(o.logits[0])  # (Q, 256)
            ntok = c["tok_masks"].shape[1]
            probs = probs[:, :ntok]
            per_prompt = torch.stack([probs[:, m].max(dim=-1).values for m in c["tok_masks"]], dim=1)  # (Q, P)
            s, k = per_prompt.max(dim=1)
            keep = s > self.score_thresh
            cx, cy, bw, bh = o.pred_boxes[0].unbind(-1)
            xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * torch.tensor(
                [w, h, w, h], dtype=torch.float32, device=s.device)
            boxes.append(_to_np(xyxy[keep]))
            scores.append(_to_np(s[keep]))
            labels += [self.prompts[c["idx"][int(i)]] for i in k[keep].cpu()]
        tm.mark("postprocess")
        self.last_timing = tm.result()
        if not labels:
            return empty_dets()
        return make_dets(np.concatenate(boxes), np.concatenate(scores), labels, self.max_det, (h, w))

    def _config(self) -> dict:
        return {"checkpoint": self.ckpt, "input": "processor default (short side 800, long side cap 1333, keep aspect)",
                "nms_iou": "none", "queries": 900, "prompts": self.prompts,
                "prompt_format": "lowercase phrases joined by ' . ', ending ' .'",
                "text_chunks": [c["idx"] for c in self.chunks], "max_text_tokens": self.MAX_TOKENS,
                "postprocess": "custom: per-prompt score = max token prob over the prompt's tokens; one box per query "
                "per chunk (argmax prompt); chunks concatenated, top-max_det. Not processor.post_process_grounded_object_detection "
                "(its phrase labels can be empty or partial at low thresholds)"}
