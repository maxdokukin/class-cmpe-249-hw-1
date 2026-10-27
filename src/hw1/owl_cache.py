"""Cache OWLv2's text-independent image side once per image, then score any vocabulary in milliseconds."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .data import iter_images, load_gt
from .detectors.base import empty_dets, make_dets

CKPT = "google/owlv2-base-patch16-ensemble"
CACHE_DIR = Path(__file__).resolve().parents[2] / "runs" / "cache"


def _load(ckpt, device):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    return Owlv2Processor.from_pretrained(ckpt), Owlv2ForObjectDetection.from_pretrained(ckpt).to(device).eval()


def build_cache(split: str, out_path=None, device: str = "cpu", max_images: int | None = None,
                ckpt: str = CKPT, emb_dtype=torch.float16, shard_size: int | None = None) -> Path:
    """One .pt for the split, or with shard_size a directory of resumable shards (for splits too big for RAM)."""
    gt = load_gt(split)
    if max_images:
        gt = {**gt, "images": gt["images"][:max_images]}
    proc, model = _load(ckpt, device)
    if not shard_size:
        out_path = Path(out_path or CACHE_DIR / f"owlv2__{split}.pt")
        return _build(gt, split, out_path, proc, model, device, ckpt, emb_dtype)
    out_dir = Path(out_path or CACHE_DIR / f"owlv2__{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    images = gt["images"]
    for n, start in enumerate(range(0, len(images), shard_size)):
        shard = out_dir / f"shard_{n:03d}.pt"
        if shard.exists():
            continue
        part = {**gt, "images": images[start:start + shard_size]}
        print(f"shard {n}: images {start}-{start + len(part['images']) - 1}", flush=True)
        tmp = shard.with_suffix(".tmp")
        _build(part, split, tmp, proc, model, device, ckpt, emb_dtype)
        tmp.rename(shard)
    return out_dir


@torch.inference_mode()
def _build(gt, split, out_path: Path, proc, model, device, ckpt, emb_dtype) -> Path:
    ids, hw, emb, shift, scale, boxes, obj = [], [], [], [], [], [], []
    t0 = time.time()
    for i, (iid, img) in enumerate(iter_images(gt, split)):
        pix = proc.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device)
        fmap, _ = model.image_embedder(pixel_values=pix)
        feats = fmap.reshape(1, -1, fmap.shape[-1])
        head = model.class_head
        ce = head.dense0(feats)
        ce = ce / (torch.linalg.norm(ce, dim=-1, keepdim=True) + 1e-6)
        ids.append(iid)
        hw.append(img.shape[:2])
        emb.append(ce[0].to(emb_dtype).cpu())
        shift.append(head.logit_shift(feats)[0, :, 0].float().cpu())
        scale.append((head.elu(head.logit_scale(feats)) + 1)[0, :, 0].float().cpu())
        boxes.append(model.box_predictor(feats, fmap)[0].float().cpu())
        obj.append(model.objectness_predictor(feats)[0].float().cpu())
        print(f"{i + 1}/{len(gt['images'])}  {(time.time() - t0) / (i + 1):.2f} s/img", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ckpt": ckpt, "image_ids": ids, "hw": hw, "class_embeds": torch.stack(emb), "logit_shift": torch.stack(shift),
                "logit_scale": torch.stack(scale), "boxes_cxcywh": torch.stack(boxes), "objectness": torch.stack(obj)}, out_path)
    return out_path


class CachedOwl:
    """Scores vocabularies against a cache file, or a shard directory (loaded one shard at a time)."""

    def __init__(self, cache_path, device: str = "cpu"):
        path = Path(cache_path)
        self.shards = sorted(path.glob("shard_*.pt")) if path.is_dir() else [path]
        if not self.shards:
            raise FileNotFoundError(f"no cache shards in {path}")
        self.device = device
        self._use(self.shards[0])
        self.proc, self.model = _load(self.d["ckpt"], device)
        self._text: dict[str, torch.Tensor] = {}

    def _use(self, shard: Path):
        if getattr(self, "_current", None) == shard:
            return
        d = torch.load(shard, map_location="cpu")
        self.d, self._current = d, shard
        self.ids, self.hw = d["image_ids"], d["hw"]
        self.index = {iid: k for k, iid in enumerate(self.ids)}

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        new = [p for p in dict.fromkeys(prompts) if p not in self._text]
        if new:
            t = self.proc.tokenizer(new, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
            t = {k: v.to(self.device) for k, v in t.items()}
            q = self.model.owlv2.get_text_features(**t).pooler_output.float().cpu()
            q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)
            self._text.update(zip(new, q))
        return torch.stack([self._text[p] for p in prompts])

    @torch.inference_mode()
    def _scores(self, k: int, q: torch.Tensor) -> torch.Tensor:
        d = self.d
        logits = (d["class_embeds"][k].float() @ q.T + d["logit_shift"][k][:, None]) * d["logit_scale"][k][:, None]
        return logits.max(-1)

    def predict(self, image_id, prompts: list[str], score_thresh: float = 0.001, max_det: int = 100) -> dict:
        k = self.index[image_id]
        h, w = self.hw[k]
        best = self._scores(k, self.encode_text(prompts))
        scores, labels = torch.sigmoid(best.values), best.indices
        keep = scores > score_thresh
        if not keep.any():
            return empty_dets()
        cx, cy, bw, bh = self.d["boxes_cxcywh"][k][keep].unbind(-1)
        s = max(h, w)
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * s
        return make_dets(boxes.numpy(), scores[keep].numpy(), [prompts[i] for i in labels[keep].tolist()], max_det, (h, w))

    def predict_all(self, prompts: list[str], score_thresh: float = 0.001, max_det: int = 100) -> list[dict]:
        out = []
        for shard in self.shards:
            self._use(shard)
            for iid in self.ids:
                p = self.predict(iid, prompts, score_thresh, max_det)
                out.append({"image_id": iid, "boxes": np.round(p["boxes"], 2).tolist(),
                            "scores": np.round(p["scores"], 5).tolist(), "labels": list(p["labels"])})
        return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="kitti_dev")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--out")
    ap.add_argument("--shard-size", type=int, help="write a directory of resumable shards of this many images")
    a = ap.parse_args(argv)
    p = build_cache(a.split, a.out, a.device, a.max_images, shard_size=a.shard_size)
    size = sum(f.stat().st_size for f in ([p] if p.is_file() else p.glob("shard_*.pt")))
    print(f"wrote {p}  {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
