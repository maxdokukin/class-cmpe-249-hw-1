"""Smoke test: every detector on CPU, 2 COCO val images. Usage: smoke_detectors.py [--device cpu] [names...]"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw1.detectors import NAMES, build_detector  # noqa: E402
from hw1.timing import MemSampler  # noqa: E402

IMAGES = ["000000000139.jpg", "000000000632.jpg"]
VOCAB = ["person", "car", "chair", "tv", "potted plant"]
CORE = ["yolo11m", "yolo26m", "frcnn_v2", "rtdetr_r50", "owlv2", "gdino_t"]


def draw(img: np.ndarray, dets: dict, path: Path, min_score: float = 0.25) -> None:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for b, s, l in zip(dets["boxes"], dets["scores"], dets["labels"]):
        if s < min_score:
            continue
        d.rectangle([float(x) for x in b], outline=(255, 0, 0), width=2)
        d.text((float(b[0]) + 2, float(b[1]) + 2), f"{l} {s:.2f}", fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def check(img: np.ndarray, dets: dict, coco_names: set[str], open_vocab: bool) -> list[str]:
    h, w = img.shape[:2]
    b, s = dets["boxes"], dets["scores"]
    errs = []
    if b.dtype != np.float32 or s.dtype != np.float32 or b.shape != (len(s), 4) or len(dets["labels"]) != len(s):
        errs.append("bad dtype/shape")
    if len(s) and not (0 <= s.min() and s.max() <= 1):
        errs.append("score out of [0,1]")
    if len(s) and (b[:, 0].min() < -1 or b[:, 1].min() < -1 or b[:, 2].max() > w + 1 or b[:, 3].max() > h + 1):
        errs.append(f"box outside image: x[{b[:, 0].min():.0f},{b[:, 2].max():.0f}] y[{b[:, 1].min():.0f},{b[:, 3].max():.0f}] vs {w}x{h}")
    if len(s) and (b[:, 2] < b[:, 0]).any() | (b[:, 3] < b[:, 1]).any():
        errs.append("inverted box")
    allowed = set(VOCAB) if open_vocab else coco_names
    bad = set(dets["labels"]) - allowed
    if bad:
        errs.append(f"unknown labels {sorted(bad)[:5]}")
    if not len(s):
        errs.append("no detections")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=CORE)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ann = ROOT / "data/coco/annotations/instances_val2017.json"
    coco_names = {c["name"] for c in json.load(open(ann))["categories"]} if ann.exists() else set()
    imgs = [np.array(Image.open(ROOT / "data/coco/val2017" / f).convert("RGB")) for f in IMAGES]
    table = []
    for name in a.names:
        print(f"\n===== {name} =====", flush=True)
        try:
            det = build_detector(name, device=a.device)
            if det.open_vocab:
                det.set_vocabulary(VOCAB)
            mem = MemSampler(a.device)
            det.predict(imgs[0])  # warmup
            ms, errs, counts = [], [], []
            for f, img in zip(IMAGES, imgs):
                out = det.predict(img)
                mem.sample()
                ms.append(det.last_timing["total_ms"])
                counts.append(len(out["scores"]))
                errs += check(img, out, coco_names, det.open_vocab)
                print(f"{f}: N={len(out['scores'])} timing={ {k: (round(v, 1) if v is not None else None) for k, v in det.last_timing.items()} }")
                for b, s, l in list(zip(out["boxes"], out["scores"], out["labels"]))[:5]:
                    print(f"   {l:14s} {s:.3f} {np.round(b, 1).tolist()}")
                print(f"   scores>0.25: {(out['scores'] > 0.25).sum()}  >0.05: {(out['scores'] > 0.05).sum()}  min={out['scores'].min() if len(out['scores']) else None}")
                if f == IMAGES[0]:
                    draw(img, out, ROOT / "runs/smoke" / f"{name}.png")
            print("config:", json.dumps(det.config(), indent=1, default=str))
            print("mem:", mem.summary())
            if min(counts) < 20:
                errs.append(f"only {min(counts)} boxes at low threshold")
            table.append((name, "PASS" if not errs else "FAIL", float(np.mean(ms)), counts, "; ".join(dict.fromkeys(errs))))
            del det
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            table.append((name, "FAIL", float("nan"), [], f"{type(e).__name__}: {e}"[:200]))
    print("\n%-12s %-5s %10s  %-10s %s" % ("model", "", "ms/img", "N boxes", "notes"))
    for n, st, ms, c, note in table:
        print("%-12s %-5s %10.0f  %-10s %s" % (n, st, ms, c, note))


if __name__ == "__main__":
    main()
