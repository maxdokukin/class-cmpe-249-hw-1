"""Run one detector over one split and save raw predictions plus a run record.

    python -m hw1.run --model yolo11m --split kitti_test --device mps

Writes runs/preds/<model>__<split>[__tag].jsonl.gz (raw boxes, scores and labels per image, before any
taxonomy mapping) and runs/preds/<...>.meta.json (config, environment, latency, memory, exact command).
Scoring is a separate step (hw1.score), so mapping variants never require re-running a model.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import gzip
import hashlib
import json
import platform
import shlex
import socket
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import ROOT, load_gt, iter_images
from .detectors import build_detector
from .detectors.base import COCO80
from .taxonomy import KITTI_NAME_PROMPTS
from .timing import MemSampler

PRED_DIR = ROOT / "runs" / "preds"


def src_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "src" / "hw1").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def default_vocab(split: str) -> list[str]:
    return list(COCO80) if split.startswith("coco") else list(KITTI_NAME_PROMPTS)


def latency_summary(rows: list[dict], warmup: int, burst: int = 200) -> dict:
    out = {"warmup_images": warmup}
    timed = rows[warmup:]
    for label, sub in (("burst", timed[:burst]), ("sustained", timed)):
        if not sub:
            continue
        tot = np.array([r["total_ms"] for r in sub])
        s = {"n": len(sub), "median_ms": float(np.median(tot)), "p90_ms": float(np.percentile(tot, 90)),
             "mean_ms": float(tot.mean()), "fps_median": float(1000 / np.median(tot))}
        for k in ("preprocess_ms", "forward_ms", "postprocess_ms"):
            v = [r[k] for r in sub if r.get(k) is not None]
            if v:
                s[f"{k[:-3]}_median_ms"] = float(np.median(v))
        out[label] = s
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--vocab", default=None, help="open-vocab: JSON list of prompts or a path to one")
    ap.add_argument("--tag", default="", help="suffix for the output name, e.g. e2e_off or vocab_agent")
    ap.add_argument("--set", nargs="*", default=[], help="detector overrides key=value (JSON values)")
    args = ap.parse_args(argv)

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v

    gt = load_gt(args.split)
    if args.max_images:
        gt = {**gt, "images": gt["images"][: args.max_images]}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        det = build_detector(args.model, device=args.device, **overrides)
        vocab = None
        if det.open_vocab:
            if args.vocab:
                vocab = json.loads(Path(args.vocab).read_text() if Path(args.vocab).exists() else args.vocab)
            else:
                vocab = default_vocab(args.split)
            det.set_vocabulary(vocab)

        name = f"{args.model}__{args.split}" + (f"__{args.tag}" if args.tag else "")
        if args.max_images:
            name += f"__n{args.max_images}"
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        pred_path = PRED_DIR / f"{name}.jsonl.gz"
        mem = MemSampler(args.device)
        timings, n_boxes = [], []
        t_start = time.time()
        with gzip.open(pred_path, "wt") as f:
            for i, (iid, img) in enumerate(iter_images(gt, args.split)):
                pred = det.predict(img)
                mem.sample()
                timings.append(det.last_timing)
                n_boxes.append(len(pred["scores"]))
                f.write(json.dumps({
                    "image_id": iid,
                    "boxes": np.round(np.asarray(pred["boxes"]), 2).tolist(),
                    "scores": np.round(np.asarray(pred["scores"]), 5).tolist(),
                    "labels": list(pred["labels"]),
                    "ms": round(det.last_timing.get("total_ms", float("nan")), 2),
                }) + "\n")
                if (i + 1) % 100 == 0:
                    el = time.time() - t_start
                    print(f"[{name}] {i + 1}/{len(gt['images'])}  {el / (i + 1) * 1000:.0f} ms/img wall", flush=True)
        wall = time.time() - t_start

    fallback = sorted({str(w.message)[:300] for w in caught if "fall back" in str(w.message).lower() or "mps" in str(w.message).lower()})
    meta = {
        "name": name,
        "model": args.model,
        "split": args.split,
        "n_images": len(gt["images"]),
        "vocabulary": vocab,
        "detector_config": det.config(),
        "latency": latency_summary(timings, args.warmup),
        "memory": mem.summary(),
        "boxes_per_image_mean": float(np.mean(n_boxes)) if n_boxes else 0.0,
        "wall_s": round(wall, 1),
        "mps_fallback_warnings": fallback,
        "env": {"host": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
                "processor": platform.processor(), "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")},
        "src_hash": src_hash(),
        "command": "python -m hw1.run " + " ".join(shlex.quote(a) for a in (argv or sys.argv[1:])),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pred_file": str(pred_path.relative_to(ROOT)),
    }
    (PRED_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    with open(ROOT / "runs" / "COMMANDS.md", "a") as f:
        f.write(f"- `{meta['command']}`  ({meta['finished_utc']}, {meta['env']['host']}, {args.device}, {meta['wall_s']} s)\n")
    lat = meta["latency"].get("sustained", {})
    print(f"done {name}: {meta['n_images']} imgs, {wall:.0f} s, median {lat.get('median_ms', float('nan')):.1f} ms/img")


if __name__ == "__main__":
    main()
