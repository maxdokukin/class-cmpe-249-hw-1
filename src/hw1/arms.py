"""E4 test arms: run each prompt map chosen on kitti_dev once on kitti_test, then score it.

    python -m hw1.arms --detector owlv2 --device mps      # OWLv2 via the sharded kitti_test feature cache
    python -m hw1.arms --detector gdino_t --device mps    # Grounding DINO: one full kitti_test run per arm

Arms come from runs/agent_logs/<detector>__kitti_dev__seed<k>/map_*.json: raw_names and ngdet_synonyms
(fixed baselines), oneshot (the agent's first proposal, before any feedback) and best (best dev mAP).
"""
from __future__ import annotations

import argparse
import gzip
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .data import ROOT, load_gt
from .run import PRED_DIR, src_hash
from .score import METRIC_DIR, main as score_main

LOGS = ROOT / "runs" / "agent_logs"


def collect_arms(detector: str, split: str = "kitti_dev") -> dict[str, Path]:
    arms: dict[str, Path] = {}
    for d in sorted(LOGS.glob(f"{detector}__{split}__seed*")):
        seed = d.name.split("__seed", 1)[1].replace("__", "_")  # e.g. "0_haiku-4-5"
        for base in ("raw_names", "ngdet_synonyms"):
            if (d / f"map_{base}.json").exists():
                arms.setdefault(base, d / f"map_{base}.json")
        for kind in ("oneshot", "best"):
            if (d / f"map_{kind}.json").exists():
                arms[f"{kind}_s{seed}"] = d / f"map_{kind}.json"
    return arms


def owl_arm_preds(name: str, map_path: Path, cache, n_images: int) -> Path:
    pred_path = PRED_DIR / f"{name}.jsonl.gz"
    pm = json.loads(map_path.read_text())
    rows = cache.predict_all(sorted(pm))
    with gzip.open(pred_path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"name": name, "model": "owlv2", "split": "kitti_test", "n_images": n_images, "vocabulary": sorted(pm),
            "prompt_map_file": str(map_path.relative_to(ROOT)),
            "detector_config": {"name": "owlv2", "open_vocab": True, "source": "sharded feature cache (hw1.owl_cache)",
                                "score_thresh": 0.001, "max_det": 100},
            "src_hash": src_hash(), "command": "python -m hw1.arms " + " ".join(map(shlex.quote, sys.argv[1:])),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pred_file": str(pred_path.relative_to(ROOT))}
    (PRED_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    return pred_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=["owlv2", "gdino_t"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args(argv)

    arms = collect_arms(args.detector)
    if not arms:
        print(f"no agent logs for {args.detector}; run hw1.agent first")
        return
    print(f"{args.detector} arms: {', '.join(arms)}")
    cache = None
    for arm, map_path in arms.items():
        name = f"{args.detector}__kitti_test__arm-{arm}"
        pred = PRED_DIR / f"{name}.jsonl.gz"
        if not (PRED_DIR / f"{name}.meta.json").exists():
            if args.detector == "owlv2":
                if cache is None:
                    from .owl_cache import CACHE_DIR, CachedOwl

                    cache = CachedOwl(CACHE_DIR / "owlv2__kitti_test", device=args.device)
                owl_arm_preds(name, map_path, cache, len(load_gt("kitti_test")["images"]))
            else:
                vocab = map_path.with_suffix(".vocab.json")
                subprocess.run([sys.executable, "-m", "hw1.run", "--model", args.detector, "--split", "kitti_test",
                                "--device", args.device, "--vocab", str(vocab), "--tag", f"arm-{arm}"], check=True)
        if not list(METRIC_DIR.glob(f"{name}__*pm-{map_path.stem}.json")):
            score_main([str(pred), "--prompt-map", str(map_path), "--bootstrap", str(args.bootstrap)])
    paired_vs_raw(args.detector, arms)


def paired_vs_raw(detector: str, arms: dict[str, Path], n: int = 1000):
    """Paired bootstrap of each arm's test mAP minus the raw-names arm, on identical resampled images."""
    from .data import load_gt
    from .evaluate import paired_bootstrap, run_cocoeval
    from .score import kitti_dets, load_preds
    from .taxonomy import GTRules, PredRules

    gt = load_gt("kitti_test", GTRules())

    def ev(arm):
        pm = {k: int(v) for k, v in json.loads(arms[arm].read_text()).items()}
        preds = load_preds(PRED_DIR / f"{detector}__kitti_test__arm-{arm}.jsonl.gz")
        return run_cocoeval(gt, kitti_dets(preds, PredRules(prompt_map=pm), True, 0.0))

    if "raw_names" not in arms:
        return
    base = ev("raw_names")
    out = {arm: paired_bootstrap(ev(arm), base, n=n) for arm in arms if arm != "raw_names"}
    path = METRIC_DIR / f"{detector}__kitti_test__paired_vs_raw_names.json"
    path.write_text(json.dumps(out, indent=1))
    for arm, r in out.items():
        print(f"{arm:28s} delta mAP {r['delta']:+.4f}  95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]  P(delta<=0) {r['p_le_0']:.3f}")


if __name__ == "__main__":
    main()
