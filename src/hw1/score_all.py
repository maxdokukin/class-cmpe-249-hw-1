"""Score every saved prediction file under the headline variant and the ablations; skips finished ones.

    python -m hw1.score_all [--bootstrap 500]

Headline: van-merge, cyclist-merge (closed-set) / KITTI-name prompts (open-vocab), no score cutoff.
Ablations on kitti_test closed-set runs: cyclist naive and none, van as ignore, a 0.3 score cutoff (ngdet's default).
Ablation on COCO runs: the 0.3 score cutoff. Agent arms are scored by hw1.arms; dev-split runs are skipped.
"""
from __future__ import annotations

import argparse
import json

from .run import PRED_DIR
from .score import METRIC_DIR, main as score_main

KITTI_ABLATIONS = [["--cyclist", "naive"], ["--cyclist", "none"], ["--van", "ignore"], ["--min-score", "0.3"]]


def expected_tag(args: list[str], open_vocab: bool, kitti: bool) -> str:
    v = dict(zip(args[::2], args[1::2]))
    parts = []
    if kitti:
        parts += [f"van-{v.get('--van', 'merge')}", f"cyc-{'prompt' if open_vocab else v.get('--cyclist', 'merge')}"]
    if "--min-score" in v:
        parts.append(f"thr{float(v['--min-score'])}")
    return "__".join(parts) or "default"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=500)
    a = ap.parse_args(argv)
    for meta_path in sorted(PRED_DIR.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        name, split = meta["name"], meta["split"]
        pred = PRED_DIR / f"{name}.jsonl.gz"
        if not pred.exists() or "__arm-" in name or "parity_" in name or split.startswith("kitti_dev"):
            continue
        ov, kitti = meta["detector_config"]["open_vocab"], split.startswith("kitti")
        boot = ["--bootstrap", str(a.bootstrap)] if split in ("kitti_test", "coco_val") else []
        jobs = [([], boot)]  # (variant args, extra args); ablations skip bootstrap and difficulty views
        if kitti and not ov:
            jobs += [(v, ["--fast"]) for v in KITTI_ABLATIONS]
        elif not kitti:
            jobs += [(["--min-score", "0.3"], ["--fast"])]
        for variant, extra in jobs:
            if (METRIC_DIR / f"{name}__{expected_tag(variant, ov, kitti)}.json").exists():
                continue
            score_main([str(pred), *variant, *extra])


if __name__ == "__main__":
    main()
