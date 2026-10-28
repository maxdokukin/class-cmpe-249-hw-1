"""Score saved raw predictions under one evaluation variant.

    python -m hw1.score runs/preds/yolo11m__kitti_test.jsonl.gz --cyclist merge --van merge --bootstrap 500

Writes runs/metrics/<pred name>__<variant>.json with COCO metrics, per-class AP, KITTI difficulty views and
KITTI-style IoU AP (KITTI splits only), and bootstrap CIs.

    python -m hw1.score runs/preds/yolo11m__coco_val.jsonl.gz --subset coco_val1k --fast

--subset <split> restricts predictions and GT to that split's image ids (a fair comparison on the 1000 COCO images the
open-vocab models ran on); the output name gets '__sub-<split>' appended to the variant tag.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .data import DIFFICULTY, ROOT, difficulty_view, load_gt
from .evaluate import bootstrap_map, filter_min_height, run_cocoeval, stats_from_eval, to_coco_results
from .taxonomy import GTRules, PredRules, map_predictions

METRIC_DIR = ROOT / "runs" / "metrics"


def load_preds(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def coco_dets(preds: list[dict], gt: dict, min_score: float) -> list[dict]:
    name_to_id = {c["name"]: c["id"] for c in gt["categories"]}
    out = []
    for p in preds:
        keep = [i for i, (l, s) in enumerate(zip(p["labels"], p["scores"])) if l in name_to_id and s >= min_score]
        if keep:
            b = np.asarray(p["boxes"])[keep]
            out += to_coco_results(p["image_id"], b, np.asarray(p["scores"])[keep], [name_to_id[p["labels"][i]] for i in keep])
    return out


def kitti_dets(preds: list[dict], rules: PredRules, open_vocab: bool, min_score: float) -> list[dict]:
    out = []
    for p in preds:
        b, s, c = map_predictions(p, rules, open_vocab)
        m = s >= min_score
        out += to_coco_results(p["image_id"], b[m], s[m], c[m])
    return out


def score(pred_path: Path, van: str = "merge", cyclist: str = "merge", prompt_map: dict | None = None,
          min_score: float = 0.0, n_boot: int = 0, extended: bool = True, subset: str | None = None) -> dict:
    meta = json.loads(Path(str(pred_path).replace(".jsonl.gz", ".meta.json")).read_text())
    split, open_vocab = meta["split"], meta["detector_config"]["open_vocab"]
    preds = load_preds(pred_path)
    if subset:
        keep_ids = {im["id"] for im in load_gt(subset)["images"]}
        preds = [p for p in preds if p["image_id"] in keep_ids]
    img_ids = {p["image_id"] for p in preds}
    res = {"pred": meta["name"], "split": split, "model": meta["model"], "n_images": len(img_ids),
           "variant": {"min_score": min_score}}
    if subset:
        res["variant"]["subset"] = subset

    if split.startswith("coco"):
        gt = load_gt(split)
        dets = coco_dets(preds, gt, min_score)
    else:
        gt = load_gt(split, GTRules(van=van))
        rules = PredRules(cyclist=cyclist, prompt_map=prompt_map)
        dets = kitti_dets(preds, rules, open_vocab, min_score)
        res["variant"] |= {"van": van, "cyclist": cyclist if not open_vocab else "prompt",
                           "prompt_map": prompt_map}
    gt = {**gt, "images": [im for im in gt["images"] if im["id"] in img_ids]}
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in img_ids]

    E = run_cocoeval(gt, dets)
    res["coco"] = stats_from_eval(E)
    if n_boot:  # COCO: 80 classes, so only the overall-mAP interval, with fewer resamples
        coco = split.startswith("coco")
        res["bootstrap"] = bootstrap_map(E, n=min(n_boot, 200) if coco else n_boot, per_class=not coco)

    if not split.startswith("coco"):
        res["mAP_core3"] = float(np.nanmean([res["coco"]["per_class"][n]["AP"] for n in ("car", "truck", "pedestrian")]))
        if extended:
            res["difficulty"] = {}
            for lv, (min_h, _, _) in DIFFICULTY.items():
                view = difficulty_view(gt, lv)
                d = filter_min_height(dets, min_h)
                res["difficulty"][lv] = {
                    "coco_mAP": stats_from_eval(run_cocoeval(view, d))["mAP"],
                    "kitti_AP": {
                        "car@0.7": stats_from_eval(run_cocoeval(view, d, cat_ids=[1], iou_thrs=[0.7]))["mAP"],
                        "pedestrian@0.5": stats_from_eval(run_cocoeval(view, d, cat_ids=[3], iou_thrs=[0.5]))["mAP"],
                        "cyclist@0.5": stats_from_eval(run_cocoeval(view, d, cat_ids=[4], iou_thrs=[0.5]))["mAP"],
                    },
                }
    return res


def variant_tag(r: dict) -> str:
    v = r["variant"]
    parts = []
    if "van" in v:
        parts += [f"van-{v['van']}", f"cyc-{v['cyclist']}"]
    if v["min_score"]:
        parts.append(f"thr{v['min_score']}")
    if v.get("subset"):
        parts = (parts or ["default"]) + [f"sub-{v['subset']}"]
    return "__".join(parts) or "default"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--van", default="merge", choices=["merge", "ignore"])
    ap.add_argument("--cyclist", default="merge", choices=["merge", "naive", "none"])
    ap.add_argument("--prompt-map", default=None, help="JSON file {prompt: kitti class id}")
    ap.add_argument("--min-score", type=float, default=0.0, help="drop detections below this (threshold ablation)")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--subset", default=None, help="restrict predictions and GT to this split's images (e.g. coco_val1k)")
    ap.add_argument("--fast", action="store_true", help="skip difficulty / KITTI-IoU views")
    args = ap.parse_args(argv)
    pm = {k: int(v) for k, v in json.loads(Path(args.prompt_map).read_text()).items()} if args.prompt_map else None
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    for p in args.preds:
        r = score(Path(p), args.van, args.cyclist, pm, args.min_score, args.bootstrap, not args.fast, args.subset)
        tag = variant_tag(r) + (f"__pm-{Path(args.prompt_map).stem}" if args.prompt_map else "")
        out = METRIC_DIR / f"{r['pred']}__{tag}.json"
        out.write_text(json.dumps(r, indent=2, default=float))
        c = r["coco"]
        extra = f"  core3 {r['mAP_core3']:.3f}" if "mAP_core3" in r else ""
        print(f"{r['pred']:45s} {tag:28s} mAP {c['mAP']:.3f}  AP50 {c.get('AP50', float('nan')):.3f}{extra}  -> {out.name}")


if __name__ == "__main__":
    main()
