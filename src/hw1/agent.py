"""E4: an LLM agent that tunes the vocabulary of an open-vocab detector on KITTI dev.

    python -m hw1.agent --detector owlv2 --seed 0            # needs ANTHROPIC_API_KEY
    python -m hw1.agent --detector owlv2 --dry-run           # scores the fixed baselines only, no API calls

The agent proposes a prompt map (prompt string -> KITTI class). The only tool, evaluate_prompt_map, runs the
detector on kitti_dev and returns per-class AP, per-prompt hit/false-positive counts and failure crops.
The code, not the agent, picks the best map by dev mAP. Test evaluation is a separate step
(hw1.run + hw1.score on kitti_test) that the agent never sees.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import ROOT, image_root, load_gt
from .evaluate import run_cocoeval, stats_from_eval, to_coco_results
from .taxonomy import KITTI_CLASSES, KITTI_IDS, KITTI_NAME_PROMPTS, NGDET_PROMPTS, PredRules, map_predictions

AGENT_SPLITS = {"kitti_dev", "kitti_dev50"}
LOG_ROOT = ROOT / "runs" / "agent_logs"
PRICES = {  # USD per MTok: input, output, cache read, cache write (5 min)
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5),
    "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25),
}
MAX_PER_CLASS, MAX_TOTAL = 5, 30

SYSTEM = """You are tuning the text vocabulary of a zero-shot, open-vocabulary object detector ({detector}) so that it \
detects the object classes of the KITTI autonomous-driving benchmark well. You cannot change model weights, \
thresholds or image size: the only thing you control is a prompt map, a list of short text prompts, each assigned to \
one KITTI class. The detector scores every image region against every prompt; a box found by any prompt of a class \
counts as a detection of that class, and duplicate boxes from synonyms of one class are merged by NMS.

KITTI classes (definitions from the KITTI object devkit and annotation conventions):
- car: passenger cars, including vans and minivans (KITTI "Car" and "Van" are both scored as car here).
- truck: trucks, box trucks, delivery trucks, lorries.
- pedestrian: people walking or standing. Seated people and riders are not pedestrians; they are ignored, \
neither rewarded nor penalized.
- cyclist: a person riding a bicycle. The ground-truth box covers BOTH the rider and the bicycle as one object.
- tram: street trams / light rail vehicles running on city streets (common in the Karlsruhe scenes).
Regions labelled "DontCare" (far-away or unlabeled objects) are ignored, so detections there cost nothing.

Images: front camera of a car driving through Karlsruhe, Germany (urban, residential, highway), 1242x375 pixels.
Objects are often small, distant, occluded or truncated.

Metric: COCO mAP@[0.5:0.95] averaged over the 5 classes, on a fixed development split of {n_images} images. \
Every class counts equally, and a class with no prompt scores 0.

Rules for the prompt map: lowercase English phrases, 1-6 words each, at most {max_per_class} prompts per class \
and at most {max_total} prompts in total, at least one prompt per class. More prompts are not free: each extra \
prompt can add false positives.

You have a budget of {budget} calls to evaluate_prompt_map. Each call returns per-class AP, which prompts produced \
correct and false detections, and image crops of high-scoring false positives (red box) and missed ground-truth \
objects (green box). Use the feedback to form hypotheses about what the detector's text encoder responds to, and \
change one or two things at a time so you can tell what helped. Always respond by calling evaluate_prompt_map with \
your next proposal and a one-sentence rationale. The best map you find is selected automatically."""

TOOL = {
    "name": "evaluate_prompt_map",
    "description": "Run the detector on the KITTI development split with this prompt map and return per-class AP, "
                   "per-prompt statistics and failure crops.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["prompt_map", "rationale"],
        "properties": {
            "prompt_map": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["prompt", "kitti_class"],
                    "properties": {
                        "prompt": {"type": "string"},
                        "kitti_class": {"type": "string", "enum": list(KITTI_CLASSES.values())},
                    },
                },
            },
            "rationale": {"type": "string"},
        },
    },
}


class DevEvaluator:
    """Scores a prompt map on an agent split. Holds the only GT the agent can reach."""

    def __init__(self, detector: str, split: str = "kitti_dev", device: str = "cpu", n_crops: int = 6):
        if split not in AGENT_SPLITS:
            raise ValueError(f"agent may only evaluate on {sorted(AGENT_SPLITS)}, not {split}")
        self.split, self.n_crops, self.detector = split, n_crops, detector
        self.gt = load_gt(split)
        self.backend = self._backend(detector, device)
        self.cache: dict[tuple, list[dict]] = {}

    def _backend(self, detector: str, device: str):
        if detector == "owlv2":
            from .owl_cache import CachedOwl

            cache = ROOT / "runs" / "cache" / f"owlv2__{self.split}.pt"
            if not cache.exists():
                raise FileNotFoundError(f"build it first: python -m hw1.owl_cache --split {self.split} --device {device}")
            owl = CachedOwl(cache)
            return lambda prompts: owl.predict_all(prompts)

        from .data import iter_images
        from .detectors import build_detector

        det = build_detector(detector, device=device)

        def run(prompts):
            det.set_vocabulary(prompts)
            out = []
            for iid, img in iter_images(self.gt, self.split):
                p = det.predict(img)
                out.append({"image_id": iid, "boxes": np.asarray(p["boxes"]).tolist(),
                            "scores": np.asarray(p["scores"]).tolist(), "labels": list(p["labels"])})
            return out

        return run

    def predictions(self, prompts: list[str]) -> list[dict]:
        key = tuple(sorted(prompts))
        if key not in self.cache:
            self.cache[key] = self.backend(list(key))
        return self.cache[key]

    def evaluate(self, prompt_map: dict[str, int], with_feedback: bool = True) -> dict:
        t0 = time.time()
        preds = self.predictions(list(prompt_map))
        rules = PredRules(prompt_map=prompt_map)
        dets = []
        for p in preds:
            b, s, c, src = map_predictions(p, rules, open_vocab=True, with_source=True)
            for d, prompt in zip(to_coco_results(p["image_id"], b, s, c), src):
                d["prompt"] = prompt
                dets.append(d)
        E = run_cocoeval(self.gt, dets)
        st = stats_from_eval(E)
        res = {"mAP": st["mAP"], "AP50": st.get("AP50"),
               "per_class": {k: {"AP": v["AP"], "AP50": v.get("AP50"), "n_gt": v["n_gt"]} for k, v in st["per_class"].items()},
               "eval_s": round(time.time() - t0, 1)}
        if with_feedback:
            res |= self._feedback(E)
        return res

    def _feedback(self, E) -> dict:
        """Per-prompt TP/FP at score >= 0.2 and IoU 0.5, plus the worst false positives and misses."""
        dt, gt = E.cocoDt.anns, E.cocoGt.anns
        per_prompt: dict[str, list[int]] = {}
        fps, fns = [], []
        for e in E.evalImgs:
            if e is None or e["aRng"] != E.params.areaRng[0]:
                continue
            for j, did in enumerate(e["dtIds"]):
                if e["dtIgnore"][0, j]:
                    continue
                d = dt[did]
                tp = e["dtMatches"][0, j] > 0
                if d["score"] >= 0.2:
                    per_prompt.setdefault(d["prompt"], [0, 0])[0 if tp else 1] += 1
                if not tp:
                    fps.append(d)
            for j, gid in enumerate(e["gtIds"]):
                if not e["gtIgnore"][j] and e["gtMatches"][0, j] == 0:
                    fns.append(gt[gid])
        fps.sort(key=lambda d: -d["score"])
        fns.sort(key=lambda g: -g["bbox"][3])
        k = self.n_crops // 2
        return {
            "per_prompt_at_score_0.2": {p: {"correct": v[0], "false_pos": v[1]} for p, v in sorted(per_prompt.items())},
            "crops": [("false positive", d) for d in fps[:k]] + [("missed", g) for g in fns[:k]],
        }

    def crop_png_b64(self, kind: str, ann: dict, max_side: int = 224) -> str:
        im = next(i for i in self.gt["images"] if i["id"] == ann["image_id"])
        with Image.open(image_root(self.split) / im["file_name"]) as f:
            img = f.convert("RGB")
        x, y, w, h = ann["bbox"]
        pad = max(w, h) * 0.6 + 10
        box = (max(0, x - pad), max(0, y - pad), min(img.width, x + w + pad), min(img.height, y + h + pad))
        crop = img.crop(box)
        ImageDraw.Draw(crop).rectangle([x - box[0], y - box[1], x + w - box[0], y + h - box[1]],
                                       outline=(255, 0, 0) if kind == "false positive" else (0, 255, 0), width=2)
        crop.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        return base64.standard_b64encode(buf.getvalue()).decode()


def validate_map(items: list[dict]) -> tuple[dict[str, int] | None, str]:
    pm: dict[str, int] = {}
    for it in items:
        p = " ".join(str(it["prompt"]).lower().split())
        if not p or len(p.split()) > 6:
            return None, f"invalid prompt {it['prompt']!r}: use 1-6 words"
        if p in pm and pm[p] != KITTI_IDS[it["kitti_class"]]:
            return None, f"prompt {p!r} assigned to two classes"
        pm[p] = KITTI_IDS[it["kitti_class"]]
    counts = {c: sum(1 for v in pm.values() if v == cid) for cid, c in KITTI_CLASSES.items()}
    if missing := [c for c, n in counts.items() if n == 0]:
        return None, f"every class needs at least one prompt; missing: {missing}"
    if over := [c for c, n in counts.items() if n > MAX_PER_CLASS]:
        return None, f"at most {MAX_PER_CLASS} prompts per class; too many for: {over}"
    if len(pm) > MAX_TOTAL:
        return None, f"at most {MAX_TOTAL} prompts in total; got {len(pm)}"
    return pm, ""


def tool_result_content(ev: DevEvaluator, res: dict, best: float) -> list[dict]:
    summary = {k: v for k, v in res.items() if k != "crops"}
    summary["best_mAP_so_far"] = best
    content = [{"type": "text", "text": json.dumps(summary, indent=1, default=float)}]
    for kind, ann in res.get("crops", []):
        label = KITTI_CLASSES[ann["category_id"]]
        cap = (f"{kind}: predicted {label} via prompt {ann['prompt']!r}, score {ann['score']:.2f}" if kind == "false positive"
               else f"{kind}: ground-truth {label}, {ann['bbox'][3]:.0f} px tall, not detected at IoU 0.5")
        content.append({"type": "text", "text": cap})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": ev.crop_png_b64(kind, ann)}})
    return content


def request_options(model: str, effort: str) -> dict:
    """Per-model request settings: Haiku 4.5 rejects `effort` and has no server-side refusal fallback."""
    if model.startswith("claude-haiku"):
        return {}
    return {"output_config": {"effort": effort}, "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}


def cost_usd(model: str, usage) -> float:
    pin, pout, pread, pwrite = PRICES[model]
    return ((usage.input_tokens or 0) * pin + (usage.output_tokens or 0) * pout
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * pread
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * pwrite) / 1e6


def save_map(path: Path, pm: dict[str, int]):
    path.write_text(json.dumps(pm, indent=1))
    path.with_suffix(".vocab.json").write_text(json.dumps(sorted(pm)))


def run_baselines(ev: DevEvaluator, out: Path) -> dict:
    res = {}
    for name, pm in (("raw_names", KITTI_NAME_PROMPTS), ("ngdet_synonyms", NGDET_PROMPTS)):
        r = ev.evaluate(pm, with_feedback=False)
        res[name] = r
        save_map(out / f"map_{name}.json", pm)
        print(f"[baseline] {name:15s} dev mAP {r['mAP']:.4f}  " + "  ".join(f"{k} {v['AP']:.3f}" for k, v in r["per_class"].items()))
    (out / "baselines.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def run_agent(ev: DevEvaluator, out: Path, model: str, budget: int, effort: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    system = SYSTEM.format(detector=ev.detector, n_images=len(ev.gt["images"]), max_per_class=MAX_PER_CLASS,
                           max_total=MAX_TOTAL, budget=budget)
    messages: list = [{"role": "user", "content": "Propose your first prompt map."}]
    history, total_cost, best = [], 0.0, {"mAP": -1.0}
    log = open(out / "transcript.jsonl", "a")
    n_eval, turns = 0, 0
    while n_eval < budget and turns < budget * 3:
        turns += 1
        resp = client.beta.messages.create(
            model=model, max_tokens=16000, system=system, tools=[TOOL], messages=messages,
            cache_control={"type": "ephemeral"}, **request_options(model, effort),
        )
        c = cost_usd(model, resp.usage)
        total_cost += c
        log.write(json.dumps({"turn": turns, "model": resp.model, "stop_reason": resp.stop_reason,
                              "usage": resp.usage.to_dict(), "cost_usd": c, "request_id": resp._request_id,
                              "content": [b.to_dict() for b in resp.content if b.type in ("text", "tool_use")]}) + "\n")
        log.flush()
        if resp.stop_reason == "refusal":
            print(f"refusal: {resp.stop_details}")
            break
        messages.append({"role": "assistant", "content": resp.content})
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            messages.append({"role": "user", "content": "Please call evaluate_prompt_map with your next proposal."})
            continue
        results = []
        for call in calls:
            if n_eval >= budget:
                results.append({"type": "tool_result", "tool_use_id": call.id, "is_error": True,
                                "content": "evaluation budget exhausted"})
                continue
            pm, err = validate_map(call.input["prompt_map"])
            if pm is None:
                results.append({"type": "tool_result", "tool_use_id": call.id, "is_error": True, "content": err})
                continue
            res = ev.evaluate(pm)
            n_eval += 1
            rec = {"iter": n_eval - 1, "prompt_map": pm, "rationale": call.input.get("rationale", ""),
                   "mAP": res["mAP"], "per_class": res["per_class"], "per_prompt": res["per_prompt_at_score_0.2"],
                   "cost_usd_cum": total_cost}
            history.append(rec)
            (out / f"iter_{n_eval - 1:02d}.json").write_text(json.dumps(rec, indent=1, default=float))
            if n_eval == 1:
                save_map(out / "map_oneshot.json", pm)
            if res["mAP"] > best["mAP"]:
                best = rec
                save_map(out / "map_best.json", pm)
            print(f"[iter {n_eval - 1:2d}] dev mAP {res['mAP']:.4f} (best {best['mAP']:.4f})  {len(pm)} prompts  "
                  f"${total_cost:.3f}  | {rec['rationale'][:90]}", flush=True)
            results.append({"type": "tool_result", "tool_use_id": call.id,
                            "content": tool_result_content(ev, res, best["mAP"])})
        messages.append({"role": "user", "content": results})
    summary = {"model": model, "effort": effort, "budget": budget, "evaluations": n_eval, "turns": turns,
               "cost_usd": total_cost, "best_iter": best.get("iter"), "best_dev_mAP": best["mAP"],
               "oneshot_dev_mAP": history[0]["mAP"] if history else None,
               "trajectory": [h["mAP"] for h in history]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="owlv2", choices=["owlv2", "gdino_t"])
    ap.add_argument("--split", default="kitti_dev", choices=sorted(AGENT_SPLITS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0, help="run index; sampling uses the model's default temperature")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true", help="score the baselines only, no API calls")
    args = ap.parse_args(argv)

    out = LOG_ROOT / f"{args.detector}__{args.split}__seed{args.seed}__{args.model.removeprefix('claude-')}"
    out.mkdir(parents=True, exist_ok=True)
    ev = DevEvaluator(args.detector, args.split, args.device)
    run_baselines(ev, out)
    if not args.dry_run:
        s = run_agent(ev, out, args.model, args.budget, args.effort)
        print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
