import json

import numpy as np
import pytest

from hw1.agent import DevEvaluator, tool_result_content, validate_map
from hw1.data import load_gt
from hw1.taxonomy import KITTI_CLASSES


def fake_evaluator(split="kitti_dev50"):
    ev = DevEvaluator.__new__(DevEvaluator)
    ev.split, ev.n_crops, ev.detector, ev.cache = split, 6, "fake", {}
    ev.gt = load_gt(split)
    rng = np.random.default_rng(0)

    def backend(prompts):
        out = {im["id"]: {"image_id": im["id"], "boxes": [], "scores": [], "labels": []} for im in ev.gt["images"]}
        for a in ev.gt["annotations"]:
            if a["iscrowd"] or rng.random() < 0.3:
                continue
            name = KITTI_CLASSES[a["category_id"]]
            if name not in prompts:
                continue
            x, y, w, h = a["bbox"]
            o = out[a["image_id"]]
            o["boxes"].append([x, y, x + w, y + h]); o["scores"].append(float(rng.random())); o["labels"].append(name)
            o["boxes"].append([x + 200, y, x + w + 200, y + h]); o["scores"].append(0.9); o["labels"].append(name)
        return list(out.values())

    ev.backend = backend
    return ev


def test_validate_map():
    ok = [{"prompt": c, "kitti_class": c} for c in KITTI_CLASSES.values()]
    pm, err = validate_map(ok)
    assert pm and not err
    assert validate_map(ok[:-1])[0] is None
    assert validate_map(ok + [{"prompt": f"car {i}", "kitti_class": "car"} for i in range(5)])[0] is None
    assert validate_map(ok + [{"prompt": "car", "kitti_class": "truck"}])[0] is None


def test_evaluate_feedback_and_crops():
    ev = fake_evaluator()
    pm = {c: i for i, c in KITTI_CLASSES.items()}
    res = ev.evaluate(pm)
    assert 0 < res["mAP"] < 1
    pp = res["per_prompt_at_score_0.2"]
    assert pp["car"]["false_pos"] > 0 and pp["car"]["correct"] > 0
    kinds = [k for k, _ in res["crops"]]
    assert "false positive" in kinds and "missed" in kinds
    content = tool_result_content(ev, res, best=res["mAP"])
    assert content[0]["type"] == "text" and json.loads(content[0]["text"])["mAP"] == pytest.approx(res["mAP"])
    imgs = [c for c in content if c["type"] == "image"]
    assert len(imgs) == len(res["crops"]) and len(imgs[0]["source"]["data"]) > 100


def test_agent_cannot_use_test_split():
    with pytest.raises(ValueError):
        DevEvaluator("owlv2", split="kitti_test")
