import numpy as np

from hw1.data import load_gt, difficulty_view
from hw1.evaluate import bootstrap_map, evaluate, run_cocoeval, stats_from_eval
from hw1.taxonomy import PredRules, map_predictions


def gt_as_dets(gt, jitter=0.0, drop=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dets = []
    for a in gt["annotations"]:
        if a["iscrowd"] or rng.random() < drop:
            continue
        x, y, w, h = a["bbox"]
        j = rng.normal(0, jitter, 4) * [w, h, w, h]
        dets.append({"image_id": a["image_id"], "category_id": a["category_id"],
                     "bbox": [x + j[0], y + j[1], max(w + j[2], 1), max(h + j[3], 1)], "score": float(rng.random())})
    return dets


def test_perfect():
    gt = load_gt("kitti_dev", cache=False)
    s = evaluate(gt, gt_as_dets(gt))
    assert abs(s["mAP"] - 1.0) < 1e-6, s["mAP"]


def test_bootstrap_point_matches_cocoeval():
    gt = load_gt("kitti_dev", cache=False)
    E = run_cocoeval(gt, gt_as_dets(gt, jitter=0.08, drop=0.2))
    s = stats_from_eval(E)
    b = bootstrap_map(E, n=20)
    assert abs(b["mAP"]["point"] - s["mAP"]) < 1e-9, (b["mAP"]["point"], s["mAP"])
    assert b["mAP"]["lo"] <= s["mAP"] <= b["mAP"]["hi"]
    for name, row in s["per_class"].items():
        assert abs(b[name]["point"] - row["AP"]) < 1e-9, name


def test_dets_in_dontcare_are_not_fps():
    gt = load_gt("kitti_dev", cache=False)
    base = gt_as_dets(gt)
    junk = [{"image_id": a["image_id"], "category_id": 1, "bbox": a["bbox"], "score": 0.99}
            for a in gt["annotations"] if a["iscrowd"] and a["kitti_type"] == "DontCare" and a["category_id"] == 1]
    assert junk
    assert abs(evaluate(gt, base + junk)["mAP"] - 1.0) < 1e-6
    no_dc = load_gt("kitti_dev", cache=False)
    for a in no_dc["annotations"]:
        if a["kitti_type"] == "DontCare":
            a["category_id"] = -1
    no_dc["annotations"] = [a for a in no_dc["annotations"] if a["category_id"] != -1]
    assert evaluate(no_dc, base + junk)["per_class"]["car"]["AP"] < 0.99


def test_kitti_iou_and_difficulty():
    gt = load_gt("kitti_dev", cache=False)
    dets = gt_as_dets(gt, jitter=0.05)
    s = evaluate(gt, dets, cat_ids=[1], iou_thrs=[0.7])
    assert s["iou_thrs"] == [0.7] and 0 < s["mAP"] <= 1
    mod = difficulty_view(gt, "moderate")
    assert evaluate(mod, dets)["mAP"] > 0


def test_cyclist_merge():
    pred = {"boxes": [[100, 50, 140, 150], [95, 100, 150, 170], [300, 100, 340, 200]],
            "scores": [0.9, 0.8, 0.7], "labels": ["person", "bicycle", "person"]}
    b, s, c = map_predictions(pred, PredRules(cyclist="merge"), open_vocab=False)
    cyc = b[c == 4]
    assert len(cyc) == 1 and cyc[0].tolist() == [95, 50, 150, 170]
    assert (c == 3).sum() == 2
    b, s, c = map_predictions(pred, PredRules(cyclist="naive"), open_vocab=False)
    assert b[c == 4][0].tolist() == [95, 100, 150, 170]


def test_open_vocab_synonym_nms():
    pred = {"boxes": [[0, 0, 10, 10], [0, 0, 10, 10.5], [50, 50, 60, 60]], "scores": [0.9, 0.5, 0.4],
            "labels": ["sedan", "suv", "person walking"]}
    pm = {"sedan": 1, "suv": 1, "person walking": 3}
    b, s, c = map_predictions(pred, PredRules(prompt_map=pm), open_vocab=True)
    assert sorted(c.tolist()) == [1, 3] and s[c == 1][0] == np.float32(0.9)
