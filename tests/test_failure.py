import gzip
import json

import numpy as np

from hw1 import score as score_mod
from hw1.data import load_gt
from hw1.evaluate import evaluate
from hw1.failure import ERRORS, FP, IGN, TP, Analysis, FPTYPES, analysis_dict
from hw1.taxonomy import KITTI_CLASSES

CATS = [{"id": k, "name": v} for k, v in KITTI_CLASSES.items()]


def ann(i, img, cat, box, crowd=0, occ=0, trunc=0.0):
    x1, y1, x2, y2 = box
    return {"id": i, "image_id": img, "category_id": cat, "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": crowd,
            "height_px": y2 - y1, "occluded": occ, "truncated": trunc, "area": (x2 - x1) * (y2 - y1)}


def mk(anns, n_img=1):
    return {"images": [{"id": i, "file_name": f"{i}.png", "width": 1000, "height": 400} for i in range(1, n_img + 1)],
            "annotations": anns, "categories": CATS}


def dets(rows):
    """rows: (image, box, score, cls) -> dict image -> arrays"""
    out = {}
    for im, box, s, c in rows:
        b, ss, cc = out.setdefault(im, ([], [], []))
        b.append(box), ss.append(s), cc.append(c)
    return {im: (np.array(b, float), np.array(s, float), np.array(c, int)) for im, (b, s, c) in out.items()}


def test_each_fp_type_and_ignore_region():
    # GT: car A (0..100), pedestrian B (300..340 x 0..100), car C (600..700), ignore region for car (800..900)
    gt = mk([ann(1, 1, 1, [0, 0, 100, 100]), ann(2, 1, 3, [300, 0, 340, 100]), ann(3, 1, 1, [600, 0, 700, 100]),
             ann(4, 1, 1, [800, 0, 900, 100], crowd=1)])
    rows = [
        (1, [0, 0, 100, 100], 0.99, 1),      # TP on A
        (1, [0, 0, 100, 100], 0.90, 1),      # Dupe: A already matched
        (1, [600, 0, 700, 100], 0.80, 3),    # Cls: pedestrian over car C (IoU 1.0 with another class)
        (1, [630, 0, 730, 100], 0.70, 1),    # IoU 70/130 = 0.54: TP on C
        (1, [650, 0, 750, 100], 0.60, 1),    # IoU(C) = 50/150 = 0.33 -> Loc
        (1, [300, 0, 380, 50], 0.50, 1),     # IoU 0.25 with pedestrian B only: Both
        (1, [450, 200, 500, 300], 0.40, 1),  # Bkg
        (1, [810, 10, 890, 90], 0.95, 1),    # inside ignore region: not an error
    ]
    an = Analysis(gt, dets(rows))
    by_score = {round(s, 2): (st, FPTYPES[t]) for s, st, t in zip(an.d_score, an.d_status, an.d_type)}
    assert by_score[0.99] == (TP, "")
    assert by_score[0.90] == (FP, "Dupe")
    assert by_score[0.80] == (FP, "Cls")
    assert by_score[0.60] == (FP, "Loc")
    assert by_score[0.50] == (FP, "Both")
    assert by_score[0.40] == (FP, "Bkg")
    assert by_score[0.95] == (IGN, "")
    assert by_score[0.70] == (TP, "")  # IoU 0.54 with car C
    c = an.counts()
    assert c["n_dets"] == len(rows) == c["TP"] + c["ignored"] + sum(c[e] for e in ERRORS)
    assert c["ignored"] == 1 and c["Missed"] == 1  # pedestrian B is missed


def test_loc_det_matches_when_iou_high_enough():
    gt = mk([ann(1, 1, 1, [0, 0, 100, 100])])
    an = Analysis(gt, dets([(1, [10, 0, 110, 100], 0.9, 1)]))  # IoU 90/110 = 0.82 -> TP
    assert an.d_status.tolist() == [TP] and an.counts()["Missed"] == 0


def test_oracles_raise_ap_and_bkg_in_particular():
    gt = mk([ann(i, 1, 1, [i * 100, 0, i * 100 + 50, 50]) for i in range(1, 6)])
    rows = [(1, [i * 100, 0, i * 100 + 50, 50], 0.5 + 0.01 * i, 1) for i in (1, 2, 3)]  # 3 of 5 found
    rows += [(1, [i * 100 + 5, 300, i * 100 + 55, 350], 0.95 - 0.01 * i, 1) for i in range(1, 6)]  # 5 high-score Bkg
    rows += [(1, [200, 0, 250, 50], 0.4, 1)]  # Dupe
    an = Analysis(gt, dets(rows))
    base, _ = an.baseline_ap50()
    res = analysis_dict(an)
    assert 0 < base < 0.6
    assert res["dAP50"]["Bkg"] > 0.1 and res["dAP50"]["Dupe"] >= 0 and res["dAP50"]["Missed"] > 0
    assert all(v >= -1e-9 for v in res["dAP50"].values())
    bkg_ap, _ = an.oracle("Bkg")
    assert abs(bkg_ap - 0.6) < 0.01  # 3/5 recall at precision 1.0 (101-point: recall points <= 0.6)
    assert an.counts()["Bkg"] == 5 and an.counts()["Dupe"] == 1 and an.counts(0.5)["Bkg"] == 5


def test_loc_and_cls_oracles_fix_to_tp():
    gt = mk([ann(1, 1, 1, [0, 0, 100, 100]), ann(2, 1, 3, [500, 0, 540, 100])])
    rows = [(1, [60, 0, 160, 100], 0.9, 1),    # Loc for the car (IoU 40/160 = 0.25)
            (1, [500, 0, 540, 100], 0.8, 1)]   # Cls: car label on the pedestrian
    an = Analysis(gt, dets(rows))
    assert [FPTYPES[t] for t in an.d_type] == ["Loc", "Cls"]
    assert analysis_dict(an)["AP50"] == 0.0
    assert abs(an.oracle("Loc")[0] - 0.5) < 1e-9 and abs(an.oracle("Cls")[0] - 0.5) < 1e-9
    assert analysis_dict(an)["dAP50"]["Loc"] > 0 and analysis_dict(an)["dAP50"]["Cls"] > 0


def test_recall_bins_sum_and_values():
    anns = [ann(1, 1, 1, [0, 0, 50, 20], occ=0, trunc=0.0),      # height 20
            ann(2, 1, 1, [100, 0, 150, 30], occ=1, trunc=0.1),   # 30
            ann(3, 1, 1, [200, 0, 260, 60], occ=2, trunc=0.5),   # 60
            ann(4, 1, 1, [300, 0, 400, 100], occ=3, trunc=0.0),  # 100
            ann(5, 1, 3, [500, 0, 520, 90], occ=0, trunc=0.0)]
    an = Analysis(mk(anns), dets([(1, [200, 0, 260, 60], 0.9, 1), (1, [300, 0, 400, 100], 0.8, 1)]))
    r = an.recall_bins()
    assert r["car"]["n"] == 4 and r["car"]["hits"] == 2 and r["all"]["n"] == 5
    for k in ("car", "pedestrian", "all"):
        for dim in ("height", "occlusion", "truncation"):
            assert sum(b["n"] for b in r[k][dim].values()) == r[k]["n"], (k, dim)
            assert sum(b["hits"] for b in r[k][dim].values()) == r[k]["hits"]
    assert [r["car"]["height"][b]["recall"] for b in ("<25", "25-40", "40-80", ">=80")] == [0, 0, 1, 1]
    assert r["car"]["truncation"][">0.3"]["hits"] == 1 and r["car"]["occlusion"]["3"]["recall"] == 1


def test_baseline_ap50_matches_hw1_evaluate():
    gt = load_gt("kitti_dev", cache=False)
    rng = np.random.default_rng(1)
    rows, cocodets = {}, []
    for a in gt["annotations"]:
        if a["iscrowd"] or rng.random() < 0.2:
            continue
        x, y, w, h = a["bbox"]
        j = rng.normal(0, 0.1, 4) * [w, h, w, h]
        box = [x + j[0], y + j[1], x + w + j[2], y + h + j[3]]
        cls = a["category_id"] if rng.random() > 0.1 else int(rng.integers(1, 6))
        s = float(rng.random())
        rows.setdefault(a["image_id"], []).append((box, s, cls))
        cocodets.append({"image_id": a["image_id"], "category_id": cls, "score": s,
                         "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]]})
    d = {i: (np.array([r[0] for r in v]), np.array([r[1] for r in v]), np.array([r[2] for r in v])) for i, v in rows.items()}
    for im in gt["images"]:
        d.setdefault(im["id"], (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
    an = Analysis(gt, d)
    ref = evaluate(gt, cocodets, iou_thrs=[0.5])["mAP"]
    assert abs(an.baseline_ap50()[0] - ref) < 1e-6, (an.baseline_ap50()[0], ref)


def test_score_subset_flag(tmp_path, monkeypatch):
    gt = load_gt("kitti_dev", cache=False)
    sub_ids = {im["id"] for im in load_gt("kitti_dev50", cache=False)["images"]}
    lines = []
    for im in gt["images"]:
        bs = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
              for a in gt["annotations"] if a["image_id"] == im["id"] and not a["iscrowd"] and a["category_id"] == 1]
        lines.append(json.dumps({"image_id": im["id"], "boxes": bs, "scores": [0.9] * len(bs), "labels": ["car"] * len(bs)}))
    pred = tmp_path / "fake__kitti_dev.jsonl.gz"
    with gzip.open(pred, "wt") as f:
        f.write("\n".join(lines) + "\n")
    (tmp_path / "fake__kitti_dev.meta.json").write_text(json.dumps(
        {"name": "fake__kitti_dev", "model": "fake", "split": "kitti_dev", "detector_config": {"open_vocab": False}}))
    full = score_mod.score(pred, extended=False)
    sub = score_mod.score(pred, extended=False, subset="kitti_dev50")
    assert full["n_images"] == 500 and sub["n_images"] == len(sub_ids) == 50
    assert score_mod.variant_tag(full) == "van-merge__cyc-merge"
    assert score_mod.variant_tag(sub) == "van-merge__cyc-merge__sub-kitti_dev50"
    assert abs(sub["coco"]["per_class"]["car"]["AP"] - 1.0) < 1e-6
    monkeypatch.setattr(score_mod, "METRIC_DIR", tmp_path / "m")
    score_mod.main([str(pred), "--fast", "--subset", "kitti_dev50"])
    assert (tmp_path / "m" / "fake__kitti_dev__van-merge__cyc-merge__sub-kitti_dev50.json").exists()
