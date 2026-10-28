"""COCO-protocol evaluation (pycocotools) with per-class AP, custom IoU thresholds and bootstrap CIs."""
from __future__ import annotations

import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

AREAS = ["all", "small", "medium", "large"]


def to_coco_results(image_id: int, boxes: np.ndarray, scores: np.ndarray, cat_ids: np.ndarray) -> list[dict]:
    out = []
    for (x1, y1, x2, y2), s, c in zip(np.asarray(boxes).tolist(), np.asarray(scores).tolist(), np.asarray(cat_ids).tolist()):
        out.append({"image_id": int(image_id), "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    return out


def filter_min_height(dets: list[dict], min_h: float) -> list[dict]:
    """KITTI devkit: detections shorter than the difficulty's minimum height are not counted."""
    return [d for d in dets if d["bbox"][3] >= min_h]


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def run_cocoeval(gt: dict, dets: list[dict], cat_ids: list[int] | None = None,
                 iou_thrs: list[float] | None = None, max_dets: int = 100) -> COCOeval:
    coco_gt = COCO()
    coco_gt.dataset = gt
    _quiet(coco_gt.createIndex)
    img_ids = [im["id"] for im in gt["images"]]
    dets = [d for d in dets if d["image_id"] in set(img_ids)]
    if dets:
        coco_dt = _quiet(coco_gt.loadRes, dets)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": gt["images"], "categories": gt["categories"], "annotations": []}
        _quiet(coco_dt.createIndex)
    E = COCOeval(coco_gt, coco_dt, "bbox")
    E.params.imgIds = img_ids
    if cat_ids is not None:
        E.params.catIds = list(cat_ids)
    if iou_thrs is not None:
        E.params.iouThrs = np.asarray(iou_thrs, dtype=float)
    E.params.maxDets = [1, 10, max_dets]
    _quiet(E.evaluate)
    _quiet(E.accumulate)
    return E


def _mean_valid(x: np.ndarray) -> float:
    v = x[x > -1]
    return float(v.mean()) if v.size else float("nan")


def stats_from_eval(E: COCOeval) -> dict:
    """Headline + per-class metrics, read from E.eval so any IoU threshold set works."""
    P, R = E.eval["precision"], E.eval["recall"]  # P: [T,R,K,A,M], R: [T,K,A,M]
    thr = list(np.round(E.params.iouThrs, 2))
    m = len(E.params.maxDets) - 1

    def ap(t=None, a=0, k=None):
        p = P[:, :, :, a, m] if t is None else P[[t], :, :, a, m]
        if k is not None:
            p = p[:, :, [k]]
        return _mean_valid(p)

    out = {"iou_thrs": [float(t) for t in thr], "mAP": ap()}
    for name, val in (("AP50", 0.5), ("AP75", 0.75)):
        if val in thr:
            out[name] = ap(t=thr.index(val))
    for a in (1, 2, 3):
        out[f"AP_{AREAS[a]}"] = ap(a=a)
    for i, md in enumerate(E.params.maxDets):
        out[f"AR@{md}"] = _mean_valid(R[:, :, 0, i])
    for a in (1, 2, 3):
        out[f"AR_{AREAS[a]}"] = _mean_valid(R[:, :, a, m])

    cats = {c["id"]: c["name"] for c in E.cocoGt.dataset["categories"]}
    per = {}
    for k, cid in enumerate(E.params.catIds):
        row = {"AP": ap(k=k), "AR": _mean_valid(R[:, [k], 0, m])}
        if 0.5 in thr:
            row["AP50"] = ap(t=thr.index(0.5), k=k)
        for a in (1, 2, 3):
            row[f"AP_{AREAS[a]}"] = ap(a=a, k=k)
        row["n_gt"] = int(sum(1 for g in E.cocoGt.anns.values() if g["category_id"] == cid and not g.get("iscrowd", 0)))
        per[cats.get(cid, str(cid))] = row
    out["per_class"] = per
    return out


def evaluate(gt: dict, dets: list[dict], **kw) -> dict:
    return stats_from_eval(run_cocoeval(gt, dets, **kw))


def _ap_for_images(E: COCOeval, img_idx: np.ndarray, k_idx: list[int]) -> float:
    """COCO 101-point AP over a (possibly repeated) set of image indices; mirrors COCOeval.accumulate."""
    p = E.params
    I, A = len(p.imgIds), len(p.areaRng)
    T, rec_thrs, max_det = len(p.iouThrs), p.recThrs, p.maxDets[-1]
    aps = []
    for k in k_idx:
        es = [E.evalImgs[k * A * I + i] for i in img_idx]
        es = [e for e in es if e is not None]
        if not es:
            continue
        gt_ig = np.concatenate([e["gtIgnore"] for e in es])
        npig = np.count_nonzero(gt_ig == 0)
        if npig == 0:
            continue
        scores = np.concatenate([e["dtScores"][:max_det] for e in es])
        order = np.argsort(-scores, kind="mergesort")
        dtm = np.concatenate([e["dtMatches"][:, :max_det] for e in es], axis=1)[:, order]
        dtig = np.concatenate([e["dtIgnore"][:, :max_det] for e in es], axis=1)[:, order]
        tps = np.logical_and(dtm, np.logical_not(dtig)).cumsum(axis=1).astype(float)
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtig)).cumsum(axis=1).astype(float)
        prec_k = np.zeros((T, len(rec_thrs)))
        for t in range(T):
            tp, fp = tps[t], fps[t]
            rc = tp / npig
            pr = tp / (fp + tp + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1] if pr.size else pr
            q = np.zeros(len(rec_thrs))
            inds = np.searchsorted(rc, rec_thrs, side="left")
            valid = inds < len(pr)
            q[valid] = pr[inds[valid]]
            prec_k[t] = q
        aps.append(prec_k.mean())
    return float(np.mean(aps)) if aps else float("nan")


def bootstrap_map(E: COCOeval, n: int = 500, seed: int = 0, per_class: bool = True) -> dict:
    """95% CI for mAP (and per-class AP) by resampling images with replacement."""
    rng = np.random.default_rng(seed)
    I = len(E.params.imgIds)
    K = len(E.params.catIds)
    samples = rng.integers(0, I, size=(n, I))
    full = np.arange(I)
    cats = {c["id"]: c["name"] for c in E.cocoGt.dataset["categories"]}
    targets = {"mAP": list(range(K))}
    if per_class:
        targets |= {cats[c]: [k] for k, c in enumerate(E.params.catIds)}
    out = {}
    for name, ks in targets.items():
        vals = np.array([_ap_for_images(E, s, ks) for s in samples])
        vals = vals[~np.isnan(vals)]
        point = _ap_for_images(E, full, ks)
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals.size else (np.nan, np.nan))
        out[name] = {"point": point, "lo": float(lo), "hi": float(hi), "n": int(vals.size)}
    return out


def paired_bootstrap(E_a: COCOeval, E_b: COCOeval, n: int = 1000, seed: int = 0) -> dict:
    """95% CI of mAP(a) - mAP(b) over the same resampled images; both evals must share images and categories."""
    assert list(E_a.params.imgIds) == list(E_b.params.imgIds) and list(E_a.params.catIds) == list(E_b.params.catIds)
    rng = np.random.default_rng(seed)
    I, ks = len(E_a.params.imgIds), list(range(len(E_a.params.catIds)))
    full = np.arange(I)
    diffs = []
    for s in rng.integers(0, I, size=(n, I)):
        diffs.append(_ap_for_images(E_a, s, ks) - _ap_for_images(E_b, s, ks))
    diffs = np.array(diffs)
    point = _ap_for_images(E_a, full, ks) - _ap_for_images(E_b, full, ks)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta": float(point), "lo": float(lo), "hi": float(hi), "p_le_0": float((diffs <= 0).mean()), "n": n}
