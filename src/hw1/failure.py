"""Failure analysis of saved KITTI predictions: TIDE-style errors, recall by object size / occlusion / truncation, galleries.

    python -m hw1.failure [--split kitti_test] [--models yolo11m rtdetr_r50 ...] [--out runs/analysis]

Default input: every runs/preds/*__<split>*.jsonl.gz whose split field is exactly <split> and whose name has no
'__arm-' (so kitti_test200 runs and agent arms are skipped). Predictions are mapped onto KITTI classes exactly as the
headline score does (GT van-merge; closed-set cyclist-merge; open-vocab KITTI-name prompts). Output goes to
runs/analysis/: <pred name>.json per run, summary.md across runs, gallery/*.jpg.

Matching (a faithful numpy re-implementation of pycocotools at one IoU threshold)
    Per image and class, detections are sorted by score (top 100 kept) and matched greedily to that class's GT at
    IoU >= 0.5, each GT used once, best-IoU GT first. KITTI ignore regions (iscrowd=1 annotations of that class: DontCare,
    Misc, ignored Van, Person_sitting, riders for pedestrians) can absorb a detection (IoU = intersection / detection
    area, as in COCO); an absorbed detection is neither TP nor FP. The baseline AP50 is the COCO 101-point AP averaged
    over KITTI classes that have GT, and equals hw1.evaluate's AP50 (checked in tests).

Error types (each unmatched, non-ignored detection gets exactly one; first rule that fires)
    Cls   IoU >= 0.5 with a GT of another class
    Dupe  IoU >= 0.5 with a same-class GT (necessarily already matched by a higher-scored detection)
    Loc   0.1 <= IoU < 0.5 with a same-class GT (best same-class IoU)
    Both  0.1 <= IoU < 0.5 with a GT of another class (and nothing same-class at >= 0.1)
    Bkg   IoU < 0.1 with every non-ignore GT
    Missed: a GT that no detection matched (counted over all detections, or over detections with score >= 0.3; the
    matching is greedy by score, so thresholding a run leaves the matches of the kept detections unchanged).
    IoUs against GT use only real GT (iscrowd=0), never ignore regions.

ΔAP50 oracles (TIDE, Bolya et al. ECCV 2020; each oracle is applied alone to the baseline, ΔAP50 = AP50(fixed) - AP50)
    Loc    each Loc FP, in descending score, is snapped to its best same-class GT: it becomes a TP if that GT is still
           unmatched, otherwise it is removed (it would be a duplicate). Because a GT already matched in the
           baseline cannot be matched again, this follows TIDE's flag-based fix rather than re-running the matcher.
    Cls    each Cls FP is relabelled to the class of its best other-class GT (IoU >= 0.5): TP in that class if the GT is
           unmatched, otherwise removed.
    Dupe   removed.   Bkg  removed.   Both  removed (TIDE also only removes them).
    Missed missed GT are dropped from the recall denominator of their class (detections untouched).
    Per-class numbers are the same oracle's change in that class's AP50; the overall number is the mean over classes.

Recall@100 uses the same matching (IoU 0.5, all scores, <= 100 detections per image and class): a GT is a hit when a
detection of its class matched it. Bins: GT height_px (<25, 25-40, 40-80, >=80), KITTI occlusion 0/1/2/3 (3 = unknown),
truncation (0, (0, 0.3], > 0.3). Every bin carries n, hits and recall; bins of a dimension partition each class's GT.

Gallery: per run the 3 images with the most high-confidence errors (FP with score >= 0.5 + missed GT with
height >= 40 px), GT green (missed GT thick and tagged MISS), detections with score >= 0.3 red when FP and orange
otherwise. A shared set of up to 3 images holds large (>= 40 px) GT that every closed-set run (unsuffixed
<model>__<split> files only) misses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import KITTI_DIR, ROOT, load_gt
from .score import load_preds
from .taxonomy import KITTI_CLASSES, GTRules, PredRules, map_predictions

PRED_DIR = ROOT / "runs" / "preds"
OUT_DIR = ROOT / "runs" / "analysis"
IOU_POS, IOU_BG, MAX_DET = 0.5, 0.1, 100
TP, IGN, FP, REM = 0, 1, 2, 3  # detection status ("REM" = removed by an oracle)
FPTYPES = ["", "Cls", "Loc", "Both", "Dupe", "Bkg"]
ERRORS = ["Cls", "Loc", "Both", "Dupe", "Bkg"]
HEIGHT_BINS = [("<25", 0, 25), ("25-40", 25, 40), ("40-80", 40, 80), (">=80", 80, 1e9)]
OCC_BINS = [("0", 0), ("1", 1), ("2", 2), ("3", 3)]
TRUNC_BINS = [("0", lambda t: t <= 0), ("(0,0.3]", lambda t: 0 < t <= 0.3), (">0.3", lambda t: t > 0.3)]


def iou_matrix(a: np.ndarray, b: np.ndarray, crowd_of_a: bool = False) -> np.ndarray:
    """IoU of xyxy boxes a [N,4] vs b [M,4]; with crowd_of_a, intersection over a's area (COCO crowd rule)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    aa = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    if crowd_of_a:
        return inter / np.maximum(aa, 1e-9)
    ab = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(aa + ab - inter, 1e-9)


def ap101(scores: np.ndarray, tp: np.ndarray, npig: int) -> float:
    """COCO 101-point AP for one class at one IoU; input in global (image, score) order, ignored dets already dropped."""
    if npig <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    t = tp[order]
    tps, fps = np.cumsum(t).astype(float), np.cumsum(~t).astype(float)
    rc = tps / npig
    pr = tps / (fps + tps + np.spacing(1))
    if pr.size:
        pr = np.maximum.accumulate(pr[::-1])[::-1]
    q = np.zeros(101)
    inds = np.searchsorted(rc, np.linspace(0, 1, 101), side="left")
    ok = inds < len(pr)
    q[ok] = pr[inds[ok]]
    return float(q.mean())


class Analysis:
    """Baseline matching and error typing of mapped detections against COCO-format KITTI GT."""

    def __init__(self, gt: dict, dets: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """dets: image_id -> (boxes xyxy [N,4], scores [N], KITTI class ids [N]); images absent from dets are dropped."""
        self.gt = gt
        self.classes = [c["id"] for c in gt["categories"]]
        anns_by_img: dict[int, list[dict]] = {}
        for a in gt["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        self.images = [im for im in gt["images"] if im["id"] in dets]

        G = [a for im in self.images for a in anns_by_img.get(im["id"], []) if not a["iscrowd"]]
        self.g_ann = np.array([a["id"] for a in G], int)
        self.g_img = np.array([a["image_id"] for a in G], int)
        self.g_cls = np.array([a["category_id"] for a in G], int)
        self.g_box = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                               for a in G], float).reshape(-1, 4)
        self.g_height = np.array([a["height_px"] for a in G], float)
        self.g_occ = np.array([a["occluded"] for a in G], int)
        self.g_trunc = np.array([a["truncated"] for a in G], float)
        self.g_match = np.full(len(G), -1, int)  # detection index that matched, else -1
        gpos = {int(i): k for k, i in enumerate(self.g_ann)}

        d_img, d_cls, d_score, d_box, d_status, d_type, d_same, d_other = [], [], [], [], [], [], [], []
        n = 0
        for im in self.images:
            iid = im["id"]
            b, s, c = (np.asarray(x) for x in dets[iid])
            b, s, c = b.reshape(-1, 4).astype(float), s.astype(float), c.astype(int)
            sel = []
            for k in self.classes:  # per class: score order, top MAX_DET (COCOeval's per image-category cut)
                idx = np.flatnonzero(c == k)
                sel.extend(idx[np.argsort(-s[idx], kind="mergesort")][:MAX_DET].tolist())
            sel = np.array(sel, int)
            b, s, c = b[sel], s[sel], c[sel]
            real = [a for a in anns_by_img.get(iid, []) if not a["iscrowd"]]
            gi = np.array([gpos[a["id"]] for a in real], int)
            gb = self.g_box[gi] if len(gi) else np.zeros((0, 4))
            gc = self.g_cls[gi] if len(gi) else np.zeros(0, int)
            iou_all = iou_matrix(b, gb)
            status = np.full(len(s), FP, int)
            ftype = np.zeros(len(s), int)
            same = np.full(len(s), -1, int)
            other = np.full(len(s), -1, int)
            for k in self.classes:
                dk = np.flatnonzero(c == k)
                if not len(dk):
                    continue
                gk = np.flatnonzero(gc == k)
                crowd = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                                  for a in anns_by_img.get(iid, []) if a["iscrowd"] and a["category_id"] == k],
                                 float).reshape(-1, 4)
                ious = np.concatenate([iou_all[np.ix_(dk, gk)], iou_matrix(b[dk], crowd, crowd_of_a=True)], axis=1)
                ncg = len(gk)
                taken = np.zeros(ious.shape[1], bool)
                for r, d in enumerate(dk):  # pycocotools evaluateImg loop at a single threshold
                    best, m = IOU_POS, -1
                    for g in range(ious.shape[1]):
                        if taken[g] and g < ncg:
                            continue
                        if m > -1 and m < ncg and g >= ncg:
                            break
                        if ious[r, g] < best:
                            continue
                        best, m = ious[r, g], g
                    if m < 0:
                        continue
                    taken[m] = True
                    if m < ncg:
                        status[d] = TP
                        self.g_match[gi[gk[m]]] = n + d
                    else:
                        status[d] = IGN
                # error typing of the unmatched, non-ignored detections
                for r, d in enumerate(dk):
                    if status[d] != FP:
                        continue
                    row = iou_all[d] if len(gb) else np.zeros(0)
                    sm = np.where(gc == k, row, -1.0)
                    om = np.where(gc != k, row, -1.0)
                    si, oi = (int(sm.argmax()), int(om.argmax())) if len(row) else (-1, -1)
                    sv, ov = (sm[si], om[oi]) if len(row) else (-1.0, -1.0)
                    if sv >= 0:
                        same[d] = gi[si]
                    if ov >= 0:
                        other[d] = gi[oi]
                    ftype[d] = (1 if ov >= IOU_POS else 4 if sv >= IOU_POS else 2 if sv >= IOU_BG
                                else 3 if ov >= IOU_BG else 5)
            d_img += [iid] * len(s)
            d_cls.append(c), d_score.append(s), d_box.append(b), d_status.append(status), d_type.append(ftype)
            d_same.append(same), d_other.append(other)
            n += len(s)
        cat = lambda xs, dt, sh=(): np.concatenate(xs).astype(dt) if xs else np.zeros((0, *sh), dt)  # noqa: E731
        self.d_img = np.array(d_img, int)
        self.d_cls, self.d_score = cat(d_cls, int), cat(d_score, float)
        self.d_box, self.d_status = cat(d_box, float, (4,)), cat(d_status, int)
        self.d_type, self.d_same, self.d_other = cat(d_type, int), cat(d_same, int), cat(d_other, int)

    # ---- AP and oracles ----
    def _ap50(self, status, cls, npig) -> tuple[float, dict]:
        per = {}
        for k in self.classes:
            m = (cls == k) & (status != IGN) & (status != REM)
            per[k] = ap101(self.d_score[m], status[m] == TP, int(npig[k]))
        vals = [v for v in per.values() if not np.isnan(v)]
        return (float(np.mean(vals)) if vals else float("nan")), per

    def _npig(self) -> dict:
        return {k: int((self.g_cls == k).sum()) for k in self.classes}

    def baseline_ap50(self) -> tuple[float, dict]:
        return self._ap50(self.d_status, self.d_cls, self._npig())

    def oracle(self, name: str) -> tuple[float, dict]:
        """AP50 (mean, per class) after fixing one error type; see the module docstring."""
        status, cls, npig = self.d_status.copy(), self.d_cls.copy(), self._npig()
        matched = self.g_match >= 0
        if name in ("Dupe", "Bkg", "Both"):
            status[self.d_type == FPTYPES.index(name)] = REM
        elif name in ("Loc", "Cls"):
            tgt = self.d_same if name == "Loc" else self.d_other
            code = FPTYPES.index(name)
            for d in sorted(np.flatnonzero(self.d_type == code), key=lambda i: -self.d_score[i]):
                g = tgt[d]
                if g >= 0 and not matched[g]:
                    status[d], matched[g] = TP, True
                    if name == "Cls":
                        cls[d] = self.g_cls[g]
                else:
                    status[d] = REM
        elif name == "Missed":
            for k in self.classes:
                npig[k] -= int(((self.g_cls == k) & ~matched).sum())
        else:
            raise KeyError(name)
        return self._ap50(status, cls, npig)

    # ---- counts ----
    def counts(self, min_score: float = 0.0) -> dict:
        keep = self.d_score >= min_score
        out = {"n_dets": int(keep.sum()), "TP": int((keep & (self.d_status == TP)).sum()),
               "ignored": int((keep & (self.d_status == IGN)).sum())}
        for e in ERRORS:
            out[e] = int((keep & (self.d_type == FPTYPES.index(e)) & (self.d_status == FP)).sum())
        hit = np.zeros(len(self.g_match), bool)
        ok = self.g_match >= 0
        hit[ok] = self.d_score[self.g_match[ok]] >= min_score
        out["Missed"] = int((~hit).sum())
        out["n_gt"] = int(len(hit))
        return out

    def counts_by_class(self, min_score: float = 0.0) -> dict:
        res = {}
        for k in self.classes:
            keep = (self.d_score >= min_score) & (self.d_cls == k)
            row = {"n_dets": int(keep.sum()), "TP": int((keep & (self.d_status == TP)).sum())}
            for e in ERRORS:
                row[e] = int((keep & (self.d_type == FPTYPES.index(e)) & (self.d_status == FP)).sum())
            gk = self.g_cls == k
            ok = self.g_match >= 0
            hit = np.zeros(len(self.g_match), bool)
            hit[ok] = self.d_score[self.g_match[ok]] >= min_score
            row["Missed"], row["n_gt"] = int((gk & ~hit).sum()), int(gk.sum())
            res[KITTI_CLASSES.get(k, str(k))] = row
        return res

    # ---- recall bins ----
    def recall_bins(self) -> dict:
        hit = self.g_match >= 0
        dims = {
            "height": [(n, (self.g_height >= lo) & (self.g_height < hi)) for n, lo, hi in HEIGHT_BINS],
            "occlusion": [(n, self.g_occ == v) for n, v in OCC_BINS],
            "truncation": [(n, np.array([f(t) for t in self.g_trunc], bool)) for n, f in TRUNC_BINS],
        }
        out = {}
        for k in [*self.classes, 0]:
            cm = np.ones(len(hit), bool) if k == 0 else self.g_cls == k
            name = "all" if k == 0 else KITTI_CLASSES.get(k, str(k))
            out[name] = {"n": int(cm.sum()), "hits": int((cm & hit).sum()), "recall": _rate((cm & hit).sum(), cm.sum())}
            for dim, bins in dims.items():
                out[name][dim] = {n: {"n": int((cm & bm).sum()), "hits": int((cm & bm & hit).sum()),
                                      "recall": _rate((cm & bm & hit).sum(), (cm & bm).sum())} for n, bm in bins}
        return out

    # ---- gallery ranking ----
    def error_images(self, top: int = 3) -> list[tuple[int, int, int, int]]:
        """[(image_id, score, n_fp_ge_0.5, n_missed_ge_40px)] best first."""
        fp = self.d_img[(self.d_status == FP) & (self.d_score >= 0.5)]
        ms = self.g_img[(self.g_match < 0) & (self.g_height >= 40)]
        ids = sorted(set(fp.tolist()) | set(ms.tolist()))
        rows = [(i, int((fp == i).sum()) + int((ms == i).sum()), int((fp == i).sum()), int((ms == i).sum())) for i in ids]
        rows.sort(key=lambda r: (-r[1], r[0]))
        return rows[:top]

    def missed_large(self) -> set[int]:
        return {int(a) for a in self.g_ann[(self.g_match < 0) & (self.g_height >= 40)]}


def _rate(h, n) -> float | None:
    return float(h) / float(n) if n else None


def analysis_dict(an: Analysis) -> dict:
    base, base_per = an.baseline_ap50()
    name = lambda k: KITTI_CLASSES.get(k, str(k))  # noqa: E731
    res = {"AP50": base, "AP50_per_class": {name(k): v for k, v in base_per.items()}, "dAP50": {}, "dAP50_per_class": {}}
    for o in [*ERRORS, "Missed"]:
        ap, per = an.oracle(o)
        res["dAP50"][o] = ap - base
        res["dAP50_per_class"][o] = {name(k): (per[k] - base_per[k]) for k in an.classes
                                     if not (np.isnan(per[k]) or np.isnan(base_per[k]))}
    return res


# ---------------- drawing ----------------
def _font(size=13):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _box(draw, box, color, label, width, font, below=False):
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        tb = draw.textbbox((0, 0), label, font=font)
        w, h = tb[2] - tb[0], tb[3] - tb[1]
        ty = (y2 + 1) if below else max(y1 - h - 3, 0)
        draw.rectangle([x1, ty, x1 + w + 4, ty + h + 3], fill=color)
        draw.text((x1 + 2, ty - tb[1] + 1), label, fill="black", font=font)


def draw_image(gt_im: dict, an: Analysis | None, image_id: int, path: Path, highlight: set[int] | None = None):
    """GT green, detections >= 0.3 (red FP / orange otherwise); `highlight` = ann ids drawn magenta (shared misses)."""
    with Image.open(KITTI_DIR / gt_im["file_name"]) as f:
        img = f.convert("RGB")
    draw, font = ImageDraw.Draw(img), _font()
    gsel = [i for i in range(len(an.g_img)) if an.g_img[i] == image_id]
    for g in gsel:
        missed = an.g_match[g] < 0
        hl = highlight is not None and int(an.g_ann[g]) in highlight
        tag = KITTI_CLASSES[an.g_cls[g]] + (" MISS" if missed else "")
        _box(draw, an.g_box[g], (255, 0, 255) if hl else (0, 200, 0), tag if (missed or hl) else "", 3 if (hl or missed) else 2,
             font, below=True)
    for d in np.flatnonzero((an.d_img == image_id) & (an.d_score >= 0.3)):
        fp = an.d_status[d] == FP
        _box(draw, an.d_box[d], (230, 30, 30) if fp else (255, 150, 0), f"{KITTI_CLASSES[an.d_cls[d]]}:{an.d_score[d]:.2f}",
             2, font)
    if img.width > 1280:
        img = img.resize((1280, round(img.height * 1280 / img.width)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=80, optimize=True)


# ---------------- driver ----------------
def select_runs(split: str, models: list[str] | None, pred_dir: Path = PRED_DIR) -> list[Path]:
    out = []
    for p in sorted(pred_dir.glob("*.jsonl.gz")):
        name = p.name[: -len(".jsonl.gz")]
        parts = name.split("__")
        if len(parts) < 2 or parts[1] != split or "__arm-" in name:
            continue
        if models and parts[0] not in models and name not in models:
            continue
        out.append(p)
    return out


def map_all(preds: list[dict], open_vocab: bool) -> dict:
    rules = PredRules()  # headline: cyclist-merge (closed-set), KITTI-name prompts (open-vocab)
    return {p["image_id"]: map_predictions(p, rules, open_vocab)[:3] for p in preds}


def analyze_run(pred_path: Path, out_dir: Path, gt_cache: dict) -> tuple[dict, Analysis, dict]:
    meta = json.loads(Path(str(pred_path).replace(".jsonl.gz", ".meta.json")).read_text())
    split, ov = meta["split"], meta["detector_config"]["open_vocab"]
    if split not in gt_cache:
        gt_cache[split] = load_gt(split, GTRules(van="merge"))
    gt = gt_cache[split]
    an = Analysis(gt, map_all(load_preds(pred_path), ov))
    res = {"pred": meta["name"], "model": meta["model"], "split": split, "open_vocab": ov, "n_images": len(an.images),
           "variant": {"van": "merge", "cyclist": "prompt" if ov else "merge", "iou": IOU_POS, "max_det": MAX_DET},
           "counts": {"all": an.counts(0.0), "score>=0.3": an.counts(0.3)},
           "counts_by_class": {"all": an.counts_by_class(0.0), "score>=0.3": an.counts_by_class(0.3)},
           **analysis_dict(an), "recall100": an.recall_bins()}
    return res, an, gt


def make_gallery(res: dict, an: Analysis, gt: dict, out_dir: Path) -> list[dict]:
    ims = {im["id"]: im for im in gt["images"]}
    shown = []
    for rank, (iid, n, nfp, nms) in enumerate(an.error_images(3), 1):
        rel = f"gallery/{res['pred']}__{rank}.jpg"
        try:
            draw_image(ims[iid], an, iid, out_dir / rel)
        except FileNotFoundError:
            print(f"  gallery: image {ims[iid]['file_name']} missing, skipped")
            continue
        shown.append({"image_id": iid, "file": rel, "high_conf_fp": nfp, "missed_ge40px": nms})
    return shown


def make_shared(runs: dict[str, tuple[Analysis, dict]], out_dir: Path) -> list[dict]:
    if len(runs) < 2:
        return []
    common = None
    for an, _ in runs.values():
        s = an.missed_large()
        common = s if common is None else common & s
    if not common:
        return []
    an0, gt = next(iter(runs.values()))
    ims = {im["id"]: im for im in gt["images"]}
    per_img: dict[int, list[int]] = {}
    for a, i, h in zip(an0.g_ann, an0.g_img, an0.g_height):
        if int(a) in common:
            per_img.setdefault(int(i), []).append(int(a))
    heights = {int(a): h for a, h in zip(an0.g_ann, an0.g_height)}
    order = sorted(per_img, key=lambda i: (-len(per_img[i]), -sum(heights[a] for a in per_img[i]), i))[:3]
    shown = []
    for rank, iid in enumerate(order, 1):
        rel = f"gallery/shared_miss__{rank}.jpg"
        try:  # drawn with the first run's detections for context; the highlighted GT is missed by every closed-set run
            draw_image(ims[iid], an0, iid, out_dir / rel, highlight=set(per_img[iid]))
        except FileNotFoundError:
            continue
        shown.append({"image_id": iid, "file": rel, "n_common_missed": len(per_img[iid]),
                      "ann_ids": per_img[iid], "detections_from": next(iter(runs))})
    return shown


def _fmt(x, p=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def write_summary(out_dir: Path, split: str, shared: list[dict]):
    rs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json")) if p.name != "shared.json"]
    rs = [r for r in rs if r.get("split") == split]
    if not rs:
        return
    L = [f"# Failure analysis on {split}", "",
         "IoU 0.5, van-merge GT, ignore regions respected; detections mapped with the headline rules "
         "(closed-set cyclist-merge, open-vocab KITTI-name prompts). Definitions: docstring of `src/hw1/failure.py`.", ""]
    L += ["## Delta AP50 from fixing one error type (TIDE oracles)", "",
          "| run | AP50 | " + " | ".join(ERRORS) + " | Missed |", "|---|---|" + "---|" * (len(ERRORS) + 1)]
    for r in rs:
        L.append(f"| {r['pred']} | {_fmt(r['AP50'])} | " + " | ".join(_fmt(r["dAP50"][e]) for e in [*ERRORS, "Missed"]) + " |")
    for key, title in (("all", "all detections"), ("score>=0.3", "detections with score >= 0.3")):
        L += ["", f"## Error counts, {title}", "",
              "| run | dets | TP | ignored | " + " | ".join(ERRORS) + " | Missed GT | GT |",
              "|---|---|---|---|" + "---|" * (len(ERRORS) + 2)]
        for r in rs:
            c = r["counts"][key]
            L.append(f"| {r['pred']} | {c['n_dets']} | {c['TP']} | {c['ignored']} | "
                     + " | ".join(str(c[e]) for e in ERRORS) + f" | {c['Missed']} | {c['n_gt']} |")
    L += ["", "## Per-class Delta AP50", ""]
    classes = [n for n in KITTI_CLASSES.values()]
    for cls in classes:
        L += [f"**{cls}**", "", "| run | AP50 | " + " | ".join(ERRORS) + " | Missed |", "|---|---|" + "---|" * (len(ERRORS) + 1)]
        for r in rs:
            d = r["dAP50_per_class"]
            L.append(f"| {r['pred']} | {_fmt(r['AP50_per_class'].get(cls))} | "
                     + " | ".join(_fmt(d[e].get(cls)) for e in [*ERRORS, "Missed"]) + " |")
        L.append("")
    L += ["## Recall@100 (IoU 0.5) by GT attribute", "", "Cells are recall (n GT in the bin); n is run-independent.", ""]
    for cls in ["all", *classes]:
        for dim, bins in (("height", [b[0] for b in HEIGHT_BINS]), ("occlusion", [b[0] for b in OCC_BINS]),
                          ("truncation", [b[0] for b in TRUNC_BINS])):
            if not rs[0]["recall100"].get(cls, {}).get("n"):
                continue
            L += [f"**{cls}, by {dim}**", "", "| run | overall | " + " | ".join(bins) + " |", "|---|---|" + "---|" * len(bins)]
            for r in rs:
                rr = r["recall100"][cls]
                L.append(f"| {r['pred']} | {_fmt(rr['recall'])} | "
                         + " | ".join(f"{_fmt(rr[dim][b]['recall'])} ({rr[dim][b]['n']})" for b in bins) + " |")
            L.append("")
    if shared:
        L += ["## GT missed by every closed-set run (height >= 40 px)", ""]
        L += [f"- `{s['file']}`: image {s['image_id']}, {s['n_common_missed']} common miss(es)" for s in shared]
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="kitti_test")
    ap.add_argument("--models", nargs="*", default=None, help="model names or full prediction names")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--pred-dir", default=str(PRED_DIR))
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = select_runs(args.split, args.models, Path(args.pred_dir))
    print(f"failure analysis on {args.split}: {len(paths)} prediction files")
    gt_cache, closed = {}, {}
    for p in paths:
        if not Path(str(p).replace(".jsonl.gz", ".meta.json")).exists():
            print(f"  {p.name}: no meta file, skipped")
            continue
        res, an, gt = analyze_run(p, out_dir, gt_cache)
        res["gallery"] = make_gallery(res, an, gt, out_dir)
        (out_dir / f"{res['pred']}.json").write_text(json.dumps(res, indent=1, default=float))
        if not res["open_vocab"] and res["pred"] == f"{res['model']}__{res['split']}":
            closed[res["pred"]] = (an, gt)
        d = res["dAP50"]
        print(f"  {res['pred']:40s} AP50 {res['AP50']:.3f}  " + " ".join(f"d{k} {v:+.3f}" for k, v in d.items()))
    shared = make_shared(closed, out_dir)
    if shared:
        (out_dir / "shared.json").write_text(json.dumps(shared, indent=1))
    write_summary(out_dir, args.split, shared)
    print(f"wrote {out_dir}/summary.md")


if __name__ == "__main__":
    main()
