"""Regenerate every report figure and table from runs/metrics and runs/preds.

    python -m hw1.figures [--runs runs] [--only F1,T3]

Writes runs/figures/F*.png|pdf, runs/figures/captions.md and runs/tables/T*.md|csv. Reads only; nothing is
recomputed from predictions. Output is deterministic (no timestamps in PDF/PNG metadata).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- model registry
# colour = family hue (validated blue/orange/aqua/magenta), shade + marker + hatch tell members apart so the
# charts survive grayscale print.
BLUE, BLUE2 = "#2a78d6", "#17457f"
ORANGE = "#eb6834"
AQUA = "#159a6b"
MAG, MAG2 = "#e87ba4", "#a62c63"

MODELS = [
    dict(key="yolo11m", name="YOLO11m", family="one-stage CNN", model="yolo11m", tag="", coco="coco_val",
         color=BLUE, marker="o", hatch=""),
    dict(key="yolo26m", name="YOLO26m", family="one-stage CNN", model="yolo26m", tag="", coco="coco_val",
         color=BLUE2, marker="s", hatch="///"),
    dict(key="frcnn_v2", name="Faster R-CNN v2", family="two-stage", model="frcnn_v2", tag="", coco="coco_val",
         color=ORANGE, marker="^", hatch=""),
    dict(key="rtdetr_r50", name="RT-DETR R50", family="DETR / query-based", model="rtdetr_r50", tag="",
         coco="coco_val", color=AQUA, marker="D", hatch=""),
    dict(key="owlv2", name="OWLv2", family="open-vocabulary", model="owlv2", tag="", coco="coco_val1k",
         color=MAG, marker="P", hatch="", open_vocab=True),
    dict(key="gdino_t", name="Grounding DINO-T", family="open-vocabulary", model="gdino_t", tag="",
         coco="coco_val1k", color=MAG2, marker="X", hatch="///", open_vocab=True),
]
# extra configs: used in F1 and tables only
EXTRA = [
    dict(key="yolo26m_nms", name="YOLO26m (NMS head)", family="one-stage CNN", model="yolo26m", tag="nms",
         coco="coco_val", color=BLUE2, marker="s", hatch="///", hollow=True),
    dict(key="yolo11m_1280", name="YOLO11m @1280", family="one-stage CNN", model="yolo11m", tag="img1280",
         coco=None, color=BLUE, marker="h", hatch=""),
]
BY_KEY = {m["key"]: m for m in MODELS + EXTRA}
TABLE_MODELS = [BY_KEY[k] for k in ("yolo11m", "yolo11m_1280", "yolo26m", "yolo26m_nms", "frcnn_v2",
                                    "rtdetr_r50", "owlv2", "gdino_t")]

CLASSES = ["car", "truck", "pedestrian", "cyclist", "tram"]
CLS_NAMES = {"car": "Car", "truck": "Truck", "pedestrian": "Pedestrian", "cyclist": "Cyclist", "tram": "Tram"}
CORE3 = [("car", "car"), ("truck", "truck"), ("person", "pedestrian")]  # (COCO name, KITTI name)

# caveat strings (shared by captions, figure footnotes and table notes)
CAV_CPU = "Grounding DINO-T was measured on the M4 CPU (an interrupted parity check), so its latency is not comparable to the MPS runs."
CAV_SUBSET = "OWLv2 and Grounding DINO-T COCO numbers are on a seeded 1000-image val2017 subset (the others use all 5000)."
CAV_MEM = "Memory on MPS is the unified-memory allocator peak (torch.mps.driver_allocated_memory), not CUDA memory; the CPU run reports process RSS."
CAV_AP = "KITTI-style AP uses COCO-style 101-point interpolation and is not comparable to the KITTI leaderboard."
CAV_KMAP = "KITTI mAP is COCO-style mAP@[.5:.95] over the 5 mapped classes (car, truck, pedestrian, cyclist, tram); COCO mAP is over 80 classes."

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "hatch.linewidth": 0.7, "lines.linewidth": 1.4, "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- data access
class Runs:
    def __init__(self, root: Path):
        self.root = root
        self.metrics = root / "metrics"
        self.preds = root / "preds"
        self._c: dict[Path, dict | None] = {}

    def _load(self, p: Path):
        if p not in self._c:
            self._c[p] = json.loads(p.read_text()) if p.exists() else None
        return self._c[p]

    @staticmethod
    def pred_name(m, split):
        return f"{m['model']}__{split}" + (f"__{m['tag']}" if m["tag"] else "")

    def kitti_variant(self, m, kind="headline"):
        if kind == "headline":
            return "van-merge__cyc-prompt" if m.get("open_vocab") else "van-merge__cyc-merge"
        return {"naive": "van-merge__cyc-naive", "none": "van-merge__cyc-none", "van-ignore": "van-ignore__cyc-merge",
                "thr0.3": "van-merge__cyc-merge__thr0.3"}[kind]

    def kitti(self, m, kind="headline"):
        return self._load(self.metrics / f"{self.pred_name(m, 'kitti_test')}__{self.kitti_variant(m, kind)}.json")

    def coco(self, m, variant="default"):
        if not m.get("coco"):
            return None
        return self._load(self.metrics / f"{self.pred_name(m, m['coco'])}__{variant}.json")

    def meta(self, m, split):
        split = m["coco"] if split == "coco" else split
        if not split:
            return None
        return self._load(self.preds / f"{self.pred_name(m, split)}.meta.json")


def f3(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def nz(x):
    return float("nan") if x is None else x


def pc(d, cls, key="AP"):
    """Per-class value from a metrics dict, None when absent or -1 (COCOeval's 'no GT')."""
    if d is None:
        return None
    v = d["coco"]["per_class"].get(cls, {}).get(key)
    return None if v is None or v < 0 else v


def cm(d, key):
    if d is None:
        return None
    v = d["coco"].get(key)
    return None if v is None or v < 0 else v


# ---------------------------------------------------------------- output helpers
def save_fig(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.06, metadata={"Software": None})
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06,
                metadata={"CreationDate": None, "Creator": "hw1.figures", "Producer": "matplotlib"})
    plt.close(fig)
    print(f"  figure {name}")


def footnote(fig, text, y=-0.005, width=125):
    import textwrap
    fig.text(0.01, y, "\n".join(textwrap.wrap(text, width)), ha="left", va="top", fontsize=6.3, color="#52514e")


def bar_kw(m, alpha=1.0, hollow=False):
    if m["hatch"]:
        return dict(facecolor=m["color"], edgecolor="white", hatch=m["hatch"], linewidth=0.5)
    return dict(facecolor=m["color"], edgecolor=m["color"], linewidth=0.5)


def model_legend(fig, models, ncol=3, kind="bar", y=0.0, loc="upper center", extra=()):
    handles = []
    for m in models:
        if kind == "bar":
            handles.append(Patch(label=m["name"], **bar_kw(m)))
        else:
            handles.append(Line2D([], [], marker=m["marker"], color=m["color"], markersize=5.5, lw=1.2,
                                  markerfacecolor="white" if m.get("hollow") else m["color"], label=m["name"],
                                  markeredgecolor=m["color"], markeredgewidth=1.0))
    handles += list(extra)
    return fig.legend(handles=handles, loc=loc, ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y),
                      handlelength=1.8, columnspacing=1.4)


def style_ax(ax, ygrid=True):
    ax.grid(axis="x", visible=False) if ygrid else ax.grid(axis="y", visible=False)
    ax.tick_params(length=2.5, width=0.6)


# ---------------------------------------------------------------- figures
def fig_f1(R: Runs, out: Path, notes):
    """mAP vs sustained latency (log x). Left COCO, right KITTI."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharex=True)
    xt = [20, 50, 100, 200, 500, 1000, 3000]
    for ax, split, title in ((axes[0], "coco", "COCO val2017"), (axes[1], "kitti", "KITTI val-3769")):
        ymax = 0
        for m in MODELS + EXTRA:
            d = R.coco(m) if split == "coco" else R.kitti(m)
            mt = R.meta(m, "coco" if split == "coco" else "kitti_test")
            if d is None or mt is None:
                continue
            x, y = mt["latency"]["sustained"]["median_ms"], d["coco"]["mAP"]
            ymax = max(ymax, y)
            cpu = mt["detector_config"].get("device") == "cpu"
            ax.scatter([x], [y], s=46, marker=m["marker"], zorder=3, linewidths=1.2,
                       facecolor="white" if (m.get("hollow") or cpu) else m["color"], edgecolor=m["color"])
            if cpu:
                ax.annotate("CPU run,\nnot comparable", (x, y), xytext=(0, 9), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.3, color="#52514e")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlim(14, 4200)
        ax.set_ylim(0, ymax * 1.18)
        ax.set_title(title)
        ax.set_xlabel("sustained median latency per image (ms, log scale)")
        ax.tick_params(axis="x", labelsize=6.8)
        ax.grid(axis="x", visible=True)
    axes[0].set_ylabel("mAP@[.5:.95]")
    axes[1].set_ylabel("mAP@[.5:.95]")
    axes[0].text(0.02, 0.03, "80-class mAP; OWLv2 and\nGrounding DINO-T on 1k subset", transform=axes[0].transAxes,
                 va="bottom", fontsize=6.3, color="#52514e")
    axes[1].text(0.02, 0.03, "5-class mAP (car, truck, ped.,\ncyclist, tram); YOLO11m also at 1280", transform=axes[1].transAxes,
                 va="bottom", fontsize=6.3, color="#52514e")
    fig.subplots_adjust(wspace=0.28)
    cpu_h = Line2D([], [], marker="X", color=MAG2, markerfacecolor="white", linestyle="", markersize=5.5,
                   label="hollow = CPU run / NMS head")
    leg = model_legend(fig, MODELS + EXTRA[:1] + EXTRA[1:], ncol=4, kind="line", y=-0.01)
    footnote(fig, CAV_CPU + " " + CAV_SUBSET + " Latency is batch 1 including pre/post-processing; the x range is shared. "
             "Hollow markers: YOLO26m with an NMS head, and the CPU-measured Grounding DINO-T.", y=-0.2)
    save_fig(fig, out, "F1_speed_accuracy")
    notes["F1_speed_accuracy"] = ("Speed-accuracy trade-off. Sustained median latency per image at batch 1 (pre + forward + post) against "
                                  "mAP@[.5:.95], on COCO val2017 (left) and KITTI val-3769 (right). " + CAV_CPU + " " + CAV_SUBSET + " " + CAV_KMAP)


def fig_f2(R: Runs, out: Path, notes):
    """Slope chart COCO -> KITTI for the comparable classes plus their mean."""
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.9), sharey=True)
    series = {}
    for m in MODELS:
        dc, dk = R.coco(m), R.kitti(m)
        if dc is None or dk is None:
            continue
        v = {}
        for cc, kc in CORE3:
            v[kc] = (pc(dc, cc), pc(dk, kc))
        v["mean"] = (np.mean([v[k][0] for _, k in CORE3]), np.mean([v[k][1] for _, k in CORE3]))
        series[m["key"]] = v
    titles = [("car", "Car"), ("truck", "Truck"), ("pedestrian", "Person → Pedestrian"), ("mean", "Mean of the 3")]
    for ax, (k, t) in zip(axes, titles):
        for m in MODELS:
            if m["key"] not in series:
                continue
            a, b = series[m["key"]][k]
            ax.plot([0, 1], [a, b], color=m["color"], marker=m["marker"], markersize=5, lw=1.3,
                    ls="--" if m["hatch"] else "-", zorder=3)
        ax.set_xticks([0, 1], ["COCO", "KITTI"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(0, 0.82)
        ax.set_title(t, fontsize=7.8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=2.5, width=0.6)
    axes[0].set_ylabel("AP@[.5:.95]")
    fig.subplots_adjust(wspace=0.12)
    model_legend(fig, MODELS, ncol=6, kind="line", y=0.02)
    footnote(fig, "Same AP scale in all panels. 'Mean of the 3' is mAP over car, truck and person/pedestrian only (the classes both datasets "
             "share), so it differs from the headline 80-class / 5-class mAP. Dashed = second model of a family. "
             + CAV_SUBSET, y=-0.09, width=135)
    save_fig(fig, out, "F2_domain_shift")
    notes["F2_domain_shift"] = ("Domain shift COCO val2017 to KITTI, per comparable class (car, truck, person/pedestrian) and the mean AP over those three. "
                                "Lines connect the same model. " + CAV_SUBSET)
    return series


def fig_f3(R: Runs, out: Path, notes):
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    n = len(MODELS)
    off = np.linspace(-0.34, 0.34, n)
    for i, m in enumerate(MODELS):
        d = R.kitti(m)
        if d is None:
            continue
        for j, c in enumerate(CLASSES):
            b = d["bootstrap"].get(c)
            if b is None:
                continue
            ax.errorbar(j + off[i], b["point"], yerr=[[b["point"] - b["lo"]], [b["hi"] - b["point"]]], fmt=m["marker"],
                        color=m["color"], markersize=4.8, capsize=2, elinewidth=1.0, capthick=0.8,
                        markerfacecolor=m["color"], markeredgewidth=1.0, zorder=3)
    for j in range(len(CLASSES) - 1):
        ax.axvline(j + 0.5, color="#e4e3df", lw=0.6)
    ax.set_xticks(range(len(CLASSES)), [CLS_NAMES[c] for c in CLASSES])
    ax.set_xlim(-0.55, len(CLASSES) - 0.45)
    ax.set_ylim(0, None)
    ax.set_ylabel("KITTI AP@[.5:.95]  (95% bootstrap CI)")
    ax.grid(axis="x", visible=False)
    model_legend(fig, MODELS, ncol=6, kind="line", y=0.0)
    footnote(fig, "Bars are 95% percentile CIs from 500 image-level bootstrap resamples of kitti_test (3769 images). Open-vocabulary models use the default class vocabulary with the Cyclist merge. " + CAV_KMAP, y=-0.09)
    save_fig(fig, out, "F3_kitti_per_class")
    notes["F3_kitti_per_class"] = ("KITTI per-class AP@[.5:.95] for every model with 95% bootstrap confidence intervals (500 resamples over images). "
                                   "Van counts as car; Cyclist is the person+bicycle union box (merge) for closed-set models. " + CAV_KMAP)


def fig_f4(R: Runs, out: Path, notes):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)
    keys = [("AP_small", "Small"), ("AP_medium", "Medium"), ("AP_large", "Large")]
    ys = np.arange(len(MODELS))[::-1]
    h = 0.36
    for ax, (k, t) in zip(axes, keys):
        for y, m in zip(ys, MODELS):
            dc, dk = R.coco(m), R.kitti(m)
            a, b = cm(dc, k), cm(dk, k)
            if a is not None:
                ax.barh(y + h / 2, a, height=h * 0.92, facecolor=m["color"], alpha=0.30, edgecolor=m["color"], linewidth=0.8)
                ax.text(a + 0.008, y + h / 2, f"{a:.2f}", va="center", fontsize=6, color="#52514e")
            if b is not None:
                ax.barh(y - h / 2, b, height=h * 0.92, **bar_kw(m))
                ax.text(b + 0.008, y - h / 2, f"{b:.2f}", va="center", fontsize=6, color="#0b0b0b")
        ax.set_title(t + " objects")
        ax.set_xlim(0, 0.85)
        ax.set_xlabel("AP@[.5:.95]")
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=2.5, width=0.6)
    axes[0].set_yticks(ys, [m["name"] for m in MODELS])
    fig.subplots_adjust(wspace=0.08)
    fig.legend(handles=[Patch(facecolor="#bbbbbb", edgecolor="#555555", label="COCO val2017 (light bar)"),
                        Patch(facecolor="#555555", edgecolor="#555555", label="KITTI val-3769 (solid bar)")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    footnote(fig, "COCO small/medium/large = area < 32^2, 32^2-96^2, > 96^2 px; the same thresholds are applied to KITTI boxes. KITTI AP_S/M/L "
             "is over the 5 mapped classes, COCO over 80 classes. " + CAV_SUBSET, y=-0.075)
    save_fig(fig, out, "F4_scale")
    notes["F4_scale"] = ("AP by object size (small / medium / large, COCO area thresholds) on COCO (light) and KITTI (solid), per model. "
                         + CAV_KMAP + " " + CAV_SUBSET)


ABL = [("naive", "Cyclist =\nbicycle only\n(naive)", "kitti"), ("none", "Cyclist\nclass\ndropped", "kitti"),
       ("van-ignore", "Van as\nignore\nregion", "kitti"), ("thr0.3", "Score >= 0.3\n(KITTI)", "kitti"),
       ("coco-thr0.3", "Score >= 0.3\n(COCO)", "coco")]


def ablation_deltas(R: Runs, m):
    """{ablation key: (abs, delta)} relative to the headline run of the same model."""
    out = {}
    hk = R.kitti(m)
    for k, _, kind in ABL:
        if kind == "kitti":
            d = R.kitti(m, k) if hk is not None else None
            base = hk
        else:
            d, base = R.coco(m, "thr0.3"), R.coco(m)
        if d is None or base is None:
            out[k] = (None, None)
        else:
            out[k] = (d["coco"]["mAP"], d["coco"]["mAP"] - base["coco"]["mAP"])
    return out


def fig_f5(R: Runs, out: Path, notes):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    n = len(MODELS)
    w = 0.13
    off = (np.arange(n) - (n - 1) / 2) * w
    na = {}
    for i, m in enumerate(MODELS):
        dl = ablation_deltas(R, m)
        for j, (k, _, _) in enumerate(ABL):
            v = dl[k][1]
            if v is None:
                na.setdefault(j, []).append(m["name"])
                continue
            ax.bar(j + off[i], v, width=w * 0.92, **bar_kw(m))
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(range(len(ABL)), [a[1] for a in ABL])
    ax.set_ylabel("Δ mAP vs headline run\n(same model, same split)")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, max(ymax, 0.004))
    for j, names in na.items():
        ax.text(j, 0.015, "n/a: " + ", ".join("OWLv2" if x == "OWLv2" else "GDINO-T" if x.startswith("Ground") else x for x in names),
                ha="center", va="bottom", fontsize=5.8, color="#52514e", transform=ax.get_xaxis_transform())
    model_legend(fig, MODELS, ncol=6, y=-0.12)
    footnote(fig, "Each bar = mAP(variant) - mAP(headline) for one model; negative means the variant scores lower. Headline = Van merged into Car, Cyclist = "
             "person+bicycle union box (closed-set) or prompted cyclist (open-vocab), score threshold 0.001 (0 for DETR). KITTI ablations were not run for "
             "OWLv2 / Grounding DINO-T (marked n/a); their Cyclist is already prompted. "
             + CAV_KMAP + " " + CAV_SUBSET, y=-0.2)
    save_fig(fig, out, "F5_ablations")
    notes["F5_ablations"] = ("Ablations: change in mAP relative to each model's headline run. Cyclist naive (bicycle boxes only) and Cyclist dropped, Van as ignore "
                             "region instead of car, and a 0.3 score cutoff (ngdet's default) on KITTI and on COCO. " + CAV_KMAP + " " + CAV_SUBSET)


def fig_f6(R: Runs, out: Path, notes):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    levels = ["easy", "moderate", "hard"]
    n = len(MODELS)
    w = 0.13
    off = (np.arange(n) - (n - 1) / 2) * w
    for ax, (key, t) in zip(axes, (("car@0.7", "Car, IoU 0.7"), ("pedestrian@0.5", "Pedestrian, IoU 0.5"))):
        for i, m in enumerate(MODELS):
            d = R.kitti(m)
            if d is None:
                continue
            vals = [d["difficulty"][lv]["kitti_AP"][key] for lv in levels]
            ax.bar(np.arange(3) + off[i], vals, width=w * 0.92, **bar_kw(m))
        ax.set_xticks(range(3), [l.capitalize() for l in levels])
        ax.set_title(t)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=2.5, width=0.6)
    axes[0].set_ylabel("KITTI-style AP (101-pt)")
    fig.subplots_adjust(wspace=0.08)
    model_legend(fig, MODELS, ncol=6, y=0.01)
    footnote(fig, CAV_AP + " Easy / moderate / hard are implemented as ignore regions (height, occlusion, truncation) following the KITTI definitions. "
             "Cyclist@0.5 is in table T4.", y=-0.075)
    save_fig(fig, out, "F6_difficulty")
    notes["F6_difficulty"] = ("KITTI-style AP by difficulty (easy, moderate, hard) for Car at IoU 0.7 and Pedestrian at IoU 0.5. " + CAV_AP)


def agent_data(R: Runs):
    logs = R.root / "agent_logs"
    traj = {}
    for d in sorted(logs.glob("*__kitti_dev__seed*")):
        its = sorted(d.glob("iter_*.json"))
        if not its:
            continue
        ys = [json.loads(p.read_text())["mAP"] for p in its]
        det, seed = d.name.split("__")[0], d.name.rsplit("seed", 1)[1]
        base = json.loads((d / "baselines.json").read_text()) if (d / "baselines.json").exists() else {}
        traj[(det, seed)] = (ys, {k: v["mAP"] for k, v in base.items() if isinstance(v, dict) and "mAP" in v})
    arms = {}
    for p in sorted(R.metrics.glob("*__arm-*.json")):
        det = p.name.split("__")[0]
        arm = p.name.split("__arm-")[1].split("__")[0]
        d = json.loads(p.read_text())
        b = d.get("bootstrap", {}).get("mAP", {})
        arms[(det, arm)] = (d["coco"]["mAP"], b.get("lo"), b.get("hi"))
    return traj, arms


def fig_f7(R: Runs, out: Path, notes):
    traj, arms = agent_data(R)
    if not traj and not arms:
        print("  F7 skipped (no agent iteration logs or arm metrics yet)")
        return
    det_style = {"owlv2": (MAG, "OWLv2", "P"), "gdino_t": (MAG2, "Grounding DINO-T", "X")}
    ncol = int(bool(traj)) + int(bool(arms))
    fig, axes = plt.subplots(1, ncol, figsize=(7.0, 3.0), squeeze=False)
    axes = list(axes[0])
    if traj:
        ax = axes.pop(0)
        seeds = sorted({s for _, s in traj})
        ls = {s: ["-", "--", ":", "-."][i % 4] for i, s in enumerate(seeds)}
        base_drawn = set()
        for (det, seed), (ys, base) in traj.items():
            c, nm, mk = det_style.get(det, ("#444444", det, "o"))
            ax.plot(range(len(ys)), ys, color=c, marker=mk, ms=4, ls=ls[seed], label=f"{nm}, seed {seed}")
            for bk, bl in (("raw_names", "raw names"), ("ngdet_synonyms", "ngdet synonyms")):
                if bk in base and (det, bk) not in base_drawn:
                    base_drawn.add((det, bk))
                    ax.axhline(base[bk], color=c, lw=0.9, ls=(0, (1, 1.5)) if bk == "raw_names" else (0, (4, 2)), alpha=0.9)
                    ax.text(0.0, base[bk], f" {nm.split()[0]} {bl}", fontsize=5.8, va="bottom", color="#52514e")
        ax.set_xlabel("agent evaluation (iteration)")
        ax.set_ylabel("dev mAP@[.5:.95] (kitti_dev, 500 imgs)")
        ax.set_title("Agent trajectory (dev)")
        ax.legend(frameon=False, loc="lower right")
    if arms:
        ax = axes.pop(0)
        order = ["raw_names", "ngdet_synonyms", "oneshot", "best"]
        rows = sorted(arms, key=lambda k: (k[0], next((i for i, o in enumerate(order) if k[1].startswith(o)), 9), k[1]))
        for i, k in enumerate(rows):
            v, lo, hi = arms[k]
            c, nm, mk = det_style.get(k[0], ("#444444", k[0], "o"))
            ax.errorbar(v, i, xerr=None if lo is None else [[v - lo], [hi - v]], fmt=mk, color=c, capsize=2, ms=5)
        ax.set_yticks(range(len(rows)), [f"{det_style.get(d, (0, d))[1]}: {a}" for d, a in rows])
        ax.invert_yaxis()
        ax.set_xlabel("kitti_test mAP@[.5:.95] (95% CI)")
        ax.set_title("Test-split arms")
        ax.grid(axis="y", visible=False)
    if ncol == 2:
        fig.subplots_adjust(wspace=0.85)
    footnote(fig, "Prompt maps are chosen on kitti_dev only; the selected map is run once on kitti_test. " + CAV_KMAP, y=-0.02)
    save_fig(fig, out, "F7_agent")
    notes["F7_agent"] = ("Agent experiment: dev mAP per evaluation with the human baselines as horizontal lines, and kitti_test mAP with 95% CIs per prompt arm. "
                         + CAV_KMAP)


# ---------------------------------------------------------------- tables
def write_table(out: Path, name, header, rows_csv, rows_md=None, notes=(), md_sections=None):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows_csv)

    def pipe(h, rows):
        esc = lambda s: str(s).replace("|", "\\|")
        lines = ["| " + " | ".join(esc(x) for x in h) + " |", "|" + "|".join("---" for _ in h) + "|"]
        lines += ["| " + " | ".join(esc(x) for x in r) + " |" for r in rows]
        return "\n".join(lines)

    parts = []
    if md_sections:
        for title, h, rows in md_sections:
            parts.append(f"**{title}**\n\n{pipe(h, rows)}")
    else:
        parts.append(pipe(header, rows_md if rows_md is not None else rows_csv))
    txt = "\n\n".join(parts) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    (out / f"{name}.md").write_text(txt)
    print(f"  table {name}")


def ci_str(b):
    return "" if not b else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


def table_t1(R: Runs, out: Path):
    setup_h = ["Model", "Family", "Checkpoint", "Input resolution", "Score thr.", "NMS IoU", "Device"]
    acc_cols = ["mAP", "AP50", "AP75", "APs", "APm", "APl", "AR@100"]
    acc_keys = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR@100"]
    csv_h = setup_h + ["COCO n_images"] + [f"COCO {c}" for c in acc_cols] + ["KITTI mAP lo", "KITTI mAP hi"] + \
        [f"KITTI {c}" for c in acc_cols] + ["burst ms", "burst FPS", "sustained ms", "sustained FPS", "COCO sustained ms",
                                            "KITTI peak mem MB", "COCO peak mem MB", "mem kind"]
    rows, setup, coco_t, kitti_t, speed = [], [], [], [], []
    for m in TABLE_MODELS:
        dk, dc = R.kitti(m), R.coco(m)
        mk, mc = R.meta(m, "kitti_test"), R.meta(m, "coco")
        if dk is None or mk is None:
            continue
        c = mk["detector_config"]
        nms = c.get("nms_iou")
        dev = c.get("device", "").upper() + ("*" if c.get("device") == "cpu" else "")
        s = [m["name"], m["family"], Path(str(c.get("checkpoint", ""))).name or "", c.get("input", ""),
             c.get("score_thresh", ""), "none" if nms in (None, "none") else nms, dev]
        cv = [dc["coco"][k] if dc else None for k in acc_keys]
        kv = [dk["coco"][k] for k in acc_keys]
        b = dk["bootstrap"].get("mAP", {})
        lat = mk["latency"]
        mem = mk["memory"]
        memv = mem.get("peak_driver_allocated_mb", mem.get("peak_mb"))
        cmem = None
        if mc:
            cmem = mc["memory"].get("peak_driver_allocated_mb", mc["memory"].get("peak_mb"))
        kind = {"mps_unified": "MPS unified-memory allocator peak", "cpu_rss": "CPU process RSS peak"}.get(mem["kind"], mem["kind"])
        sp = [lat["burst"]["median_ms"], lat["burst"]["fps_median"], lat["sustained"]["median_ms"], lat["sustained"]["fps_median"],
              mc["latency"]["sustained"]["median_ms"] if mc else None, memv, cmem, kind]
        rows.append(s + [dc["n_images"] if dc else None] + [nz(x) for x in cv] + [b.get("lo"), b.get("hi")] + kv + sp)
        setup.append(s)
        coco_t.append([m["name"] + ("" if not m.get("open_vocab") else " (1k)"), dc["n_images"] if dc else ""] + [f3(x) for x in cv])
        kitti_t.append([m["name"], f"{b.get('point', dk['coco']['mAP']):.3f} [{b.get('lo', float('nan')):.3f}, {b.get('hi', float('nan')):.3f}]"]
                       + [f3(x) for x in kv[1:]])
        speed.append([m["name"] + (" (CPU*)" if c.get("device") == "cpu" else ""), f"{lat['burst']['median_ms']:.1f}", f"{lat['burst']['fps_median']:.1f}",
                      f"{lat['sustained']['median_ms']:.1f}", f"{lat['sustained']['fps_median']:.1f}",
                      f"{memv:.1f}", f"{cmem:.1f}" if cmem is not None else "-", kind])
    rows_csv = [[f3(x) if isinstance(x, float) and h.split()[-1] in acc_cols + ["lo", "hi"] else
                 (round(x, 2) if isinstance(x, float) else ("" if x is None else x)) for h, x in zip(csv_h, r)] for r in rows]
    sections = [
        ("T1a. Setup", setup_h, setup),
        ("T1b. Accuracy, COCO val2017 (AP, AR in [0,1])", ["Model", "n images"] + acc_cols, coco_t),
        ("T1c. Accuracy, KITTI val-3769 (mAP with 95% bootstrap CI)", ["Model", "mAP [95% CI]"] + acc_cols[1:], kitti_t),
        ("T1d. Speed and memory (KITTI runs, batch 1, pre + forward + post)",
         ["Model", "burst ms", "burst FPS", "sustained ms", "sustained FPS", "peak mem MB (KITTI run)", "peak mem MB (COCO run)", "memory kind"], speed),
    ]
    notes = ["Burst = median of the first 200 timed images; sustained = median over the whole run (after 10 warm-up images). FPS = 1000 / median ms.",
             "* " + CAV_CPU + " Its memory is process RSS, not an allocator peak.",
             CAV_SUBSET + " They also trained on Visual Genome, which overlaps COCO, so COCO is not clean zero-shot for them (KITTI is).",
             CAV_MEM,
             "Peak MPS memory differs a lot between runs of the same model (for example YOLO11m 1156 MB on COCO, 162 MB on KITTI at 640, 1146 MB on KITTI at 1280); "
             "it tracks input size: Ultralytics letterboxes the wide KITTI frames to about 640x192 at imgsz 640, versus 640x640 "
             "for most COCO images and about 1280x384 at imgsz 1280, while RT-DETR and OWLv2 resize to fixed squares and use the "
             "same memory on both datasets. It is the allocator peak, so treat it as indicative, not as model size.",
             CAV_KMAP + " YOLO11m @1280 has no COCO run.",
             "Score threshold 0.001 (0.0 for RT-DETR) and max 100 detections for every model; NMS IoU as each model ships. OWLv2 / Grounding DINO-T KITTI "
             "numbers use the default class vocabulary (headline variant cyc-prompt)."]
    write_table(out, "T1_main_results", csv_h, rows_csv, notes=notes, md_sections=sections)


def table_t2(R: Runs, out: Path):
    h_csv = ["Model", "mAP", "mAP lo", "mAP hi"]
    for c in CLASSES:
        h_csv += [f"{CLS_NAMES[c]} AP", f"{CLS_NAMES[c]} lo", f"{CLS_NAMES[c]} hi"]
    h_md = ["Model", "mAP [95% CI]"] + [CLS_NAMES[c] for c in CLASSES]
    rc, rm = [], []
    for m in TABLE_MODELS:
        d = R.kitti(m)
        if d is None:
            continue
        bs = d["bootstrap"]
        r = [m["name"]]
        for k in ["mAP"] + CLASSES:
            b = bs.get(k, {})
            r += [f3(b.get("point")), f3(b.get("lo")), f3(b.get("hi"))]
        rc.append(r)
        rm.append([m["name"], ci_str(bs.get("mAP"))] + [ci_str(bs.get(c)) for c in CLASSES])
    n_gt = {}
    d = R.kitti(MODELS[0])
    if d:
        n_gt = {c: d["coco"]["per_class"][c]["n_gt"] for c in CLASSES}
    notes = ["AP@[.5:.95] on KITTI val-3769; 95% percentile CIs from 500 image-level bootstrap resamples. Van is merged into car; Cyclist = person+bicycle union box "
             "(closed-set) or prompted (open-vocab).",
             "Ground-truth counts (non-ignored): " + ", ".join(f"{CLS_NAMES[c]} {n}" for c, n in n_gt.items()) + ".",
             CAV_KMAP, "Grounding DINO-T latency is not part of this table; see T1 for its CPU caveat."]
    write_table(out, "T2_kitti_per_class", h_csv, rc, rm, notes)


def table_t3(R: Runs, out: Path):
    h = ["Model", "Headline KITTI mAP"]
    for k, lab, _ in ABL:
        lab = lab.replace("\n", " ")
        h += [f"{lab}: mAP", f"{lab}: Δ"]
    h += ["Cyclist AP merge/prompt", "Cyclist AP naive", "Cyclist AP none"]
    h += ["COCO headline mAP"]
    rows = []
    for m in TABLE_MODELS:
        hk = R.kitti(m)
        if hk is None:
            continue
        dl = ablation_deltas(R, m)
        dc = R.coco(m)
        r = [m["name"], f3(hk["coco"]["mAP"])]
        for k, _, _ in ABL:
            a, dd = dl[k]
            r += [f3(a), "" if dd is None else f"{dd:+.3f}"]
        r += [f3(pc(hk, "cyclist"))]
        for k in ("naive", "none"):
            r += [f3(pc(R.kitti(m, k), "cyclist"))]
        r += [f3(dc["coco"]["mAP"]) if dc else ""]
        rows.append(r)
    notes = ["Δ = variant mAP - headline mAP of the same model and split (KITTI columns relative to the KITTI headline, COCO column relative to the COCO headline). Blank = not run.",
             "Naive: only bicycle boxes count as Cyclist. None: no detection is mapped to Cyclist, so Cyclist AP is 0 while its GT is still scored. Van-ignore: Van GT becomes an ignore region instead of merging into car. "
             "Score >= 0.3: predictions below 0.3 removed before scoring (ngdet's default).",
             "The KITTI ablations were run for the closed-set models only; OWLv2 / Grounding DINO-T prompt Cyclist directly. " + CAV_KMAP + " " + CAV_SUBSET,
             "'Cyclist AP' columns are KITTI Cyclist AP@[.5:.95] under each treatment (headline merge, naive, none); naive and none give ~0 by construction (a COCO detector has no Cyclist class)."]
    write_table(out, "T3_ablations", h, rows, notes=notes)


def table_t4(R: Runs, out: Path):
    levels = ["easy", "moderate", "hard"]
    cols = [("car@0.7", "Car@0.7"), ("pedestrian@0.5", "Ped@0.5"), ("cyclist@0.5", "Cyc@0.5")]
    h = ["Model"] + [f"{lab} {lv}" for _, lab in cols for lv in levels] + [f"COCO-style mAP {lv}" for lv in levels]
    rows = []
    for m in TABLE_MODELS:
        d = R.kitti(m)
        if d is None:
            continue
        r = [m["name"]]
        for k, _ in cols:
            r += [f3(d["difficulty"][lv]["kitti_AP"].get(k)) for lv in levels]
        r += [f3(d["difficulty"][lv]["coco_mAP"]) for lv in levels]
        rows.append(r)
    notes = [CAV_AP, "Easy / moderate / hard use the KITTI height, occlusion and truncation rules as ignore regions; each level includes the easier ones' objects as in KITTI.",
             "Last three columns: COCO-style mAP@[.5:.95] over the 5 classes with the same difficulty filtering.", CAV_KMAP]
    write_table(out, "T4_difficulty", h, rows, notes=notes)


IGNORE_META = re.compile(r"coco_val200|kitti_dev50|__n\d+|parity")


def table_t5(R: Runs, out: Path):
    h = ["Model", "Split", "Tag", "n images", "Device", "Host", "torch", "torchvision", "ultralytics", "transformers", "Command", "Wall time (s)", "src hash"]
    rows = []
    for p in sorted(R.preds.glob("*.meta.json")):
        if IGNORE_META.search(p.name):
            continue
        m = R._load(p)
        c = m["detector_config"]
        v = c.get("versions", {})
        parts = m["name"].split("__")
        split = parts[1] if len(parts) > 1 else m.get("split", "")
        tag = "__".join(parts[2:])
        key = next((x["name"] for x in MODELS + EXTRA if x["model"] == parts[0] and x["tag"] == tag), None)
        disp = next((x["name"] for x in MODELS if x["model"] == parts[0]), parts[0])
        rows.append([disp, split, tag, m["n_images"], c.get("device", ""), m.get("env", {}).get("host", ""), v.get("torch", ""),
                     v.get("torchvision", ""), v.get("ultralytics", ""), v.get("transformers", ""), m.get("command", ""),
                     f"{m['wall_s']:.1f}", m.get("src_hash", "")])
    order = {x["model"]: i for i, x in enumerate(MODELS)}
    rows.sort(key=lambda r: (order.get(next((x["model"] for x in MODELS if x["name"] == r[0]), ""), 99), r[1], r[2]))
    notes = ["One row per prediction run in runs/preds (dev-VM sanity runs and the MPS/CPU parity checks are excluded). Every run is deterministic given the command: fixed seeds for subsets, batch size 1.",
             "Wall time covers model load, inference and writing predictions; scoring is separate (python -m hw1.score).",
             CAV_SUBSET + " " + CAV_CPU,
             "Full command history: runs/COMMANDS.md."]
    write_table(out, "T5_reproducibility", h, rows, notes=notes)


# ---------------------------------------------------------------- captions
def write_captions(out: Path, notes):
    lines = ["# Suggested figure captions", "", "Generated by hw1.figures; each caption carries the caveats that apply to its figure.", ""]
    for k in sorted(notes):
        lines += [f"**{k}.** {notes[k]}", ""]
    (out / "captions.md").write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--only", default="", help="comma list, e.g. F1,T3 (default: everything)")
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    R = Runs(runs)
    figs, tabs = runs / "figures", runs / "tables"
    only = {s.strip().upper() for s in a.only.split(",") if s.strip()}
    want = lambda k: not only or k in only
    notes = {}
    for k, fn in (("F1", fig_f1), ("F2", fig_f2), ("F3", fig_f3), ("F4", fig_f4), ("F5", fig_f5), ("F6", fig_f6), ("F7", fig_f7)):
        if want(k):
            fn(R, figs, notes)
    for k, fn in (("T1", table_t1), ("T2", table_t2), ("T3", table_t3), ("T4", table_t4), ("T5", table_t5)):
        if want(k):
            fn(R, tabs)
    if not only:
        write_captions(figs, notes)


if __name__ == "__main__":
    main()
