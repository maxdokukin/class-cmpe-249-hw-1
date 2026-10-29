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
    dict(key="yolo26m", name="YOLO26m (NMS-free)", family="one-stage CNN", model="yolo26m", tag="", coco="coco_val",
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
    dict(key="yolo26m_nms", name="YOLO26m (NMS)", family="one-stage CNN", model="yolo26m", tag="nms",
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
def save_fig(fig, out: Path, name: str, dpi=300, pdf=True):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.06, metadata={"Software": None})
    if pdf:
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
                   label="hollow = CPU run / NMS")
    leg = model_legend(fig, MODELS + EXTRA[:1] + EXTRA[1:], ncol=4, kind="line", y=-0.01)
    footnote(fig, CAV_CPU + " " + CAV_SUBSET + " Latency is batch 1 including pre/post-processing; the x range is shared. "
             "Hollow markers: YOLO26m with the NMS head (Ultralytics default; the solid YOLO26m run used the opt-in NMS-free head), and the CPU-measured Grounding DINO-T.", y=-0.2)
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


# ---------------------------------------------------------------- agent study (F7, T6)
# agent models get their own colour set (validated: gold, red, violet, green), distinct from the detector palette.
AGENTS = [
    dict(key="haiku-4-5", name="Haiku 4.5", color="#b8860b", marker="o"),
    dict(key="haiku-5-5", name="Haiku 5.5", color="#b3262e", marker="s"),
    dict(key="sonnet-5-5", name="Sonnet 5.5", color="#7b4fc4", marker="^"),
    dict(key="opus-5-5", name="Opus 5.5", color="#4d8f2a", marker="D"),
]
SEEDS = (0, 1, 2)
NGDET_C = "#52514e"
SEED_LS = {0: "-", 1: (0, (5, 1.5)), 2: (0, (1, 1.2))}


def _transcript_tokens(d: Path):
    """Output tokens summed over a run's transcript turns (used when summary.json has no token block)."""
    t = d / "transcript.jsonl"
    if not t.exists():
        return None
    tot = 0
    for line in t.read_text().splitlines():
        if line.strip():
            tot += (json.loads(line).get("usage") or {}).get("output_tokens", 0) or 0
    return tot


def agent_study(R: Runs):
    """{agent key: {seed: dict(traj, best_iter, best_dev, oneshot_dev, cost, out_tokens, evals, raw, ngdet)}}"""
    logs = R.root / "agent_logs"
    study = {}
    for a in AGENTS:
        for s in SEEDS:
            d = logs / f"owlv2__kitti_dev__seed{s}__{a['key']}"
            if not (d / "summary.json").exists():
                continue
            sm = json.loads((d / "summary.json").read_text())
            bl = json.loads((d / "baselines.json").read_text())
            tok = sm.get("tokens")
            study.setdefault(a["key"], {})[s] = dict(
                traj=sm["trajectory"], best_iter=sm["best_iter"], best_dev=sm["best_dev_mAP"],
                oneshot_dev=sm["oneshot_dev_mAP"], cost=sm["cost_usd"], evals=sm["evaluations"],
                out_tokens=tok["output"] if tok else _transcript_tokens(d), tok_from_summary=bool(tok),
                raw=bl["raw_names"]["mAP"], ngdet=bl["ngdet_synonyms"]["mAP"])
    return study


def agent_test(R: Runs):
    """({arm: test mAP}, {arm: dict(delta, lo, hi)}) for the owlv2 prompt arms on kitti_test."""
    maps = {}
    for p in sorted(R.metrics.glob("owlv2__kitti_test__arm-*.json")):
        arm = p.name.split("__arm-")[1].split("__")[0]
        maps[arm] = json.loads(p.read_text())["coco"]["mAP"]
    pp = R.metrics / "owlv2__kitti_test__paired_vs_raw_names.json"
    paired = json.loads(pp.read_text()) if pp.exists() else {}
    return maps, paired


def fig_f7(R: Runs, out: Path, notes):
    study = agent_study(R)
    maps, paired = agent_test(R)
    if not study:
        print("  F7 skipped (no agent logs)")
        return
    raw = next(iter(next(iter(study.values())).values()))["raw"]
    ngd = next(iter(next(iter(study.values())).values()))["ngdet"]
    fig = plt.figure(figsize=(7.0, 6.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.5, wspace=0.07)
    axs = [fig.add_subplot(gs[0, 0])]
    axs += [fig.add_subplot(gs[0, i], sharey=axs[0]) for i in range(1, 4)]
    for ax, a in zip(axs, AGENTS):
        ax.axhline(raw, color=NGDET_C, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
        ax.axhline(ngd, color=NGDET_C, lw=0.9, ls=(0, (4, 2)), zorder=1)
        for s, r in sorted(study.get(a["key"], {}).items()):
            ys = r["traj"]
            ax.plot(range(len(ys)), ys, color=a["color"], lw=1.0, ls=SEED_LS[s], zorder=2)
            ax.scatter([r["best_iter"]], [ys[r["best_iter"]]], s=22, color=a["color"], marker=a["marker"],
                       edgecolor="white", linewidth=0.6, zorder=4)
        ax.set_title(a["name"], fontsize=8)
        ax.set_xlim(-0.5, 14.5)
        ax.set_xticks([0, 5, 10, 14])
        ax.tick_params(length=2.5, width=0.6)
        ax.grid(axis="x", visible=False)
        if ax is not axs[0]:
            plt.setp(ax.get_yticklabels(), visible=False)
    axs[0].set_ylabel("dev mAP@[.5:.95]\n(kitti_dev, 500 images)")
    ymin, ymax = axs[0].get_ylim()
    fig.canvas.draw()
    y0 = axs[0].get_position().y0
    fig.text(0.5, y0 - 0.045, "agent evaluation (0 = one-shot map)", ha="center", va="top", fontsize=8, color="#0b0b0b")
    fig.text(0.0, 0.93, "(a) Dev-set trajectory, 3 runs per agent model", fontsize=8.5, fontweight="bold", ha="left")
    # --- panel (b)
    ax = fig.add_subplot(gs[1, :])
    groups = [("ngdet\nsynonyms", None)] + [(a["name"], a) for a in AGENTS]
    pitch, w = 1.0, 0.12
    ticks = []
    for gi, (lab, a) in enumerate(groups):
        if a is None:
            q = paired.get("ngdet_synonyms")
            if q:
                ax.errorbar(gi, q["delta"], yerr=[[q["delta"] - q["lo"]], [q["hi"] - q["delta"]]], fmt="o", color=NGDET_C,
                            ms=4.5, capsize=2, elinewidth=1.0, zorder=3)
            ticks.append(gi)
            continue
        pos = 0
        for kind in ("best", "oneshot"):
            for s in SEEDS:
                q = paired.get(f"{kind}_s{s}_{a['key']}")
                x = gi + (pos - 2.5 - (0.25 if pos >= 3 else -0.25)) * w
                pos += 1
                if not q:
                    continue
                ax.errorbar(x, q["delta"], yerr=[[q["delta"] - q["lo"]], [q["hi"] - q["delta"]]], fmt=a["marker"],
                            color=a["color"], ms=4.6, capsize=1.8, elinewidth=1.0, capthick=0.8,
                            markerfacecolor=a["color"] if kind == "best" else "white", markeredgewidth=1.0, zorder=3)
        ticks.append(gi)
    ax.axhline(0, color="#0b0b0b", lw=0.9, zorder=1)
    ax.set_xticks(ticks, [g[0] for g in groups])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel("Δ test mAP vs raw names\n(paired 95% CI)")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.text(0.995, 0.97, f"raw names test mAP = {maps.get('raw_names', float('nan')):.3f}", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.5, color="#52514e")
    ax.set_title("(b) kitti_test, prompt map selected on dev, vs the raw class names", loc="left", x=-0.07, fontsize=8.5, pad=8)
    leg = [Line2D([], [], color=NGDET_C, lw=0.9, ls=(0, (1, 1.5)), label="raw names (dev)"),
           Line2D([], [], color=NGDET_C, lw=0.9, ls=(0, (4, 2)), label="ngdet synonyms (dev)"),
           Line2D([], [], color="#52514e", lw=1.0, ls=SEED_LS[0], label="seed 0"),
           Line2D([], [], color="#52514e", lw=1.0, ls=SEED_LS[1], label="seed 1"),
           Line2D([], [], color="#52514e", lw=1.0, ls=SEED_LS[2], label="seed 2"),
           Line2D([], [], marker="o", color="#52514e", ls="", ms=4.5, label="best map (filled)"),
           Line2D([], [], marker="o", color="#52514e", ls="", ms=4.5, markerfacecolor="white", label="one-shot map (hollow)")]
    fig.legend(handles=leg, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.045), columnspacing=1.0,
               handlelength=2.0, fontsize=6.6)
    footnote(fig, "(a) Larger marker = the evaluation the agent kept as its best map. (b) Within each model: best maps of seeds 0, 1, 2, then one-shot maps of "
             "seeds 0, 1, 2. Bars are paired bootstrap CIs (1000 image resamples of kitti_test) and reflect image sampling only, not run-to-run variance of the "
             "agent. 3 runs per model. Haiku 4.5 ran without the effort setting (not accepted); the other three ran at effort medium. " + CAV_KMAP, y=-0.045)
    save_fig(fig, out, "F7_agent")
    notes["F7_agent"] = ("Agent prompt-tuning study (OWLv2, 12 runs = 4 agent models x 3 seeds). (a) Dev mAP per evaluation, one small panel per model, "
                         "against the raw-names and ngdet-synonym dev baselines. (b) kitti_test mAP minus raw names for every selected map, with paired 95% bootstrap CIs. "
                         "Caveats: paired CIs cover image resampling only, not run-to-run variance of the agent; 3 runs per model; Haiku 4.5 runs without the "
                         "effort setting (it does not accept it) while the other three run at effort medium; Haiku 4.5's one-shot and best maps are selected on the "
                         "same 500 dev images that define the dev curve. " + CAV_KMAP)


# ---------------------------------------------------------------- failure analysis (F8, F9, F10, T8)
ERR = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Missed"]
ERR_LABEL = {"Cls": "Cls\n(wrong class)", "Loc": "Loc\n(poor box)", "Both": "Both\n(box + class)", "Dupe": "Dupe\n(duplicate)",
             "Bkg": "Bkg\n(background)", "Missed": "Missed\n(no detection)"}


def analysis(R: Runs, m):
    return R._load(R.root / "analysis" / f"{R.pred_name(m, 'kitti_test')}.json")


def fig_f8(R: Runs, out: Path, notes):
    ms = [m for m in MODELS if analysis(R, m)]
    if not ms:
        print("  F8 skipped (no runs/analysis)")
        return
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), gridspec_kw=dict(height_ratios=[1, 1], hspace=0.5))
    n = len(ms)
    w = 0.13
    off = (np.arange(n) - (n - 1) / 2) * w
    for ax, (getter, title, ylab) in zip(axes, (
            (lambda a, e: a["dAP50"][e], "(a) All five classes", "ΔAP50 if this error\nwere fixed"),
            (lambda a, e: a["dAP50_per_class"][e]["cyclist"], "(b) Cyclist only", "Cyclist ΔAP50"))):
        for i, m in enumerate(ms):
            a = analysis(R, m)
            vals = [getter(a, e) for e in ERR]
            ax.bar(np.arange(len(ERR)) + off[i], vals, width=w * 0.92, **bar_kw(m))
            if ax is axes[1]:
                for j, v in enumerate(vals):
                    if v >= 0.3:
                        ax.text(j + off[i], v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        ax.set_xticks(range(len(ERR)), [ERR_LABEL[e] for e in ERR])
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    axes[1].set_ylim(0, 0.42)
    a_owl = analysis(R, BY_KEY["owlv2"])
    axes[1].annotate("OWLv2 cyclist boxes land on the\nright object with the wrong extent",
                     xy=(1 + off[ms.index(BY_KEY["owlv2"])], a_owl["dAP50_per_class"]["Loc"]["cyclist"]),
                     xytext=(2.1, 0.30), fontsize=6.5, color="#0b0b0b", va="center",
                     arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.7))
    model_legend(fig, ms, ncol=6, y=0.02)
    footnote(fig, "TIDE-style oracle: each bar is AP50 gained (IoU 0.5, kitti_test, 3769 images) when only that error type is fixed, one oracle at a time; bigger = "
             "that error costs more. Loc = box overlaps a same-class object at IoU 0.1-0.5; Missed = ground truth never matched. Headline runs only "
             "(YOLO11m @1280 and YOLO26m (NMS) variants are in table T8). Closed-set models use the Cyclist merge; open-vocabulary models prompt Cyclist directly. "
             + CAV_AP.replace("KITTI-style AP", "AP50 here"), y=-0.06)
    save_fig(fig, out, "F8_failure_tide")
    notes["F8_failure_tide"] = ("TIDE-style error analysis on kitti_test at IoU 0.5: ΔAP50 from fixing one error type at a time (Cls, Loc, Both, Dupe, Bkg, Missed), "
                                "over all classes (a) and for Cyclist only (b). OWLv2's Cyclist Loc ΔAP50 is 0.341, the largest single bar: its cyclist boxes sit on the right "
                                "object with the wrong extent. Oracles are applied one at a time, so the bars do not sum to the AP gap. One image set, one run per model, "
                                "so no confidence intervals. Headline runs only; img1280 / YOLO26m (NMS) variants are in T8.")


def _recall_pool(a, cls, dim, key):
    b = a["recall100"][cls][dim][key]
    return b["recall"], b["n"]


def fig_f9(R: Runs, out: Path, notes):
    ms = [m for m in MODELS if analysis(R, m)]
    if not ms:
        print("  F9 skipped (no runs/analysis)")
        return
    occ = [("0", "0\nfully\nvisible"), ("1", "1\npartly\noccluded"), ("2", "2\nlargely\noccluded"), ("3", "3\nunknown")]
    hts = [("<25", "< 25 px"), ("25-40", "25-40 px"), ("40-80", "40-80 px"), (">=80", ">= 80 px")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True, gridspec_kw=dict(hspace=0.62, wspace=0.08))
    for r, (cls, cl) in enumerate((("all", "All classes"), ("cyclist", "Cyclist"))):
        for c, (dim, bins, t) in enumerate((("occlusion", occ, "KITTI occlusion level"), ("height", hts, "Box height"))):
            ax = axes[r, c]
            ref = analysis(R, ms[0])
            for m in ms:
                a = analysis(R, m)
                ys = [a["recall100"][cls][dim][k]["recall"] for k, _ in bins]
                ax.plot(range(4), [nz(y) for y in ys], color=m["color"], marker=m["marker"], ms=4.6, lw=1.2,
                        ls="--" if m["hatch"] else "-", zorder=3)
            labs = [f"{lab}\nn={ref['recall100'][cls][dim][k]['n']}" for k, lab in bins]
            ax.set_xticks(range(4), labs, fontsize=6.3)
            ax.set_xlim(-0.3, 3.3)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{cl}: by {t.lower()}", fontsize=7.8)
            ax.grid(axis="x", visible=False)
            ax.tick_params(length=2.5, width=0.6)
    for r in range(2):
        axes[r, 0].set_ylabel("recall@100 (IoU 0.5)")
    model_legend(fig, ms, ncol=6, kind="line", y=-0.035)
    footnote(fig, "Recall = share of ground-truth boxes matched at IoU 0.5 by a detection of the right class, any score, at most 100 detections per image and class. "
             "n = ground-truth boxes in the bin (identical for every model). Occlusion 3 is KITTI's 'unknown'. Dashed = second model of a family. Cyclist is the "
             "person+bicycle union box for closed-set models; the smallest cyclist height bin holds only 81 boxes, so that point is noisy. " + CAV_KMAP,
             y=-0.115, width=130)
    save_fig(fig, out, "F9_recall_bins")
    notes["F9_recall_bins"] = ("Recall@100 at IoU 0.5 by KITTI occlusion level (left) and by box height (right), pooled over all five classes (top) and for Cyclist only "
                               "(bottom), per model. Ground-truth counts are shown under each bin; small bins (cyclist < 25 px: 81 boxes) are noisy and there are "
                               "no confidence intervals on these points. Single run per model.")


GAL = Path("analysis") / "gallery"


def fig_f10(R: Runs, out: Path, notes):
    from PIL import Image
    an = R.root / "analysis"
    picks = [
        ("owlv2__kitti_test__1.jpg", "(a) OWLv2, image 2666: 9 ground-truth objects >= 40 px missed (mostly pedestrians, tagged MISS) and one confident unmatched box (red, truck 0.59)"),
        ("shared_miss__2.jpg", "(b) Faster R-CNN v2, same image 2666: 5 ground-truth objects >= 40 px that every closed-set model misses (magenta)"),
        ("shared_miss__1.jpg", "(c) Faster R-CNN v2, image 4343: 6 ground-truth objects >= 40 px missed by every closed-set model (trams, a pedestrian); a confident cyclist box (red) does not match the annotated one"),
    ]
    picks = [(f, c) for f, c in picks if (an / "gallery" / f).exists()]
    if not picks:
        print("  F10 skipped (no gallery)")
        return
    import textwrap
    imh, tth = 7.0 * 370 / 1224, 0.58  # image height at full width, title band (inches)
    H = len(picks) * (imh + tth)
    fig = plt.figure(figsize=(7.0, H))
    for i, (f, cap) in enumerate(picks):
        top = 1 - i * (imh + tth) / H
        ax = fig.add_axes([0, top - (imh + tth) / H, 1, imh / H])
        ax.imshow(Image.open(an / "gallery" / f), aspect="auto")
        ax.axis("off")
        fig.text(0.0, top - 0.06 / H, "\n".join(textwrap.wrap(cap, 118)), fontsize=6.6, va="top", ha="left")
    footnote(fig, "Green = ground truth (thick, tagged MISS when no detection matched), orange = detections with score >= 0.3 that matched, red = unmatched (false "
             "positive), magenta = missed ground truth in the shared-miss images. Images come from runs/analysis/gallery, chosen by most high-confidence errors "
             "(a) and ground truth >= 40 px missed by every closed-set model (b, c). No OWLv2 cyclist-localisation image appears among the auto-selected "
             "gallery images, so the cyclist Loc finding is supported by F8, not by a picture.", y=-0.005, width=135)
    save_fig(fig, out, "F10_gallery", dpi=125, pdf=False)
    notes["F10_gallery"] = ("Failure gallery on kitti_test: (a) OWLv2 on image 2666, where a crowd of pedestrians is missed; (b) the same image seen by the closed-set "
                            "shared-miss set (detections from Faster R-CNN v2); (c) image 4343 with large objects missed by all closed-set models. Illustrative single "
                            "images selected by an automatic ranking, not representative samples.")


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
             CAV_SUBSET + " "
             "OWLv2's ensemble checkpoint was fine-tuned on LVIS, whose images are COCO images, and Grounding DINO's GoldG grounding data includes Visual Genome, whose images come from COCO, so neither model's COCO score is strictly zero-shot; KITTI is.",
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
        disp = next((x["name"] for x in MODELS + EXTRA if x["model"] == parts[0] and x["tag"] == tag), None) or next((x["name"] for x in MODELS if x["model"] == parts[0]), parts[0])
        rows.append([disp, split, tag, m["n_images"], c.get("device", ""), m.get("env", {}).get("host", ""), v.get("torch", ""),
                     v.get("torchvision", ""), v.get("ultralytics", ""), v.get("transformers", ""), m.get("command", ""),
                     (f"{m['wall_s']:.1f}" if m.get('wall_s') is not None else ""), m.get("src_hash", "")])
    order = {x["model"]: i for i, x in enumerate(MODELS)}
    rows.sort(key=lambda r: (order.get(next((x["model"] for x in MODELS if x["name"] == r[0]), ""), 99), r[1], r[2]))
    notes = ["One row per prediction run in runs/preds (dev-VM sanity runs and the MPS/CPU parity checks are excluded). Every run is deterministic given the command: fixed seeds for subsets, batch size 1.",
             "Wall time covers model load, inference and writing predictions; scoring is separate (python -m hw1.score).",
             CAV_SUBSET + " " + CAV_CPU,
             "Full command history: runs/COMMANDS.md."]
    write_table(out, "T5_reproducibility", h, rows, notes=notes)


# ---------------------------------------------------------------- new tables (T6, T7, T8)
def table_t6(R: Runs, out: Path):
    study = agent_study(R)
    maps, paired = agent_test(R)
    if not study:
        print("  T6 skipped (no agent logs)")
        return
    h = ["Agent model", "runs", "evaluations (total)", "cost per run USD (mean ± sd)", "output tokens per run (mean)",
         "best dev mAP (mean)", "best map test Δ vs raw (mean [min, max])", "best maps significant (of 3)",
         "one-shot test Δ (mean [min, max])", "mean |dev step|", "worst single-step dev drop"]
    rows = []
    for a in AGENTS:
        rs = study.get(a["key"], {})
        if not rs:
            continue
        cost = np.array([r["cost"] for r in rs.values()])
        toks = [r["out_tokens"] for r in rs.values() if r["out_tokens"] is not None]
        steps = [np.diff(r["traj"]) for r in rs.values()]
        allsteps = np.concatenate(steps)
        bd = [paired.get(f"best_s{s}_{a['key']}") for s in rs]
        od = [paired.get(f"oneshot_s{s}_{a['key']}") for s in rs]
        bd, od = [q for q in bd if q], [q for q in od if q]
        sig = sum(1 for q in bd if q["lo"] > 0 or q["hi"] < 0)
        fmt = lambda qs: (f"{np.mean([q['delta'] for q in qs]):+.4f} [{min(q['delta'] for q in qs):+.4f}, {max(q['delta'] for q in qs):+.4f}]"
                          if qs else "")
        rows.append([a["name"] + ("*" if a["key"] == "haiku-4-5" else ""), len(rs), sum(r["evals"] for r in rs.values()),
                     f"{cost.mean():.3f} ± {cost.std(ddof=1):.3f}", f"{np.mean(toks):.0f}" if toks else "",
                     f"{np.mean([r['best_dev'] for r in rs.values()]):.4f}", fmt(bd), f"{sig} of {len(bd)}", fmt(od),
                     f"{np.mean(np.abs(allsteps)):.4f}", f"{allsteps.min():+.4f}"])
    p = PRICES_NOTE()
    notes = [f"Dev baselines (kitti_dev, 500 images): raw names {next(iter(next(iter(study.values())).values()))['raw']:.4f}, ngdet synonyms "
             f"{next(iter(next(iter(study.values())).values()))['ngdet']:.4f}. Test mAP of raw names {maps.get('raw_names', float('nan')):.4f}. "
             f"ngdet synonyms test Δ vs raw names: {paired['ngdet_synonyms']['delta']:+.4f} [{paired['ngdet_synonyms']['lo']:+.4f}, {paired['ngdet_synonyms']['hi']:+.4f}].",
             "Test Δ = kitti_test mAP of the selected prompt map minus raw names (OWLv2). 'Significant' = the paired-bootstrap 95% CI (1000 image resamples) excludes 0; "
             "it covers image resampling only, not run-to-run variance of the agent. 3 runs per model, so mean [min, max] over 3 seeds.",
             "Evaluations: dev evaluations the agent used (budget 15; one Haiku 5.5 run stopped at 14). Dev step = change in dev mAP between consecutive evaluations within a run; "
             "the 'worst drop' is the most negative single step over all runs of that model.",
             "Cost per run = (input x p_in + output x p_out + cache_read x p_read + cache_write x p_write) / 1e6 from the usage the API reported, with USD per million "
             "tokens from PRICES in src/hw1/agent.py: " + p + ". Cost covers agent tokens only, not OWLv2 compute.",
             "* Haiku 4.5 ran without the effort setting (it does not accept it); the others ran at effort medium. Its summary.json has no token block, so its output tokens "
             "are summed from the per-turn usage in transcript.jsonl."]
    write_table(out, "T6_agent_study", h, rows, notes=notes)


def PRICES_NOTE():
    src = (ROOT / "src" / "hw1" / "agent.py").read_text()
    blk = re.search(r"PRICES = \{(.*?)\n\}", src, re.S).group(1)
    items = re.findall(r'"(claude-[\w-]+)":\s*\(([^)]*)\)', blk)
    return "; ".join(f"{k}: in/out/read/write = " + "/".join(x.strip() for x in v.split(",")) for k, v in items)


def table_t7(R: Runs, out: Path):
    keys = [("mAP", "mAP"), ("AP50", "AP50"), ("AP_small", "APs"), ("AP_medium", "APm"), ("AP_large", "APl")]
    h = ["Model", "1k: n images"] + [f"1k {lab}" for _, lab in keys] + ["5000: mAP", "5000: AP50", "Δ mAP (1k - 5000)"]
    rows = []
    for m in MODELS:
        if m.get("open_vocab"):
            d1 = R._load(R.metrics / f"{m['model']}__coco_val1k__default.json")
            d5 = None
        else:
            d1 = R._load(R.metrics / f"{m['model']}__coco_val__default__sub-coco_val1k.json")
            d5 = R.coco(m)
        if d1 is None:
            continue
        r = [m["name"], d1["n_images"]] + [f3(cm(d1, k)) for k, _ in keys]
        r += [f3(cm(d5, "mAP")) if d5 else "n/a", f3(cm(d5, "AP50")) if d5 else "n/a",
              f"{cm(d1, 'mAP') - cm(d5, 'mAP'):+.3f}" if d5 else "n/a"]
        rows.append(r)
    notes = ["All six models scored on the same seeded 1000-image COCO val2017 subset (the closed-set models were re-scored on it from their full-val predictions). "
             "5000 = the full val2017 numbers of T1; OWLv2 and Grounding DINO-T were run only on the 1k subset.",
             "Δ mAP shows how much the 1k subset shifts each closed-set model's mAP from the full set; compare models within the 1k columns, not across 1k and 5000.",
             "OWLv2's ensemble checkpoint was fine-tuned on LVIS, whose images are COCO images, and Grounding DINO's GoldG grounding data includes Visual Genome, whose images come from COCO, so neither model's COCO score is strictly zero-shot; KITTI is. COCO mAP is over 80 classes; "
             "one subset, one run each, no confidence intervals."]
    write_table(out, "T7_coco_same_subset", h, rows, notes=notes)


def table_t8(R: Runs, out: Path):
    h = ["Model", "AP50"] + [f"{e} ΔAP50" for e in ERR] + ["Cyclist AP50"] + [f"Cyclist {e} ΔAP50" for e in ERR]
    rows = []
    for m in TABLE_MODELS:
        a = analysis(R, m)
        if a is None:
            continue
        rows.append([m["name"], f3(a["AP50"])] + [f3(a["dAP50"][e]) for e in ERR] + [f3(a["AP50_per_class"]["cyclist"])]
                    + [f3(a["dAP50_per_class"][e]["cyclist"]) for e in ERR])
    notes = ["TIDE-style oracles at IoU 0.5 on kitti_test (3769 images): ΔAP50 = AP50 after fixing only that error type minus AP50 (definitions in the docstring of "
             "src/hw1/failure.py). Overall = mean over the five KITTI classes. Oracles are applied one at a time, so columns do not add up to the AP gap.",
             "Cls = box matches an object of another class; Loc = poor localisation of the right class (IoU 0.1-0.5); Both = poor box and wrong class; Dupe = duplicate; "
             "Bkg = background; Missed = ground truth never detected.",
             "OWLv2 Cyclist Loc ΔAP50 = 0.341: its cyclist boxes sit on the right object with the wrong extent.",
             "One run per model; no confidence intervals. YOLO11m @1280 and YOLO26m (NMS) are the extra configurations of T1. " + CAV_AP]
    write_table(out, "T8_tide", h, rows, notes=notes)


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
    for k, fn in (("F1", fig_f1), ("F2", fig_f2), ("F3", fig_f3), ("F4", fig_f4), ("F5", fig_f5), ("F6", fig_f6), ("F7", fig_f7),
                   ("F8", fig_f8), ("F9", fig_f9), ("F10", fig_f10)):
        if want(k):
            fn(R, figs, notes)
    for k, fn in (("T1", table_t1), ("T2", table_t2), ("T3", table_t3), ("T4", table_t4), ("T5", table_t5), ("T6", table_t6),
                   ("T7", table_t7), ("T8", table_t8)):
        if want(k):
            fn(R, tabs)
    if not only:
        write_captions(figs, notes)


if __name__ == "__main__":
    main()
