"""Datasets as COCO-format ground truth plus an image iterator.

Splits:
  coco_val       COCO val2017, all 5000 images
  coco_val1k     seeded 1000-image subset of coco_val (for the slow open-vocab models)
  kitti_test     KITTI 3DOP val split, 3769 images (held out; scored once per arm)
  kitti_dev      seeded 500 images from the KITTI 3DOP train split (the only split the agent sees)
  kitti_dev50 / kitti_test200 / coco_val200   small seeded subsets for smoke tests and timing
"""
from __future__ import annotations

import copy
import json
import random
import struct
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .taxonomy import KITTI_CLASSES, GTRules

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
COCO_DIR = DATA / "coco"
KITTI_DIR = DATA / "kitti"
GT_DIR = DATA / "gt"
SEED = 249

# KITTI devkit difficulty: min box height (px), max occlusion level, max truncation
DIFFICULTY = {"easy": (40, 0, 0.15), "moderate": (25, 1, 0.30), "hard": (25, 2, 0.50)}


def _png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def _subset(ids: list, n: int, seed: int = SEED) -> list:
    ids = sorted(ids)
    return sorted(random.Random(seed).sample(ids, n))


def kitti_split_ids(split: str) -> list[str]:
    def read(name):
        return [l.strip() for l in (KITTI_DIR / "ImageSets" / f"{name}.txt").read_text().split() if l.strip()]

    if split == "kitti_test":
        return read("val")
    if split == "kitti_dev":
        return _subset(read("train"), 500)
    if split == "kitti_dev50":
        return _subset(kitti_split_ids("kitti_dev"), 50)
    if split == "kitti_test200":
        return _subset(read("val"), 200)
    raise KeyError(split)


def parse_kitti_label(path: Path) -> list[dict]:
    objs = []
    for line in path.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        x1, y1, x2, y2 = map(float, p[4:8])
        objs.append({"type": p[0], "truncated": float(p[1]), "occluded": int(p[2]), "bbox": [x1, y1, x2, y2]})
    return objs


def build_kitti_gt(split: str, rules: GTRules = GTRules()) -> dict:
    """COCO-format GT. Ignore regions are iscrowd=1 annotations, one per class they apply to."""
    images, anns, missing = [], [], 0
    for name in kitti_split_ids(split):
        img_path = KITTI_DIR / "training" / "image_2" / f"{name}.png"
        if img_path.exists():
            w, h = _png_size(img_path)
        else:
            w, h, missing = 1242, 375, missing + 1
        iid = int(name)
        images.append({"id": iid, "file_name": f"training/image_2/{name}.png", "width": w, "height": h})
        for o in parse_kitti_label(KITTI_DIR / "training" / "label_2" / f"{name}.txt"):
            x1, y1, x2, y2 = o["bbox"]
            box = [x1, y1, x2 - x1, y2 - y1]
            gt_for, ignore_for = rules.targets(o["type"])
            for cid, crowd in [(c, 0) for c in gt_for] + [(c, 1) for c in ignore_for]:
                anns.append({
                    "id": len(anns) + 1, "image_id": iid, "category_id": cid, "bbox": box,
                    "area": box[2] * box[3], "iscrowd": crowd, "kitti_type": o["type"],
                    "height_px": y2 - y1, "truncated": o["truncated"], "occluded": o["occluded"],
                })
    return {
        "info": {"split": split, "gt_rules": asdict(rules), "missing_images": missing},
        "images": images,
        "annotations": anns,
        "categories": [{"id": k, "name": v} for k, v in KITTI_CLASSES.items()],
    }


def difficulty_view(gt: dict, level: str) -> dict:
    """KITTI-devkit-style view: GT outside the difficulty becomes an ignore region (not removed)."""
    min_h, max_occ, max_trunc = DIFFICULTY[level]
    out = copy.deepcopy(gt)
    for a in out["annotations"]:
        if a["iscrowd"] == 0 and (a["height_px"] < min_h or a["occluded"] > max_occ or a["truncated"] > max_trunc):
            a["iscrowd"] = 1
    out["info"]["difficulty"] = level
    return out


def coco_gt(split: str) -> dict:
    gt = json.loads((COCO_DIR / "annotations" / "instances_val2017.json").read_text())
    if split == "coco_val":
        return gt
    n = {"coco_val1k": 1000, "coco_val200": 200}[split]
    keep = set(_subset([i["id"] for i in gt["images"]], n))
    gt["images"] = [i for i in gt["images"] if i["id"] in keep]
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in keep]
    return gt


def load_gt(split: str, rules: GTRules = GTRules(), cache: bool = True) -> dict:
    if split.startswith("coco"):
        return coco_gt(split)
    tag = f"{split}__van-{rules.van}__cyc{int(rules.cyclist_ignore_for_ped)}__dc{int(rules.dontcare_ignore)}"
    path = GT_DIR / f"{tag}.json"
    if cache and path.exists():
        return json.loads(path.read_text())
    gt = build_kitti_gt(split, rules)
    if cache and gt["info"]["missing_images"] == 0:
        GT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(gt))
    return gt


def image_root(split: str) -> Path:
    return COCO_DIR / "val2017" if split.startswith("coco") else KITTI_DIR


def iter_images(gt: dict, split: str):
    """Yield (image_id, HxWx3 uint8 RGB array)."""
    root = image_root(split)
    for im in gt["images"]:
        with Image.open(root / im["file_name"]) as f:
            yield im["id"], np.asarray(f.convert("RGB"))


def gt_summary(gt: dict) -> dict:
    out = {"images": len(gt["images"])}
    for c in gt["categories"]:
        n = sum(1 for a in gt["annotations"] if a["category_id"] == c["id"] and not a.get("iscrowd", 0))
        out[c["name"]] = n
    out["ignore_regions"] = sum(1 for a in gt["annotations"] if a.get("iscrowd", 0))
    return out
