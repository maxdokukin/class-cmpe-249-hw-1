"""MPS vs CPU parity on the same 20 COCO images: mAP must agree within 0.01.  python scripts/parity.py yolo11m"""
import subprocess
import sys
from pathlib import Path

from hw1.score import score

model = sys.argv[1]
maps = {}
for dev in ("cpu", "mps"):
    pred = Path(f"runs/preds/{model}__coco_val200__parity_{dev}__n20.jsonl.gz")
    meta = pred.with_name(pred.name.replace(".jsonl.gz", ".meta.json"))
    if not (pred.exists() and meta.exists()):  # meta is written last, so a missing one means an interrupted run
        subprocess.run([sys.executable, "-m", "hw1.run", "--model", model, "--split", "coco_val200", "--max-images", "20",
                        "--device", dev, "--warmup", "2", "--tag", f"parity_{dev}"], check=True)
    maps[dev] = score(pred)["coco"]["mAP"]
d = abs(maps["cpu"] - maps["mps"])
print(f"{model}: mAP cpu {maps['cpu']:.4f}  mps {maps['mps']:.4f}  |diff| {d:.4f} -> {'PARITY OK' if d < 0.01 else 'PARITY FAIL'}")
