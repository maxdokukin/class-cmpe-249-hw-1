"""Write report/appendix_tables.typ from runs/preds/*.meta.json (latency, memory, commands per main run)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"yolo11m": "YOLO11m", "yolo26m": "YOLO26m", "frcnn_v2": "Faster R-CNN v2", "rtdetr_r50": "RT-DETR R50",
         "owlv2": "OWLv2", "gdino_t": "Grounding DINO-T"}
SPLITS = {"coco_val", "coco_val1k", "kitti_test"}


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("#", "\\#").replace("[", "\\[").replace("]", "\\]").replace("$", "\\$")


rows = []
for f in sorted((ROOT / "runs/preds").glob("*.meta.json")):
    m = json.loads(f.read_text())
    if m["split"] not in SPLITS or "__arm-" in m["name"] or "parity" in m["name"]:
        continue
    lat, mem = m["latency"], m["memory"]
    peak = mem.get("peak_driver_allocated_mb", mem.get("peak_mb"))
    tag = m["name"].split("__")[2] if m["name"].count("__") >= 2 else ""
    label = NAMES.get(m["model"], m["model"])
    if m["model"] == "yolo26m":
        label += " (NMS)" if tag == "nms" else " (NMS-free)"
    elif tag:
        label += f" ({tag})"
    rows.append((label, m["split"], m["n_images"],
                 m["detector_config"]["device"], lat.get("burst", {}).get("median_ms", 0), lat.get("sustained", {}).get("median_ms", 0),
                 peak, round(m["wall_s"] / 60, 1), m["command"]))

out = ["#figure(table(columns: 8, align: (left, left, center, center, center, center, center, center),",
       "  [*Run*], [*Split*], [*Images*], [*Device*], [*Burst ms*], [*Sustained ms*], [*Peak MB*], [*Wall min*],"]
for r in rows:
    out.append(f"  [{esc(r[0])}], [`{r[1]}`], [{r[2]}], [{r[3].upper()}], [{r[4]:.1f}], [{r[5]:.1f}], [{r[6]:.0f}], [{r[7]}],")
out.append("), caption: [Per-run latency, memory and wall time (batch 1, after 10 warm-up images). Peak memory is the MPS "
           "unified-memory allocator peak, or process RSS for CPU runs.]) <tab:runs>")
out.append("")
out.append("#figure(table(columns: 2, align: (left, left),")
out.append("  [*Run*], [*Exact command (from the run's metadata; also in `runs/COMMANDS.md`)*],")
for r in rows:
    out.append(f"  [{esc(r[0])}, `{r[1]}`], [`{r[8]}`],")
out.append("), caption: [Commands for every main run. Each also records checkpoint, input rule, thresholds, library "
           "versions and a source hash in `runs/preds/<run>.meta.json`.]) <tab:cmds>")
(ROOT / "report/appendix_tables.typ").write_text("\n".join(out) + "\n")
print(f"{len(rows)} runs")
