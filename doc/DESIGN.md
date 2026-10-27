# HW1 design: zero-shot detection under domain shift, with a prompt-tuning agent

Status: **draft for approval** (2026-10-27). Built from `research/01–04`. Nothing is installed or downloaded yet.

Question the report answers: *How much accuracy do COCO-pretrained detectors lose on KITTI, where does the loss come
from (classes, object scale, taxonomy), and can an LLM agent recover the part caused by vocabulary mismatch, without
touching any weights?*

## 1. Models (6 configs, 5 families)

| # | Config | Family | Source | Why it's here |
|---|---|---|---|---|
| M1 | YOLO11m | one-stage CNN, anchor-free, NMS | `ultralytics` `yolo11m.pt` | modern real-time baseline |
| M2 | YOLO26m | one-stage CNN, NMS-free head | `ultralytics` `yolo26m.pt` | NMS vs NMS-free within one family |
| M3 | Faster R-CNN R50-FPN v2 | two-stage, anchor-based | `torchvision` `COCO_V1` | classic reference |
| M4 | RT-DETR R50 | DETR, query-based, NMS-free | HF `PekingU/rtdetr_r50vd` | transformer vs CNN |
| M5 | OWLv2-B/16 | open-vocab, ViT | HF `google/owlv2-base-patch16-ensemble` | agent target (prompt-independent image features → cheap loop) |
| M6 | Grounding DINO-T | open-vocab, text-fused DETR | HF `IDEA-Research/grounding-dino-tiny` | second agent target; **not** `-base` (it saw COCO in training) |

Fallbacks if an MPS smoke test fails: D-FINE-M or Ultralytics `rtdetr-l.pt` for M4; YOLO-World v2 for M6.
Stretch only: FCOS (anchor-free dense), x-size YOLOs.

## 2. Data and splits

| Domain | Set | Use | Size |
|---|---|---|---|
| Source | COCO val2017 | in-domain reference. All 5000 images for M1–M4; a fixed seeded 1000-image subset for M5–M6 | 1.1 GB |
| Target | KITTI 2D, **val-3769** (3DOP split, sequence-disjoint from train-3712) | `kitti_test`: held-out, scored once per arm | ~6 GB of PNGs out of a 12.6 GB zip |
| Target | KITTI **train-3712**, seeded 500-image subset | `kitti_dev`: the only split the agent sees | in the same zip |
| Stretch | nuImages (mini for the pipeline, val if time allows) | second shift domain (Boston/Singapore, night/rain) | 118 MB / 16 GB |

The 3DOP split keeps sequences disjoint, so dev→test leakage through neighbouring frames can't happen.

**KITTI → COCO mapping** (full table in `research/02 §3.1`):
- Car, Van → car. Truck → truck. Pedestrian → person.
- Tram → train, reported separately.
- Person_sitting and Misc → ignore regions.
- DontCare → `iscrowd=1` for each evaluated class.
- Cyclist: a COCO model sees a rider and a bike as two objects, so it gets three treatments:
  - (a) naive bicycle→cyclist (ngdet's approach, expected AP ≈ 0);
  - (b) our merge of person+bicycle pairs into one union box;
  - (c) Cyclist boxes as ignore regions for `person`.
- Each choice is ablated once on one model, and the AP difference is reported.

## 3. Evaluation protocol (same for every model)

- **Score threshold 0.001** (0.0 for DETR-style models) and max 100 detections. Frameworks default to 0.2–0.5 and
  ngdet to 0.3, which cuts off the PR curve and deflates mAP. We show this effect once, as a finding.
- NMS as each model ships it, logged: Ultralytics IoU 0.7, torchvision 0.5, none for DETR/YOLO26.
- Resolution: 640 for YOLO/RT-DETR, the torchvision default 800/1333, and the HF processor defaults for M5/M6. One ablation
  runs at 1280 on KITTI, because letterboxing a 1242×375 frame to 640 shrinks small objects.
- Metrics:
  - scored with pycocotools: mAP@[.5:.95], AP50/75, AP S/M/L, per-class AP, AR@1/10/100;
  - a KITTI-style view: Car AP@0.7 and Ped/Cyc AP@0.5, with the easy/moderate/hard filter implemented as ignore regions
    (`research/02 §3.2`). It is labelled as not comparable to the leaderboard;
  - TIDE error decomposition for the failure analysis;
  - bootstrap 95% CIs on `kitti_test`.
- Speed: batch 1, fixed warmup, `torch.mps.synchronize()`. Burst latency is the median of 200 images; sustained latency is a 10-min run.
  Pre- and post-processing are included and also reported separately. Peak `torch.mps.driver_allocated_memory()` is sampled
  per image and labelled as unified memory. A CPU latency column comes from the VM.
- Every result JSON records the model and checkpoint hash, dataset and split, resolution, thresholds, device, library versions, git SHA
  and the exact command. Every command is appended to `runs/COMMANDS.md`.

## 4. Experiments → rubric analyses

| ID | Experiment | Rubric analysis |
|---|---|---|
| E1 | M1–M6 on COCO and KITTI: per-class and per-scale drop | cross-dataset generalization (required) |
| E2 | YOLO11 vs YOLO26 vs RT-DETR vs Faster R-CNN: NMS vs NMS-free, one-stage vs two-stage; duplicate and localization errors from TIDE | architecture comparison |
| E3 | latency, memory and mAP Pareto plot (M4 GPU, plus CPU) | speed–accuracy |
| E4 | **Agent experiment (core):** five arms on OWLv2 and GDINO-T, evaluated once on `kitti_test`: (1) raw KITTI names, (2) ngdet's hand-written synonyms (human baseline), (3) one-shot LLM with no feedback, (4) feedback agent: propose prompt map → `evaluate(split="dev")` returns per-class AP, confusions and failure crops → revise, 15 iterations, 3 seeds; (5) closed-set M1–M4 as the reference | closed-set vs open-vocabulary, plus the agent angle |
| E5 | failure analysis: TIDE, the Cyclist/Van/Tram case study, a gallery of the worst images | failure analysis (required section) |
| S1 | *stretch:* detector + VLM verifier cascade (Qwen2.5-VL-3B on MLX re-scores uncertain boxes) | adds a speed–accuracy curve |
| S2 | *stretch:* nuImages as a third domain | stronger generalization claim |

Agent rules, enforced in code:
- The `evaluate` tool is wired to `kitti_dev` only, and the agent can't call it on test.
- Prompts are capped at 5 per class.
- Every prompt and tool call is logged to `runs/agent_logs/`.
- The selected prompt map is the best on dev; it is run once on test.
- The dev→test drop is reported.

The agent runs on the Claude API (Sonnet 5.5), at an estimated cost under $5.

## 5. Code and harness

A thin harness of our own in `hw_1/src/hw1/`: one adapter per framework (ultralytics, torchvision, HF), a COCO-format writer,
our taxonomy and evaluator, latency, and the agent.
- **Why not run ngdet directly?** It has the four protocol weaknesses above: the 0.3 threshold, DontCare dropped,
  bicycle→cyclist, and a 120-image evaluation.
- We cite it as the course's reference harness, borrow its adapter structure and synonym list, and run ngdet once on
  one model as a cross-check, which shows the effect of the threshold.

Environment: `uv`, Python 3.12, torch 2.14 / torchvision 0.29 (as research/04 recommends; to be confirmed
when installing), one `uv.lock` for macOS arm64 and Linux CPU. Code is written and smoke-tested on the VM (CPU, 20 images), pushed to
GitHub, pulled on the M4 for the real runs, and the results are committed back.

## 6. Schedule (due Sat 2026-10-31; stretch goals S1/S2 cut unless ahead of plan)

| Day | Work | Where |
|---|---|---|
| Tue 10-27 | env (done), data download and conversion, KITTI→COCO with ignore regions, evaluator tested against a toy set, model adapters | VM |
| Wed 10-28 | 20-image smoke tests, MPS-vs-CPU parity, 50-image timing; E1/E3 full runs overnight with cooldowns | VM → M4 |
| Thu 10-29 | agent loop (E4): OWLv2 with cached features, then GDINO-T; threshold/Cyclist/1280 ablations | M4 + API |
| Fri 10-30 | E2/E5 analysis, figures, report draft | VM |
| Sat 10-31 | finish the report, export the PDF, package code and command log, submit | — |

## 7. Decisions (answered 2026-10-27)

1. Due **Saturday 2026-10-31**.
2. A Claude API key will be provided. It is used only by the E4 agent loop (the LLM that proposes and revises prompts), < $5.
3. M4: **24 GB RAM, 200 GB SSD**, enough for everything including a local VLM.
4. **No repo for now**: work in `hw_1/` here; code moves to the M4 by copy (rsync/scp) when the runs start.
5. **Own harness**, with ngdet cited and cross-checked once.

## 8. Caveats found during setup (2026-10-27)

- **COCO is not a clean zero-shot domain for the open-vocab models.** OWLv2 (ensemble checkpoint) and Grounding DINO-T
  trained on Visual Genome / GoldG data. Visual Genome reuses many COCO images, so their COCO mAP is in-domain-ish;
  KITTI is the clean test for them. On 30 shared COCO images, OWLv2 0.595 ≈ YOLO11m 0.575 ≈ GDINO-T 0.574. *(To verify
  against the papers' data sections before writing the claim.)*
- Small COCO subsets are easy: every model scores 3–9 points above its published full-val AP on coco_val200 and
  its first 60 images. Only same-image comparisons or the full 5000-image runs are used for claims.
- Grounding DINO uses custom per-prompt scoring (max token probability per prompt), because HF post-processing returns
  empty phrase labels at low thresholds. It matches YOLO11m / RT-DETR on the same 60 COCO images (0.602 vs 0.605 / 0.609).
- Threshold effect, measured: a 0.3 score cutoff (ngdet's default) takes YOLO11m from 0.547 to 0.490 mAP on coco_val200.
