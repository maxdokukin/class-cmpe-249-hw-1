# HW1 plan: Track A, 2D detection evaluation under domain shift

Status: draft v0 (2026-10-27). Research phase in progress; specifics get locked in at Gate 1.

## Goal

Design, run and critically analyze a reproducible object-detection experiment:
COCO-pretrained detectors, evaluated zero-shot on autonomous-driving data, with an
agent / vision-language angle.

Deliverable: a 6–10 page PDF report plus a public repo with code, configs and the command log.

## Working hypothesis (to confirm after research)

| Axis | Candidate choice | Why |
|---|---|---|
| Families (≥2, aim for 4) | CNN one-stage (YOLO11 / YOLOv8), transformer query-based (RT-DETR or D-FINE), two-stage (Faster R-CNN, torchvision), open-vocabulary (Grounding DINO / YOLO-World / OWLv2) | covers closed vs open set, NMS vs NMS-free, anchor vs anchor-free |
| Configs (≥3) | ~5–6 total, including 2 sizes of one family for speed/accuracy | grading counts families, not sizes |
| Datasets (≥2) | COCO val2017 (source domain), KITTI 2D (target), optionally nuImages or BDD100K | domain shift, ties to lectures 11–12 (KITTI geometry) |
| Analyses (≥2, aim for 3) | cross-dataset generalization (per-class and per-scale drop); closed-set vs open-vocabulary; speed–accuracy | optional 4th: small fine-tune on KITTI |
| Agent angle | open-vocab / VLM detector driven by prompts; an LLM agent that writes prompts and taxonomy mappings, or does failure triage | the user is interested in AI agents; it must stay a measured experiment |

## Compute (updated 2026-10-27)

- **Primary: MacBook Air M4** (PyTorch MPS, unified memory, fanless, so it throttles). "GPU memory" =
  `torch.mps.driver_allocated_memory`, with the unified-memory caveat stated. Fixed warmup, a cooldown between runs.
- Dev VM (this box): 20 vCPU, 16 GB RAM, no GPU. Code, data conversion, CPU smoke tests.
- Mac Pro 2013 (2× D700): likely CPU-only (no CUDA; x86-macOS torch wheels stop at 2.2). Pending verdict in research/04.
- SJSU HPC: long queue, fallback only.
- Deadline is soon, so evaluate on fixed subsets (e.g. 500 COCO val + 500 KITTI val images, seeded) and
  scale up only if time allows. Subset sizes are reported in the report.

## Instructor tutorial: ngdet (key finding)

https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md

The instructor's `ngdet` already implements this exact experiment: `run_eval.py` (multi-model × multi-dataset),
`taxonomy.py` (`driving3`/`driving5` presets), `evaluator.py` (`COCOUnifiedEvaluator`), `latency.py`, `report.py`,
with backends for torchvision (Faster R-CNN/RetinaNet/FCOS), HF DETR/RT-DETR, Ultralytics YOLO, Grounding DINO/OWLv2,
and VLMs (Qwen2.5-VL, LocateAnything). It supports COCO, KITTI, nuImages and Waymo.
Reference results (RTX 3090): COCO 0.51–0.61 mAP, KITTI 0.19–0.28, nuImages 0.24–0.31; open-vocab leads
on nuImages after prompt tuning; Cyclist AP ≈ 0 for COCO models.

Implication: **use ngdet as the harness** (saves days; it's the expected path), run it ourselves on the M4
(`--device mps`, to be verified), and put our own contribution on top:
1. our runs, logs and failure analysis (not copied reference numbers);
2. the **agent experiment**: an LLM agent that tunes open-vocab prompts and thresholds on a KITTI *tune* split,
   evaluated on a disjoint *test* split. This directly targets ngdet's "prompt sensitivity" caveat, and/or a VLM verifier cascade;
3. a check of the Cyclist taxonomy fix (person+bicycle box merge) as a failure-analysis item.

## Phases

0. **Research (now).** Parallel subagents write findings to `doc/research/`:
   models & toolchains, datasets & taxonomy, agent/VLM angle, compute (HPC + CPU feasibility).
1. **Gate 1: lock the design.** Pick the models, datasets, splits and analyses. Write `doc/DESIGN.md`. Agree with the user.
2. **Environment.** `uv`/conda env, pinned versions; `env/` folder; smoke-test each model on 5 images (CPU).
3. **Data.** Download and convert everything to COCO-format JSON. Write `data/README.md`
   with the KITTI→COCO class mapping (car/van/truck/pedestrian/cyclist/…, how DontCare is handled,
   ignore regions). Fixed val splits with seeds.
4. **Eval harness.** One runner: `python -m hw1.eval --model X --data Y --cfg cfg.yaml` → predictions JSON +
   metrics JSON (COCO mAP, AP50/75, APs/m/l, per-class AP, AR, latency, peak memory). Log the
   thresholds, resolution, hardware and versions into every result file.
5. **Runs.** CPU subset runs locally; full runs on GPU. Append every command to `runs/COMMANDS.md`.
6. **Analyses.** Per-class/per-scale degradation tables; closed vs open-vocab; speed–accuracy plot;
   failure analysis (TIDE-style error breakdown + qualitative failure gallery); agent experiment.
7. **Optional fine-tune.** One model on the KITTI train split, compared against zero-shot.
8. **Report.** 6–10 pages, ≥8 references (≥4 papers, ≥2 repos/docs). Export to PDF.

## Rubric checklist

- [ ] ≥3 configs, ≥2 families
- [ ] ≥2 datasets, documented taxonomy mapping
- [ ] mAP, AP50/75, APs/m/l, per-class AP, AR, latency/FPS, GPU memory
- [ ] For every experiment: model + checkpoint, dataset + split, input resolution, conf/NMS thresholds, hardware, framework version, exact command
- [ ] ≥2 technical analyses
- [ ] failure analysis, conclusions
- [ ] ≥8 references
- [ ] code, configs and command log linked

## Open questions

1. Is there a local GPU somewhere (not this VM)? Otherwise the HPC handles GPU numbers.
2. The instructions mention "our tutorial" with a link that didn't survive the copy. What is it?
3. Due date?
