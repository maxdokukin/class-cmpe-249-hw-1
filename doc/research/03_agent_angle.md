# 03 — Agent / VLM angle for HW1 (Track A)

Researched 2026-10-27. This file covers VLMs that output boxes, agentic detection pipelines, driving-specific
VLM work, and 3 small experiment designs that add an agent angle while still producing COCO-style numbers.

Conventions: numbers quoted from a fetched paper/page carry its URL. **UNVERIFIED** marks claims taken
from search-engine snippets or secondary pages that were not checked against the primary source, and my own
compute/cost estimates (marked **ESTIMATE**) that must be measured before they go in the report.

---

## 1. Key takeaways

1. **VLMs are weak COCO detectors when scored with COCO mAP.** Qwen2.5-VL-7B gets ~18–19 mAP on COCO val2017,
   while Grounding DINO gets 52.5. The gap comes mostly from the metric: VLMs emit a few boxes with no
   confidence scores, so mAP has nothing to rank by and recall is capped. On recall/precision/F1 at IoU 0.5,
   the gap mostly closes (Qwen2.5-VL-7B F1 64.0 vs Grounding DINO-T 69.8).
2. **There is no standard way to give VLM boxes a score.** Papers set every score to 1.0 (RF100-VL, Qwen3-VL
   on ODinW), use box area as a pseudo-score, or report F1 instead of AP (Rex-Omni, CoT4Det). Whatever we
   choose, we have to state it. Comparing ranking schemes is a cheap and legitimate analysis in its own right.
3. **"Zero-shot COCO" is contaminated for most VLMs and open-vocabulary detectors** (their training data
   includes COCO images or COCO-derived data). KITTI is the cleaner zero-shot test. Say so in the report.
4. **Prompt/vocabulary engineering helps a little on average and a lot on some classes.** On nuImages,
   synonyms raised Detic from 14.40 to 14.92 mAP. LLM-written descriptions raised MQ-GLIP from 21.42 to 22.09,
   but individual classes jumped (pushable_pullable 3.6 → 15.3 AP). RF100-VL found that MLLMs often get
   *worse* when given rich annotator instructions instead of class names. So the honest hypothesis is "small
   mean gain, large gain on taxonomy-mismatched classes" (on KITTI: Cyclist, Van, Person_sitting, Tram).
5. **Detect-then-verify is a well-established pattern** (medical, ATR, driving safety). The usual claim is fewer
   false positives at some latency cost. Verification can only re-rank or delete boxes, so recall can never
   go above the base detector's.
6. **Agentic detection products report F1, not mAP**, on internal benchmarks (LandingAI). Rubric-grade work
   has to rebuild the COCO-style evaluation itself.

---

## 1b. What the instructor's harness (ngdet) already does — don't duplicate it

Source: [TUTORIAL.md](https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md)
and the code in `DeepDataMiningLearning/ngdet/` (read 2026-10-27: `taxonomy.py`, `run_eval.py`, `datasets.py`,
`evaluator.py`, `detectors/{grounding_dino,qwen_vl}.py`).

**Already in ngdet (treat as baseline infrastructure, not our contribution):**
- Closed-set detectors (torchvision Faster R-CNN/RetinaNet/FCOS, HF DETR/RT-DETR/Deformable DETR, Ultralytics YOLOv8/11/26),
  open-vocab detectors (Grounding DINO tiny/base, OWLv2), VLM detectors (Qwen2.5-VL, default `Qwen/Qwen2.5-VL-3B-Instruct`;
  NVIDIA LocateAnything-3B), on COCO / KITTI / Waymo / nuScenes / nuImages, with pycocotools metrics and a latency tool.
- Its taxonomy is **coarse**: `driving3` = vehicle / person / cyclist by default (`driving5` = car/truck/bus/person/cyclist).
  KITTI Van, Tram, Truck all fold into "vehicle"; COCO "bicycle" folds into "cyclist" (a caveat documented in `taxonomy.py`).
- **Hand-curated, fixed prompt synonyms** (`PROMPT_SYNONYMS`, e.g. vehicle → `car . truck . bus . van`). The tutorial
  notes that open-vocab models are "highly prompt- and score-scale-sensitive" ("vehicle" is much worse than "car . truck . bus").
  **There is no automated prompt search** (§12.4 of the tutorial only changes per-model score thresholds: Grounding DINO @0.15, OWLv2 @0.05).
- VLM boxes get **order-based scores** ("rank = order"). LocateAnything went from 0.055 to 0.274 nuImages mAP once it
  was prompted per class.
- Reported zero-shot KITTI mAP (driving3): YOLO26x 0.264, RT-DETR-R50 0.262, Faster R-CNN v2 0.279, Grounding-DINO-tiny 0.223.
  The tutorial's conclusion: "on KITTI every model lands ~0.23–0.28 regardless of architecture."
- **No agent / LLM loop, no verifier or cascade, no dev/test split.** `EvalDataset` takes the first `--max-images` sorted
  images (`idxs[:max_images]`).
- Predictions are **thresholded at `--score-thr` (default 0.3) before pycocotools**. This truncates the PR curve and
  biases mAP against models whose scores run on a low scale (OWL). The per-model thresholds in §12.4 work around it by hand.
- CUDA-only in the docs (`--device cuda`). The HF-based detectors should accept `--device mps`, but that is untested (UNVERIFIED).

**What this means for our design:**
1. Reuse ngdet's loaders/evaluator/adapters where possible (it's the course's reference, and graders will recognize it), but
   (a) evaluate mAP at a near-zero threshold (0.001, maxDets 100) and keep `score_thr` for the *operating-point* metrics only,
   and (b) add an explicit **drive-disjoint dev/test split** of KITTI.
2. Our distinct contributions are: (i) an **automated, leakage-free agent loop** over the prompt map (and the per-class
   operating thresholds), on the **fine KITTI taxonomy** (Car/Van/Truck/Pedestrian/Person_sitting/Cyclist/Tram), where the
   vocabulary mismatch actually bites, compared against ngdet's hand-curated `PROMPT_SYNONYMS` as the human baseline; and
   (ii) a **detector + VLM verifier cascade**. Both are absent from ngdet.
3. Design C (VLM as detector) mostly **duplicates** ngdet (Qwen2.5-VL and LocateAnything with order scoring). Keep only the
   cheap part that is new: the **ranking-rule ablation** (order vs 1.0 vs token log-prob vs area) on the same Qwen2.5-VL-3B outputs.

**Note on tuning thresholds:** with mAP at maxDets=100, lowering the score threshold almost never hurts AP, so "tuning
the threshold for mAP" is degenerate (the answer is always "as low as possible"). The agent should tune thresholds only for
an operating-point metric: per-class F1 or precision at a fixed recall on dev. That is also the number a driving stack would use.

---

## 2. VLMs that output boxes: measured detection numbers

| Model | Size | Box format | COCO val2017 | Other | Source |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 7B | JSON `bbox_2d` [x1,y1,x2,y2], absolute pixels on the resized image | 17.7 mAP (VLM-FO1 table); 19.1 mAP all-80-class prompt, 26.3 mAP when prompted only with GT classes (CoT4Det); F1@0.5 = 64.0 (Rex-Omni) | ODinW-13 43.1 mAP (72B, tech report) | [VLM-FO1](https://arxiv.org/html/2509.25916v1), [CoT4Det](https://arxiv.org/html/2512.06663), [Rex-Omni](https://arxiv.org/html/2510.12798v1), [Qwen2.5-VL report](https://arxiv.org/pdf/2502.13923) |
| Qwen2.5-VL-7B + area-ratio pseudo-score | 7B | same | 33.6 mAP reported with score = box area / image area | — | UNVERIFIED (search snippet attributed to an ablation, probably [One RL to See Them All](https://arxiv.org/pdf/2505.18129)). If true, the ranking rule alone moves mAP by ~15 points. |
| Qwen2.5-VL-72B | 72B | same | — | RF100-VL zero-shot 5.6 mAP (score = 1.0) | [RF100-VL](https://arxiv.org/html/2505.20612v3) |
| Qwen3-VL-235B-A22B | MoE | boxes (relative 0–1000 coordinates: UNVERIFIED) | — | ODinW-13 48.6 mAP, with all scores set to 1.0 | [Qwen3-VL report](https://arxiv.org/pdf/2511.21631) |
| InternVL2.5-8B | 8B | text boxes | 12.1 mAP | — | [VLM-FO1](https://arxiv.org/html/2509.25916v1) |
| ChatRex-7B | 7B | proposal-indexed | 4.3 mAP standard (48.2 under its own non-standard protocol) | — | [VLM-FO1](https://arxiv.org/html/2509.25916v1) |
| VLM-FO1-3B | 3B | picks from an external proposal network (OPN), so it has real scores | 44.4 mAP | — | [VLM-FO1](https://arxiv.org/html/2509.25916v1) |
| Rex-Omni | 3B (Qwen2.5-VL base, coord tokens 0–999) | quantized coordinate tokens | F1@0.5 = 72.0 (P 68.1, R 76.3); authors explicitly avoid AP, citing "lack reliable confidence estimation" | — | [Rex-Omni](https://arxiv.org/html/2510.12798v1) |
| Florence-2-base / -large | 0.23B / 0.77B | `<OD>` task, quantized location tokens, no scores | 34.7 / 37.5 mAP "zero-shot" (model card) | — | [HF model card](https://huggingface.co/GuiAworld/Florence-2-large) |
| PaliGemma 2 3B / 10B | 3B / 10B | `<locXXXX>` tokens | **after fine-tuning on COCO** (not zero-shot): 30.4/30.3 @224px, 38.5/39.2 @448px, 42.3/43.6 @896px | — | [PaliGemma 2](https://arxiv.org/pdf/2412.03555) |
| Gemini 2.5 Pro | hosted | [ymin,xmin,ymax,xmax] 0–1000 | 0.340 mAP on all 5k val images; it emits a score that only "partially correlated" with correctness; 5–25 invalid outputs per 5k images | Gemini 3 Pro Preview 0.407 mAP | [simedw blog](https://simedw.com/2025/07/10/gemini-bounding-boxes/) (independent blog, not peer-reviewed) |
| Gemini 2.5 Pro | hosted | — | — | RF100-VL zero-shot 11.6 mAP | [RF100-VL](https://arxiv.org/html/2505.20612v3) |
| GPT-4o | hosted | — | 0 F1 on LandingAI's internal benchmark | — | [LandingAI blog](https://landing.ai/blog/what-is-agentic-object-detection) (UNVERIFIED, vendor) |
| Molmo | 7B/72B | points, not boxes | — | used for hazard captioning on COOOL | [COOOL case study](https://arxiv.org/html/2412.19944v1) |
| *Reference:* Grounding DINO-T / OmDet-Turbo | 172M / — | scored boxes | 52.5 / 53.4 mAP | RF100-VL 15.7 mAP | [VLM-FO1](https://arxiv.org/html/2509.25916v1), [RF100-VL](https://arxiv.org/html/2505.20612v3) |

**Note on Florence-2's "zero-shot" claim:** FLD-5B was auto-annotated with specialist models trained on
COCO-like data, so treat 37.5 as "not trained on COCO labels", not as a clean domain transfer. UNVERIFIED in detail.

### 2.1 How to score free-form VLM boxes with COCO metrics

pycocotools ranks detections by score, so a VLM's unscored boxes need a ranking rule. The options seen in the
literature, plus the ones we can compute ourselves:

| Ranking rule | Used by | Pros / cons |
|---|---|---|
| score = 1.0 for every box | RF100-VL, Qwen3-VL (ODinW) | Simple, standard. All boxes tie, so the PR curve collapses to a single operating point. Underestimates AP compared with a scored detector. |
| score = box area / image area | ablation cited in §2 (UNVERIFIED) | Big boxes are usually right, so AP rises a lot. It's a heuristic, though, and it penalizes small objects, which matter on KITTI. |
| score = order of emission (first box highest) | common in practice (UNVERIFIED) | Free; VLMs tend to list salient objects first. |
| score = mean token log-prob of the label + coordinate tokens | possible with local models (HF `output_scores=True`, vLLM `logprobs`) | A real confidence signal; costs nothing extra. Hosted APIs (Claude, Gemini) do not expose logprobs for images, or only partially. |
| ask the model for a 0–1 confidence in the JSON | Gemini blog | Weakly calibrated, as the Gemini blog found. |
| skip AP; report P/R/F1 @IoU 0.5 (and 0.75) | Rex-Omni, CoT4Det, LandingAI | Honest, but not COCO mAP. Report alongside AP, never instead. |

**Recommendation:** for every VLM, report (a) mAP with score = 1.0, (b) mAP with the log-prob score,
(c) P/R/F1@0.5, all through the same pycocotools harness, and run the closed-set detectors through F1@0.5
at their best threshold too, so the comparison is fair in both directions. The ranking-rule ablation is a
small, novel-looking table that comes cheaply.

---

## 3. Agentic detection: what exists

### 3.1 LLM that chooses the vocabulary, then calls an open-vocab detector
- **LLM-Guided Agentic Object Detection** ([arXiv 2507.10844](https://arxiv.org/html/2507.10844v1)): Gemma-3 lists the
  objects it sees, and YOLO-World-X localizes them. It introduces class-agnostic AP (CAAP) and a CLIP-similarity
  naming AP (SNAP). On COCO val: CAAP 0.25, SNAP 0.54. The authors note the LLM overhead hurts real-time use.
  Not a prompt-*optimization* loop; one shot per image.
- **DetToolChain** (ECCV 2024, [arXiv 2403.12488](https://arxiv.org/abs/2403.12488)): GPT-4V/Gemini with a toolkit of
  visual prompts (zoom-in, rulers/compass overlays, scene graphs) and a detection chain-of-thought that
  diagnoses and refines its own boxes. Reports +21.5 AP50 on COCO novel classes (open-vocab split) over prior SOTA.
  This is the closest published template for "an agent that iteratively improves boxes".
- **CoT4Det** ([arXiv 2512.06663](https://arxiv.org/html/2512.06663)): decomposes detection into classify → count →
  ground steps for Qwen2.5-VL-7B; COCO mAP 19.1 → 32.7 (all classes). Shows that structured multi-step prompting
  of a single VLM is worth ~13 mAP.
- **LandingAI Agentic Object Detection** ([blog](https://landing.ai/blog/what-is-agentic-object-detection)): an agent
  that plans and calls OD tools plus a VLM. Internal benchmark F1 79.7 vs OWLv2 43.2 (recall 81.0, precision 29.5),
  Florence-2 39.7, Qwen2.5-VL-7B 35.1, GPT-4o 0. Vendor numbers, not reproducible: **UNVERIFIED**. Still, OWLv2's
  pattern (high recall, terrible precision) is exactly what a verifier stage should fix.

### 3.2 Prompt / vocabulary optimization for open-vocab detectors
- **Foundational FSOD** (Madan, Peri, Kong, Ramanan, NeurIPS 2024, [arXiv 2312.14494](https://arxiv.org/html/2312.14494v4)).
  This is the most relevant prior work because it is *driving data* (nuImages). Zero-shot nuImages AP:
  Detic 14.40, Grounding DINO 12.05, GLIP 17.01. Synonyms gave Detic 14.40 → 14.92. ChatGPT-4o iterative
  descriptions gave MQ-GLIP 21.42 → 22.09 mAP, with big per-class jumps (pushable_pullable 3.6 → 15.29,
  barrier 11.6 → 15.31). Their key quote: "trucks on the web may be defined differently from trucks for target
  applications such as autonomous vehicle perception." That sentence is the motivation for our Design A.
  The CVPR 2024 challenge winner reached 45.35 AP (with few-shot fine-tuning, not prompting alone).
- **Multi-Modal Classifiers for OVOD** (Kaul et al., ICML 2023, [paper](https://proceedings.mlr.press/v202/kaul23a.html)):
  LLM-generated class descriptions give +1.6 APr / +0.4 mAP on LVIS. Same message: small mean gain, larger on rare classes.
- **RF100-VL** ([arXiv 2505.20612](https://arxiv.org/html/2505.20612v3)): MLLMs "struggle to benefit from multi-modal
  annotator instructions", and replacing class names with descriptions sometimes *hurts*. That is a useful
  negative prior, and a reason to give the agent a feedback signal instead of letting it write descriptions blind.
- **LLMDet** (CVPR 2025 highlight, [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_LLMDet_Learning_Strong_Open-Vocabulary_Object_Detectors_under_the_Supervision_of_CVPR_2025_paper.html)):
  LLM supervision used at *training* time (captions) to strengthen a Grounding-DINO-style detector. LLMDet is a
  usable open-vocab checkpoint in its own right; it's related work, not an agent.

### 3.3 Detect-then-verify (detector proposals + VLM verification)
- Zero-shot ATR with LVLMs ([arXiv 2501.07396](https://arxiv.org/pdf/2501.07396)): an open-vocab detector proposes, an LVLM
  verifies and filters, and false positives drop.
- Adaptive detector–verifier for zero-shot polyp detection ([arXiv 2512.12492](https://arxiv.org/pdf/2512.12492)): a cascade
  in which an RL-tuned verifier still struggles with hard negatives. A good citation for the "verifier failure modes" discussion.
- SOTIF comparison of 10 LVLMs vs YOLOv5 / RT-DETRv4 on PeSOTIF driving scenes ([arXiv 2601.22830](https://arxiv.org/abs/2601.22830)):
  top LVLMs (Gemini 3) beat YOLOv5 recall by >25% and roughly match RT-DETRv4 under natural degradation. Specialized
  detectors stay ahead on geometric precision. The authors recommend LVLMs as *high-level safety validators*, which
  is the verifier role. Directly relevant driving citation.

### 3.4 Agent as annotator (auto-labeling → distill)
- **Autodistill** ([GitHub](https://github.com/autodistill/autodistill)): a base model (Grounding DINO / Grounded-SAM)
  labels images from a text ontology such as `{"sedan": "car"}`, and a target model (YOLOv8) trains on those labels.
  The ontology dict is exactly the "prompt → dataset class" mapping an agent would optimize. Accuracy claims in blog
  posts ("70–85% of human-label accuracy") are **UNVERIFIED** marketing.

### 3.5 Driving-specific VLM work (mostly not box-mAP)
- **DriveLM** (ECCV 2024, [arXiv 2312.14150](https://arxiv.org/pdf/2312.14150)): Graph-VQA over nuScenes/CARLA
  (perception → prediction → planning). Evaluated with language metrics, not detection AP. Cite for context only.
- **CODA-LM** (WACV 2025, [arXiv 2404.10595](https://arxiv.org/html/2404.10595v2)): corner-case LVLM benchmark with
  regional perception, judged by a text-only LLM. Again not mAP.
- **COOOL** (WACV 2025 workshop, [case study](https://arxiv.org/html/2412.19944v1)): out-of-label hazards (animals, debris);
  zero-shot pipelines with CLIP and Molmo. Useful to justify open-vocab detection in driving.
- **PeSOTIF LVLM comparison** (above): the only one here with detection metrics in driving.

**Gap we can occupy:** no paper found does a *controlled* comparison on KITTI of (closed-set COCO detectors)
vs (open-vocab with naive prompts) vs (open-vocab with agent-optimized prompts, tuned on a dev split) vs
(VLM-verified cascade) with COCO-style metrics. That is small enough for a homework and still a real experiment.

---

## 4. Experiment designs

> **Compute update (coordinator, 2026-10-27):** the machine is a **MacBook Air M4, 16 GB unified memory, MPS/MLX, fanless**,
> and the deadline is short (aim for ≤ 1 day of compute). §4.0 has the M4 estimates. The CPU-VM/HPC numbers inside
> each design are kept for reference only.

### 4.0 M4 (16 GB) feasibility and cost (all ESTIMATES, measure in a 20-image smoke test)

Base M4 memory bandwidth is ~120 GB/s, versus ~546 GB/s on an M4 Max. LLM decode is bandwidth-bound, so scale M4-Max
benchmarks down by ~4×. As a reference point, Qwen2.5-VL-7B Q4 runs at ~95 tok/s on an M4 Max
([willitrunai](https://willitrunai.com/de/can-run/qwen-2.5-vl-7b-on-m4-max-64gb), UNVERIFIED), which suggests ~20 tok/s on a base M4.
The Air is fanless, so expect throttling after ~10–20 min of sustained load. Run in chunks and report the latency from a
cool, warmed-up run.

| Component | Runtime on M4 | Memory | Per-unit time | Job on M4 |
|---|---|---|---|---|
| OWLv2-B/16 (HF, MPS) | PyTorch MPS | ~1–2 GB | ~0.3–0.8 s/img | cache 500 dev imgs ≈ 5 min; each agent prompt eval after that takes seconds (class head only) |
| Grounding DINO-T (HF, MPS + `PYTORCH_ENABLE_MPS_FALLBACK=1`) | MPS, some ops on CPU | ~2–3 GB | ~0.5–1.5 s/img | 15 iters × 300 dev imgs ≈ 1–2 h; test 1,500 imgs ≈ 15–40 min |
| YOLO-World-S/M (Ultralytics, MPS) | MPS | <1 GB | ~0.05–0.1 s/img | full loop well under 1 h |
| YOLO11 / RT-DETR base detectors | MPS | <2 GB | ~0.03–0.15 s/img | minutes |
| Qwen2.5-VL-3B 4-bit verifier (`mlx-vlm`) | MLX | ~2.5 GB | crop ~64 visual tokens + 1-token answer ≈ 0.3–0.8 s/crop | 5k crops (≈500 test imgs × ~10 band boxes) ≈ 0.5–1 h |
| Qwen2.5-VL-7B 4-bit (`mlx-vlm`) | MLX | ~5–6 GB | ≈ 1–2 s/crop; as a full-image detector ~15–30 s/img | verifier on 500 imgs ≈ 2–3 h; as a detector, only ~100 imgs |
| Qwen2.5-VL-3B as full-image detector (`mlx-vlm` or ngdet's HF adapter on MPS) | MLX/MPS | ~3–7 GB | ~5–10 s/img (≈600 visual tokens prefill + ~200 output tokens) | 500 imgs ≈ 1 h (only needed for the ranking-rule ablation) |

Notes: the yes/no token probability for the verifier score is available from local models (MLX/HF logits).
Whether `mlx-vlm` exposes logprobs through its Python API is UNVERIFIED; the HF `transformers` path on MPS
(`output_scores=True`) is the safe fallback, at maybe 2–3× slower.

**Hosted alternative (no local load, better verifier quality, no logprobs):**
- Agent loop (Design A) on Claude Opus 5.5 ($4/$20 per MTok): ~$1–3 per detector for 15 iterations; Sonnet 5.5 costs about half.
- Verifier (Design B) on Claude Haiku 4.5 ($1/$5, Batch API −50%): 5k crops ≈ $0.5–1 batched. Gemini 2.5 Flash
  ($0.30/$2.50): similar. Batch results come back within ~24 h, so submit early. Use the synchronous API for a 200-crop latency sample.
- Whole homework, all agent/verifier API spend: **< $10** (ESTIMATE).

Shared setup assumed from the other research files: KITTI 2D train images, split by *sequence/drive* into
`kitti_dev` (agent tuning, e.g. 500 images) and `kitti_test` (held-out evaluation, e.g. 1,500+ images); COCO
val2017 (or a fixed 500-image subset for VLMs). Evaluation through one pycocotools harness, with KITTI DontCare
regions as ignore. KITTI classes of interest: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram.

### Design A (recommended core): prompt-optimizing agent for open-vocab detectors on KITTI

**Question.** How much of the closed-set vs open-vocab gap on KITTI comes from the *taxonomy mismatch*
(COCO "person + bicycle" vs KITTI "Cyclist", no COCO "van"), and can an LLM agent close it by rewriting prompts,
without touching weights?

**Arms (all evaluated once on `kitti_test`):**
1. Closed-set baselines (YOLO11, RT-DETR, Faster R-CNN) with a hand-written COCO→KITTI mapping.
2. Open-vocab detector with raw KITTI class names as prompts (`car . van . truck . pedestrian . cyclist . tram .`).
3. Open-vocab detector with *human* synonyms written by the student in 15 minutes (the human baseline).
4. **No-feedback LLM**: one LLM call that writes synonyms from the class definitions in the KITTI devkit
   (tests whether the LLM's knowledge alone helps).
5. **Feedback agent**: the LLM proposes a prompt set → tool `evaluate(prompt_map, split="dev")` returns per-class AP,
   the confusion matrix and ~8 failure crops (FP/FN) → the LLM revises. Budget: K = 15 iterations; select the best
   prompt map *on dev*, then run it on test exactly once.

The prompt map is many-to-one: `{"sedan": Car, "hatchback": Car, "SUV": Car, "minivan": Van, "box truck": Truck,
"person riding a bicycle": Cyclist, "person sitting": Person_sitting, ...}`. After detection, merge to KITTI labels
and run class-wise NMS (IoU 0.6) across synonym hits to the same class.

**Detectors.** Use two, so the result isn't a one-model artifact:
- **OWLv2** (HF `transformers`): image embeddings don't depend on the prompt. The class head is a dot product of
  per-patch box embeddings with text embeddings, so you can cache image features for `kitti_dev` once and score any
  new prompt set in seconds, even on CPU. That makes the agent loop nearly free. (Check this in
  `OwlViTForObjectDetection` / `Owlv2ForObjectDetection`: `image_embedder` + `class_predictor`.)
- **Grounding DINO** (or MM-Grounding-DINO / LLMDet) and/or **YOLO-World**: text is fused early (feature enhancer /
  RepVL-PAN), so every prompt change needs a full forward pass. YOLO-World-S/M is fast enough on CPU; Grounding DINO
  needs a GPU for 15 iterations × 500 images.

**Metrics.** mAP, AP50/75, AP_s/m/l, per-class AP, and AR@100 on `kitti_test` for arms 1–5; the dev→test gap
for arm 5; number of LLM calls/tokens/$; wall-clock of the loop. Bootstrap 95% CI over test images (1,000
resamples) so a +0.5 AP change isn't over-read.

**Compute (ESTIMATE, measure in the smoke test):**
- OWLv2-B/16 on 20 vCPU: ~1–3 s/image forward → 500 dev images ≈ 15–25 min once for the cache, then each
  iteration costs seconds. Test-set pass ≈ 1 h on CPU, minutes on GPU.
- Grounding DINO-T on CPU: ~2–5 s/image, so 15 iterations × 500 images ≈ 4–10 h. Put it on the HPC GPU (~0.1 s/image,
  so the whole loop takes <30 min).
- YOLO-World-S on CPU: ~0.1–0.3 s/image, so the loop takes about an hour on CPU.

**API cost (ESTIMATE).** The agent runs on the Claude API with tools (Tool Runner). Per iteration: ~6k text tokens
(metrics, history) plus 8 crops × ~100–600 image tokens (Claude image tokens ≈ w·h/750) ≈ 10k input, ~2k output.
Claude Opus 5.5 ($4/$20 per MTok): ≈ $0.08/iteration → 15 iterations ≈ $1–3 per detector, including history growth.
Claude Sonnet 5.5 ($2/$10) costs about half. Prices from the Anthropic model table (cached 2026-09-25); recheck on the pricing page.
A local LLM (Qwen3 8B on the HPC) also works and costs $0, but it's a weaker optimizer.

**Risks.**
- **Test leakage**: the agent must never see `kitti_test` metrics or images. Enforce it in code (the tool hard-wires
  `split="dev"`), split by drive/sequence to avoid near-duplicate frames, and log every prompt the agent tried.
- **Selection bias / multiple comparisons**: 15 tries on dev means the best dev score is optimistic. Report the
  dev→test drop, and run the agent with 3 seeds (temperature or different first prompts) to show variance.
- **Synonym explosion**: more prompts mean more false positives and more latency (Grounding DINO's text length is
  capped at 256 tokens). Report AP *and* the number of prompts, and give the agent a cap (e.g. ≤ 5 prompts per class).
- **Gaming the evaluation**: e.g. dropping hard classes. The metric is fixed mAP over all target classes; a missing
  class scores 0 AP.
- **Grader skepticism**: frame it as "vocabulary alignment as hyperparameter search with an LLM proposer", with
  arms 3 and 4 as ablations that isolate what the feedback loop adds.

**Rubric mapping.** Closed-set vs open-vocabulary (main), cross-dataset generalization (per-class drop
COCO→KITTI, and which classes the agent recovers: expected Cyclist/Van/Person_sitting), failure analysis
(the agent's failure crops double as the qualitative gallery). It does not need fine-tuning, so the report
can still call it zero-shot.

### Design B: detector + VLM verifier cascade (precision at fixed recall, cost in latency)

**Question.** Does a VLM verifier lift precision at fixed recall (and AP) on KITTI for a fast detector, and what
latency does it add per image?

**Pipeline.** The base detector (YOLO11-s and/or OWLv2 with the Design-A prompts) runs at a low threshold (0.05).
Boxes in an *uncertain band* (e.g. 0.05 ≤ s < 0.5) are cropped with 20% context padding and sent to the verifier
with "Is the object in the red box a {class}? Answer yes or no." Boxes ≥ 0.5 pass through unchanged. The band is
the "agentic triage" element: the system decides where to spend compute. New score:
`s' = s · p_yes` (local VLM: `p_yes` from the softmax of the yes/no token logits) or `s' = s` if yes else `s · 0.1`
(hosted model with no logprobs). Re-evaluate with the same pycocotools harness.

**Verifiers (pick 1–2):** Qwen2.5-VL-7B or Qwen3-VL-8B (local, logprobs available), Florence-2-large (cheap, CPU),
Claude Haiku 4.5 or Gemini 2.5 Flash (hosted).

**Metrics.** AP / AP50 / per-class AP before vs after; **precision at recall = 0.5 / 0.7** per class; FP count
by TIDE error type (background, classification, duplicate); AR (must not exceed the base, as a sanity check);
added latency per image = (#boxes in band) × (verifier latency), reported as mean and p95; $/1k images for hosted
verifiers. Sweep the band width to draw an *accuracy vs added latency* curve, which feeds the speed–accuracy analysis.

**Compute (ESTIMATE).** KITTI averages maybe 5–15 uncertain boxes/image (measure). For 1,500 test images that's
~10–20k verifier calls.
- Qwen2.5-VL-7B on the 20-vCPU / 16 GB VM: bf16 weights (~15 GB) don't fit comfortably; a Q4 GGUF via llama.cpp
  (~5 GB) does. A 224×224 crop is ~64 visual tokens and the answer is 1 token, so prefill dominates: maybe
  2–6 s/crop → 10k crops ≈ 6–16 h. Feasible overnight on a 500-image subset only. Florence-2-large or
  Qwen2.5-VL-3B are the CPU-friendly choices.
- One HPC GPU (A100/H100, vLLM, batched): ~20–100 crops/s → 20k crops in minutes. Report per-crop latency at
  batch size 1 separately, since that's what matters for a car.
- Hosted: Claude Haiku 4.5 ($1/$5 per MTok, Batch API −50%): a crop is ~70–150 image tokens + ~80 prompt tokens,
  so 20k crops ≈ 4–5M input tokens ≈ $2–3 batched, plus output under $0.50. Gemini 2.5 Flash ($0.30/$2.50,
  [pricing summary](https://www.morphllm.com/gemini-api-pricing), recheck) is similar or cheaper. Hosted latency
  (network, ~0.5–2 s/call) makes the "latency" column meaningless for deployment; report it but label it API latency.

**Risks.** The verifier can't recover missed objects, so AR stays flat at best; a yes-biased VLM changes nothing;
crops of small/far KITTI objects (< 32 px) are too small for the VLM. Upscale the crops, and report AP_small
separately, since that is where the verifier is expected to fail. Threshold/band tuning must happen on
`kitti_dev`, not test. Hosted-model rate limits.

**Rubric mapping.** Speed–accuracy tradeoff (accuracy gain vs ms/image and $), architecture comparison
(single-stage detector vs detector+VLM two-stage), failure analysis (which FP types the verifier removes).

### Design C: VLM as a detector, compared head-to-head with closed-set and open-vocab detectors

> Largely covered by ngdet already (Qwen2.5-VL-3B and LocateAnything-3B adapters, order-based scores). Only the
> ranking-rule ablation below is new. See §1b and §5.

**Question.** Where do generative VLM detectors sit against YOLO / RT-DETR / Grounding DINO on COCO and KITTI,
and how much of the gap is the ranking rule rather than localization?

**Models.** Florence-2-large (`<OD>` and `<OPEN_VOCABULARY_DETECTION>` tasks; 0.77B, runs on CPU) and one of
Qwen2.5-VL-7B / Qwen3-VL-8B (HPC GPU). Optional: Gemini 2.5 Flash on a 200-image subset for a hosted reference.
This makes the VLM a *third detector family* (autoregressive coordinate-token decoding), which strengthens the
"≥2 families" requirement rather than only decorating it.

**Protocol.** COCO val2017 fixed 500-image subset + `kitti_test` (or a 500-image subset). Prompt: list all
instances of {classes} as JSON. Parse failures count as empty predictions and are reported. Score with the three
ranking rules from §2.1 + F1@0.5, alongside the same numbers for closed-set models.

**Compute (ESTIMATE).** Florence-2-large on CPU: ~1–3 s/image → 1,000 images in ~1 h. Qwen-7B on CPU: a full KITTI
image (1242×375) is ~600 visual tokens, and decoding 10–30 boxes is ~150–400 tokens at ~3–6 tok/s for Q4 on CPU,
so ~1 min/image. Too slow except for ~100 images. On one GPU with vLLM, ~1–3 s/image unbatched, much faster batched →
1,000 images in < 30 min. Gemini 2.5 Flash: ~600 input + ~400 output tokens/image → ~$1–2 per 1,000 images (thinking off).

**Risks.** COCO contamination in the VLM's training data (state it; KITTI is the cleaner test). Coordinate-frame bugs
(Qwen2.5-VL uses absolute pixels of the *resized* image, Gemini [ymin,xmin,ymax,xmax] on a 0–1000 scale, Qwen3-VL
reportedly relative 0–1000, UNVERIFIED). Unit-test the parser on 5 images with drawn boxes before running anything. VLMs
under-report crowded scenes (recall cap), and JSON truncation from `max_new_tokens`.

**Rubric mapping.** Architecture comparison (regression heads + NMS vs set prediction vs autoregressive
coordinate tokens), closed-set vs open-vocabulary, speed–accuracy (VLMs are 10–100× slower per image),
cross-dataset generalization.

### Optional Design D: agent as annotator → fine-tune (covers the fine-tuning analysis)

Use the Design-A prompt map with Grounding DINO/OWLv2 to auto-label ~2k KITTI train images (Autodistill-style).
Fine-tune YOLO11-n on (i) auto-labels and (ii) GT labels for the same images, with the same schedule, and compare both
against zero-shot YOLO11-n on `kitti_test`. The interesting number is the fraction of the GT-fine-tune gain recovered
with zero human labels. Needs ~1–2 GPU-hours (ESTIMATE). Risk: only if time allows; it shouldn't block the core.

### Optional add-on: failure-triage agent

An LLM reads per-class and per-scale tables, TIDE output and ~100 failure crops, then labels each failure
(occlusion, truncation, small/far, taxonomy confusion, label noise, night/glare). Measure agreement with the
student's own labels on the same 100 failures (Cohen's κ). Cheap (<$2), feeds the required failure analysis,
and is measurable. It's honest about being an assistant, not a result.

---

## 5. Recommendation (updated for the M4 and ngdet)

- **Core: Design A, an automated leakage-free prompt + threshold agent on the fine 7-class KITTI taxonomy.**
  - Detectors: OWLv2 (cached image embeddings, so each agent iteration takes seconds on the M4) and Grounding DINO-T (MPS, ~1–2 h loop).
  - Arms: closed-set baselines; raw class names; **ngdet's hand-curated `PROMPT_SYNONYMS` (the human baseline)**;
    a one-shot LLM with no feedback; the feedback agent.
  - The agent runs on the Claude API (~$2–5 in total). Tune on a drive-disjoint `kitti_dev`, evaluate once on `kitti_test`,
    with bootstrap CIs and 3 agent seeds.
  - mAP is computed at threshold 0.001. The agent tunes thresholds only for per-class F1 / P@R.
  - It extends ngdet's stated "prompt sensitivity" finding instead of repeating it, and covers *closed vs open-vocab* +
    *cross-dataset generalization*.
- **Second: Design B, a verifier cascade.**
  - YOLO11 or OWLv2 boxes in an uncertain score band are re-scored by Qwen2.5-VL-3B 4-bit on MLX
    (~0.5–1 h for 500 test images; local, so it has yes-probabilities).
  - Optionally add Claude Haiku 4.5 via the Batch API (< $1) as a stronger hosted verifier.
  - Sweep the band width to get the *speed–accuracy* curve. ngdet has nothing like this.
- **Design C is demoted**, since ngdet already runs Qwen2.5-VL/LocateAnything as detectors. Keep only the ranking-rule
  ablation (order vs 1.0 vs log-prob vs area) on ~300 images with Qwen2.5-VL-3B (~30–60 min on the M4), if time allows.
- Skip Design D (fine-tuning on auto-labels) under the deadline unless GPU time turns up.
- Hard rules for every design: tune only on `kitti_dev` (split by drive), evaluate on `kitti_test` once, bootstrap CIs,
  log every agent prompt/tool call to `runs/agent_logs/`, and record model IDs, prices and dates for reproducibility.
  Cite ngdet as the base harness, and state exactly which parts are ours.

## 6. Sources

- Qwen2.5-VL Technical Report — https://arxiv.org/pdf/2502.13923
- Qwen3-VL Technical Report — https://arxiv.org/pdf/2511.21631
- VLM-FO1 (COCO table for Qwen2.5-VL, InternVL2.5, ChatRex, Griffon-G, Grounding DINO) — https://arxiv.org/html/2509.25916v1
- Rex-Omni / Detect Anything via Next Point Prediction — https://arxiv.org/html/2510.12798v1
- CoT4Det — https://arxiv.org/html/2512.06663
- One RL to See Them All (possible source of area-ratio ablation, UNVERIFIED) — https://arxiv.org/pdf/2505.18129
- Roboflow100-VL — https://arxiv.org/html/2505.20612v3
- Florence-2 model card — https://huggingface.co/GuiAworld/Florence-2-large
- PaliGemma 2 — https://arxiv.org/pdf/2412.03555
- Gemini bounding-box benchmark blog — https://simedw.com/2025/07/10/gemini-bounding-boxes/
- LandingAI Agentic Object Detection — https://landing.ai/blog/what-is-agentic-object-detection
- LLM-Guided Agentic Object Detection — https://arxiv.org/html/2507.10844v1
- DetToolChain (ECCV 2024) — https://arxiv.org/abs/2403.12488
- Revisiting Few-Shot OD with VLMs / Foundational FSOD (NeurIPS 2024) — https://arxiv.org/html/2312.14494v4
- Multi-Modal Classifiers for OVOD (ICML 2023) — https://proceedings.mlr.press/v202/kaul23a.html
- LLMDet (CVPR 2025) — https://openaccess.thecvf.com/content/CVPR2025/html/Fu_LLMDet_Learning_Strong_Open-Vocabulary_Object_Detectors_under_the_Supervision_of_CVPR_2025_paper.html
- LVLMs for 2D detection under SOTIF conditions — https://arxiv.org/abs/2601.22830
- Zero-shot ATR with LVLMs (detect-then-verify) — https://arxiv.org/pdf/2501.07396
- Adaptive detector–verifier for polyp detection — https://arxiv.org/pdf/2512.12492
- DriveLM (ECCV 2024) — https://arxiv.org/pdf/2312.14150
- CODA-LM (WACV 2025) — https://arxiv.org/html/2404.10595v2
- COOOL zero-shot hazard identification — https://arxiv.org/html/2412.19944v1
- Autodistill — https://github.com/autodistill/autodistill
- ngdet tutorial and code (instructor harness) — https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md
- Qwen2.5-VL-7B on M4 Max throughput (secondary, UNVERIFIED) — https://willitrunai.com/de/can-run/qwen-2.5-vl-7b-on-m4-max-64gb
- Qwen2.5-VL grounding output format (LearnOpenCV) — https://learnopencv.com/object-detection-with-vlms-ft-qwen2-5-vl/
- Gemini API pricing summary (secondary; recheck on ai.google.dev) — https://www.morphllm.com/gemini-api-pricing
- Claude pricing: Anthropic model table (cached 2026-09-25): Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 4.5 $1/$5 per MTok; Batch API 50% off.
