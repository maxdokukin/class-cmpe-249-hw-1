# 01 — Models and toolchains (research, 2026-10-27)

Scope: which 2D detectors to evaluate for Track A, how to install and run them, what COCO val2017 numbers
they claim, whether they run on the dev box (20 vCPU, no GPU) and on the primary accelerator
(MacBook Air M4, PyTorch MPS), and what to watch for so the mAP and latency numbers are honest.

Nothing was installed. Every number has a source URL. "UNVERIFIED" marks claims I could not confirm
from a primary source, or that rest on my own knowledge and need a smoke test.

## TL;DR — recommended set

One Python 3.12 env: `torch` + `torchvision` + `ultralytics` + `transformers` + `pycocotools`.
No mmdet/mmcv, no repo clones, no compiled CUDA ops. Every model below is either in the course
tutorial's `ngdet` backends or one name-change away from one.

| # | Config | Family | Package / checkpoint | COCO val2017 AP (claimed) | MPS OK? |
|---|---|---|---|---|---|
| 1 | YOLO11m | one-stage CNN, anchor-free, NMS | `ultralytics` / `yolo11m.pt` | 51.5 | yes (documented device) |
| 2 | YOLO26m | one-stage CNN, NMS-free one-to-one head | `ultralytics` / `yolo26m.pt` | 53.1 (52.5 e2e) | yes (documented device) |
| 3 | Faster R-CNN R50-FPN v2 | two-stage, anchor-based | `torchvision` / `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` | 46.7 | yes (nms/roi_align MPS kernels since tv 0.16) |
| 4 | RT-DETR R50 (v1, or v2) | DETR-family real-time, query-based, NMS-free | `transformers` / `PekingU/rtdetr_r50vd` (or `PekingU/rtdetr_v2_r50vd`) | 53.1 (v1) / 53.4 (v2-L) | probably (grid_sample forward exists on MPS); smoke-test |
| 5 | Grounding DINO-T | open-vocabulary, text-conditioned DETR | `transformers` / `IDEA-Research/grounding-dino-tiny` | 48.4 zero-shot | probably, with `PYTORCH_ENABLE_MPS_FALLBACK=1`; smoke-test |
| 6 | YOLO-World v2 (s or l) | open-vocabulary CNN | `ultralytics` / `yolov8s-worldv2.pt` or `yolov8l-worldv2.pt` | see §5 (O365-trained, "zero-shot") | yes (same stack as #1) |
| opt | FCOS R50-FPN | anchor-free dense, one-stage | `torchvision` / `FCOS_ResNet50_FPN_Weights.COCO_V1` | 39.2 | yes |

Why this set:
- It covers 5 families (CNN one-stage with NMS, two-stage, DETR query-based, open-vocab transformer,
  open-vocab CNN), so the "≥2 families" rule is met several times over, and three analyses fall out
  directly: YOLO vs RT-DETR (NMS vs NMS-free), Faster R-CNN vs YOLO (two-stage vs one-stage), and
  closed-set vs open-vocab (YOLO11 vs YOLO-World vs Grounding DINO). YOLO11 vs YOLO26 adds an NMS vs
  NMS-free comparison inside one family with the same code path.
- Install friction is three `pip install`s. All of them have wheels for Python 3.10–3.14 except
  `transformers`, which needs ≥3.11, so pin Python 3.12 (also what the tutorial uses).
- It matches the instructor's `ngdet` tutorial backends (`torchvision`, `hf_detr`, `yolo`, `gdino`,
  `owlv2`), so the tutorial's code can be reused or cross-checked.
- Every model is a plain PyTorch module, so one harness can time it and score it the same way.
- Size choice: "m" sizes keep M4 runtimes manageable before the deadline. The tutorial uses "x"
  (`yolov8x.pt, yolo11x.pt, yolo26x.pt`); switch to x only if time allows.

Swap-ins if one fails a smoke test: OWLv2 (`google/owlv2-base-patch16-ensemble`, in the tutorial)
for Grounding DINO; D-FINE-M (`ustc-community/dfine-medium-coco`) or Ultralytics `rtdetr-l.pt` for
HF RT-DETR; RetinaNet v2 for FCOS.

## 1. Course tutorial (instructor's `ngdet`)

Source: https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md

- Backends and example checkpoints (quoted from the tutorial):
  - `torchvision`: `fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2, fcos_resnet50_fpn` (`detectors/torchvision_det.py`)
  - `hf_detr`: `facebook/detr-resnet-50, PekingU/rtdetr_r50vd, SenseTime/deformable-detr` (`detectors/hf_detr.py`)
  - `yolo`: `yolov8x.pt, yolo11x.pt, yolo26x.pt, yolov8s-worldv2.pt` (`detectors/yolo.py`)
  - `gdino`: `IDEA-Research/grounding-dino-tiny, …-base`; `owlv2`: `google/owlv2-base-patch16-ensemble`;
    `locate_anything`: `nvidia/LocateAnything-3B`
- Datasets: COCO, KITTI, Waymo, nuScenes (3D boxes projected to 2D), nuImages.
- Eval: `python -m DeepDataMiningLearning.ngdet.run_eval --models ... --datasets ... --max-images 50 --score-thr 0.3 --device cuda`.
  `evaluator.py` uses pycocotools under a unified taxonomy (`driving3` = vehicle/person/cyclist by
  default; also `vehicle_person`, `driving5`).
  (https://github.com/lkk688/DeepDataMiningLearning/tree/main/DeepDataMiningLearning/ngdet)
- Env: `torch torchvision transformers ultralytics pycocotools opencv-python matplotlib pyarrow`, conda `py312`.

Things in the tutorial code to be careful about (from fetched source of `run_eval.py`, `detectors/yolo.py`, `latency.py`):
- `--score-thr` defaults to 0.3 (`run_eval.py`), applied to every detector. A 0.3 cut lowers COCO mAP
  because AP integrates over the full precision-recall curve; the numbers are fine for a uniform
  "deployment" comparison but are not comparable with the official numbers below. Report both:
  low-threshold (≈0.001) for mAP, and 0.3 for the tutorial-style comparison.
- Its mAPs are over the 3-class unified taxonomy, not COCO-80. The tutorial's headline result: "on KITTI
  every model lands ~0.23–0.28 regardless of architecture."
- `--device` defaults to `cuda`; there is no mention of `mps` anywhere in `run_eval.py`.
  `detectors/yolo.py` parses the device with `device.split(":")[-1]` and falls back to `cpu` for non-CUDA
  strings, so `--device mps` may quietly run YOLO on CPU (UNVERIFIED, read from a summary of the
  source; check before relying on it).
- `latency.py` times end-to-end `detector.predict(image)` with warmup, `torch.cuda.synchronize()`,
  `time.perf_counter()`, 30 iterations, a single KITTI image; it measures no memory.
- YOLO predicts with `conf=0.0` and filters later (good); YOLO-World gets `set_classes(taxonomy.prompts())`.

## 2. Toolchain versions (as of 2026-10-27)

| Package | Latest | Python | Source |
|---|---|---|---|
| ultralytics | 8.4.190 (2026-10-26) | ≥3.8 | https://pypi.org/project/ultralytics/ |
| transformers | 5.21.0 (2026-10-26) | ≥3.11, PyTorch ≥2.6 | https://pypi.org/project/transformers/ |
| torchvision | 0.29.1 (2026-09-30) | 3.10–3.14 | https://pypi.org/project/torchvision/ |
| torch | 2.14 (stable docs); 2.15 at RC; 2.14 is last with 3.10 wheels | 3.10–3.14 | https://dev-discuss.pytorch.org/t/notice-python-3-10-support-is-being-removed-from-pytorch-2-15-2-14-is-the-last-release-with-3-10-wheels/3440 |
| mmdetection | v3.3.0 (2024-01-05), no release since | — | https://github.com/open-mmlab/mmdetection/releases |

Python 3.13: torch/torchvision/transformers/ultralytics all publish wheels up to 3.14, so 3.13 would
work, but `pycocotools` and the tutorial target 3.12; use 3.12 to avoid surprises (UNVERIFIED that
pycocotools has 3.13 macOS arm64 wheels; it builds from source easily if not).

**Avoid mmdet/mmcv.** Last mmdet release is January 2024; MMDetection 3.x pins `mmcv>=2.0.0,<2.1.0`
(https://mmdetection.readthedocs.io/en/3.x/notes/faq.html), mmcv-full has reported install problems
on Python 3.12 (https://www.inflearn.com/community/questions/1663399), and mmcv's compiled ops have no
MPS path (UNVERIFIED). Everything mmdet would give us (Cascade R-CNN, MM-Grounding-DINO) is either
not needed or available in transformers.

## 3. Closed-set candidates

### 3.1 One-stage CNN (Ultralytics)

All: `pip install ultralytics`, AGPL-3.0 (or Enterprise), COCO-80 weights auto-download.
YOLO11 has no paper ("Ultralytics has not published a formal research paper for YOLO11").

| Model | Paper | Released | Size | COCO AP50-95 | CPU ONNX ms | GPU ms | Params M | FLOPs B | Source |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n/s/m/l/x | none (software) | 2023-01-10 | 640 | 37.3 / 44.9 / 50.2 / 52.9 / 53.9 | 80.4 / 128.4 / 234.7 / 375.2 / 479.1 | A100 TRT 0.99 / 1.20 / 1.83 / 2.39 / 3.53 | 3.2 / 11.2 / 25.9 / 43.7 / 68.2 | 8.7 / 28.6 / 78.9 / 165.1 / 257.8 | https://docs.ultralytics.com/models/yolov8/ |
| YOLO11n/s/m/l/x | none (software) | 2024-09-27 | 640 | 39.5 / 47.0 / 51.5 / 53.4 / 54.7 | 56.1 / 90.0 / 183.2 / 238.6 / 462.8 | T4 TRT10 1.5 / 2.5 / 4.7 / 6.2 / 11.3 | 2.6 / 9.4 / 20.1 / 25.3 / 56.9 | 6.5 / 21.6 / 68.1 / 87.2 / 195.3 | https://docs.ultralytics.com/models/yolo11/ |
| YOLO12n/s/m/l/x | arXiv 2502.12524 (2025) | 2025 | 640 | 40.6 / 48.0 / 52.5 / 53.7 / 55.2 | — | T4 1.64 / 2.61 / 4.86 / 6.77 / 11.79 | 2.6 / 9.3 / 20.2 / 26.4 / 59.1 | 7.5 / 23.3 / 70.7 / 95.4 / 208.9 | https://docs.ultralytics.com/models/yolo12/ |
| YOLO26n/s/m/l/x | arXiv 2606.03748 (2026) | 2026-01 | 640 | 40.9 / 48.6 / 53.1 / 55.0 / 57.5 (e2e: 40.1 / 47.8 / 52.5 / 54.4 / 56.9) | 38.9 / 87.2 / 220.0 / 286.2 / 525.8 | T4 TRT10 1.7 / 2.5 / 4.7 / 6.2 / 11.8 | 2.4 / 9.5 / 20.4 / 24.8 / 55.7 | 5.5 / 20.9 / 68.4 / 86.8 / 194.4 | https://docs.ultralytics.com/models/yolo26/ |

Notes:
- YOLO26 is the newest Ultralytics model (paper "Ultralytics YOLO26: Unified Real-Time End-to-End Vision
  Models", https://arxiv.org/abs/2606.03748). It has a dual head: "Prediction and validation default to
  the one-to-many head for accuracy. Use `nms=False` for NMS-free inference." Its speed numbers use the
  NMS-free head. So the "mAP" column is with NMS, the "e2e" column is NMS-free. Report which one you ran.
  The YOLO26 docs claim "up to 43% faster CPU ONNX inference" for 26n vs 11n.
- YOLO12 is attention-based; Ultralytics itself warns it "may exhibit training instability, elevated
  memory consumption, and slower CPU throughput" and recommends YOLO11 or YOLO26
  (https://docs.ultralytics.com/models/yolo12/). Skip it.
- CPU: n/s/m sizes are fine on the 20-vCPU dev box for subsets (YOLO11m ≈ 183 ms/img in ONNX on
  Ultralytics' reference CPU; PyTorch eager will be slower; UNVERIFIED on this box).

Ultralytics eval defaults (https://docs.ultralytics.com/modes/val/, https://docs.ultralytics.com/modes/predict/):

| Arg | `val` default | `predict` default |
|---|---|---|
| conf | **0.001** | **0.25** |
| iou (NMS) | 0.7 | 0.7 |
| max_det | 300 | 300 |
| imgsz | 640 | 640 |
| rect | True | n/a (letterbox) |
| agnostic_nms | False | False |
| nms | None | None (NMS on) |

Gotchas:
- If you build your own harness on `model.predict`, pass `conf=0.001` (or 0.0) explicitly; the predict
  default of 0.25 cuts recall and lowers mAP.
- `val` with `rect=True` pads differently from `predict`: "a square 640×640 image validated at imgsz=640
  with stride 32 uses a 672×672 input with 16 px of padding on each side." Use `rect=False` if you want
  val and predict geometry to match.
- `val(save_json=True)` scores with `faster-coco-eval` and reports APs/m/l. For one scorer across all
  frameworks, export predictions and score with pycocotools yourself.
- Ultralytics class ids are contiguous 0–79. COCO JSON needs the 91-style category ids; Ultralytics has a
  helper `ultralytics.data.converter.coco80_to_coco91_class()` (UNVERIFIED path, from memory).
- RT-DETR is also in Ultralytics (`rtdetr-l.pt` 53.0 AP, `rtdetr-x.pt` 54.8 AP,
  https://docs.ultralytics.com/models/rtdetr/), so RT-DETR can run through the YOLO code path if HF fails.

### 3.2 Two-stage and anchor-free dense (torchvision)

`pip install torchvision`; BSD-3 code; weights are COCO-trained (check per-weight license notes;
UNVERIFIED). Numbers from https://docs.pytorch.org/vision/main/models.html and
https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html.

| Builder (weights) | Paper | COCO box AP | Params | GFLOPs |
|---|---|---|---|---|
| `fasterrcnn_resnet50_fpn_v2` (`COCO_V1`) | Faster R-CNN arXiv 1506.01497 (2015); v2 recipe from "Benchmarking Detection Transfer Learning with ViTs" arXiv 2111.11429 (UNVERIFIED attribution) | 46.7 | 43.7M | 280.37 |
| `fasterrcnn_resnet50_fpn` | 1506.01497 | 37.0 | 41.8M | 134.38 |
| `fasterrcnn_mobilenet_v3_large_fpn` | — | 32.8 | 19.4M | 4.49 |
| `retinanet_resnet50_fpn_v2` | RetinaNet arXiv 1708.02002 (2017) | 41.5 | 38.2M | 152.24 |
| `retinanet_resnet50_fpn` | 1708.02002 | 36.4 | 34.0M | 151.54 |
| `fcos_resnet50_fpn` | FCOS arXiv 1904.01355 (2019) | 39.2 | 32.3M | 128.21 |
| `ssdlite320_mobilenet_v3_large` | — | 21.3 | 3.4M | 0.58 |

Inference defaults (fetched from `torchvision/models/detection/*.py` on main):

| Model | min/max size | score thresh | NMS IoU | detections/img | other |
|---|---|---|---|---|---|
| FasterRCNN | 800 / 1333 | `box_score_thresh=0.05` | 0.5 | 100 | `rpn_pre/post_nms_top_n_test=1000` |
| RetinaNet | 800 / 1333 | `score_thresh=0.05` | 0.5 | 300 | `topk_candidates=1000` |
| FCOS | 800 / 1333 | **`score_thresh=0.2`** | 0.6 | 100 | `topk_candidates=1000`, `center_sampling_radius=1.5` |

Gotchas:
- FCOS's default `score_thresh=0.2` truncates the PR curve; pass `score_thresh=0.05` (or 0.001) to the
  builder for honest mAP. Faster R-CNN's 0.05 is the standard COCO-eval value.
- Input is resized to shorter side 800 (KITTI 1242×375 becomes ~2651×800 capped by max 1333, so
  ≈1333×402). Record the effective resolution; it differs from YOLO's 640 letterbox.
- torchvision label ids follow the 91-slot COCO list with "N/A" placeholders, so label == COCO
  `category_id` (UNVERIFIED; check `weights.meta["categories"]` in the smoke test).
- Cascade R-CNN is not in torchvision; it needs mmdet or detectron2. Skip it.
- CPU: Faster R-CNN v2 at 800px is heavy (280 GFLOPs); expect >1 s/img on the dev box (UNVERIFIED).

### 3.3 DETR-family real-time

| Model | Paper | Easiest impl | Checkpoint(s) | COCO AP | Params | License | Source |
|---|---|---|---|---|---|---|---|
| RT-DETR (v1) | arXiv 2304.08069 (2023) | transformers `RTDetrForObjectDetection`; also Ultralytics | `PekingU/rtdetr_r18vd` … `PekingU/rtdetr_r50vd`, `_r101vd` | R50 53.1, R101 54.3 (paper abstract) | — | Apache-2.0 | https://arxiv.org/abs/2304.08069, https://github.com/lyuwenyu/RT-DETR |
| RT-DETRv2 | arXiv 2407.17140 (2024) | transformers `RTDetrV2ForObjectDetection` (added 2025-02-06) | `PekingU/rtdetr_v2_r18vd`, `_r34vd`, `_r50vd`, `_r101vd` | S 48.1 / M 51.9 / L 53.4 / X 54.3 | 20 / 36 / 42 / 76 M | Apache-2.0 | https://huggingface.co/docs/transformers/model_doc/rt_detr_v2, https://github.com/lyuwenyu/RT-DETR |
| D-FINE | arXiv 2410.13842 (2024) | transformers `DFineForObjectDetection` (added 2025-04-29) | `ustc-community/dfine-{small,medium,large,xlarge}-coco`, `-obj2coco` variants | COCO-only N 42.8 / S 48.5 / M 52.3 / L 54.0 / X 55.8; O365→COCO S 50.7 / M 55.1 / L 57.3 / X 59.3 | 4 / 10 / 19 / 31 / 62 M | Apache-2.0 | https://github.com/Peterande/D-FINE, https://huggingface.co/docs/transformers/model_doc/d_fine |
| DEIMv2 | arXiv 2509.20787 (2025) | transformers `Deimv2ForObjectDetection` | doc example uses `harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers` (community conversion); originals `Intellindust/DEIMv2_*` | X 57.8 (50.3M), S 50.9 (9.71M), Pico 38.5 (1.5M) | — | UNVERIFIED | https://huggingface.co/docs/transformers/model_doc/deimv2 |
| RF-DETR | arXiv 2511.09554 (2025) | `pip install rfdetr` (Py ≥3.10); also transformers `RfDetrForObjectDetection` | `Roboflow/rf-detr-base`, `Roboflow/rf-detr-medium` (HF) | N 48.4 (384px) / M 54.7 (576) / L 56.5 (704) / 2XL 60.1 (880) | 30.5 / 33.7 / 33.9 / 126.9 M | Apache-2.0 for N/S/M/L; PML 1.0 for XL/2XL and tiny sizes | https://github.com/roboflow/rf-detr, https://huggingface.co/docs/transformers/model_doc/rf_detr, https://arxiv.org/abs/2511.09554 |

Gotchas (HF DETR-family):
- `post_process_object_detection(outputs, threshold=0.5, target_sizes=..., use_focal_loss=True)`: default
  `threshold=0.5` (from `image_processing_rt_detr.py` on main). It takes sigmoid scores, then
  `torch.topk` over queries×classes (≤300 boxes). For mAP pass `threshold=0.0` or 0.001; the doc examples
  use 0.5 and 0.9.
- `RTDetrImageProcessor` resizes to a fixed `{"height": 640, "width": 640}` without keeping aspect
  ratio. The RT-DETRv2 docs warn "Reshaping to other sizes will generally degrade performance." KITTI's
  3.3:1 frames get squashed vertically, which is a domain-shift factor to name in the report.
- `id2label` for these checkpoints is contiguous 0–79 (UNVERIFIED; `logits` shape `[1, 300, 80]` in the
  docs suggests so), so map to COCO category ids by name. Older `facebook/detr-resnet-50` uses 91 slots.
- DETRs are NMS-free by design: no IoU threshold to report, only score threshold and top-k = 300.

### 3.4 MPS notes for closed-set models

- torchvision 0.16 added MPS kernels for "`nms` and roi-align kernels (`roi_align`, `roi_pool`,
  `ps_roi_align`, `ps_roi_pool`)" (https://newreleases.io/project/pypi/torchvision/release/0.16.0), so
  Faster R-CNN, RetinaNet and FCOS inference should run on MPS. `deform_conv2d` got no MPS kernel (not
  used by these models).
- RT-DETR / D-FINE / DEIM decoders use deformable attention via `F.grid_sample`. PyTorch made
  `grid_sampler_2d` available on MPS behind an MPSGraph enum guard (search result pointing to the
  PyTorch MPS op-tracking issue https://github.com/pytorch/pytorch/issues/141287); `grid_sampler_3d`
  is still missing. Inference (forward only) should work on a recent macOS; UNVERIFIED on the M4 —
  smoke-test and keep `PYTORCH_ENABLE_MPS_FALLBACK=1` set.
- RF-DETR on MPS: `aten::_upsample_bicubic2d_aa.out` is not supported and falls back to CPU
  (https://discuss.roboflow.com/t/can-rf-detr-run-with-mac-silicon/10580). It runs, but slower.
- Ultralytics supports `device="mps"` (https://github.com/ultralytics/ultralytics/issues/13033). I found
  no confirmed MPS-vs-CPU mAP discrepancy report. Run COCO val on a small subset on both devices once
  and compare to rule it out.

## 4. Open-vocabulary candidates

| Model | Paper | Easiest impl | Checkpoint | COCO zero-shot box AP | Other | License | MPS OK? |
|---|---|---|---|---|---|---|---|
| Grounding DINO-T | arXiv 2303.05499 (2023) | transformers `AutoModelForZeroShotObjectDetection` | `IDEA-Research/grounding-dino-tiny` | 48.4 (O365, GoldG, Cap4M; no COCO) | fine-tuned 57.2 | Apache-2.0 | probably; deformable attn → `grid_sample` path; set fallback env |
| Grounding DINO-B | same | same | `IDEA-Research/grounding-dino-base` | **not zero-shot**: 56.7, trained with COCO in the mix | — | Apache-2.0 | same |
| MM-Grounding-DINO | arXiv 2401.02361 (2024) | transformers (added 2025-08-01) | `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg` | 50.6 (tiny) | LVIS minival 41.4 | UNVERIFIED | same as GDINO |
| LLMDet | arXiv 2501.18954 (2025) | transformers ≥4.55 (`AutoModelForZeroShotObjectDetection`) | `iSEE-Laboratory/llmdet_{tiny,base,large}` | UNVERIFIED (training data includes COCO-derived GroundingCap; not a clean zero-shot) | LVIS minival 50.7 / 54.3 / 56.6 | Apache-2.0 | same as GDINO |
| OWLv2 | arXiv 2306.09683 (2023) | transformers `Owlv2ForObjectDetection` | `google/owlv2-base-patch16-ensemble` | UNVERIFIED: 43.4 AP (third-party card), 45.5 in SAM 3 paper's table | ~0.2B params, 960×960 input | Apache-2.0 | likely (plain ViT); slow |
| YOLO-World v2 | arXiv 2401.17270 (2024) | `ultralytics` `YOLOWorld` | `yolov8{s,m,l,x}-worldv2.pt` | x-worldv2 47.1 per Ultralytics docs, trained on O365+GoldG | `set_classes([...])` | AGPL-3.0 | yes |
| YOLOE | arXiv 2503.07465 (2025) | `ultralytics` `YOLOE` | `yoloe-26{n,s,m,l,x}-seg.pt`, `yoloe-11{s,m,l}-seg.pt`, `*-pf.pt` prompt-free | no COCO val example; LVIS text-prompt AP 24.7 / 30.8 / 35.4 / 37.8 / 40.6 (26n..x) | text encoder ~254 MB first-call download | UNVERIFIED (likely AGPL) | yes |
| Florence-2 | arXiv 2311.06242 (2023) | transformers `Florence2ForConditionalGeneration` | `microsoft/Florence-2-base`, `-large` | 34.7 / 37.5 (model card "zero-shot") | `<OD>` task; **no confidence scores** | MIT | likely; slow (autoregressive) |
| SAM 3 | "SAM 3: Segment Anything with Concepts" (ICLR 2026) | transformers `Sam3Model` | `facebook/sam3` (848M) | 53.6 COCO box AP (vs DINO-X 52.4, OWLv2 45.5; secondary source) | one text concept per pass | UNVERIFIED (Meta license; repo may be gated) | UNVERIFIED |
| LocateAnything-3B | NVIDIA, 2026-06-29 | HF `nvidia/LocateAnything-3B`; in tutorial | — | UNVERIFIED | VLM, 2.98–4.40 s/img in tutorial | NVIDIA non-commercial | UNVERIFIED |

Sources: Grounding DINO zoo https://github.com/IDEA-Research/GroundingDINO ; HF doc
https://huggingface.co/docs/transformers/model_doc/grounding-dino ; MM-GDINO
https://huggingface.co/docs/transformers/en/model_doc/mm-grounding-dino ; LLMDet
https://huggingface.co/iSEE-Laboratory/llmdet_large/blob/main/README.md and https://arxiv.org/abs/2501.18954 ;
OWLv2 https://huggingface.co/google/owlv2-base-patch16-ensemble and https://superlinked.com/models/google-owlv2-base-patch16-ensemble.md ;
YOLO-World https://docs.ultralytics.com/models/yolo-world/ ; YOLOE https://docs.ultralytics.com/models/yoloe/ ;
Florence-2 https://huggingface.co/microsoft/Florence-2-base ; SAM 3 https://huggingface.co/docs/transformers/model_doc/sam3 and
https://en.papernotes.org/ICLR2026/segmentation/sam_3_segment_anything_with_concepts/ ;
LocateAnything https://docs.voxel51.com/model_zoo/models/nvidia_LocateAnything_3B.html and the ngdet tutorial.
Newer 2026 open-vocab work exists (e.g. OV-DEIM, https://arxiv.org/abs/2602.00531, which reports beating
YOLOE on LVIS rare classes) but has no pip-installable release I could confirm (UNVERIFIED).

Gotchas:
- "Zero-shot on COCO" mostly means "no COCO images", not "unseen concepts": Grounding DINO-T and
  YOLO-World are trained on Objects365, whose classes cover nearly all of COCO's. Say this in the
  report's closed-vs-open analysis. On KITTI the zero-shot claim is cleaner.
- Grounding DINO in transformers: classes go in one string separated by periods, e.g. `"a cat. a dog."`;
  use `processor.post_process_grounded_object_detection(outputs, input_ids, threshold=0.25,
  text_threshold=0.25, target_sizes=...)` (defaults 0.25/0.25); `num_queries=900`, `max_text_len=256`
  tokens. 80 COCO names fit in 256 BERT tokens (UNVERIFIED; count them in the smoke test, otherwise
  split into chunks and merge). Labels come back as text phrases, so map phrase → class by string match.
  For mAP set `threshold` low (≈0.05) and keep top-100 per image by score.
- OWLv2: `post_process_object_detection` default threshold 0.1; one text query per class; 960×960 input.
- Florence-2 returns boxes and labels with no scores, so you can't rank detections and COCO AP is not
  well-defined. Use it only for qualitative or fixed-threshold precision/recall, or skip it.
- SAM 3 and LocateAnything are too heavy for an M4 Air before the deadline; mention them as related work.
- For KITTI, prompts are a free design choice ("car. van. truck. pedestrian. cyclist." vs
  COCO names). Fix them in the config and log them; the tutorial reports open-vocab mAP "after prompt tuning".

## 5. Getting COCO-compatible output from each framework

Recommendation: one harness, one scorer. Each adapter returns `[{"image_id", "category_id", "bbox":
[x, y, w, h] in original-image pixels, "score"}]`; score everything with `pycocotools.COCOeval`
(`maxDets=[1,10,100]`; gives AP, AP50, AP75, APs/m/l, AR1/10/100), and per-class AP from
`eval.eval["precision"][:, :, k, 0, 2]`. This is what the tutorial's `evaluator.py` does too.

| Framework | Raw output | Box format | Class ids | What to do |
|---|---|---|---|---|
| Ultralytics | `Results.boxes.xyxy / conf / cls` (orig. pixels) | xyxy | 0–79 contiguous | xyxy→xywh; map with `coco80_to_coco91_class()`; `conf=0.001`, `iou=0.7`, `max_det=300` |
| torchvision | `[{"boxes","scores","labels"}]` (orig. pixels) | xyxy | 91-slot COCO ids (UNVERIFIED) | xyxy→xywh; set `box_score_thresh`/`score_thresh` low |
| HF closed-set DETRs | `post_process_object_detection(..., target_sizes)` | xyxy abs | `id2label` 0–79 (check) | map by name; `threshold=0.0` |
| HF Grounding DINO / OWLv2 | grounded post-process returns text labels | xyxy abs | strings | phrase → class lookup; low threshold; top-100 |

Ultralytics' own `val(save_json=True)` uses faster-coco-eval. Results should match pycocotools closely,
but use one scorer for the report and mention the cross-check.

## 6. Latency and memory, measured the same way everywhere

Protocol (one function wrapping every adapter's `predict(image)`):
1. Fixed image set (e.g. 200 KITTI val images), batch 1, FP32, model in `eval()`, `torch.inference_mode()`.
2. Warm up 10–20 iterations (MPS compiles kernels on first use; CUDA picks cuDNN algos).
3. Time each call with `time.perf_counter()` bracketed by a device sync:
   `torch.cuda.synchronize()` on CUDA, `torch.mps.synchronize()` on MPS
   (https://docs.pytorch.org/docs/2.14/mps.html). Report mean, p50, p90 and FPS = 1000/mean.
4. Report two numbers when you can: model forward only, and end-to-end (preprocess + forward +
   postprocess/NMS). Ultralytics `Results.speed` already splits pre/inference/post; for HF and
   torchvision, time the processor, the model call and the post-process separately.
5. Log device, OS, torch/torchvision/ultralytics/transformers versions, input resolution and
   thresholds into each result JSON.

Memory:
- CUDA: `torch.cuda.reset_peak_memory_stats()` before the run, then `torch.cuda.max_memory_allocated()`
  (tensors) and `torch.cuda.max_memory_reserved()` (allocator cache). NVML/`nvidia-smi` adds the
  context (~300–500 MB, UNVERIFIED), so say which one you report.
- MPS: there is no peak-memory API. `torch.mps` offers `current_allocated_memory()` (tensors),
  `driver_allocated_memory()` (everything the Metal driver gave the process) and
  `recommended_max_memory()` (https://docs.pytorch.org/docs/2.14/mps.html). Sample
  `driver_allocated_memory()` after every forward and keep the max; call `torch.mps.empty_cache()` between
  models. On Apple silicon this is unified memory shared with the CPU, not dedicated VRAM; say so in the report.
- CPU: peak RSS via `resource.getrusage(RUSAGE_SELF).ru_maxrss` or `psutil`, measured in a fresh process per model.
- Run each model in its own process so allocator caches don't carry over between models.

The tutorial's `latency.py` does warmup, cuda sync and 30 timed iterations end-to-end, with no memory
and no MPS sync. Extend that pattern rather than inventing a different one.

## 7. Per-model gotcha checklist (what to log for reproducibility)

| Config | Resolution | Score thr for mAP | NMS | Top-k | Known traps |
|---|---|---|---|---|---|
| YOLO11m | letterbox 640 (`rect` val 672 pad) | 0.001 | IoU 0.7, class-aware | 300 | predict default conf 0.25 |
| YOLO26m | 640 | 0.001 | default head = NMS; `nms=False` = e2e one-to-one | 300 | say which head |
| Faster R-CNN v2 | short side 800, max 1333 | 0.05 (default) or 0.001 | IoU 0.5 per class | 100 | slow on CPU |
| FCOS | 800/1333 | set 0.05 (default 0.2!) | IoU 0.6 | 100 | default threshold |
| RT-DETR R50 | fixed 640×640, aspect not kept | 0.0 | none | 300 | squashes KITTI aspect |
| Grounding DINO-T | processor default (shortest edge 800, UNVERIFIED) | ~0.05 box, text_thr 0.25 | none | 900 queries → top-100 | prompt string, token limit, phrase→class mapping |
| YOLO-World v2 | 640 | 0.001 | IoU 0.7 | 300 | `set_classes` before predict; CLIP download |
