# 04 Compute and environment

Research date: 2026-10-27. This file covers where the HW1 runs happen, how long they take, one reproducible environment,
and one way to measure latency and memory.
Revision 2: after the user's update, the **MacBook Air M4 (MPS) is the primary accelerator**. The Linux VM is for CPU development, the
Mac Pro 2013 is a verdict only, and the SJSU HPC is a fallback.
Claims I could not confirm from a primary source are marked **UNVERIFIED**.

## TL;DR

- **Main runs on the M4 Air with `device="mps"`.** Full COCO val2017 + KITTI val (~8.7k images) for all seven models should take about
  **6-10 h of total compute**, dominated by Grounding DINO and OWLv2 (~2-3 h each). That fits in two or three overnight sessions.
  All of it is an estimate (**UNVERIFIED**) until a 50-image calibration run.
- Every target op has an MPS path: torchvision `nms` and `roi_align` have had MPS kernels since torchvision 0.16. Even so, **set
  `PYTORCH_ENABLE_MPS_FALLBACK=1`, log any fallback warnings, and run an MPS-vs-CPU parity check** (same boxes on 20 images, same AP
  on a 500-image subset) before trusting MPS mAP.
- **"GPU memory" on MPS is not CUDA memory.** Report `torch.mps.driver_allocated_memory()` (peak sampled per image) and
  `current_allocated_memory()`, label them as unified-memory allocator numbers, and don't put them in the same column as CUDA `max_memory_allocated`.
- **The Air is fanless and throttles.** Use a fixed warmup, a cooldown between models, and report both **burst** and **sustained** latency.
- **This Linux VM has no AVX2/FMA** (Xeon E5-2697 v2, Ivy Bridge, 2013). Use it for smoke tests, 500-image CPU subsets and a CPU-latency
  column. Polars, which Ultralytics needs, requires the `rtcompat` build here.
- **Mac Pro 2013 (2x FirePro D700):** treat it as CPU-only and skip it. Its GPUs have no usable modern PyTorch path, and its CPU is the same Ivy Bridge
  generation as the VM, at best ~10-20 % faster.
- **Environment:** `uv`, Python **3.12**, one `pyproject.toml` + `uv.lock`, **torch 2.14 / torchvision 0.29**. macOS arm64 gets PyPI wheels (MPS
  built in), Linux gets the CPU index, and an optional `cu126` extra serves the HPC/CUDA.
- **HPC (fallback):** A100/H100/P100 nodes, the faculty sponsors access, there's no internet on compute nodes, and the default job time is 4 h. See section 6.

## 1. Primary: MacBook Air M4 (MPS)

### 1.1 Support by library

| Library / model | MPS status | Notes |
|---|---|---|
| Ultralytics YOLO11 n/s/m | Works: `model.predict(..., device="mps")`, `yolo val ... device=mps` | M3 Air data point: YOLO11n MPS **9.2 ms inference / 15.9 ms total** at 384x640, YOLO11m **23.3 ms / 36.6 ms**; CPU YOLO11n 25.8 ms ([yolo.hexdocs.pm M3 benchmark](https://yolo.hexdocs.pm/macbook_air_m3.html)). Ultralytics documents an MPS bug for **pose** models (not detection) and fixed MPS memory reporting in 8.3.90 (from search-result summaries, **UNVERIFIED**). Post-processing (NMS) is a large share of the total on MPS (5-12 ms in the M3 numbers), so report forward and end-to-end separately. |
| Ultralytics RT-DETR-L | Probably works (pure PyTorch; uses `F.grid_sample`, which has an MPS forward) | **UNVERIFIED** on MPS. Do the parity check. |
| torchvision Faster R-CNN R50-FPN | `nms`, `roi_align`, `roi_pool`, `ps_roi_*` have **MPS kernels since torchvision 0.16** ([0.16 release notes](https://newreleases.io/project/pypi/torchvision/release/0.16.0); kernel source [roi_align_kernel.mm](https://sources.debian.org/src/pytorch-vision/0.21.0-3/torchvision/csrc/ops/mps/roi_align_kernel.mm)) | `roi_align` backward doesn't support fp16 on MPS, which doesn't matter for inference. `MultiScaleRoIAlign` + `batched_nms` should stay on the GPU. |
| HF RT-DETR / RT-DETRv2 / D-FINE | Pure-PyTorch multi-scale deformable attention (`grid_sample`) | **UNVERIFIED**. Expect it to work, possibly with CPU fallbacks for some ops. Run with fallback on and grep the warnings. |
| HF Grounding DINO-T | Pure-PyTorch deformable attention path (custom CUDA kernels are optional; HF falls back to `grid_sample`) | **UNVERIFIED** on MPS. This is the most likely place to hit fallbacks or numerical drift. Parity check required. |
| HF OWLv2-B/16 | Plain ViT + CLIP text tower | Should be the easiest transformer on MPS (**UNVERIFIED**). 960x960 = 3600 tokens, so attention memory is ~a few GB. |

General MPS rules ([PyTorch MPS docs](https://docs.pytorch.org/docs/2.14/mps.html), [HF Apple silicon guide](https://huggingface.co/docs/transformers/main/perf_train_special)):
- `export PYTORCH_ENABLE_MPS_FALLBACK=1` runs unsupported ops on the CPU (with a one-time warning per op). Fallbacks add copies and
  distort latency, so **log them and mention them in the report** (e.g. "Grounding DINO: N ops fall back to the CPU").
- Move both the model and every input tensor to `mps`. There's no `device_map` offload, because the whole model sits in unified memory.
- FP32 is the main setting. FP16 (`model.half()` or `torch.autocast("mps", dtype=torch.float16)`) is a secondary column, only if mAP stays within ~0.3 AP.
- Known PyTorch-MPS correctness bugs have hit batched matmul and clamp in the past
  ([pytorch#122123](https://github.com/pytorch/pytorch/issues/122123) and others). This is the reason for the parity check.

### 1.2 CoreML as an alternative latency number

- Ultralytics exports YOLO11 to CoreML (`yolo export model=yolo11n.pt format=coreml nms=True`) and can run `yolo val` on the
  `.mlpackage` ([Ultralytics CoreML docs](https://docs.ultralytics.com/integrations/coreml/)). CoreML can use the Neural Engine + GPU
  and is usually faster than PyTorch MPS. The M3 example ran YOLO11n at 15.6 ms CoreML inference vs 9.2 ms MPS at the same size, so
  the gain is not guaranteed. **Measure it.**
- RT-DETR to CoreML through Ultralytics: **UNVERIFIED**. The HF models (Grounding DINO, OWLv2) would need a hand conversion with `coremltools`,
  which is out of scope.
- Verdict: an **optional extra "deployment" column for the YOLO family only**. The main table stays PyTorch-MPS for all models, so
  the comparison runs on one runtime. CoreML memory can't be measured the same way, so only report latency for it.

### 1.3 Estimated M4 wall clock (full COCO val 5000 + KITTI val ~3.7k, about 8.7k images)

Assumptions (**all UNVERIFIED**): M4 Air with a 10-core GPU (~4 TFLOP/s FP32 peak; the base 8-core-GPU model is ~20 % slower), PyTorch MPS FP32, batch 1,
end-to-end including pre/post but with JPEG decode overlapped in a DataLoader, ~25 % sustained throttling penalty. Scaled from the M3 Air
YOLO11 numbers above and from FLOPs. KITTI images (375x1242) cost about the same as COCO for letterboxed 640 models and less for the
800/1333-resize models.

| Model | Input | Est. ms/img (burst) | Full COCO+KITTI (sustained) |
|---|---|---|---|
| YOLO11n | 640 | ~15-25 | ~5 min |
| YOLO11s | 640 | ~20-30 | ~6 min |
| YOLO11m | 640 | ~40-60 | ~10 min |
| RT-DETR-L (or D-FINE-L) | 640 | ~60-100 | ~15 min |
| Faster R-CNN R50-FPN | 800/1333 | ~200-400 | ~45-75 min |
| Grounding DINO-T | 800/1333 | ~500-1000 (x2 if the 80-class COCO prompt must be split into chunks) | ~1.5-3 h (up to ~5 h with chunks) |
| OWLv2-B/16 | 960 | ~600-1000 | ~1.5-3 h |
| **All seven** | | | **~6-10 h**, two or three overnight sessions |

Unified memory: with batch 1, none of these needs more than ~4-6 GB (**UNVERIFIED**), so a 16 GB Air is fine. Close other apps, because
memory pressure leads to swapping and latency spikes.

## 2. Measurement protocol (works on MPS, CUDA and CPU)

Put every model behind one wrapper, `predict(images) -> dets`, with three timed stages: `preprocess` (decoded image -> device tensor),
`forward` (model call only), `postprocess` (decode, NMS, rescale, threshold). JPEG decode and disk IO are done before the timer and
**excluded**.

1. **Setup:** `model.eval()`, `torch.inference_mode()`, batch = 1, fixed input resolution per model (log it), conf threshold logged
   (postprocess time depends on it; mAP runs use 0.001). Text prompts for GDINO/OWLv2 are tokenized once and cached, which is noted.
2. **Sync at every stage boundary:** `torch.mps.synchronize()` on MPS, `torch.cuda.synchronize()` on CUDA, nothing on CPU, then
   `time.perf_counter()`. MPS ops are asynchronous, so without the sync the forward time comes out near 0 and postprocess absorbs it.
   A helper `sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(dev, lambda: None)` keeps the code identical.
3. **Warmup:** 20 iterations on GPU/MPS (5 on CPU) with real images. MPS compiles Metal kernels on first use, so the first iterations
   are much slower.
4. **Burst latency:** after the warmup, 200 images (50 on the VM's CPU), the same fixed-seed list for every model. Report the median and p90 of forward
   and end-to-end ms. FPS = 1000 / median end-to-end.
5. **Sustained latency (the fanless Air):** loop the same image list for **10 minutes** and report the median over the **last 2 minutes**.
   Also log the burst/sustained ratio. Between models, take a **5-minute cooldown** (idle). Conditions to log: plugged in, Low Power Mode
   off, lid open, nothing else running, ambient temperature roughly constant. To log thermal state, `pmset -g therm` (CPU speed limit) or
   `sudo powermetrics --samplers smc,gpu_power -i 1000` during the run (**UNVERIFIED** exact sampler names on macOS 26).
   The full-dataset mAP runs are not latency runs. Throttling only changes their wall clock, not their accuracy.
6. **Memory:**
   - CUDA: `torch.cuda.reset_peak_memory_stats()` before the loop, then `max_memory_allocated()` (tensors) and `max_memory_reserved()`.
   - MPS: there is **no peak-memory API** ([torch.mps docs](https://docs.pytorch.org/docs/2.14/mps.html) list only
     `current_allocated_memory`, `driver_allocated_memory`, `recommended_max_memory`, `empty_cache`, `set_per_process_memory_fraction`).
     Call `torch.mps.empty_cache()` before the loop, then **sample both after every forward and keep the max**. Report
     "peak MPS tensor memory" (`current_allocated_memory`) and "peak Metal driver memory" (`driver_allocated_memory`, which includes allocator caching).
   - Unified-memory caveat: CPU and GPU share one RAM pool, so these numbers are allocator footprints, not dedicated VRAM, and CPU-fallback
     ops allocate outside them. Also log the process peak RSS (`resource.getrusage(RUSAGE_SELF).ru_maxrss`, which is in **bytes on macOS** and
     KB on Linux). In the report, put the MPS and CUDA memory in separate, clearly labelled columns.
   - CPU (VM): peak RSS, with a fresh process per model.
7. **Precision:** the main table is FP32 everywhere. FP16 is a secondary column on MPS/CUDA. Never use FP16 on the VM's CPU.
8. **Threads:** CPU runs use a fixed `torch.set_num_threads(N)`, logged (VM: 10 vs 20 tested once; the M4 needs no setting).
9. **Isolation and logging:** one model per process. Log the device name, OS, torch/torchvision/transformers/ultralytics versions,
   `uv.lock` hash, any MPS fallback warnings, and burst + sustained numbers into every result JSON.
10. **Library-specific hooks:** for Ultralytics, time your own wrapper and keep `results[0].speed` only as a cross-check. RT-DETR has no NMS.
   For torchvision, the model's `GeneralizedRCNNTransform` (resize + normalize) runs inside `forward`; note it or move it. For HF,
   `processor(...)` = pre, `model(**inputs)` = forward, `processor.post_process_grounded_object_detection` / `post_process_object_detection` = post.

## 3. Secondary: this Linux VM (CPU only; verified with `lscpu`, `which`)

| Item | Value |
|---|---|
| CPU | Intel Xeon E5-2697 v2 @ 2.70 GHz (turbo 3.5), Ivy Bridge-EP, 12 cores / 24 threads on the host |
| vCPUs | 24 listed, 20 online; 1 NUMA node, 30 MB L3 |
| ISA | `avx`, `f16c`, `sse4_2`. **No `avx2`, `fma`, `avx512*`, `avx_vnni` or `amx*`** |
| RAM | 16 GB + 4 GB swap |
| Tools | system Python 3.13.5, `pip` present; **no `uv`, no `conda`** |

- Peak is ~0.5 TFLOP/s FP32 (AVX without FMA). Expect ~100-150 GFLOP/s effective (**UNVERIFIED**). That makes it 3-5x slower per core than
  the Cascade Lake CPU behind the Ultralytics CPU-ONNX numbers ([Ultralytics README](https://github.com/ultralytics/ultralytics): AWS EC2 P4d).
- Hyperthreading adds almost nothing. Test 10 vs 20 threads once.
- There's no bf16/VNNI/AMX, so CPU quantization isn't worth the effort. OpenVINO generally beats ONNX Runtime, which beats PyTorch eager on Intel
  ([arXiv 2607.11356](https://arxiv.org/pdf/2607.11356)), but expect at most ~1.3-2x on this old ISA (**UNVERIFIED**). Stay in PyTorch eager.
- **Polars:** standard wheels require AVX2/FMA/BMI and fail on this CPU. Use `polars[rtcompat]` or `polars-lts-cpu`
  ([polars install docs](https://cocalc.com/github/pola-rs/polars/blob/main/docs/source/user-guide/installation.md),
  [polars-lts-cpu](https://pypi.org/project/polars-lts-cpu/1.8.1)). Ultralytics lists `polars>=0.20.0` as a core dependency
  ([pyproject](https://raw.githubusercontent.com/ultralytics/ultralytics/main/pyproject.toml)). Any other wheel built for
  x86-64-v3 would die with "Illegal instruction", so the smoke test must import everything.

Estimated CPU cost on the VM (**UNVERIFIED**, PyTorch FP32, 10 threads; FLOPs from the
[YOLO11 docs](https://docs.ultralytics.com/models/yolo11/), [RT-DETR paper](https://arxiv.org/abs/2304.08069),
[torchvision card](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html);
Grounding DINO-T reaches only 1.5 FPS even on a V100 per [YOLO-World Table 2](https://arxiv.org/html/2401.17270)):

| Model | GFLOPs | s/img | Full 8.7k | 500-image subset |
|---|---|---|---|---|
| YOLO11n | 6.5 | ~0.15 | ~22 min | ~1.5 min |
| YOLO11s | 21.6 | ~0.3 | ~45 min | ~2.5 min |
| YOLO11m | 68 | ~0.8 | ~2 h | ~7 min |
| RT-DETR-L | 110 | ~1.2 | ~3 h | ~10 min |
| Faster R-CNN R50-FPN | 134 | ~3 (COCO), ~1.8 (KITTI) | ~6 h | ~25 min |
| Grounding DINO-T | ~460 (**UNVERIFIED**) | ~8 | ~20-40 h | ~1-1.5 h |
| OWLv2-B/16 @960 | ~800 (own arithmetic) | ~10 | ~24 h | ~1.5 h |

Role: smoke tests (5 images), 500-image subsets during development while the Mac runs the full sets, and an optional "old x86 CPU"
latency column labelled with the CPU model and the missing AVX2. Running the full COCO+KITTI set for YOLO11n on both the VM and the M4 gives a cheap cross-device mAP
consistency check.

## 4. Mac Pro 2013 (2x FirePro D700): verdict

**Treat it as CPU-only, and probably don't use it at all.** The D700s are GCN 1.0 (Tahiti XT, 6 GB each) GPUs with no CUDA. AMD's ROCm never
supported macOS or GCN 1.0. PyTorch dropped Intel-macOS wheels after **2.2.2**, and that version is too old for current transformers
(which needs a newer torch, **UNVERIFIED** exact floor) and recent Ultralytics features. PyTorch MPS nominally supported AMD GPUs on macOS 12.3+, and
the Late-2013 Mac Pro tops out at Monterey 12.x. That makes "torch 2.2 + MPS on a GCN 1.0 GPU" theoretically possible, but it is
untested, would need a separate, frozen dependency stack, and the GPU's limited Metal feature support makes op gaps likely (**UNVERIFIED**).
OpenCL (OpenCV DNN / old PlaidML) is a dead end for these models. On the CPU side, the Mac Pro's top configuration is the **same Xeon E5-2697 v2**
(12 cores, Ivy Bridge, AVX only) that the VM runs on. With all 12 cores and 24 threads unvirtualized it would be at most ~10-20 % faster than
the 20-vCPU VM (**UNVERIFIED**; lower-end Mac Pro configs with 6 or 8 cores are slower). It would also need an Intel-macOS Python stack that
the lockfile doesn't cover. The only sensible use is as a second CPU worker running a different 500-image shard, and even that likely isn't worth the
setup time.

## 5. Environment: one `uv` spec for macOS arm64 (MPS), Linux CPU, and optionally CUDA

- **uv over conda:** a single user-space binary (`curl -LsSf https://astral.sh/uv/install.sh | sh`), a cross-platform `uv.lock`, and it
  downloads a standalone Python 3.12, so the system 3.13 doesn't matter. `.python-version` contains `3.12`.
- **Python 3.12:** pycocotools, tidecv, research repos (D-FINE) and coremltools are safest on 3.12. Avoid mmcv/mmdet
  (no prebuilt wheels for recent torch).
- **torch 2.14 / torchvision 0.29.** macOS arm64 wheels on PyPI include MPS. Linux uses the CPU index. 2.14 is also the **last release with cu126
  wheels**, the only CUDA build that still covers the HPC's P100/V100 (2.15+ is cu13x-only, sm_75+)
  ([PyTorch notice, 2026-09-04](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)).
- Pattern from the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/):

```toml
[project]
name = "hw1"
requires-python = ">=3.12,<3.13"
dependencies = [
  "ultralytics>=8.3",
  "polars[rtcompat]",                     # needed on the AVX-only VM; harmless elsewhere
  "transformers", "accelerate", "timm",   # pin exact versions after the smoke test
  "pycocotools", "numpy", "opencv-python-headless", "pyyaml", "pandas", "matplotlib",
]

[project.optional-dependencies]
cpu    = ["torch==2.14.*", "torchvision==0.29.*"]   # macOS arm64 (MPS) and Linux CPU
cu126  = ["torch==2.14.*", "torchvision==0.29.*"]   # HPC / any CUDA box
coreml = ["coremltools; sys_platform == 'darwin'"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu126" }]]

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu",   extra = "cpu",   marker = "sys_platform == 'linux'" },
  { index = "pytorch-cu126", extra = "cu126", marker = "sys_platform == 'linux'" },
]   # on darwin no source matches, so the default PyPI arm64 wheel (with MPS) is used
torchvision = [
  { index = "pytorch-cpu",   extra = "cpu",   marker = "sys_platform == 'linux'" },
  { index = "pytorch-cu126", extra = "cu126", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

- Commands: M4 `uv sync --extra cpu --extra coreml`; VM `uv sync --extra cpu`; HPC `uv sync --extra cu126 --frozen`. One committed
  `uv.lock`. Lock resolution of the extras together with the markers is **UNVERIFIED** until `uv lock` runs. If it fails, drop the markers and add a third
  `mac` extra with no source override.
- Device selection in code: `"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`, overridable from
  the YAML config.
- Smoke test on each machine: `uv run python -c "import torch, torchvision, polars, ultralytics, transformers; print(torch.__version__, torch.backends.mps.is_available(), torch.cuda.is_available())"`, then 5 images per model.

## 6. Fallback: SJSU CoE HPC (brief)

Sources: [official page](https://www.sjsu.edu/cmpe/resources/hpc.php) (updated Sep 4, 2026) and K. Liu's tutorials
([1](https://deepdatamininglearning.readthedocs.io/en/latest/HPC2.html), [2](https://mytutorial-lkk.readthedocs.io/en/latest/HPC2.html); 2022-24, possibly stale).

- **Access:** `ssh <SJSUID>@coe-hpc1.sjsu.edu`, only from the campus network or VPN. **A CoE faculty member must request the account**
  ([form](https://docs.google.com/forms/d/e/1FAIpQLSd2-RFOPvpgcbrj-kJwLIrZGKx2CBI2BG6Wo4tS8HsRg6va-A/viewform)). Class accounts last up to 6 months.
- **GPUs (Slurm):** H100 `g[2,6]`, A100 `g[7,13]`, P100 `g[1,3-5,9,11-12,14-16]`; 7-day limit; **default job time is 4 h**. Tutorials use
  `-p gpu --gres=gpu`. Confirm the partition and gres names with `sinfo -o "%P %l %D %N %G"`. A separate COE-HPC3 (`g17`) is reached from coe-hpc1.
- **Storage:** home quota was 20 GB per the tutorial (**UNVERIFIED** now). Put the venv, `UV_CACHE_DIR`, `HF_HOME` and the data in `/data/<course>` or `/scratch/<course>`.
- **Network:** compute nodes have **no internet**. The tutorial lists a proxy, `http://172.16.1.2:3128` (**UNVERIFIED**). Sync and download on the login node, then run
  jobs with `HF_HUB_OFFLINE=1`.
- **Modules:** Lmod (`module avail`). No CUDA module is needed with pip wheels, but the driver must support CUDA 12.6 (check `nvidia-smi`).
- Minimal job:

```bash
#!/bin/bash
#SBATCH --job-name=hw1-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # or --nodelist=g7 (A100)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /data/<course-or-user>/hw1
export HF_HOME=$PWD/.cache/hf TORCH_HOME=$PWD/.cache/torch HF_HUB_OFFLINE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,driver_version --format=csv
.venv/bin/python -m hw1.eval --model "$MODEL" --data "$DATA" --cfg "$CFG"
```

## Sources

- PyTorch MPS API: https://docs.pytorch.org/docs/2.14/mps.html
- HF Apple silicon notes: https://huggingface.co/docs/transformers/main/perf_train_special
- torchvision 0.16 (MPS nms/roi ops): https://newreleases.io/project/pypi/torchvision/release/0.16.0 ; https://sources.debian.org/src/pytorch-vision/0.21.0-3/torchvision/csrc/ops/mps/roi_align_kernel.mm
- YOLO11 on M3 Air (MPS/CPU/CoreML): https://yolo.hexdocs.pm/macbook_air_m3.html
- Ultralytics CoreML: https://docs.ultralytics.com/integrations/coreml/ ; YOLO11 table: https://docs.ultralytics.com/models/yolo11/ ; README: https://github.com/ultralytics/ultralytics ; deps: https://raw.githubusercontent.com/ultralytics/ultralytics/main/pyproject.toml
- PyTorch MPS matmul issue: https://github.com/pytorch/pytorch/issues/122123
- PyTorch cu126 / Pascal-Volta notice: https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432
- uv + PyTorch: https://docs.astral.sh/uv/guides/integration/pytorch/
- torchvision Faster R-CNN card: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
- RT-DETR: https://arxiv.org/abs/2304.08069 ; YOLO-World (GDINO-T FPS): https://arxiv.org/html/2401.17270
- CPU runtimes comparison: https://arxiv.org/pdf/2607.11356
- Polars on old CPUs: https://cocalc.com/github/pola-rs/polars/blob/main/docs/source/user-guide/installation.md ; https://pypi.org/project/polars-lts-cpu/1.8.1
- SJSU HPC: https://www.sjsu.edu/cmpe/resources/hpc.php ; https://deepdatamininglearning.readthedocs.io/en/latest/HPC2.html ; https://mytutorial-lkk.readthedocs.io/en/latest/HPC2.html
