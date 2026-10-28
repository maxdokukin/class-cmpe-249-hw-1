**T1a. Setup**

| Model | Family | Checkpoint | Input resolution | Score thr. | NMS IoU | Device |
|---|---|---|---|---|---|---|
| YOLO11m | one-stage CNN | yolo11m.pt | letterbox, long side 640, minimal stride-32 padding (predict-style, rect) | 0.001 | 0.7 | MPS |
| YOLO11m @1280 | one-stage CNN | yolo11m.pt | letterbox, long side 1280, minimal stride-32 padding (predict-style, rect) | 0.001 | 0.7 | MPS |
| YOLO26m | one-stage CNN | yolo26m.pt | letterbox, long side 640, minimal stride-32 padding (predict-style, rect) | 0.001 | none (end-to-end one-to-one head, nms=False) | MPS |
| YOLO26m (NMS head) | one-stage CNN | yolo26m.pt | letterbox, long side 640, minimal stride-32 padding (predict-style, rect) | 0.001 | 0.7 | MPS |
| Faster R-CNN v2 | two-stage | FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 | GeneralizedRCNNTransform resize short side 800, long side cap 1333, keep aspect | 0.001 | 0.5 | MPS |
| RT-DETR R50 | DETR / query-based | rtdetr_r50vd | fixed 640x640 resize, aspect not kept | 0.0 | none | MPS |
| OWLv2 | open-vocabulary | owlv2-base-patch16-ensemble | pad to square (gray), resize 960x960 | 0.001 | none | MPS |
| Grounding DINO-T | open-vocabulary | grounding-dino-tiny | processor default (short side 800, long side cap 1333, keep aspect) | 0.001 | none | CPU* |

**T1b. Accuracy, COCO val2017 (AP, AR in [0,1])**

| Model | n images | mAP | AP50 | AP75 | APs | APm | APl | AR@100 |
|---|---|---|---|---|---|---|---|---|
| YOLO11m | 5000 | 0.505 | 0.670 | 0.549 | 0.325 | 0.559 | 0.664 | 0.650 |
| YOLO11m @1280 |  |  |  |  |  |  |  |  |
| YOLO26m | 5000 | 0.520 | 0.690 | 0.568 | 0.354 | 0.566 | 0.682 | 0.698 |
| YOLO26m (NMS head) | 5000 | 0.521 | 0.693 | 0.567 | 0.354 | 0.570 | 0.681 | 0.674 |
| Faster R-CNN v2 | 5000 | 0.469 | 0.679 | 0.514 | 0.310 | 0.508 | 0.604 | 0.630 |
| RT-DETR R50 | 5000 | 0.529 | 0.709 | 0.575 | 0.345 | 0.576 | 0.700 | 0.693 |
| OWLv2 (1k) | 1000 | 0.498 | 0.696 | 0.553 | 0.356 | 0.527 | 0.654 | 0.681 |
| Grounding DINO-T (1k) | 1000 | 0.509 | 0.668 | 0.559 | 0.375 | 0.515 | 0.638 | 0.682 |

**T1c. Accuracy, KITTI val-3769 (mAP with 95% bootstrap CI)**

| Model | mAP [95% CI] | AP50 | AP75 | APs | APm | APl | AR@100 |
|---|---|---|---|---|---|---|---|
| YOLO11m | 0.217 [0.208, 0.229] | 0.440 | 0.190 | 0.146 | 0.208 | 0.288 | 0.365 |
| YOLO11m @1280 | 0.237 [0.229, 0.250] | 0.491 | 0.201 | 0.144 | 0.242 | 0.288 | 0.394 |
| YOLO26m | 0.232 [0.224, 0.244] | 0.481 | 0.198 | 0.169 | 0.226 | 0.295 | 0.439 |
| YOLO26m (NMS head) | 0.223 [0.215, 0.234] | 0.456 | 0.193 | 0.132 | 0.211 | 0.307 | 0.350 |
| Faster R-CNN v2 | 0.239 [0.231, 0.251] | 0.493 | 0.200 | 0.154 | 0.249 | 0.287 | 0.457 |
| RT-DETR R50 | 0.241 [0.232, 0.253] | 0.480 | 0.206 | 0.180 | 0.238 | 0.307 | 0.454 |
| OWLv2 | 0.195 [0.186, 0.207] | 0.412 | 0.159 | 0.130 | 0.207 | 0.275 | 0.379 |
| Grounding DINO-T | 0.232 [0.223, 0.245] | 0.421 | 0.218 | 0.244 | 0.243 | 0.269 | 0.477 |

**T1d. Speed and memory (KITTI runs, batch 1, pre + forward + post)**

| Model | burst ms | burst FPS | sustained ms | sustained FPS | peak mem MB (KITTI run) | peak mem MB (COCO run) | memory kind |
|---|---|---|---|---|---|---|---|
| YOLO11m | 20.1 | 49.9 | 20.3 | 49.3 | 161.8 | 1155.9 | MPS unified-memory allocator peak |
| YOLO11m @1280 | 58.7 | 17.0 | 63.9 | 15.6 | 1145.8 | - | MPS unified-memory allocator peak |
| YOLO26m | 19.8 | 50.4 | 20.0 | 50.1 | 153.8 | 1156.0 | MPS unified-memory allocator peak |
| YOLO26m (NMS head) | 20.4 | 49.1 | 20.5 | 48.7 | 153.8 | 1156.0 | MPS unified-memory allocator peak |
| Faster R-CNN v2 | 267.5 | 3.7 | 337.3 | 3.0 | 1217.7 | 2250.0 | MPS unified-memory allocator peak |
| RT-DETR R50 | 115.6 | 8.6 | 187.1 | 5.3 | 1248.8 | 1248.8 | MPS unified-memory allocator peak |
| OWLv2 | 634.5 | 1.6 | 865.9 | 1.2 | 1084.7 | 1084.7 | MPS unified-memory allocator peak |
| Grounding DINO-T (CPU*) | 1379.3 | 0.7 | 1394.6 | 0.7 | 2438.1 | 3426.7 | CPU process RSS peak |

- Burst = median of the first 200 timed images; sustained = median over the whole run (after 10 warm-up images). FPS = 1000 / median ms.
- * Grounding DINO-T was measured on the M4 CPU (an interrupted parity check), so its latency is not comparable to the MPS runs. Its memory is process RSS, not an allocator peak.
- OWLv2 and Grounding DINO-T COCO numbers are on a seeded 1000-image val2017 subset (the others use all 5000). They also trained on Visual Genome, which overlaps COCO, so COCO is not clean zero-shot for them (KITTI is).
- Memory on MPS is the unified-memory allocator peak (torch.mps.driver_allocated_memory), not CUDA memory; the CPU run reports process RSS.
- Peak MPS memory differs a lot between runs of the same model (for example YOLO11m 1156 MB on COCO, 162 MB on KITTI at 640, 1146 MB on KITTI at 1280); it tracks input size: Ultralytics letterboxes the wide KITTI frames to about 640x192 at imgsz 640, versus 640x640 for most COCO images and about 1280x384 at imgsz 1280, while RT-DETR and OWLv2 resize to fixed squares and use the same memory on both datasets. It is the allocator peak, so treat it as indicative, not as model size.
- KITTI mAP is COCO-style mAP@[.5:.95] over the 5 mapped classes (car, truck, pedestrian, cyclist, tram); COCO mAP is over 80 classes. YOLO11m @1280 has no COCO run.
- Score threshold 0.001 (0.0 for RT-DETR) and max 100 detections for every model; NMS IoU as each model ships. OWLv2 / Grounding DINO-T KITTI numbers use the default class vocabulary (headline variant cyc-prompt).
