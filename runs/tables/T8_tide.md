| Model | AP50 | Cls ΔAP50 | Loc ΔAP50 | Both ΔAP50 | Dupe ΔAP50 | Bkg ΔAP50 | Missed ΔAP50 | Cyclist AP50 | Cyclist Cls ΔAP50 | Cyclist Loc ΔAP50 | Cyclist Both ΔAP50 | Cyclist Dupe ΔAP50 | Cyclist Bkg ΔAP50 | Cyclist Missed ΔAP50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11m | 0.440 | 0.115 | 0.073 | 0.008 | 0.001 | 0.020 | 0.231 | 0.235 | 0.027 | 0.055 | 0.029 | 0.002 | 0.008 | 0.416 |
| YOLO11m @1280 | 0.491 | 0.083 | 0.090 | 0.011 | 0.001 | 0.019 | 0.214 | 0.393 | 0.051 | 0.087 | 0.042 | 0.003 | 0.010 | 0.309 |
| YOLO26m (NMS-free) | 0.481 | 0.102 | 0.078 | 0.010 | 0.005 | 0.021 | 0.180 | 0.335 | 0.041 | 0.070 | 0.042 | 0.007 | 0.016 | 0.277 |
| YOLO26m (NMS) | 0.456 | 0.078 | 0.085 | 0.007 | 0.001 | 0.017 | 0.272 | 0.291 | 0.029 | 0.063 | 0.029 | 0.002 | 0.012 | 0.404 |
| Faster R-CNN v2 | 0.493 | 0.069 | 0.090 | 0.011 | 0.001 | 0.035 | 0.149 | 0.358 | 0.021 | 0.092 | 0.039 | 0.001 | 0.033 | 0.262 |
| RT-DETR R50 | 0.480 | 0.113 | 0.077 | 0.011 | 0.002 | 0.020 | 0.151 | 0.340 | 0.050 | 0.056 | 0.050 | 0.002 | 0.020 | 0.210 |
| OWLv2 | 0.412 | 0.070 | 0.172 | 0.005 | 0.001 | 0.048 | 0.152 | 0.103 | 0.011 | 0.341 | 0.004 | 0.000 | 0.017 | 0.182 |
| Grounding DINO-T | 0.421 | 0.125 | 0.046 | 0.008 | 0.002 | 0.076 | 0.113 | 0.250 | 0.087 | 0.026 | 0.020 | 0.007 | 0.070 | 0.041 |

- TIDE-style oracles at IoU 0.5 on kitti_test (3769 images): ΔAP50 = AP50 after fixing only that error type minus AP50 (definitions in the docstring of src/hw1/failure.py). Overall = mean over the five KITTI classes. Oracles are applied one at a time, so columns do not add up to the AP gap.
- Cls = box matches an object of another class; Loc = poor localisation of the right class (IoU 0.1-0.5); Both = poor box and wrong class; Dupe = duplicate; Bkg = background; Missed = ground truth never detected.
- OWLv2 Cyclist Loc ΔAP50 = 0.341: its cyclist boxes sit on the right object with the wrong extent.
- One run per model; no confidence intervals. YOLO11m @1280 and YOLO26m (NMS) are the extra configurations of T1. KITTI-style AP uses COCO-style 101-point interpolation and is not comparable to the KITTI leaderboard.
