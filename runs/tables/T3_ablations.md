| Model | Headline KITTI mAP | Cyclist = bicycle only (naive): mAP | Cyclist = bicycle only (naive): Δ | Cyclist class dropped: mAP | Cyclist class dropped: Δ | Van as ignore region: mAP | Van as ignore region: Δ | Score >= 0.3 (KITTI): mAP | Score >= 0.3 (KITTI): Δ | Score >= 0.3 (COCO): mAP | Score >= 0.3 (COCO): Δ | Cyclist AP merge/prompt | Cyclist AP naive | Cyclist AP none | COCO headline mAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11m | 0.217 | 0.205 | -0.012 | 0.205 | -0.012 | 0.222 | +0.005 | 0.199 | -0.018 | 0.440 | -0.065 | 0.059 | 0.000 | 0.000 | 0.505 |
| YOLO11m @1280 | 0.237 | 0.216 | -0.021 | 0.216 | -0.022 | 0.243 | +0.006 | 0.221 | -0.016 |  |  | 0.108 | 0.001 | 0.000 |  |
| YOLO26m (NMS-free) | 0.232 | 0.215 | -0.017 | 0.215 | -0.017 | 0.236 | +0.004 | 0.196 | -0.035 | 0.438 | -0.082 | 0.084 | 0.000 | 0.000 | 0.520 |
| YOLO26m (NMS) | 0.223 | 0.208 | -0.015 | 0.208 | -0.015 | 0.228 | +0.005 | 0.208 | -0.015 | 0.455 | -0.066 | 0.074 | 0.000 | 0.000 | 0.521 |
| Faster R-CNN v2 | 0.239 | 0.219 | -0.020 | 0.218 | -0.020 | 0.243 | +0.004 | 0.223 | -0.016 | 0.454 | -0.016 | 0.101 | 0.001 | 0.000 | 0.469 |
| RT-DETR R50 | 0.241 | 0.222 | -0.019 | 0.222 | -0.019 | 0.245 | +0.004 | 0.230 | -0.011 | 0.503 | -0.026 | 0.095 | 0.001 | 0.000 | 0.529 |
| OWLv2 | 0.195 |  |  |  |  |  |  |  |  | 0.449 | -0.048 | 0.020 |  |  | 0.498 |
| Grounding DINO-T | 0.232 |  |  |  |  |  |  |  |  | 0.472 | -0.037 | 0.100 |  |  | 0.509 |

- Δ = variant mAP - headline mAP of the same model and split (KITTI columns relative to the KITTI headline, COCO column relative to the COCO headline). Blank = not run.
- Naive: only bicycle boxes count as Cyclist. None: no detection is mapped to Cyclist, so Cyclist AP is 0 while its GT is still scored. Van-ignore: Van GT becomes an ignore region instead of merging into car. Score >= 0.3: predictions below 0.3 removed before scoring (ngdet's default).
- The KITTI ablations were run for the closed-set models only; OWLv2 / Grounding DINO-T prompt Cyclist directly. KITTI mAP is COCO-style mAP@[.5:.95] over the 5 mapped classes (car, truck, pedestrian, cyclist, tram); COCO mAP is over 80 classes. OWLv2 and Grounding DINO-T COCO numbers are on a seeded 1000-image val2017 subset (the others use all 5000).
- 'Cyclist AP' columns are KITTI Cyclist AP@[.5:.95] under each treatment (headline merge, naive, none); naive and none give ~0 by construction (a COCO detector has no Cyclist class).
