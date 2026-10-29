| Model | Car@0.7 easy | Car@0.7 moderate | Car@0.7 hard | Ped@0.5 easy | Ped@0.5 moderate | Ped@0.5 hard | Cyc@0.5 easy | Cyc@0.5 moderate | Cyc@0.5 hard | COCO-style mAP easy | COCO-style mAP moderate | COCO-style mAP hard |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11m | 0.908 | 0.782 | 0.609 | 0.837 | 0.748 | 0.614 | 0.381 | 0.254 | 0.245 | 0.324 | 0.246 | 0.204 |
| YOLO11m @1280 | 0.900 | 0.787 | 0.613 | 0.874 | 0.798 | 0.653 | 0.555 | 0.430 | 0.409 | 0.345 | 0.270 | 0.223 |
| YOLO26m (NMS-free) | 0.929 | 0.801 | 0.616 | 0.844 | 0.756 | 0.622 | 0.473 | 0.338 | 0.326 | 0.327 | 0.265 | 0.218 |
| YOLO26m (NMS) | 0.921 | 0.792 | 0.616 | 0.850 | 0.762 | 0.625 | 0.446 | 0.303 | 0.291 | 0.321 | 0.250 | 0.207 |
| Faster R-CNN v2 | 0.930 | 0.810 | 0.613 | 0.865 | 0.783 | 0.647 | 0.478 | 0.389 | 0.372 | 0.356 | 0.280 | 0.231 |
| RT-DETR R50 | 0.929 | 0.810 | 0.643 | 0.838 | 0.754 | 0.619 | 0.467 | 0.367 | 0.345 | 0.342 | 0.274 | 0.227 |
| OWLv2 | 0.729 | 0.611 | 0.437 | 0.558 | 0.480 | 0.409 | 0.157 | 0.107 | 0.099 | 0.278 | 0.224 | 0.180 |
| Grounding DINO-T | 0.929 | 0.811 | 0.633 | 0.652 | 0.470 | 0.381 | 0.254 | 0.215 | 0.202 | 0.354 | 0.282 | 0.228 |

- KITTI-style AP uses COCO-style 101-point interpolation and is not comparable to the KITTI leaderboard.
- Easy / moderate / hard use the KITTI height, occlusion and truncation rules as ignore regions; each level includes the easier ones' objects as in KITTI.
- Last three columns: COCO-style mAP@[.5:.95] over the 5 classes with the same difficulty filtering.
- KITTI mAP is COCO-style mAP@[.5:.95] over the 5 mapped classes (car, truck, pedestrian, cyclist, tram); COCO mAP is over 80 classes.
