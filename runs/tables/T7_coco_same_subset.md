| Model | 1k: n images | 1k mAP | 1k AP50 | 1k APs | 1k APm | 1k APl | 5000: mAP | 5000: AP50 | Δ mAP (1k - 5000) |
|---|---|---|---|---|---|---|---|---|---|
| YOLO11m | 1000 | 0.544 | 0.716 | 0.375 | 0.577 | 0.694 | 0.505 | 0.670 | +0.039 |
| YOLO26m (NMS-free) | 1000 | 0.551 | 0.722 | 0.379 | 0.586 | 0.712 | 0.520 | 0.690 | +0.031 |
| Faster R-CNN v2 | 1000 | 0.494 | 0.706 | 0.344 | 0.510 | 0.627 | 0.469 | 0.679 | +0.025 |
| RT-DETR R50 | 1000 | 0.559 | 0.746 | 0.385 | 0.586 | 0.714 | 0.529 | 0.709 | +0.030 |
| OWLv2 | 1000 | 0.498 | 0.696 | 0.356 | 0.527 | 0.654 | n/a | n/a | n/a |
| Grounding DINO-T | 1000 | 0.509 | 0.668 | 0.375 | 0.515 | 0.638 | n/a | n/a | n/a |

- All six models scored on the same seeded 1000-image COCO val2017 subset (the closed-set models were re-scored on it from their full-val predictions). 5000 = the full val2017 numbers of T1; OWLv2 and Grounding DINO-T were run only on the 1k subset.
- Δ mAP shows how much the 1k subset shifts each closed-set model's mAP from the full set; compare models within the 1k columns, not across 1k and 5000.
- OWLv2's ensemble checkpoint was fine-tuned on LVIS, whose images are COCO images, and Grounding DINO's GoldG grounding data includes Visual Genome, whose images come from COCO, so neither model's COCO score is strictly zero-shot; KITTI is. COCO mAP is over 80 classes; one subset, one run each, no confidence intervals.
