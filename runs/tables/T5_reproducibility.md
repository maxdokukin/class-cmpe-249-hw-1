| Model | Split | Tag | n images | Device | Host | torch | torchvision | ultralytics | transformers | Command | Wall time (s) | src hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11m | coco_val |  | 5000 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model yolo11m --split coco_val --device mps | 234.7 | 9a1b67aa7f53 |
| YOLO11m | kitti_test |  | 3769 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model yolo11m --split kitti_test --device mps | 107.7 | 9a1b67aa7f53 |
| YOLO11m | kitti_test | img1280 | 3769 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model yolo11m --split kitti_test --device mps --tag img1280 --set imgsz=1280 | 273.8 | 9a1b67aa7f53 |
| YOLO26m | coco_val |  | 5000 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model yolo26m --split coco_val --device mps | 301.7 | 9a1b67aa7f53 |
| YOLO26m | coco_val | nms | 5000 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model yolo26m --split coco_val --device mps --tag nms --set end2end=false | 235.3 | 9a1b67aa7f53 |
| YOLO26m | kitti_test |  | 3769 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model yolo26m --split kitti_test --device mps | 107.2 | 9a1b67aa7f53 |
| YOLO26m | kitti_test | nms | 3769 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model yolo26m --split kitti_test --device mps --tag nms --set end2end=false | 107.4 | 9a1b67aa7f53 |
| Faster R-CNN v2 | coco_val |  | 5000 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model frcnn_v2 --split coco_val --device mps | 2448.1 | 9a1b67aa7f53 |
| Faster R-CNN v2 | kitti_test |  | 3769 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model frcnn_v2 --split kitti_test --device mps | 1336.6 | 9a1b67aa7f53 |
| RT-DETR R50 | coco_val |  | 5000 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model rtdetr_r50 --split coco_val --device mps | 844.6 | 9a1b67aa7f53 |
| RT-DETR R50 | kitti_test |  | 3769 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model rtdetr_r50 --split kitti_test --device mps | 743.3 | 9a1b67aa7f53 |
| OWLv2 | coco_val1k |  | 1000 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model owlv2 --split coco_val1k --device mps | 800.5 | 9a1b67aa7f53 |
| OWLv2 | kitti_test |  | 3769 | mps | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model owlv2 --split kitti_test --device mps | 3337.8 | 9a1b67aa7f53 |
| Grounding DINO-T | coco_val1k |  | 1000 | cpu | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model gdino_t --split coco_val1k --device cpu | 2590.3 | 9a1b67aa7f53 |
| Grounding DINO-T | kitti_test |  | 3769 | cpu | Mac | 2.14.1 | 0.29.1 | 8.4.190 | 5.21.0 | python -m hw1.run --model gdino_t --split kitti_test --device cpu | 5486.9 | 9a1b67aa7f53 |

- One row per prediction run in runs/preds (dev-VM sanity runs and the MPS/CPU parity checks are excluded). Every run is deterministic given the command: fixed seeds for subsets, batch size 1.
- Wall time covers model load, inference and writing predictions; scoring is separate (python -m hw1.score).
- OWLv2 and Grounding DINO-T COCO numbers are on a seeded 1000-image val2017 subset (the others use all 5000). Grounding DINO-T was measured on the M4 CPU (an interrupted parity check), so its latency is not comparable to the MPS runs.
- Full command history: runs/COMMANDS.md.
