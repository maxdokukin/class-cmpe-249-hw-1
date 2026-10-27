#!/usr/bin/env bash
# One-time setup: Python environment, COCO val2017 + KITTI 2D downloads (~13.6 GB, resumable), GT build, tests.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

fetch() {  # fetch <url> <expected bytes>
  local f="data/downloads/$(basename "$1")"
  if [[ -f $f && $(wc -c < "$f" | tr -d ' ') == "$2" ]]; then echo "have $f"; return; fi
  echo "downloading $f"
  curl -fL --retry 5 -C - -o "$f" "$1" || curl -fL --retry 5 -o "$f" "$1"
}
mkdir -p data/downloads data/coco data/kitti/ImageSets
fetch http://images.cocodataset.org/zips/val2017.zip 815585330
fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip 252907541
fetch https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip 5601213
fetch https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip 12569945557

[[ $(ls data/coco/val2017 2>/dev/null | wc -l) -eq 5000 ]] || (cd data/coco && unzip -qo ../downloads/val2017.zip)
[[ -f data/coco/annotations/instances_val2017.json ]] || \
  (cd data/coco && unzip -qo ../downloads/annotations_trainval2017.zip annotations/instances_val2017.json)
[[ $(ls data/kitti/training/label_2 2>/dev/null | wc -l) -eq 7481 ]] || (cd data/kitti && unzip -qo ../downloads/data_object_label_2.zip)
[[ $(ls data/kitti/training/image_2 2>/dev/null | wc -l) -eq 7481 ]] || \
  (cd data/kitti && unzip -qo ../downloads/data_object_image_2.zip 'training/*')
for s in train val; do
  [[ -s data/kitti/ImageSets/$s.txt ]] || curl -fsL -o "data/kitti/ImageSets/$s.txt" \
    "https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/data/kitti/ImageSets/$s.txt"
done

uv run python - <<'PY'
import torch
from hw1.data import gt_summary, load_gt
for s in ("kitti_test", "kitti_dev"):
    g = load_gt(s)
    assert g["info"]["missing_images"] == 0, f"{s}: images missing"
    print(s, gt_summary(g))
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
PY
uv run pytest -q tests
echo "setup OK. Next: ./run.sh   (plugged in, lid open)"
