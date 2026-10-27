# 02 Datasets, taxonomy mapping and evaluation protocol

Scope: Track A, COCO-pretrained 2D detectors evaluated zero-shot on driving data. Written 2026-10-27.
Compute note: the first brief said "dev box, no GPU, GPU runs on SJSU HPC"; the later update says the working machine is a MacBook Air M4 with a close deadline. This file therefore favors small downloads.
Marker: UNVERIFIED means I could not confirm it from a primary source in this session. "Checked" means I ran a command (HTTP HEAD, pip install, toy run) and saw the result.

## 0. Recommendation (read this first)

| Role | Dataset | Why | Download |
|---|---|---|---|
| Source domain (required) | COCO val2017 | in-domain reference, 5000 images | 815.6 MB images + 252.9 MB annotations (checked, HEAD) |
| Target 1 (required) | KITTI 2D object, 3769-image val split | the homework names it; covers lectures 11-12 | 12.57 GB left color images zip + 5.6 MB labels (checked, HEAD); no registration for the S3 mirror |
| Target 2 (recommended) | nuImages (val) | real human 2D boxes, 1600x900, rain/night/dense-urban content, different country and camera than KITTI | mini = 117.9 MB (checked, HEAD) for a smoke test; full samples = 16.4 GB (UNVERIFIED, from a Hugging Face mirror listing) |
| Optional 3 (shift study) | BDD100K val, night vs day split | strongest, cheapest "same sensor, only lighting changes" study; 3929 night val images | needs a free account on the BDD site, which was unreachable from my sandbox, so size UNVERIFIED (commonly quoted ~5.3 GB for the 100K images zip, UNVERIFIED) |

If time is short: COCO + KITTI + nuImages mini (118 MB) is a valid "3 domains" experiment. nuImages mini is only a few hundred images (count UNVERIFIED), so use it for the pipeline and use full val for the headline number if the 16.4 GB download is affordable. Waymo is not recommended: registration plus a Google-Cloud data license, large files, and the 2D labels are in parquet (v2). Cityscapes needs a registration approved by hand; ACDC is a segmentation-first dataset (see section 1).

KITTI trick for the 12.6 GB zip: the S3 host answers `Accept-Ranges: bytes` (checked), so a ranged reader (for example the `remotezip` package, UNVERIFIED on this host) could fetch only the 3769 val PNGs (about 6 GB) instead of the whole archive. The label zip is 5.6 MB and always fully needed.

## 1. Dataset facts

### 1.1 COCO val2017
- Files: `http://images.cocodataset.org/zips/val2017.zip` (815,585,330 bytes), `annotations_trainval2017.zip` (252,907,541 bytes). Checked by HEAD.
- 5000 images, 80 classes (91-id space with gaps), JSON in COCO format, boxes `[x,y,w,h]`, `iscrowd` flag. Images are mostly 640x480-ish. No registration. Annotations CC BY 4.0, images under Flickr terms (https://cocodataset.org/#termsofuse).
- Take only `instances_val2017.json` out of the annotations zip.

### 1.2 KITTI 2D object detection
- Source: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d (fetched). 7481 training images, 7518 test images (no public test labels; test needs a server submission). Images are PNG.
- Direct mirror (checked by HEAD): `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip` (12,569,945,557 bytes), `data_object_label_2.zip` (5,601,213 bytes), `devkit_object.zip` (63,794 bytes). The site itself lists left color images 12 GB, labels 5 MB, calib 16 MB. Calibration is not needed for 2D.
- Registration: the website asks you to log in for the page; the S3 URLs above answered 200 without a login. UNVERIFIED that this stays true.
- License: CC BY-NC-SA 3.0 (stated on the KITTI site; UNVERIFIED in this session).
- Resolution: about 1242x375 (varies slightly, read it from the image).
- Label format: one txt per image, one object per line: `type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry`. Types: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare. Boxes are `x1,y1,x2,y2` (xyxy), pixel coordinates; truncated is a float in [0,1]; occluded is 0 (visible), 1 (partly), 2 (largely), 3 (unknown).
- Common split: the 3DOP/MV3D split, 3712 train and 3769 val (many papers state this; for example https://arxiv.org/pdf/1812.04244 and others). The split files are in OpenPCDet at `data/kitti/ImageSets/{train,val}.txt`. I downloaded them: they contain 3711 and 3768 newline-terminated lines (3712/3769 entries when the last line has no trailing newline; treat as 3712/3769). Raw URL: https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/data/kitti/ImageSets/val.txt . The split avoids putting frames of one drive in both parts, so it is leakage-safe for fine-tuning. For zero-shot work the split only decides which images to score; use val (3769) so numbers are comparable with fine-tuned runs later.
- Official evaluation rules (fetched from the KITTI page): Easy = min box height 40 px, fully visible, truncation <= 15%. Moderate = min height 25 px, occlusion up to partly occluded, truncation <= 30%. Hard = min height 25 px, up to largely occluded, truncation <= 50%. IoU threshold 0.7 for Car, 0.5 for Pedestrian and Cyclist. Since Oct 2019 AP uses 40 recall positions instead of 11. Detections in DontCare regions or below the minimum size are not counted as false positives. (The occlusion levels per difficulty are from the devkit convention; the page text I fetched summarized them loosely, so treat the exact occlusion wording as UNVERIFIED and re-read `evaluate_object.cpp` in the devkit zip before quoting.)
- Converters to COCO: the repo `lkk688/DeepDataMiningLearning` has a KITTI loader with a COCO-style export (instructor code, see section 4); MMDetection has `tools/dataset_converters/kitti_converter` style scripts for 3D (UNVERIFIED for plain 2D). Writing our own 40-line converter is safer because we need custom ignore handling (section 3).

### 1.3 nuImages
- Official page: https://www.nuscenes.org/nuimages (reachable, but my fetch returned no text; facts below come from the paper summary https://arxiv.org/html/2512.02448 and the devkit).
- 93,000 annotated key frames (plus 6 past and 6 future unannotated frames at 2 Hz = 13-frame clips, about 1.2 M images in all) from about 500 logs. 75% chosen by active learning for rare classes (bicycles) and adverse conditions (rain, snow, night), 25% uniform. This means the set is deliberately harder than a uniform sample (relevant when comparing with KITTI).
- Splits: train 67,279, val 16,445, test 9,752 (one search result; another says val 14,772; UNVERIFIED which is right, count the JSON yourself).
- Resolution 1600x900 (cropped from 1600x1200). Labels: 2D boxes, instance masks and semantic masks. Object classes: 23 per the dataset summary (names from memory, UNVERIFIED: animal; human.pedestrian.{adult, child, construction_worker, personal_mobility, police_officer, stroller, wheelchair}; movable_object.{barrier, debris, pushable_pullable, trafficcone}; static_object.bicycle_rack; vehicle.{bicycle, bus.bendy, bus.rigid, car, construction, emergency.ambulance, emergency.police, motorcycle, trailer, truck}). Plus two surface classes, flat.driveable_surface and vehicle.ego.
- Format: nuScenes-style JSON tables in `v1.0-<split>/` (`object_ann.json` has `bbox` as xyxy pixels and `category_token`; `sample_data.json`; `category.json`; `attribute.json`). Attributes include cycle.with_rider / cycle.without_rider and vehicle.parked / moving (UNVERIFIED names).
- Download: free account on nuscenes.org, then links. Checked: `https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz` answered 200 with 117,929,607 bytes (the nuScenes cloudfront mirror gave 117,873,411). The URLs for `nuimages-v1.0-all-samples.tgz` and `...all-metadata.tgz` on nuscenes.org returned a 6574-byte page (a login or redirect page), so they need the account. A Hugging Face mirror listing shows samples at 16.4 GB (UNVERIFIED, third-party). Only metadata + samples are needed; sweeps are optional and large.
- License: same as nuScenes (CC BY-NC-SA 4.0; non-commercial). Fine for coursework. UNVERIFIED in this session because the terms page returned no text.
- Converters: MMDetection3D `tools/dataset_converters/nuimage_converter.py` writes COCO JSON; its name map (fetched) is: barrier, bicycle, bus (bendy+rigid), car, construction_vehicle, motorcycle, pedestrian (adult/child/construction_worker/police_officer), traffic_cone, trailer, truck. Note it drops personal_mobility, stroller, wheelchair, animal, debris, pushable_pullable, ambulance and police cars. Source: https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/tools/dataset_converters/nuimage_converter.py
- Rider rule: the nuScenes annotation guide says a bicycle or motorcycle box includes the rider and any passenger, and a person standing next to a bike is a separate pedestrian (https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md, fetched). nuImages shares the taxonomy, but I did not confirm the 2D box rule for nuImages itself (UNVERIFIED). If it holds, nuImages `bicycle` boxes are rider+bike boxes just like KITTI Cyclist.

### 1.4 BDD100K
- Official site `bdd-data.berkeley.edu` and docs `doc.bdd100k.com` did not resolve from my sandbox (DNS/connection failure, checked), so download details come from secondary sources: registration (free account) at https://bdd-data.berkeley.edu, then "100K Images" plus "Detection 2020 Labels" (https://www.github.com/SysCV/bdd100k-models/blob/main/doc/PREPARE_DATASET.md).
- Detection set: 70,000 train, 10,000 val images, 1280x720. License: BSD-3-Clause per a secondary source (UNVERIFIED; the paper is https://arxiv.org/abs/1805.04687).
- 10 detection classes (from memory, UNVERIFIED): pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign. Note `rider` is a separate class whose box is the person on the bike/motorcycle; the bike has its own box. This is the COCO-style convention (person + bicycle), the opposite of KITTI Cyclist.
- Attributes per image: weather, scene, timeofday (daytime / night / dawn-dusk). Val night subset: 3,929 images, day 5,258 (secondary source, https://arxiv.org/pdf/2112.02891 and others; UNVERIFIED).
- COCO converter: `python -m bdd100k.label.to_coco -m det -i bdd100k/labels/det_20/det_val.json -o det_val_cocofmt.json` (from the PREPARE_DATASET page above, fetched).
- Cost: the 100K image archive contains train/val/test, so the whole archive must be downloaded to get val; this is the main friction.

### 1.5 Waymo Open Dataset (2D camera)
- 1,000 segments, 11.8 M 2D boxes with tracking ids, classes Vehicle, Pedestrian, Cyclist, Sign (https://waymo.com/intl/en-ca/open/data/perception/ via search snippet). v2 stores labels in parquet files on Google Cloud Storage; access needs a Google account and acceptance of the Waymo license (non-commercial). Image sizes UNVERIFIED (about 1920x1280 front camera). The instructor harness has a loader (`dataset_waymov3_1.Waymo2DDataset`). Not recommended on a laptop with a deadline: registration, `gsutil`, large shards.

### 1.6 Cityscapes
- Needs a registered account approved manually (can take days); fine annotations are instance polygons, not boxes (detection boxes are tightest rectangles of instance masks, as done in https://arxiv.org/pdf/1803.03243). `leftImg8bit_trainvaltest.zip` is about 11 GB (UNVERIFIED). 2048x1024. Only worth it for matching the Cityscapes-to-KITTI literature. Skip.

### 1.7 Adverse-condition options
- ACDC: 4006 adverse images (1000-ish each of fog, night, rain, snow) plus 4006 normal-condition counterparts, annotations are panoptic/semantic pixel labels, 1920x1080; supports detection through instance masks (https://arxiv.org/html/2104.13395v5, search snippet). Registration at https://acdc.vision.ee.ethz.ch. Detection boxes would have to be derived from masks; labels for the test split are withheld. More work than BDD.
- BDD100K night/rain: cheapest because box labels exist and the sensor is identical, so a day vs night comparison isolates lighting. Recommended as optional third if the BDD account works.
- DAWN: small (about 1000 images, weather), COCO-style boxes; I did not verify download details. UNVERIFIED.

## 2. Evaluation protocol decisions

1. Score threshold: run every detector with a very low confidence threshold (0.001 for YOLO and Faster R-CNN, 0.0 to 0.01 for DETR-style) and max 100 detections per image. AP is computed over the full precision-recall curve; cutting at 0.3 truncates the curve and deflates AP. The instructor harness default is `--score-thr 0.3` (see section 4), which is a weakness for AP numbers.
2. NMS: Ultralytics default IoU 0.7 for val; Faster R-CNN torchvision box_nms_thresh 0.5; DETR-style: none. Record each in the config.
3. Report three taxonomies for transparency: (a) strict 3 class (car, person, truck) for the headline, (b) "vehicle / person / cyclist" coarse for comparison with the instructor harness, (c) the KITTI-native subset (Car, Pedestrian, Cyclist) with KITTI IoU 0.7/0.5 using custom iouThrs.
4. Input resolution: fix per model (640 for YOLO/RT-DETR, torchvision default min_size 800 for Faster R-CNN) and log it. KITTI is 1242x375 (wide), so letterboxing at 640 shrinks small objects heavily; add one 1280-wide run as an ablation.

## 3. Concrete label mappings

COCO category ids used: person 1, bicycle 2, car 3, motorcycle 4, bus 6, train 7, truck 8.

### 3.1 KITTI to COCO

| KITTI | COCO class | Treatment in GT | Reason |
|---|---|---|---|
| Car | car (3) | normal GT | direct |
| Van | car (3) | headline: normal car GT (merge). Strict KITTI-devkit variant: ignore region for car (iscrowd=1) | A COCO model usually calls a minivan "car" (and some vans "truck"); the devkit treats Van as a neighbor class that is neither required nor penalized |
| Truck | truck (8) | normal GT | direct |
| Tram | train (7) | normal GT, but report it separately; or ignore | few instances; COCO "train" is a rail vehicle, but a street tram is out of distribution; detectors often say bus or train |
| Pedestrian | person (1) | normal GT | direct |
| Person_sitting | person (1) | ignore region for person (devkit neighbor rule); alternative: merge | rare; sitting people are poses COCO covers well, but the devkit ignores them |
| Cyclist | see 3.1.1 | | KITTI box = rider + bike |
| Misc | none | ignore region for car and truck (iscrowd=1) or dropped | heterogeneous (trailers, etc.) |
| DontCare | all evaluated classes | one `iscrowd=1` annotation per evaluated category, same box | marks unlabeled far objects; any detection mostly inside it must not be a false positive |

Both the Van-as-car merge and the strict devkit variant should be run once on one model and the AP difference reported; the Adithya-Rajendran repo (https://github.com/Adithya-Rajendran/HW1-2D-Detection, fetched) says DontCare-as-ignore adds 0.5 to 1.1 AP to every model, which is a published comparison point for that choice.

#### 3.1.1 Cyclist (person + bicycle)
Facts: KITTI Cyclist is one box around rider and bike. COCO has `person` and `bicycle` as separate objects, and a COCO model will output two boxes. The rider's person box covers only the rider, so IoU with the Cyclist box is usually below 0.5; the bike box covers only the bike. The instructor harness reports cyclist AP about 0 for COCO models, which is mostly this geometry effect, not a detection failure.
Three options, in order of what I recommend:
1. Merged "cyclist" prediction: for each predicted `bicycle` box, find the predicted `person` box with the largest overlap (intersection over person area > 0.5 and person center above the bike box center); output the union box with score = min or mean of the two. Unmatched bicycles are dropped for the cyclist class; unmatched persons stay as pedestrians. Evaluate "cyclist" with the Cyclist GT. This mimics what a downstream KITTI pipeline would do and is the fairest test of whether the model sees the pair. Implementation is about 30 lines of numpy. Report both the merged and the naive result.
2. Naive: map predicted `bicycle` to cyclist and score against the Cyclist box (the harness approach); expect near-zero AP; use it as the "what a lazy mapping does" row.
3. Exclusion: treat Cyclist GT boxes as ignore regions for the `person` class (so rider detections are not false positives for pedestrian) and drop the cyclist class from the headline mAP. Do this regardless in the pedestrian evaluation; otherwise riders detected as `person` count as pedestrian false positives and pedestrian AP drops for the wrong reason.
Report cyclist separately and do not average it into a headline mAP without saying so.

### 3.2 KITTI evaluation details: difficulty vs COCO size buckets
- COCO buckets are by box area: small < 32^2 = 1024 px, medium 1024-9216, large > 9216. KITTI difficulty is by height (and truncation/occlusion), so they cut the data differently. A KITTI car of height 25 px (the Moderate minimum) with typical width about 2x height has an area near 1250 px, i.e. just medium. A pedestrian of height 25 px is about 12 px wide, area about 300, so small. Most KITTI Hard/Moderate pedestrians are COCO-small, while many KITTI cars are medium; report both views and do not assume Easy = large.
- Implementation: store `height`, `truncated`, `occluded` as extra keys on each GT annotation. Build three GT JSONs (easy, moderate, hard). In each, GT that falls outside the difficulty (height below minimum, truncation or occlusion above the limit) is kept but set `iscrowd=1` so detections matching it are ignored, exactly the devkit behavior. pycocotools ignores crowd GT and does not penalize unmatched detections that overlap them; it matches detections to non-ignored GT first (checked in pycocotools source, `cocoeval.py` lines 251-282).
- Minimum height: detections shorter than 25 px (or 40 for Easy) that match nothing are false positives in COCO but are ignored in the devkit. To mimic the devkit, drop predictions with height below the difficulty minimum before COCOeval (documented in the KITTI page: "detections below the minimum size thresholds don't count as false positives").
- Crowd matching semantic: pycocotools IoU for crowd GT is intersection over detection area, which matches the devkit rule for DontCare (detection is ignored if mostly inside the region).
- KITTI IoU: set `params.iouThrs` to a single 0.7 (Car) and 0.5 (Pedestrian, Cyclist) in separate runs for "KITTI-style AP"; COCO AP uses 0.50:0.95 with 101 recall points, KITTI uses 40 recall points. Numbers are not comparable with the KITTI leaderboard; say so in the report.

### 3.3 nuImages to COCO (recommended second target)

| nuImages | COCO class | Note |
|---|---|---|
| vehicle.car | car | |
| vehicle.truck | truck | |
| vehicle.bus.rigid / bus.bendy | bus | |
| vehicle.trailer | truck (or ignore) | a trailer alone has no COCO class |
| vehicle.construction | truck (or ignore) | |
| vehicle.emergency.* | car (or ignore) | rare; mmdet3d drops them |
| vehicle.motorcycle | motorcycle | |
| vehicle.bicycle | bicycle (and, if rider included, person+bicycle pair) | see 1.3 rider rule |
| human.pedestrian.adult / child / construction_worker / police_officer | person | |
| human.pedestrian.personal_mobility / stroller / wheelchair | person ignore region | the nuScenes guide includes the occupant in the box, so the box is not a clean person |
| movable_object.trafficcone / barrier / debris / pushable_pullable, static_object.bicycle_rack, animal | no class (traffic cone, barrier have no COCO class); drop | predictions of unrelated COCO classes (traffic light, dog, etc.) are filtered out before scoring |

Attributes: use `vehicle.parked` etc. only for failure analysis. No DontCare equivalent exists; heavily occluded instances are included, so recall is capped lower than on KITTI.

## 4. Comparison with the instructor harness (ngdet)

Files read: https://raw.githubusercontent.com/lkk688/DeepDataMiningLearning/main/DeepDataMiningLearning/ngdet/{taxonomy.py, datasets.py, evaluator.py, TUTORIAL.md}, and the KITTI loader `detection/dataset_kitti.py`.

What it does:
- Presets `driving3` (vehicle, person, cyclist) and `driving5` (car, truck, bus, person, cyclist). Mapping by lowercase name with synonym lists. `driving3` folds car, truck, bus, van, trailer, construction_vehicle, tram, train, motorcycle into `vehicle`; person, pedestrian, person_sitting into `person`; cyclist, bicycle, bike into `cyclist`. In `driving5`, motorcycle goes to cyclist.
- nuImages: parses `object_ann.json` directly, right-to-left token match (so vehicle.bicycle goes to cyclist via "bicycle").
- Evaluator: pycocotools `COCOeval` bbox on the unified classes, prints the 12 standard stats plus per-class AP.

Weaknesses relative to my proposal:
1. DontCare is dropped, not used as an ignore region (`dataset_kitti.py`: `if objecttype != 'DontCare' and ...`), and `evaluator.py` writes `iscrowd: 0` for every GT. So detections of unlabeled far objects count as false positives; the Adithya-Rajendran repo measured +0.5 to +1.1 AP from handling this (above).
2. Cyclist handling is the naive option 2: COCO `bicycle` is relabeled `cyclist`, rider `person` boxes are scored as pedestrians, no pairing, no ignore for riders. The tutorial itself reports cyclist AP about 0. The tutorial blames box geometry; that is consistent with my analysis, but it also means the 3-class mean mAP is dragged by a mapping artifact, and rider detections also create pedestrian false positives.
3. Van is merged into `vehicle` without an option to treat it as ignore; Tram, train and motorcycle also fold into one class, so a detector confusing car with truck or bus is invisible in `driving3`.
4. No KITTI difficulty filtering: no minimum height, truncation or occlusion limit, so tiny far objects count in full; no easy/moderate/hard breakdown. The harness `KittiDataset` does record `truncated` and `occluded`, so adding it is cheap.
5. Score threshold default 0.3 (`run_eval.py --score-thr`), which truncates the PR curve and deflates AP; sample sizes are small (120 images for COCO/KITTI/Waymo, 500 for nuImages) so per-class numbers are noisy. Our full-val runs (3769 KITTI, 5000 COCO) would be more reliable.
6. Split: if `ImageSets` is missing, `KittiDataset` creates a random 80/20 split with seed 42, which is not the 3712/3769 split and has temporal leakage between neighbors. Irrelevant zero-shot, but matters if we fine-tune.
7. nuImages: the right-to-left rule sends `vehicle.construction`, `vehicle.emergency.*`, and `human.pedestrian.stroller/wheelchair/personal_mobility` to vehicle or person without checking whether the box is clean (the nuScenes rule includes occupants). Attributes (cycle.with_rider) are unused. Key frames are sorted by token for subsampling, which is deterministic but not a documented random sample (the tutorial says seed 0 in another repo; for ngdet UNVERIFIED).
8. Metric label space is merged before scoring, so per-original-class AP (Van vs Car, Truck vs Tram) cannot be recovered afterward.
Strengths worth reusing: name-based mapping via synonyms (robust across models with different id spaces), the nuImages direct JSON parser with no devkit dependency, curated open-vocabulary prompt synonyms for Grounding DINO/OWLv2, and the tutorial's advice to prefer nuImages over nuScenes (projected 3D boxes are loose; tutorial shows about 0.06 vs about 0.30 mAP on the same models).

## 5. Evaluation tooling

Environment checked: Python 3.13.5, Linux x86_64, numpy 2.5.4, fresh venv. macOS arm64 (the M4) behavior is UNVERIFIED; wheels exist for these packages on PyPI but I did not test on arm64.

| Tool | Version tested | Python 3.13 / numpy 2 | Notes |
|---|---|---|---|
| pycocotools | 2.0.11 (abi3 wheel) | works | reference implementation; COCOeval output in my toy run: AP 0.598, AP50 0.792, AR@100 0.639 |
| faster-coco-eval | 1.8.0 (cp313 wheel) | works | same AP as pycocotools on my toy data (0.59787634 vs 0.5978763699); faster C++ backend; also has `extended_metrics()` (per-class mAP, precision, recall, best-F1) |
| torchmetrics `MeanAveragePrecision` | 1.9.0 | works but needs torchvision installed (import is gated on it) | `backend="pycocotools"` and `"faster_coco_eval"` both gave mAP 0.5978763699 and identical `map_per_class` on my toy data; `class_metrics=True` gives per-class; accepts `iscrowd` in targets. Needs boxes in tensors; easy to use but heavier than raw COCOeval |
| tidecv (TIDE) | 1.0.1 | imports and runs on numpy 2.5.4 | caveats below |

TIDE caveats found by running it:
- It imports `cv2`; with `opencv-python` the import failed on a bare Linux box (missing libxcb); `opencv-python-headless` fixed it.
- `datasets.COCO(path)` crashes with `KeyError: 'segmentation'` if annotations lack a `segmentation` field (box-only datasets such as KITTI exports). Fix: add a polygon from the bbox, `[[x,y,x+w,y,x+w,y+h,x,y+h]]`. After that `TIDE().evaluate(datasets.COCO(gt), datasets.COCOResult(dt), mode=TIDE.BOX)` printed the Cls/Loc/Both/Dupe/Bkg/Miss table (toy output: Miss 20.79 dAP only because I deleted 20% of detections).
- It treats `iscrowd` annotations as ignore regions (source `tidecv/datasets.py` line 104), compatible with our DontCare handling. It evaluates one IoU threshold (default 0.5 foreground); state that threshold in the report.
- Category ids and `info` keys: results file must be a list of dicts with `image_id`, `category_id`, `bbox`, `score`.

How to get the required numbers with plain pycocotools:
- AP / AP50 / AP75 / APs / APm / APl and AR@1/10/100, ARs/m/l: `E.stats[0..11]` after `evaluate(); accumulate(); summarize()`.
- Per-class AP: `E.eval['precision']` has shape [T iou, R recall, K class, A area, M maxDets]; for class k: `p = E.eval['precision'][:, :, k, 0, 2]; ap = p[p > -1].mean()` (verified in my toy script: matched the per-class values returned by torchmetrics).
- Per-class and per-size: use area index A = 1, 2, 3 for small, medium, large instead of 0; per-class AR from `E.eval['recall']` ([T, K, A, M]).
- Custom areas for driving (for example splitting by height): change `E.params.areaRng` and `areaRngLbl` before `evaluate`.
- Per-class evaluation over a subset: set `E.params.catIds`.
- Speed: pycocotools on 3769 KITTI images is seconds to a minute; faster-coco-eval is a drop-in (`from faster_coco_eval import COCO, COCOeval_faster`) if iteration is slow.

## 6. Published numbers to compare against (with limits)

None of these is the same protocol as ours. Use them as order-of-magnitude anchors.

| Source | Setting | Number |
|---|---|---|
| Simple multi-dataset detection, Zhou et al., CVPR 2022 (https://arxiv.org/pdf/2102.13086, Table 6, fetched and read) | Cascade R-CNN R50 trained on COCO only, zero-shot, mAP50 | KITTI 30.5 (evaluated on the KITTI train set, their 8-class space), Cityscapes 39.6, WildDash 25.9, VIPER 13.9, VOC 80.0, CrowdHuman 73.9. Unified model: KITTI 39.9. Note their metric is AP at IoU 0.5 with a mapped label space, so it is closer to our AP50 than our mAP |
| Instructor harness ngdet TUTORIAL section 12.1 (https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md, read) | 120 images, vehicle/person/cyclist, mAP@[.5:.95], RTX 3090 | COCO 0.47-0.61; KITTI 0.19-0.28 (Faster R-CNN v2 0.279, YOLO26x 0.264, RT-DETR-R50 0.262, YOLOv8x 0.255, GDINO-tiny 0.223, OWLv2 0.191); nuImages (500 val images) 0.22-0.31 (GDINO-tiny 0.306, Faster R-CNN v2 0.299). Small samples, score-thr 0.3, cyclist about 0: treat as rough |
| Adithya-Rajendran/HW1-2D-Detection (a prior student repo for this same course, https://github.com/Adithya-Rajendran/HW1-2D-Detection, read) | YOLO26m at 640, four shared classes | COCO 58.6 AP mean, KITTI 26.2 (-32.4); per-class KITTI from 11.3 (truck) to 50.7 (car); val split 3769, nuImages 3000 val frames |
| Domain Adaptive Faster R-CNN, Chen et al., CVPR 2018 (https://arxiv.org/pdf/1803.03243, Tables 1 and 3, fetched and read) | Faster R-CNN source-only, Car AP at IoU 0.5, trained on the source dataset only | Cityscapes to KITTI 53.5; KITTI to Cityscapes 30.2; Sim10k to Cityscapes 30.12; with domain adaptation 64.1 / 38.5 / 38.97. Cityscapes to Foggy Cityscapes (8 classes) mAP 18.8 source-only |
| 2PCNet, CVPR 2023 (https://arxiv.org/pdf/2303.13853, Table 1, fetched and read) | Faster R-CNN R50, BDD100K day to night, lower bound = trained on labelled day, tested on night; upper bound = trained on night | AP (metric likely AP50, UNVERIFIED) lower bound 41.1 vs upper bound 46.2; per class lower bound: pedestrian 50.0, rider 28.9, car 66.6, truck 47.8, bus 47.5, motorcycle 32.8, bicycle 39.5, traffic light 41.0, sign 56.5. Table 2 reports APcoco 22.1 lower bound, 23.9 upper bound for car/bus/truck. The day-to-night gap is small for supervised in-domain training |
| Towards Domain Generalization in Object Detection (https://arxiv.org/pdf/2203.14387, Table 1, fetched and read) | leave-one-out across BDD100K, Cityscapes, Sim10k, KITTI, ImageNet-pretrained R50 | Faster R-CNN mAP on held-out BDD100K 0.371, Cityscapes 0.452, Sim10k 0.303, KITTI 0.484 (metric/class set UNVERIFIED; likely AP50 on car-like classes) |

Gaps: I did not find a clean published table of COCO-pretrained YOLO/Faster R-CNN zero-shot mAP@[.5:.95] on KITTI val or on BDD100K or nuImages with a documented mapping beyond the student repos above. Searches for such tables returned only the items listed. Do not cite the ngdet or student-repo numbers as peer-reviewed.

Sanity expectation for our results (my inference, not a source): COCO-pretrained detectors lose about 30 AP points between COCO and KITTI on shared classes (consistent with the -32.4 above); car is highest, truck/cyclist/person lowest, small objects worst; modern architectures compress to similar KITTI numbers.

## 7. Suggested minimum data layout and commands (no downloads performed here)

```
data/coco/val2017/ + annotations/instances_val2017.json
data/kitti/training/image_2/*.png, label_2/*.txt, ImageSets/val.txt (3769 ids)
data/nuimages/{samples,v1.0-mini|v1.0-val}/
```
Converters to write: `kitti2coco.py` (keeps `height`, `truncated`, `occluded`, `kitti_type` as extra keys; DontCare and neighbor classes as `iscrowd=1`), `nuimages2coco.py` (reuse the mmdet3d name map above, add our ignore rules). Output one GT JSON per (dataset, taxonomy, difficulty).

## Source list
- KITTI benchmark page: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
- KITTI S3 files: https://s3.eu-central-1.amazonaws.com/avg-kitti/ (data_object_image_2.zip, data_object_label_2.zip, devkit_object.zip)
- COCO downloads: http://images.cocodataset.org/zips/val2017.zip
- OpenPCDet split files: https://github.com/open-mmlab/OpenPCDet/tree/master/data/kitti/ImageSets
- nuImages: https://www.nuscenes.org/nuimages ; mini: https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz ; paper summary https://arxiv.org/html/2512.02448
- nuScenes annotation rules: https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md
- mmdet3d nuImages converter: https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/tools/dataset_converters/nuimage_converter.py
- BDD100K: https://bdd-data.berkeley.edu ; converter doc https://www.github.com/SysCV/bdd100k-models/blob/main/doc/PREPARE_DATASET.md ; paper https://arxiv.org/abs/1805.04687
- Waymo: https://waymo.com/open/ ; ACDC: https://acdc.vision.ee.ethz.ch , https://arxiv.org/html/2104.13395v5
- Papers: https://arxiv.org/pdf/2102.13086 , https://arxiv.org/pdf/1803.03243 , https://arxiv.org/pdf/2303.13853 , https://arxiv.org/pdf/2203.14387
- Instructor harness: https://github.com/lkk688/DeepDataMiningLearning/tree/main/DeepDataMiningLearning/ngdet
- Student repos: https://github.com/Adithya-Rajendran/HW1-2D-Detection , https://github.com/harshitxjangam/cross-domain-detection-audit (protocol only, no results yet)
- Tools: pycocotools, faster-coco-eval, tidecv, torchmetrics on PyPI (versions tested above)
