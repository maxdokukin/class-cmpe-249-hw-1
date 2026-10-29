# HW1 references and claim checks (checked 2026-10-28)

Status key: VERIFIED = I opened the URL given and the title/authors/venue/number matched. UNVERIFIED = I could not open a primary source for that item (the item is listed from memory or from a search snippet only). Where only one field is unverified, it is stated.

## A. References (IEEE style)

Original research papers (>= 4 required): [1]-[4], [6]-[14]. Official repos / framework docs (>= 2 required): [15]-[25].

[1] S. Ren, K. He, R. Girshick, J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," in Proc. NeurIPS, 2015. arXiv:1506.01497. VERIFIED: https://papers.nips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html and https://arxiv.org/abs/1506.01497

[2] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, S. Belongie, "Feature pyramid networks for object detection," in Proc. IEEE CVPR, 2017. arXiv:1612.03144. VERIFIED: https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html

[3] Y. Li, S. Xie, X. Chen, P. Dollar, K. He, R. Girshick, "Benchmarking detection transfer learning with vision transformers," arXiv:2111.11429, 2021. VERIFIED: https://arxiv.org/abs/2111.11429 . This is the paper the torchvision `fasterrcnn_resnet50_fpn_v2` page cites (checked on the torchvision page, see [20]). Cite it as an arXiv preprint; I found no venue.

[4] J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You only look once: Unified, real-time object detection," in Proc. IEEE CVPR, 2016. arXiv:1506.02640. VERIFIED: https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html

[5] G. Jocher, J. Qiu, "Ultralytics YOLO11," software, v11.0.0, 2024 (AGPL-3.0). VERIFIED: https://docs.ultralytics.com/models/yolo11/ (the page's own citation block) and https://github.com/ultralytics/ultralytics

[6] G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, M. E. Kalfaoglu, "Ultralytics YOLO26: Unified real-time end-to-end vision models," arXiv:2606.03748, Jun. 2026. VERIFIED: https://arxiv.org/abs/2606.03748 (abstract page opened; title, authors, submitted 2 Jun 2026) and https://docs.ultralytics.com/models/yolo26/ (cites this exact arXiv ID, DOI 10.48550/arXiv.2606.03748). The notes' ID is correct.

[7] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko, "End-to-end object detection with transformers," in Proc. ECCV, 2020. arXiv:2005.12872. arXiv title/authors VERIFIED: https://arxiv.org/abs/2005.12872 . ECCV 2020 venue UNVERIFIED (not shown on the page I opened; from memory).

[8] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, J. Chen, "DETRs beat YOLOs on real-time object detection," in Proc. IEEE/CVF CVPR, 2024. arXiv:2304.08069. VERIFIED: https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html and https://arxiv.org/abs/2304.08069

[9] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, J. Zhu, L. Zhang, "Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection," in Proc. ECCV, 2024. arXiv:2303.05499. VERIFIED: https://arxiv.org/abs/2303.05499 and https://arxiv.org/html/2303.05499 . The ECCV 2024 venue and DOI 10.1007/978-3-031-72970-6_3 come from search results (mlanthology / unpaywall); the Springer page itself would not open (login redirect), so the venue is secondary-source only.

[10] M. Minderer, A. Gritsenko, N. Houlsby, "Scaling open-vocabulary object detection," in Advances in Neural Information Processing Systems 36 (NeurIPS), 2023. arXiv:2306.09683. VERIFIED: https://proceedings.neurips.cc/paper_files/paper/2023/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html and https://arxiv.org/html/2306.09683

[11] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, C. L. Zitnick, "Microsoft COCO: Common objects in context," in Proc. ECCV, 2014. arXiv:1405.0312. arXiv title/authors VERIFIED: https://arxiv.org/abs/1405.0312 . ECCV 2014 venue UNVERIFIED (from memory).

[12] A. Geiger, P. Lenz, R. Urtasun, "Are we ready for autonomous driving? The KITTI vision benchmark suite," in Proc. IEEE CVPR, 2012, pp. 3354-3361, doi:10.1109/CVPR.2012.6248074. Bibliographic data from a search result (https://mlanthology.org/cvpr/2012/geiger2012cvpr-we/); I did not open an IEEE or CVF page, so UNVERIFIED as a primary source. The benchmark page itself is VERIFIED: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

[13] X. Chen, K. Kundu, Y. Zhu, A. G. Berneshawi, H. Ma, S. Fidler, R. Urtasun, "3D object proposals for accurate object class detection," in Proc. NeurIPS, 2015, pp. 424-432. VERIFIED (title, authors, venue): https://papers.nips.cc/paper/2015/hash/6da37dd3139aa4d9aa55b8d237ec5d4a-Abstract.html . I read the paper PDF (https://www.cs.utoronto.ca/~fidler/papers/3dopNIPS15.pdf); see claim check 6b for what it says about the split.

[14] D. Bolya, S. Foley, J. Hays, J. Hoffman, "TIDE: A general toolbox for identifying object detection errors," in Proc. ECCV, 2020. arXiv:2008.08115. VERIFIED: https://arxiv.org/abs/2008.08115 and https://ar5iv.labs.arxiv.org/html/2008.08115 (ECCV 2020 stated there).

[15] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall, 1993, ISBN 978-0-412-04231-7. Publisher, year, authorship and ISBN VERIFIED via Open Library: https://openlibrary.org/search.json?q=an+introduction+to+the+bootstrap+efron (I did not open the book itself).

[16] R. Krishna et al., "Visual Genome: Connecting language and vision using crowdsourced dense image annotations," Int. J. Comput. Vis., 2017. arXiv:1602.07332. Supporting reference for claim 2. I read the arXiv PDF (https://arxiv.org/abs/1602.07332): VERIFIED for content. IJCV venue from memory.

[17] cocodataset/cocoapi (pycocotools source), GitHub. VERIFIED: https://github.com/cocodataset/cocoapi (page opens; README describes the COCO API with a Python API folder).

[18] Ultralytics repository and docs. VERIFIED: https://github.com/ultralytics/ultralytics , https://docs.ultralytics.com/models/yolo26/ , https://docs.ultralytics.com/models/yolo11/

[19] Hugging Face Transformers documentation: RT-DETR https://huggingface.co/docs/transformers/model_doc/rt_detr ; Grounding DINO https://huggingface.co/docs/transformers/model_doc/grounding-dino ; OWLv2 https://huggingface.co/docs/transformers/model_doc/owlv2 . All three VERIFIED (opened; each cites the paper listed above).

[20] PyTorch torchvision detection models docs. VERIFIED: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html (cites arXiv:2111.11429, COCO_V1 box_map 46.7) and https://docs.pytorch.org/vision/main/models/faster_rcnn.html

[21] L. Kun Kim (lkk688), "ngdet Tutorial: Zero-shot 2D detection across datasets," DeepDataMiningLearning, GitHub. VERIFIED (raw file read): https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md . It lists `hf_detr:PekingU/rtdetr_r50vd`, `gdino:IDEA-Research/grounding-dino-tiny`, `owlv2:google/owlv2-base-patch16-ensemble`. (The "L. Kun Kim" author expansion is my guess from the GitHub handle; cite the handle `lkk688` if unsure.)

[22] OpenPCDet, `data/kitti/ImageSets` (train.txt, val.txt, test.txt), GitHub. VERIFIED: https://github.com/open-mmlab/OpenPCDet/tree/master/data/kitti/ImageSets . Raw files counted: train.txt 3712 ids, val.txt 3769 ids, test.txt 7518 ids (3712 + 3769 = 7481 = KITTI training set size).

[23] Anthropic, "Models overview," Claude API docs. VERIFIED: https://platform.claude.com/docs/en/models/overview ; model pages https://platform.claude.com/docs/en/models/haiku-5-5/overview and https://platform.claude.com/docs/en/models/haiku-4-5/overview

[24] Hugging Face model cards: PekingU/rtdetr_r50vd https://huggingface.co/PekingU/rtdetr_r50vd ; IDEA-Research/grounding-dino-tiny https://huggingface.co/IDEA-Research/grounding-dino-tiny ; google/owlv2-base-patch16-ensemble https://huggingface.co/google/owlv2-base-patch16-ensemble . All three VERIFIED (opened; raw README read for the last two). Caveats in the claim checks below.

[25] A. Gupta, P. Dollar, R. Girshick, "LVIS: A dataset for large vocabulary instance segmentation," in Proc. CVPR, 2019. arXiv:1908.03195. Supporting reference for claim 3. UNVERIFIED as primary (found only in search results: https://openaccess.thecvf.com/content_CVPR_2019/papers/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.pdf ; I did not open it).

[26] A. Kamath et al., "MDETR: Modulated detection for end-to-end multi-modal understanding," arXiv:2104.12763, 2021. Supporting reference for GoldG. Content VERIFIED at https://arxiv.org/html/2104.12763 (venue ICCV 2021 not checked, cite as arXiv).

Tally: 26 entries. Primary papers: [1]-[4], [6]-[11], [13], [14], [16], [25], [26] plus book [15]. Fully VERIFIED including venue: [1] [2] [4] [5] [6] [8] [10] [13] [14] [15] [17]-[24] (18 entries). Verified for content but with one field (venue/DOI) from memory or a secondary source: [3] (preprint, no venue to check), [7] [9] [11] [12] [16] [26]. Entirely UNVERIFIED as a primary source: [25]. Requirements (>= 8 references, >= 4 papers, >= 2 repos/docs) are met with margin using only fully verified entries: papers [1] [2] [4] [6] [8] [10] [13] [14]; repos/docs [5] [17]-[24].

## B. Claim checks

### 1. Grounding DINO-T pre-training and COCO AP: TRUE (with a trap)
- Paper Table (arXiv HTML v5, https://arxiv.org/html/2303.05499), zero-shot COCO row: "Grounding DINO T (Ours) Swin-T O365,GoldG,Cap4M 48.4 [zero-shot AP] 57.2 [fine-tuned AP]". Other rows: T with O365 only 46.7; T with O365,GoldG 48.1.
- The HF `grounding-dino-tiny` conversion script maps the checkpoint to `groundingdino_swint_ogc.pth` (https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/grounding_dino/convert_grounding_dino_to_hf.py). The GitHub README (https://github.com/IDEA-Research/GroundingDINO) lists GroundingDINO-T as O365,GoldG,Cap4M, 48.4 zero-shot, 57.2 fine-tuned. The link between the "ogc" file name and the README row is by naming; it is not stated in one sentence anywhere I read.
- Trap: the HF model card for grounding-dino-tiny says "The model achieves remarkable results, such as 52.5 AP on COCO zero-shot." That 52.5 is the Swin-L model (O365, OI, GoldG), not the tiny checkpoint. Do not quote 52.5 for the tiny model.

### 2. GoldG contents and Visual Genome / COCO overlap: TRUE
- Grounding DINO paper: "GoldG contains images in Flickr30k entities and Visual Genome" (preprocessed by MDETR). https://arxiv.org/html/2303.05499
- Visual Genome paper, section 5.1: "The Visual Genome dataset consists of all 108,249 images from the intersection of MS-COCO's 328,000 images and YFCC's 100 million images." So the overlap is not just heavy: by construction every VG image is also a COCO image. https://arxiv.org/abs/1602.07332
- Caveat: Grounding DINO defines zero-shot as "the training split of the test dataset is not utilized", so VG/COCO image sharing is allowed under their definition. MDETR says it removed val/test images of its downstream tasks from pre-training (https://arxiv.org/html/2104.12763). A search snippet from a third-party paper says GoldG excludes COCO images; I did not open that paper, so treat it as UNVERIFIED. None of this touches KITTI.

### 3. OWLv2 base-patch16-ensemble training data: PARTLY (the "Objects365 and/or Visual Genome" fine-tuning part is wrong)
- WebLI self-training: TRUE. "We use the WebLI dataset as the source of weak supervision for self-training." Pseudo-boxes come from an OWL-ViT CLIP-L/14 annotator queried with caption N-grams and/or a curated vocabulary built from LVIS, Objects365, OpenImagesV4 and Visual Genome class names. https://arxiv.org/html/2306.09683
- Objects365 and Visual Genome were used to train the OWL-ViT L/14 annotator (paper Table 1 and Table A1 label it "trained on Objects365 and Visual Genome"), not to fine-tune the OWLv2 B/16 ensemble.
- The ensemble is a weight-space average of the self-trained checkpoint and the checkpoint fine-tuned on LVIS-base (LVIS with rare-category annotations removed). Paper: "create an ensemble of the model before and after fine-tuning by averaging the model weights"; appendix: fine-tuned weight 0.4. The Scenic README (https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) lists "OWLv2 CLIP B/16 ST/FT ens" as separate from the "fine-tuned on Objects365 and Visual Genome" checkpoints (`...st-ngrams-ft-o365vg...`), which are a different set.
- Mapping caveat: the HF conversion script has an `owlv2-base-patch16-ensemble` branch and points to the Scenic checkpoint list, but it does not name the exact Scenic file. That the HF ensemble equals "B/16 ST/FT ens" is a strong inference, not a quote.
- The HF model card for the ensemble says the heads and backbone "are fine-tuned on publicly available object detection datasets such as COCO and OpenImages" (that is OWL-ViT v1 boilerplate, the card says "to be updated for v2"). It disagrees with the paper. Do not cite it for training data.
- Does the paper report COCO AP for this checkpoint? No. The only COCO number is for models fine-tuned on COCO: "our B/16 and L/14 models reach 54.3% and 56.0% COCO mAP". The ensemble checkpoint was not trained on COCO labels.
- Contamination note for the write-up: LVIS annotates COCO 2017 images (search result, UNVERIFIED primary, see [25]), so LVIS-base fine-tuning means this checkpoint has seen COCO images with LVIS labels. The WebLI data was filtered against COCO/LVIS, Objects365 and Visual Genome splits. KITTI is not involved either way.
- Suggested sentence: "OWLv2-B/16-ensemble is a weight-space ensemble of a WebLI self-trained model (pseudo-boxes from an OWL-ViT annotator trained on Objects365 and Visual Genome) and the same model fine-tuned on LVIS-base."

### 4. Official COCO numbers: TRUE for every number
- RT-DETR R50: 53.1% AP on COCO val2017 (paper abstract: "RT-DETR-R50 / R101 achieves 53.1% / 54.3% AP on COCO and 108 / 74 FPS on T4"), https://arxiv.org/abs/2304.08069 ; HF card for PekingU/rtdetr_r50vd lists 53.1 AP (71.3 AP50, 57.7 AP75), trained on COCO 2017 train. The Objects365-pretrained variant (55.3) is a different checkpoint. Note: this checkpoint is trained on COCO, so it is in-domain on COCO and zero-shot only with respect to KITTI.
- YOLO11m: 51.5 mAP val 50-95. https://docs.ultralytics.com/models/yolo11/ (n 39.5, s 47.0, m 51.5, l 53.4, x 54.7)
- YOLO26m: 53.1 with the default NMS (one-to-many) head, 52.5 with the NMS-free end-to-end head (column "mAP val 50-95(e2e)"). Full table at https://docs.ultralytics.com/models/yolo26/ : n 40.9/40.1, s 48.6/47.8, m 53.1/52.5, l 55.0/54.4, x 57.5/56.9. So "53.1 (NMS-free 52.5?)" is correct as written: 53.1 is NOT the NMS-free number.
- torchvision fasterrcnn_resnet50_fpn_v2, COCO_V1 weights: box_map 46.7 on COCO val2017. https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html

### 5. YOLO26 NMS-free by default: FALSE (the default is the NMS head)
- Ultralytics docs (https://docs.ultralytics.com/models/yolo26/): "YOLO26 supports native end-to-end inference with nms=False, producing predictions without a separate NMS pass. The default one-to-many path uses NMS for accuracy." Also: "One-to-Many Head (Default): Generates traditional YOLO outputs requiring NMS post-processing" and "One-to-One Head (nms=False): Produces end-to-end predictions without NMS." Code comment in the docs: "Use one-to-many head (default, Ultralytics applies NMS)" and "Opt into the NMS-free one-to-one head ... nms=False". Also: "Prediction and validation default to the one-to-many head for accuracy ... Both heads are trained regardless of this choice." The overview calls it "The optional one-to-one detection head".
- Correct statement: "YOLO26 has a dual head. Ultralytics predict/val/export default to the one-to-many head followed by NMS; the NMS-free one-to-one head is opt-in with nms=False. The reported 52.5 (m) is the NMS-free number and 53.1 is the default-head number." If the harness ran yolo26 with default arguments, it used NMS. Check how ngdet's `yolo` detector calls predict.

### 6. KITTI difficulty criteria: TRUE
- KITTI site, exact text: Easy "Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %"; Moderate "25 Px, Partly occluded, 30 %"; Hard "25 Px, Difficult to see, 50 %". The occlusion levels 0/1/2 in label files correspond to fully visible / partly occluded / largely occluded (difficult to see); the site gives the words, the numeric codes are the devkit's convention (I did not open the devkit; UNVERIFIED for the numeric codes). https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
- Also on that page: IoU 70% for cars, 50% for pedestrians and cyclists; since 2019 AP uses 40 recall positions; methods are ranked on Moderate; about 2% of Hard boxes are not recognised by humans.

### 6b. 3DOP split (3712 train / 3769 val): PARTLY
- The 3DOP paper says only: "we split the KITTI training set into train and validation sets (each containing half of the images). We ensure that our training and validation set do not come from the same video sequences." It does not print 3712/3769. The counts come from the split files: OpenPCDet `ImageSets/train.txt` = 3712 ids and `val.txt` = 3769 ids (sum 7481). Cite [13] for the method and [22] for the numbers.

### 7. TIDE error types and oracle ΔAP: TRUE
- Six types (ar5iv text of https://arxiv.org/abs/2008.08115): Classification (right box, wrong class), Localization (right class, poor box), Both (wrong class and poor box), Duplicate (right class, but the ground truth is already matched by a higher-scoring detection), Background (detected where there is no object), Missed (undetected ground truth). Short names Cls, Loc, Both, Dupe, Bkg, Miss match.
- ΔAP: each error type is fixed by an "oracle" separately, ΔAP_o = AP_o - AP, so errors are not fixed cumulatively; fixing all errors gives 100 mAP. The tool's PDF would not render for me, so this rests on the ar5iv HTML copy; wording of the six definitions is paraphrased from it. Venue ECCV 2020 is stated there.

### 8. Claude models: PARTLY (Haiku 4.5 is not current)
- Haiku 5.5: "Released October 7, 2026", Latest, model ID `claude-haiku-5-5`. TRUE. https://platform.claude.com/docs/en/models/haiku-5-5/overview
- Haiku 4.5: status "Active (legacy)", released October 15, 2025, ID `claude-haiku-4-5-20251001` (alias `claude-haiku-4-5`), "Retirement: Not sooner than October 15, 2026", which is 18 days from today. The docs say "Claude Haiku 4.5 is a legacy model; Claude Haiku 5.5 is the current Haiku model." So the claim is FALSE for Haiku 4.5. https://platform.claude.com/docs/en/models/haiku-4-5/overview
- Sonnet 5.5 (`claude-sonnet-5-5`) and Opus 5.5 (`claude-opus-5-5`) are current: TRUE. The lineup also includes Claude Fable 5.1 (`claude-fable-5-1`, $10 / $50). https://platform.claude.com/docs/en/models/overview
- Prices per MTok (input / output):

| Model | Input | Output | Note |
|---|---|---|---|
| Claude Haiku 5.5 | $0.10 | $0.50 | for prompts up to 100,000 tokens; $0.50 / $2.50 above 100,000 tokens (the overview shows "From $0.10 / $0.50") |
| Claude Haiku 4.5 (legacy) | $1 | $5 | |
| Claude Sonnet 5.5 | $2 | $10 | |
| Claude Opus 5.5 | $4 | $20 | |

  Batch API is 50% off. Haiku 5.5 uses the newer tokenizer, so the same text counts as about 30% more tokens than on Haiku 4.5; per-token price is lower but compare cost per task, not per token.

## C. Other discrepancies found
- Notes' YOLO26 arXiv ID 2606.03748 is correct (title "Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models", submitted 2 Jun 2026).
- HF grounding-dino-tiny card quotes 52.5 AP (the Swin-L result); real tiny number is 48.4.
- HF owlv2 card still describes OWL-ViT v1 fine-tuning data (COCO, OpenImages); the paper says LVIS-base for the ensemble.
- Not checked: the KITTI devkit source (only the website), the Springer/IEEE pages for [9] and [12], COCO and DETR venue pages.
