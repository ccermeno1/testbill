# Model Analysis Report

Hub slug **`rotated_retinanet_dota_le90_3x`**. Official DOTA v1.0 Task 1 **73.89%** (AP75 47.11, COCO mAP 44.63). Deploy `production.score_threshold` **0.25** (eval-val F1 0.30 − 0.05). **OBB matching** (`use_hbb_for_matching: false`, MMRotate `RBboxOverlaps2D`); +2.17 Task 1 / +3.65 AP75 vs Hub 1× OBB **71.72%**. Circum-HBB 3× is `rotated_retinanet_dota_le90_3x_hbb`. The mAP50 below is leaky eval-val (val tiles are in train), not the published number.

## Official Task 1 (hidden test)

| Class | AP50 |
| --- | ---: |
| plane | 0.8944 |
| baseball-diamond | 0.8289 |
| bridge | 0.4426 |
| ground-track-field | 0.7581 |
| small-vehicle | 0.7907 |
| large-vehicle | 0.7174 |
| ship | 0.8609 |
| tennis-court | 0.9054 |
| basketball-court | 0.8365 |
| storage-tank | 0.8138 |
| soccer-ball-field | 0.6155 |
| roundabout | 0.6423 |
| harbor | 0.6424 |
| swimming-pool | 0.7159 |
| helicopter | 0.6189 |
| **mAP50** | **0.7389** |

COCO-style: AP50 **0.7389**, AP75 **0.4711**, mAP **0.4463**.

- Generated at: `2026-09-21T06:47:31.362033`

## Model metadata
- Experiment dir: `runs/rotated_retinanet/20260920-054001`
- Checkpoint: `runs/rotated_retinanet/20260920-054001/checkpoints/best_mAP_0.80.pth`
- Checkpoint modified: `2026-09-21T05:06:14.775918`
- Config: `runs/rotated_retinanet/20260920-054001/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `206641`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.3000`
- Precision at best threshold: `0.7449`
- Recall at best threshold: `0.8282`
- F1 at best threshold: `0.7844`
- F2 at best threshold: `0.8101`
- mAP50: `0.7856` (78.56%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7593`
- Global mean best IoU (same class): `0.7555` (median `0.8046`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.8374 | 0.8370 | 0.8611 |
| `basketball-court` | 278 | 0.8951 | 0.8918 | 0.9086 |
| `bridge` | 666 | 0.6631 | 0.6601 | 0.7079 |
| `ground-track-field` | 216 | 0.8442 | 0.8432 | 0.8800 |
| `harbor` | 4298 | 0.7173 | 0.7146 | 0.7593 |
| `helicopter` | 157 | 0.7783 | 0.7598 | 0.8104 |
| `large-vehicle` | 9398 | 0.7845 | 0.7758 | 0.8138 |
| `plane` | 4731 | 0.8351 | 0.8351 | 0.8748 |
| `roundabout` | 256 | 0.8198 | 0.8056 | 0.8790 |
| `ship` | 18534 | 0.7729 | 0.7712 | 0.8059 |
| `small-vehicle` | 11357 | 0.7117 | 0.7055 | 0.7641 |
| `soccer-ball-field` | 260 | 0.7847 | 0.7740 | 0.8929 |
| `storage-tank` | 5031 | 0.6919 | 0.6908 | 0.7886 |
| `swimming-pool` | 693 | 0.6702 | 0.6701 | 0.7108 |
| `tennis-court` | 1529 | 0.9076 | 0.9049 | 0.9186 |
| **global** | 57768 | 0.7593 | 0.7555 | 0.8046 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 1440 | 0.981 | 0.8145 |
| `basketball-court` | 278 | 829 | 0.996 | 0.8735 |
| `bridge` | 666 | 10624 | 0.829 | 0.5693 |
| `ground-track-field` | 216 | 1242 | 0.972 | 0.8286 |
| `harbor` | 4298 | 11002 | 0.893 | 0.7406 |
| `helicopter` | 157 | 577 | 0.949 | 0.8863 |
| `large-vehicle` | 9398 | 37496 | 0.946 | 0.8300 |
| `plane` | 4731 | 8705 | 0.956 | 0.8869 |
| `roundabout` | 256 | 1163 | 0.922 | 0.7963 |
| `ship` | 18534 | 42427 | 0.953 | 0.7154 |
| `small-vehicle` | 11357 | 65588 | 0.886 | 0.7418 |
| `soccer-ball-field` | 260 | 1125 | 0.819 | 0.7845 |
| `storage-tank` | 5031 | 17093 | 0.804 | 0.7200 |
| `swimming-pool` | 693 | 4648 | 0.885 | 0.7216 |
| `tennis-court` | 1529 | 2682 | 0.992 | 0.8751 |
| **mAP** | | | | 0.7856 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.7500 | 0.8195 | 0.8984 | 0.8571 | 327 | 72 | 37 |
| `basketball-court` | 0.5500 | 0.8977 | 0.9784 | 0.9363 | 272 | 31 | 6 |
| `bridge` | 0.3500 | 0.5884 | 0.5495 | 0.5683 | 366 | 256 | 300 |
| `ground-track-field` | 0.6000 | 0.8230 | 0.9259 | 0.8715 | 200 | 43 | 16 |
| `harbor` | 0.2000 | 0.7371 | 0.8381 | 0.7843 | 3602 | 1285 | 696 |
| `helicopter` | 0.5000 | 0.9545 | 0.8025 | 0.8720 | 126 | 6 | 31 |
| `large-vehicle` | 0.3000 | 0.8029 | 0.8509 | 0.8262 | 7997 | 1963 | 1401 |
| `plane` | 0.4000 | 0.9313 | 0.9146 | 0.9229 | 4327 | 319 | 404 |
| `roundabout` | 0.7500 | 0.8333 | 0.7227 | 0.7741 | 185 | 37 | 71 |
| `ship` | 0.3500 | 0.6874 | 0.8696 | 0.7678 | 16117 | 7329 | 2417 |
| `small-vehicle` | 0.3500 | 0.8309 | 0.6894 | 0.7536 | 7830 | 1594 | 3527 |
| `soccer-ball-field` | 0.7000 | 0.9434 | 0.7692 | 0.8475 | 200 | 12 | 60 |
| `storage-tank` | 0.3000 | 0.7799 | 0.7203 | 0.7489 | 3624 | 1023 | 1407 |
| `swimming-pool` | 0.4000 | 0.7321 | 0.6941 | 0.7126 | 481 | 176 | 212 |
| `tennis-court` | 0.6000 | 0.9255 | 0.9745 | 0.9493 | 1490 | 120 | 39 |

## Confusion matrix

Computed at score threshold `0.3000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 349 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |
| `basketball-court` | 0 | 275 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |
| `bridge` | 0 | 0 | 409 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 257 |
| `ground-track-field` | 0 | 0 | 0 | 209 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| `harbor` | 0 | 0 | 0 | 0 | 3037 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1260 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 143 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 7985 | 0 | 0 | 2 | 83 | 0 | 0 | 0 | 0 | 1328 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 4413 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 316 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 213 | 0 | 0 | 0 | 0 | 0 | 0 | 43 |
| `ship` | 0 | 0 | 2 | 0 | 3 | 0 | 1 | 0 | 0 | 16718 | 0 | 0 | 0 | 1 | 0 | 1809 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 134 | 0 | 0 | 1 | 8167 | 0 | 0 | 0 | 0 | 3055 |
| `soccer-ball-field` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 212 | 0 | 0 | 0 | 48 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3624 | 0 | 0 | 1407 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 539 | 0 | 154 |
| `tennis-court` | 4 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1505 | 18 |
| `False Positive` | 226 | 70 | 377 | 118 | 718 | 43 | 1840 | 447 | 129 | 8397 | 2198 | 99 | 1023 | 311 | 187 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
