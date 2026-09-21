# Model Analysis Report

Hub slug **`rotated_retinanet_dota_le90_1x`**. Official DOTA v1.0 Task 1 **71.72%** (AP75 43.46, COCO mAP 42.18). Deploy `production.score_threshold` **0.25** (eval-val F1 0.30 − 0.05). **OBB matching** (`use_hbb_for_matching: false`, MMRotate `RBboxOverlaps2D`); this is the published Hub 1× default — ahead of MMRotate OBB (**68.42%**). Circum-HBB is `rotated_retinanet_dota_le90_1x_hbb`. The mAP50 below is leaky eval-val (val tiles are in train), not the published number.

## Official Task 1 (hidden test)

| Class | AP50 |
| --- | ---: |
| plane | 0.8984 |
| baseball-diamond | 0.8082 |
| bridge | 0.4051 |
| ground-track-field | 0.7413 |
| small-vehicle | 0.7841 |
| large-vehicle | 0.6901 |
| ship | 0.8386 |
| tennis-court | 0.9056 |
| basketball-court | 0.8263 |
| storage-tank | 0.8267 |
| soccer-ball-field | 0.5768 |
| roundabout | 0.6452 |
| harbor | 0.5871 |
| swimming-pool | 0.6854 |
| helicopter | 0.5397 |
| **mAP50** | **0.7172** |

COCO-style: AP50 **0.7172**, AP75 **0.4346**, mAP **0.4218**.

- Generated at: `2026-09-20T02:55:35.971775`

## Model metadata
- Experiment dir: `runs/rotated_retinanet/20260919-093746`
- Checkpoint: `runs/rotated_retinanet/20260919-093746/checkpoints/best_mAP_0.72.pth`
- Checkpoint modified: `2026-09-19T17:40:08.897616`
- Config: `runs/rotated_retinanet/20260919-093746/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `298918`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.3000`
- Precision at best threshold: `0.7120`
- Recall at best threshold: `0.7727`
- F1 at best threshold: `0.7411`
- F2 at best threshold: `0.7597`
- mAP50: `0.7112` (71.12%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7281`
- Global mean best IoU (same class): `0.7221` (median `0.7787`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7657 | 0.7588 | 0.8107 |
| `basketball-court` | 278 | 0.8569 | 0.8536 | 0.8731 |
| `bridge` | 666 | 0.5913 | 0.5846 | 0.6267 |
| `ground-track-field` | 216 | 0.7782 | 0.7761 | 0.8281 |
| `harbor` | 4298 | 0.6594 | 0.6539 | 0.6941 |
| `helicopter` | 157 | 0.7298 | 0.6552 | 0.7390 |
| `large-vehicle` | 9398 | 0.7539 | 0.7406 | 0.7831 |
| `plane` | 4731 | 0.8119 | 0.8078 | 0.8534 |
| `roundabout` | 256 | 0.7522 | 0.7354 | 0.8310 |
| `ship` | 18534 | 0.7480 | 0.7451 | 0.7864 |
| `small-vehicle` | 11357 | 0.6882 | 0.6821 | 0.7450 |
| `soccer-ball-field` | 260 | 0.7308 | 0.7069 | 0.8365 |
| `storage-tank` | 5031 | 0.6523 | 0.6496 | 0.7551 |
| `swimming-pool` | 693 | 0.6101 | 0.6050 | 0.6718 |
| `tennis-court` | 1529 | 0.8775 | 0.8722 | 0.9067 |
| **global** | 57768 | 0.7281 | 0.7221 | 0.7787 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 3288 | 0.931 | 0.7687 |
| `basketball-court` | 278 | 2158 | 0.982 | 0.8183 |
| `bridge` | 666 | 21625 | 0.734 | 0.4065 |
| `ground-track-field` | 216 | 4013 | 0.944 | 0.7729 |
| `harbor` | 4298 | 14024 | 0.824 | 0.6854 |
| `helicopter` | 157 | 1150 | 0.828 | 0.6965 |
| `large-vehicle` | 9398 | 60692 | 0.924 | 0.7805 |
| `plane` | 4731 | 12642 | 0.948 | 0.8872 |
| `roundabout` | 256 | 2627 | 0.879 | 0.6851 |
| `ship` | 18534 | 49537 | 0.932 | 0.7026 |
| `small-vehicle` | 11357 | 88755 | 0.864 | 0.6798 |
| `soccer-ball-field` | 260 | 3632 | 0.788 | 0.6540 |
| `storage-tank` | 5031 | 25782 | 0.751 | 0.6625 |
| `swimming-pool` | 693 | 4124 | 0.802 | 0.6022 |
| `tennis-court` | 1529 | 4869 | 0.975 | 0.8650 |
| **mAP** | | | | 0.7112 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.5500 | 0.7961 | 0.7830 | 0.7895 | 285 | 73 | 79 |
| `basketball-court` | 0.5000 | 0.8556 | 0.8309 | 0.8431 | 231 | 39 | 47 |
| `bridge` | 0.3500 | 0.5423 | 0.4520 | 0.4930 | 301 | 254 | 365 |
| `ground-track-field` | 0.5500 | 0.8177 | 0.7269 | 0.7696 | 157 | 35 | 59 |
| `harbor` | 0.2000 | 0.6407 | 0.7641 | 0.6969 | 3284 | 1842 | 1014 |
| `helicopter` | 0.4000 | 0.8125 | 0.6624 | 0.7298 | 104 | 24 | 53 |
| `large-vehicle` | 0.3000 | 0.7504 | 0.7800 | 0.7649 | 7330 | 2438 | 2068 |
| `plane` | 0.4000 | 0.9026 | 0.9015 | 0.9021 | 4265 | 460 | 466 |
| `roundabout` | 0.4000 | 0.6431 | 0.7461 | 0.6908 | 191 | 106 | 65 |
| `ship` | 0.3000 | 0.6578 | 0.8512 | 0.7421 | 15777 | 8209 | 2757 |
| `small-vehicle` | 0.3500 | 0.7929 | 0.6420 | 0.7095 | 7291 | 1904 | 4066 |
| `soccer-ball-field` | 0.5000 | 0.7897 | 0.6500 | 0.7131 | 169 | 45 | 91 |
| `storage-tank` | 0.3000 | 0.7682 | 0.6502 | 0.7043 | 3271 | 987 | 1760 |
| `swimming-pool` | 0.4500 | 0.7206 | 0.5657 | 0.6338 | 392 | 152 | 301 |
| `tennis-court` | 0.5500 | 0.9148 | 0.9339 | 0.9243 | 1428 | 133 | 101 |

## Confusion matrix

Computed at score threshold `0.3000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 326 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 38 |
| `basketball-court` | 0 | 245 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 33 |
| `bridge` | 0 | 0 | 339 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 327 |
| `ground-track-field` | 0 | 0 | 0 | 190 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 21 |
| `harbor` | 0 | 0 | 0 | 0 | 2759 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1538 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 108 | 0 | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 7307 | 0 | 0 | 3 | 132 | 0 | 0 | 0 | 0 | 1956 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4372 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 359 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 205 | 0 | 0 | 0 | 1 | 0 | 0 | 50 |
| `ship` | 0 | 0 | 2 | 0 | 6 | 0 | 7 | 0 | 0 | 15776 | 0 | 0 | 0 | 2 | 0 | 2741 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 212 | 0 | 0 | 1 | 7541 | 0 | 0 | 0 | 0 | 3603 |
| `soccer-ball-field` | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 183 | 0 | 0 | 0 | 72 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3271 | 0 | 0 | 1760 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 463 | 0 | 230 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1465 | 54 |
| `False Positive` | 284 | 96 | 442 | 209 | 983 | 88 | 2242 | 644 | 200 | 8205 | 2573 | 170 | 986 | 344 | 262 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
