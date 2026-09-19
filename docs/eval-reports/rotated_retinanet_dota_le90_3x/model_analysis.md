# Model Analysis Report

Hub slug **`rotated_retinanet_dota_le90_3x`**. Official DOTA v1.0 Task 1 **70.70%** (AP75 43.34, COCO mAP 41.59). Deploy `production.score_threshold` **0.35** (eval-val F1 0.40 − 0.05). **Circum-HBB matching** (`use_hbb_for_matching: true`); +2.83 Task 1 / +3.26 AP75 vs Hub 1× **67.87%**. The mAP50 below is leaky eval-val (val tiles are in train), not the published number.

## Official Task 1 (hidden test)

| Class | AP50 |
| --- | ---: |
| plane | 0.8729 |
| baseball-diamond | 0.8400 |
| bridge | 0.4595 |
| ground-track-field | 0.7324 |
| small-vehicle | 0.6602 |
| large-vehicle | 0.5775 |
| ship | 0.7107 |
| tennis-court | 0.9054 |
| basketball-court | 0.8189 |
| storage-tank | 0.8272 |
| soccer-ball-field | 0.5779 |
| roundabout | 0.6352 |
| harbor | 0.6527 |
| swimming-pool | 0.7189 |
| helicopter | 0.6151 |
| **mAP50** | **0.7070** |

COCO-style: AP50 **0.7070**, AP75 **0.4334**, mAP **0.4159**.

- Generated at: `2026-09-14T01:14:19.789503`

## Model metadata
- Experiment dir: `runs/rotated_retinanet/20260913-031811`
- Checkpoint: `runs/rotated_retinanet/20260913-031811/checkpoints/best_mAP_0.80.pth`
- Checkpoint modified: `2026-09-13T21:26:33.747190`
- Config: `runs/rotated_retinanet/20260913-031811/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `223571`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.4000`
- Precision at best threshold: `0.7455`
- Recall at best threshold: `0.7604`
- F1 at best threshold: `0.7529`
- F2 at best threshold: `0.7574`
- mAP50: `0.7651` (76.51%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7096`
- Global mean best IoU (same class): `0.7013` (median `0.7908`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.8255 | 0.8252 | 0.8393 |
| `basketball-court` | 278 | 0.8805 | 0.8792 | 0.9049 |
| `bridge` | 666 | 0.6935 | 0.6905 | 0.7431 |
| `ground-track-field` | 216 | 0.8487 | 0.8480 | 0.8665 |
| `harbor` | 4298 | 0.7315 | 0.7290 | 0.7701 |
| `helicopter` | 157 | 0.7772 | 0.7541 | 0.7798 |
| `large-vehicle` | 9398 | 0.7243 | 0.7108 | 0.7859 |
| `plane` | 4731 | 0.8261 | 0.8259 | 0.8691 |
| `roundabout` | 256 | 0.7982 | 0.7874 | 0.8641 |
| `ship` | 18534 | 0.6646 | 0.6515 | 0.7853 |
| `small-vehicle` | 11357 | 0.6864 | 0.6805 | 0.7591 |
| `soccer-ball-field` | 260 | 0.7829 | 0.7630 | 0.8749 |
| `storage-tank` | 5031 | 0.6883 | 0.6872 | 0.7859 |
| `swimming-pool` | 693 | 0.6667 | 0.6667 | 0.7153 |
| `tennis-court` | 1529 | 0.8996 | 0.8934 | 0.9165 |
| **global** | 57768 | 0.7096 | 0.7013 | 0.7908 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 1682 | 0.984 | 0.8312 |
| `basketball-court` | 278 | 897 | 0.986 | 0.8737 |
| `bridge` | 666 | 8310 | 0.886 | 0.6582 |
| `ground-track-field` | 216 | 1293 | 0.995 | 0.7887 |
| `harbor` | 4298 | 16238 | 0.913 | 0.7726 |
| `helicopter` | 157 | 771 | 0.949 | 0.8957 |
| `large-vehicle` | 9398 | 40491 | 0.856 | 0.6763 |
| `plane` | 4731 | 10966 | 0.951 | 0.8854 |
| `roundabout` | 256 | 1508 | 0.914 | 0.7611 |
| `ship` | 18534 | 37953 | 0.783 | 0.5709 |
| `small-vehicle` | 11357 | 70378 | 0.844 | 0.7018 |
| `soccer-ball-field` | 260 | 1345 | 0.838 | 0.7943 |
| `storage-tank` | 5031 | 25113 | 0.793 | 0.6861 |
| `swimming-pool` | 693 | 3787 | 0.872 | 0.7096 |
| `tennis-court` | 1529 | 2839 | 0.982 | 0.8706 |
| **mAP** | | | | 0.7651 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.7000 | 0.8203 | 0.8901 | 0.8538 | 324 | 71 | 40 |
| `basketball-court` | 0.7000 | 0.8775 | 0.9532 | 0.9138 | 265 | 37 | 13 |
| `bridge` | 0.5000 | 0.6763 | 0.7057 | 0.6907 | 470 | 225 | 196 |
| `ground-track-field` | 0.5500 | 0.7811 | 0.9583 | 0.8607 | 207 | 58 | 9 |
| `harbor` | 0.5000 | 0.7911 | 0.8343 | 0.8121 | 3586 | 947 | 712 |
| `helicopter` | 0.4500 | 0.9085 | 0.8854 | 0.8968 | 139 | 14 | 18 |
| `large-vehicle` | 0.3500 | 0.7564 | 0.8001 | 0.7776 | 7519 | 2421 | 1879 |
| `plane` | 0.5000 | 0.9356 | 0.9119 | 0.9236 | 4314 | 297 | 417 |
| `roundabout` | 0.6500 | 0.7866 | 0.7344 | 0.7596 | 188 | 51 | 68 |
| `ship` | 0.5500 | 0.6645 | 0.7364 | 0.6986 | 13648 | 6891 | 4886 |
| `small-vehicle` | 0.4000 | 0.8281 | 0.6539 | 0.7307 | 7426 | 1542 | 3931 |
| `soccer-ball-field` | 0.6000 | 0.8996 | 0.7923 | 0.8425 | 206 | 23 | 54 |
| `storage-tank` | 0.3500 | 0.8389 | 0.6698 | 0.7449 | 3370 | 647 | 1661 |
| `swimming-pool` | 0.4000 | 0.6859 | 0.7027 | 0.6942 | 487 | 223 | 206 |
| `tennis-court` | 0.7500 | 0.9304 | 0.9621 | 0.9460 | 1471 | 110 | 58 |

## Confusion matrix

Computed at score threshold `0.4000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 353 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
| `basketball-court` | 0 | 271 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| `bridge` | 0 | 0 | 505 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 161 |
| `ground-track-field` | 0 | 0 | 0 | 211 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| `harbor` | 0 | 0 | 0 | 0 | 3728 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 564 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 141 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 7259 | 0 | 0 | 0 | 80 | 0 | 0 | 0 | 0 | 2059 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4363 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 368 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 201 | 0 | 0 | 0 | 0 | 0 | 0 | 55 |
| `ship` | 0 | 0 | 0 | 0 | 7 | 0 | 2 | 0 | 0 | 14172 | 0 | 0 | 0 | 1 | 0 | 4352 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 106 | 0 | 0 | 0 | 7402 | 0 | 0 | 0 | 0 | 3849 |
| `soccer-ball-field` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 214 | 0 | 0 | 0 | 46 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3096 | 0 | 0 | 1935 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 487 | 0 | 206 |
| `tennis-court` | 3 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1488 | 32 |
| `False Positive` | 171 | 59 | 352 | 94 | 1260 | 17 | 2004 | 380 | 91 | 8043 | 1486 | 71 | 394 | 222 | 175 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
