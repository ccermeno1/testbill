# Model Analysis Report

Hub slug **`rotated_retinanet_dota_le90_1x`**. Official DOTA v1.0 Task 1 **67.87%** (AP75 40.08, COCO mAP 38.91). Deploy `production.score_threshold` **0.35** (eval-val F1 0.40 − 0.05). **Circum-HBB matching** (`use_hbb_for_matching: true`); this is the published Hub 1× column — ahead of MMRotate HBB (**64.55%**). An **OBB** recipe (`dota_le90_1x_obb.json`, MMRotate `RBboxOverlaps2D`) is still underway and is **not** this Hub slug. The mAP50 below is leaky eval-val (val tiles are in train), not the published number.

## Official Task 1 (hidden test)

| Class | AP50 |
| --- | ---: |
| plane | 0.8873 |
| baseball-diamond | 0.7680 |
| bridge | 0.4309 |
| ground-track-field | 0.7009 |
| small-vehicle | 0.6922 |
| large-vehicle | 0.5479 |
| ship | 0.7413 |
| tennis-court | 0.9054 |
| basketball-court | 0.7491 |
| storage-tank | 0.8198 |
| soccer-ball-field | 0.5314 |
| roundabout | 0.6310 |
| harbor | 0.6145 |
| swimming-pool | 0.6665 |
| helicopter | 0.4943 |
| **mAP50** | **0.6787** |

COCO-style: AP50 **0.6787**, AP75 **0.4008**, mAP **0.3891**.

- Generated at: `2026-09-13T01:01:27.976112`

## Model metadata
- Experiment dir: `runs/rotated_retinanet/20260912-105343`
- Checkpoint: `runs/rotated_retinanet/20260912-105343/checkpoints/best_mAP_0.71.pth`
- Checkpoint modified: `2026-09-12T17:24:48.791920`
- Config: `runs/rotated_retinanet/20260912-105343/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `308406`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.4000`
- Precision at best threshold: `0.7141`
- Recall at best threshold: `0.7303`
- F1 at best threshold: `0.7221`
- F2 at best threshold: `0.7270`
- mAP50: `0.6820` (68.20%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.6903`
- Global mean best IoU (same class): `0.6809` (median `0.7658`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.7626 | 0.7553 | 0.7862 |
| `basketball-court` | 278 | 0.8394 | 0.8353 | 0.8675 |
| `bridge` | 666 | 0.5991 | 0.5975 | 0.6498 |
| `ground-track-field` | 216 | 0.7758 | 0.7612 | 0.8088 |
| `harbor` | 4298 | 0.6714 | 0.6665 | 0.7056 |
| `helicopter` | 157 | 0.7212 | 0.6378 | 0.7293 |
| `large-vehicle` | 9398 | 0.6888 | 0.6712 | 0.7525 |
| `plane` | 4731 | 0.8025 | 0.7969 | 0.8460 |
| `roundabout` | 256 | 0.7309 | 0.7143 | 0.8035 |
| `ship` | 18534 | 0.6836 | 0.6749 | 0.7759 |
| `small-vehicle` | 11357 | 0.6630 | 0.6545 | 0.7376 |
| `soccer-ball-field` | 260 | 0.7218 | 0.6839 | 0.7991 |
| `storage-tank` | 5031 | 0.6340 | 0.6307 | 0.7473 |
| `swimming-pool` | 693 | 0.6146 | 0.6034 | 0.6713 |
| `tennis-court` | 1529 | 0.8760 | 0.8695 | 0.9050 |
| **global** | 57768 | 0.6903 | 0.6809 | 0.7658 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 6245 | 0.945 | 0.7554 |
| `basketball-court` | 278 | 3128 | 0.960 | 0.8244 |
| `bridge` | 666 | 22263 | 0.731 | 0.4681 |
| `ground-track-field` | 216 | 5341 | 0.940 | 0.7067 |
| `harbor` | 4298 | 21125 | 0.846 | 0.6821 |
| `helicopter` | 157 | 1346 | 0.796 | 0.6127 |
| `large-vehicle` | 9398 | 50905 | 0.820 | 0.6315 |
| `plane` | 4731 | 13410 | 0.939 | 0.8852 |
| `roundabout` | 256 | 3052 | 0.832 | 0.6338 |
| `ship` | 18534 | 44486 | 0.830 | 0.6098 |
| `small-vehicle` | 11357 | 93796 | 0.816 | 0.6422 |
| `soccer-ball-field` | 260 | 4718 | 0.800 | 0.6679 |
| `storage-tank` | 5031 | 24219 | 0.729 | 0.6500 |
| `swimming-pool` | 693 | 6870 | 0.811 | 0.5973 |
| `tennis-court` | 1529 | 7502 | 0.967 | 0.8633 |
| **mAP** | | | | 0.6820 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.5500 | 0.7686 | 0.7665 | 0.7675 | 279 | 84 | 85 |
| `basketball-court` | 0.5000 | 0.8529 | 0.8345 | 0.8436 | 232 | 40 | 46 |
| `bridge` | 0.4500 | 0.5399 | 0.5075 | 0.5232 | 338 | 288 | 328 |
| `ground-track-field` | 0.5000 | 0.6966 | 0.7546 | 0.7244 | 163 | 71 | 53 |
| `harbor` | 0.5000 | 0.7409 | 0.7396 | 0.7402 | 3179 | 1112 | 1119 |
| `helicopter` | 0.5000 | 0.8587 | 0.5032 | 0.6345 | 79 | 13 | 78 |
| `large-vehicle` | 0.4000 | 0.7066 | 0.6931 | 0.6998 | 6514 | 2705 | 2884 |
| `plane` | 0.4500 | 0.8987 | 0.9055 | 0.9021 | 4284 | 483 | 447 |
| `roundabout` | 0.4500 | 0.6455 | 0.6758 | 0.6603 | 173 | 95 | 83 |
| `ship` | 0.4500 | 0.6539 | 0.7911 | 0.7160 | 14662 | 7759 | 3872 |
| `small-vehicle` | 0.3500 | 0.7603 | 0.6378 | 0.6937 | 7244 | 2284 | 4113 |
| `soccer-ball-field` | 0.5500 | 0.8223 | 0.6231 | 0.7090 | 162 | 35 | 98 |
| `storage-tank` | 0.3000 | 0.7717 | 0.6289 | 0.6930 | 3164 | 936 | 1867 |
| `swimming-pool` | 0.4500 | 0.6359 | 0.5974 | 0.6161 | 414 | 237 | 279 |
| `tennis-court` | 0.6000 | 0.9162 | 0.9366 | 0.9263 | 1432 | 131 | 97 |

## Confusion matrix

Computed at score threshold `0.4000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 308 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 56 |
| `basketball-court` | 0 | 253 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 23 |
| `bridge` | 0 | 0 | 366 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 300 |
| `ground-track-field` | 0 | 1 | 0 | 177 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 31 |
| `harbor` | 0 | 0 | 1 | 0 | 3408 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 881 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 84 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 6510 | 0 | 0 | 0 | 71 | 0 | 0 | 0 | 0 | 2817 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 4306 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 424 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 180 | 0 | 0 | 0 | 0 | 0 | 0 | 76 |
| `ship` | 0 | 0 | 2 | 0 | 11 | 0 | 0 | 0 | 0 | 14868 | 0 | 0 | 0 | 4 | 0 | 3649 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 180 | 0 | 0 | 0 | 6927 | 0 | 0 | 0 | 0 | 4250 |
| `soccer-ball-field` | 0 | 1 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 175 | 0 | 0 | 0 | 77 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2706 | 0 | 0 | 2325 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 428 | 0 | 265 |
| `tennis-court` | 4 | 6 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1449 | 69 |
| `False Positive` | 192 | 65 | 418 | 122 | 1655 | 33 | 2529 | 528 | 119 | 8252 | 1675 | 122 | 382 | 301 | 208 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
