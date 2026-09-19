# Model Analysis Report

Hub slug **`oriented_rcnn_dota_le90_3x`**. Official DOTA v1.0 Task 1 **74.88%** (AP75 51.23, COCO mAP 46.91). Deploy `production.score_threshold` **0.55** (this sweep’s best F1 0.60 − 0.05). The mAP50 below is leaky eval-val (val tiles are in train), not the published number.

## Official Task 1 (hidden test)

| Class | AP50 |
| --- | ---: |
| plane | 0.8884 |
| baseball-diamond | 0.8418 |
| bridge | 0.5443 |
| ground-track-field | 0.7512 |
| small-vehicle | 0.7410 |
| large-vehicle | 0.7844 |
| ship | 0.8856 |
| tennis-court | 0.9057 |
| basketball-court | 0.7979 |
| storage-tank | 0.7874 |
| soccer-ball-field | 0.6054 |
| roundabout | 0.6225 |
| harbor | 0.7692 |
| swimming-pool | 0.7329 |
| helicopter | 0.5752 |
| **mAP50** | **0.7488** |

COCO-style: AP50 **0.7488**, AP75 **0.5123**, mAP **0.4691**.

- Generated at: `2026-09-17T10:51:42.847279`

## Model metadata
- Experiment dir: `runs/oriented_rcnn/20260911-102320`
- Checkpoint: `runs/oriented_rcnn/20260911-102320/checkpoints/best_mAP_0.80.pth`
- Checkpoint modified: `2026-09-15T16:27:56.121174`
- Config: `runs/oriented_rcnn/20260911-102320/config.json`

## Source data
- Data root: `/path/to/data/DOTA-v1.0-tiled`
- Data split: `val`
- Total images: `7669`
- Total ground truth objects: `57768`
- Total predictions: `91500`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.6000`
- Precision at best threshold: `0.8172`
- Recall at best threshold: `0.8840`
- F1 at best threshold: `0.8493`
- F2 at best threshold: `0.8698`
- mAP50: `0.8292` (82.92%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.8025`
- Global mean best IoU (same class): `0.8008` (median `0.8455`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 0.8523 | 0.8523 | 0.8719 |
| `basketball-court` | 278 | 0.9079 | 0.9079 | 0.9202 |
| `bridge` | 666 | 0.7396 | 0.7378 | 0.7933 |
| `ground-track-field` | 216 | 0.8550 | 0.8537 | 0.8860 |
| `harbor` | 4298 | 0.7833 | 0.7822 | 0.8280 |
| `helicopter` | 157 | 0.8031 | 0.8026 | 0.8301 |
| `large-vehicle` | 9398 | 0.8361 | 0.8329 | 0.8570 |
| `plane` | 4731 | 0.8637 | 0.8637 | 0.8978 |
| `roundabout` | 256 | 0.8122 | 0.8122 | 0.8785 |
| `ship` | 18534 | 0.8307 | 0.8304 | 0.8463 |
| `small-vehicle` | 11357 | 0.7575 | 0.7534 | 0.8039 |
| `soccer-ball-field` | 260 | 0.8546 | 0.8498 | 0.9017 |
| `storage-tank` | 5031 | 0.6695 | 0.6695 | 0.8399 |
| `swimming-pool` | 693 | 0.6870 | 0.6870 | 0.7464 |
| `tennis-court` | 1529 | 0.9211 | 0.9174 | 0.9419 |
| **global** | 57768 | 0.8025 | 0.8008 | 0.8455 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 364 | 716 | 0.989 | 0.8163 |
| `basketball-court` | 278 | 416 | 0.996 | 0.8930 |
| `bridge` | 666 | 2583 | 0.913 | 0.7599 |
| `ground-track-field` | 216 | 447 | 0.972 | 0.8432 |
| `harbor` | 4298 | 6301 | 0.945 | 0.8543 |
| `helicopter` | 157 | 200 | 0.975 | 0.9072 |
| `large-vehicle` | 9398 | 14395 | 0.979 | 0.9007 |
| `plane` | 4731 | 5352 | 0.968 | 0.8871 |
| `roundabout` | 256 | 527 | 0.934 | 0.8203 |
| `ship` | 18534 | 30061 | 0.989 | 0.7141 |
| `small-vehicle` | 11357 | 19738 | 0.939 | 0.8589 |
| `soccer-ball-field` | 260 | 505 | 0.958 | 0.8874 |
| `storage-tank` | 5031 | 7152 | 0.777 | 0.7064 |
| `swimming-pool` | 693 | 1333 | 0.872 | 0.7271 |
| `tennis-court` | 1529 | 1774 | 0.986 | 0.8621 |
| **mAP** | | | | 0.8292 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 0.9500 | 0.7884 | 0.9313 | 0.8539 | 339 | 91 | 25 |
| `basketball-court` | 0.9500 | 0.8907 | 0.9964 | 0.9406 | 277 | 34 | 1 |
| `bridge` | 0.8500 | 0.7039 | 0.7853 | 0.7424 | 523 | 220 | 143 |
| `ground-track-field` | 0.9000 | 0.8000 | 0.9259 | 0.8584 | 200 | 50 | 16 |
| `harbor` | 0.6000 | 0.8577 | 0.8748 | 0.8662 | 3760 | 624 | 538 |
| `helicopter` | 0.3000 | 0.9613 | 0.9490 | 0.9551 | 149 | 6 | 8 |
| `large-vehicle` | 0.7500 | 0.9489 | 0.9224 | 0.9355 | 8669 | 467 | 729 |
| `plane` | 0.6500 | 0.9433 | 0.9535 | 0.9484 | 4511 | 271 | 220 |
| `roundabout` | 0.9000 | 0.7735 | 0.8672 | 0.8177 | 222 | 65 | 34 |
| `ship` | 0.7500 | 0.7250 | 0.9340 | 0.8163 | 17310 | 6567 | 1224 |
| `small-vehicle` | 0.4500 | 0.8644 | 0.8117 | 0.8373 | 9219 | 1446 | 2138 |
| `soccer-ball-field` | 0.9500 | 0.9414 | 0.8654 | 0.9018 | 225 | 14 | 35 |
| `storage-tank` | 0.6000 | 0.9037 | 0.6921 | 0.7839 | 3482 | 371 | 1549 |
| `swimming-pool` | 0.7500 | 0.7770 | 0.7590 | 0.7679 | 526 | 151 | 167 |
| `tennis-court` | 0.7500 | 0.9272 | 0.9745 | 0.9503 | 1490 | 117 | 39 |

## Confusion matrix

Computed at score threshold `0.6000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | baseball-diamond | basketball-court | bridge | ground-track-field | harbor | helicopter | large-vehicle | plane | roundabout | ship | small-vehicle | soccer-ball-field | storage-tank | swimming-pool | tennis-court | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseball-diamond` | 357 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| `basketball-court` | 0 | 277 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `bridge` | 0 | 0 | 569 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 97 |
| `ground-track-field` | 0 | 0 | 0 | 204 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| `harbor` | 0 | 0 | 0 | 0 | 3760 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 535 |
| `helicopter` | 0 | 0 | 0 | 0 | 0 | 141 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 |
| `large-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 8833 | 0 | 0 | 0 | 35 | 0 | 0 | 0 | 0 | 530 |
| `plane` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4519 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 212 |
| `roundabout` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 231 | 0 | 0 | 0 | 1 | 0 | 0 | 24 |
| `ship` | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 17718 | 0 | 0 | 0 | 0 | 0 | 813 |
| `small-vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 86 | 0 | 0 | 0 | 8668 | 0 | 0 | 0 | 0 | 2603 |
| `soccer-ball-field` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 243 | 0 | 0 | 0 | 16 |
| `storage-tank` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3482 | 0 | 0 | 1549 |
| `swimming-pool` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 553 | 0 | 140 |
| `tennis-court` | 4 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1492 | 27 |
| `False Positive` | 171 | 48 | 413 | 73 | 621 | 3 | 663 | 285 | 107 | 7238 | 894 | 59 | 370 | 235 | 128 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
- **eval-val / Task 1 score floor is 0.05** (`evaluation.preds_score_threshold` null → 0.05; NMS IoU 0.10). In-train periodic mAP on this run used **0.7** because the Sept 11 code still let `production.score_threshold` override train-val; that does not affect published eval-val or Task 1.
