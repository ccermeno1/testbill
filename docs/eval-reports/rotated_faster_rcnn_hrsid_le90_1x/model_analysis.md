# Rotated Faster R-CNN 1× on HRSID

Recipe: [`configs/rotated_faster_rcnn/hrsid_le90_1x.json`](../../../configs/rotated_faster_rcnn/hrsid_le90_1x.json). Finetune from `hf://rotated_faster_rcnn_dota_le90_1x` (1-way `ship` head re-init). Official **train / held-out test** (no val). `make eval-val` mAP50 **78.55%** (`runs/rotated_faster_rcnn/20260918-134544`, RTX 3090 Ti, 1h 32m). Test chips are **not** in training. No Hub slug.

Wei et al. HBB AP50 is **>84.7%** (horizontal COCO boxes). This number is **rotated** IoU on min-area rectangles from polygons — high 70s is in the documented healthy band. Protocol: [Data guide — HRSID](../../user-guide/data.md#hrsid-1x).

# Model Analysis Report

- Generated at: `2026-09-18T16:25:06.080262`

## Model metadata
- Experiment dir: `runs/rotated_faster_rcnn/20260918-134544`
- Checkpoint: `runs/rotated_faster_rcnn/20260918-134544/checkpoints/best_mAP_0.72.pth`
- Checkpoint modified: `2026-09-18T15:18:39.163594`
- Config: `runs/rotated_faster_rcnn/20260918-134544/config.json`

## Source data
- Data root: `/path/to/data/HRSID_JPG`
- Data split: `val`
- Total images: `1962`
- Total ground truth objects: `5918`
- Total predictions: `7572`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.6000`
- Precision at best threshold: `0.8954`
- Recall at best threshold: `0.7410`
- F1 at best threshold: `0.8109`
- F2 at best threshold: `0.7674`
- mAP50: `0.7855` (78.55%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.6753`
- Global mean best IoU (same class): `0.6753` (median `0.7619`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `ship` | 5918 | 0.6753 | 0.6753 | 0.7619 |
| **global** | 5918 | 0.6753 | 0.6753 | 0.7619 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `ship` | 5918 | 7572 | 0.826 | 0.7855 |
| **mAP** | | | | 0.7855 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ship` | 0.6000 | 0.8954 | 0.7410 | 0.8109 | 4385 | 512 | 1533 |

## Confusion matrix

Computed at score threshold `0.6000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | ship | Missed |
| --- | ---: | ---: |
| `ship` | 4385 | 1533 |
| `False Positive` | 512 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
