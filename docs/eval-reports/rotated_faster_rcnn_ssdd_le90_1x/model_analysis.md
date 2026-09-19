# Rotated Faster R-CNN 1× on SSDD

Recipe: [`configs/rotated_faster_rcnn/ssdd_le90_1x.json`](../../../configs/rotated_faster_rcnn/ssdd_le90_1x.json). Finetune from `hf://rotated_faster_rcnn_dota_le90_1x` (1-way `ship` head re-init). Official last-digit **train / held-out test**. `make eval-val` mAP50 **90.34%** (`runs/rotated_faster_rcnn/20260918-130546`, RTX 3090 Ti, 15 m). Test chips are **not** in training. No Hub slug.

Literature Faster R-CNN on SSDD is often ~89% AP50 (Guo et al., *Sensors* 2021: **88.96%** overall). This run is **in band**. Protocol: [Data guide — SSDD](../../user-guide/data.md#ssdd-1x-faster-rcnn).

# Model Analysis Report

- Generated at: `2026-09-18T13:29:57.140476`

## Model metadata
- Experiment dir: `runs/rotated_faster_rcnn/20260918-130546`
- Checkpoint: `runs/rotated_faster_rcnn/20260918-130546/checkpoints/best_mAP_0.90.pth`
- Checkpoint modified: `2026-09-18T13:21:11.237426`
- Config: `runs/rotated_faster_rcnn/20260918-130546/config.json`

## Source data
- Data root: `/path/to/data/Official-SSDD-OPEN`
- Data split: `val`
- Total images: `232`
- Total ground truth objects: `546`
- Total predictions: `622`

## Evaluation setup
- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): `0.50`
- NMS IoU (deduplication): `0.10`
- Threshold sweep: `0.0` to `1.0` step `0.05`

## Key outcomes
- Best threshold (F1): `0.4000`
- Precision at best threshold: `0.9397`
- Recall at best threshold: `0.9139`
- F1 at best threshold: `0.9266`
- F2 at best threshold: `0.9190`
- mAP50: `0.9034` (90.34%)

## GT alignment (mean best IoU vs raw detections)

- Global mean best IoU (any class): `0.7390`
- Global mean best IoU (same class): `0.7390` (median `0.7717`)

Per-class breakdown (each GT: max rotated IoU vs detections on the same image):

| Class | gts | mean_any | mean_same | med_same |
| --- | ---: | ---: | ---: | ---: |
| `ship` | 546 | 0.7390 | 0.7390 | 0.7717 |
| **global** | 546 | 0.7390 | 0.7390 | 0.7717 |

## Per-class metrics (mAP50)

| Class | gts | dets | recall | AP |
| --- | ---: | ---: | ---: | ---: |
| `ship` | 546 | 622 | 0.945 | 0.9034 |
| **mAP** | | | | 0.9034 |

## Per-class best thresholds (max F1 over the same sweep)

| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ship` | 0.4000 | 0.9397 | 0.9139 | 0.9266 | 499 | 32 | 47 |

## Confusion matrix

Computed at score threshold `0.4000` and IoU `0.50`.

Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.

| Actual \ Predicted | ship | Missed |
| --- | ---: | ---: |
| `ship` | 499 | 47 |
| `False Positive` | 32 | 0 |

## Artifacts
- Predictions JSON: `predictions.json`
- Analysis JSON: `analysis_iou0.50.json`
- PR curve: `pr_curve.png`
- Threshold metrics: `threshold_metrics.png`

## Notes
- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.
- Per-class table: best threshold per class maximizes F1 on the same threshold grid (see `best_threshold_per_class` in the analysis JSON).
- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.
