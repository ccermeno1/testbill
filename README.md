# PP-YOLOE-R Banknote Detector (plain PyTorch)

Oriented-bounding-box detection of euro banknotes with **PP-YOLOE-R**, reimplemented in plain
**PyTorch**. The same code runs on **CPU, CUDA and Apple MPS**, with no dependency on Paddle,
PaddleDetection or any compiled op.

It is a port of the PaddleDetection implementation, checked against it numerically. See
*Parity with Paddle* below.

## Layout

```text
data/                                   datasets (not versioned)
  banknotes_obb/                          images/ + annotations/*.json (COCO with polygons)
  billetesprueba/                         external test set of real photos
models/ppyoloe_r/                       .pt checkpoints
src/ppyoloer_mps/
  ppyoloe_obb/
    boxes.py        OBB geometry: rbox <-> polygon, ProbIoU, exact rotated IoU
    ops.py          rotated NMS and post-processing
    model.py        CSPResNet + CustomCSPPAN + PPYOLOERHead
    assigner.py     RotatedTaskAlignedAssigner
    losses.py       VariFocal + ProbIoU + DFL
    data.py         COCO-polygon dataset, preprocessing and augmentation
    engine.py       device selection, EMA, train/eval loops
    evaluation.py   mAP50/75/50-95, P/R/F1
    checkpoint.py   .pdparams -> .pt conversion, saving and loading
  train.py  evaluate.py  infer.py
  convert_paddle.py                     PaddleDetection .pdparams -> .pt
  export_onnx.py  onnx_example.py       ONNX export and the client-side pre/post-processing
  data_prep/                            dataset preparation (no Paddle)
tests/                                  runtime tests (pytest)
```

## Environment

```bash
uv venv .venv --python 3.11
uv sync --extra test --extra export      # installs from uv.lock
# without the lock:  uv pip install -e ".[test,export]"
# PyTorch build per platform:
#   macOS (MPS) and CPU:  the default wheel is fine
#   CUDA:                 uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
uv run pytest
```

`--device auto` picks CUDA, then MPS, then CPU. On macOS pass `--workers 0`, since the OpenCV
worker processes are not stable there, and if some op is missing on MPS,
`export PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Usage

```bash
# predict on single photos or a folder
uv run python src/ppyoloer_mps/infer.py -w models/ppyoloe_r/ppyoloe_r_s_banknotes_torch.pt \
    --images photos/ --out-dir runs/pred --vis --device auto

# evaluate (mAP50/75/50-95 plus P/R/F1)
uv run python src/ppyoloer_mps/evaluate.py -w models/ppyoloe_r/ppyoloe_r_s_banknotes_torch.pt \
    --data data/banknotes_obb --split valid test --device auto

# train, the recipe used for the shipped model
uv run python src/ppyoloer_mps/train.py \
    --data data/banknotes_obb --train-split train_plus_extra --val-split valid \
    --init models/ppyoloe_r/ppyoloe_r_s_dota.pt \
    --work-dir runs/extra --epochs 60 --mosaic-epochs 50 --batch 4 --device auto --workers 0
```

Other flags worth knowing on `train.py`: `--init` (starting weights, e.g. the converted DOTA
checkpoint; classification layers with a different class count are dropped automatically),
`--resume`, `--img-size`, `--lr`, `--no-ema`, `--val-interval`, `--nms-iou`, `--save-best`.

## Data

The format is COCO with a 4-point polygon per box in `segmentation`, the same as on the Paddle
side, so the JSONs in `data_manifests/annotations/` can be used as they are:

```text
data/banknotes_obb/
  images/<stem>.jpg
  annotations/train.json  valid.json  test.json  train_plus_extra.json  ...
```

## Checkpoints

| file | where it comes from |
|---|---|
| `ppyoloe_r_s_banknotes_torch.pt` | **trained with this repo** (60 epochs, 95.78 / 81.19 / 69.53) |
| `ppyoloe_r_s_banknotes_extra.pt` | the same model trained with Paddle, converted |
| `ppyoloe_r_s_dota.pt` | official DOTA checkpoint, the starting point for training |

To convert another PaddleDetection checkpoint:

```bash
uv run python src/ppyoloer_mps/convert_paddle.py \
    --src model_final.pdparams --out models/ppyoloe_r/other.pt --num-classes 1 --classes euro_banknote
```

Only that script needs Paddle installed (`uv pip install -e ".[convert]"`), and only when the
input is a `.pdparams`. A `.npz` exported on another machine works without it.

## Parity with Paddle

### Inference

Same checkpoint, same images, on CPU:

| check | result |
|---|---|
| state_dict load | 461 tensors, 0 missing, 0 unexpected |
| neck feature maps | max error 6e-5 (values around 12) |
| head scores | max error 3.8e-6 |
| decoded boxes | max error 3e-3 px on ~480 px |
| training loss (cls/iou/dfl) | relative difference below 8e-6, mosaic batches included |
| 20 SGD steps on one batch | matching trajectories (2.4799 -> 0.6123 vs 0.6234) |

### Training from scratch

60 epochs from the DOTA checkpoint, same recipe and same metric, measured on valid:

| | PyTorch port | PaddleDetection |
|---|---|---|
| mAP50 | **95.78** | 95.10 |
| mAP75 | **81.19** | 80.94 |
| mAP50-95 | 69.53 | 69.53 |
| P / R / F1 @.5 | 0.969 / 0.874 / 0.919 | 0.969 / 0.874 / 0.919 |

Getting there took six fidelity fixes against ppdet, each with a regression test. They are
written down because none of them shows up when you compare modules in isolation:

| # | difference | symptom when missed |
|---|---|---|
| 1 | `RRotate` uses `auto_bound`: it shrinks the image instead of cutting corners | partly cut objects trained against their full box |
| 2 | `Poly2RBox` drops boxes with a side under 2 px | ProbIoU hits `log(0)` and the loss goes to **NaN** |
| 3 | in `RandomDistort`, `prob` is the chance of **skipping** the op, and it runs through PIL `ImageEnhance` | washed-out images (mean 127 vs 107, std 54 vs 74) |
| 4 | rotations pad with black, not with grey 114 | same image statistics drifting |
| 5 | `ModelEMA` uses `ema_decay_type='threshold'`, not the exponential ramp | decay 0.34 instead of 0.99, so the EMA tracks the model instead of averaging it |
| 6 | **`RResize` clips the polygon vertices to the canvas** | trained on the part of the object that is not visible: **mAP75 collapses** (10 vs 40) |

Number 6 was the main one. It turned up by measuring the Paddle pipeline stage by stage:
median box area per image went mosaic 0.2348 -> plus rotations 0.1717 -> plus RResize 0.1202.

### Thresholds, aligned with mmrotate

The defaults come from mmrotate's official `test_cfg` (`rotated_retinanet`), so this can be
compared against other rotated detectors on equal terms:

| parameter | value | where |
|---|---|---|
| `score_thr` (before NMS) | 0.05 | `--score-threshold` |
| `nms_iou` | 0.1 | `--nms-iou` |
| `nms_pre` / `max_per_img` | 2000 | `ops.batched_postprocess` |
| threshold for P/R/F1 and production | 0.5 | `--conf` |
| best-checkpoint criterion | `0.9*mAP50-95 + 0.1*mAP50` | `--save-best` |

With stacked banknotes it pays to raise NMS to 0.5: 17 % of the valid boxes overlap a
neighbour above IoU 0.1. Measured on the model trained here:

| dataset | score 0.05 / NMS 0.1 (mmrotate) | score 0.01 / NMS 0.5 |
|---|---|---|
| valid | 90.54 / 80.99 / 67.76 | **95.78 / 81.19 / 69.53** |
| test | 100.00 / 87.48 / 77.46 | **100.00 / 89.42 / 78.12** |
| billetesprueba (external) | 98.34 / 86.02 / 68.64 | **98.69 / 86.00 / 68.78** |

(mAP50 / mAP75 / mAP50-95.) Keep the mmrotate defaults when comparing detectors; for a real
deployment use `--nms-iou 0.5`.

## ONNX export

```bash
uv pip install -e ".[export]"
uv run python src/ppyoloer_mps/export_onnx.py models/ppyoloe_r/ppyoloe_r_s_banknotes_torch.pt \
    --out models/ppyoloe_r/ppyoloe_r_s_banknotes_640.onnx --img-size 640
```

The graph takes **float32 RGB in 0..255**, already resized and padded (normalisation is inside
it), and returns `scores` (B, C, L) with sigmoid applied and `boxes` (B, L, 5) already decoded
in input pixels. That leaves the client three steps: threshold, rotated NMS, and undoing the
resize scale. A `.metadata.json` written next to the `.onnx` spells out the whole contract.

`onnx_example.py` implements those three steps in numpy and cross-checks them against PyTorch:

```bash
uv run python src/ppyoloer_mps/onnx_example.py photo.jpg \
    --onnx models/ppyoloe_r/ppyoloe_r_s_banknotes_640.onnx \
    --checkpoint models/ppyoloe_r/ppyoloe_r_s_banknotes_torch.pt
# ONNX: 2 detections (conf >= 0.5, NMS IoU 0.1)
# PyTorch: 2 detections
#   max box difference: 0.0002 px
#   max score difference: 0.000000
```

Mind the corner convention when reimplementing `rbox2poly` outside Python: getting the sign of
the angle term wrong gives plausible-looking boxes that are rotated the wrong way. A test pins
it down.

## Devices and speed

- **macOS / MPS**: the reason this port exists. Training and inference both run without Paddle.
- **CUDA**: works with a cu118 wheel even on older drivers (checked on 461.33). The cu126
  wheels need driver 525 or newer.
- **CPU**: fine for predicting and evaluating, around 1 s per image at 640 px. Training on CPU
  is slow, roughly 2 s per iteration at batch 2.
