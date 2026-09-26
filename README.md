# YOLOX-OBB Banknote Detector

Pure PyTorch implementation of **YOLOX-OBB** for oriented banknote detection. It combines the
network and loss of [DDGRCF/YOLOX_OBB](https://github.com/DDGRCF/YOLOX_OBB) (plus the official
[Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) networks with an oriented head)
with the training runtime of the RTMDet-R repo (`testbill`, branch `feature/rt_refactor`). It
runs on CPU, CUDA and Apple MPS, without compiled operators, BboxToolkit, mmcv or YAML configs.

## What comes from each repo

| Piece | Origin |
| --- | --- |
| Network (YOLOv5-s backbone with ReLU, PAN, decoupled head) | YOLOX_OBB `configs/modules/yoloxs_obb.yaml` |
| SimOTA label assignment (centre inside the rotated box + radius 2.5, dynamic-k) | YOLOX_OBB `OBBDetectX.get_assignments` |
| Losses: BCE obj + BCE cls (IoU target) + 5 x PolyIoU + extra L1 | YOLOX_OBB `configs/losses/yolox_losses_obb.yaml` |
| SGD nesterov, `yoloxwarmcos`, EMA 0.9998, lr 0.01/64 per image | YOLOX_OBB `yolox_obb_base.py`, `trainer.py` |
| Post-processing: score = obj x cls, class-aware rotated NMS | YOLOX_OBB `obbpostprocess` |
| YOLOv8-OBB dataset, augmentations (mosaic, mixup, rotation, HSV, flip, stage 2) | testbill |
| Training loop, `log.jsonl`, TensorBoard, `best_epoch_N.pth` by fitness | testbill |
| Rotated mAP evaluation, inference, ONNX export, tests | testbill |
| Differentiable rotated IoU, pairwise IoU and NMS in pure PyTorch | testbill |

The last `--stage2-epochs` epochs are the YOLOX "no aug" phase: the L1 loss is switched on,
the LR stays at its minimum and, with `--strong-aug`, training switches to the light pipeline.

Deliberate differences from the original YOLOX_OBB:

* The augmentations are the testbill ones, not the YOLOX ones (copy-paste, random affine).
  There is no multi-scale training either.
* Box angles are clockwise, as in testbill. The network keeps the original (counter-clockwise)
  semantics and the sign is flipped when decoding, so the DOTA weights remain valid.
* In the original, the L1 target of the angle stays at 0 (only x, y, w, h are filled in).
  Here the real ground-truth angle is used.

## Architectures (`--model`)

| `--model` | Network | Parameters | Initial weights (default `--init`) |
| --- | --- | --- | --- |
| `ddgrcf_s` (default) | YOLOX-OBB-s by DDGRCF (YOLOv5-s, ReLU) | 8.06 M | `yolox_s_dota1_0.pth` (DOTA, OBB) |
| `yolox_nano` | Official YOLOX nano (depthwise CSPDarknet, SiLU) | 0.90 M | `yolox_nano.pth` (COCO) |
| `yolox_tiny` | Official YOLOX tiny | 5.1 M | `yolox_tiny.pth` (COCO) |
| `yolox_s` | Official YOLOX s | 8.9 M | `yolox_s.pth` (COCO) |

`ddgrcf_s` keeps the state-dict layout of YOLOX_OBB (`model.0.conv.weight` ...
`model.33.cls_preds.2.bias`), so its DOTA checkpoint loads directly. YOLOX_OBB only publishes
the **s** OBB variant.

The official variants (`yolox_obb/official.py`) are CSPDarknet + PAFPN + `YOLOXHead` from
Megvii YOLOX with the same weight names. The only change is `reg_preds`, which has 5 channels
(x, y, w, h, angle): when the COCO weights are loaded, the 4 pretrained xywh channels are kept
and the angle channel starts at 0 (horizontal boxes). The COCO weights of release `0.1.1rc0`
(input BGR 0..255, no normalisation) are downloaded from GitHub:

```bash
curl -L -o models/yolox_obb/checkpoints/yolox_nano.pth \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
curl -L -o models/yolox_obb/checkpoints/yolox_s.pth \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

All of them share training, losses, SimOTA and post-processing (`OBBDetector` in `model.py`).
The architecture is stored in every checkpoint, so `evaluate.py`, `infer.py` and
`export_onnx.py` detect it on their own (older checkpoints without it are `ddgrcf_s`).

## Layout

```text
data/
  dataset/                  YOLOv8-OBB export with train/, valid/, test/
  augmented/                offline augmentations (optional)
  external_dataset/         external evaluation: images/ + labels/

models/yolox_obb/
  checkpoints/              initial weights (e.g. yolox_s_dota1_0.pth)
  experiments/              one directory per run

src/yolox_mps/              training, evaluation, inference and export scripts
  yolox_obb/                model, SimOTA, losses, data, training engine
tests/                      runtime tests
```

## Environment

The environment is managed with `uv` (Python 3.11):

```bash
uv sync --extra export --extra test
# with TensorBoard:
uv sync --extra export --extra test --extra monitoring
```

On macOS, Pillow is used for resizing (`YOLOX_RESIZE_BACKEND=pil`, the default there) and
`PYTORCH_ENABLE_MPS_FALLBACK=1` is recommended.

The pinned `torch` wheel from PyPI is CPU-only on Windows. For CUDA, install a PyTorch build
that matches the NVIDIA driver (`nvidia-smi` shows the highest supported CUDA version).

## DOTA weights

The YOLOX_OBB model zoo is on Baidu Pan: <https://pan.baidu.com/s/1k1k1JCq56Z-g9NrRtHNWhQ>
(code `tdm6`). Download `YOLOX_s_dota1_0` and save it as:

```text
models/yolox_obb/checkpoints/yolox_s_dota1_0.pth
```

With 1 class, only the 6 `cls_preds` layers are dropped (15 classes in DOTA). The rest of the
network, including the regression, is reused. Without pretrained weights, `--init none` trains
from scratch, which is clearly worse with a few hundred images.

## Training

`--device auto` picks CUDA, then MPS, then CPU. `train.py` uses `data/dataset` by default,
resolves `--init` in `models/yolox_obb/checkpoints` and writes to
`models/yolox_obb/experiments/<work-dir>`.

### Roboflow export + frozen `v1` split

With the full export and the frozen split (`train.txt`, `valid.txt`, `test.txt` with one id per
line; the `.rf.XXX` hash is ignored when matching):

```powershell
$P = "C:\Users\Clara\OneDrive\Escritorio\paddletest"
uv run python src/yolox_mps/train.py --model yolox_nano `
  --data "$P\Annotated banknotes 2.yolov8-obb" --split-dir "$P\v1" --extra-train "$P\augmented" `
  --work-dir v1_nano100 --img-size 640 --batch 4 --accumulate 2 `
  --epochs 100 --stage2-epochs 10 --strong-aug --mosaic-prob 1 --mixup-prob 0.5 --val-interval 5
```

This gives 355 + 710 training images (the 710 in `augmented` come only from the `v1` training
images, checked against its `manifest.json`), 100 validation and 47 test images.

Validation/test with the same split, and generalisation on the external set `billetesprueba 2`
(143 photos, 59 of them HEIC, read with `pillow-heif`):

```powershell
uv run python src/yolox_mps/evaluate.py v1_nano100/best_epoch_100.pth `
  --data "$P\Annotated banknotes 2.yolov8-obb" --split-dir "$P\v1" --split test --img-size 640
uv run python src/yolox_mps/evaluate.py v1_nano100/best_epoch_100.pth `
  --data "$P\billetesprueba 2.yolov8-obb" --split train --img-size 640 `
  --tensorboard-logdir models/yolox_obb/experiments/v1_nano100/tensorboard --tensorboard-tag external
```

### CUDA (Windows / Linux)

```powershell
uv run python src/yolox_mps/train.py `
  --model ddgrcf_s `
  --work-dir dota_100 `
  --extra-train augmented `
  --img-size 640 `
  --batch 8 `
  --epochs 100 `
  --stage2-epochs 10 `
  --strong-aug `
  --mosaic-prob 1 `
  --mixup-prob 0.5 `
  --device cuda `
  --workers 4 `
  --val-interval 5 `
  --tensorboard
```

On a 4 GB GPU shared with the display, use `--batch 4 --accumulate 2` (same effective batch
and LR, about 1.2 GB peak at 640 px).

### Apple MPS

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python src/yolox_mps/train.py \
  --model yolox_nano --work-dir nano_100 --extra-train augmented \
  --img-size 640 --batch 8 --epochs 100 --stage2-epochs 10 \
  --strong-aug --mosaic-prob 1 --mixup-prob 0.5 \
  --device mps --workers 0 --val-interval 5 --seed 0
```

On macOS, use `--workers 0` with this augmentation pipeline.

### CPU

```bash
uv run python src/yolox_mps/train.py \
  --model yolox_nano --work-dir cpu_run \
  --img-size 512 --batch 2 --epochs 100 --stage2-epochs 10 --strong-aug \
  --device cpu --workers 0
```

The default LR is `0.01 / 64 * batch * accumulate` (YOLOX `basic_lr_per_img`); `--lr` sets it
by hand. Every run writes `args.json`, `log.jsonl`, `epoch_N.pth`, `latest.pth` and
`best_epoch_N.pth`. The best checkpoint is selected on validation with:

```text
fitness = 0.9 * mAP@.5:.95 + 0.1 * mAP@0.50
```

To resume, repeat the command with `--resume <work-dir>/latest.pth`. TensorBoard:

```bash
uv run --extra monitoring tensorboard --logdir models/yolox_obb/experiments
```

## Evaluation

testbill protocol: AP with score >= 0.05 and rotated NMS at IoU 0.1. Precision, recall and F1
are measured at the 0.5 operating threshold.

```bash
uv run python src/yolox_mps/evaluate.py dota_100/best_epoch_100.pth \
  --data data/dataset --split test --img-size 640 --batch 2 --workers 0 \
  --score-thr 0.05 --report-score-thr 0.5 --nms-iou 0.1
```

For a flat external set (`images/` + `labels/`): `--data data/external_dataset --split train`.
With `--tensorboard-logdir models/yolox_obb/experiments/dota_100/tensorboard --tensorboard-tag external`
the metrics are added to the same dashboard.

## Inference

Writes one annotated image per input and a `predictions.json`. Ground truth is drawn in red and
predictions in green.

```bash
uv run python src/yolox_mps/infer.py dota_100/best_epoch_100.pth data/external_dataset/images \
  --out inference_800 --img-size 800 --score-thr 0.5 --nms-iou 0.3 \
  --gt data/external_dataset/labels
```

JPG, PNG and HEIC (through `pillow-heif`) are accepted.

## Export

```bash
# inference weights only (EMA) + metadata
uv run python src/yolox_mps/export_checkpoint.py \
  models/yolox_obb/experiments/v1_gpu100/best_epoch_100.pth models/yolox_obb/checkpoints/banknotes_obb.pth

# ONNX (single file) + .metadata.json contract, next to the checkpoint
uv run python src/yolox_mps/export_onnx.py v1_gpu100/best_epoch_100.pth --out model_640.onnx --img-size 640
```

The decoding is inside the graph, so the client only does what is left outside:

| Step | What it does |
| --- | --- |
| Pre-processing | Photo with EXIF orientation applied → `scale = S / max(h, w)` → resize to `(round(w·scale), round(h·scale))` (INTER_AREA when shrinking) → paste at the top-left of an `S×S` canvas filled with 114 → NCHW float32 BGR **0..255, no normalisation** |
| Network (`images` → `boxes`, `scores`) | `boxes (1, N, 5)`: cx, cy, w, h in input pixels, angle in radians (clockwise). `scores (1, N, C)`: sigmoid(obj) · sigmoid(cls) |
| Post-processing | per row `label = argmax`, `score = max` → `score > score_thr` → class-aware rotated NMS (IoU > `nms_iou`) → `cx, cy, w, h /= scale` → corners `c ± (w/2)(cos a, sin a) ± (h/2)(−sin a, cos a)` |

`N = (S/8)² + (S/16)² + (S/32)²` (8400 at 640 px). The `.metadata.json` records this contract,
the class names and the recommended thresholds (0.5 / 0.3).

`src/yolox_mps/onnx_example.py` is the reference implementation for the app: it only uses
`onnxruntime`, `numpy` and OpenCV (rotated NMS with `cv2.rotatedRectangleIntersection`), reads
JPG and HEIC and draws the results. `--check` compares every photo with the PyTorch model:

```bash
uv run python src/yolox_mps/onnx_example.py models/yolox_obb/experiments/v1_gpu100/model_640.onnx \
  photos/ --out onnx_vis --check models/yolox_obb/experiments/v1_gpu100/best_epoch_100.pth
```

## Tests

```bash
uv run pytest -q
```

They cover forward passes and shapes, the weight layouts of `yoloxs_obb.yaml` and of the
official YOLOX, loss and backward, SimOTA, mAP, data loading, and the ONNX export against
PyTorch. If `models/yolox_obb/checkpoints/yolox_s_dota1_0.pth` exists, loading it is tested too.
