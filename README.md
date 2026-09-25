# RTMDet-R Banknote Detector

Pure PyTorch implementation of RTMDet-R for oriented banknote detection. The
runtime supports CPU, CUDA and Apple MPS.

## Repository layout

```text
data/
  Annotated banknotes 2.yolov8-obb/   training dataset
  augmented/                          offline training augmentations
  external_dataset/                   external evaluation images and labels
  train/, valid/, test/            physical train, valid and test split

models/rtmdet/
  checkpoints/                        reusable initial weights
  experiments/                        checkpoints and results per run

src/rtmdet_mps/                       training and inference code
tests/                                runtime tests
docs/                                 detailed RTMDet notes
```

The datasets, checkpoints and experiment outputs are local files. They are not
part of the source distribution.

## Environment

The environment is managed with `uv`. Python is pinned to the 3.11 series in

Create or update the environment from the repository root:

```bash
uv sync --extra export --extra test
```

The `export` extra installs ONNX tools. The `test` extra installs pytest.
For TensorBoard monitoring, add the `monitoring` extra:

```bash
uv sync --extra export --extra test --extra monitoring
```

On macOS, Pillow is used for image resizing by default. This avoids the ARM
OpenCV resize implementation that can crash inside `kleidicv`. It also applies
antialiasing when large photos are reduced. To make the choice explicit:

```bash
export RTMDET_RESIZE_BACKEND=pil
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` can be enabled if a PyTorch operation is not
implemented by MPS:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Devices

The same code runs on CPU, Apple MPS and CUDA. Use `--device auto` to select
CUDA first, then MPS, then CPU.

Check the current PyTorch build before training:

```bash
uv run python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print('mps:', torch.backends.mps.is_available())"
```

### CPU

CPU training is supported but slower. Use a small batch and no data-loader
workers when debugging:

```bash
uv run python src/rtmdet_mps/train.py \
  --init rotated_rtmdet_tiny-3x-dota-9d821076.pth \
  --work-dir cpu_run \
  --extra-train augmented \
  --img-size 512 --batch 1 --epochs 200 --stage2-epochs 30 \
  --strong-aug --mosaic-prob 1 --mixup-prob 0.5 \
  --device cpu --workers 0
```

### CUDA on Windows or Linux

Install a PyTorch build matching the CUDA driver on the target machine. CUDA
can use a larger batch and multiple loader workers when memory allows:

```powershell
uv run python src/rtmdet_mps/train.py `
  --init rotated_rtmdet_tiny-3x-dota-9d821076.pth `
  --work-dir cuda_run `
  --extra-train augmented `
  --img-size 512 `
  --batch 8 `
  --epochs 200 `
  --stage2-epochs 30 `
  --strong-aug `
  --mosaic-prob 1 `
  --mixup-prob 0.5 `
  --device cuda `
  --workers 4
```

On macOS, use `--workers 0` with this augmentation pipeline. On CUDA, increase
`--batch` and `--workers` only after checking GPU memory and data-loader
stability.

## Training from DOTA

The DOTA checkpoint is stored at:

```text
models/rtmdet/checkpoints/rotated_rtmdet_tiny-3x-dota-9d821076.pth
```

For a comparison run from COCO pretraining, use:

```text
models/rtmdet/checkpoints/rtmdet_tiny_8xb32-300e_coco_78e30dcc.pth
```

Its backbone and neck match RTMDet tiny. The oriented banknote head is
initialized separately.

```bash
uv run python src/rtmdet_mps/train.py \
  --init rtmdet_tiny_8xb32-300e_coco_78e30dcc.pth \
  --work-dir mps_coco_200 \
  --extra-train augmented \
  --img-size 512 --batch 2 --epochs 200 --stage2-epochs 30 \
  --strong-aug --mosaic-prob 1 --mixup-prob 0.5 \
  --device mps --workers 0 --val-interval 5 --nms-iou 0.3 --seed 0
```

The following command trains for 200 epochs. It uses the frozen `v1` split,
offline augmentations, mosaic, mixup and a light final stage during the last 30
epochs. On macOS, keep `--workers 0` because OpenCV worker processes are not
stable with this augmentation pipeline.

Run it from the repository root:

```bash
export RTMDET_RESIZE_BACKEND=pil
export PYTORCH_ENABLE_MPS_FALLBACK=1

uv run python src/rtmdet_mps/train.py \
  --init rotated_rtmdet_tiny-3x-dota-9d821076.pth \
  --work-dir dota_200 \
  --extra-train augmented \
  --img-size 512 \
  --batch 2 \
  --epochs 200 \
  --stage2-epochs 30 \
  --strong-aug \
  --mosaic-prob 1 \
  --mixup-prob 0.5 \
  --device mps \
  --workers 0 \
  --val-interval 5 \
  --nms-iou 0.3 \
  --seed 0 \
  --tensorboard
```

With the standard layout, `train.py` defaults to `data/dataset`, resolves
checkpoint filenames from `models/rtmdet/checkpoints` and writes named
experiments under `models/rtmdet/experiments`. Add `--extra-train augmented`
when you want to include the offline augmented copies:

```bash
uv run python src/rtmdet_mps/train.py \
  --init rotated_rtmdet_tiny-3x-dota-9d821076.pth \
  --work-dir dota_200 \
  --extra-train augmented \
  --img-size 512 --batch 8 --epochs 200 --stage2-epochs 30 \
  --strong-aug --mosaic-prob 1 --mixup-prob 0.5 \
  --device mps --workers 0 --val-interval 5 --nms-iou 0.1 --seed 0
```

Each run writes `args.json`, `log.jsonl`, checkpoints and validation results to
its experiment directory. The best checkpoint is selected on validation using
the fitness score:

```text
fitness = 0.9 * mAP@.5:.95 + 0.1 * mAP@0.50
```

The fitness value is written to `log.jsonl`, TensorBoard and the best checkpoint
metadata.

With `--tensorboard`, event files are written to the experiment's `tensorboard/`
directory. Start the dashboard from the repository root:

```bash
uv run --extra monitoring tensorboard --logdir models/rtmdet/experiments
```

Then open `http://localhost:6006`.

To add test or external metrics to the same dashboard, pass the experiment's
TensorBoard directory to `evaluate.py`:

```bash
uv run python src/rtmdet_mps/evaluate.py \
  dota_200/best_epoch_30.pth \
  --data external_dataset --split train --img-size 800 \
  --batch 2 --workers 0 --nms-iou 0.3 --device mps \
  --tensorboard-logdir "models/rtmdet/experiments/dota_200/tensorboard" \
  --tensorboard-tag external
```

TensorBoard records `mAP@0.50`, `mAP@0.75`, `mAP@.5:.95`,
`precision@0.50`, `recall@0.50` and `f1@0.50` for validation, test and external
runs. Precision, recall and F1 use the final operating point at IoU 0.50; mAP
is computed over the full score ranking.

To resume a run:

```bash
uv run python src/rtmdet_mps/train.py \
  --resume dota_200/latest.pth \
  --work-dir dota_200 \
  --extra-train augmented \
  --img-size 512 --batch 2 --epochs 200 --stage2-epochs 30 \
  --strong-aug --mosaic-prob 1 --mixup-prob 0.5 \
  --device mps --workers 0 --val-interval 5 --nms-iou 0.3 --seed 0
```

## Evaluation

The frozen split contains 355 training images, 100 validation images and 47
test images. Evaluate a checkpoint on validation or test as follows:

The evaluation follows MMRotate's official operating point for AP: detections
are kept from score `0.05` and rotated NMS uses IoU `0.1`. Precision, recall and
F1 are reported separately at the deployment threshold `0.5`.

```bash
uv run python src/rtmdet_mps/evaluate.py \
  dota_200/best_epoch_30.pth \
  --data dataset \
  --split valid \
  --img-size 512 \
  --batch 2 --workers 0 --score-thr 0.05 --report-score-thr 0.5 \
  --nms-iou 0.1 --device mps
```

Change `--split valid` to `--split test` when the test evaluation is intended.
The test split is not read during training.

The external set is a flat YOLO-OBB directory with `images/` and `labels/`:

```bash
uv run python src/rtmdet_mps/evaluate.py \
  dota_200/best_epoch_30.pth \
  --data external_dataset \
  --split train \
  --img-size 800 \
  --batch 2 --workers 0 --score-thr 0.05 --report-score-thr 0.5 \
  --nms-iou 0.1 --device mps
```

Use `512` for faster inference or `800` for large phone photos. The external
evaluation set contains 143 images.

## Inference images

`infer.py` writes one annotated image per input and a `predictions.json` file.
Ground truth is drawn in red, predictions in green and each prediction includes
its confidence score.

```bash
uv run python src/rtmdet_mps/infer.py \
  "models/rtmdet/experiments/dota_200/best_epoch_30.pth" \
  external_dataset/images \
  --out inference_external_800 \
  --img-size 800 \
  --score-thr 0.5 \
  --nms-iou 0.3 \
  --device mps \
  --gt external_dataset/labels
```

The model accepts JPG, PNG and HEIC input when `pillow-heif` is installed.

## ONNX export

Export uses a trained checkpoint and writes the ONNX model together with its
metadata into the same experiment directory:

```bash
uv run python src/rtmdet_mps/export_onnx.py \
  dota_200/best_epoch_30.pth \
  --out model_800.onnx \
  --img-size 800
```

The ONNX graph contains the RTMDet-R network and pixel normalization. The app
must still perform image resize and padding, decode oriented boxes, run rotated
NMS and map coordinates back to the original image. The generated
`model_800.metadata.json` describes the input and output tensors.

## Tests

Run the runtime tests from the repository root:

```bash
uv run pytest -q
```

The MMCV comparison test is optional and requires a build of MMCV with compiled
rotated operators:

```bash
uv run python tests/test_ops_vs_mmcv.py --device cuda
```
