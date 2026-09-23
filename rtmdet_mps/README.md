# RTMDet-R in pure PyTorch — oriented banknote detection on CPU / CUDA / Apple MPS

A self-contained reimplementation of **RTMDet-R** (rotated RTMDet, mmrotate 1.x) with no
mmcv / mmdet / mmengine dependency: only `torch`, `numpy` and `opencv`. Training and
inference run on CUDA, **Apple Silicon (MPS)** and CPU. The compiled CUDA kernels that
mmrotate needs (`box_iou_rotated`, `diff_iou_rotated_2d`, `nms_rotated`) are rewritten in
plain torch (`rtmdet_obb/ops.py`).

Trained here on a single-class euro-banknote dataset (oriented boxes). The **tiny** model
(4.9 M parameters, 19 MB) beats a ResNet-50 Rotated RetinaNet (36 M) both on the test split
and on real phone photos.

## Verification against the reference implementation

| what | result |
|---|---|
| `box_iou_rotated`, `diff_iou_rotated_2d` (values + gradients), `nms_rotated` vs mmcv 1.7.2 CUDA | identical (abs err < 5e-5), same NMS keep sets — `tests/test_ops_vs_mmcv.py` |
| loading the official `rotated_rtmdet_tiny-3x-dota` checkpoint | 482/482 tensors, `strict=True` |
| forward pass (cls / reg / angle, 3 levels) vs mmrotate 1.0.0rc1 + mmdet 3.0 | difference **0.0** |
| loss (`DynamicSoftLabelAssigner` + QFL + `RotatedIoULoss`) vs reference | 0.390993 / 1.590471 vs 0.390995 / 1.590471 |
| `predict` (score filter, decode, rotated NMS) vs reference | same detections, boxes within 0.16 px |

The pure-torch IoU is also more robust than mmcv's kernel on identical thin boxes (float32
corner rounding at large coordinates): corners are computed in a local frame and the
corner-in-box test has a 0.01 px tolerance.

## Install (Mac with MPS, or anything else)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # torch>=2.1 torchvision numpy opencv-python pillow pillow-heif
python -c "import torch; print(torch.backends.mps.is_available())"
```

`--device auto` picks `cuda` > `mps` > `cpu`. The code avoids everything MPS lacks: no
float64, no `torch.topk` (k ≤ 16 limit on some builds), the sequential part of the NMS runs
on CPU over an IoU matrix computed on the device. If your torch version still reports an
unimplemented MPS op, `export PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Data

`train.py` / `evaluate.py` read a **Roboflow YOLOv8-OBB export** directly:

```
<data>/{train,valid,test}/images/*.jpg
<data>/{train,valid,test}/labels/*.txt     # <cls> x1 y1 x2 y2 x3 y3 x4 y4  (normalised)
```

- `--split-dir <dir>` overrides the export's folders with id lists `<dir>/{train,valid,test}.txt`
  (ids are matched by base name, i.e. without the Roboflow `.rf.<hash>` suffix, so the split
  survives re-downloads).
- `--extra-train <dir> ...` appends flat folders (`<dir>/images`, `<dir>/labels`, YOLO-OBB) to
  the training set, e.g. an offline-augmented copy.
- `evaluate.py --dota <dir>` evaluates a folder in DOTA layout (`images/` + `annfiles/` with
  `x1 y1 ... x4 y4 <class> <difficulty>` in pixels).
- `infer.py` accepts any folder of JPG/PNG/HEIC photos; `--gt <labels dir>` overlays the ground
  truth (red) next to the predictions (green).

Images are loaded in BGR (OpenCV) and normalised inside the model with the DOTA checkpoint
statistics. Preprocessing = resize keeping the aspect ratio so the long side equals
`--img-size`, then pad to a square with grey 114.

## Train

```bash
# recommended recipe (run C below): strong augmentation, 512 px, batch 8
python train.py --data "<yolov8-obb export>" --split-dir splits_v1 --extra-train augmented \
    --init checkpoints/rotated_rtmdet_tiny-3x-dota-9d821076.pth \
    --work-dir work_dirs/rtmdet_tiny --img-size 512 --batch 8 --epochs 100 \
    --strong-aug --stage2-epochs 30 --device mps
```

Initial weights: the official mmrotate RTMDet-R tiny DOTA checkpoint
(https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota/rotated_rtmdet_tiny-3x-dota-9d821076.pth,
48 MB). Everything is loaded except the three `rtm_cls` layers (15 → N classes), which are
re-initialised with the focal-loss bias. `--size s|m|l` selects the bigger presets (use the
matching DOTA checkpoint).

Recipe (mmrotate `rotated_rtmdet_*-3x-dota` / `*-100e-aug-dota`): AdamW (wd 0.05, no decay on
norm/bias), linear warmup + cosine over the second half, EMA (momentum 2e-4), QFL +
RotatedIoULoss (weight 2), DynamicSoftLabelAssigner (topk 13). Default lr = 0.00025 ·
effective batch / 8.

Augmentation:

- basic (default): random flip h/v/diagonal (p 0.75) + random rotation ±180° (p 0.5).
- `--strong-aug`: mosaic of 4 images on a 2× canvas (`--mosaic-prob`) → random resize
  (`--resize-range`, default 0.5–1.5) → rotation → random crop → HSV jitter → flip
  (`--flip-prob` 0.5) → mixup (`--mixup-prob` 0.5). The last `--stage2-epochs` epochs switch
  to a light pipeline (resize 0.9–1.1 + rotation + flip), as mmdet's `PipelineSwitchHook`.
  As in mmdet, the mosaic and the mixup take their extra samples from caches of recent ones
  (`--mosaic-cache` 40 / `--mixup-cache` 20 entries per worker), so each sample decodes one
  image, not eight. Each worker holds ~50 MB of cache; lower the caches or `--workers` on a
  small machine.

Other options: `--accumulate N` (gradient accumulation, effective batch = batch·N, verified
equivalent to a real batch), `--resume work_dirs/.../latest.pth`, `--val-interval`,
`--nms-iou` for validation (0.5 default; DOTA's 0.1 suppresses stacked notes), `--workers`.

Validation runs every epoch with the EMA weights; `best_epoch_N.pth` is chosen by
`mAP@.5:.95` (area AP). Checkpoints contain the EMA weights (`state_dict`), the raw weights,
the optimizer and the EMA step, so `--resume` continues exactly. `export_checkpoint.py`
strips a checkpoint to the 19 MB inference weights.

Memory: tiny at 512 px / batch 8 peaks at ~2.4 GB; 640 px / batch 4 at ~2.6 GB (batch 8 at
640 does not fit in 4 GB). On an 8 GB Mac use `--batch 4` (or `--accumulate 2`).

## Evaluate / infer

```bash
# rotated mAP (area) on a split, several NMS thresholds
python evaluate.py checkpoints/rtmdet_r_tiny_banknotes_C_strongaug.pth \
    --data "<export>" --split-dir splits_v1 --split test --img-size 512 --nms-iou 0.3 0.5

# DOTA-layout folder (e.g. real photos), sweep the input size
python evaluate.py checkpoints/rtmdet_r_tiny_banknotes_C_strongaug.pth --data data/photos --dota --img-size 800

# photos (JPG/HEIC) -> images with boxes + predictions.json (obb [cx, cy, w, h, angle_rad] + polygon, original pixels)
python infer.py checkpoints/rtmdet_r_tiny_banknotes_C_strongaug.pth photos/ --out out/ --img-size 800 --score-thr 0.5 --nms-iou 0.3
```

## Trained checkpoints (`checkpoints/`, inference weights only)

Dataset: 502 images / 416×416, 1 class `euro_banknote`, frozen split **`splits_v1/`**
(train 355 / val 100 / test 47, manifest v3, seed 20260910). Note: this is not the same
partition as the repository's top-level `splits/` (351 / 101 / 50); all numbers below use
`splits_v1/`. `augmented/`: 710 offline copies of the train images (rotation, shear,
perspective). External set: 143 real phone photos (up to 4032 px), 208 boxes, never seen in
training. All numbers are area AP, NMS 0.3.

| file | run | train data | img | epochs | GPU time (GTX 1650) |
|---|---|---|---|---|---|
| `rtmdet_r_tiny_banknotes_A_v1.pth` | A | v1 train | 640 | 36 (basic aug), best 35 | 30 min |
| `rtmdet_r_tiny_banknotes_B_v1_aug.pth` | B | v1 + augmented | 640 | 36 (basic aug), best 33 | 76 min |
| **`rtmdet_r_tiny_banknotes_C_strongaug.pth`** | **C** | v1 + augmented | **512** | **100 (strong aug, last 30 light)** | 2 h 30 |
| `rtmdet_r_tiny_banknotes_C_ep72_prelight.pth` | C @72 | same | 512 | before the light stage | — |

**Test split v1** (47 images, 63 boxes)

| model | params | input | AP@0.5 | AP@0.75 | AP@.5:.95 |
|---|---|---|---|---|---|
| Rotated RetinaNet R50 (mmrotate 0.3.4) | 36.4 M | 640 | 0.978 | 0.946 | 0.836 |
| Rotated RetinaNet R18 (mmrotate 0.3.4) | 19.8 M | 640 | 0.957 | 0.866 | 0.773 |
| A | 4.9 M | 640 | 0.968 | 0.920 | 0.793 |
| B | 4.9 M | 640 | 0.968 | 0.952 | 0.854 |
| **C** | 4.9 M | 512 | 0.968 | 0.952 | **0.861** |
| C | 4.9 M | 640 | 0.978 | 0.952 | 0.836 |

**External photos** (143 images, 208 boxes)

| model | input | recall | AP@0.5 | AP@0.75 | AP@.5:.95 |
|---|---|---|---|---|---|
| RetinaNet R50 | 1024 | 0.995 | 0.971 | 0.829 | 0.701 |
| RetinaNet R18 | 1024 | 0.981 | 0.839 | 0.638 | 0.554 |
| A | 1024 | 0.923 | 0.812 | 0.620 | 0.516 |
| B | 1024 | 0.957 | 0.925 | 0.811 | 0.653 |
| C @72 (pre-light) | 800 | 0.995 | 0.975 | 0.913 | **0.736** |
| **C** | **800** | 0.990 | 0.974 | 0.910 | **0.732** |
| C | 640 | 0.981 | 0.962 | 0.894 | 0.715 |
| C | 512 | 0.971 | 0.947 | 0.861 | 0.693 |
| C | 1024 | 0.986 | 0.966 | 0.865 | 0.714 |

## What we learned

- **Offline augmented copies matter mostly out of domain**: A → B is +0.06 on the test split
  but +0.14 on real photos (the shear/perspective copies are not reproduced by the online
  flips/rotations). Note that B also does 3× the iterations per epoch.
- **Strong augmentation (mosaic + resize + mixup) is what closes the gap with the R50**: C is
  the best model on both sets with 7× fewer parameters, and its boxes are much tighter
  (AP@0.75 0.91 vs 0.83 on photos).
- **Inference size**: without strong augmentation the model needs 1024 px on large photos
  (banknotes are ~150 px at 640, ~400 px in training). With strong augmentation the sweet spot
  drops to **640–800** and 1024 no longer helps. 512 works for framed banknotes; it loses the
  small/shadowed ones (recall 0.971 vs 0.990).
- The `--resize-range` of the strong pipeline sets the object sizes seen in training (0.5–1.5
  × ~280 px at 512 → 140–420 px). Lowering it to 0.3 should make 512-px inference as robust as
  800 on phone photos — not tried yet.
- **The light final stage** (last 30 epochs without mosaic/mixup) raised val mAP@.5:.95 from
  0.74 to 0.81 and the test split from 0.82 to 0.86, but left the external set flat
  (0.736 → 0.732): it specialises slightly to the training domain. `C_ep72_prelight` is the
  alternative if generalisation matters most; a shorter light stage (10 epochs) is a cheap
  experiment.
- **More epochs alone will not help**: val is flat from epoch 86 while the train loss keeps
  falling. Next steps by expected gain: real scene photos in training (even a few dozen),
  shorter light stage / `--resize-range 0.3 1.5`, then `--size s` (8.9 M params, ~1.8× cost).
- The best epoch chosen by AP@0.5 saturates early; select by mAP@.5:.95 (done here).

## macOS / MPS troubleshooting

`Segmentation fault` after a few iterations is almost always one of these; isolate it in this
order (two minutes):

1. `--workers 0`. Slower but it removes multiprocessing entirely, so it is the reliable way to
   keep training while debugging. On macOS the code already sets `cv2.setNumThreads(0)`, the
   `spawn` start method, the `file_system` sharing strategy, and turns persistent workers off
   (they are reused across epochs, and that reuse is a common crash point — `--persistent-workers`
   re-enables them). If it dies at an epoch boundary rather than mid-epoch, suspect the workers.
2. `--device cpu` (with `--workers 0`). If it still crashes it is not MPS — look at the data
   (a corrupt JPEG, a HEIC without `pillow-heif`).
3. If it only crashes on `mps`, it is a torch/Metal bug or memory pressure: update torch,
   lower `--batch`, and run with `PYTORCH_ENABLE_MPS_FALLBACK=1`.

If it always dies on the same sample it is the data: run

```bash
python check_dataset.py --data "<export>" --split-dir splits_v1 --split train     --extra-train augmented --img-size 512 --strong-aug --repeat 4
```

It prints each id before processing it (flushed), so the last id printed when the process dies
is the offending file; it also reports unreadable images, odd dtypes/sizes and degenerate or
out-of-image boxes. Over the 1065 training samples used here it reports nothing, so a hit on
your machine means that file (or its label) differs.

The decisive evidence is the macOS crash report: `~/Library/Logs/DiagnosticReports/python-*.ips`
(or Console.app → Crash Reports). The first frames name the faulting library —
`libtorch_cpu`/`MPSGraph` vs `libopencv` vs `CoreFoundation` — which says immediately which of
the three it is.

## Windows notes (only relevant to the machine used for training)

Each DataLoader worker commits ~5 GB of virtual memory for the CUDA DLLs. When the
commit limit is exhausted a worker dies with `WinError 1455` and the main process hangs.
Mitigations already in the code: the old loader's persistent workers are shut down before the
stage-2 switch, validation uses no workers. Do not run other torch processes concurrently;
lower `--workers` if it still happens. `scripts/` contains the exact commands of runs A/B/C.

## Layout

```
rtmdet_obb/
  ops.py         rotated IoU (pairwise + differentiable) and rotated NMS in torch
  model.py       CSPNeXt + CSPNeXtPAFPN + RotatedRTMDetSepBNHead; loss() and predict()
  assigner.py    DynamicSoftLabelAssigner (MPS-safe, no topk)
  losses.py      QualityFocalLoss, RotatedIoULoss
  boxes.py       le90 utils, distance2obb, poly <-> rbox, flip / rotate of boxes
  data.py        YOLOv8-OBB / DOTA datasets, basic + strong augmentation, letterbox
  evaluation.py  rotated mAP (VOC area / 11-point)
  engine.py      device, AdamW param groups, LR schedule, EMA, train / eval loops
  checkpoint.py  loads mmrotate checkpoints without mmengine
train.py / evaluate.py / infer.py / export_checkpoint.py
checkpoints/     trained weights (above); the DOTA init checkpoint is downloaded separately
scripts/         run_gpu_experiments.sh (A, B), run_c.sh (C)
tests/test_ops_vs_mmcv.py   needs an mmcv with compiled ops (run on the Windows/CUDA box)
```
