"""Train RTMDet-R on a YOLOv8-OBB export (pure PyTorch: CUDA, Apple MPS or CPU).

Example (Mac, from the official DOTA checkpoint, frozen split v1)::

    python train.py --data "Annotated banknotes 2.yolov8-obb" --split-dir v1 \
        --init checkpoints/rotated_rtmdet_tiny-3x-dota-9d821076.pth \
        --work-dir work_dirs/rtmdet_tiny_banknotes --epochs 36 --batch 8 --img-size 640
"""
import argparse
import faulthandler
import json
import os
import os.path as osp
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from rtmdet_obb import RTMDetR, load_pretrained  # noqa: E402
from rtmdet_obb.data import StrongAug, YoloObbDataset, collate  # noqa: E402
from rtmdet_obb.engine import pick_device, train  # noqa: E402


faulthandler.enable()  # print the Python frame that was running if we get a SIGSEGV


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='Roboflow YOLOv8-OBB export folder')
    p.add_argument('--split-dir', help='folder with train.txt / valid.txt / test.txt (ids); default: export folders')
    p.add_argument('--extra-train', nargs='*', default=[], help='extra flat folders (images/ + labels/) added to train')
    p.add_argument('--classes', type=int, default=1)
    p.add_argument('--size', default='tiny', choices=['tiny', 's', 'm', 'l'])
    p.add_argument('--init', help='checkpoint to initialise from (mmrotate or this repo); cls layer re-init if shapes differ')
    p.add_argument('--resume', help='latest.pth of this repo to resume')
    p.add_argument('--work-dir', default='work_dirs/rtmdet')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--epochs', type=int, default=36)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--accumulate', type=int, default=1, help='gradient accumulation steps (effective batch = batch * accumulate)')
    p.add_argument('--lr', type=float, help='default 0.00025 * effective batch / 8 (mmrotate: 0.004/16 for batch 8)')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--warmup-iters', type=int, default=200)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--persistent-workers', dest='persistent_workers', action='store_true', default=None,
                   help='keep DataLoader workers alive between epochs (default: on, off on macOS)')
    p.add_argument('--no-persistent-workers', dest='persistent_workers', action='store_false',
                   help='respawn the workers every epoch; try this if training dies at an epoch boundary')
    p.add_argument('--mosaic-cache', type=int, default=40, help='samples cached per worker for the mosaic')
    p.add_argument('--mixup-cache', type=int, default=20, help='samples cached per worker for the mixup')
    p.add_argument('--device', default='auto', help='auto | cuda | mps | cpu')
    p.add_argument('--val-interval', type=int, default=1)
    p.add_argument('--no-ema', action='store_true')
    p.add_argument('--grad-clip', type=float, default=None)
    p.add_argument('--nms-iou', type=float, default=0.5, help='NMS IoU for validation (0.1 in DOTA; 0.5 keeps stacked notes)')
    p.add_argument('--no-rotate', action='store_true', help='disable the random rotation augmentation')
    p.add_argument('--strong-aug', action='store_true',
                   help='RTMDet "aug" pipeline: mosaic + random resize + rotate + crop + HSV + flip + mixup, '
                        'switching to the light pipeline for the last --stage2-epochs')
    p.add_argument('--mosaic-prob', type=float, default=1.0)
    p.add_argument('--mixup-prob', type=float, default=0.5)
    p.add_argument('--resize-range', type=float, nargs=2, default=[0.5, 1.5])
    p.add_argument('--stage2-resize-range', type=float, nargs=2, default=[0.9, 1.1])
    p.add_argument('--flip-prob', type=float, default=0.5, help='(strong aug only; the basic pipeline uses 0.75)')
    p.add_argument('--stage2-epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if sys.platform == 'darwin' and args.workers > 0:
        # fork + Core Graphics / Metal in a worker segfaults; spawn is the safe context,
        # and the file_system sharing strategy avoids the fd-passing crashes on macOS
        torch.multiprocessing.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')
    device = pick_device(args.device)
    lr = args.lr if args.lr else 0.00025 * args.batch * args.accumulate / 8
    print(f'device {device}, lr {lr:.2e}, batch {args.batch} x {args.accumulate}, img {args.img_size}')

    if args.persistent_workers is None:  # macOS workers crash on reuse across epochs
        args.persistent_workers = sys.platform != 'darwin'
    aug_kw = dict(mosaic_prob=args.mosaic_prob, mixup_prob=args.mixup_prob, resize_range=args.resize_range,
                  stage2_resize_range=args.stage2_resize_range, flip_prob=args.flip_prob,
                  rotate_prob=0.0 if args.no_rotate else 0.5,
                  mosaic_max_cached=args.mosaic_cache, mixup_max_cached=args.mixup_cache)
    train_ds = YoloObbDataset(args.data, 'train', args.img_size, train=True, split_dir=args.split_dir,
                              rotate_prob=0.0 if args.no_rotate else 0.5, extra_dirs=args.extra_train,
                              strong_aug=StrongAug(**aug_kw) if args.strong_aug else None)
    stage2_loader = stage2_epoch = None
    if args.strong_aug:
        stage2_ds = YoloObbDataset(args.data, 'train', args.img_size, train=True, split_dir=args.split_dir,
                                   extra_dirs=args.extra_train, strong_aug=StrongAug(stage2=True, **aug_kw))
        stage2_epoch = max(0, args.epochs - args.stage2_epochs)
    val_ds = YoloObbDataset(args.data, 'valid', args.img_size, train=False, split_dir=args.split_dir)
    print(f'train {len(train_ds)} images, val {len(val_ds)} images')
    pin = device.type == 'cuda'
    persistent = args.workers > 0 and args.persistent_workers
    train_loader = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate,
                              drop_last=True, pin_memory=pin, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, max(1, args.batch // 2), shuffle=False, num_workers=0, collate_fn=collate,
                            pin_memory=pin)
    if args.strong_aug:
        stage2_loader = DataLoader(stage2_ds, args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate,
                                   drop_last=True, pin_memory=pin, persistent_workers=persistent)

    model = RTMDetR(num_classes=args.classes, size=args.size)
    if args.init and not args.resume:
        load_pretrained(model, args.init)

    os.makedirs(args.work_dir, exist_ok=True)
    with open(osp.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    train(model, train_loader, val_loader, device, args.work_dir, args.epochs, lr, args.weight_decay,
          args.warmup_iters, args.val_interval, use_ema=not args.no_ema, grad_clip=args.grad_clip, resume=args.resume,
          val_kwargs=dict(nms_iou=args.nms_iou), stage2_loader=stage2_loader, stage2_epoch=stage2_epoch,
          accumulate=args.accumulate)


if __name__ == '__main__':
    main()
