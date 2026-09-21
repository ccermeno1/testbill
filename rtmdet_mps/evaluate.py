"""Evaluate a checkpoint on a split of a YOLOv8-OBB export (rotated mAP, area mode).

    python evaluate.py work_dirs/rtmdet_tiny_banknotes/best_epoch_XX.pth --data "Annotated banknotes 2.yolov8-obb" \
        --split-dir v1 --split test --img-size 640 --nms-iou 0.3 0.5
"""
import argparse
import json
import os.path as osp
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from rtmdet_obb import RTMDetR, load_state_dict_file  # noqa: E402
from rtmdet_obb.data import DotaObbDataset, YoloObbDataset, collate  # noqa: E402
from rtmdet_obb.engine import evaluate, pick_device  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('--data', required=True)
    p.add_argument('--split-dir')
    p.add_argument('--dota', action='store_true', help='--data is a DOTA-layout folder (images/ + annfiles/), no splits')
    p.add_argument('--class-names', default='euro_banknote', help='comma separated (DOTA layout only)')
    p.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    p.add_argument('--classes', type=int, default=1)
    p.add_argument('--size', default='tiny', choices=['tiny', 's', 'm', 'l'])
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--score-thr', type=float, default=0.05)
    p.add_argument('--nms-iou', type=float, nargs='+', default=[0.5])
    p.add_argument('--mode', default='area', choices=['area', '11points'])
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    device = pick_device(args.device)
    model = RTMDetR(num_classes=args.classes, size=args.size)
    model.load_state_dict(load_state_dict_file(args.checkpoint))
    model.to(device).eval()
    if args.dota:
        ds = DotaObbDataset(args.data, args.class_names.split(','), args.img_size)
    else:
        ds = YoloObbDataset(args.data, args.split, args.img_size, train=False, split_dir=args.split_dir)
    loader = DataLoader(ds, args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate)
    print(f'{args.split}: {len(ds)} images on {device}')
    for nms in args.nms_iou:
        m = evaluate(model, loader, device, args.classes, score_thr=args.score_thr, nms_iou=nms, mode=args.mode)
        print(f'nms {nms}: ' + ' '.join(f'{k} {v:.4f}' for k, v in m.items()))


if __name__ == '__main__':
    main()
