"""Find the sample that kills the training process.

Loads every image and label of a split, reports anything unusual (unreadable file,
odd dtype/channels, huge size, degenerate or out-of-image boxes) and then runs the
real augmentation pipeline over each sample. The id is printed *before* the work and
flushed, so if the process dies (segfault), the last id printed is the culprit.

    python check_dataset.py --data "<export>" --split-dir splits_v1 --split train \
        --extra-train augmented --strong-aug --repeat 4
"""
import argparse
import os.path as osp
import sys
import traceback

import cv2
import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from rtmdet_obb.data import StrongAug, YoloObbDataset  # noqa: E402


def describe(img, boxes, labels, path):
    """Return a list of warnings about one raw sample."""
    out = []
    if img is None:
        return [f'cv2.imread returned None (corrupt or unsupported): {path}']
    h, w = img.shape[:2]
    if img.ndim != 3 or img.shape[2] != 3:
        out.append(f'unexpected shape {img.shape}')
    if img.dtype != np.uint8:
        out.append(f'unexpected dtype {img.dtype}')
    if not img.flags['C_CONTIGUOUS']:
        out.append('not C-contiguous')
    if max(h, w) > 6000:
        out.append(f'very large image {w}x{h}')
    if min(h, w) < 32:
        out.append(f'very small image {w}x{h}')
    if len(boxes):
        if not np.isfinite(boxes).all():
            out.append('non-finite values in boxes')
        wh = boxes[:, 2:4]
        if (wh <= 1).any():
            out.append(f'degenerate box (w or h <= 1 px): {boxes[wh.min(1) <= 1][:2].tolist()}')
        ctr = boxes[:, :2]
        outside = (ctr[:, 0] < 0) | (ctr[:, 0] > w) | (ctr[:, 1] < 0) | (ctr[:, 1] > h)
        if outside.any():
            out.append(f'{int(outside.sum())} box centre(s) outside the image')
        if (wh.max(1) > 4 * max(h, w)).any():
            out.append('box much larger than the image')
    if len(labels) and (labels < 0).any():
        out.append('negative class id')
    return out


def loader_check(ds, args):
    """Iterate the real DataLoader with workers, printing progress; no model involved."""
    import time
    import torch
    from torch.utils.data import DataLoader
    from rtmdet_obb.data import collate
    if sys.platform == 'darwin' and args.loader_workers > 0:
        torch.multiprocessing.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')
    loader = DataLoader(ds, args.batch, shuffle=True, num_workers=args.loader_workers,
                        collate_fn=collate, drop_last=True, persistent_workers=False)
    print(f'iterating {args.epochs} epoch(s) x {len(loader)} batches with '
          f'{args.loader_workers} worker(s), batch {args.batch}', flush=True)
    t0 = time.time()
    for ep in range(args.epochs):
        for bi, batch in enumerate(loader):
            if (bi + 1) % 50 == 0:
                print(f'  epoch {ep + 1} batch {bi + 1}/{len(loader)} '
                      f'({(time.time() - t0) / (bi + 1):.3f} s/batch)', flush=True)
        print(f'epoch {ep + 1} finished', flush=True)
    print('loader check finished without crashing', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--split-dir')
    p.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    p.add_argument('--extra-train', nargs='*', default=[])
    p.add_argument('--img-size', type=int, default=512)
    p.add_argument('--strong-aug', action='store_true')
    p.add_argument('--mosaic-prob', type=float, default=1.0, help='set to 0 to isolate the mosaic')
    p.add_argument('--mixup-prob', type=float, default=0.5, help='set to 0 to isolate the mixup')
    p.add_argument('--no-hsv', action='store_true', help='disable the HSV jitter (isolates cv2.cvtColor)')
    p.add_argument('--stage2', action='store_true', help='check the light pipeline instead')
    p.add_argument('--repeat', type=int, default=2, help='augmented passes per image')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--quiet', action='store_true', help='only print ids that produce warnings')
    p.add_argument('--loader-workers', type=int, help='instead of the per-sample scan, iterate a real '
                   'DataLoader with this many workers (no model, no MPS): isolates crashes in the data pipeline')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--epochs', type=int, default=2, help='passes over the dataset in --loader-workers mode')
    args = p.parse_args()

    np.random.seed(args.seed)
    aug = StrongAug(stage2=args.stage2, mosaic_prob=args.mosaic_prob, mixup_prob=args.mixup_prob,
                    hsv=(0, 0, 0) if args.no_hsv else (5, 30, 30)) if args.strong_aug else None
    ds = YoloObbDataset(args.data, args.split, args.img_size, train=True, split_dir=args.split_dir,
                        extra_dirs=args.extra_train, strong_aug=aug, filter_empty=False)

    if args.loader_workers is not None:
        loader_check(ds, args)
        return
    print(f'{len(ds)} samples, img_size {args.img_size}, '
          f'{"strong" if args.strong_aug else "basic"}{" stage2" if args.stage2 else ""} pipeline, '
          f'{args.repeat} pass(es) each', flush=True)

    problems, empty = [], []
    for i in range(len(ds)):
        item = ds.items[i]
        if not args.quiet:
            print(f'[{i + 1}/{len(ds)}] {item["id"]}', flush=True)
        try:
            img, boxes, labels = ds.load(i)
        except Exception as e:
            problems.append((item['id'], f'load failed: {e}'))
            print(f'  !! load failed: {e}', flush=True)
            continue
        for w in describe(img, boxes, labels, item['img']):
            problems.append((item['id'], w))
            print(f'  !! {w}', flush=True)
        if not len(boxes):
            empty.append(item['id'])
        for r in range(args.repeat):
            try:
                sample = ds[i]
            except Exception:
                problems.append((item['id'], 'augmentation raised'))
                print(f'  !! augmentation raised on pass {r}:\n{traceback.format_exc()}', flush=True)
                break
            b = sample['boxes'].numpy()
            if len(b) and not np.isfinite(b).all():
                problems.append((item['id'], 'non-finite box after augmentation'))
                print('  !! non-finite box after augmentation', flush=True)
            im = sample['image']
            if im.shape != (3, args.img_size, args.img_size):
                problems.append((item['id'], f'bad tensor shape {tuple(im.shape)}'))
                print(f'  !! bad tensor shape {tuple(im.shape)}', flush=True)

    print(f'\ndone: {len(problems)} warning(s) over {len(ds)} samples', flush=True)
    for pid, msg in problems:
        print(f'  {pid}: {msg}')
    if empty:
        print(f'{len(empty)} sample(s) without boxes: {empty[:5]}{" ..." if len(empty) > 5 else ""}')


if __name__ == '__main__':
    main()
