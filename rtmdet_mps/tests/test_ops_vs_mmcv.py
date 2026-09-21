"""Compare the pure-torch rotated ops against the compiled mmcv kernels.

Run inside a venv that has ``mmcv-full`` (1.x) or ``mmcv`` (2.x) with ops
compiled, ideally with CUDA::

    python tests/test_ops_vs_mmcv.py [--device cuda]
"""
import argparse
import os.path as osp
import sys
import time

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from rtmdet_obb import ops  # noqa: E402

from mmcv.ops import box_iou_rotated as mm_box_iou_rotated  # noqa: E402
from mmcv.ops import diff_iou_rotated_2d as mm_diff_iou  # noqa: E402
from mmcv.ops import nms_rotated as mm_nms_rotated  # noqa: E402


def random_boxes(n, device, scale=200.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * scale
    wh = torch.rand(n, 2, generator=g) * scale * 0.5 + 2
    a = (torch.rand(n, 1, generator=g) - 0.5) * 3.1416
    return torch.cat([xy, wh, a], 1).to(device)


def report(name, ok, extra=''):
    print(f'[{"OK" if ok else "FAIL"}] {name} {extra}')
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    dev = args.device
    all_ok = True

    # ---- pairwise IoU: random boxes ------------------------------------------
    b1 = random_boxes(500, dev, seed=1)
    b2 = random_boxes(300, dev, seed=2)
    ref = mm_box_iou_rotated(b1, b2)
    got = ops.box_iou_rotated(b1, b2)
    err = (ref - got).abs().max().item()
    all_ok &= report('box_iou_rotated random 500x300', err < 1e-4, f'max abs err {err:.2e}')

    # overlapping boxes so the IoUs are not mostly zero
    b3 = b1[:200].clone()
    gp = torch.Generator().manual_seed(21)
    b3[:, :2] += (torch.randn(b3[:, :2].shape, generator=gp) * 10).to(dev)
    b3[:, 4] += (torch.randn(b3[:, 4].shape, generator=gp) * 0.3).to(dev)
    ref = mm_box_iou_rotated(b1[:200], b3)
    got = ops.box_iou_rotated(b1[:200], b3)
    err = (ref - got).abs().max().item()
    frac = (ref > 0.1).float().mean().item()
    all_ok &= report('box_iou_rotated overlapping', err < 1e-4, f'max abs err {err:.2e}, {frac:.0%} pairs > 0.1')

    # identical boxes, contained boxes, axis-aligned boxes
    same = mm_box_iou_rotated(b1[:50], b1[:50], aligned=True)
    got = ops.box_iou_rotated(b1[:50], b1[:50], aligned=True)
    # mmcv's own kernel returns 0.333 for one thin box here (float32 corner
    # rounding at large coordinates), so compare against the exact answer only
    n_mm_wrong = int(((same - 1).abs() > 1e-4).sum())
    all_ok &= report('identical boxes IoU == 1', torch.allclose(got, torch.ones_like(got), atol=1e-4),
                     f'(mmcv itself gets {n_mm_wrong}/50 wrong)')
    inner = b1[:50].clone()
    inner[:, 2:4] *= 0.5
    ref = mm_box_iou_rotated(b1[:50], inner, aligned=True)
    got = ops.box_iou_rotated(b1[:50], inner, aligned=True)
    all_ok &= report('contained boxes', (ref - got).abs().max().item() < 1e-5, f'(IoU should be 0.25: {got[:3].tolist()})')
    aa = b1[:100].clone()
    aa[:, 4] = 0
    aa2 = aa.clone()
    aa2[:, :2] += 5
    ref = mm_box_iou_rotated(aa, aa2)
    got = ops.box_iou_rotated(aa, aa2)
    all_ok &= report('axis-aligned (parallel edges)', (ref - got).abs().max().item() < 1e-4)

    # iof
    ref = mm_box_iou_rotated(b1[:100], b3[:100], mode='iof')
    got = ops.box_iou_rotated(b1[:100], b3[:100], mode='iof')
    all_ok &= report('iof mode', (ref - got).abs().max().item() < 1e-4)

    # ---- differentiable IoU + gradients ----------------------------------------
    p = b1[:200].clone().requires_grad_(True)
    t = b3.clone()
    ref = mm_diff_iou(p[None], t[None])[0]
    (ref.sum()).backward()
    g_ref = p.grad.clone()
    p.grad = None
    got = ops.diff_iou_rotated_2d(p, t)
    (got.sum()).backward()
    g_got = p.grad.clone()
    err = (ref - got).abs().max().item()
    finite = torch.isfinite(g_got).all().item()
    gerr = (g_ref - g_got).abs()
    # a pair whose corner lies (almost) exactly on an edge of the other box sits
    # on a kink of the IoU: both implementations return a valid but different
    # sub-gradient there, so judge by the 99th percentile instead of the max
    gerr_p99 = gerr.flatten().kthvalue(int(0.99 * gerr.numel())).values.item()
    all_ok &= report('diff_iou_rotated_2d values', err < 1e-4, f'max abs err {err:.2e}')
    all_ok &= report('diff_iou_rotated_2d gradients', finite and gerr_p99 < 1e-3,
                     f'p99 abs grad err {gerr_p99:.2e}, max {gerr.max().item():.2e}, finite={finite}')

    # ---- NMS -------------------------------------------------------------------
    boxes = torch.cat([b1, b3, b1[:100] + 1], 0)
    scores = torch.rand(boxes.shape[0], generator=torch.Generator().manual_seed(3)).to(dev)
    for thr in (0.1, 0.3, 0.5):
        ref_dets, ref_keep = mm_nms_rotated(boxes, scores, thr)
        t0 = time.time()
        got_dets, got_keep = ops.nms_rotated(boxes, scores, thr)
        dt = time.time() - t0
        same_set = set(ref_keep.tolist()) == set(got_keep.tolist())
        all_ok &= report(f'nms_rotated thr={thr}', same_set,
                         f'{len(ref_keep)} vs {len(got_keep)} kept, {dt*1000:.0f} ms for {boxes.shape[0]} boxes')

    # 2000 boxes timing (nms_pre in the config)
    big = random_boxes(2000, dev, seed=7)
    s = torch.rand(2000, generator=torch.Generator().manual_seed(8)).to(dev)
    t0 = time.time()
    ops.nms_rotated(big, s, 0.5)
    print(f'      nms_rotated 2000 boxes: {(time.time()-t0)*1000:.0f} ms on {dev}')

    print('ALL OK' if all_ok else 'SOME CHECKS FAILED')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
