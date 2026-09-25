"""Strip a training checkpoint down to the inference weights (EMA ``state_dict`` + metadata).

    python export_checkpoint.py ../../models/rtmdet/experiments/run/epoch_100.pth ../../models/rtmdet/checkpoints/rtmdet_r_tiny_banknotes.pth
"""
import argparse
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('src')
    p.add_argument('dst')
    p.add_argument('--note', default='')
    args = p.parse_args()
    ckpt = torch.load(args.src, map_location='cpu')
    out = dict(state_dict=ckpt['state_dict'], meta=dict(ckpt.get('meta', {}), epoch=ckpt.get('epoch'), note=args.note))
    torch.save(out, args.dst)
    n = sum(v.numel() for v in out['state_dict'].values())
    print(f'{args.dst}: {len(out["state_dict"])} tensors, {n / 1e6:.2f} M values')


if __name__ == '__main__':
    main()
