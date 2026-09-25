"""Load YOLOX_OBB checkpoints and the checkpoints written by this repo."""
from typing import Dict, Tuple

import torch
import torch.nn as nn


def load_state_dict_file(path: str) -> Dict[str, torch.Tensor]:
    """Return the weights of a checkpoint.

    Accepts the files of this repo (``state_dict``, EMA weights), the YOLOX_OBB
    trainer checkpoints (``model``, also EMA weights when ``exp.ema`` was on) and
    a bare state dict.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'model'):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def load_pretrained(model: nn.Module, path: str, strict: bool = False, verbose: bool = True) -> Tuple[list, list]:
    """Load weights into ``model``, skipping tensors whose shape differs (e.g. the
    15-class ``cls_preds`` layers of the DOTA checkpoint when fine-tuning on 1 class).
    ``model.adapt_state_dict`` (if any) converts the checkpoint first.

    Returns ``(skipped_shape_mismatch, missing_keys)``.
    """
    state = load_state_dict_file(path)
    if hasattr(model, 'adapt_state_dict'):
        state = model.adapt_state_dict(state)
    own = model.state_dict()
    filtered, mismatched = {}, []
    for k, v in state.items():
        if k in own:
            if own[k].shape == v.shape:
                filtered[k] = v
            else:
                mismatched.append((k, tuple(v.shape), tuple(own[k].shape)))
    result = model.load_state_dict(filtered, strict=False)
    unexpected = [k for k in state if k not in own]
    if strict and (mismatched or result.missing_keys or unexpected):
        raise RuntimeError(f'strict load failed: mismatched={mismatched}, missing={result.missing_keys}, '
                           f'unexpected={unexpected}')
    if verbose:
        print(f'loaded {len(filtered)}/{len(own)} tensors from {path}')
        for k, s_src, s_own in mismatched:
            print(f'  skipped {k}: checkpoint {s_src} vs model {s_own}')
        if result.missing_keys:
            print(f'  {len(result.missing_keys)} tensors kept at random init: {result.missing_keys[:6]}'
                  + (' ...' if len(result.missing_keys) > 6 else ''))
        if unexpected:
            print(f'  {len(unexpected)} unused checkpoint tensors (e.g. {unexpected[0]})')
    return mismatched, result.missing_keys
