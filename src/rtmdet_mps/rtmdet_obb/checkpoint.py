"""Load mmrotate / mmdet checkpoints without mmengine installed."""
import pickle
import types
from typing import Dict, Tuple

import torch
import torch.nn as nn


class _Stub:
    """Placeholder for mmengine objects referenced in checkpoint metadata."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


class _Unpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module.startswith('mmengine') or module.startswith('mmcv') or module.startswith('mmdet'):
            return _Stub
        return super().find_class(module, name)


_pickle_module = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__='pickle')


def load_state_dict_file(path: str) -> Dict[str, torch.Tensor]:
    """Return the ``state_dict`` of a torch/mmengine checkpoint (EMA weights if that is what was saved)."""
    ckpt = torch.load(path, map_location='cpu', pickle_module=_pickle_module)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt


def load_pretrained(model: nn.Module, path: str, strict: bool = False, verbose: bool = True) -> Tuple[list, list]:
    """Load weights into ``model``, skipping tensors whose shape differs (e.g. the
    15-class ``rtm_cls`` layers when fine-tuning on 1 class).

    Returns ``(skipped_shape_mismatch, missing_keys)``.
    """
    state = load_state_dict_file(path)
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
