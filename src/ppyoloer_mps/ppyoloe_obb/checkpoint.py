"""
PaddleDetection (.pdparams) to PyTorch (.pt) weight conversion, plus loading helpers.

The mapping is 1:1: the modules here are named exactly as in Paddle, and the checkpoint holds
only 4D tensors (convolutions, same NCHW layout) and 1D ones (BatchNorm, biases). There is no
linear layer to transpose. Two suffixes change:

    bn._mean      -> bn.running_mean
    bn._variance  -> bn.running_var

Paddle has no num_batches_tracked, so it is created as zero. That does not affect inference
because BatchNorm uses the running statistics.
"""
from __future__ import annotations

import os
from typing import Any

import torch

__all__ = ["convert_paddle_state_dict", "load_checkpoint", "save_checkpoint"]

_SUFFIX_MAP = {"._mean": ".running_mean", "._variance": ".running_var"}


def _rename(key: str) -> str:
    for old, new in _SUFFIX_MAP.items():
        if key.endswith(old):
            return key[: -len(old)] + new
    return key


def convert_paddle_state_dict(paddle_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert a Paddle state_dict (numpy arrays or paddle.Tensor) into torch tensors."""
    out: dict[str, torch.Tensor] = {}
    for key, value in paddle_state.items():
        array = value.numpy() if hasattr(value, "numpy") else value
        out[_rename(key)] = torch.as_tensor(array).clone()
    return out


def save_checkpoint(path: str, model: torch.nn.Module, meta: dict | None = None) -> None:
    payload = {"model": model.state_dict(), "meta": meta or {}}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str, strict: bool = True) -> dict:
    """Load a .pt from this repo, or a bare state_dict. Returns the metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    # num_batches_tracked does not come from Paddle, and angle_proj_conv is a constant the
    # model rebuilds in _init_weights, so both are fine to be missing
    missing = [k for k in missing if not k.endswith(("num_batches_tracked", "angle_proj_conv.weight"))]
    if strict and (missing or unexpected):
        raise RuntimeError(f"pesos incompatibles.\n  faltan: {missing[:8]}\n  sobran: {unexpected[:8]}")
    return payload.get("meta", {}) if isinstance(payload, dict) else {}
