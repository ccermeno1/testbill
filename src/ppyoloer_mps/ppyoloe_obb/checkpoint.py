"""
Conversion de pesos PaddleDetection (.pdparams) -> PyTorch (.pt) y utilidades de carga.

El mapeo es 1:1 porque los modulos del port se llaman igual que en Paddle y el checkpoint
solo contiene tensores 4D (convoluciones, mismo layout NCHW) y 1D (BatchNorm, sesgos):
no hay ninguna capa lineal que hubiera que transponer. Solo cambian dos sufijos:

    bn._mean      -> bn.running_mean
    bn._variance  -> bn.running_var

`num_batches_tracked` no existe en Paddle; se crea a cero (no afecta a la inferencia
porque BatchNorm usa las estadisticas acumuladas).
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
    """Convierte un state_dict de Paddle (numpy o paddle.Tensor) a tensores de PyTorch."""
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
    """Carga un .pt de este repo o un state_dict suelto. Devuelve los metadatos."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    # num_batches_tracked no viene de Paddle y angle_proj_conv es una constante que el
    # modelo reconstruye en _init_weights: es aceptable que falten
    missing = [k for k in missing if not k.endswith(("num_batches_tracked", "angle_proj_conv.weight"))]
    if strict and (missing or unexpected):
        raise RuntimeError(f"pesos incompatibles.\n  faltan: {missing[:8]}\n  sobran: {unexpected[:8]}")
    return payload.get("meta", {}) if isinstance(payload, dict) else {}
