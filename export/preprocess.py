"""Image → ONNX input tensor (training-style resize + RGB mean/std).

The ONNX graph expects ``[1, 3, H, W]`` float32 RGB, already normalized:
``(x / 255 - mean) / std``. Resize is bilinear (PIL), matching oriented-det
``resize_mode=fixed``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
from PIL import Image

_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)

try:
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - Pillow < 9.1
    _RESAMPLE = Image.BILINEAR


def _as_float3(values: Sequence[float] | None, default: Sequence[float]) -> np.ndarray:
    src = default if values is None else values
    arr = np.asarray(list(src), dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"expected 3-channel mean/std, got {arr.size}")
    return arr


def canvas_size_from_meta(meta: Mapping[str, Any]) -> tuple[int, int]:
    """Return ``(height, width)`` from export meta."""
    prep = meta.get("preprocess") or {}
    ts = prep.get("target_size")
    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
        return int(ts[0]), int(ts[1])
    shape = (meta.get("input") or {}).get("shape") or []
    if len(shape) >= 4:
        return int(shape[2]), int(shape[3])
    size = int(meta.get("image_size") or 1024)
    return size, size


def mean_std_from_meta(meta: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    prep = meta.get("preprocess") or {}
    mean = prep.get("normalize_mean") or meta.get("normalize_mean") or _DEFAULT_MEAN
    std = prep.get("normalize_std") or meta.get("normalize_std") or _DEFAULT_STD
    return _as_float3(mean, _DEFAULT_MEAN), _as_float3(std, _DEFAULT_STD)


def to_rgb_uint8(image: Union[Image.Image, np.ndarray], *, bgr: bool = False) -> np.ndarray:
    """Return HWC uint8 RGB."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected HWC image with 3 or 4 channels, got {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if bgr:
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def preprocess_rgb(
    image: Union[Image.Image, np.ndarray],
    height: int,
    width: int,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    *,
    bgr: bool = False,
) -> tuple[np.ndarray, int, int]:
    """RGB (or BGR) image → ``[1, 3, H, W]`` float32 plus original ``(h, w)``."""
    rgb = to_rgb_uint8(image, bgr=bgr)
    orig_h, orig_w = int(rgb.shape[0]), int(rgb.shape[1])
    pil = Image.fromarray(rgb, mode="RGB")
    if pil.size != (int(width), int(height)):
        pil = pil.resize((int(width), int(height)), _RESAMPLE)
    chw = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean_a = _as_float3(mean, _DEFAULT_MEAN).reshape(3, 1, 1)
    std_a = _as_float3(std, _DEFAULT_STD).reshape(3, 1, 1)
    out = (chw - mean_a) / std_a
    return out[np.newaxis, ...], orig_h, orig_w


def preprocess_from_meta(
    image: Union[Image.Image, np.ndarray],
    meta: Mapping[str, Any],
    *,
    bgr: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Preprocess using ``normalize_mean`` / ``target_size`` from export meta."""
    height, width = canvas_size_from_meta(meta)
    mean, std = mean_std_from_meta(meta)
    return preprocess_rgb(image, height, width, mean, std, bgr=bgr)


def preprocess_path(
    path: Union[str, Path],
    meta: Mapping[str, Any],
) -> tuple[np.ndarray, int, int]:
    """Load a file as RGB and preprocess."""
    img = Image.open(path).convert("RGB")
    return preprocess_from_meta(img, meta)


def scale_obb_to_original(
    boxes: np.ndarray,
    orig_w: int,
    orig_h: int,
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Scale ``(cx, cy, w, h, angle)`` from canvas pixels to original image pixels."""
    if boxes.size == 0:
        return boxes.reshape(0, 5)
    out = np.array(boxes, dtype=np.float32, copy=True)
    out[:, 0] *= float(orig_w) / float(canvas_w)
    out[:, 1] *= float(orig_h) / float(canvas_h)
    out[:, 2] *= float(orig_w) / float(canvas_w)
    out[:, 3] *= float(orig_h) / float(canvas_h)
    return out
