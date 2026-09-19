"""ONNX Runtime device selection and session cache."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

_ORT_DEVICE_OVERRIDE: Optional[str] = None
_SESSION_CACHE: Dict[Tuple[str, Tuple[str, ...]], object] = {}


def set_ort_device(device: Optional[str]) -> None:
    """Set runtime ORT device for this process (``cpu``, ``cuda``, ``auto``)."""
    global _ORT_DEVICE_OVERRIDE
    _ORT_DEVICE_OVERRIDE = device.lower().strip() if device else None


def get_ort_device() -> str:
    """Resolved ORT device string (override, env, or default ``cpu``)."""
    if _ORT_DEVICE_OVERRIDE:
        return _ORT_DEVICE_OVERRIDE
    return (os.environ.get("ORIENTED_DET_ORT_DEVICE") or "cpu").lower().strip()


def ort_providers_for_device(device: Optional[str] = None) -> List[str]:
    """Map device string to ONNX Runtime ``providers`` list."""
    import onnxruntime as ort

    d = (device if device is not None else get_ort_device()).lower().strip()
    if d in ("cuda", "gpu"):
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "ORT device=cuda requested but CUDAExecutionProvider is not available. "
                "Install onnxruntime-gpu matching your CUDA driver "
                "(e.g. pip install onnxruntime-gpu[cuda,cudnn]; do not also install onnxruntime)."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if d == "auto":
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if d != "cpu":
        raise ValueError(f"Unknown ORT device {device!r}; use cpu, cuda, or auto.")
    return ["CPUExecutionProvider"]


def _cuda_provider_options() -> Dict[str, Any]:
    """ORT CUDA EP options (device id, arena, optional mem cap)."""
    opts: Dict[str, Any] = {
        "device_id": int(os.environ.get("ORIENTED_DET_ORT_CUDA_DEVICE_ID", "0")),
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": os.environ.get(
            "ORIENTED_DET_ORT_CUDNN_CONV_ALGO", "HEURISTIC"
        ),
    }
    mem = os.environ.get("ORIENTED_DET_ORT_GPU_MEM_LIMIT")
    if mem:
        opts["gpu_mem_limit"] = int(mem)
    return opts


def _session_providers(
    providers: List[str],
) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
    """Expand provider names into ORT session provider list (with CUDA options)."""
    out: List[Union[str, Tuple[str, Dict[str, Any]]]] = []
    for p in providers:
        if p == "CUDAExecutionProvider":
            out.append((p, _cuda_provider_options()))
        else:
            out.append(p)
    return out


def configure_ort_device(device: Optional[str] = None) -> List[str]:
    """Apply device override and return provider names."""
    if device is not None:
        set_ort_device(device)
    return ort_providers_for_device()


def clear_ort_session_cache() -> None:
    """Drop cached ORT sessions (tests)."""
    _SESSION_CACHE.clear()


def get_ort_session(onnx_path: str, device: Optional[str] = None):
    """Return a cached ``onnxruntime.InferenceSession`` for ``onnx_path``."""
    import onnxruntime as ort

    providers = ort_providers_for_device(device)
    session_providers = _session_providers(providers)
    # Cache key uses provider *names* only (options are env-stable per process).
    key = (str(onnx_path), tuple(providers))
    if key not in _SESSION_CACHE:
        so = ort.SessionOptions()
        so.enable_mem_pattern = True
        _SESSION_CACHE[key] = ort.InferenceSession(
            str(onnx_path),
            sess_options=so,
            providers=session_providers,
        )
    return _SESSION_CACHE[key]
