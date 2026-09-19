"""Tests for ORT device / provider resolution (no GPU required)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")

from export.ort_runtime import (
    clear_ort_session_cache,
    configure_ort_device,
    get_ort_device,
    ort_providers_for_device,
    set_ort_device,
)

_DEMO = Path(__file__).resolve().parents[1] / "demo" / "planes_pleiades_neo.jpg"
_ONNX = Path(__file__).resolve().parents[2] / "onnx_export" / "model.onnx"


def test_cpu_providers():
    set_ort_device("cpu")
    assert ort_providers_for_device() == ["CPUExecutionProvider"]


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="Unknown ORT device"):
        ort_providers_for_device("tpu")


def test_configure_returns_cpu_by_default():
    clear_ort_session_cache()
    set_ort_device(None)
    providers = configure_ort_device("cpu")
    assert providers == ["CPUExecutionProvider"]
    assert get_ort_device() == "cpu"


def test_cuda_without_gpu_raises():
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("CUDA EP available; skip missing-CUDA test")
    set_ort_device("cuda")
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        ort_providers_for_device()


def test_ort_cuda_smoke_real_tile_1024():
    """One real 1024 tile must run on ORT CUDA without ROI Transpose OOM."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider not available")

    import numpy as np
    from PIL import Image

    if not _ONNX.is_file():
        pytest.skip("onnx_export/model.onnx not present")
    if not _DEMO.is_file():
        pytest.skip("bundled demo image missing")

    from export.ort_runtime import get_ort_session

    clear_ort_session_cache()
    providers = configure_ort_device("cuda")
    assert "CUDAExecutionProvider" in providers

    sess = get_ort_session(str(_ONNX), device="cuda")
    img = Image.open(_DEMO).convert("RGB").resize((1024, 1024))
    x = (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    outs = sess.run(None, {"images": x})
    assert len(outs) >= 4
    assert int(np.asarray(outs[3]).reshape(-1)[0]) >= 0
