from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from rtmdet_obb import RTMDetR  # noqa: E402
from export_onnx import OUTPUT_NAMES, RawOutputs  # noqa: E402


def test_tiny_export_has_expected_outputs(tmp_path):
    model = RTMDetR(num_classes=1, size="tiny").eval()
    output = tmp_path / "tiny.onnx"
    torch.onnx.export(
        RawOutputs(model), torch.zeros(1, 3, 64, 64), str(output),
        input_names=["images_bgr_0_255"], output_names=list(OUTPUT_NAMES),
        opset_version=18,
    )
    assert output.exists() and output.stat().st_size > 0
