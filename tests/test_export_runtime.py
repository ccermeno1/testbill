import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from yolox_obb import build_model  # noqa: E402
from export_onnx import DeployModel, OUTPUT_NAMES, export  # noqa: E402
from onnx_example import corners, letterbox, postprocess  # noqa: E402


@pytest.mark.parametrize("arch", ["ddgrcf_s", "yolox_nano"])
def test_onnx_matches_pytorch_decoded_outputs(tmp_path, arch):
    torch.manual_seed(0)
    model = build_model(arch, 1).eval()
    path = tmp_path / "model.onnx"
    export(model, str(path), 64)
    x = (np.random.rand(1, 3, 64, 64) * 255).astype(np.float32)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()] == list(OUTPUT_NAMES)
    boxes, scores = session.run(None, {"images": x})
    with torch.no_grad():
        ref_boxes, ref_scores = DeployModel(model)(torch.from_numpy(x))
    assert boxes.shape == (1, 8 * 8 + 4 * 4 + 2 * 2, 5) and scores.shape == (1, 84, 1)
    np.testing.assert_allclose(scores, ref_scores.numpy(), atol=1e-5)
    np.testing.assert_allclose(boxes, ref_boxes.numpy(), rtol=1e-4, atol=1e-3)


def test_numpy_postprocess_suppresses_overlaps_and_rescales():
    boxes = np.array([[100, 100, 80, 40, 0.3], [102, 101, 80, 40, 0.3], [300, 300, 50, 20, -0.5]], np.float32)
    scores = np.array([[0.9], [0.8], [0.7]], np.float32)
    out_boxes, out_scores, labels = postprocess(boxes, scores, scale=0.5, score_thr=0.5, nms_iou=0.3)
    assert len(out_boxes) == 2 and out_scores[0] == pytest.approx(0.9)
    np.testing.assert_allclose(out_boxes[0, :4], [200, 200, 160, 80])
    assert corners(out_boxes).shape == (2, 4, 2)


def test_letterbox_matches_training_preprocessing():
    from yolox_obb.data import letterbox_image
    img = (np.random.rand(30, 50, 3) * 255).astype(np.uint8)
    x, scale = letterbox(img, 40)
    ref, ref_scale = letterbox_image(img, 40)
    assert scale == pytest.approx(ref_scale)
    np.testing.assert_array_equal(x[0], ref.numpy().astype(np.float32))
