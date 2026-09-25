import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from yolox_obb import YOLOXOBB, load_pretrained  # noqa: E402
from yolox_obb.boxes import regularize_mintheta  # noqa: E402
from yolox_obb.ops import box_iou_rotated  # noqa: E402


def test_forward_has_three_finite_output_levels():
    model = YOLOXOBB(num_classes=1).eval()
    with torch.no_grad():
        cls, reg, obj = model(torch.zeros(1, 3, 64, 64))
    assert [x.shape[-2:] for x in cls] == [(8, 8), (4, 4), (2, 2)]
    assert [x.shape[1] for x in reg] == [5, 5, 5]
    assert [x.shape[1] for x in obj] == [1, 1, 1]
    assert all(torch.isfinite(x).all() for group in (cls, reg, obj) for x in group)


def test_state_dict_keys_follow_the_yolox_obb_yaml_layout():
    """Names and shapes of the checkpoint written by YOLOX_OBB for ``yoloxs_obb.yaml``."""
    state = YOLOXOBB(num_classes=15).state_dict()
    expected = {
        "model.0.conv.weight": (32, 3, 6, 6),
        "model.2.m.0.cv2.conv.weight": (32, 32, 3, 3),
        "model.4.m.2.cv1.conv.weight": (64, 64, 1, 1),
        "model.8.cv2.conv.weight": (512, 1024, 1, 1),
        "model.13.cv1.conv.weight": (128, 512, 1, 1),
        "model.23.cv3.conv.weight": (512, 512, 1, 1),
        "model.26.conv.weight": (128, 512, 1, 1),
        "model.32.bn.running_var": (128,),
        "model.33.cls_preds.0.weight": (15, 128, 1, 1),
        "model.33.reg_preds.2.weight": (5, 128, 1, 1),
        "model.33.obj_preds.1.bias": (1,),
    }
    for key, shape in expected.items():
        assert tuple(state[key].shape) == shape, key
    assert not any(k.startswith(("model.11.", "model.12.", "model.33.cls_preds.3")) for k in state)


def test_loss_is_finite_and_backpropagates():
    torch.manual_seed(0)
    model = YOLOXOBB(num_classes=2).train()
    images = torch.rand(2, 3, 128, 128) * 255
    boxes = [torch.tensor([[40., 50., 60., 30., 0.3], [90., 90., 20., 40., -1.0]]), torch.zeros((0, 5))]
    labels = [torch.tensor([0, 1]), torch.zeros((0,), dtype=torch.long)]
    losses = model.loss(images, boxes, labels)
    assert set(losses) == {"loss_obj", "loss_cls", "loss_reg"}
    model.use_l1 = True
    losses = model.loss(images, boxes, labels)
    assert "loss_l1" in losses
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert model.model[0].conv.weight.grad is not None
    assert sum(conv.weight.grad.abs().sum() for conv in model.model[33].reg_preds) > 0


def test_predict_returns_boxes_in_input_pixels():
    model = YOLOXOBB(num_classes=1).eval()
    head = model.model[33]
    for conv in list(head.cls_preds) + list(head.obj_preds):
        torch.nn.init.constant_(conv.bias, 5.0)  # everything above threshold
    res = model.predict(torch.zeros(1, 3, 64, 64), score_thr=0.5, nms_iou=0.1, max_per_img=10)[0]
    assert 0 < len(res["boxes"]) <= 10
    assert res["boxes"].shape[1] == 5
    assert torch.isfinite(res["boxes"]).all()


def test_mintheta_keeps_the_same_rotated_box():
    boxes = torch.tensor([[10., 20., 30., 10., 1.2], [0., 0., 8., 16., -0.2], [5., 5., 4., 9., 3.0]])
    reg = regularize_mintheta(boxes)
    assert (reg[:, 4].abs() <= math.pi / 4 + 1e-6).all()
    ious = box_iou_rotated(boxes, reg, aligned=True)
    assert torch.allclose(ious, torch.ones(3), atol=1e-4)


def test_local_dota_checkpoint_loads_when_present():
    checkpoint = Path(__file__).parents[1] / "models/yolox_obb/checkpoints/yolox_s_dota1_0.pth"
    if not checkpoint.exists():
        pytest.skip("local YOLOX_OBB DOTA checkpoint is not available")
    model = YOLOXOBB(num_classes=1)
    mismatched, missing = load_pretrained(model, str(checkpoint), verbose=False)
    # The DOTA checkpoint has 15 classes: only cls_preds weight and bias differ at each level.
    assert len(mismatched) == 6
    assert all(k.startswith("model.33.cls_preds") for k in missing)


@pytest.mark.parametrize("size", ["nano", "tiny", "s"])
def test_official_yolox_forward_and_loss(size):
    from yolox_obb import YOLOXOfficialOBB
    model = YOLOXOfficialOBB(num_classes=1, size=size)
    assert model.arch == f"yolox_{size}"
    model.eval()
    with torch.no_grad():
        cls, reg, obj = model(torch.zeros(1, 3, 64, 64))
    assert [x.shape[1] for x in reg] == [5, 5, 5]
    model.train()
    losses = model.loss(torch.rand(1, 3, 128, 128) * 255, [torch.tensor([[60., 60., 50., 26., 0.5]])],
                        [torch.tensor([0])])
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()


def test_official_yolox_keeps_coco_xywh_regression():
    from yolox_obb import YOLOXOfficialOBB
    model = YOLOXOfficialOBB(num_classes=1, size="nano")
    coco = {k: torch.randn(4, *v.shape[1:]) if ".reg_preds." in k else v.clone()
            for k, v in model.state_dict().items()}
    adapted = model.adapt_state_dict(coco)
    w = adapted["head.reg_preds.0.weight"]
    assert w.shape[0] == 5
    assert torch.equal(w[:4], coco["head.reg_preds.0.weight"])
    assert (w[4] == 0).all()


def test_build_model_and_checkpoint_arch(tmp_path):
    from yolox_obb import MODELS, build_model, checkpoint_arch
    for name in MODELS:
        assert build_model(name, 1).arch == name
    path = tmp_path / "ckpt.pth"
    torch.save(dict(state_dict={}, meta=dict(arch="yolox_nano")), path)
    assert checkpoint_arch(str(path)) == "yolox_nano"
    torch.save(dict(state_dict={}, meta={}), path)
    assert checkpoint_arch(str(path)) == "ddgrcf_s"
