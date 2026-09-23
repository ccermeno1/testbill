"""Rotated RTMDet: CSPNeXt/PAFPN shapes, MMDet key layout, assigner, losses, train/eval, wiring."""

import math
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from oriented_det.models import RotatedRTMDet
from oriented_det.models.backbones.cspnext import (
    build_cspnext_pafpn,
    load_mmdet_weights,
)
from oriented_det.models.bbox_coder import DistanceAnglePointCoder
from oriented_det.models.rotated_rtmdet import (
    decode_rtmdet_boxes,
    dynamic_soft_label_assign,
    generate_rtmdet_priors,
    pairwise_rbox_iou_exact,
    points_in_rboxes,
    quality_focal_loss,
    rotated_rtmdet_kwargs_from_config,
)


def _tiny(num_classes: int = 1, **kw) -> RotatedRTMDet:
    kw.setdefault("pretrained_backbone", False)
    return RotatedRTMDet(num_classes=num_classes, variant="tiny", **kw)


def _target(boxes, labels):
    return {
        "rboxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def test_cspnext_tiny_channels_and_strides():
    backbone, neck, out_c = build_cspnext_pafpn("tiny")
    assert backbone.out_channels == [96, 192, 384]
    assert out_c == 96
    x = torch.randn(1, 3, 128, 160)
    feats = neck(backbone(x))
    assert [tuple(f.shape) for f in feats] == [
        (1, 96, 16, 20),
        (1, 96, 8, 10),
        (1, 96, 4, 5),
    ]


def test_cspnext_handles_non_divisible_input():
    backbone, neck, _ = build_cspnext_pafpn("tiny")
    feats = neck(backbone(torch.randn(1, 3, 100, 70)))
    assert len(feats) == 3


def test_state_dict_keys_follow_mmdet_layout():
    keys = set(_tiny(num_classes=15).state_dict())
    for k in (
        "backbone.stem.0.conv.weight",
        "backbone.stem.0.bn.running_mean",
        "backbone.stage1.1.attention.fc.weight",
        "backbone.stage1.1.blocks.0.conv2.depthwise_conv.conv.weight",
        "backbone.stage4.1.conv1.conv.weight",
        "neck.reduce_layers.0.conv.weight",
        "neck.top_down_blocks.0.main_conv.conv.weight",
        "neck.out_convs.2.bn.weight",
        "head.cls_convs.1.0.conv.weight",
        "head.reg_convs.2.1.bn.running_var",
        "head.rtm_cls.0.weight",
        "head.rtm_reg.1.weight",
        "head.rtm_ang.2.weight",
    ):
        assert k in keys, k


def test_tiny_parameter_count_matches_mmrotate():
    # MMRotate model zoo: rotated_rtmdet_tiny (DOTA, 15 classes) = 4.88M params.
    model = _tiny(num_classes=15)
    n = sum(p.numel() for p in model.parameters())
    assert 4.7e6 < n < 5.0e6, n


def test_share_conv_ties_level_weights():
    head = _tiny().head
    assert head.cls_convs[0][0].conv.weight is head.cls_convs[2][0].conv.weight
    assert head.cls_convs[0][0].bn is not head.cls_convs[2][0].bn


def test_priors_offset_zero():
    pts = generate_rtmdet_priors([(2, 3)], [8], torch.float32, torch.device("cpu"))[0]
    assert pts[:4].tolist() == [[0.0, 0.0], [8.0, 0.0], [16.0, 0.0], [0.0, 8.0]]


def test_decode_matches_distance_angle_coder_for_positive_distances():
    torch.manual_seed(0)
    pts = torch.rand(20, 2) * 100
    ltrb = torch.rand(20, 4) * 30 + 1
    ang = (torch.rand(20, 1) - 0.5) * 1.2  # inside le90, no w/h swap
    ours = decode_rtmdet_boxes(pts, ltrb, ang)
    ref = DistanceAnglePointCoder().decode(pts, torch.cat([ltrb, ang], dim=-1))
    iou = pairwise_rbox_iou_exact(ours, ref).diagonal()
    assert torch.all(iou > 0.999)


def test_decode_negative_extent_is_absolute():
    box = decode_rtmdet_boxes(
        torch.zeros(1, 2), torch.tensor([[-3.0, 1.0, 1.0, 1.0]]), torch.zeros(1, 1)
    )
    assert box[0, 2].item() == pytest.approx(2.0)


def test_points_in_rboxes_rotated():
    box = torch.tensor([[50.0, 50.0, 40.0, 4.0, math.pi / 4]])
    pts = torch.tensor([[60.0, 60.0], [60.0, 40.0], [50.0, 50.0]])
    assert points_in_rboxes(pts, box)[:, 0].tolist() == [True, False, True]


def test_pairwise_iou_exact_known_values():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0], [100.0, 100.0, 4.0, 4.0, 0.3]])
    b = torch.tensor([[5.0, 0.0, 10.0, 10.0, 0.0], [0.0, 0.0, 10.0, 10.0, math.pi / 2]])
    iou = pairwise_rbox_iou_exact(a, b)
    assert iou[0, 0].item() == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert iou[0, 1].item() == pytest.approx(1.0, abs=1e-4)
    assert iou[1].abs().sum().item() == 0.0


def test_quality_focal_loss_zero_at_perfect_prediction():
    logits = torch.tensor([[20.0, -20.0], [-20.0, -20.0]])
    labels = torch.tensor([0, 2])
    scores = torch.tensor([1.0, 0.0])
    assert quality_focal_loss(logits, labels, scores).sum().item() < 1e-6


def test_assigner_picks_priors_inside_gt_with_good_boxes():
    pts = generate_rtmdet_priors([(8, 8)], [8], torch.float32, torch.device("cpu"))[0]
    strides = torch.full((pts.size(0),), 8.0)
    gt = torch.tensor([[32.0, 32.0, 30.0, 14.0, 0.4]])
    decoded = gt.expand(pts.size(0), 5).clone()
    decoded[:, :2] = pts  # boxes centered on each prior
    logits = torch.zeros(pts.size(0), 1)
    assigned, ious = dynamic_soft_label_assign(pts, strides, decoded, logits, gt, torch.tensor([0]))
    fg = assigned >= 0
    assert fg.any()
    assert torch.all(points_in_rboxes(pts[fg], gt)[:, 0])
    assert torch.all(ious[fg] > 0) and torch.all(ious[~fg] == 0)


def test_assigner_empty_gt():
    pts = torch.rand(10, 2)
    assigned, ious = dynamic_soft_label_assign(
        pts, torch.ones(10), torch.rand(10, 5), torch.zeros(10, 1), torch.zeros(0, 5), torch.zeros(0)
    )
    assert torch.all(assigned == -1) and torch.all(ious == 0)


def test_forward_train_losses_finite_and_backward():
    torch.manual_seed(0)
    model = _tiny(num_classes=2)
    model.train()
    images = [torch.randn(3, 128, 128), torch.randn(3, 128, 128)]
    targets = [
        _target([[64.0, 64.0, 60.0, 30.0, 0.5]], [1]),
        {"rboxes": torch.zeros((0, 5)), "labels": torch.zeros((0,), dtype=torch.int64)},
    ]
    losses = model(images, targets)
    assert set(losses) == {"loss_classifier", "loss_box_reg"}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert model.head.rtm_reg[0].weight.grad is not None


def test_forward_eval_output_format():
    model = _tiny(num_classes=3, score_threshold=0.0, nms_pre=50, max_detections_per_image=7)
    model.eval()
    with torch.no_grad():
        out = model([torch.randn(3, 96, 96)])
    assert len(out) == 1
    assert len(out[0]["rboxes"]) == out[0]["scores"].numel() == out[0]["labels"].numel()
    assert out[0]["scores"].numel() <= 7
    if out[0]["labels"].numel():
        assert out[0]["labels"].min() >= 1 and out[0]["labels"].max() <= 3


def test_overfit_single_rotated_box():
    """Sanity: the tiny model can fit one rotated rectangle on a synthetic image."""
    torch.manual_seed(0)
    img = torch.zeros(3, 128, 128)
    gt = [64.0, 64.0, 70.0, 30.0, 0.6]
    # Paint the GT box so the task is learnable.
    ys, xs = torch.meshgrid(torch.arange(128.0), torch.arange(128.0), indexing="ij")
    inside = points_in_rboxes(torch.stack([xs.flatten(), ys.flatten()], -1), torch.tensor([gt]))
    img[:, inside[:, 0].view(128, 128)] = 2.0
    model = _tiny(num_classes=1, score_threshold=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    model.train()
    # Losses are weighted by the assigned IoU, so they start near 0 (zero-size boxes)
    # and are not monotonic; AdamW is scale-invariant and still trains.
    for _ in range(120):
        losses = model([img], [_target([gt], [1])])
        loss = sum(losses.values())
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Single-image toy run: BN running stats (momentum 0.03) lag the weights, so
    # recalibrate them on the image (precise BN) before eval.
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average
    with torch.no_grad():
        model.head(model.extract_features(img[None]))
    model.eval()
    with torch.no_grad():
        out = model([img])[0]
    assert out["scores"].numel() > 0
    r = out["rboxes"][0]  # highest score first
    best = torch.tensor([[r.cx, r.cy, r.width, r.height, r.angle]])
    assert pairwise_rbox_iou_exact(best, torch.tensor([gt]))[0, 0] > 0.5


def test_input_bgr_resolution():
    assert _tiny(pretrained_weights="coco").input_bgr is True
    assert _tiny(pretrained_weights="dota").input_bgr is True
    assert _tiny(pretrained_weights="imagenet").input_bgr is False
    assert _tiny(pretrained_weights="coco", input_bgr=False).input_bgr is False


def test_load_openmmlab_style_checkpoint_with_missing_classes(tmp_path):
    """Checkpoints pickle mmengine objects; loading must not need those packages."""
    src = _tiny(num_classes=80)
    fake_mod = types.ModuleType("mmengine_fake_for_test")

    class HistoryBuffer:  # stands in for mmengine.logging.HistoryBuffer
        def __init__(self):
            self.data = [1, 2, 3]

    HistoryBuffer.__module__ = fake_mod.__name__
    HistoryBuffer.__qualname__ = "HistoryBuffer"
    fake_mod.HistoryBuffer = HistoryBuffer
    sys.modules[fake_mod.__name__] = fake_mod
    try:
        # MMDet naming: our ``head.*`` is ``bbox_head.*`` in OpenMMLab checkpoints.
        state = {
            ("bbox_head." + k[len("head."):]) if k.startswith("head.") else k: v
            for k, v in src.state_dict().items()
        }
        ckpt = {"state_dict": state, "message_hub": {"buf": HistoryBuffer()}}
        path = tmp_path / "fake_mmdet.pth"
        torch.save(ckpt, path)
    finally:
        del sys.modules[fake_mod.__name__]

    dst = _tiny(num_classes=1)
    loaded, skipped = load_mmdet_weights(
        dst, path, {"backbone.": "backbone.", "neck.": "neck.", "bbox_head.": "head."}
    )
    assert "backbone.stem.0.conv.weight" in loaded
    assert "head.rtm_reg.0.weight" in loaded
    assert "head.rtm_cls.0.weight" in skipped  # 80 vs 1 classes
    assert torch.equal(dst.backbone.stem[0].conv.weight, src.backbone.stem[0].conv.weight)
    assert torch.equal(dst.head.rtm_ang[1].weight, src.head.rtm_ang[1].weight)


def test_kwargs_from_config_and_train_wiring():
    from unittest.mock import MagicMock, patch
    import importlib

    from oriented_det.train.config import TrainingExperimentConfig

    root = Path(__file__).resolve().parents[1]
    cfg = TrainingExperimentConfig.load(root / "configs" / "rotated_rtmdet" / "billetes_tiny_le90.json")
    assert cfg.model_type == "rotated_rtmdet"
    assert cfg.training.optimizer == "adamw"
    kw = rotated_rtmdet_kwargs_from_config(cfg.model)
    assert kw["variant"] == "tiny"
    assert kw["pretrained_weights"] == "coco"
    assert kw["fpn_strides"] == [8, 16, 32]
    assert kw["box_reg_weight"] == pytest.approx(2.0)
    assert kw["final_nms_iou_threshold"] == pytest.approx(0.3)
    assert kw["frozen_stages"] == -1

    tools_dir = str(root / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    train = importlib.import_module("train")
    inst = MagicMock()
    inst.to = MagicMock(return_value=inst)
    mock_cls = MagicMock(return_value=inst)
    with patch.object(train, "RotatedRTMDet", mock_cls):
        model, _ = train.create_model_from_config(
            cfg, num_classes=1, device=torch.device("cpu"), roi_class_weights=None
        )
    assert model is inst
    call_kw = mock_cls.call_args.kwargs
    assert call_kw["num_classes"] == 1
    assert call_kw["variant"] == "tiny"
    assert call_kw["nms_pre"] == 2000


def test_adamw_optimizer_skips_decay_on_norm_and_bias():
    import importlib

    from oriented_det.train.config import TrainingConfig, TrainingExperimentConfig

    root = Path(__file__).resolve().parents[1]
    tools_dir = str(root / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    train = importlib.import_module("train")
    model = _tiny()
    cfg = TrainingExperimentConfig(
        training=TrainingConfig(optimizer="adamw", learning_rate=1e-3, weight_decay=0.05)
    )
    opt = train.create_optimizer(model, [{"params": list(model.parameters())}], cfg)
    assert isinstance(opt, torch.optim.AdamW)
    by_id = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    assert by_id[id(model.backbone.stem[0].bn.weight)] == 0.0
    assert by_id[id(model.head.rtm_cls[0].bias)] == 0.0
    assert by_id[id(model.backbone.stem[0].conv.weight)] == pytest.approx(0.05)
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))
