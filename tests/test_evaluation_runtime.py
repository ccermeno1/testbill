"""Loss, training loop, EMA and metrics."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ppyoloer_mps.ppyoloe_obb import build_ppyoloe_r
from ppyoloer_mps.ppyoloe_obb.data import ObbDataset, collate
from ppyoloer_mps.ppyoloe_obb.engine import ModelEMA, evaluate_model, train_one_epoch
from ppyoloer_mps.ppyoloe_obb.evaluation import evaluate_detections
from ppyoloer_mps.ppyoloe_obb.losses import PPYOLOERLoss


def test_training_can_overfit(make_learnable_dataset):
    """The whole loop (assigner, losses, optimiser) has to learn a trivial case.

    The loss climbs for the first epochs: with task-aligned assignment from random weights
    there are barely any positives at the start. What this checks is that it converges.
    """
    torch.manual_seed(0)
    ann, img_dir = make_learnable_dataset()
    ds = ObbDataset(ann, img_dir, img_size=128, train=False)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate)
    model = build_ppyoloe_r(num_classes=1, size="s")
    loss_fn = PPYOLOERLoss(num_classes=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    device = torch.device("cpu")
    losses = [train_one_epoch(model, loss_fn, loader, opt, device, e, log_interval=0)["loss"] for e in range(90)]
    assert all(math.isfinite(v) for v in losses), "loss diverged"
    assert losses[-1] < max(losses) / 2, f"did not converge: {losses[0]:.3f} -> {losses[-1]:.3f}"
    res = evaluate_model(model, loader, device, name="overfit", logger=lambda *_: None)
    assert res["mAP50"] > 0.5, f"failed to learn the trivial case: mAP50={res['mAP50']:.3f}"


def test_evaluate_model_runs(make_dataset):
    ann, img_dir = make_dataset()
    ds = ObbDataset(ann, img_dir, img_size=320, train=False)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    model = build_ppyoloe_r(num_classes=1, size="s")
    res = evaluate_model(model, loader, torch.device("cpu"), name="t", logger=lambda *_: None)
    assert res["n_gt"] == 4
    assert 0.0 <= res["mAP50"] <= 1.0


def test_perfect_predictions_give_perfect_metrics():
    gt = [{"rboxes": torch.tensor([[100.0, 100.0, 40.0, 20.0, 0.2]])} for _ in range(3)]
    pred = [{"rboxes": g["rboxes"].clone(), "scores": np.array([0.9])} for g in gt]
    res = evaluate_detections(pred, gt, conf=0.5)
    assert pytest.approx(res["mAP50"], abs=1e-6) == 1.0
    assert pytest.approx(res["mAP75"], abs=1e-6) == 1.0
    assert res["conf0.5_iou0.5"]["precision"] == 1.0
    assert res["conf0.5_iou0.5"]["recall"] == 1.0


def test_metrics_penalise_shifted_boxes():
    """A shifted box should count at IoU 0.5 but not at 0.75."""
    gt = [{"rboxes": torch.tensor([[100.0, 100.0, 40.0, 20.0, 0.0]])}]
    pred = [{"rboxes": torch.tensor([[108.0, 100.0, 40.0, 20.0, 0.0]]), "scores": np.array([0.9])}]
    res = evaluate_detections(pred, gt, conf=0.5)
    assert res["mAP50"] > 0.9
    assert res["mAP75"] < 0.1


def test_ema_follows_ppdet_threshold_schedule():
    """Decay must grow as min(decay, (1+step)/(10+step)), not with the exponential ramp.

    With the wrong ramp the decay is 0.34 at step 800 instead of 0.99, so the EMA tracks the
    model rather than averaging it and the localisation gain is lost.
    """
    model = build_ppyoloe_r(num_classes=1, size="s")
    ema = ModelEMA(model, decay=0.9998)
    ema.step = 0
    assert pytest.approx(ema._next_decay(), abs=1e-6) == 0.1
    ema.step = 820
    assert ema._next_decay() > 0.98
    ema.step = 10**6
    assert pytest.approx(ema._next_decay(), abs=1e-6) == 0.9998


def test_ema_averages_and_corrects_bias():
    torch.manual_seed(0)
    model = build_ppyoloe_r(num_classes=1, size="s")
    ema = ModelEMA(model, decay=0.9998)
    key = "yolo_head.pred_cls.0.bias"
    # with constant weights the bias correction leaves the EMA essentially equal to the model
    for _ in range(50):
        ema.update(model)
    assert torch.allclose(ema.apply().state_dict()[key], model.state_dict()[key], rtol=1e-3, atol=1e-3)
