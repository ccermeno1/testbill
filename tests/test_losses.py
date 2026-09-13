"""The loss recipes: each one whole, on the same own head.

What is checked and why:

- That the head each recipe demands is the one that comes out, and that a
  checkpoint carries it inside: loading DFL weights into a direct head would
  blow up halfway.
- That `target_distances` is the EXACT inverse of the decoder: if it were
  not, the DFL and the late L1 would learn towards a box that is not the truth.
- Properties of TAL and DFL that the papers guarantee.
- That every recipe produces finite losses with finite gradient on an
  untrained network, and that each reports ITS terms and not another's.
"""

from __future__ import annotations

import math

import pytest
import torch

from testbank.config import Config
from testbank.models.assign import AnchorGrid, build_anchor_grid, tal_assign
from testbank.models.decode import decode_outputs
from testbank.models.losses import (
    RECIPES,
    distribution_focal_loss,
    head_spec_for,
    losses_for_image,
    target_distances,
)
from testbank.models.overlap import pairwise_probiou
from testbank.models.train import load_model
from testbank.models.yolox_obb import STRIDES, HeadSpec, YoloxObb, decode_angle

torch.manual_seed(0)
SIDE = 128


def _config(recipe: str, **detector) -> Config:
    base = Config()
    loss = base.detector.loss.model_copy(update={"recipe": recipe})
    return base.model_copy(
        update={
            "detector": base.detector.model_copy(
                update={"image_size": SIDE, "epochs": 2, "loss": loss, **detector}
            )
        }
    )


def _single_image_outputs(model):
    outputs = model(torch.randn(1, 3, SIDE, SIDE))
    return outputs


def _targets():
    boxes = torch.tensor(
        [[64.0, 64.0, 60.0, 30.0, 0.4], [30.0, 90.0, 40.0, 20.0, 1.2]]
    )
    return boxes, torch.zeros(2, dtype=torch.long)


# --- the head each recipe demands ------------------------------------------


def test_each_recipe_fixes_its_head():
    assert head_spec_for(_config("own")) == HeadSpec()
    assert head_spec_for(_config("yolox_obb_fork")) == HeadSpec()
    ul = head_spec_for(_config("ultralytics_obb"))
    assert (ul.regression, ul.angle, ul.objectness) == ("dfl", "scalar", False)


def test_the_dfl_head_outputs_the_right_shapes():
    model = YoloxObb("nano", head=HeadSpec(regression="dfl", angle="scalar", objectness=False))
    for o in _single_image_outputs(model):
        assert o.distances.shape[1] == 4
        assert o.distribution.shape[1] == 4 * 16
        assert o.angle.shape[1] == 1
        assert o.objectness is None
        # The expectation of the distribution falls in [0, reg_max - 1].
        assert (o.distances >= 0).all() and (o.distances <= 15).all()


def test_the_direct_head_does_not_change():
    model = YoloxObb("nano")
    for o in _single_image_outputs(model):
        assert o.distribution is None and o.objectness is not None
        assert o.angle.shape[1] == 2


def test_the_checkpoint_carries_the_head_inside(tmp_path):
    """Loading DFL weights into a direct head would fail on sizes. The head
    specification travels with the weights, not with the config."""
    model = YoloxObb("nano", head=HeadSpec(regression="dfl", angle="scalar", objectness=False))
    path = tmp_path / "w.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "variant": "nano",
            "num_classes": 1,
            "head": model.head_spec.to_dict(),
        },
        path,
    )
    loaded = load_model(path)
    assert loaded.head_spec == model.head_spec


def test_an_old_checkpoint_without_head_loads_as_direct(tmp_path):
    model = YoloxObb("nano")
    path = tmp_path / "w.pt"
    torch.save({"model": model.state_dict(), "variant": "nano", "num_classes": 1}, path)
    assert load_model(path).head_spec == HeadSpec()


# --- scalar angle ----------------------------------------------------------


def test_the_scalar_angle_covers_half_a_turn():
    logits = torch.linspace(-12, 12, 200).view(1, 1, 200, 1)
    theta = decode_angle(logits.permute(0, 1, 2, 3).reshape(200, 1), "scalar")
    assert (theta >= 0).all() and (theta < math.pi).all()
    # Sigmoid from -inf to +inf sweeps (-pi/4, 3pi/4): modulo pi, the whole range.
    assert theta.max() - theta.min() > 0.95 * math.pi


# --- target_distances is the inverse of the decoder -------------------------


def test_target_distances_undoes_the_decoder():
    """Distances are built, decoded to boxes, and taken back: the same thing
    has to come out. Otherwise the DFL would learn towards another box."""
    from testbank.models.yolox_obb import HeadOutput, encode_angle

    grid_sizes = [(SIDE // s, SIDE // s) for s in STRIDES]
    grid = build_anchor_grid(grid_sizes, STRIDES, device=torch.device("cpu"))
    outputs = []
    torch.manual_seed(1)
    for (h, w), stride in zip(grid_sizes, STRIDES):
        distances = torch.rand(1, 4, h, w) * 4 + 0.5
        theta = torch.rand(1, h, w) * math.pi
        angle = encode_angle(theta).permute(0, 3, 1, 2)
        outputs.append(
            HeadOutput(
                distances=distances,
                angle=angle,
                objectness=torch.zeros(1, 1, h, w),
                classes=torch.zeros(1, 1, h, w),
                stride=stride,
            )
        )
    from testbank.dataio.formats import ImageSize

    boxes, _ = decode_outputs(outputs, ImageSize(SIDE, SIDE))
    back = target_distances(grid.centers, grid.strides, boxes)
    original = torch.cat([o.distances[0].permute(1, 2, 0).reshape(-1, 4) for o in outputs])
    assert torch.allclose(back, original, atol=1e-4), (back - original).abs().max()


# --- TAL ------------------------------------------------------------------


def _grid_for(side=SIDE) -> AnchorGrid:
    return build_anchor_grid([(side // s, side // s) for s in STRIDES], STRIDES, device=torch.device("cpu"))


def test_tal_only_picks_cells_inside_the_box():
    grid = _grid_for()
    boxes, classes = _targets()
    n = len(grid)
    predicted = torch.cat([grid.centers, torch.full((n, 1), 30.0), torch.full((n, 1), 15.0), torch.zeros(n, 1)], dim=1)
    scores = torch.rand(n, 1)
    a = tal_assign(predicted, scores, boxes, classes, grid, overlap=pairwise_probiou)
    from testbank.models.assign import points_in_rotated_boxes

    inside = points_in_rotated_boxes(grid.centers, boxes)
    assert a.positive.any()
    assert inside[a.positive].any(dim=1).all(), "a positive outside every box"


def test_tal_gives_at_most_topk_per_banknote_and_soft_targets_in_zero_one():
    grid = _grid_for()
    boxes, classes = _targets()
    n = len(grid)
    predicted = torch.cat([grid.centers, torch.full((n, 1), 30.0), torch.full((n, 1), 15.0), torch.zeros(n, 1)], dim=1)
    scores = torch.rand(n, 1)
    a = tal_assign(predicted, scores, boxes, classes, grid, overlap=pairwise_probiou, topk=5)
    for target in range(2):
        assert int(((a.matched == target) & a.positive).sum()) <= 5
    ts = a.target_scores
    assert ts is not None and (ts >= 0).all() and (ts <= 1).all()
    assert (ts[~a.positive] == 0).all(), "the background has target zero"
    assert (ts[a.positive].sum(dim=1) > 0).all()


def test_tal_without_banknotes_returns_all_background():
    grid = _grid_for()
    n = len(grid)
    predicted = torch.cat([grid.centers, torch.ones(n, 2) * 10, torch.zeros(n, 1)], dim=1)
    a = tal_assign(predicted, torch.rand(n, 1), torch.zeros(0, 5), torch.zeros(0, dtype=torch.long), grid, overlap=pairwise_probiou)
    assert not a.positive.any() and (a.target_scores == 0).all()


def test_tal_the_best_aligned_cell_receives_its_best_overlap():
    """It is TOOD's normalization: max(metric) -> max(overlap) per banknote."""
    grid = _grid_for()
    boxes, classes = _targets()
    n = len(grid)
    predicted = torch.cat([grid.centers, torch.full((n, 1), 60.0), torch.full((n, 1), 30.0), torch.full((n, 1), 0.4)], dim=1)
    scores = torch.full((n, 1), 0.9)
    a = tal_assign(predicted, scores, boxes, classes, grid, overlap=pairwise_probiou)
    for target in range(2):
        mine = a.positive & (a.matched == target)
        if not mine.any():
            continue
        assert a.target_scores[mine].max().item() == pytest.approx(a.matched_iou[mine].max().item(), abs=1e-5)


# --- DFL -----------------------------------------------------------------


def test_dfl_is_minimal_when_the_mass_is_in_the_right_bins():
    reg_max = 16
    target = torch.tensor([[3.25, 7.0, 0.5, 14.9]])
    logits = torch.full((1, 4 * reg_max), -20.0).view(1, 4, reg_max)
    for side, y in enumerate(target[0]):
        lo = int(y.floor())
        logits[0, side, lo] = math.log(float(lo + 1 - y) + 1e-9) + 20
        logits[0, side, min(lo + 1, reg_max - 1)] = math.log(float(y - lo) + 1e-9) + 20
    perfect = distribution_focal_loss(logits.view(1, -1), target, reg_max)
    worse = distribution_focal_loss(torch.zeros(1, 4 * reg_max), target, reg_max)
    assert perfect.item() < worse.item()
    assert perfect.item() >= 0


def test_dfl_clamps_the_target_to_the_last_bin():
    """A target beyond reg_max-1 cannot be represented: it is clamped instead
    of indexing out of bounds and blowing up."""
    logits = torch.randn(3, 4 * 16)
    loss = distribution_focal_loss(logits, torch.full((3, 4), 40.0), 16)
    assert torch.isfinite(loss).all()


# --- the recipes, end to end --------------------------------------------------


@pytest.mark.parametrize("recipe", RECIPES)
def test_each_recipe_gives_a_finite_loss_with_gradient(recipe):
    config = _config(recipe)
    model = YoloxObb("nano", head=head_spec_for(config))
    outputs = _single_image_outputs(model)
    boxes, classes = _targets()
    terms = losses_for_image(outputs, boxes, classes, config, torch.device("cpu"), epoch=1, total_epochs=2)
    assert torch.isfinite(terms.total)
    assert terms.num_positives > 0
    terms.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_each_recipe_reports_its_terms_and_not_another_recipe_s():
    boxes, classes = _targets()
    reported = {}
    for recipe in RECIPES:
        config = _config(recipe)
        model = YoloxObb("nano", head=head_spec_for(config))
        terms = losses_for_image(_single_image_outputs(model), boxes, classes, config, torch.device("cpu"), epoch=1, total_epochs=2)
        reported[recipe] = terms.to_dict()
    assert reported["own"]["angle"] > 0 and reported["own"]["dfl"] == 0 and reported["own"]["l1"] == 0
    assert reported["yolox_obb_fork"]["angle"] == 0 and reported["yolox_obb_fork"]["dfl"] == 0
    assert reported["yolox_obb_fork"]["objectness"] > 0
    assert reported["ultralytics_obb"]["dfl"] > 0
    assert reported["ultralytics_obb"]["objectness"] == 0 and reported["ultralytics_obb"]["angle"] == 0


def test_the_fork_l1_only_switches_on_at_the_end():
    boxes, classes = _targets()
    config = _config("yolox_obb_fork", epochs=20)
    model = YoloxObb("nano")
    outputs = _single_image_outputs(model)
    early = losses_for_image(outputs, boxes, classes, config, torch.device("cpu"), epoch=0, total_epochs=20)
    late = losses_for_image(outputs, boxes, classes, config, torch.device("cpu"), epoch=19, total_epochs=20)
    assert early.to_dict()["l1"] == 0.0
    assert late.to_dict()["l1"] > 0.0


def test_the_ultralytics_recipe_requires_the_dfl_head():
    config = _config("ultralytics_obb")
    model = YoloxObb("nano")  # direct head, on purpose
    boxes, classes = _targets()
    with pytest.raises(ValueError, match="DFL head"):
        losses_for_image(_single_image_outputs(model), boxes, classes, config, torch.device("cpu"))


def test_the_default_gains_are_those_of_each_original():
    config = Config()
    assert config.detector.loss.fork.box_gain == 5.0
    assert config.detector.loss.fork.tau == 1.0
    ul = config.detector.loss.ultralytics
    assert (ul.box_gain, ul.cls_gain, ul.dfl_gain) == (7.5, 0.5, 1.5)
    assert (ul.reg_max, ul.tal_topk, ul.tal_alpha, ul.tal_beta) == (16, 10, 0.5, 6.0)


def test_the_recipe_is_a_literal_and_accepts_nothing_else():
    """Through the constructor, which validates. `model_copy(update=...)` does
    NOT validate -- it is the footgun documented in `StrictModel` -- and that
    is why `_config` is no good here: it would accept the string and blow up
    later in the dispatch."""
    from testbank.config import LossConfig

    with pytest.raises(ValueError):
        LossConfig(recipe="mixed_gaussian")


def test_an_unvalidated_recipe_that_slips_through_blows_up_in_the_dispatch_with_a_message():
    boxes, classes = _targets()
    config = _config("mixed_gaussian")  # slips through: model_copy does not validate
    with pytest.raises(ValueError, match="unknown recipe"):
        losses_for_image(_single_image_outputs(YoloxObb("nano")), boxes, classes, config, torch.device("cpu"))
