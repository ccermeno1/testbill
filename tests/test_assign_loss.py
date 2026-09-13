"""SimOTA assignment and losses of the own candidate."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="the own candidate needs torch")

from testbank.config import AngleWeightConfig
from testbank.models.assign import (
    Assignment,
    build_anchor_grid,
    enclosing_boxes,
    pairwise_iou,
    points_in_rotated_boxes,
    simota_assign,
)
from testbank.models.losses import angle_loss, compute_losses, iou_loss


def box(cx, cy, w, h, theta=0.0):
    return torch.tensor([[cx, cy, w, h, theta]], dtype=torch.float32)


# --- the grid -------------------------------------------------------------


def test_the_grid_has_one_cell_per_position():
    grid = build_anchor_grid([(52, 52), (26, 26), (13, 13)], (8, 16, 32))
    assert len(grid) == 52 * 52 + 26 * 26 + 13 * 13


def test_points_fall_at_the_CENTER_of_the_cell():
    """Without the half pixel, every box comes out biased half a cell: at
    stride 32 that is 16 pixels of systematic bias."""
    grid = build_anchor_grid([(2, 2)], (32,))
    assert torch.allclose(grid.centers[0], torch.tensor([16.0, 16.0]))
    assert torch.allclose(grid.centers[-1], torch.tensor([48.0, 48.0]))


# --- geometry -------------------------------------------------------------


def test_the_center_is_always_inside():
    points = torch.tensor([[100.0, 100.0]])
    assert bool(points_in_rotated_boxes(points, box(100, 100, 40, 20, 0.7))[0, 0])


def test_a_far_point_is_outside():
    points = torch.tensor([[300.0, 300.0]])
    assert not bool(points_in_rotated_boxes(points, box(100, 100, 40, 20))[0, 0])


def test_rotation_changes_which_points_are_inside():
    """If it did not, the check would be ignoring the angle."""
    point = torch.tensor([[100.0, 118.0]])  # above the center
    horizontal = box(100, 100, 60, 20, 0.0)
    vertical = box(100, 100, 60, 20, math.pi / 2)
    assert not bool(points_in_rotated_boxes(point, horizontal)[0, 0])
    assert bool(points_in_rotated_boxes(point, vertical)[0, 0])


def test_the_envelope_of_an_aligned_box_is_itself():
    out = enclosing_boxes(box(100, 100, 40, 20, 0.0))[0]
    assert torch.allclose(out, torch.tensor([80.0, 90.0, 120.0, 110.0]), atol=1e-4)


def test_the_envelope_of_a_rotated_one_is_larger():
    """It is the approximation that supports the cost, and one must know it is loose."""
    straight = enclosing_boxes(box(100, 100, 60, 20, 0.0))[0]
    rotated = enclosing_boxes(box(100, 100, 60, 20, math.pi / 4))[0]
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    assert area(rotated) > area(straight)


def test_iou_of_identical_boxes_is_one():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert float(pairwise_iou(a, a)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[50.0, 50.0, 60.0, 60.0]])
    assert float(pairwise_iou(a, b)) == pytest.approx(0.0)


# --- the assignment -------------------------------------------------------


def _scenario(n_targets=1):
    grid = build_anchor_grid([(16, 16)], (16,))
    targets = torch.cat(
        [box(80 + 100 * i, 128, 60, 30, 0.0) for i in range(n_targets)]
    )
    predicted = torch.cat(
        [grid.centers, torch.full((len(grid), 2), 40.0), torch.zeros(len(grid), 1)],
        dim=1,
    )
    scores = torch.full((len(grid),), 0.5)
    return grid, targets, predicted, scores


def test_an_image_without_banknotes_assigns_nothing():
    """Legitimate case: the area filter can empty an image of strips."""
    grid, _, predicted, scores = _scenario()
    result = simota_assign(predicted, scores, torch.zeros((0, 5)), grid)
    assert result.num_positives == 0
    assert not result.positive.any()


def test_every_banknote_receives_at_least_one_positive():
    """A banknote without positives generates no gradient: it is like not annotating it."""
    grid, targets, predicted, scores = _scenario(n_targets=2)
    result = simota_assign(predicted, scores, targets, grid)
    assigned = set(result.matched[result.positive].tolist())
    assert assigned == {0, 1}


def test_positives_fall_near_the_banknote():
    grid, targets, predicted, scores = _scenario()
    result = simota_assign(predicted, scores, targets, grid)
    centers = grid.centers[result.positive]
    distance = (centers - targets[0, :2]).abs().max()
    assert float(distance) < 100.0


def test_no_cell_serves_two_banknotes():
    """It would receive two different targets and learn the average, which
    is neither of the two."""
    grid, targets, predicted, scores = _scenario(n_targets=2)
    result = simota_assign(predicted, scores, targets, grid)
    assert isinstance(result, Assignment)
    # `matched` is a single index per cell: exclusivity is structural.
    assert result.matched.shape == (len(grid),)
    assert result.matched[result.positive].min() >= 0


def test_positives_are_a_minority():
    """If almost everything were positive, the background would stop being taught."""
    grid, targets, predicted, scores = _scenario()
    result = simota_assign(predicted, scores, targets, grid)
    assert 0 < result.num_positives < len(grid) // 4


# --- the losses -----------------------------------------------------------


def test_a_perfect_box_has_no_box_loss():
    b = box(100, 100, 60, 30, 0.0)
    assert float(iou_loss(b, b)) == pytest.approx(0.0, abs=1e-5)


def test_a_far_box_has_maximum_loss():
    assert float(iou_loss(box(0, 0, 10, 10), box(500, 500, 10, 10))) == pytest.approx(1.0)


def test_the_right_angle_has_no_loss():
    theta = torch.tensor([0.7])
    predicted = torch.stack((torch.sin(2 * theta), torch.cos(2 * theta)), dim=-1)
    wh = torch.tensor([[60.0, 20.0]])
    loss, _ = angle_loss(predicted, theta, wh)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_getting_the_angle_plus_180_is_not_penalized_either():
    """The property that motivates the whole representation, now in the loss."""
    theta = torch.tensor([0.3])
    predicted_rotated = torch.stack(
        (torch.sin(2 * (theta + math.pi)), torch.cos(2 * (theta + math.pi))), dim=-1
    )
    wh = torch.tensor([[60.0, 20.0]])
    loss, _ = angle_loss(predicted_rotated, theta, wh)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_the_attenuator_lowers_the_loss_of_a_nearly_square_box():
    """It is its reason to exist: not punishing an angle that is undefined."""
    theta = torch.tensor([0.0])
    predicted = torch.tensor([[1.0, 0.0]])  # 90 degrees of error
    nearly_square = torch.tensor([[50.0, 49.0]])

    with_att, weights = angle_loss(predicted, theta, nearly_square)
    without, _ = angle_loss(predicted, theta, nearly_square, enabled=False)
    # ratio 50/49 = 1.02, barely 20% of the way to the 1.1 threshold
    assert float(weights[0]) == pytest.approx(0.108, abs=0.01)
    assert float(with_att) == pytest.approx(0.108, abs=0.01)
    assert float(without) == pytest.approx(1.0)


def test_the_attenuator_does_not_touch_a_well_defined_box():
    theta = torch.tensor([0.0])
    predicted = torch.tensor([[1.0, 0.0]])
    elongated = torch.tensor([[100.0, 40.0]])
    with_att, weights = angle_loss(predicted, theta, elongated)
    without, _ = angle_loss(predicted, theta, elongated, enabled=False)
    assert float(weights[0]) == pytest.approx(1.0)
    assert float(with_att) == pytest.approx(float(without))


def test_normalized_by_N_and_not_by_the_sum_of_weights():
    """Dividing by the sum of weights would UNDO the attenuation.

    With a single box of weight 0.1, `(1 * 0.1) / 0.1` gives 1.0 again; and
    with a whole batch of ambiguous boxes the gradient would come out at full
    power, which is exactly what the attenuator exists to avoid.
    """
    theta = torch.zeros(2)
    predicted = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    # One well defined (weight 1) and one nearly square (weight ~0.108).
    wh = torch.tensor([[100.0, 40.0], [50.0, 49.0]])
    loss, weights = angle_loss(predicted, theta, wh)
    expected = (float(weights[0]) + float(weights[1])) / 2
    assert float(loss) == pytest.approx(expected, abs=1e-4)
    assert float(loss) < 1.0, "attenuating has to really reduce the magnitude"


def test_a_whole_batch_of_ambiguous_boxes_barely_weighs():
    """The case the normalization by sum of weights broke completely."""
    theta = torch.zeros(4)
    predicted = torch.tensor([[1.0, 0.0]] * 4)
    nearly_square = torch.tensor([[50.0, 49.0]] * 4)
    loss, _ = angle_loss(predicted, theta, nearly_square)
    assert float(loss) < 0.2


def test_without_positives_only_the_background_loss_remains():
    n = 32
    empty = Assignment(
        positive=torch.zeros(n, dtype=torch.bool),
        matched=torch.zeros(n, dtype=torch.long),
        matched_iou=torch.zeros(n),
    )
    terms = compute_losses(
        torch.zeros(n, 5), torch.zeros(n, 2), torch.zeros(n), torch.zeros(n, 1),
        torch.zeros((0, 5)), torch.zeros(0, dtype=torch.long), empty,
    )
    assert terms.num_positives == 0
    assert float(terms.box) == 0.0
    assert float(terms.objectness) > 0.0
    assert float(terms.total) > 0.0


def test_losses_are_reported_separately():
    """A single scalar hides which of the three terms is failing."""
    grid, targets, predicted, scores = _scenario()
    assignment = simota_assign(predicted, scores, targets, grid)
    terms = compute_losses(
        predicted,
        torch.zeros(len(grid), 2),
        torch.zeros(len(grid)),
        torch.zeros(len(grid), 1),
        targets,
        torch.zeros(1, dtype=torch.long),
        assignment,
        angle_config=AngleWeightConfig(),
    )
    keys = terms.to_dict()
    # `dfl` and `l1` belong to other recipes and are 0 here: they are always
    # reported so the log of every run has the same columns.
    assert set(keys) == {
        "box", "angle", "objectness", "classes", "dfl", "l1", "total", "num_positives"
    }
    assert keys["dfl"] == 0.0 and keys["l1"] == 0.0
    assert keys["num_positives"] > 0


def test_the_gradient_flows_from_the_total_loss():
    grid, targets, predicted, scores = _scenario()
    assignment = simota_assign(predicted, scores, targets, grid)
    angle = torch.zeros(len(grid), 2, requires_grad=True)
    objectness = torch.zeros(len(grid), requires_grad=True)
    terms = compute_losses(
        predicted, angle, objectness, torch.zeros(len(grid), 1),
        targets, torch.zeros(1, dtype=torch.long), assignment,
    )
    terms.total.backward()
    assert objectness.grad is not None and torch.any(objectness.grad != 0)
