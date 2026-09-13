"""The four loss recipes, each one WHOLE, and what they share.

A recipe is not a box loss: it is the combination of assigner, targets, terms,
normalization and gains that one concrete network uses. Mixing the box of one
with the assigner of another would give something that is neither of the two,
and the table would call it by a name that does not belong to it. So here each
recipe is a closed function, and `losses_for_image` only picks which one.

    own              aligned IoU + attenuated angle + obj + cls; SimOTA
    yolox_obb_fork   KLD x5 + obj + cls by overlap + late L1; SimOTA with KLD
    ultralytics_obb  ProbIoU x7.5 + DFL x1.5 + cls x0.5 with soft target; TAL
    ddgrcf           exact PolyIoU x5 + obj + cls by IoU + late L1; SimOTA -log IoU

What is faithful and what is not, said up front
-----------------------------------------------
**Fork.** Its code is Apache-2.0 and has been read in full. Its `get_losses` is
reproduced: same terms, same gains, same normalization by number of positives,
same class target (one-hot by overlap), same SimOTA with the KLD as cost. Two
deviations, both of head and not of loss: they regress the angle in degrees
directly and here `(sin 2t, cos 2t)` is used; and their late L1 goes over
`(dx, dy, log w, log h)` while here it goes over the four distances, which is
the raw regression of THIS head. Normalization is per image and then batch
mean, instead of over the whole batch: with one or two banknotes per photo the
difference is one of weighting between images, not of form.

**Ultralytics.** Its code is AGPL and has NOT been read. The composition --
which terms, which gains, which assigner -- comes from its public
documentation; the formulas come from the papers (ProbIoU: Llerena 2021; DFL:
Li 2020; TAL: Feng 2021). It is a reproduction of the DESCRIBED recipe, not a
copy. If their code had an undocumented detail, it is not here.

**DDGRCF.** Apache-2.0, read in full; see `_ddgrcf` for the one stated deviation.

=== The own recipe, in detail ===
Losses of the own candidate.

It is decomposed into three, instead of optimizing the rotated IoU directly:

    box       aligned IoU over the envelope
    angle     cosine over (sin 2t, cos 2t), ATTENUATED by the ratio
    presence  BCE for objectness and class

Why decompose
-------------
Optimizing the true rotated IoU would require building and intersecting
polygons on every iteration, and that does not fit in the loop (see
`assign.py`). The decomposition is an approximation: it does not optimize
exactly what is measured afterwards.

It is assumed knowingly, and it is measurable. The evaluation metric IS STILL
the rotated IoU by shapely, so if the decomposition allocates the trade-off
between shape and rotation badly, the final number says so. Training with an
approximation and measuring with the good thing is the correct order; the
reverse would be fooling oneself.

The angle attenuator
--------------------
It is the only non-standard thing. When the true box is nearly square its
angle is undefined -- `(w, h, t)` and `(h, w, t+90)` are the same rectangle --
so the angle loss is reduced. Parametrizable and switchable off so its
contribution can be measured. See the README.

=== The angle attenuator of the own recipe ===
Weight of the angle loss according to how square the box is.

Why it is needed
----------------
The `(cx, cy, w, h, theta)` representation is ambiguous when `w ~ h`: the box
`(w, h, t)` and the box `(h, w, t+90)` are the SAME rectangle. The
`(sin 2t, cos 2t)` encoding solves `t` and `t+180` being the same, but NOT
this: it sends the two versions to opposite points of the circle, so the model
would receive two contradictory targets for the same box.

Measured on the export: 145 of 762 annotations (19%) have ratio < 1.1.

Why attenuate and not fix it
----------------------------
Because in the ambiguous case the angle DOES NOT MATTER for what we care about.
A nearly square rectangle cropped with 5 degrees of error covers practically
the same, and the annotation policy already says an approximate rectangle is
enough. Punishing the model for missing something that is neither well defined
nor changes the result is spending capacity on noise.

The alternative would be the Gaussian representation, which absorbs the
ambiguity naturally. Discarded on purpose: it moves away from the shapely
rotated IoU we measure with, and that traceability weighs more than the
elegance of the formulation.

Everything parametrizable
-------------------------
Threshold, shape and floor go in `detector.loss.angle_weight`, with `enabled`
to switch it off. The question "how much does this contribute" is answered by
training with and without, not by reasoning. See `AngleWeightConfig`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from testbank.config import Config
from testbank.models.assign import (
    AnchorGrid,
    Assignment,
    build_anchor_grid,
    enclosing_boxes,
    pairwise_iou,
    simota_assign,
    tal_assign,
)
from testbank.models.decode import decode_outputs
from testbank.models.overlap import (
    kld_loss,
    pairwise_kld_loss,
    pairwise_probiou,
    pairwise_rotated_iou,
    probiou,
    rotated_iou,
)
from testbank.models.yolox_obb import STRIDES, HeadSpec, YoloxObb

# --------------------------------------------------------------------------
# Angle attenuator (own recipe)
# --------------------------------------------------------------------------

#: Rise shapes between the perfect square and the threshold. The key is what
#: `decay` accepts in the config; adding one is adding an entry here.
DECAYS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    # Rises straight. The simplest to interpret: the weight is the fraction of
    # the way travelled towards the threshold.
    "linear": lambda t: t,
    # Starts slowly and brakes at the end. Leaves the most ambiguous band
    # almost without weight instead of rising from the first moment.
    "smoothstep": lambda t: t * t * (3.0 - 2.0 * t),
    # Intermediate: starts slowly but does not brake.
    "quadratic": lambda t: t * t,
    # Step. Serves as a reference to measure whether the smooth transition
    # contributes anything over a hard cut.
    "step": lambda t: (t >= 1.0).to(t.dtype),
}


def angle_weight(
    ratio: torch.Tensor,
    *,
    enabled: bool = True,
    ratio_threshold: float = 1.1,
    min_weight: float = 0.0,
    decay: str = "smoothstep",
) -> torch.Tensor:
    """Weight in `[min_weight, 1]` for the angle loss of each box.

    `ratio` is long side / short side of the TRUE box, in pixels. In pixels and
    not normalized: normalizing scales x and y by different factors, and the
    ratio would stop being the geometric one -- the same mistake that already
    bit twice in this project.

    With `enabled=False` it returns ones: it is the control branch of the
    experiment, and it has to cost exactly the same to write as the other.
    """
    if not enabled:
        return torch.ones_like(ratio)
    if ratio_threshold <= 1.0:
        raise ValueError(
            f"ratio_threshold must be > 1, got {ratio_threshold}"
        )
    try:
        shape = DECAYS[decay]
    except KeyError:
        raise ValueError(
            f"unknown decay shape {decay!r}; available: {sorted(DECAYS)}"
        ) from None

    # A ratio below 1 does not exist: it is the long side over the short one.
    # If it arrives, someone swapped them, and truncating at 1 avoids negative
    # weights without hiding the problem (the weight would be minimal, not
    # absurd).
    progress = ((ratio.clamp(min=1.0) - 1.0) / (ratio_threshold - 1.0)).clamp(0.0, 1.0)
    return min_weight + (1.0 - min_weight) * shape(progress)


def side_ratio_px(width: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """Long side / short side, without assuming which of the two is which.

    The order of `w` and `h` is precisely what the ambiguity makes arbitrary,
    so the ratio cannot depend on it.
    """
    long_side = torch.maximum(width, height)
    short_side = torch.minimum(width, height)
    return long_side / short_side.clamp(min=1e-6)


# --------------------------------------------------------------------------
# Own recipe: terms
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LossTerms:
    """Each term separately. Merging them into a scalar hides which one fails."""

    box: torch.Tensor
    angle: torch.Tensor
    objectness: torch.Tensor
    classes: torch.Tensor
    total: torch.Tensor
    num_positives: int
    #: Only the Ultralytics recipe. Zero in the others, so the log of every
    #: run always has the same columns.
    dfl: torch.Tensor | None = None
    #: Only the fork recipe, and only in its last epochs.
    l1: torch.Tensor | None = None

    def to_dict(self) -> dict:
        """Only for logging. `detach` on purpose: converting to float a tensor
        still attached to the graph warns, and dragging the graph into a
        report dictionary is how memory leaks get in."""
        def value(t):
            return 0.0 if t is None else t.detach().item()

        return {
            "box": value(self.box),
            "angle": value(self.angle),
            "objectness": value(self.objectness),
            "classes": value(self.classes),
            "dfl": value(self.dfl),
            "l1": value(self.l1),
            "total": value(self.total),
            "num_positives": self.num_positives,
        }


def iou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """`1 - IoU` over the axis-aligned envelopes.

    IoU and not L1 over the coordinates: L1 treats a 5-pixel error the same on
    a small banknote and on a large one, when in the first it is fatal and in
    the second irrelevant. IoU is relative to size by construction.
    """
    if predicted.numel() == 0:
        return predicted.new_zeros(())
    boxes_p = enclosing_boxes(predicted)
    boxes_t = enclosing_boxes(target)
    iou = pairwise_iou(boxes_p, boxes_t).diagonal()
    return (1.0 - iou).mean()


def angle_loss(
    predicted_angle: torch.Tensor,
    target_theta: torch.Tensor,
    target_wh: torch.Tensor,
    *,
    enabled: bool = True,
    ratio_threshold: float = 1.1,
    min_weight: float = 0.0,
    decay: str = "smoothstep",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns `(loss, weights)`. The weights come out so they can be logged.

    `predicted_angle` is (P, 2) with `(sin 2t, cos 2t)` NOT normalized: the
    network outputs two free numbers. They are normalized here so the loss
    measures only direction; the magnitude means nothing and leaving it loose
    would give the model a way to lower the loss without getting the angle
    right.
    """
    if predicted_angle.numel() == 0:
        zero = predicted_angle.new_zeros(())
        return zero, predicted_angle.new_zeros((0,))

    predicted = F.normalize(predicted_angle, dim=-1, eps=1e-6)
    target = torch.stack(
        (torch.sin(2 * target_theta), torch.cos(2 * target_theta)), dim=-1
    )
    # 1 - cosine: zero when they point the same way, 2 in the worst case.
    # Continuous around the whole circle, which is the reason for doubling
    # the angle.
    per_box = 1.0 - (predicted * target).sum(dim=-1)

    weights = angle_weight(
        side_ratio_px(target_wh[:, 0], target_wh[:, 1]),
        enabled=enabled,
        ratio_threshold=ratio_threshold,
        min_weight=min_weight,
        decay=decay,
    )
    # Divided by the NUMBER of boxes, not by the sum of weights.
    #
    # Dividing by the sum undoes the attenuation: with a single box of weight
    # 0.1, `(1 * 0.1) / 0.1` gives 1.0 again, and with a whole batch of
    # ambiguous boxes the gradient comes out at full power -- exactly what the
    # attenuator exists to avoid. Normalizing by N, attenuating really reduces
    # the magnitude.
    #
    # That this lowers the total loss distorts nothing: candidates are compared
    # by the evaluation METRICS, not by the value of a loss, and comparing
    # losses across different loss functions means nothing anyway.
    return (per_box * weights).sum() / per_box.numel(), weights


def compute_losses(
    predicted_boxes: torch.Tensor,
    predicted_angle: torch.Tensor,
    predicted_objectness: torch.Tensor,
    predicted_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    assignment: Assignment,
    *,
    angle_config=None,
    box_gain: float = 5.0,
    angle_gain: float = 1.0,
    obj_gain: float = 1.0,
    cls_gain: float = 1.0,
) -> LossTerms:
    """Puts the three terms together. Everything in pixels."""
    positive = assignment.positive
    matched = assignment.matched[positive]
    device = predicted_objectness.device

    # Objectness: ALL cells take part. It is the only signal that teaches that
    # background is background, and with ~3500 cells and 2 banknotes it is
    # almost all the signal.
    obj_target = positive.to(predicted_objectness.dtype)
    objectness = F.binary_cross_entropy_with_logits(
        predicted_objectness, obj_target, reduction="mean"
    )

    if not positive.any():
        zero = torch.zeros((), device=device)
        return LossTerms(zero, zero, objectness * obj_gain, zero, objectness * obj_gain, 0)

    boxes_p = predicted_boxes[positive]
    boxes_t = target_boxes[matched]

    box = iou_loss(boxes_p, boxes_t)

    options = {} if angle_config is None else {
        "enabled": angle_config.enabled,
        "ratio_threshold": angle_config.ratio_threshold,
        "min_weight": angle_config.min_weight,
        "decay": angle_config.decay,
    }
    angle, _ = angle_loss(
        predicted_angle[positive], boxes_t[:, 4], boxes_t[:, 2:4], **options
    )

    classes = F.binary_cross_entropy_with_logits(
        predicted_classes[positive],
        F.one_hot(
            target_classes[matched], predicted_classes.shape[-1]
        ).to(predicted_classes.dtype),
        reduction="mean",
    )

    total = (
        box_gain * box + angle_gain * angle + obj_gain * objectness + cls_gain * classes
    )
    return LossTerms(
        box=box,
        angle=angle,
        objectness=objectness,
        classes=classes,
        total=total,
        num_positives=assignment.num_positives,
    )


# --------------------------------------------------------------------------
# The four recipes and the dispatch
# --------------------------------------------------------------------------

RECIPES = ("own", "yolox_obb_fork", "ultralytics_obb", "ddgrcf")


def head_spec_for(config: Config) -> HeadSpec:
    """The head each recipe demands. It is not selectable separately."""
    recipe = config.detector.loss.recipe
    if recipe == "ultralytics_obb":
        return HeadSpec(
            regression="dfl",
            reg_max=config.detector.loss.ultralytics.reg_max,
            angle="scalar",
            objectness=False,
        )
    if recipe == "ddgrcf":
        from testbank.models.ddgrcf import DdgrcfYoloxObb

        return DdgrcfYoloxObb.HEAD_SPEC
    return HeadSpec()


def build_model(config: Config):
    """The model the recipe demands. `ddgrcf` is a different NETWORK, not just
    a head: the port of its yaml, so that its DOTA weights load."""
    if config.detector.loss.recipe == "ddgrcf":
        from testbank.models.ddgrcf import DdgrcfYoloxObb

        return DdgrcfYoloxObb(num_classes=1)
    return YoloxObb(config.detector.variant, num_classes=1, head=head_spec_for(config))


def architecture_of(model) -> str:
    """What goes into the checkpoint to rebuild the network on load."""
    return "ddgrcf" if model.__class__.__name__ == "DdgrcfYoloxObb" else "yolox_obb"


# --- shared utilities ------------------------------------------------------


def _flatten(outputs):
    """The three levels in one list of cells. Everything (N, ·)."""
    angle = torch.cat([o.angle[0].permute(1, 2, 0).reshape(-1, o.angle.shape[1]) for o in outputs])
    classes = torch.cat(
        [o.classes[0].permute(1, 2, 0).reshape(-1, o.classes.shape[1]) for o in outputs]
    )
    objectness = (
        torch.cat([o.objectness[0].reshape(-1) for o in outputs])
        if outputs[0].objectness is not None
        else None
    )
    distances = torch.cat([o.distances[0].permute(1, 2, 0).reshape(-1, 4) for o in outputs])
    distribution = (
        torch.cat(
            [
                o.distribution[0].permute(1, 2, 0).reshape(-1, o.distribution.shape[1])
                for o in outputs
            ]
        )
        if outputs[0].distribution is not None
        else None
    )
    return angle, classes, objectness, distances, distribution


def _grid(outputs, device) -> AnchorGrid:
    return build_anchor_grid(
        [tuple(o.distances.shape[-2:]) for o in outputs], STRIDES, device=device
    )


def target_distances(
    centers: torch.Tensor, strides: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    """`(l, t, r, b)` in stride units from each cell to ITS rotated box.

    It is the exact inverse of `decode_level`: the offset of the box center
    with respect to the cell is taken into the box frame, and there

        l = w/2 - u    r = w/2 + u    t = h/2 - v    b = h/2 + v

    It serves as target both for the fork's late L1 and for Ultralytics' DFL.
    It can come out negative if the cell is outside the box; the assigners do
    not pick those cells, but it is clamped anyway just in case.
    """
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    dx, dy = cx - centers[:, 0], cy - centers[:, 1]
    cos, sin = torch.cos(theta), torch.sin(theta)
    u = dx * cos + dy * sin
    v = -dx * sin + dy * cos
    ltrb = torch.stack((w / 2 - u, h / 2 - v, w / 2 + u, h / 2 + v), dim=-1)
    return (ltrb / strides[:, None]).clamp(min=0.0)


# --- own recipe ------------------------------------------------------------


def _own(outputs, boxes, classes, config: Config, device) -> LossTerms:
    predicted_boxes, scores = decode_outputs(outputs, _image_size(config))
    angle, cls, obj, _, _ = _flatten(outputs)
    grid = _grid(outputs, device)
    assignment = simota_assign(predicted_boxes.detach(), scores.detach(), boxes, grid)
    return compute_losses(
        predicted_boxes,
        angle,
        obj,
        cls,
        boxes,
        classes,
        assignment,
        angle_config=config.detector.loss.angle_weight,
    )


# --- fork recipe -----------------------------------------------------------


def _fork(outputs, boxes, classes, config: Config, device, *, epoch, total_epochs) -> LossTerms:
    """`get_losses` of buzhidaoshenme/YOLOX-OBB (Apache-2.0), on our head.

    From the fork, literally: the KLD as SimOTA cost (x3.0) and as box loss
    (x5.0); `cls_target = one_hot * overlap`, with overlap `= 1 - kld_loss`;
    everything `sum / num_fg`; and the L1 in the last epochs.
    """
    options = config.detector.loss.fork
    predicted_boxes, scores = decode_outputs(outputs, _image_size(config))
    _, cls, obj, distances, _ = _flatten(outputs)
    grid = _grid(outputs, device)

    def overlap(p, t):
        return 1.0 - pairwise_kld_loss(p, t)

    assignment = simota_assign(
        predicted_boxes.detach(),
        scores.detach(),
        boxes,
        grid,
        iou_weight=3.0,
        overlap=overlap,
        overlap_cost="one_minus",
    )
    positive = assignment.positive
    num_fg = max(assignment.num_positives, 1)
    zero = torch.zeros((), device=device)

    objectness = F.binary_cross_entropy_with_logits(
        obj, positive.to(obj.dtype), reduction="sum"
    ) / num_fg
    if not positive.any():
        return LossTerms(zero, zero, objectness, zero, objectness, 0, dfl=zero, l1=zero)

    matched = assignment.matched[positive]
    boxes_p, boxes_t = predicted_boxes[positive], boxes[matched]

    box = kld_loss(boxes_p, boxes_t, tau=options.tau).sum() / num_fg
    cls_target = (
        F.one_hot(classes[matched], cls.shape[-1]).to(cls.dtype)
        * assignment.matched_iou[positive, None]
    )
    class_loss = F.binary_cross_entropy_with_logits(
        cls[positive], cls_target, reduction="sum"
    ) / num_fg

    l1 = zero
    if options.l1_last_epochs > 0 and epoch >= total_epochs - options.l1_last_epochs:
        target = target_distances(grid.centers[positive], grid.strides[positive], boxes_t)
        l1 = F.l1_loss(distances[positive], target, reduction="sum") / num_fg

    total = options.box_gain * box + objectness + class_loss + l1
    return LossTerms(
        box=box,
        angle=zero,
        objectness=objectness,
        classes=class_loss,
        total=total,
        num_positives=assignment.num_positives,
        dfl=zero,
        l1=l1,
    )


# --- Ultralytics recipe ----------------------------------------------------


def distribution_focal_loss(
    logits: torch.Tensor, target: torch.Tensor, reg_max: int
) -> torch.Tensor:
    """DFL (Li et al., 2020). `logits` (P, 4*reg_max), `target` (P, 4). -> (P,).

    Each target distance `y` falls between two integer bins `yl <= y < yr`; the
    loss is the cross-entropy against both, weighted by closeness:

        DFL = -(yr - y) log p(yl) - (y - yl) log p(yr)

    So the distribution learns to concentrate mass around the real value
    instead of only getting its expectation right. Mean over the four distances.
    """
    positives = logits.shape[0]
    logits = logits.view(positives, 4, reg_max)
    target = target.clamp(min=0.0, max=reg_max - 1 - 0.01)
    left = target.floor().long()
    right = left + 1
    weight_left = right.to(target.dtype) - target
    weight_right = 1.0 - weight_left
    log_probabilities = logits.log_softmax(dim=-1)
    loss = -(
        log_probabilities.gather(-1, left[..., None]).squeeze(-1) * weight_left
        + log_probabilities.gather(-1, right[..., None]).squeeze(-1) * weight_right
    )
    return loss.mean(dim=-1)


def _ultralytics(outputs, boxes, classes, config: Config, device) -> LossTerms:
    """The v8-OBB recipe, rebuilt from its documentation and the papers.

    No objectness: the class BCE is computed over ALL cells against the soft
    target of the TAL, which is zero in the background and the normalized
    localization quality in the positives. Box and DFL are weighted by that
    same target. Everything is divided by its sum, which is the normalization
    Ultralytics documents (`target_scores_sum`).
    """
    options = config.detector.loss.ultralytics
    predicted_boxes, _ = decode_outputs(outputs, _image_size(config))
    _, cls, _, _, distribution = _flatten(outputs)
    if distribution is None:
        raise ValueError(
            "the ultralytics_obb recipe requires the DFL head; build the model "
            "with head_spec_for(config)"
        )
    grid = _grid(outputs, device)
    assignment = tal_assign(
        predicted_boxes.detach(),
        torch.sigmoid(cls).detach(),
        boxes,
        classes,
        grid,
        overlap=pairwise_probiou,
        topk=options.tal_topk,
        alpha=options.tal_alpha,
        beta=options.tal_beta,
    )
    target_scores = assignment.target_scores
    assert target_scores is not None
    scores_sum = target_scores.sum().clamp(min=1.0)
    zero = torch.zeros((), device=device)

    class_loss = F.binary_cross_entropy_with_logits(
        cls, target_scores, reduction="sum"
    ) / scores_sum

    positive = assignment.positive
    if not positive.any():
        total = options.cls_gain * class_loss
        return LossTerms(zero, zero, zero, class_loss, total, 0, dfl=zero, l1=zero)

    matched = assignment.matched[positive]
    boxes_p, boxes_t = predicted_boxes[positive], boxes[matched]
    weight = target_scores.sum(dim=-1)[positive]

    box = ((1.0 - probiou(boxes_p, boxes_t)) * weight).sum() / scores_sum
    target = target_distances(grid.centers[positive], grid.strides[positive], boxes_t)
    dfl = (
        distribution_focal_loss(distribution[positive], target, options.reg_max) * weight
    ).sum() / scores_sum

    total = options.box_gain * box + options.cls_gain * class_loss + options.dfl_gain * dfl
    return LossTerms(
        box=box,
        angle=zero,
        objectness=zero,
        classes=class_loss,
        total=total,
        num_positives=assignment.num_positives,
        dfl=dfl,
        l1=zero,
    )


# --- DDGRCF recipe ---------------------------------------------------------


def regularize_angle_ddgrcf(boxes: torch.Tensor) -> torch.Tensor:
    """Their `mintheta_obb`: the angle in `(-pi/4, pi/4]`, swapping w and h.

    `(w, h, t)` and `(h, w, t + pi/2)` are the same rectangle; their convention
    picks the representation with the smaller |angle|. Their network predicts
    the raw angle, so the target has to be in that range or it would not be
    reachable with a small gradient.
    """

    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    theta1 = (theta + math.pi / 2) % math.pi - math.pi / 2
    theta2 = (theta + math.pi) % math.pi - math.pi / 2
    swap = theta1.abs() >= theta2.abs()
    return torch.stack(
        (
            cx,
            cy,
            torch.where(swap, h, w),
            torch.where(swap, w, h),
            torch.where(swap, theta2, theta1),
        ),
        dim=-1,
    )


def _ddgrcf(outputs, boxes, classes, config: Config, device, *, epoch, total_epochs) -> LossTerms:
    """`get_losses` of DDGRCF/YOLOX_OBB (Apache-2.0), on the port of their network.

    Literal from their `detectx.py` + `obbdetectx.py` + loss yaml: SimOTA with
    cost `-log(IoU)` x3 and EXACT IoU; box `1 - IoU` x5; obj and cls BCE with
    target `one_hot * IoU`; everything `sum / num_fg`; extra L1 over the raw
    regression in the last epochs.

    One deviation, stated: their extra L1 leaves the ANGLE target at zero
    (their `get_reg_l1_target` fills 4 of 5 components), which pushes the raw
    angle towards 0 in the last epochs. Here the target is the real angle. It
    is almost certainly an oversight of theirs, not a decision, and replicating
    it would be copying the bug under the name of fidelity.
    """
    options = config.detector.loss.ddgrcf
    boxes = regularize_angle_ddgrcf(boxes)
    predicted_boxes, scores = decode_outputs(outputs, _image_size(config))
    angle, cls, obj, raw, _ = _flatten(outputs)
    grid = _grid(outputs, device)

    assignment = simota_assign(
        predicted_boxes.detach(),
        scores.detach(),
        boxes,
        grid,
        iou_weight=3.0,
        overlap=pairwise_rotated_iou,
        overlap_cost="neg_log",
    )
    positive = assignment.positive
    num_fg = max(assignment.num_positives, 1)
    zero = torch.zeros((), device=device)

    objectness = F.binary_cross_entropy_with_logits(
        obj, positive.to(obj.dtype), reduction="sum"
    ) / num_fg
    if not positive.any():
        return LossTerms(zero, zero, objectness, zero, objectness, 0, dfl=zero, l1=zero)

    matched = assignment.matched[positive]
    boxes_p, boxes_t = predicted_boxes[positive], boxes[matched]

    box = (1.0 - rotated_iou(boxes_p, boxes_t)).sum() / num_fg
    cls_target = (
        F.one_hot(classes[matched], cls.shape[-1]).to(cls.dtype)
        * assignment.matched_iou[positive, None]
    )
    class_loss = F.binary_cross_entropy_with_logits(
        cls[positive], cls_target, reduction="sum"
    ) / num_fg

    l1 = zero
    if options.l1_last_epochs > 0 and epoch >= total_epochs - options.l1_last_epochs:
        # Their `get_reg_l1_target`: (cx/stride - i, cy/stride - j, log w/stride,
        # log h/stride), with (i, j) the cell corner. Our grid stores the cell
        # CENTER, hence the half stride.
        stride = grid.strides[positive]
        corner = (grid.centers[positive] - stride[:, None] / 2) / stride[:, None]
        target = torch.stack(
            (
                boxes_t[:, 0] / stride - corner[:, 0],
                boxes_t[:, 1] / stride - corner[:, 1],
                torch.log(boxes_t[:, 2] / stride + 1e-8),
                torch.log(boxes_t[:, 3] / stride + 1e-8),
                boxes_t[:, 4],
            ),
            dim=-1,
        )
        raw_all = torch.cat((raw[positive], angle[positive]), dim=-1)
        l1 = F.l1_loss(raw_all, target, reduction="sum") / num_fg

    total = options.box_gain * box + objectness + class_loss + l1
    return LossTerms(
        box=box,
        angle=zero,
        objectness=objectness,
        classes=class_loss,
        total=total,
        num_positives=assignment.num_positives,
        dfl=zero,
        l1=l1,
    )


# --- dispatch --------------------------------------------------------------


def losses_for_image(
    outputs, boxes, classes, config: Config, device, *, epoch: int = 0, total_epochs: int = 1
) -> LossTerms:
    recipe = config.detector.loss.recipe
    if recipe == "own":
        return _own(outputs, boxes, classes, config, device)
    if recipe == "yolox_obb_fork":
        return _fork(
            outputs, boxes, classes, config, device, epoch=epoch, total_epochs=total_epochs
        )
    if recipe == "ultralytics_obb":
        return _ultralytics(outputs, boxes, classes, config, device)
    if recipe == "ddgrcf":
        return _ddgrcf(
            outputs, boxes, classes, config, device, epoch=epoch, total_epochs=total_epochs
        )
    raise ValueError(f"unknown recipe: {recipe!r}; available: {RECIPES}")


def _image_size(config: Config):
    from testbank.dataio.formats import ImageSize

    side = config.detector.image_size
    return ImageSize(side, side)


__all__ = [
    "DECAYS",
    "RECIPES",
    "LossTerms",
    "angle_loss",
    "angle_weight",
    "architecture_of",
    "build_model",
    "compute_losses",
    "distribution_focal_loss",
    "head_spec_for",
    "iou_loss",
    "losses_for_image",
    "regularize_angle_ddgrcf",
    "side_ratio_px",
    "target_distances",
]
