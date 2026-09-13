"""Configuration validated with pydantic. Every run stores the resolved config."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Immutable config with no extra fields.

    Warning about `model_copy(update=...)`: pydantic v2 does NOT validate what
    is passed there. An `update={"out_of_bounds": "pad"}` leaves a `str` where
    an enum should be, and the frozen `config.yaml` of the run serializes it as
    is. To override, build the correct type -- which is what the CLI does -- and
    wherever the value may come from outside, coerce it at the point of use.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    root: Path = Path("data/raw")
    #: Container of split versions (`splits/v1`, `splits/v2`, ...).
    splits_dir: Path = Path("splits")
    #: Which version to read: `vN`, or None for the latest. Whatever gets
    #: resolved is written into the run record, so "latest" never stays
    #: ambiguous in a `run.json`.
    split_version: str | None = None
    derived_dir: Path = Path("data/derived")


class SplitConfig(StrictModel):
    """Only used in create mode. In adopt mode the partition is already given."""

    train: float = 0.70
    valid: float = 0.15
    test: float = 0.15
    seed: int = 20260910
    group_key: str = "none"
    group_regex: str | None = None
    group_manifest: Path | None = None
    i_confirm_independence: bool = False

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> SplitConfig:
        total = self.train + self.valid + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1, they sum to {total}")
        return self

    @property
    def ratios(self) -> tuple[float, float, float]:
        return (self.train, self.valid, self.test)


class AnnotationPolicyConfig(StrictModel):
    """Visibility threshold of the annotation policy.

    It is actually applied when annotating, not here: the code can only flag
    annotations that contradict it. Documented in the README as a guide.
    """

    visibility_threshold: float = Field(default=0.25, gt=0.0, lt=1.0)
    #: Filter ON LOAD: every annotation whose area is smaller than this fraction
    #: of the largest annotation in the same image is dropped, so that in every
    #: image the banknote in front is kept. It approximates
    #: `visibility_threshold`; it is not a measure of occlusion, and it deletes
    #: nothing: the source files stay intact and what is dropped is reported.
    #: Raising or lowering it does not require re-exporting the dataset.
    min_relative_area: float = Field(default=0.25, ge=0.0, le=1.0)


class CropConfig(StrictModel):
    """The margin decides coverage and contamination, so it lives in the config.

    Without fixing and recording it, two runs are not comparable: the margin
    trades one metric for the other directly.
    """

    margin: float = Field(default=0.05, ge=0.0, le=1.0)
    #: Sweep reported alongside, so that the trade-off stays visible.
    margin_sweep: tuple[float, ...] = (0.0, 0.05, 0.10)


class ContaminationConfig(StrictModel):
    """Two thresholds, because the distribution is bimodal, not continuous.

    Measured with a PERFECT detector (prediction = truth) on train+valid of the
    current export, at margin 0.05, with `contamination_floor`:

        single banknote   n=301   p95 = 0.0000
        fan               n=378   p95 = 0.8906

    A single threshold would be useless in both directions at once: too strict
    for fans, where not even a perfect detector can go below 0.89 -- if a
    banknote is partially covered, its box necessarily contains pixels of the
    one covering it -- and too lax for single-banknote images, where the floor
    is exactly zero and any contamination is a real error.

    Warning when reading the fan threshold: between the floor (0.89) and the
    ceiling (1.0) there are 11 points, so it discriminates little. To compare
    candidates on fans look at the MEDIAN, which the perfect detector leaves at
    0.023 and has plenty of headroom.

    Both numbers come from the data, so they must be re-derived when the export
    changes: `contamination_floor` in `metrics/crop.py` recomputes them.
    """

    #: Measured floor: 0.0000. Any contamination here is a real error.
    single_max: float = Field(default=0.01, ge=0.0, le=1.0)
    #: Measured floor: 0.8906. Leaves ~3 points of headroom, a quarter of the
    #: range that remains up to 1.0.
    fan_max: float = Field(default=0.92, ge=0.0, le=1.0)
    #: Percentile the two thresholds are applied to.
    percentile: float = Field(default=95.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _fan_is_not_stricter_than_single(self) -> ContaminationConfig:
        if self.fan_max < self.single_max:
            raise ValueError(
                "fan_max cannot be lower than single_max: a fan can never come "
                "out cleaner than a single-banknote image"
            )
        return self


class MetricsConfig(StrictModel):
    match_iou: float = Field(default=0.5, gt=0.0, lt=1.0)
    coverage_target: float = Field(default=0.98, gt=0.0, le=1.0)
    coverage_percentile: float = Field(default=5.0, ge=0.0, le=100.0)
    contamination: ContaminationConfig = ContaminationConfig()
    #: Confidence at which the counts of hits and false positives are REPORTED.
    #: Different from `detector.confidence_threshold` (0.01), which is the
    #: inference one and is deliberately low so that the precision-recall
    #: curve keeps its tail: AP needs all of it.
    #:
    #: With that 0.01, the raw count gives 1099 false positives against 116
    #: hits, and not because the model fires at everything -- they are
    #: detections of confidence 0.02 that AP already penalizes. The number
    #: measures how many boxes the NMS let through, not quality, and it does
    #: not improve with more training.
    #:
    #: This threshold answers the other question, the one people read: how
    #: many false positives there would be AT DEPLOYMENT. Both figures are
    #: reported, labelled.
    report_confidence: float = Field(default=0.25, gt=0.0, lt=1.0)
    bootstrap_samples: int = 2000
    #: The resampling unit is the image, not the detection: several banknotes
    #: in the same image are correlated, and resampling detections narrows the
    #: intervals artificially.
    bootstrap_unit: str = "image"
    seed: int = 20260910

    @field_validator("bootstrap_unit")
    @classmethod
    def _only_image(cls, value: str) -> str:
        if value != "image":
            raise ValueError(
                "the bootstrap unit has to be 'image': resampling detections "
                "ignores the correlation within each image"
            )
        return value


class OutOfBoundsPolicy(str, Enum):
    """What to do with banknotes that cross the image border.

    Our reader accepts them on purpose (tolerant range [-0.5, 1.5]): a banknote
    straddling the frame legitimately has vertices outside [0,1]. Ultralytics
    considers those labels corrupt and DROPS THE WHOLE IMAGE. Measured on the
    current export: 12 images lost out of 452, and precisely the hard cases.

    Also measured, how much banknote actually sticks out:

        99 annotations out of 679, in 82 images out of 452
        area outside the frame: median 0.4%, p90 5.8%, max 16.5%
        86% have less than 5% of their area outside
    """

    #: Clips the quads to the frame. No image is lost and not a single pixel
    #: is invented. The box becomes the VISIBLE part of the banknote, which is
    #: all a crop can contain anyway.
    CLIP = "clip"
    #: Adds a border so the full extent fits. Beware: the padding has to be
    #: UNIFORM AND ALWAYS ON, at inference too -- there is no annotation there
    #: to compute how much is needed. Fitting everything needs 23% per side,
    #: i.e. 53% of every image invented, in exchange for recovering a median
    #: of 0.4% of banknote.
    PAD = "pad"
    #: Touch nothing. Ultralytics will drop those images; how many is recorded
    #: instead of lost silently.
    KEEP = "keep"


class AngleWeightConfig(StrictModel):
    """Attenuates the angle loss on nearly square boxes.

    The problem: the `(cx, cy, w, h, theta)` representation is ambiguous when
    `w ~ h`. The box `(w, h, t)` and the box `(h, w, t+90)` are the SAME
    rectangle, but `(sin 2t, cos 2t)` sends them to opposite points of the
    circle. The model would receive two contradictory targets for one box.

    Not theoretical: 145 of the 762 annotations in the export (19%) have ratio
    < 1.1. It is the same 19% that triggers the unstable-anchor warning.

    The fix here is cheap and fits the annotation policy: if the rectangle is
    nearly square, the angle barely changes the crop, so there is no point in
    punishing the model for missing it. Its weight is lowered.

    It lives in the config and not hard-coded SO THAT IT CAN BE MEASURED: with
    `enabled=False` the model trains without the attenuator and is compared.
    It is an experiment, not a buried constant.

    BEWARE when interpreting: that 19% is suspected to be an artifact of the
    resize to 416x416, not of the domain. See the README.
    """

    enabled: bool = True
    #: Above this ratio the weight is 1: the angle is well defined.
    ratio_threshold: float = Field(default=1.1, gt=1.0)
    #: Weight at the perfect square (ratio 1). Zero ignores it entirely.
    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    #: How the weight rises between the square and the threshold. See `DECAYS`.
    decay: str = "smoothstep"

    @field_validator("decay")
    @classmethod
    def _known_decay(cls, value: str) -> str:
        from testbank.models.losses import DECAYS

        if value not in DECAYS:
            raise ValueError(f"unknown decay shape {value!r}; available: {sorted(DECAYS)}")
        return value


class ForkRecipeConfig(StrictModel):
    """The `buzhidaoshenme/YOLOX-OBB` recipe, with ITS defaults.

    Copied from its `yolo_head_obb_kld.py` (Apache-2.0): `reg_weight = 5.0`,
    `taf = 1.0`, and the L1 that switches on during the last
    `no_aug_epochs = 15`. Changing them here is legitimate -- they are config
    -- but then it is no longer "the fork's recipe" and the table has to say so.
    """

    box_gain: float = Field(default=5.0, ge=0.0)
    tau: float = Field(default=1.0, gt=0.0)
    #: The fork switches on an L1 over the raw regression during the last epochs
    #: (the ones without mosaic). Here: the last `l1_last_epochs`.
    l1_last_epochs: int = Field(default=15, ge=0)


class UltralyticsRecipeConfig(StrictModel):
    """The Ultralytics YOLO-OBB recipe, with the values it DOCUMENTS.

    Gains `box=7.5, cls=0.5, dfl=1.5`, `reg_max=16`, and the TAL assigner with
    `topk=10, alpha=0.5, beta=6.0`. They come from its public documentation,
    not from its code, which is AGPL and has not been read. The losses
    themselves are implemented from the papers: ProbIoU (Llerena 2021) and DFL
    (Li 2020); TAL from TOOD (Feng 2021).
    """

    box_gain: float = Field(default=7.5, ge=0.0)
    cls_gain: float = Field(default=0.5, ge=0.0)
    dfl_gain: float = Field(default=1.5, ge=0.0)
    reg_max: int = Field(default=16, ge=2)
    tal_topk: int = Field(default=10, ge=1)
    tal_alpha: float = Field(default=0.5, ge=0.0)
    tal_beta: float = Field(default=6.0, ge=0.0)


class DdgrcfRecipeConfig(StrictModel):
    """The `DDGRCF/YOLOX_OBB` recipe, with the values of ITS losses yaml.

    `configs/losses/yolox_losses_obb.yaml` (Apache-2.0): linear PolyIoU x5, obj
    and cls BCE x1 summed and normalized by positives, and an "extra" L1 that
    its trainer switches on during the last `no_aug_epochs = 3` epochs (the
    DOTA exp uses 2). The IoU is EXACT, polygon-based: here in pure torch
    (`models/overlap.py`) instead of its compiled operator.
    """

    box_gain: float = Field(default=5.0, ge=0.0)
    l1_last_epochs: int = Field(default=3, ge=0)


class LossConfig(StrictModel):
    """Which loss recipe trains the in-house head. Recorded with the run.

    Four recipes, each one WHOLE and never mixed with the others:

        own              ours: aligned IoU + attenuated angle + BCE, SimOTA
        yolox_obb_fork   KLD x5 + obj + IoU-weighted cls + late L1, SimOTA with KLD
        ultralytics_obb  ProbIoU x7.5 + DFL x1.5 + soft cls x0.5, TAL, no obj
        ddgrcf           EXACT PolyIoU x5 + obj + IoU-weighted cls + late L1, SimOTA

    The recipe also fixes the HEAD (`models/yolox_obb.HeadSpec`): Ultralytics
    requires distributional regression and a scalar angle, and has no
    objectness. And `ddgrcf` fixes the whole NETWORK as well: it is the port of
    its yaml, so that its DOTA weights can be loaded (`models/ddgrcf.py`).
    """

    recipe: Literal["own", "yolox_obb_fork", "ultralytics_obb", "ddgrcf"] = "own"
    angle_weight: AngleWeightConfig = AngleWeightConfig()
    fork: ForkRecipeConfig = ForkRecipeConfig()
    ultralytics: UltralyticsRecipeConfig = UltralyticsRecipeConfig()
    ddgrcf: DdgrcfRecipeConfig = DdgrcfRecipeConfig()


class DetectorConfig(StrictModel):
    """What gets handed to whichever trainer. It lives in the config and is
    recorded: two runs with different epochs are not comparable."""

    #: Deployment is mobile, so nano/small are prioritized.
    variant: str = "nano"
    epochs: int = Field(default=100, gt=0)
    image_size: int = Field(default=640, gt=0)
    batch_size: int = Field(default=8, gt=0)
    #: Evaluate on `valid` every N epochs (and always on the last one) and keep
    #: the best checkpoint as `best.pt`. 0 disables it: then `best.pt` is just
    #: the last epoch. It is checkpoint SELECTION, not early stopping: the
    #: cosine schedule runs whole, because stopping it halfway leaves the
    #: learning rate hanging where it should have decayed.
    eval_every: int = Field(default=5, ge=0)
    #: What "best" means. mAP50 by default and not coverage p5, which is the
    #: metric that decides: coverage saturates at 1.0 early and stops telling
    #: checkpoints apart, while mAP50 keeps moving. The final comparison of
    #: candidates still reads coverage and contamination.
    selection_metric: Literal["map50", "coverage_p5"] = "map50"
    #: Foreign checkpoint to start from, or None to train from scratch. For the
    #: in-house head, a Megvii `yolox_*.pth.tar` (COCO; loads backbone and neck,
    #: discards its head). For the DDGRCF port, its DOTA checkpoint. It lives
    #: in the config so it is frozen with the run: two runs with and without
    #: pretraining are not comparable.
    pretrained: Path | None = None
    #: Confidence threshold at INFERENCE. Deliberately low: detection metrics
    #: need the low-confidence tail to trace the precision-recall curve;
    #: cutting it higher inflates AP artificially.
    confidence_threshold: float = Field(default=0.01, gt=0.0, lt=1.0)
    #: IoU of the rotated NMS.
    nms_iou: float = Field(default=0.5, gt=0.0, lt=1.0)
    loss: LossConfig = LossConfig()
    out_of_bounds: OutOfBoundsPolicy = OutOfBoundsPolicy.CLIP
    #: Only with `out_of_bounds = pad`. Fraction of the side added on EACH
    #: border. 0.25 covers the maximum measured overflow (0.231).
    pad_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    #: Whether padding is also applied when INFERRING. By default NO: the
    #: decision is deliberate and remains open.
    #:
    #: With False, the model trains on padded images and then sees unpadded
    #: ones. It is a REAL MISMATCH: after the resize to `image_size`, a
    #: banknote occupies more pixels at inference than in training, because in
    #: training it competed with a border that added 53% of area.
    #:
    #: What to keep in mind when reading the numbers: a run with `pad` and no
    #: padding at inference measures the MISMATCHED pipeline. If it does badly,
    #: it does not show that padding is useless; it shows that training and
    #: predicting with different framings does not work, which was already
    #: known. To judge padding itself, set this to True.
    pad_at_inference: bool = False


class VizConfig(StrictModel):
    #: Fixed validation sample, always the same, to compare by eye.
    sample_count: int = 12
    #: Minimum confidence to DRAW a prediction. Different from the inference
    #: one (0.01), which is deliberately low so the precision-recall curve
    #: keeps its tail. Drawing that tail fills the image with confidence-0.01
    #: boxes and makes it unreadable: measured, 11-19 boxes per image covering
    #: the banknote. Here one looks, one does not measure.
    confidence_threshold: float = Field(default=0.25, ge=0.0, lt=1.0)
    seed: int = 20260910
    output_dir: Path = Path("runs/_inspection")
    dpi: int = 110


class Config(StrictModel):
    data: DataConfig = DataConfig()
    splits: SplitConfig = SplitConfig()
    annotation_policy: AnnotationPolicyConfig = AnnotationPolicyConfig()
    crop: CropConfig = CropConfig()
    detector: DetectorConfig = DetectorConfig()
    metrics: MetricsConfig = MetricsConfig()
    viz: VizConfig = VizConfig()
    runs_dir: Path = Path("runs")

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    def dump_yaml(self) -> str:
        return yaml.safe_dump(
            yaml.safe_load(self.model_dump_json()), sort_keys=False, allow_unicode=True
        )
