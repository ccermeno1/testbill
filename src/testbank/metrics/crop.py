"""The two metrics that decide: coverage and contamination.

They measure whether the CROP works, which is all the downstream stain
classifier cares about. mAP can be excellent and the crop useless.

Coverage
    Fraction of the real banknote that lands inside the predicted crop with
    margin. A crop that cuts half a stain ruins the classifier. Target: >= 0.98
    at the 5th percentile.

Contamination
    Fraction of the crop that belongs to ANOTHER banknote. Background is
    harmless noise; a piece of the neighbouring banknote can bring in a foreign
    stain and cause a false positive.

An undetected banknote counts as coverage 0; it is not excluded. Excluding it
would make a detector that only finds the easy cases look better, which is the
inverted conclusion. The three figures are still reported separately --
detection rate, conditioned on detection, and aggregate -- because mixing them
hides which of the two problems a candidate has.

What counts as "another banknote"
---------------------------------
Also the quads the area filter left out. For scoring DETECTION they are
ignored, because penalizing the detector for finding them would be unfair.
For contamination they are not: they are real physical banknotes, and if a
piece of one enters the crop, the foreign stain enters with it. However we
keep the books on annotations does not change what is in the pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from testbank.metrics.core import ImageEval, PolygonCache, expand, union_of
from testbank.metrics.matching import DEFAULT_MATCH_IOU, match_image

DEFAULT_MARGIN = 0.05


class SceneType(str, Enum):
    """An image with a single banknote or with several.

    Contamination behaves completely differently in each case, and summarizing
    them together hides both. See `ContaminationConfig`.
    """

    SINGLE = "single"
    FAN = "fan"


def scene_type(item: ImageEval) -> SceneType:
    """Counts the area-filtered ones too.

    An image with one live annotation and one dropped neighbour IS a fan for
    contamination purposes: the neighbour is still in the pixels and still
    dirties the crop. Classifying it as "single banknote" would measure it
    against the strict threshold because of a banknote we decided not to use.
    """
    return (
        SceneType.FAN
        if len(item.truths) + len(item.ignored) > 1
        else SceneType.SINGLE
    )


@dataclass(frozen=True, slots=True)
class CropSample:
    """One truth and what happened to it. `detected=False` implies coverage 0."""

    sample_id: str
    truth_index: int
    detected: bool
    coverage: float
    contamination: float | None
    scene: SceneType = SceneType.SINGLE


def crop_samples(
    item: ImageEval,
    cache: PolygonCache | None = None,
    *,
    margin: float = DEFAULT_MARGIN,
    match_iou: float = DEFAULT_MATCH_IOU,
    contaminate_with_ignored: bool = True,
) -> list[CropSample]:
    cache = cache or PolygonCache.build(item)
    matching = match_image(item, cache, match_iou=match_iou)
    by_truth = matching.truth_to_prediction()

    extra = list(cache.ignored) if contaminate_with_ignored else []
    scene = scene_type(item)

    out: list[CropSample] = []
    for truth_index, truth in enumerate(cache.truths):
        pair = by_truth.get(truth_index)
        if pair is None:
            out.append(
                CropSample(item.sample_id, truth_index, False, 0.0, None, scene)
            )
            continue

        crop = expand(cache.predictions[pair.prediction_index], margin)
        coverage = (
            truth.intersection(crop).area / truth.area if truth.area > 0 else 0.0
        )

        others = [p for i, p in enumerate(cache.truths) if i != truth_index]
        rest = union_of(others + extra)
        contamination = (
            crop.intersection(rest).area / crop.area
            if rest is not None and crop.area > 0
            else 0.0
        )
        out.append(
            CropSample(
                item.sample_id,
                truth_index,
                True,
                min(coverage, 1.0),
                min(contamination, 1.0),
                scene,
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class CropReport:
    margin: float
    n_truths: int
    detected: int
    #: Low percentile of coverage over ALL truths (undetected = 0).
    coverage_p5_all: float
    #: The same percentile only over the detected ones. Kept apart on purpose.
    coverage_p5_detected: float
    #: The 10th percentile over all truths. With ~140 truths the 5th sits on
    #: the boundary between the undetected (0) and the rest and jumps; the
    #: 10th moves smoothly and discriminates between checkpoints.
    coverage_p10_all: float
    coverage_median: float
    contamination_p95: float
    contamination_median: float

    @property
    def detection_rate(self) -> float:
        return self.detected / self.n_truths if self.n_truths else 0.0

    def meets(self, target: float) -> bool:
        return self.coverage_p5_all >= target

    def to_dict(self) -> dict:
        return {
            "margin": self.margin,
            "n_truths": self.n_truths,
            "detected": self.detected,
            "detection_rate": self.detection_rate,
            "coverage_p5_all": self.coverage_p5_all,
            "coverage_p5_detected": self.coverage_p5_detected,
            "coverage_p10_all": self.coverage_p10_all,
            "coverage_median": self.coverage_median,
            "contamination_p95": self.contamination_p95,
            "contamination_median": self.contamination_median,
        }


def summarize(
    samples: list[CropSample], *, margin: float, percentile: float = 5.0
) -> CropReport:
    if not samples:
        return CropReport(margin, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    all_coverage = np.array([s.coverage for s in samples])
    detected = [s for s in samples if s.detected]
    det_coverage = (
        np.array([s.coverage for s in detected]) if detected else np.array([0.0])
    )
    contamination = (
        np.array([s.contamination for s in detected if s.contamination is not None])
        if detected
        else np.array([0.0])
    )
    return CropReport(
        margin=margin,
        n_truths=len(samples),
        detected=len(detected),
        coverage_p5_all=float(np.percentile(all_coverage, percentile)),
        coverage_p5_detected=float(np.percentile(det_coverage, percentile)),
        coverage_p10_all=float(np.percentile(all_coverage, 10.0)),
        coverage_median=float(np.median(all_coverage)),
        contamination_p95=float(np.percentile(contamination, 95)),
        contamination_median=float(np.median(contamination)),
    )


@dataclass(frozen=True, slots=True)
class SceneContamination:
    """Contamination of one scene type, against its own threshold."""

    scene: SceneType
    n: int
    median: float
    p95: float
    threshold: float

    @property
    def passes(self) -> bool:
        """Without samples there is no failing: no evidence, no verdict."""
        return self.n == 0 or self.p95 <= self.threshold

    def to_dict(self) -> dict:
        return {
            "scene": self.scene.value,
            "n": self.n,
            "median": self.median,
            "p95": self.p95,
            "threshold": self.threshold,
            "passes": self.passes,
        }


def contamination_by_scene(
    samples: list[CropSample],
    *,
    single_max: float,
    fan_max: float,
    percentile: float = 95.0,
) -> dict[str, SceneContamination]:
    """Single banknote and fans separately, each against its threshold.

    Measured only over crops that EXIST: an undetected truth produces no crop,
    so it has no contamination to measure. Its cost is already paid by
    coverage, which counts it as 0.
    """
    out: dict[str, SceneContamination] = {}
    for scene, threshold in (
        (SceneType.SINGLE, single_max),
        (SceneType.FAN, fan_max),
    ):
        values = np.array(
            [
                s.contamination
                for s in samples
                if s.scene is scene and s.detected and s.contamination is not None
            ]
        )
        out[scene.value] = SceneContamination(
            scene=scene,
            n=int(values.size),
            median=float(np.median(values)) if values.size else 0.0,
            p95=float(np.percentile(values, percentile)) if values.size else 0.0,
            threshold=threshold,
        )
    return out


def contamination_floor(
    items,
    *,
    margin: float = DEFAULT_MARGIN,
    percentile: float = 95.0,
) -> dict[str, float]:
    """Reachable floor: contamination of a PERFECT detector.

    It predicts exactly the truth, so what remains is what the annotation
    geometry imposes, and no candidate can go below it. The defaults of
    `ContaminationConfig` come from here; run it again when the export
    changes, because the floor changes with the data.

    `items` is used only for its truths and its ignored ones: any predictions
    it carries are discarded.
    """
    from testbank.metrics.core import ImageEval, Prediction

    collected: list[CropSample] = []
    for item in items:
        perfect = ImageEval(
            sample_id=item.sample_id,
            size=item.size,
            truths=item.truths,
            predictions=tuple(Prediction(q, 1.0) for q in item.truths),
            ignored=item.ignored,
        )
        collected.extend(crop_samples(perfect, margin=margin))

    by_scene = contamination_by_scene(
        collected, single_max=1.0, fan_max=1.0, percentile=percentile
    )
    return {name: entry.p95 for name, entry in by_scene.items()}
