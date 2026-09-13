"""mAP50 and mAP50-95 with rotated IoU, accumulated over ALL images.

101-point interpolation, like COCO. It is a choice, not a detail: the
alternative (all points, VOC 2010+) gives slightly different numbers, and if
we are going to compare against the published mAP of RTMDet-R or Ultralytics,
we have to compute it like they do.

The ranking is GLOBAL by confidence, not per image. Averaging the AP of each
image is a different metric and gives a different number: an image with a
single easy banknote would weigh the same as a fan of six.

Why there is a precompute phase
-------------------------------
The bootstrap resamples IMAGES 2000 times, and mAP50-95 walks 10 thresholds.
Matching inside the loop would be 2000 x 10 x 101 matchings with shapely:
measured, it does not finish in useful time.

But the matching of one image does NOT depend on which other images came out
in the resample -- it is local to the image. Only the global ranking and the
truth count depend on the set. So each image is matched once per threshold,
`(score, hit)` is stored per detection, and every bootstrap replicate reduces
to concatenating and sorting. Same number, three orders of magnitude faster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from testbank.metrics.core import PolygonCache
from testbank.metrics.matching import DEFAULT_MATCH_IOU, Outcome, match_all

#: The ten COCO thresholds: 0.50, 0.55, ..., 0.95.
COCO_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))

#: Recall points at which precision is interpolated.
_RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True, slots=True)
class ImageDetectionStats:
    """Matching of ONE image at ONE threshold, reduced to what AP uses.

    `scored` carries `(score, hit)` per detection, without the ignored ones:
    they add neither hit nor miss, so they leave the count entirely.
    """

    sample_id: str
    n_truths: int
    scored: tuple[tuple[float, int], ...]
    true_positives: int
    false_positives: int
    ignored: int


@dataclass(frozen=True, slots=True)
class DetectionCounts:
    true_positives: int
    false_positives: int
    ignored: int
    n_truths: int

    @property
    def detection_rate(self) -> float:
        return self.true_positives / self.n_truths if self.n_truths else 0.0


def image_stats(
    items,
    caches=None,
    *,
    iou_threshold: float = DEFAULT_MATCH_IOU,
    use_ignored: bool = True,
) -> list[ImageDetectionStats]:
    """Match once and keep just enough. It is what the bootstrap resamples."""
    items = list(items)
    caches = caches or [PolygonCache.build(i) for i in items]
    matchings = match_all(
        items, caches, match_iou=iou_threshold, use_ignored=use_ignored
    )
    out = []
    for matching in matchings:
        scored, tp, fp, ign = [], 0, 0, 0
        for _, outcome, score in matching.outcomes:
            if outcome is Outcome.IGNORED:
                ign += 1
                continue
            hit = 1 if outcome is Outcome.TRUE_POSITIVE else 0
            tp += hit
            fp += 1 - hit
            scored.append((score, hit))
        out.append(
            ImageDetectionStats(
                sample_id=matching.sample_id,
                n_truths=matching.n_truths,
                scored=tuple(scored),
                true_positives=tp,
                false_positives=fp,
                ignored=ign,
            )
        )
    return out


def ap_from_stats(stats) -> float:
    """AP over an already matched set. Only concatenates, sorts and accumulates."""
    stats = list(stats)
    n_truths = sum(s.n_truths for s in stats)
    if n_truths == 0:
        return float("nan")

    scored = [pair for s in stats for pair in s.scored]
    if not scored:
        return 0.0

    scored.sort(key=lambda pair: pair[0], reverse=True)
    hits = np.array([hit for _, hit in scored], dtype=np.float64)
    tp = np.cumsum(hits)
    fp = np.cumsum(1.0 - hits)
    recall = tp / n_truths
    precision = tp / np.maximum(tp + fp, 1e-12)

    # Monotone decreasing envelope: the precision at a given recall is the best
    # achievable at that recall or beyond.
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    # `searchsorted`, not `np.interp`. With false positives AFTER the last hit
    # -- the normal case with the low inference confidence -- recall stays put
    # and the vector has repeated values. `np.interp` with duplicate x returns
    # the LAST one, which is the lowest precision of the run, and sinks the AP.
    # Here the FIRST position with recall >= r is taken, which together with
    # the envelope gives the maximum of the tail, which is the definition.
    positions = np.searchsorted(recall, _RECALL_POINTS, side="left")
    interpolated = np.where(
        positions < precision.size, precision[np.minimum(positions, precision.size - 1)], 0.0
    )
    return float(interpolated.mean())


def counts_from_stats(stats) -> DetectionCounts:
    stats = list(stats)
    return DetectionCounts(
        true_positives=sum(s.true_positives for s in stats),
        false_positives=sum(s.false_positives for s in stats),
        ignored=sum(s.ignored for s in stats),
        n_truths=sum(s.n_truths for s in stats),
    )


# --- convenience facades ---------------------------------------------------


def average_precision(
    items,
    caches=None,
    *,
    iou_threshold: float = DEFAULT_MATCH_IOU,
    use_ignored: bool = True,
) -> float:
    return ap_from_stats(
        image_stats(
            items, caches, iou_threshold=iou_threshold, use_ignored=use_ignored
        )
    )


def mean_average_precision(
    items,
    caches=None,
    *,
    thresholds=COCO_THRESHOLDS,
    use_ignored: bool = True,
) -> float:
    values = [
        average_precision(items, caches, iou_threshold=t, use_ignored=use_ignored)
        for t in thresholds
    ]
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def counts(
    items,
    caches=None,
    *,
    iou_threshold: float = DEFAULT_MATCH_IOU,
    use_ignored: bool = True,
) -> DetectionCounts:
    return counts_from_stats(
        image_stats(
            items, caches, iou_threshold=iou_threshold, use_ignored=use_ignored
        )
    )


# --------------------------------------------------------------------------
# Bootstrap by image
# --------------------------------------------------------------------------
#
# Bootstrap intervals. Every row of the comparison table carries one.
#
# With ~500 images, validation is ~100 and the differences between candidates
# fall within the noise. An mAP of 0.81 against one of 0.78 says nothing if the
# intervals overlap widely, and a bare table of means invites reading as
# improvement what is dispersion.
#
# The resampling unit is the IMAGE, never the detection. Several banknotes in
# the same image share background, lighting, camera and annotator: they are
# correlated. Resampling detections treats them as independent and narrows the
# intervals artificially, which is the only thing worse than not having them,
# because it gives false confidence.

DEFAULT_SAMPLES = 2000
DEFAULT_SEED = 20260910


@dataclass(frozen=True, slots=True)
class Interval:
    value: float
    ci_low: float
    ci_high: float
    n: int
    samples: int

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "bootstrap_samples": self.samples,
        }

    def describe(self) -> str:
        return f"{self.value:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}] (n={self.n})"


def bootstrap_images(
    units,
    statistic,
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> Interval:
    """Percentile bootstrap over `units`, which are IMAGES.

    `statistic` receives a list of resampled images and returns a number. It
    receives the whole list and not a precomputed aggregate on purpose:
    metrics like mAP have a global ranking and cannot be averaged per image.

    Deterministic: same `seed`, same interval.
    """
    units = list(units)
    n = len(units)
    point = float(statistic(units))
    if n < 2:
        # With one image there is no dispersion to estimate. Returning a
        # zero-width interval would be lying; the point and n are returned so
        # that `compare` shows it for what it is.
        return Interval(point, float("nan"), float("nan"), n, 0)

    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        picked = rng.integers(0, n, size=n)
        values[i] = statistic([units[j] for j in picked])

    values = values[~np.isnan(values)]
    if values.size == 0:
        return Interval(point, float("nan"), float("nan"), n, samples)

    alpha = (1.0 - confidence) / 2.0
    return Interval(
        value=point,
        ci_low=float(np.percentile(values, 100 * alpha)),
        ci_high=float(np.percentile(values, 100 * (1.0 - alpha))),
        n=n,
        samples=samples,
    )


__all__ = [
    "DetectionCounts",
    "ImageDetectionStats",
    "Interval",
    "ap_from_stats",
    "average_precision",
    "bootstrap_images",
    "counts",
    "counts_from_stats",
    "image_stats",
    "mean_average_precision",
]
