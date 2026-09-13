"""Greedy matching by descending confidence, with rotated IoU.

The delicate point is not the greedy part, it is that a detection has THREE
possible outcomes, not two.

Annotations that must neither be detected nor penalized
--------------------------------------------------------
Two mechanisms of this project leave REAL banknotes out of the truth:

1. The 25% visibility policy. A banknote showing only a strip is not
   annotated. The specification itself warned: *"in fan images the model may
   correctly detect banknotes that are not annotated and they will count as
   false positives"*.

2. The relative area filter. It drops annotations that DO exist in the source
   file.

In both cases, if the detector finds that banknote, it is right. Counting it
as a false positive punishes the detector for doing its job and sinks
precision precisely on fan images, which are the ones that matter.

Nothing can be done about case 1: if nobody annotated it, there is nothing to
compare against. Case 2 can be handled, because we have the dropped quad. It
enters as `ignored`: a detection that falls mostly on one of those quads adds
neither a hit nor a miss; it simply leaves the count.

The criterion is IoA, not IoU: `area(pred ∩ ignored) / area(pred)`. With IoU,
a detection of the WHOLE banknote against an annotated strip of that same
banknote would give a low IoU and slip through as a false positive. The
question we want to ask is "is this detection explained by a real banknote we
decided not to use?", and that is a fraction of the detection, not a
symmetric intersection.

`evaluate` reports the figures WITH and WITHOUT this exclusion, because the
size of the effect is itself a datum about annotation quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from testbank.metrics.core import ImageEval, PolygonCache, iou

DEFAULT_MATCH_IOU = 0.5


class Outcome(str, Enum):
    TRUE_POSITIVE = "tp"
    FALSE_POSITIVE = "fp"
    IGNORED = "ignored"


@dataclass(frozen=True, slots=True)
class MatchedPair:
    prediction_index: int
    truth_index: int
    iou: float
    score: float


@dataclass(frozen=True, slots=True)
class ImageMatching:
    sample_id: str
    pairs: tuple[MatchedPair, ...]
    #: (prediction index, outcome, score) for ALL predictions.
    outcomes: tuple[tuple[int, Outcome, float], ...]
    #: Indices of truths nobody detected. They count as misses, never excluded.
    missed: tuple[int, ...]

    @property
    def n_truths(self) -> int:
        return len(self.pairs) + len(self.missed)

    @property
    def detected(self) -> int:
        return len(self.pairs)

    def truth_to_prediction(self) -> dict[int, MatchedPair]:
        return {pair.truth_index: pair for pair in self.pairs}


def match_image(
    item: ImageEval,
    cache: PolygonCache | None = None,
    *,
    match_iou: float = DEFAULT_MATCH_IOU,
    use_ignored: bool = True,
) -> ImageMatching:
    """Greedy by descending confidence. A truth is matched at most once.

    Greedy and not optimal assignment on purpose: it is what the usual
    detection metrics do (COCO included), and changing it would make our
    numbers incomparable with published ones.
    """
    cache = cache or PolygonCache.build(item)
    order = sorted(
        range(len(item.predictions)),
        key=lambda i: item.predictions[i].score,
        reverse=True,
    )

    taken: set[int] = set()
    pairs: list[MatchedPair] = []
    outcomes: list[tuple[int, Outcome, float]] = []

    for pred_index in order:
        prediction = cache.predictions[pred_index]
        score = item.predictions[pred_index].score

        best_truth, best_iou = None, 0.0
        for truth_index, truth in enumerate(cache.truths):
            if truth_index in taken:
                continue
            value = iou(prediction, truth)
            if value > best_iou:
                best_truth, best_iou = truth_index, value

        if best_truth is not None and best_iou >= match_iou:
            taken.add(best_truth)
            pairs.append(
                MatchedPair(pred_index, best_truth, best_iou, score)
            )
            outcomes.append((pred_index, Outcome.TRUE_POSITIVE, score))
            continue

        if use_ignored and _falls_on_ignored(
            prediction, cache, threshold=match_iou
        ):
            outcomes.append((pred_index, Outcome.IGNORED, score))
            continue

        outcomes.append((pred_index, Outcome.FALSE_POSITIVE, score))

    missed = tuple(i for i in range(len(cache.truths)) if i not in taken)
    return ImageMatching(
        sample_id=item.sample_id,
        pairs=tuple(pairs),
        outcomes=tuple(outcomes),
        missed=missed,
    )


def _falls_on_ignored(prediction, cache: PolygonCache, *, threshold: float) -> bool:
    """IoA over the union of the ignored quads.

    Over the UNION and not pairwise: a detection shared between two adjacent
    strips, 30% each, is 60% explained by real annotations and should not
    count as a false positive for failing to reach the threshold with either.
    """
    if not cache.ignored or prediction.area <= 0:
        return False
    covered = 0.0
    from shapely.ops import unary_union

    merged = unary_union(cache.ignored)
    if merged.is_empty:
        return False
    covered = prediction.intersection(merged).area
    return covered / prediction.area >= threshold


def match_all(
    items,
    caches=None,
    *,
    match_iou: float = DEFAULT_MATCH_IOU,
    use_ignored: bool = True,
) -> list[ImageMatching]:
    items = list(items)
    caches = caches or [PolygonCache.build(i) for i in items]
    return [
        match_image(item, cache, match_iou=match_iou, use_ignored=use_ignored)
        for item, cache in zip(items, caches)
    ]
