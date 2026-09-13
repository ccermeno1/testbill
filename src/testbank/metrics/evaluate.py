"""Assembles a run's `metrics.json`.

Reading order of the report, which is the order in which decisions are made:

1. `crop` -- coverage and contamination. They decide whether the crop works.
2. `detection` -- mAP and detection rate. They say whether it finds the banknotes.
3. `geometry` -- angle and vertices. Diagnostic, never a success criterion.

Everything that decides carries an interval. What is diagnostic does not, so
as not to suggest it is being compared on.

Matching happens ONCE per image and threshold, and the bootstrap resamples
indices over what is already computed. Matching inside the loop would be 2000
replicates x 10 thresholds x 101 images of shapely, which measured does not
finish. See `detection.py`.
"""

from __future__ import annotations

import numpy as np

from testbank.config import Config
from testbank.metrics.core import (
    PolygonCache,
    angle_error_deg,
    longest_side_px,
    vertex_distances_px,
)
from testbank.metrics.crop import (
    contamination_by_scene,
    crop_samples,
    summarize,
)
from testbank.metrics.detection import (
    COCO_THRESHOLDS,
    ap_from_stats,
    bootstrap_images,
    counts_from_stats,
    image_stats,
)
from testbank.metrics.matching import match_image


def _filter_by_score(item, minimum: float):
    """Same image with only the predictions above `minimum`."""
    from testbank.metrics.core import ImageEval

    return ImageEval(
        sample_id=item.sample_id,
        size=item.size,
        truths=item.truths,
        predictions=tuple(p for p in item.predictions if p.score >= minimum),
        ignored=item.ignored,
    )


def _percentile(values, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def geometry_diagnostics(items, caches, *, match_iou: float) -> dict:
    """Angle and per-vertex distance, only over matched pairs.

    Diagnostic, not a success criterion: the specification says exact
    geometric precision is not the goal, so these numbers go without an
    interval so that nobody uses them to pick a candidate.
    """
    angles: list[float] = []
    vertex_px: list[float] = []
    vertex_frac: list[float] = []

    for item, cache in zip(items, caches):
        matching = match_image(item, cache, match_iou=match_iou)
        for pair in matching.pairs:
            truth = item.truths[pair.truth_index]
            predicted = item.predictions[pair.prediction_index].quad
            angles.append(angle_error_deg(truth, predicted, item.size))
            side = longest_side_px(truth, item.size)
            for distance in vertex_distances_px(truth, predicted, item.size):
                vertex_px.append(distance)
                if side > 0:
                    vertex_frac.append(distance / side)

    return {
        "n_matched_pairs": len(angles),
        "angle_error_deg": {
            "median": _percentile(angles, 50),
            "p95": _percentile(angles, 95),
        },
        "vertex_distance_px": {
            "median": _percentile(vertex_px, 50),
            "p95": _percentile(vertex_px, 95),
        },
        "vertex_distance_fraction_of_longest_side": {
            "median": _percentile(vertex_frac, 50),
            "p95": _percentile(vertex_frac, 95),
        },
        "note": (
            "Diagnostic. An approximate rectangle is the annotation policy, so "
            "the per-vertex deviation is not a detector failure."
        ),
    }


def evaluate(items, config: Config | None = None, *, split: str = "valid") -> dict:
    """Full metrics of a set of already predicted images."""
    config = config or Config()
    items = list(items)
    caches = [PolygonCache.build(i) for i in items]

    match_iou = config.metrics.match_iou
    samples = config.metrics.bootstrap_samples
    seed = config.metrics.seed
    indices = list(range(len(items)))

    # --- precompute, once only --------------------------------------------
    thresholds = tuple(sorted({match_iou, *COCO_THRESHOLDS}))
    stats = {
        t: image_stats(items, caches, iou_threshold=t) for t in thresholds
    }
    stats_no_ignore = image_stats(
        items, caches, iou_threshold=match_iou, use_ignored=False
    )

    crops = {}
    for margin in config.crop.margin_sweep:
        crops[margin] = [
            crop_samples(item, cache, margin=margin, match_iou=match_iou)
            for item, cache in zip(items, caches)
        ]

    # --- aggregation and bootstrap over indices ---------------------------
    map50 = bootstrap_images(
        indices,
        lambda picked: ap_from_stats([stats[match_iou][i] for i in picked]),
        samples=samples,
        seed=seed,
    )

    def map50_95(picked) -> float:
        values = [
            ap_from_stats([stats[t][i] for i in picked]) for t in COCO_THRESHOLDS
        ]
        finite = [v for v in values if not np.isnan(v)]
        return float(np.mean(finite)) if finite else float("nan")

    map50_95_interval = bootstrap_images(
        indices, map50_95, samples=samples, seed=seed
    )

    with_ignored = counts_from_stats(stats[match_iou])
    without_ignored = counts_from_stats(stats_no_ignore)

    # Counts at the DECISION confidence: the question people read is how many
    # false positives there would be at deployment, not how many boxes the NMS
    # let through at the low threshold the precision-recall curve needs.
    decision = config.metrics.report_confidence
    decided = [_filter_by_score(item, decision) for item in items]
    decided_counts = counts_from_stats(
        image_stats(decided, iou_threshold=match_iou)
    )

    crop_by_margin = {}
    for margin, per_image in crops.items():
        flat = [s for group in per_image for s in group]
        report = summarize(
            flat, margin=margin, percentile=config.metrics.coverage_percentile
        )

        def coverage_p5(picked, per_image=per_image, margin=margin) -> float:
            collected = [s for i in picked for s in per_image[i]]
            return summarize(
                collected,
                margin=margin,
                percentile=config.metrics.coverage_percentile,
            ).coverage_p5_all

        def contamination_p95(picked, per_image=per_image, margin=margin) -> float:
            collected = [s for i in picked for s in per_image[i]]
            return summarize(
                collected,
                margin=margin,
                percentile=config.metrics.coverage_percentile,
            ).contamination_p95

        by_scene = contamination_by_scene(
            flat,
            single_max=config.metrics.contamination.single_max,
            fan_max=config.metrics.contamination.fan_max,
            percentile=config.metrics.contamination.percentile,
        )

        entry = report.to_dict()
        entry["coverage_p5_all_ci"] = bootstrap_images(
            indices, coverage_p5, samples=samples, seed=seed
        ).to_dict()
        entry["contamination_p95_ci"] = bootstrap_images(
            indices, contamination_p95, samples=samples, seed=seed
        ).to_dict()
        entry["contamination_by_scene"] = {
            name: value.to_dict() for name, value in by_scene.items()
        }
        entry["meets_target"] = report.meets(config.metrics.coverage_target)
        crop_by_margin[f"{margin:.2f}"] = entry

    chosen = crop_by_margin[f"{config.crop.margin:.2f}"]
    return {
        "split": split,
        "n_images": len(items),
        "n_truths": with_ignored.n_truths,
        "crop": {
            "margin_used": config.crop.margin,
            "coverage_target": config.metrics.coverage_target,
            "coverage_percentile": config.metrics.coverage_percentile,
            "by_margin": crop_by_margin,
        },
        "coverage_p5": chosen["coverage_p5_all_ci"],
        #: Aggregate over both scenes. Kept for continuity, but the verdict is
        #: the two entries below: mixing them hides both.
        "contamination_p95": chosen["contamination_p95_ci"],
        "contamination": {
            "by_scene": chosen["contamination_by_scene"],
            "passes": all(
                v["passes"] for v in chosen["contamination_by_scene"].values()
            ),
            "note": (
                "Two thresholds because the distribution is bimodal. In "
                "single-banknote images the floor is exactly zero and any "
                "contamination is a real error. In fans not even a perfect "
                "detector goes below 0.89: if a banknote is partially covered, "
                "its box necessarily contains pixels of the one covering it. "
                "To compare candidates on fans look at the median, not the "
                "p95, which discriminates little because it is nearly saturated."
            ),
        },
        "detection": {
            "match_iou": match_iou,
            #: Confidence the inference was run at. It is the one AP uses.
            "inference_confidence": config.detector.confidence_threshold,
            #: Confidence at which the counts below are read.
            "decision_confidence": decision,
            "at_decision_confidence": {
                "true_positives": decided_counts.true_positives,
                "false_positives": decided_counts.false_positives,
                "ignored": decided_counts.ignored,
                "detection_rate": decided_counts.detection_rate,
                "note": (
                    "What would be seen at deployment with this threshold. "
                    "This is the figure to read."
                ),
            },
            "with_ignored": {
                "true_positives": with_ignored.true_positives,
                "false_positives": with_ignored.false_positives,
                "ignored": with_ignored.ignored,
                "detection_rate": with_ignored.detection_rate,
            },
            "without_ignored": {
                "true_positives": without_ignored.true_positives,
                "false_positives": without_ignored.false_positives,
                "detection_rate": without_ignored.detection_rate,
            },
            "note": (
                "`with_ignored` and `without_ignored` are at the INFERENCE "
                "confidence, deliberately low so AP keeps the tail of the "
                "curve: there the false positive count measures how many boxes "
                "the NMS let through, not quality, and does not improve with "
                "more training. To read, use `at_decision_confidence`. "
                "The difference in false_positives between the two entries is "
                "the cost of our own annotation policy: correct detections on "
                "banknotes the area filter left out. A big jump says the "
                "annotation needs review, not the detector."
            ),
        },
        "map50": map50.to_dict(),
        "map50_95": map50_95_interval.to_dict(),
        "geometry": geometry_diagnostics(items, caches, match_iou=match_iou),
        "bootstrap": {
            "unit": config.metrics.bootstrap_unit,
            "samples": samples,
            "seed": seed,
        },
    }
