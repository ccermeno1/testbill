"""Metrics: rotated IoU, matching, coverage, contamination and intervals."""

from __future__ import annotations

import math

import pytest
from conftest import rotated_rect_points

from testbank.config import Config, MetricsConfig
from testbank.dataio.formats import ImageSize
from testbank.geometry.quad import Quad, canonicalize
from testbank.metrics.core import (
    ImageEval,
    Prediction,
    angle_error_deg,
    expand,
    iou,
    to_polygon,
)
from testbank.metrics.crop import crop_samples, summarize
from testbank.metrics.detection import (
    average_precision,
    bootstrap_images,
    counts,
    mean_average_precision,
)
from testbank.metrics.evaluate import evaluate
from testbank.metrics.matching import Outcome, match_image

SIZE = ImageSize(640, 480)


def quad(cx=0.5, cy=0.5, half_long=0.15, ratio=2.0, theta=0.0) -> Quad:
    """Rotated rectangle built in PIXELS, like the real banknotes."""
    points = rotated_rect_points(
        cx * SIZE.width, cy * SIZE.height, half_long * SIZE.width, ratio, theta
    )
    return canonicalize(
        Quad.from_xy([(x / SIZE.width, y / SIZE.height) for x, y in points]),
        aspect=SIZE.aspect,
    )


def image(truths=(), predictions=(), ignored=(), sample_id="img") -> ImageEval:
    return ImageEval(sample_id, SIZE, tuple(truths), tuple(predictions), tuple(ignored))


def perfect(truths, score=0.9):
    return [Prediction(q, score) for q in truths]


# --- angle ----------------------------------------------------------------


def test_179_degrees_against_1_degree_gives_2():
    """The textual example of the specification."""
    a = quad(theta=math.radians(1))
    b = quad(theta=math.radians(179))
    assert angle_error_deg(a, b, SIZE) == pytest.approx(2.0, abs=0.01)


def test_the_angle_error_never_exceeds_90():
    for degrees in range(0, 360, 7):
        error = angle_error_deg(quad(), quad(theta=math.radians(degrees)), SIZE)
        assert 0.0 <= error <= 90.0 + 1e-9


# --- rotated IoU ----------------------------------------------------------


def test_the_same_quad_gives_iou_1():
    polygon = to_polygon(quad(), SIZE)
    assert iou(polygon, polygon) == pytest.approx(1.0)


def test_separate_quads_give_iou_0():
    a = to_polygon(quad(cx=0.15, cy=0.2), SIZE)
    b = to_polygon(quad(cx=0.85, cy=0.8), SIZE)
    assert iou(a, b) == 0.0


def test_known_iou_of_two_shifted_squares():
    a = Quad.from_xy([(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)])
    b = Quad.from_xy([(0.1, 0.0), (0.3, 0.0), (0.3, 0.2), (0.1, 0.2)])
    # They overlap by half: intersection 1/2, union 3/2 -> IoU = 1/3.
    assert iou(to_polygon(a, SIZE), to_polygon(b, SIZE)) == pytest.approx(1 / 3)


# --- matching -------------------------------------------------------------


def test_a_truth_is_matched_at_most_once():
    truth = quad()
    item = image([truth], [Prediction(truth, 0.9), Prediction(truth, 0.8)])
    matching = match_image(item)
    assert len(matching.pairs) == 1
    results = [o for _, o, _ in matching.outcomes]
    assert results.count(Outcome.TRUE_POSITIVE) == 1
    assert results.count(Outcome.FALSE_POSITIVE) == 1


def test_the_most_confident_one_chooses_first():
    truth = quad()
    worse = Prediction(truth, 0.5)
    better = Prediction(quad(cx=0.51), 0.95)
    matching = match_image(image([truth], [worse, better]))
    assert matching.pairs[0].prediction_index == 1


def test_an_undetected_truth_counts_as_a_miss():
    matching = match_image(image([quad(), quad(cx=0.85, cy=0.8)], []))
    assert matching.missed == (0, 1)
    assert matching.detected == 0
    assert matching.n_truths == 2


# --- the ignored annotations ---------------------------------------------


def test_detecting_a_filtered_banknote_is_not_a_false_positive():
    """The area filter removed that banknote; finding it is a hit, not a failure."""
    truth = quad(cx=0.25)
    filtered = quad(cx=0.75)
    item = image([truth], perfect([truth, filtered]), ignored=[filtered])
    results = [o for _, o, _ in match_image(item).outcomes]
    assert results.count(Outcome.IGNORED) == 1
    assert results.count(Outcome.FALSE_POSITIVE) == 0


def test_without_the_ignore_that_same_detection_is_a_false_positive():
    """Reporting both figures is what measures the cost of our policy."""
    truth = quad(cx=0.25)
    filtered = quad(cx=0.75)
    item = image([truth], perfect([truth, filtered]), ignored=[filtered])
    results = [o for _, o, _ in match_image(item, use_ignored=False).outcomes]
    assert results.count(Outcome.FALSE_POSITIVE) == 1


def test_a_detection_in_the_background_is_still_a_false_positive():
    """Ignoring is not a free pass: it only covers the declared quads."""
    truth = quad(cx=0.2, cy=0.2)
    item = image([truth], perfect([truth, quad(cx=0.8, cy=0.8)]), ignored=[])
    results = [o for _, o, _ in match_image(item).outcomes]
    assert results.count(Outcome.FALSE_POSITIVE) == 1


# --- coverage and contamination ------------------------------------------


def test_a_perfect_prediction_covers_fully():
    truth = quad()
    sample = crop_samples(image([truth], perfect([truth])), margin=0.0)[0]
    assert sample.detected
    assert sample.coverage == pytest.approx(1.0, abs=1e-6)


def test_an_undetected_banknote_counts_as_zero_coverage():
    samples = crop_samples(image([quad()], []))
    assert samples[0].detected is False
    assert samples[0].coverage == 0.0


def test_finding_only_the_easy_ones_cannot_come_out_better():
    """Excluding the undetected would invert the conclusion. Here it is not excluded.

    The cautious one detects one banknote of two, perfectly. The complete one
    detects both, one of them somewhat worse. Aggregated, the complete one
    has to win.
    """
    easy, hard = quad(cx=0.25), quad(cx=0.75)
    cautious = image([easy, hard], perfect([easy]))
    complete = image(
        [easy, hard], [Prediction(easy, 0.9), Prediction(quad(cx=0.76), 0.7)]
    )
    cautious_summary = summarize(crop_samples(cautious), margin=0.05)
    complete_summary = summarize(crop_samples(complete), margin=0.05)

    assert cautious_summary.coverage_p5_detected >= complete_summary.coverage_p5_detected
    assert complete_summary.coverage_p5_all > cautious_summary.coverage_p5_all
    assert complete_summary.detection_rate > cautious_summary.detection_rate


def test_an_isolated_banknote_is_not_contaminated():
    truth = quad(cx=0.5, cy=0.5, half_long=0.1)
    sample = crop_samples(image([truth], perfect([truth])), margin=0.0)[0]
    assert sample.contamination == pytest.approx(0.0, abs=1e-9)


#: Neighbour on the right that does NOT overlap the banknote at 0.35, and a
#: slightly larger prediction that does reach it. It is the realistic case:
#: the box comes out too wide and eats the edge of the banknote next door.
_A = {"cx": 0.35, "cy": 0.5, "half_long": 0.12}
_NEIGHBOUR = {"cx": 0.61, "cy": 0.5, "half_long": 0.12}
_WIDE_PREDICTION = {"cx": 0.35, "cy": 0.5, "half_long": 0.155}


def test_the_neighbouring_banknote_contaminates_the_crop():
    a = quad(**_A)
    b = quad(**_NEIGHBOUR)
    wide = Prediction(quad(**_WIDE_PREDICTION), 0.9)
    samples = crop_samples(image([a, b], [wide]), margin=0.0)

    of_a = next(s for s in samples if s.truth_index == 0)
    assert of_a.detected, "the prediction has to match to be able to contaminate"
    assert of_a.contamination > 0.0


def test_a_filtered_banknote_does_contaminate():
    """For scoring detection it is ignored; for contaminating it is not: they are real pixels."""
    truth = quad(**_A)
    filtered = quad(**_NEIGHBOUR)
    wide = Prediction(quad(**_WIDE_PREDICTION), 0.9)

    with_ignored = crop_samples(
        image([truth], [wide], ignored=[filtered]),
        margin=0.0,
        contaminate_with_ignored=True,
    )[0]
    without = crop_samples(
        image([truth], [wide], ignored=[filtered]),
        margin=0.0,
        contaminate_with_ignored=False,
    )[0]
    assert with_ignored.detected and without.detected
    assert with_ignored.contamination > 0.0
    assert without.contamination == pytest.approx(0.0, abs=1e-9)


def test_the_margin_trades_coverage_for_contamination():
    """It is the reason the margin goes in the config and is swept."""
    a = quad(cx=0.40, cy=0.5, half_long=0.12)
    b = quad(cx=0.60, cy=0.5, half_long=0.12)
    item = image([a, b], perfect([a, b]))

    narrow = summarize(crop_samples(item, margin=0.0), margin=0.0)
    wide = summarize(crop_samples(item, margin=0.30), margin=0.30)
    assert wide.contamination_p95 > narrow.contamination_p95


def test_expanding_preserves_the_quadrilateral():
    """It is scaled, not dilated: `buffer` would round the corners."""
    polygon = to_polygon(quad(), SIZE)
    grown = expand(polygon, 0.1)
    assert len(grown.exterior.coords) == len(polygon.exterior.coords)
    assert grown.area > polygon.area


def test_negative_margin_is_an_error():
    with pytest.raises(ValueError, match="margin"):
        expand(to_polygon(quad(), SIZE), -0.1)


# --- AP -------------------------------------------------------------------


def test_perfect_prediction_gives_ap_1():
    items = [image([quad(cx=0.3)], perfect([quad(cx=0.3)]), sample_id="a")]
    assert average_precision(items) == pytest.approx(1.0, abs=1e-6)


def test_without_predictions_the_ap_is_zero():
    assert average_precision([image([quad()], [])]) == 0.0


def test_map50_95_does_not_exceed_map50():
    truth = quad()
    items = [image([truth], [Prediction(quad(cx=0.505), 0.9)])]
    assert mean_average_precision(items) <= average_precision(items) + 1e-9


def test_the_ranking_is_global_not_per_image():
    """An easy image must not weigh the same as a fan of six."""
    easy = image([quad()], perfect([quad()]), sample_id="easy")
    fan_truths = [quad(cx=0.2 + 0.12 * i, half_long=0.05) for i in range(6)]
    fan = image(fan_truths, perfect(fan_truths[:1]), sample_id="fan")
    counted = counts([easy, fan])
    assert counted.n_truths == 7
    assert counted.true_positives == 2
    assert counted.detection_rate == pytest.approx(2 / 7)


# --- intervals ------------------------------------------------------------


def test_the_bootstrap_is_deterministic():
    units = list(range(30))
    stat = lambda xs: sum(xs) / len(xs)
    a = bootstrap_images(units, stat, samples=200, seed=7)
    b = bootstrap_images(units, stat, samples=200, seed=7)
    assert (a.value, a.ci_low, a.ci_high) == (b.value, b.ci_low, b.ci_high)


def test_different_seeds_give_different_intervals():
    units = list(range(30))
    stat = lambda xs: sum(xs) / len(xs)
    a = bootstrap_images(units, stat, samples=200, seed=1)
    b = bootstrap_images(units, stat, samples=200, seed=2)
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


def test_with_a_single_image_the_interval_is_nan_not_zero():
    """A zero-width interval would be lying about the uncertainty."""
    result = bootstrap_images([1.0], lambda xs: xs[0], samples=100, seed=1)
    assert result.n == 1
    assert math.isnan(result.ci_low) and math.isnan(result.ci_high)


def test_the_interval_contains_the_point():
    units = [float(i) for i in range(40)]
    stat = lambda xs: sum(xs) / len(xs)
    result = bootstrap_images(units, stat, samples=500, seed=3)
    assert result.ci_low <= result.value <= result.ci_high


def test_the_bootstrap_unit_can_only_be_image():
    """Resampling detections narrows the intervals artificially."""
    with pytest.raises(ValueError, match="bootstrap"):
        MetricsConfig(bootstrap_unit="detection")


# --- the full report ------------------------------------------------------


def _fast_config() -> Config:
    base = Config()
    return base.model_copy(
        update={"metrics": base.metrics.model_copy(update={"bootstrap_samples": 50})}
    )


def test_the_report_has_what_decides_and_what_diagnoses():
    truths = [quad(cx=0.3), quad(cx=0.7)]
    items = [
        image(truths, perfect(truths), sample_id=f"img{i}") for i in range(4)
    ]
    report = evaluate(items, _fast_config())

    assert report["n_images"] == 4 and report["n_truths"] == 8
    assert report["map50"]["value"] == pytest.approx(1.0, abs=1e-6)
    assert report["coverage_p5"]["n"] == 4
    assert "0.05" in report["crop"]["by_margin"]
    assert report["crop"]["by_margin"]["0.05"]["meets_target"] is True
    assert report["geometry"]["n_matched_pairs"] == 8
    assert report["bootstrap"]["unit"] == "image"


def test_the_report_separates_false_positives_with_and_without_ignore():
    truth = quad(cx=0.25)
    filtered = quad(cx=0.75)
    items = [
        image([truth], perfect([truth, filtered]), ignored=[filtered], sample_id=f"i{i}")
        for i in range(3)
    ]
    report = evaluate(items, _fast_config())
    detection = report["detection"]
    assert detection["with_ignored"]["false_positives"] == 0
    assert detection["with_ignored"]["ignored"] == 3
    assert detection["without_ignored"]["false_positives"] == 3


# --- the precompute cannot change the number ------------------------------


def test_the_precomputed_ap_matches_the_direct_one():
    """The bootstrap resamples over `image_stats`, it does not match again.

    It is valid because matching is LOCAL to each image: whether other images
    enter the replicate or not does not change this one's result. If that
    stopped being true, this test catches it.
    """
    from testbank.metrics.detection import ap_from_stats, image_stats

    items = [
        image(
            [quad(cx=0.3), quad(cx=0.7)],
            [Prediction(quad(cx=0.31), 0.9), Prediction(quad(cx=0.85, cy=0.8), 0.4)],
            sample_id=f"img{i}",
        )
        for i in range(5)
    ]
    stats = image_stats(items)

    # Any subset, like the one a bootstrap replicate would produce.
    picked = [0, 2, 2, 4]
    direct = average_precision([items[i] for i in picked])
    precomputed = ap_from_stats([stats[i] for i in picked])
    assert precomputed == pytest.approx(direct)


def test_the_order_of_the_images_does_not_change_the_ap():
    """The ranking is global by confidence, so shuffling images does not matter."""
    from testbank.metrics.detection import ap_from_stats, image_stats

    items = [
        image([quad(cx=0.3)], [Prediction(quad(cx=0.3 + 0.01 * i), 0.5 + 0.08 * i)],
              sample_id=f"img{i}")
        for i in range(5)
    ]
    stats = image_stats(items)
    assert ap_from_stats(stats) == pytest.approx(ap_from_stats(list(reversed(stats))))


# --- the two contamination thresholds ------------------------------------


def test_a_single_banknote_image_is_a_single_scene():
    from testbank.metrics.crop import SceneType, scene_type

    assert scene_type(image([quad()], [])) is SceneType.SINGLE


def test_a_filtered_neighbour_already_turns_the_scene_into_a_fan():
    """It is still in the pixels and still dirties, even if it is not live truth."""
    from testbank.metrics.crop import SceneType, scene_type

    item = image([quad(cx=0.3)], [], ignored=[quad(cx=0.7)])
    assert scene_type(item) is SceneType.FAN


def test_each_scene_is_measured_against_its_threshold():
    from testbank.metrics.crop import contamination_by_scene

    clean = image([quad(cx=0.3)], perfect([quad(cx=0.3)]), sample_id="alone")
    a, b = quad(**_A), quad(**_NEIGHBOUR)
    dirty = image([a, b], [Prediction(quad(**_WIDE_PREDICTION), 0.9)], sample_id="fan")

    samples = crop_samples(clean, margin=0.05) + crop_samples(dirty, margin=0.05)
    by_scene = contamination_by_scene(samples, single_max=0.01, fan_max=0.92)

    assert by_scene["single"].threshold == 0.01
    assert by_scene["fan"].threshold == 0.92
    assert by_scene["single"].p95 == pytest.approx(0.0, abs=1e-9)
    assert by_scene["single"].passes


def test_without_samples_of_a_scene_it_does_not_fail():
    """Without evidence there is no verdict; failing by default would be inventing."""
    from testbank.metrics.crop import contamination_by_scene

    samples = crop_samples(image([quad()], perfect([quad()])), margin=0.0)
    by_scene = contamination_by_scene(samples, single_max=0.01, fan_max=0.92)
    assert by_scene["fan"].n == 0
    assert by_scene["fan"].passes is True


def test_the_fan_threshold_cannot_be_stricter_than_the_single_banknote_one():
    from testbank.config import ContaminationConfig

    with pytest.raises(ValueError, match="fan_max"):
        ContaminationConfig(single_max=0.5, fan_max=0.1)


def test_the_floor_can_be_re_derived_from_the_data():
    """The defaults came from here; it has to be repeated if the export changes."""
    from testbank.metrics.crop import contamination_floor

    a, b = quad(**_A), quad(**_NEIGHBOUR)
    items = [
        image([quad(cx=0.5)], [], sample_id="alone"),
        image([a, b], [], sample_id="fan"),
    ]
    floor = contamination_floor(items, margin=0.05)
    assert floor["single"] == pytest.approx(0.0, abs=1e-9)
    assert floor["fan"] >= 0.0


def test_the_report_carries_both_scenes_with_their_verdict():
    truths = [quad(cx=0.3), quad(cx=0.7)]
    items = [image(truths, perfect(truths), sample_id=f"i{i}") for i in range(3)]
    report = evaluate(items, _fast_config())
    by_scene = report["contamination"]["by_scene"]
    assert set(by_scene) == {"single", "fan"}
    assert by_scene["fan"]["threshold"] == 0.92
    assert by_scene["single"]["threshold"] == 0.01
    assert "passes" in report["contamination"]


# --- the two confidence thresholds ----------------------------------------


def test_counts_are_given_at_both_confidences():
    """At 0.01 the count measures how many boxes the NMS let through, not quality.

    The low-confidence tail is indispensable for AP and ruinous as a
    headline, so both figures are reported, labelled.
    """
    truth = quad(cx=0.3)
    items = [
        image(
            [truth],
            [
                Prediction(truth, 0.9),                    # hit
                Prediction(quad(cx=0.8, cy=0.8), 0.02),    # noise tail
                Prediction(quad(cx=0.8, cy=0.2), 0.03),    # noise tail
            ],
            sample_id=f"i{i}",
        )
        for i in range(3)
    ]
    report = evaluate(items, _fast_config())
    detection = report["detection"]

    assert detection["decision_confidence"] == 0.25
    assert detection["with_ignored"]["false_positives"] == 6
    assert detection["at_decision_confidence"]["false_positives"] == 0
    assert detection["at_decision_confidence"]["true_positives"] == 3
    # AP is computed with the whole tail and is not affected by the report.
    assert report["map50"]["value"] == pytest.approx(1.0, abs=1e-6)


def test_the_false_positives_of_the_tail_do_not_sink_the_ap():
    """Regression: with `np.interp` and repeated recall, the AP came out sunk.

    Three perfect hits reach recall 1.0 with precision 1.0. The later false
    positives cannot lower the AP: the precision at a given recall is the BEST
    achievable at that recall or beyond, and that maximum was already reached.
    """
    truth = quad(cx=0.3)
    clean = [image([truth], [Prediction(truth, 0.9)], sample_id=f"i{i}") for i in range(3)]
    with_tail = [
        image(
            [truth],
            [Prediction(truth, 0.9)] + [
                Prediction(quad(cx=0.8, cy=0.2 + 0.15 * k), 0.02) for k in range(4)
            ],
            sample_id=f"i{i}",
        )
        for i in range(3)
    ]
    assert average_precision(clean) == pytest.approx(1.0, abs=1e-9)
    assert average_precision(with_tail) == pytest.approx(1.0, abs=1e-9)
