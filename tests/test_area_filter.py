"""Relative area filter: keeps the front banknote of each image."""

from __future__ import annotations

import hashlib

import pytest
from conftest import rotated_rect_points

from testbank.checks.visibility import check_samples
from testbank.data.discover import Sample
from testbank.dataio.obb_yolo import Annotation
from testbank.dataio.prepare import (
    DEFAULT_MIN_RELATIVE_AREA,
    filter_by_relative_area,
    load_sample,
    load_samples,
)
from testbank.geometry.quad import Quad, canonicalize


def quad(cx, cy, half_long=0.2, ratio=2.0, theta=0.0) -> Quad:
    return canonicalize(Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta)))


def ann(*args, **kwargs) -> Annotation:
    return Annotation(class_id=0, quad=quad(*args, **kwargs))


def write_sample(tmp_path, quads, sample_id="img") -> Sample:
    """Writes an obb_yolo .txt and a paired dummy image."""
    label = tmp_path / f"{sample_id}.txt"
    lines = [
        "0 " + " ".join(f"{v:.17g}" for v in q.flat()) for q in quads
    ]
    label.write_text("\n".join(lines) + "\n", encoding="utf-8")
    image = tmp_path / f"{sample_id}.jpg"
    image.write_bytes(b"")
    return Sample(sample_id=sample_id, image_path=image, label_path=label)


# --- the criterion --------------------------------------------------------


def test_a_single_annotation_is_always_kept():
    kept, dropped = filter_by_relative_area([ann(0.5, 0.5)])
    assert len(kept) == 1 and dropped == ()


def test_no_annotations_does_not_fail():
    assert filter_by_relative_area([]) == ((), ())


def test_banknotes_of_similar_size_are_all_kept():
    """Photo of two or three banknotes together: none is a covered strip."""
    anns = [ann(0.25, 0.5, half_long=0.20), ann(0.75, 0.5, half_long=0.19)]
    kept, dropped = filter_by_relative_area(anns)
    assert len(kept) == 2 and dropped == ()


def test_strip_of_covered_banknote_drops():
    """Fan: the visible strip of the one behind is much smaller than the front one."""
    front = ann(0.5, 0.5, half_long=0.30, ratio=2.0)
    strip = ann(0.5, 0.5, half_long=0.30, ratio=20.0)  # same length, very thin
    kept, dropped = filter_by_relative_area([front, strip])
    assert len(kept) == 1
    assert [index for index, _ in dropped] == [1]
    assert dropped[0][1] == pytest.approx(0.1, abs=0.01)


def test_the_largest_is_never_discarded_even_with_threshold_one():
    anns = [ann(0.3, 0.3, half_long=0.10), ann(0.7, 0.7, half_long=0.30)]
    kept, dropped = filter_by_relative_area(anns, min_relative_area=1.0)
    assert len(kept) == 1
    assert kept[0][0] == 1  # the largest, with its original index
    assert [index for index, _ in dropped] == [0]


def test_threshold_zero_discards_nothing():
    anns = [ann(0.5, 0.5, half_long=0.30), ann(0.5, 0.5, half_long=0.30, ratio=40.0)]
    kept, dropped = filter_by_relative_area(anns, min_relative_area=0.0)
    assert len(kept) == 2 and dropped == ()


def test_threshold_out_of_range_is_an_error():
    with pytest.raises(ValueError, match="min_relative_area"):
        filter_by_relative_area([ann(0.5, 0.5)], min_relative_area=1.5)


def test_the_default_threshold_is_the_policy_one():
    assert DEFAULT_MIN_RELATIVE_AREA == 0.25


def test_the_threshold_separates_above_and_below():
    """On each side of the threshold the decision is the expected one.

    EXACT equality with the threshold is not checked, and not for
    convenience: vertices snap to the dyadic grid of `quad.py`, so an area
    built to give exactly 0.25 gives 0.2499999998. As with the flip
    involution, the exact edge is not representable. Nor does it matter: an
    area ratio within 1e-9 of the threshold is arbitrary in any case, because
    the source annotation is an "approximate rectangle".
    """
    large = ann(0.5, 0.5, half_long=0.20, ratio=2.0)
    # The area is proportional to half_long^2 / ratio.
    above = ann(0.5, 0.5, half_long=0.11, ratio=2.0)  # ~0.30 relative
    kept, dropped = filter_by_relative_area([large, above], min_relative_area=0.25)
    assert len(kept) == 2 and dropped == ()

    below = ann(0.5, 0.5, half_long=0.09, ratio=2.0)  # ~0.20 relative
    kept, dropped = filter_by_relative_area([large, below], min_relative_area=0.25)
    assert [index for index, _ in dropped] == [1]


# --- indices point to the file, not to the filtered list ------------------


def test_kept_indices_are_the_file_ones(tmp_path):
    """If annotation 0 drops, the next one is still #1, not #0.

    A report that renumbered would send you to open the wrong annotation in
    Roboflow, which is exactly what the report is for.
    """
    sample = write_sample(
        tmp_path,
        [
            quad(0.5, 0.5, half_long=0.30, ratio=30.0),  # strip, drops
            quad(0.25, 0.5, half_long=0.20),
            quad(0.75, 0.5, half_long=0.20),
        ],
    )
    loaded = load_sample(sample)
    assert loaded.kept_indices == (1, 2)
    assert [d.index for d in loaded.dropped] == [0]


def test_the_report_carries_image_index_and_relative_area(tmp_path):
    sample = write_sample(
        tmp_path,
        [
            quad(0.5, 0.5, half_long=0.30, ratio=2.0),
            quad(0.5, 0.5, half_long=0.30, ratio=20.0),
        ],
        sample_id="fan",
    )
    loaded = load_sample(sample)
    assert len(loaded.dropped) == 1
    dropped = loaded.dropped[0]
    assert dropped.sample_id == "fan"
    assert dropped.index == 1
    assert 0.0 < dropped.relative_area < 0.25
    assert "fan" in dropped.describe() and "#1" in dropped.describe()


# --- filter, not deletion -------------------------------------------------


def test_the_source_file_is_not_touched(tmp_path):
    sample = write_sample(
        tmp_path,
        [
            quad(0.5, 0.5, half_long=0.30, ratio=2.0),
            quad(0.5, 0.5, half_long=0.30, ratio=20.0),
        ],
    )
    before = hashlib.sha256(sample.label_path.read_bytes()).hexdigest()
    load_sample(sample, min_relative_area=0.9)
    after = hashlib.sha256(sample.label_path.read_bytes()).hexdigest()
    assert before == after


def test_raising_the_threshold_does_not_require_re_exporting(tmp_path):
    """The same file gives different sets depending on the parameter."""
    sample = write_sample(
        tmp_path,
        [
            quad(0.25, 0.5, half_long=0.20),
            quad(0.75, 0.5, half_long=0.12),
        ],
    )
    assert len(load_sample(sample, min_relative_area=0.0).annotations) == 2
    assert len(load_sample(sample, min_relative_area=0.9).annotations) == 1


# --- aggregate ------------------------------------------------------------


def test_the_aggregate_report_adds_up(tmp_path):
    a = write_sample(
        tmp_path,
        [quad(0.5, 0.5, half_long=0.30), quad(0.5, 0.5, half_long=0.30, ratio=25.0)],
        sample_id="a",
    )
    b = write_sample(
        tmp_path, [quad(0.5, 0.5, half_long=0.20)], sample_id="b"
    )
    loaded, report = load_samples([a, b])
    assert report.images_loaded == 2
    assert report.annotations_read == 3
    assert report.annotations_kept == 2
    assert report.images_affected == 1
    assert len(loaded) == 2

    payload = report.to_dict()
    assert payload["annotations_dropped"] == 1
    assert payload["dropped"][0]["sample_id"] == "a"
    assert payload["dropped"][0]["index"] == 1
    assert "nothing was deleted" in payload["note"]


# --- integration with the visibility check --------------------------------


def test_check_samples_returns_both_reports(tmp_path):
    sample = write_sample(
        tmp_path,
        [quad(0.5, 0.5, half_long=0.30), quad(0.5, 0.5, half_long=0.30, ratio=25.0)],
    )
    report, filter_report = check_samples([sample])
    assert filter_report.annotations_read == 2
    assert filter_report.annotations_kept == 1
    # With a single live annotation there is nothing left that can be covered.
    assert report.annotations_checked == 1
    assert report.findings == []


def test_visibility_findings_number_over_the_file(tmp_path):
    """#0 is filtered by area; the two remaining are still #1 and #2.

    Without translating indices, the finding would say #0 and send you to
    review precisely the annotation the filter had already discarded.
    """
    sample = write_sample(
        tmp_path,
        [
            quad(0.5, 0.1, half_long=0.30, ratio=25.0),  # loose strip, drops
            # Relative area 0.44: survives the filter, but lies inside the
            # next one, so the visibility check does flag it.
            quad(0.5, 0.6, half_long=0.20, ratio=2.0),
            quad(0.5, 0.6, half_long=0.30, ratio=2.0),
        ],
    )
    report, filter_report = check_samples([sample], min_relative_area=0.25)
    assert [d.index for d in filter_report.dropped] == [0]
    assert [f.annotation_index for f in report.findings] == [1]
    assert report.findings[0].dominant_index == 2
