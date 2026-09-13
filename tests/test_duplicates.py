"""Near-duplicates: grouping, report and manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from testbank.data.discover import Sample
from testbank.data.duplicates import (
    DEFAULT_THRESHOLD,
    find_duplicates,
    signature,
    write_manifest,
)


def _sample(directory: Path, sample_id: str, array: np.ndarray) -> Sample:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sample_id}.png"  # PNG: lossless, controlled noise
    Image.fromarray(array.astype(np.uint8)).save(path)
    return Sample(
        sample_id=sample_id,
        image_path=path,
        label_path=directory / f"{sample_id}.txt",
    )


def _noise(seed: int, size: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (size, size, 3))


def _jitter(array: np.ndarray, amount: int, seed: int) -> np.ndarray:
    """Same image with a small perturbation, like two consecutive shots."""
    delta = np.random.default_rng(seed).integers(-amount, amount + 1, array.shape)
    return np.clip(array.astype(int) + delta, 0, 255)


# --- the signature --------------------------------------------------------


def test_the_signature_is_normalized(tmp_path):
    s = _sample(tmp_path, "a", _noise(1))
    vector = signature(s.image_path)
    assert vector.size == 16 * 16 * 3  # color: three channels
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert vector.mean() == pytest.approx(0.0, abs=1e-9)


def test_a_single_tone_image_does_not_blow_up(tmp_path):
    """Zero norm: dividing would give NaN and contaminate the whole matrix."""
    s = _sample(tmp_path, "flat", np.full((64, 64, 3), 128))
    vector = signature(s.image_path)
    assert np.all(np.isfinite(vector))
    assert np.linalg.norm(vector) == pytest.approx(0.0)


# --- the grouping ---------------------------------------------------------


def test_two_nearly_equal_shots_go_to_the_same_group(tmp_path):
    base = _noise(1)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=2))
    report = find_duplicates([a, b])
    assert len(report.pairs) == 1
    assert report.groups["a"] == report.groups["b"]
    assert report.n_groups == 1


def test_different_images_are_not_grouped(tmp_path):
    samples = [_sample(tmp_path, f"s{i}", _noise(i)) for i in range(4)]
    report = find_duplicates(samples)
    assert report.pairs == []
    assert report.n_groups == 4


def test_the_grouping_is_transitive(tmp_path):
    """A~B and B~C put the three together even if A and C do not resemble each other.

    Splitting them would leave the leak half done, which is almost the same as doing nothing.
    """
    base = _noise(7)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 30, seed=1))
    c = _sample(tmp_path, "c", _jitter(base, 60, seed=2))
    report = find_duplicates([a, b, c], threshold=0.80)
    assert report.n_groups == 1
    assert len({report.groups[i] for i in ("a", "b", "c")}) == 1


def test_the_group_key_does_not_depend_on_order(tmp_path):
    """The manifest has to come out the same however they are passed."""
    base = _noise(3)
    a = _sample(tmp_path, "aaa", base)
    b = _sample(tmp_path, "bbb", _jitter(base, 4, seed=9))
    forward = find_duplicates([a, b]).groups
    reverse = find_duplicates([b, a]).groups
    assert forward == reverse
    assert set(forward.values()) == {"aaa"}


def test_a_higher_threshold_groups_less(tmp_path):
    base = _noise(5)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 90, seed=4))
    assert find_duplicates([a, b], threshold=0.70).n_groups == 1
    assert find_duplicates([a, b], threshold=0.999).n_groups == 2


# --- the report -----------------------------------------------------------


def test_pairs_crossing_splits_are_flagged(tmp_path):
    base = _noise(11)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=1))
    c = _sample(tmp_path, "c", _noise(12))

    report = find_duplicates(
        [a, b, c], split_of={"a": "train", "b": "test", "c": "train"}
    )
    assert len(report.crossing) == 1
    assert report.crossing[0].crosses_splits
    assert report.affected == {"a", "b"}
    assert "train <-> test" in report.crossing[0].describe()


def test_a_pair_within_the_same_split_does_not_cross(tmp_path):
    base = _noise(13)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=1))
    report = find_duplicates([a, b], split_of={"a": "train", "b": "train"})
    assert len(report.pairs) == 1
    assert report.crossing == []


def test_no_samples_does_not_fail():
    report = find_duplicates([])
    assert report.n_samples == 0 and report.pairs == []


def test_the_report_says_it_does_not_re_split(tmp_path):
    """Whoever reads the JSON has to know this has NOT touched the split."""
    note = find_duplicates([_sample(tmp_path, "a", _noise(1))]).to_dict()["note"]
    assert "Does not re-split" in note
    assert "the export decides" in note
    assert "--repartition" in note


# --- the manifest ---------------------------------------------------------


def test_the_manifest_has_the_group_manifest_format(tmp_path):
    base = _noise(21)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=1))
    c = _sample(tmp_path, "c", _noise(22))

    path = write_manifest(find_duplicates([a, b, c]), tmp_path / "out" / "g.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"a", "b", "c"}
    assert data["a"] == data["b"] != data["c"]
    assert all(isinstance(v, str) for v in data.values())


def test_the_default_threshold_is_the_measured_one():
    assert DEFAULT_THRESHOLD == 0.90


def test_same_framing_with_different_color_is_not_a_duplicate(tmp_path):
    """REGRESSION: in grayscale, two stock photos with the same framing and
    banknotes of different value (an orange 50, a purple 500) came out as the
    same shot. 18 of 93 real pairs were that. Here: same drawing, different tint."""
    base = _noise(51).astype(float)
    orange = np.clip(base * [1.0, 0.6, 0.2], 0, 255)
    purple = np.clip(base * [0.6, 0.2, 1.0], 0, 255)
    a = _sample(tmp_path, "a", orange)
    b = _sample(tmp_path, "b", purple)
    assert find_duplicates([a, b]).pairs == []


def test_the_per_split_breakdown_highlights_test(tmp_path):
    """The number that matters most is how much test is contaminated.

    Regression: without this breakdown it had to be counted by hand over the
    truncated list of pairs, and that is how a 4 slipped in where there were
    21 in a report.
    """
    base = _noise(31)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=1))
    c = _sample(tmp_path, "c", _noise(32))

    report = find_duplicates(
        [a, b, c], split_of={"a": "train", "b": "test", "c": "train"}
    )
    assert len(report.touching("test")) == 1
    assert report.affected_in("test") == {"b"}
    assert report.affected_in("train") == {"a"}
    assert report.touching("valid") == []

    lines = "\n".join(report.summary_lines({"train": 2, "test": 1}))
    assert "test" in lines and "SEALED" in lines
    assert lines.index("test") < lines.index("train"), "test goes first"


def test_the_breakdown_also_goes_to_the_json(tmp_path):
    base = _noise(41)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=1))
    payload = find_duplicates(
        [a, b], split_of={"a": "train", "b": "test"}
    ).to_dict()
    assert payload["by_split"]["test"]["images"] == 1
    assert payload["by_split"]["valid"]["pairs"] == 0
