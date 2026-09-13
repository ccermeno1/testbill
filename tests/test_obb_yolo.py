from __future__ import annotations

import pytest

from testbank.dataio.obb_yolo import (
    LabelFormatError,
    read_label_file,
    significant_digits,
)

VALID = "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"


def write(tmp_path, text, name="img_0001.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# -- significant digits -----------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("0", 0),
        ("0.5", 1),
        ("0.000123", 3),
        ("123.456", 6),
        ("-0.10", 1),
        ("1e-3", 1),
        ("0.12345678901234567", 17),
        ("0.123456789012345678", 18),
    ],
)
def test_significant_digit_count(token, expected):
    assert significant_digits(token) == expected


# -- reading ----------------------------------------------------------------


def test_reads_a_valid_annotation(tmp_path):
    label = read_label_file(write(tmp_path, VALID))
    assert len(label.annotations) == 1
    assert label.annotations[0].class_id == 0
    assert label.annotations[0].quad.is_clockwise()


def test_empty_file_is_an_image_without_banknotes(tmp_path):
    assert read_label_file(write(tmp_path, "")).annotations == ()


def test_ignores_blank_lines(tmp_path):
    assert len(read_label_file(write(tmp_path, VALID + "\n\n" + VALID)).annotations) == 2


def test_the_output_already_comes_canonical(tmp_path):
    """The reader canonicalizes: nobody downstream has to remember."""
    shuffled = "0 0.5 0.3 0.1 0.3 0.1 0.1 0.5 0.1\n"
    a = read_label_file(write(tmp_path, VALID, "a.txt")).annotations[0]
    b = read_label_file(write(tmp_path, shuffled, "b.txt")).annotations[0]
    assert a.quad.points == b.quad.points


# -- explicit errors --------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1\n",
        "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3 0.9\n",
        "0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n",
    ],
)
def test_wrong_number_of_tokens_is_an_explicit_error(tmp_path, line):
    path = write(tmp_path, line)
    with pytest.raises(LabelFormatError, match="9 tokens") as exc:
        read_label_file(path)
    assert f"{path}:1" in str(exc.value)


def test_fabricated_precision_is_an_error(tmp_path):
    """18 digits: the file went through an extractor that corrupted it."""
    line = "0 0.100000000000000001 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"
    with pytest.raises(LabelFormatError, match="significant digits") as exc:
        read_label_file(write(tmp_path, line))
    assert "token 1" in str(exc.value)


def test_seventeen_digits_are_accepted(tmp_path):
    line = "0 0.10000000000000001 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"
    assert len(read_label_file(write(tmp_path, line)).annotations) == 1


def test_non_numeric_token_is_an_error(tmp_path):
    line = "0 nan 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"
    with pytest.raises(LabelFormatError, match="non-numeric"):
        read_label_file(write(tmp_path, line))


def test_non_integer_class_is_an_error(tmp_path):
    line = "0.5 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"
    with pytest.raises(LabelFormatError, match="non-negative integer"):
        read_label_file(write(tmp_path, line))


def test_coordinate_out_of_range_is_an_error(tmp_path):
    line = "0 0.1 0.1 1.9 0.1 1.9 0.3 0.1 0.3\n"
    with pytest.raises(LabelFormatError, match="outside the tolerant range"):
        read_label_file(write(tmp_path, line))


def test_the_error_identifies_the_line(tmp_path):
    path = write(tmp_path, VALID + VALID + "0 1 2 3\n")
    with pytest.raises(LabelFormatError) as exc:
        read_label_file(path)
    assert f"{path}:3" in str(exc.value)


# -- non-fatal warnings -----------------------------------------------------


def test_vertex_outside_0_1_is_a_warning_not_an_error(tmp_path):
    """Banknote crossing the image border."""
    line = "0 -0.2 0.1 0.5 0.1 0.5 0.3 -0.2 0.3\n"
    label = read_label_file(write(tmp_path, line))
    assert len(label.annotations) == 1
    assert any("outside [0,1]" in w for w in label.warnings)


# -- errors accumulate, it does not stop at the first ----------------------


def test_a_file_reports_all_its_bad_lines(tmp_path):
    """Three bad lines are seen at once, not one per run."""
    text = "0 1 2 3\n" + VALID + "0 nan 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n" + "0.5 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"
    path = write(tmp_path, text)
    with pytest.raises(LabelFormatError) as exc:
        read_label_file(path)
    assert len(exc.value.problems) == 3
    assert f"{path}:1" in str(exc.value)
    assert f"{path}:3" in str(exc.value)
    assert f"{path}:4" in str(exc.value)
    # Line 2 is valid and does not appear.
    assert f"{path}:2" not in str(exc.value)


def test_with_a_single_bad_line_the_message_does_not_change(tmp_path):
    """Do not wrap the common case: it is still the usual error."""
    path = write(tmp_path, "0 1 2 3\n")
    with pytest.raises(LabelFormatError) as exc:
        read_label_file(path)
    assert exc.value.problems == (str(exc.value),)
    assert "invalid lines" not in str(exc.value)


def test_errors_accumulate_across_files(tmp_path):
    """With 500 labels, one error per run turns the cleanup into a loop."""
    from testbank.data.discover import Sample
    from testbank.dataio.prepare import load_samples

    samples = []
    for name, text in (("a", "0 1 2 3\n"), ("b", VALID), ("c", "0 nan 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n")):
        label = write(tmp_path, text, f"{name}.txt")
        samples.append(
            Sample(sample_id=name, image_path=tmp_path / f"{name}.jpg", label_path=label)
        )

    with pytest.raises(LabelFormatError) as exc:
        load_samples(samples)
    assert len(exc.value.problems) == 2
    assert "a.txt" in str(exc.value) and "c.txt" in str(exc.value)
    assert "b.txt" not in str(exc.value)
