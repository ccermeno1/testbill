"""Reader for the canonical format: `class x1 y1 x2 y2 x3 y3 x4 y4`, normalized.

It is the format of the Roboflow export and the pivot of all other converters.
The original annotations are read-only: nothing is ever written here.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from testbank.geometry.quad import (
    DEFAULT_ASPECT,
    CoordinateRangeWarning,
    Quad,
    QuadError,
    QuadShapeWarning,
    canonicalize,
)

TOKENS_PER_LINE = 9

# A float64 cannot serialize more than 17 significant digits. A token with more
# does not come from an honest exporter: the file went through an extractor
# that corrupted it and is not a valid annotation.
MAX_SIGNIFICANT_DIGITS = 17

_NUMBER = re.compile(
    r"^[+-]?(?:(?P<int>\d+)(?:\.(?P<frac>\d*))?|\.(?P<frac2>\d+))"
    r"(?:[eE](?P<exp>[+-]?\d+))?$"
)


class LabelFormatError(ValueError):
    """The label file does not follow the format. Always identifies the line.

    Still fatal, but it accumulates: a file with three bad lines reports all
    three at once. With 500 files, failing on the first forces a full
    fix-and-re-export cycle per error, and hides up front whether the problem
    is a stray click or the entire export.
    """

    def __init__(self, message: str, problems: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        #: Each problem on its own, already formatted with its location.
        self.problems = problems or (message,)

    @classmethod
    def combine(cls, problems: list[str], header: str) -> LabelFormatError:
        """A single error carrying every problem.

        With a single one the message is the usual one, to avoid decorating the
        common case or breaking anyone searching for a specific text.
        """
        if len(problems) == 1:
            return cls(problems[0], (problems[0],))
        listed = "\n".join(f"  - {p}" for p in problems)
        return cls(f"{header} ({len(problems)}):\n{listed}", tuple(problems))


@dataclass(frozen=True, slots=True)
class Annotation:
    class_id: int
    quad: Quad


@dataclass(frozen=True, slots=True)
class LabelFile:
    path: Path
    annotations: tuple[Annotation, ...]
    #: Non-fatal warnings, already formatted with their location.
    warnings: tuple[str, ...] = ()


def significant_digits(token: str) -> int:
    """Significant digits of the mantissa.

    Sign, point, exponent, leading zeros and trailing zeros are ignored: none
    of them carries information a float64 could not produce, and what we want
    to detect is fabricated precision.
    """
    match = _NUMBER.match(token)
    if match is None:
        raise ValueError(f"non-numeric token: {token!r}")
    digits = (match.group("int") or "") + (
        match.group("frac") or match.group("frac2") or ""
    )
    return len(digits.strip("0"))


def _parse_line(
    line: str, path: Path, lineno: int, aspect: float = DEFAULT_ASPECT
) -> Annotation:
    where = f"{path}:{lineno}"
    tokens = line.split()
    if len(tokens) != TOKENS_PER_LINE:
        raise LabelFormatError(
            f"{where}: an obb_yolo line has {TOKENS_PER_LINE} tokens "
            f"(class + 4 vertices), found {len(tokens)}: {line.strip()!r}"
        )

    for index, token in enumerate(tokens):
        try:
            count = significant_digits(token)
        except ValueError as exc:
            raise LabelFormatError(f"{where}: token {index}: {exc}") from exc
        if count > MAX_SIGNIFICANT_DIGITS:
            raise LabelFormatError(
                f"{where}: token {index} has {count} significant digits "
                f"({token!r}); a float64 cannot serialize more than "
                f"{MAX_SIGNIFICANT_DIGITS}, so the file went through an "
                f"extractor that corrupted it and is not a valid annotation"
            )

    class_token = tokens[0]
    class_value = float(class_token)
    if class_value != int(class_value) or class_value < 0:
        raise LabelFormatError(
            f"{where}: the class id must be a non-negative integer, "
            f"found {class_token!r}"
        )

    try:
        quad = Quad.from_xy([float(t) for t in tokens[1:]])
    except QuadError as exc:
        raise LabelFormatError(f"{where}: {exc}") from exc

    return Annotation(
        class_id=int(class_value), quad=canonicalize(quad, aspect=aspect)
    )


def read_label_file(
    path: str | Path, *, aspect: float = DEFAULT_ASPECT
) -> LabelFile:
    """Read an obb_yolo .txt. An empty file is an image with no banknotes, valid.

    `aspect` is width/height in pixels of the corresponding image. Without it
    the canonical order is computed in normalized space, where the longest side
    may not be the real longest side. Pass it whenever you have it.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise LabelFormatError(f"{path}: label file does not exist") from exc

    annotations: list[Annotation] = []
    messages: list[str] = []
    problems: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                parsed = _parse_line(line, path, lineno, aspect)
            except LabelFormatError as exc:
                # Keep reading to report EVERY bad line of the file at once;
                # the error is raised once the file is done.
                problems.append(str(exc))
                continue
            annotations.append(parsed)
        for entry in caught:
            if issubclass(entry.category, (CoordinateRangeWarning, QuadShapeWarning)):
                messages.append(f"{path}:{lineno}: {entry.message}")

    if problems:
        raise LabelFormatError.combine(problems, f"{path}: invalid lines")

    return LabelFile(
        path=path, annotations=tuple(annotations), warnings=tuple(messages)
    )
