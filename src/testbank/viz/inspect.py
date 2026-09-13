"""Inspection visualization: quads drawn with the canonical order visible.

What must be checkable at a glance:
  - that vertex 0 is the one opening the longest side
  - that the path 0->1->2->3 is clockwise
  - that overlapping banknotes carry separate boxes and not a single giant box
  - that banknotes crossing the border have vertices outside the image

The sample is fixed and comes from the validation split, always the same, so
runs can be compared by eye.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from testbank.checks.visibility import Finding, check_quads
from testbank.data.discover import Sample
from testbank.dataio.prepare import (
    DEFAULT_MIN_RELATIVE_AREA,
    DroppedAnnotation,
    load_sample,
)
from testbank.geometry.quad import Quad

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_EDGE = (90, 210, 90)
COLOR_ANCHOR = (235, 90, 235)
COLOR_FLAGGED = (70, 70, 245)
COLOR_VERTEX_TEXT = (255, 255, 255)
COLOR_PANEL = (34, 32, 30)
COLOR_PANEL_TEXT = (225, 225, 225)
COLOR_FRAME = (120, 170, 250)
#: Dim grey: what the area filter discarded is drawn so it can be reviewed,
#: but it must not compete visually with what is still alive.
COLOR_DROPPED = (130, 130, 130)
#: Amber for what is PREDICTED. OpenCV uses BGR, not RGB: (255, 200, 40) gave
#: a cyan nearly identical to the blue of the image frame, and in the contact
#: sheet predictions were confused with the border. Truth and prediction have
#: to be told apart at a glance.
COLOR_PREDICTION = (40, 190, 255)


@dataclass(frozen=True, slots=True)
class InspectionResult:
    sample_id: str
    output_path: Path
    annotations: int
    flagged: int
    dropped: int = 0


def fixed_validation_sample(
    samples, *, count: int, seed: int
) -> list[Sample]:
    """Fixed, deterministic sample. Sorting before shuffling is what fixes it."""
    ordered = sorted(samples, key=lambda s: s.sample_id)
    rng = random.Random(seed)
    picked = ordered[:]
    rng.shuffle(picked)
    return sorted(picked[:count], key=lambda s: s.sample_id)


def _to_pixels(quad: Quad, width: int, height: int) -> np.ndarray:
    return np.array([[x * width, y * height] for x, y in quad.points], dtype=np.float32)


def _draw_quad(
    canvas: np.ndarray,
    quad: Quad,
    index: int,
    *,
    flagged: bool,
    offset: tuple[int, int],
    width: int,
    height: int,
) -> None:
    pts = _to_pixels(quad, width, height)
    pts[:, 0] += offset[0]
    pts[:, 1] += offset[1]
    edge_color = COLOR_FLAGGED if flagged else COLOR_EDGE

    for i in range(4):
        p0 = tuple(np.round(pts[i]).astype(int))
        p1 = tuple(np.round(pts[(i + 1) % 4]).astype(int))
        if i == 0:
            cv2.arrowedLine(canvas, p0, p1, COLOR_ANCHOR, 3, cv2.LINE_AA, tipLength=0.06)
        else:
            cv2.line(canvas, p0, p1, edge_color, 2, cv2.LINE_AA)

    centroid = pts.mean(axis=0)
    for i, point in enumerate(pts):
        # Push the label outwards so it does not cover the corner.
        direction = point - centroid
        norm = np.linalg.norm(direction) or 1.0
        label_pos = point + direction / norm * 15.0
        cx, cy = round(label_pos[0]), round(label_pos[1])
        radius = 12 if i == 0 else 9
        cv2.circle(canvas, (cx, cy), radius, (18, 18, 18), -1, cv2.LINE_AA)
        cv2.circle(
            canvas,
            (cx, cy),
            radius,
            COLOR_ANCHOR if i == 0 else edge_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas, str(i), (cx - 4, cy + 4), FONT, 0.4, COLOR_VERTEX_TEXT, 1, cv2.LINE_AA
        )

    aspect = width / height
    tag = f"#{index}  {quad.angle_deg(aspect):.0f}deg  r={quad.side_ratio(aspect):.2f}"
    # At the midpoint of the anchor side, pushed outwards: that way the labels
    # of overlapping quads do not all pile up in the center.
    midpoint = (pts[0] + pts[1]) / 2.0
    away = midpoint - centroid
    away = away / (np.linalg.norm(away) or 1.0)
    anchor = np.round(midpoint + away * 22.0).astype(int)
    (tw, th), _ = cv2.getTextSize(tag, FONT, 0.44, 1)
    cv2.rectangle(
        canvas,
        (anchor[0] - tw // 2 - 4, anchor[1] - th - 4),
        (anchor[0] + tw // 2 + 4, anchor[1] + 5),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        canvas,
        tag,
        (anchor[0] - tw // 2, anchor[1]),
        FONT,
        0.44,
        COLOR_VERTEX_TEXT,
        1,
        cv2.LINE_AA,
    )


def _draw_dropped(
    canvas: np.ndarray,
    dropped: DroppedAnnotation,
    *,
    offset: tuple[int, int],
    width: int,
    height: int,
) -> None:
    """Dashed grey outline, without numbered vertices or anchor arrow.

    What was discarded is no longer part of the canonical set, so drawing it
    with the same encoding would confuse. It is drawn only to be able to
    locate it and decide whether it has to be fixed in Roboflow.
    """
    pts = np.array(
        [
            (x * width + offset[0], y * height + offset[1])
            for x, y in dropped.quad.points
        ]
    )
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        segments = 9
        for s in range(0, segments, 2):
            a = tuple(np.round(p0 + (p1 - p0) * (s / segments)).astype(int))
            b = tuple(np.round(p0 + (p1 - p0) * ((s + 1) / segments)).astype(int))
            cv2.line(canvas, a, b, COLOR_DROPPED, 1, cv2.LINE_AA)

    tag = f"#{dropped.index} filtered {dropped.relative_area:.0%}"
    anchor = np.round(pts.mean(axis=0)).astype(int)
    (tw, th), _ = cv2.getTextSize(tag, FONT, 0.4, 1)
    cv2.rectangle(
        canvas,
        (anchor[0] - tw // 2 - 3, anchor[1] - th - 3),
        (anchor[0] + tw // 2 + 3, anchor[1] + 4),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        canvas,
        tag,
        (anchor[0] - tw // 2, anchor[1]),
        FONT,
        0.4,
        COLOR_DROPPED,
        1,
        cv2.LINE_AA,
    )


def _draw_prediction(
    canvas: np.ndarray,
    prediction,
    *,
    offset: tuple[int, int],
    width: int,
    height: int,
) -> None:
    """Amber outline with the confidence. No numbered vertices: the canonical
    order of a prediction is not what is being looked at here."""
    if prediction.quad is None:
        # False positive without geometry (box more than half a frame
        # outside): nothing to draw inside the image. It counts in the
        # metric, not here.
        return
    pts = np.array(
        [
            (x * width + offset[0], y * height + offset[1])
            for x, y in prediction.quad.points
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [pts], True, COLOR_PREDICTION, 2, cv2.LINE_AA)
    tag = f"{prediction.score:.2f}"
    anchor = pts[0]
    (tw, th), _ = cv2.getTextSize(tag, FONT, 0.42, 1)
    cv2.rectangle(
        canvas,
        (anchor[0] - 2, anchor[1] - th - 4),
        (anchor[0] + tw + 4, anchor[1] + 3),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        canvas, tag, (anchor[0] + 1, anchor[1]), FONT, 0.42,
        COLOR_PREDICTION, 1, cv2.LINE_AA,
    )


def render_sample(
    sample: Sample,
    *,
    visibility_threshold: float,
    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA,
    predictions=(),
    pad: int = 90,
    header: int = 34,
) -> tuple[np.ndarray, InspectionResult]:
    """Draws an image with its quads, with a margin for what falls outside the border."""
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read the image {sample.image_path}")
    height, width = image.shape[:2]

    loaded = load_sample(
        sample,
        aspect=width / height,
        min_relative_area=min_relative_area,
    )
    quads = list(loaded.quads)
    findings: list[Finding] = check_quads(
        quads,
        sample.sample_id,
        visibility_threshold=visibility_threshold,
        indices=loaded.kept_indices,
    )
    flagged_indices = {f.annotation_index for f in findings}

    canvas = np.full(
        (height + 2 * pad + header, width + 2 * pad, 3), 26, dtype=np.uint8
    )
    canvas[header + pad : header + pad + height, pad : pad + width] = image
    cv2.rectangle(
        canvas,
        (pad - 1, header + pad - 1),
        (pad + width, header + pad + height),
        COLOR_FRAME,
        1,
        cv2.LINE_AA,
    )

    # The filtered ones go first so they stay under what is in force.
    for dropped in loaded.dropped:
        _draw_dropped(
            canvas,
            dropped,
            offset=(pad, header + pad),
            width=width,
            height=height,
        )

    for position, quad in enumerate(quads):
        original = loaded.kept_indices[position]
        _draw_quad(
            canvas,
            quad,
            original,
            flagged=original in flagged_indices,
            offset=(pad, header + pad),
            width=width,
            height=height,
        )

    for prediction in predictions:
        _draw_prediction(
            canvas, prediction, offset=(pad, header + pad), width=width, height=height
        )

    title = (
        f"{sample.sample_id}   {width}x{height}   {len(quads)} annotations"
        + (f"   {len(flagged_indices)} flagged" if flagged_indices else "")
        + (f"   {len(loaded.dropped)} filtered" if loaded.dropped else "")
        + (f"   {len(predictions)} predicted" if predictions else "")
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], header), COLOR_PANEL, -1)
    cv2.putText(canvas, title, (12, 23), FONT, 0.55, COLOR_PANEL_TEXT, 1, cv2.LINE_AA)

    return canvas, InspectionResult(
        sample_id=sample.sample_id,
        output_path=Path(),
        annotations=len(quads),
        flagged=len(flagged_indices),
        dropped=len(loaded.dropped),
    )


def _legend(width: int) -> np.ndarray:
    lines = [
        "magenta arrow = long side p0->p1, the one anchoring the canonical order",
        "numbers = clockwise canonical order; 0 opens the longest side",
        "green = conforming quad  |  red = candidate to violate the visibility policy",
        "dashed grey = filtered by relative area; still in the source file",
        "amber = detector prediction, with its confidence",
        "blue frame = image border; vertices outside are banknotes crossing it",
    ]
    panel = np.full((26 * len(lines) + 16, width, 3), COLOR_PANEL, dtype=np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(
            panel, text, (12, 26 + i * 26), FONT, 0.46, COLOR_PANEL_TEXT, 1, cv2.LINE_AA
        )
    return panel


def render_contact_sheet(
    images: list[np.ndarray], *, columns: int = 3, cell_width: int = 520
) -> np.ndarray:
    cells = []
    for image in images:
        scale = cell_width / image.shape[1]
        cells.append(
            cv2.resize(
                image,
                (cell_width, max(1, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    rows = []
    for start in range(0, len(cells), columns):
        chunk = cells[start : start + columns]
        tallest = max(c.shape[0] for c in chunk)
        padded = [
            np.pad(c, ((0, tallest - c.shape[0]), (0, 0), (0, 0)), constant_values=26)
            for c in chunk
        ]
        while len(padded) < columns:
            padded.append(np.full((tallest, cell_width, 3), 26, dtype=np.uint8))
        rows.append(np.hstack(padded))
    sheet = np.vstack(rows)
    return np.vstack([sheet, _legend(sheet.shape[1])])


def inspect_samples(
    samples: list[Sample],
    output_dir: str | Path,
    *,
    visibility_threshold: float,
    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA,
    predictions: dict | None = None,
) -> list[InspectionResult]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[InspectionResult] = []
    rendered: list[np.ndarray] = []

    for sample in samples:
        canvas, result = render_sample(
            sample,
            visibility_threshold=visibility_threshold,
            min_relative_area=min_relative_area,
            predictions=(predictions or {}).get(sample.sample_id, ()),
        )
        path = output_dir / f"{sample.sample_id}.png"
        cv2.imwrite(str(path), canvas)
        rendered.append(canvas)
        results.append(
            InspectionResult(
                sample_id=result.sample_id,
                output_path=path,
                annotations=result.annotations,
                flagged=result.flagged,
                dropped=result.dropped,
            )
        )

    if rendered:
        cv2.imwrite(str(output_dir / "_contact_sheet.png"), render_contact_sheet(rendered))
    return results
