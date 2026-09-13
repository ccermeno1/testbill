"""Generates synthetic data with the structure and pathologies of the real export.

Stands in for the Roboflow export until it arrives. Deliberately reproduces
what makes the problem hard:

  - fanned and adjacent banknotes at different angles (the case that breaks
    the axis-aligned box and motivates the OBB)
  - banknotes that overlap and touch, of the same color and texture
  - banknotes crossing the image border, with vertices outside [0,1]
  - the odd nearly square banknote, which triggers the unstable-anchor warning
  - a few images that VIOLATE the visibility policy on purpose, so the
    consistency check has something real to find

Visibility is computed with shapely against the union of the banknotes drawn
AFTERWARDS, which are the ones that end up on top.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

VISIBILITY_THRESHOLD = 0.25
MIN_INSIDE_FRAME = 0.40
BILL_ASPECT = (1.85, 2.25)

# Banknotes clearly separable from the background: the inspection
# visualization is useless if the banknote cannot be told from the table.
_PALETTE = [
    (96, 158, 120),
    (168, 132, 96),
    (110, 120, 178),
    (140, 164, 104),
    (128, 112, 156),
]


@dataclass
class Bill:
    corners: np.ndarray  # (4,2) in pixels, clockwise order
    polygon: Polygon


def _rot_rect(cx, cy, half_long, ratio, theta) -> np.ndarray:
    half_short = half_long / ratio
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    local = np.array(
        [
            [-half_long, -half_short],
            [+half_long, -half_short],
            [+half_long, +half_short],
            [-half_long, +half_short],
        ]
    )
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return local @ rot.T + np.array([cx, cy])


def _bill_texture(rng: random.Random, width: int, height: int) -> np.ndarray:
    base = np.array(rng.choice(_PALETTE), dtype=np.float32)
    tint = np.array([rng.uniform(-12, 12) for _ in range(3)], dtype=np.float32)
    patch = np.ones((height, width, 3), dtype=np.float32) * (base + tint)

    grain = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 7, (height, width, 1))
    patch += grain

    cv2.rectangle(patch, (2, 2), (width - 3, height - 3), tuple(float(v * 0.82) for v in base), 2)
    cv2.ellipse(
        patch,
        (int(width * 0.74), height // 2),
        (int(width * 0.13), int(height * 0.3)),
        0,
        0,
        360,
        tuple(float(v * 1.14) for v in base),
        -1,
    )
    for i in range(5):
        y = int(height * (0.25 + 0.12 * i))
        cv2.line(patch, (int(width * 0.08), y), (int(width * 0.5), y),
                 tuple(float(v * 0.7) for v in base), 1)

    # Stains: the downstream classifier looks for exactly this.
    for _ in range(rng.randint(0, 3)):
        centre = (rng.randint(0, width), rng.randint(0, height))
        axes = (rng.randint(4, max(5, width // 10)), rng.randint(4, max(5, height // 6)))
        shade = tuple(float(v * rng.uniform(0.45, 0.7)) for v in base)
        cv2.ellipse(patch, centre, axes, rng.uniform(0, 180), 0, 360, shade, -1)

    return np.clip(patch, 0, 255)


def _background(rng: random.Random, width: int, height: int) -> np.ndarray:
    tone = rng.uniform(58, 92)
    canvas = np.ones((height, width, 3), dtype=np.float32) * tone
    gradient = np.linspace(-10, 10, width, dtype=np.float32)[None, :, None]
    canvas += gradient
    canvas += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 4, (height, width, 1))
    for _ in range(rng.randint(2, 5)):
        p0 = (rng.randint(0, width), rng.randint(0, height))
        p1 = (rng.randint(0, width), rng.randint(0, height))
        cv2.line(canvas, p0, p1, (tone * 1.18,) * 3, rng.randint(1, 3))
    return np.clip(canvas, 0, 255)


def _paste(canvas: np.ndarray, corners: np.ndarray, texture: np.ndarray) -> None:
    height, width = texture.shape[:2]
    src = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    warped = cv2.warpPerspective(texture, matrix, (canvas.shape[1], canvas.shape[0]))
    mask = cv2.warpPerspective(
        np.ones((height, width), np.float32), matrix, (canvas.shape[1], canvas.shape[0])
    )
    mask = cv2.GaussianBlur(mask, (3, 3), 0)[..., None]
    canvas *= 1.0 - mask
    canvas += warped * mask


def _scene(rng: random.Random, kind: str, width: int, height: int) -> list[np.ndarray]:
    """Returns the rectangles in drawing order: the last one ends up on top."""
    out: list[np.ndarray] = []
    span = min(width, height)

    if kind == "fan":
        cx = rng.uniform(width * 0.4, width * 0.6)
        cy = rng.uniform(height * 0.45, height * 0.62)
        base = rng.uniform(0, math.pi)
        count = rng.randint(3, 5)
        half_long = span * rng.uniform(0.16, 0.21)
        for i in range(count):
            angle = base + i * math.radians(rng.uniform(18, 32))
            shift = i * half_long * rng.uniform(0.16, 0.3)
            out.append(
                _rot_rect(
                    cx + shift * math.cos(base + math.pi / 2),
                    cy + shift * math.sin(base + math.pi / 2),
                    half_long,
                    rng.uniform(*BILL_ASPECT),
                    angle,
                )
            )
    elif kind == "overlap":
        cx = rng.uniform(width * 0.35, width * 0.65)
        cy = rng.uniform(height * 0.35, height * 0.65)
        half_long = span * rng.uniform(0.17, 0.22)
        for i in range(rng.randint(2, 4)):
            out.append(
                _rot_rect(
                    cx + rng.uniform(-0.55, 0.55) * half_long,
                    cy + rng.uniform(-0.55, 0.55) * half_long,
                    half_long,
                    rng.uniform(*BILL_ASPECT),
                    rng.uniform(0, math.pi),
                )
            )
    elif kind == "edge":
        half_long = span * rng.uniform(0.15, 0.2)
        for _ in range(rng.randint(1, 3)):
            side = rng.choice(["l", "r", "t", "b"])
            cx = {"l": half_long * 0.35, "r": width - half_long * 0.35}.get(
                side, rng.uniform(width * 0.25, width * 0.75)
            )
            cy = {"t": half_long * 0.35, "b": height - half_long * 0.35}.get(
                side, rng.uniform(height * 0.25, height * 0.75)
            )
            out.append(
                _rot_rect(cx, cy, half_long, rng.uniform(*BILL_ASPECT), rng.uniform(0, math.pi))
            )
        out.append(
            _rot_rect(
                rng.uniform(width * 0.35, width * 0.65),
                rng.uniform(height * 0.35, height * 0.65),
                half_long,
                rng.uniform(*BILL_ASPECT),
                rng.uniform(0, math.pi),
            )
        )
    elif kind == "near_square":
        out.append(
            _rot_rect(
                rng.uniform(width * 0.3, width * 0.7),
                rng.uniform(height * 0.3, height * 0.7),
                span * 0.13,
                rng.uniform(1.02, 1.08),
                rng.uniform(0, math.pi),
            )
        )
        for _ in range(rng.randint(1, 2)):
            out.append(
                _rot_rect(
                    rng.uniform(width * 0.2, width * 0.8),
                    rng.uniform(height * 0.2, height * 0.8),
                    span * rng.uniform(0.15, 0.2),
                    rng.uniform(*BILL_ASPECT),
                    rng.uniform(0, math.pi),
                )
            )
    else:  # scattered
        for _ in range(rng.randint(1, 4)):
            half_long = span * rng.uniform(0.12, 0.2)
            out.append(
                _rot_rect(
                    rng.uniform(half_long, width - half_long),
                    rng.uniform(half_long, height - half_long),
                    half_long,
                    rng.uniform(*BILL_ASPECT),
                    rng.uniform(0, math.pi),
                )
            )
    return out


def _visibility(index: int, polys: list[Polygon]) -> float:
    """Visible fraction: what the banknotes drawn afterwards do not cover."""
    above = polys[index + 1 :]
    if not above:
        return 1.0
    covered = polys[index].intersection(unary_union(above)).area
    return max(0.0, 1.0 - covered / polys[index].area)


def _inside_frame(poly: Polygon, width: int, height: int) -> float:
    return poly.intersection(box(0, 0, width, height)).area / poly.area


def render_image(
    rng: random.Random, kind: str, violate_policy: bool
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    width = rng.choice([960, 1024, 1120])
    height = rng.choice([720, 768, 840])
    canvas = _background(rng, width, height)

    rects = _scene(rng, kind, width, height)
    polys = [Polygon(r) for r in rects]

    for rect in rects:
        long_px = int(np.linalg.norm(rect[1] - rect[0]))
        short_px = int(np.linalg.norm(rect[2] - rect[1]))
        texture = _bill_texture(rng, max(24, long_px), max(16, short_px))
        _paste(canvas, rect, texture)

    annotated: list[np.ndarray] = []
    meta = {"kind": kind, "violates_policy": False, "bills_drawn": len(rects)}
    below_threshold: list[int] = []
    for index, rect in enumerate(rects):
        visible = _visibility(index, polys)
        if _inside_frame(polys[index], width, height) < MIN_INSIDE_FRAME:
            continue
        if visible >= VISIBILITY_THRESHOLD:
            annotated.append(rect)
        else:
            below_threshold.append(index)

    if violate_policy and below_threshold:
        annotated.append(rects[below_threshold[0]])
        meta["violates_policy"] = True

    return np.clip(canvas, 0, 255).astype(np.uint8), annotated, meta


def to_label_lines(rects: list[np.ndarray], width: int, height: int) -> list[str]:
    lines = []
    for rect in rects:
        coords = []
        for x, y in rect:
            coords.append(f"{min(max(x / width, -0.499), 1.499):.6f}")
            coords.append(f"{min(max(y / height, -0.499), 1.499):.6f}")
        lines.append("0 " + " ".join(coords))
    return lines


def generate(
    out_root: Path, count: int, seed: int, layout: str, violation_rate: float = 0.16
) -> dict:
    rng = random.Random(seed)
    kinds = ["scattered", "fan", "overlap", "edge", "near_square"]
    weights = [0.32, 0.24, 0.22, 0.14, 0.08]

    if layout == "roboflow":
        split_of = lambda i: "train" if i % 10 < 7 else ("valid" if i % 10 < 9 else "test")
        for split in ("train", "valid", "test"):
            (out_root / split / "images").mkdir(parents=True, exist_ok=True)
            (out_root / split / "labels").mkdir(parents=True, exist_ok=True)
    else:
        split_of = lambda i: ""
        (out_root / "images").mkdir(parents=True, exist_ok=True)
        (out_root / "labels").mkdir(parents=True, exist_ok=True)

    index_meta = {}
    for i in range(count):
        kind = rng.choices(kinds, weights=weights)[0]
        violate = rng.random() < violation_rate
        image, rects, meta = render_image(rng, kind, violate)
        height, width = image.shape[:2]

        sample_id = f"bill_{i:04d}"
        split = split_of(i)
        base = out_root / split if split else out_root
        cv2.imwrite(str(base / "images" / f"{sample_id}.jpg"), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = to_label_lines(rects, width, height)
        (base / "labels" / f"{sample_id}.txt").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )
        meta.update({"split": split or None, "annotated": len(rects),
                     "width": width, "height": height})
        index_meta[sample_id] = meta

    summary = {
        "count": count,
        "seed": seed,
        "layout": layout,
        "deliberate_violations": sorted(
            k for k, v in index_meta.items() if v["violates_policy"]
        ),
        "samples": index_meta,
    }
    (out_root / "_synthetic_truth.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--layout", choices=["roboflow", "flat"], default="roboflow")
    args = parser.parse_args()
    summary = generate(args.out, args.count, args.seed, args.layout)
    print(f"{summary['count']} images in {args.out} ({args.layout})")
    print(f"deliberate violations: {len(summary['deliberate_violations'])}")


if __name__ == "__main__":
    main()
