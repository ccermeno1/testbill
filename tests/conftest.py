from __future__ import annotations

import math

import pytest
from hypothesis import strategies as st

from testbank.geometry.quad import Quad

# ratio >= 1.2 mantiene los lados cortos fuera de la tolerancia de empate
# (1/(1-0.05) = 1.053) y por encima del umbral de ancla inestable (1.1).
SAFE_MIN_RATIO = 1.2


def rotated_rect_points(
    cx: float, cy: float, half_long: float, ratio: float, theta: float
) -> list[tuple[float, float]]:
    """Rectangulo girado en coordenadas de imagen (y hacia abajo).

    Devuelve los vertices en orden horario visual antes de canonicalizar.
    """
    half_short = half_long / ratio
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    local = [
        (-half_long, -half_short),
        (+half_long, -half_short),
        (+half_long, +half_short),
        (-half_long, +half_short),
    ]
    return [
        (cx + lx * cos_t - ly * sin_t, cy + lx * sin_t + ly * cos_t)
        for lx, ly in local
    ]


@st.composite
def rotated_rects(draw, min_ratio: float = SAFE_MIN_RATIO, max_ratio: float = 5.0):
    """Rectangulos girados que caben holgadamente en el rango tolerante."""
    half_long = draw(st.floats(min_value=0.04, max_value=0.22))
    ratio = draw(st.floats(min_value=min_ratio, max_value=max_ratio))
    theta = draw(st.floats(min_value=0.0, max_value=math.pi, exclude_max=True))
    margin = half_long * 1.45 + 0.02  # media diagonal <= half_long*sqrt(2)
    cx = draw(st.floats(min_value=margin, max_value=1.0 - margin))
    cy = draw(st.floats(min_value=margin, max_value=1.0 - margin))
    return Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta))


@pytest.fixture()
def config_fork(tmp_path):
    """Config minima para los tests del adaptador del fork de YOLOX-OBB."""
    from testbank.config import Config

    base = Config()
    return base.model_copy(
        update={
            "data": base.data.model_copy(update={"derived_dir": tmp_path / "d"}),
            "detector": base.detector.model_copy(
                update={"epochs": 3, "image_size": 416}
            ),
        }
    )
