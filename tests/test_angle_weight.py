"""Atenuador de la perdida de angulo en cajas casi cuadradas."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="el candidato propio necesita torch")

from testbank.config import AngleWeightConfig
from testbank.models.losses import (
    DECAYS,
    angle_weight,
    side_ratio_px,
)


def w(ratio, **kwargs):
    return angle_weight(torch.tensor(ratio, dtype=torch.float32), **kwargs)


# --- el comportamiento que se busca ---------------------------------------


def test_una_caja_bien_definida_pesa_uno():
    """Un billete 2:1 tiene el angulo perfectamente definido: sin descuento."""
    assert float(w([2.0])) == pytest.approx(1.0)


def test_un_cuadrado_perfecto_no_pesa_nada():
    """Su angulo no esta definido: castigar al modelo por fallarlo es ruido."""
    assert float(w([1.0])) == pytest.approx(0.0)


def test_por_encima_del_umbral_siempre_pesa_uno():
    assert torch.allclose(w([1.1, 1.5, 3.0, 10.0]), torch.ones(4), atol=1e-6)


def test_el_peso_crece_con_el_ratio():
    """Cuanto mejor definido esta el angulo, mas cuenta acertarlo."""
    values = w([1.0, 1.02, 1.05, 1.08, 1.1])
    assert torch.all(values[1:] >= values[:-1])


def test_el_peso_se_queda_en_su_rango():
    values = w(torch.linspace(0.5, 5.0, 50).tolist())
    assert torch.all(values >= 0.0)
    assert torch.all(values <= 1.0)


# --- la rama de control ---------------------------------------------------


def test_desactivado_devuelve_unos():
    """Es el brazo de control del experimento: sin atenuar, todo pesa igual."""
    values = w([1.0, 1.05, 2.0], enabled=False)
    assert torch.allclose(values, torch.ones(3))


def test_el_suelo_se_respeta():
    """`min_weight` permite atenuar sin anular, por si cero resulta ser mucho."""
    assert float(w([1.0], min_weight=0.3)) == pytest.approx(0.3)
    assert float(w([2.0], min_weight=0.3)) == pytest.approx(1.0)


# --- las formas de decaimiento --------------------------------------------


@pytest.mark.parametrize("decay", sorted(DECAYS))
def test_toda_forma_va_de_cero_a_uno(decay):
    assert float(w([1.0], decay=decay)) == pytest.approx(0.0, abs=1e-6)
    assert float(w([1.1], decay=decay)) == pytest.approx(1.0, abs=1e-6)


def test_smoothstep_arranca_mas_despacio_que_lineal():
    """Es su motivo de ser: dejar casi sin peso la franja mas ambigua."""
    mitad = 1.05  # justo a medio camino con umbral 1.1
    assert float(w([mitad], decay="smoothstep")) == pytest.approx(0.5, abs=0.01)
    cuarto = 1.025
    assert float(w([cuarto], decay="smoothstep")) < float(w([cuarto], decay="linear"))


def test_step_corta_por_lo_sano():
    """Referencia para medir si la transicion suave aporta algo."""
    assert float(w([1.09], decay="step")) == 0.0
    assert float(w([1.10], decay="step")) == 1.0


def test_forma_desconocida_dice_cuales_hay():
    with pytest.raises(ValueError, match="decaimiento desconocida"):
        w([2.0], decay="exponencial")


def test_umbral_invalido_es_error():
    with pytest.raises(ValueError, match="ratio_threshold"):
        w([2.0], ratio_threshold=1.0)


# --- el ratio -------------------------------------------------------------


def test_el_ratio_no_depende_del_orden_de_los_lados():
    """El orden de w y h es justo lo que la ambiguedad vuelve arbitrario."""
    a = side_ratio_px(torch.tensor([100.0]), torch.tensor([50.0]))
    b = side_ratio_px(torch.tensor([50.0]), torch.tensor([100.0]))
    assert torch.allclose(a, b)
    assert float(a) == pytest.approx(2.0)


def test_un_lado_de_cero_no_revienta():
    ratio = side_ratio_px(torch.tensor([100.0]), torch.tensor([0.0]))
    assert torch.isfinite(ratio).all()


def test_un_ratio_invertido_da_el_peso_minimo_no_uno_negativo():
    """Si alguien intercambia los lados, el peso sale minimo -- no absurdo."""
    assert float(w([0.5])) == pytest.approx(0.0)


# --- la config ------------------------------------------------------------


def test_la_config_rechaza_una_forma_que_no_existe():
    with pytest.raises(ValueError, match="decaimiento desconocida"):
        AngleWeightConfig(decay="ninguna")


def test_los_valores_por_defecto_son_los_documentados():
    cfg = AngleWeightConfig()
    assert cfg.enabled is True
    assert cfg.ratio_threshold == 1.1  # el mismo umbral del aviso de ancla
    assert cfg.min_weight == 0.0
    assert cfg.decay == "smoothstep"


def test_se_puede_apagar_desde_la_config():
    cfg = AngleWeightConfig(enabled=False)
    values = angle_weight(
        torch.tensor([1.0, 2.0]),
        enabled=cfg.enabled,
        ratio_threshold=cfg.ratio_threshold,
        min_weight=cfg.min_weight,
        decay=cfg.decay,
    )
    assert torch.allclose(values, torch.ones(2))


def test_el_atenuador_queda_registrado_en_la_ejecucion():
    """Sin esto no seria un experimento comparable: dos runs con atenuadores
    distintos se veran iguales en la tabla."""
    from testbank.config import Config

    yaml_text = Config().dump_yaml()
    assert "angle_weight" in yaml_text
    assert "ratio_threshold" in yaml_text
    assert "smoothstep" in yaml_text
