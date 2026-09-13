"""KLD y ProbIoU: propiedades, y fidelidad numerica al original donde se puede.

La KLD se comprueba contra la formula del fork `buzhidaoshenme/YOLOX-OBB`
(Apache-2.0), transcrita aqui con atribucion: es la implementacion que usa la
receta que estamos reproduciendo, asi que "da lo mismo" es exactamente la
afirmacion que hay que anclar. Recibe el angulo en GRADOS; la nuestra en
radianes.

La ProbIoU NO tiene referencia de codigo: Ultralytics es AGPL y no se ha leido.
Se comprueba contra las propiedades que el paper garantiza y contra una
implementacion matricial independiente de la distancia de Bhattacharyya, que
solo usa `torch.linalg` y no comparte una linea con la de produccion.
"""

from __future__ import annotations

import math

import pytest
import torch

from testbank.models.gaussian import (
    KLD_VARIANCE_DIVISOR,
    PROBIOU_VARIANCE_DIVISOR,
    bhattacharyya_distance,
    box_to_gaussian,
    kld_divergence,
    kld_loss,
    pairwise_kld_loss,
    pairwise_probiou,
    probiou,
)

torch.manual_seed(0)


def _boxes(n, *, seed):
    g = torch.Generator().manual_seed(seed)
    cx = torch.rand(n, generator=g) * 400
    cy = torch.rand(n, generator=g) * 400
    w = 20 + torch.rand(n, generator=g) * 200
    h = 10 + torch.rand(n, generator=g) * 100
    theta = (torch.rand(n, generator=g) - 0.5) * math.pi
    return torch.stack((cx, cy, w, h, theta), dim=-1)


# --- KLD: fidelidad al fork ----------------------------------------------


def _fork_kld(pred, target, taf=1.0):
    """`KLD_loss.kld_loss` de buzhidaoshenme/YOLOX-OBB (Apache-2.0), transcrita
    tal cual salvo el nombre de las variables. Angulos en GRADOS."""
    delta_x = pred[:, 0] - target[:, 0]
    delta_y = pred[:, 1] - target[:, 1]
    pre = math.pi * pred[:, 4] / 180.0
    tgt = math.pi * target[:, 4] / 180.0
    delta = pre - tgt
    kld = (
        0.5
        * (
            4 * (delta_x * torch.cos(tgt) + delta_y * torch.sin(tgt)) ** 2 / target[:, 2] ** 2
            + 4 * (delta_y * torch.cos(tgt) - delta_x * torch.sin(tgt)) ** 2 / target[:, 3] ** 2
        )
        + 0.5
        * (
            pred[:, 3] ** 2 / target[:, 2] ** 2 * torch.sin(delta) ** 2
            + pred[:, 2] ** 2 / target[:, 3] ** 2 * torch.sin(delta) ** 2
            + pred[:, 3] ** 2 / target[:, 3] ** 2 * torch.cos(delta) ** 2
            + pred[:, 2] ** 2 / target[:, 2] ** 2 * torch.cos(delta) ** 2
        )
        + 0.5
        * (
            torch.log(target[:, 3] ** 2 / pred[:, 3] ** 2)
            + torch.log(target[:, 2] ** 2 / pred[:, 2] ** 2)
        )
        - 1.0
    )
    return 1 - 1 / (taf + torch.log(kld + 1))


def test_kld_coincide_con_la_formula_del_fork():
    p, t = _boxes(500, seed=1), _boxes(500, seed=2)
    ours = kld_loss(p, t)
    en_grados = lambda b: torch.cat((b[:, :4], torch.rad2deg(b[:, 4:5])), dim=1)
    theirs = _fork_kld(en_grados(p), en_grados(t))
    assert torch.allclose(ours, theirs, atol=1e-5), (ours - theirs).abs().max()


def test_kld_es_cero_para_la_misma_caja():
    b = _boxes(50, seed=3)
    assert torch.allclose(kld_divergence(b, b), torch.zeros(50), atol=1e-5)
    assert torch.allclose(kld_loss(b, b), torch.zeros(50), atol=1e-5)


def test_kld_no_es_simetrica_y_se_sabe():
    """No es un fallo: es la KL. El test existe para que nadie la use como
    distancia sin saberlo."""
    p, t = _boxes(50, seed=4), _boxes(50, seed=5)
    assert not torch.allclose(kld_divergence(p, t), kld_divergence(t, p))


def test_kld_crece_con_el_error_de_centro():
    t = torch.tensor([[100.0, 100.0, 80.0, 40.0, 0.3]])
    valores = [kld_loss(t + torch.tensor([[dx, 0, 0, 0, 0]]), t).item() for dx in (0, 5, 20, 80)]
    assert valores == sorted(valores) and valores[0] < 1e-5


def test_kld_es_invariante_al_giro_de_180_grados():
    """`(w, h, t)` y `(w, h, t + pi)` son el mismo rectangulo. La gaussiana
    tambien: la covarianza es cuadratica en seno y coseno."""
    t = _boxes(50, seed=6)
    girada = t.clone()
    girada[:, 4] += math.pi
    assert torch.allclose(kld_loss(girada, t), torch.zeros(50), atol=1e-5)


def test_kld_esta_acotada_en_cero_uno():
    p, t = _boxes(300, seed=7), _boxes(300, seed=8)
    v = kld_loss(p, t)
    assert (v >= 0).all() and (v < 1).all()


# --- ProbIoU: propiedades y contraste matricial ----------------------------


def _bhattacharyya_matricial(p, t):
    """Referencia independiente: la B_D con matrices, sin la expansion `a,b,c`."""
    def cov(b):
        _, a, bb, c = box_to_gaussian(b, variance_divisor=PROBIOU_VARIANCE_DIVISOR)
        return torch.stack((torch.stack((a, c), -1), torch.stack((c, bb), -1)), -2)

    s1, s2 = cov(p), cov(t)
    s = 0.5 * (s1 + s2)
    d = (p[:, :2] - t[:, :2]).unsqueeze(-1)
    term1 = 0.125 * (d.transpose(1, 2) @ torch.linalg.inv(s) @ d).squeeze(-1).squeeze(-1)
    term2 = 0.5 * torch.log(
        torch.linalg.det(s) / torch.sqrt(torch.linalg.det(s1) * torch.linalg.det(s2))
    )
    return term1 + term2


def test_bhattacharyya_coincide_con_la_version_matricial():
    p, t = _boxes(400, seed=9), _boxes(400, seed=10)
    ours = bhattacharyya_distance(p, t)
    ref = _bhattacharyya_matricial(p, t).clamp(min=1e-7, max=100.0)
    assert torch.allclose(ours, ref, atol=1e-4, rtol=1e-4), (ours - ref).abs().max()


def test_probiou_es_uno_para_la_misma_caja():
    b = _boxes(50, seed=11)
    assert torch.allclose(probiou(b, b), torch.ones(50), atol=1e-3)


def test_probiou_es_simetrica():
    p, t = _boxes(100, seed=12), _boxes(100, seed=13)
    assert torch.allclose(probiou(p, t), probiou(t, p), atol=1e-6)


def test_probiou_baja_al_alejarse_y_esta_en_cero_uno():
    t = torch.tensor([[100.0, 100.0, 80.0, 40.0, 0.3]])
    valores = [probiou(t + torch.tensor([[dx, 0, 0, 0, 0]]), t).item() for dx in (0, 5, 20, 80, 300)]
    assert valores == sorted(valores, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in valores)


def test_probiou_es_invariante_al_giro_de_180_grados():
    t = _boxes(50, seed=14)
    girada = t.clone()
    girada[:, 4] += math.pi
    assert torch.allclose(probiou(girada, t), torch.ones(50), atol=1e-3)


def test_una_caja_cuadrada_girada_no_cambia_nada():
    """Un cuadrado es el mismo con cualquier angulo, y su gaussiana es
    isotropica: las dos medidas tienen que verlo. Es el caso que rompe la
    representacion `(w, h, theta)` y que las gaussianas resuelven solas."""
    cuadrado = torch.tensor([[100.0, 100.0, 60.0, 60.0, 0.0]])
    girado = torch.tensor([[100.0, 100.0, 60.0, 60.0, 0.7]])
    assert probiou(girado, cuadrado).item() == pytest.approx(1.0, abs=1e-3)
    assert kld_loss(girado, cuadrado).item() == pytest.approx(0.0, abs=1e-5)


# --- convenciones de covarianza, que no se pueden mezclar -----------------


def test_las_dos_convenciones_de_varianza_son_las_de_sus_papers():
    assert KLD_VARIANCE_DIVISOR == 4.0
    assert PROBIOU_VARIANCE_DIVISOR == 12.0


def test_box_to_gaussian_exige_el_divisor():
    with pytest.raises(TypeError):
        box_to_gaussian(_boxes(1, seed=0))  # type: ignore[call-arg]


# --- gradientes y pares ----------------------------------------------------


def test_los_gradientes_son_finitos_incluso_en_el_caso_cuadrado():
    p = _boxes(30, seed=15).requires_grad_(True)
    t = _boxes(30, seed=16)
    t[:5, 2] = t[:5, 3]  # cuadrados exactos
    for fn in (kld_loss, lambda a, b: 1 - probiou(a, b)):
        p.grad = None
        fn(p, t).sum().backward()
        assert torch.isfinite(p.grad).all()


def test_las_versiones_por_pares_coinciden_con_las_de_uno_a_uno():
    a, b = _boxes(7, seed=17), _boxes(4, seed=18)
    pk, pp = pairwise_kld_loss(a, b), pairwise_probiou(a, b)
    assert pk.shape == pp.shape == (7, 4)
    for i in range(7):
        for j in range(4):
            assert pk[i, j].item() == pytest.approx(kld_loss(a[i : i + 1], b[j : j + 1]).item(), abs=1e-6)
            assert pp[i, j].item() == pytest.approx(probiou(a[i : i + 1], b[j : j + 1]).item(), abs=1e-6)
