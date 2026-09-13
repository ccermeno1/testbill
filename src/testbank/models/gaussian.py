"""Cajas giradas como gaussianas: KLD y ProbIoU. Torch puro, sin nada compilado.

Las dos recetas ajenas que este proyecto reproduce miden la distancia entre
cajas convirtiendo cada una en una distribucion normal bidimensional: el centro
es la media y el rectangulo, girado, da la covarianza. Comparar dos cajas pasa a
ser comparar dos campanas, y eso tiene formula cerrada. Es lo que evita
intersecar poligonos en el bucle de entrenamiento.

Licencias, y por que esto se escribio desde los papers
------------------------------------------------------
- **KLD**: Yang et al., "Learning High-Precision Bounding Box for Rotated Object
  Detection via Kullback-Leibler Divergence", NeurIPS 2021. La usa el fork
  `buzhidaoshenme/YOLOX-OBB` (Apache-2.0), cuya implementacion se uso solo para
  VERIFICAR numericamente la de aqui (`tests/test_gaussian.py`).
- **ProbIoU**: Llerena et al., "Gaussian Bounding Boxes and Probabilistic
  Intersection-over-Union for Object Detection", 2021. La usa Ultralytics, que
  es **AGPL**: de su codigo no se ha leido ni copiado nada. Todo lo de abajo sale
  de las ecuaciones del paper.

Dos convenciones de covarianza, y no es un detalle
--------------------------------------------------
Los dos papers convierten `(w, h)` en varianzas de forma distinta:

    KLD      sigma_x^2 = w^2 / 4      (la caja como 2-sigma de la gaussiana)
    ProbIoU  sigma_x^2 = w^2 / 12     (la caja como soporte de una uniforme)

Mezclarlas cambia los numeros sin cambiar el nombre. Cada funcion usa la de su
paper, y `box_to_gaussian` recibe el divisor explicito para que no se pueda
llamar "a secas".
"""

from __future__ import annotations

import torch

#: `sigma^2 = lado^2 / divisor`. Ver el docstring del modulo.
KLD_VARIANCE_DIVISOR = 4.0
PROBIOU_VARIANCE_DIVISOR = 12.0

_EPS = 1e-7


def box_to_gaussian(
    boxes: torch.Tensor, *, variance_divisor: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(N, 5)` con `cx, cy, w, h, theta` -> media y covarianza `(a, b, c)`.

    La covarianza es `R diag(w^2/d, h^2/d) R^T`, expandida:

        a = (w^2 cos^2 + h^2 sin^2) / d      varianza en x
        b = (w^2 sin^2 + h^2 cos^2) / d      varianza en y
        c = (w^2 - h^2) cos sin / d          covarianza

    Se devuelven sueltas y no como matriz porque las formulas de abajo las usan
    sueltas, y montar `(N, 2, 2)` para volver a desmontarlo es ruido.
    """
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    w2, h2 = (w * w) / variance_divisor, (h * h) / variance_divisor
    a = w2 * cos * cos + h2 * sin * sin
    b = w2 * sin * sin + h2 * cos * cos
    c = (w2 - h2) * cos * sin
    return torch.stack((cx, cy), dim=-1), a, b, c


# --- KLD (Yang et al., 2021) ----------------------------------------------


def kld_divergence(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Divergencia KL `D(N_p || N_t)` entre las gaussianas de dos cajas. `(N,)`.

    Formula cerrada para dos normales bidimensionales:

        D = 1/2 (mu_p - mu_t)^T S_t^-1 (mu_p - mu_t)
          + 1/2 tr(S_t^-1 S_p)
          + 1/2 ln(|S_t| / |S_p|)
          - 1

    Se calcula en el sistema de ejes de la caja OBJETIVO, que es donde su
    covarianza es diagonal y todo se escribe con senos y cosenos del angulo
    relativo. Es lo que hace el paper y lo que hace el fork; con matrices
    saldria lo mismo mas lento.

    NO es simetrica: `D(p||t) != D(t||p)`. Para una perdida es lo que se quiere
    -- el objetivo es fijo y la prediccion se mueve hacia el -- pero conviene
    saberlo antes de usarla como "distancia".
    """
    cx_p, cy_p, w_p, h_p, t_p = predicted.unbind(dim=-1)
    cx_t, cy_t, w_t, h_t, t_t = target.unbind(dim=-1)
    d = KLD_VARIANCE_DIVISOR

    dx, dy = cx_p - cx_t, cy_p - cy_t
    cos_t, sin_t = torch.cos(t_t), torch.sin(t_t)
    # Desplazamiento del centro proyectado sobre los ejes del objetivo.
    along = dx * cos_t + dy * sin_t
    across = dy * cos_t - dx * sin_t

    var_w_t, var_h_t = (w_t * w_t) / d, (h_t * h_t) / d
    var_w_p, var_h_p = (w_p * w_p) / d, (h_p * h_p) / d
    delta = t_p - t_t
    sin2, cos2 = torch.sin(delta) ** 2, torch.cos(delta) ** 2

    mahalanobis = 0.5 * (along * along / var_w_t + across * across / var_h_t)
    trace = 0.5 * (
        var_h_p / var_w_t * sin2
        + var_w_p / var_h_t * sin2
        + var_h_p / var_h_t * cos2
        + var_w_p / var_w_t * cos2
    )
    log_det = 0.5 * (
        torch.log(var_h_t / var_h_p.clamp(min=_EPS))
        + torch.log(var_w_t / var_w_p.clamp(min=_EPS))
    )
    return mahalanobis + trace + log_det - 1.0


def kld_loss(
    predicted: torch.Tensor, target: torch.Tensor, *, tau: float = 1.0
) -> torch.Tensor:
    """La perdida del paper: `1 - 1 / (tau + ln(D + 1))`. `(N,)`, en `[0, 1)`.

    La divergencia cruda no esta acotada y crece sin freno con el error de
    centro; el envoltorio logaritmico la aplana para que una caja muy lejos no
    domine el lote. `tau = 1` es el valor del paper y del fork.
    """
    divergence = kld_divergence(predicted, target).clamp(min=0.0)
    return 1.0 - 1.0 / (tau + torch.log1p(divergence))


# --- ProbIoU (Llerena et al., 2021) ----------------------------------------


def bhattacharyya_distance(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """`B_D` entre las gaussianas de dos cajas, con las ecuaciones del paper.

    Con `(a, b, c)` las covarianzas y `(x, y)` los centros de cada una:

        B_D = 1/4 * [ (a1+a2)(y1-y2)^2 + (b1+b2)(x1-x2)^2 ] / [ (a1+a2)(b1+b2) - (c1+c2)^2 ]
            + 1/2 * [ (c1+c2)(x2-x1)(y1-y2) ]             / [ (a1+a2)(b1+b2) - (c1+c2)^2 ]
            + 1/2 * ln( [ (a1+a2)(b1+b2) - (c1+c2)^2 ] / (4 sqrt((a1 b1 - c1^2)(a2 b2 - c2^2))) )

    SI es simetrica, al contrario que la KL.
    """
    mu1, a1, b1, c1 = box_to_gaussian(predicted, variance_divisor=PROBIOU_VARIANCE_DIVISOR)
    mu2, a2, b2, c2 = box_to_gaussian(target, variance_divisor=PROBIOU_VARIANCE_DIVISOR)
    x1, y1 = mu1.unbind(dim=-1)
    x2, y2 = mu2.unbind(dim=-1)

    a, b, c = a1 + a2, b1 + b2, c1 + c2
    denominator = (a * b - c * c).clamp(min=_EPS)
    t1 = 0.25 * (a * (y1 - y2) ** 2 + b * (x1 - x2) ** 2) / denominator
    t2 = 0.5 * (c * (x2 - x1) * (y1 - y2)) / denominator
    det1 = (a1 * b1 - c1 * c1).clamp(min=_EPS)
    det2 = (a2 * b2 - c2 * c2).clamp(min=_EPS)
    t3 = 0.5 * torch.log(denominator / (4.0 * torch.sqrt(det1 * det2)).clamp(min=_EPS))
    return (t1 + t2 + t3).clamp(min=_EPS, max=100.0)


def probiou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """`ProbIoU = 1 - H_D`, con `H_D = sqrt(1 - exp(-B_D))` la de Hellinger. `(N,)`.

    Vale 1 para dos cajas iguales y baja hacia 0 al alejarse. Es lo que la
    receta de Ultralytics usa como "IoU" tanto en la perdida (`1 - probiou`)
    como en la metrica de alineacion del asignador.
    """
    hellinger = torch.sqrt(1.0 - torch.exp(-bhattacharyya_distance(predicted, target)) + _EPS)
    return 1.0 - hellinger


def pairwise_probiou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`(N, M)` de ProbIoU entre cada caja de `a` y cada una de `b`."""
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return a.new_zeros((n, m))
    left = a[:, None, :].expand(n, m, 5).reshape(-1, 5)
    right = b[None, :, :].expand(n, m, 5).reshape(-1, 5)
    return probiou(left, right).view(n, m)


def pairwise_kld_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`(N, M)` de `kld_loss(a_i, b_j)`: la prediccion en filas, el objetivo en
    columnas. El orden importa porque la KL no es simetrica."""
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return a.new_zeros((n, m))
    left = a[:, None, :].expand(n, m, 5).reshape(-1, 5)
    right = b[None, :, :].expand(n, m, 5).reshape(-1, 5)
    return kld_loss(left, right).view(n, m)


__all__ = [
    "KLD_VARIANCE_DIVISOR",
    "PROBIOU_VARIANCE_DIVISOR",
    "bhattacharyya_distance",
    "box_to_gaussian",
    "kld_divergence",
    "kld_loss",
    "pairwise_kld_loss",
    "pairwise_probiou",
    "probiou",
]

