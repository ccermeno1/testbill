"""Como medir cuanto se parecen dos cajas giradas, de tres maneras.

Las recetas de perdida no comparten forma de medir el solape, y esa diferencia
ES la diferencia entre ellas: la propia usa la envolvente alineada (en
`assign.py`), el fork de YOLOX-OBB y Ultralytics usan gaussianas, y DDGRCF usa
el IoU exacto de poligonos. Este modulo tiene las dos que no son la envolvente.

Todo en torch puro y diferenciable. Nada compilado: era la restriccion, y las
tres formas caben en el bucle de entrenamiento (medido: el IoU exacto por pares
de un asignador entero, 3549 x 2, tarda ~19 ms).

=== Gaussianas: KLD y ProbIoU ===
Cajas giradas como gaussianas: KLD y ProbIoU. Torch puro, sin nada compilado.

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

=== IoU exacto de poligonos ===
IoU EXACTO entre rectangulos girados, en torch puro y diferenciable.

Es lo que sustituye al operador compilado (`box_iou_rotated` / `convex`) de
DDGRCF/YOLOX_OBB, que lo usa en dos sitios: el coste del SimOTA y la perdida de
caja (PolyIoU). Sin esto el port no reproduce su receta; con las gaussianas la
cambiaria.

Como se calcula
---------------
Para dos cuadrilateros convexos, la interseccion es un poligono convexo cuyos
vertices son de tres tipos: esquinas del primero dentro del segundo, esquinas
del segundo dentro del primero, y cruces de aristas. Se recogen los 24
candidatos (4 + 4 + 16) con una mascara de validez, se ordenan por angulo
alrededor de su centro y se aplica la formula del cordon (shoelace) enmascarada.
Como mucho 8 son validos.

Es diferenciable respecto a las coordenadas: las esquinas y los cruces son
funciones lisas de las cajas; solo el ORDEN se calcula sin gradiente, y el orden
es constante a trozos, asi que no lo necesita. Es el mismo planteamiento que la
"Rotated IoU" de Lanxiao Li (MIT), reescrito aqui desde la geometria.

Coste: para el asignador son `celdas x billetes` pares por imagen (3549 x ~2),
todo vectorizado. Medido, es del orden de milisegundos: no es el "no cabe en el
bucle" de shapely, que va par a par en Python.
"""

from __future__ import annotations

import torch

# --------------------------------------------------------------------------
# Gaussianas
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Poligonos
# --------------------------------------------------------------------------

_EPS = 1e-8


def box_corners(boxes: torch.Tensor) -> torch.Tensor:
    """`(N, 5)` cx, cy, w, h, theta -> `(N, 4, 2)` esquinas en sentido horario
    (en coordenadas de imagen, con y hacia abajo)."""
    cx, cy, w, h, theta = boxes.unbind(dim=-1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    dx = torch.stack((-w, w, w, -w), dim=-1) / 2
    dy = torch.stack((-h, -h, h, h), dim=-1) / 2
    x = cx[:, None] + dx * cos[:, None] - dy * sin[:, None]
    y = cy[:, None] + dx * sin[:, None] + dy * cos[:, None]
    return torch.stack((x, y), dim=-1)


def _cross(o, a, b):
    """Producto vectorial 2D de (a - o) x (b - o). Signo = de que lado esta b."""
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (
        b[..., 0] - o[..., 0]
    )


def _edge_intersections(c1: torch.Tensor, c2: torch.Tensor):
    """Cruces entre las 4 aristas de cada caja: `(N, 16, 2)` y mascara `(N, 16)`."""
    a = c1  # (N, 4, 2)
    b = torch.roll(c1, -1, dims=1)
    c = c2
    d = torch.roll(c2, -1, dims=1)
    # Todas las combinaciones arista_i de 1 x arista_j de 2.
    a = a[:, :, None, :].expand(-1, 4, 4, -1)
    b = b[:, :, None, :].expand(-1, 4, 4, -1)
    c = c[:, None, :, :].expand(-1, 4, 4, -1)
    d = d[:, None, :, :].expand(-1, 4, 4, -1)
    # Parametros t (sobre ab) y u (sobre cd) del cruce de las rectas.
    ab = b - a
    cd = d - c
    ac = c - a
    denominator = ab[..., 0] * cd[..., 1] - ab[..., 1] * cd[..., 0]
    parallel = denominator.abs() < _EPS
    safe = torch.where(parallel, torch.ones_like(denominator), denominator)
    t = (ac[..., 0] * cd[..., 1] - ac[..., 1] * cd[..., 0]) / safe
    u = (ac[..., 0] * ab[..., 1] - ac[..., 1] * ab[..., 0]) / safe
    valid = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    points = a + t[..., None] * ab
    return points.reshape(-1, 16, 2), valid.reshape(-1, 16)


def _points_inside(points: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
    """`(N, P)`: si cada punto cae dentro del cuadrilatero convexo `corners`.

    Dentro = al mismo lado de las cuatro aristas. Vale para ambas orientaciones
    porque se compara el signo entre aristas, no contra un sentido fijo.
    """
    a = corners[:, None, :, :]  # (N, 1, 4, 2)
    b = torch.roll(corners, -1, dims=1)[:, None, :, :]
    p = points[:, :, None, :]  # (N, P, 1, 2)
    side = _cross(a, b, p)  # (N, P, 4)
    return (side >= -_EPS).all(dim=-1) | (side <= _EPS).all(dim=-1)


def intersection_area(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Area de la interseccion de cada par `(N, 5)` x `(N, 5)`. Diferenciable."""
    c1, c2 = box_corners(boxes_a), box_corners(boxes_b)
    crossings, crossing_valid = _edge_intersections(c1, c2)
    inside_1 = _points_inside(c1, c2)  # esquinas de A dentro de B
    inside_2 = _points_inside(c2, c1)

    vertices = torch.cat((c1, c2, crossings), dim=1)  # (N, 24, 2)
    valid = torch.cat((inside_1, inside_2, crossing_valid), dim=1)  # (N, 24)
    count = valid.sum(dim=1, keepdim=True).clamp(min=1)

    # Orden angular alrededor del centro de los validos. Sin gradiente: el
    # orden es constante a trozos, y lo que se deriva son las coordenadas.
    with torch.no_grad():
        mask = valid[..., None].to(vertices.dtype)
        center = (vertices * mask).sum(dim=1, keepdim=True) / count[..., None]
        angles = torch.atan2(vertices[..., 1] - center[..., 1], vertices[..., 0] - center[..., 0])
        angles = torch.where(valid, angles, torch.full_like(angles, 10.0))  # invalidos al final
        order = angles.argsort(dim=1)
    sorted_vertices = torch.gather(vertices, 1, order[..., None].expand(-1, -1, 2))
    sorted_valid = torch.gather(valid, 1, order)

    # Shoelace enmascarado: cada valido con el siguiente valido, y el ultimo
    # valido cierra contra el primero.
    n = sorted_valid.sum(dim=1)  # (N,)
    index = torch.arange(24, device=vertices.device)[None, :]
    next_index = torch.where(index + 1 < n[:, None], index + 1, torch.zeros_like(index))
    next_vertices = torch.gather(sorted_vertices, 1, next_index[..., None].expand(-1, -1, 2))
    cross = (
        sorted_vertices[..., 0] * next_vertices[..., 1]
        - next_vertices[..., 0] * sorted_vertices[..., 1]
    )
    area = 0.5 * (cross * sorted_valid.to(cross.dtype)).sum(dim=1).abs()
    return torch.where(n >= 3, area, torch.zeros_like(area))


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return boxes[:, 2] * boxes[:, 3]


def rotated_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """IoU exacto por pares `(N,)`, diferenciable. Es el `PolyIoU` de DDGRCF."""
    inter = intersection_area(boxes_a, boxes_b)
    union = box_area(boxes_a) + box_area(boxes_b) - inter
    return (inter / (union + eps)).clamp(min=0.0, max=1.0)


def pairwise_rotated_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`(N, M)` de IoU exacto. Para el asignador; se llama sin gradiente."""
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return a.new_zeros((n, m))
    left = a[:, None, :].expand(n, m, 5).reshape(-1, 5)
    right = b[None, :, :].expand(n, m, 5).reshape(-1, 5)
    return rotated_iou(left, right).view(n, m)


__all__ = ['KLD_VARIANCE_DIVISOR', 'PROBIOU_VARIANCE_DIVISOR', 'bhattacharyya_distance', 'box_corners', 'box_to_gaussian', 'intersection_area', 'kld_divergence', 'kld_loss', 'pairwise_kld_loss', 'pairwise_probiou', 'pairwise_rotated_iou', 'probiou', 'rotated_iou']
