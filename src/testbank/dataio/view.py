"""Preparacion comun de las muestras antes de escribirlas en cualquier formato.

Entre el fichero de origen y lo que ve un entrenador hay tres pasos que TIENEN
que ser los mismos para todos los formatos:

1. El filtro de area relativa (`dataio/loader.py`).
2. La politica de billetes que cruzan el borde (`clip`, `pad` o `keep`).
3. El tamano de imagen, que los formatos en pixeles necesitan.

Estaban dentro de `detectors/dataset.py`, que escribe la vista de Ultralytics.
Al anadir los exportadores de DOTA, VOC y COCO habria hecho falta repetirlos, y
una divergencia ahi seria silenciosa: dos candidatos entrenando con verdades
distintas y una tabla comparandolos como si fueran lo mismo. Asi que viven aqui
y los dos sitios llaman a lo mismo.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from testbank.config import Config, OutOfBoundsPolicy
from testbank.data.discover import Sample
from testbank.dataio.formats import ImageSize
from testbank.dataio.image_sizes import SizeIndex
from testbank.dataio.loader import load_samples
from testbank.geometry.quad import Quad, canonicalize

#: Ordenes en los que se recorren los extremos. Ver `_fit_extents`.
_TRIM_ORDERS = (
    ("u1", "u0", "v1", "v0"),
    ("v1", "v0", "u1", "u0"),
    ("u0", "u1", "v0", "v1"),
    ("v0", "v1", "u0", "u1"),
)


def _trim(u0, u1, v0, v1, axes, aspect, order, passes):
    """Gauss-Seidel sobre los cuatro extremos, en un orden dado.

    La regla de oro: **un extremo nunca cruza a su opuesto**. Si la cota que le
    toca lo exigiese, se deja como esta y que lo absorba el otro eje en la pasada
    siguiente. Sin esa guarda, el extremo se pinzaba contra su opuesto, la
    dimension colapsaba a cero y como el ajuste solo encoge, ya no habia vuelta:
    salia un quad degenerado en lugar de un rectangulo pequeno.
    """
    (ux, uy), (vx, vy) = axes
    # (coef_u, coef_v, tope) de cada restriccion del marco.
    limits = ((ux, vx, aspect), (uy, vy, 1.0))

    for index in range(passes):
        # Solo la ULTIMA pasada deja que cualquier eje absorba cualquier
        # restriccion. Antes manda la alineacion (ver `_fit_extents`).
        cheapest_only = index < passes - 1
        for which in order:
            on_u = which[0] == "u"
            low, high = -math.inf, math.inf
            others = (v0, v1) if on_u else (u0, u1)
            for coefficient_u, coefficient_v, cap in limits:
                own = coefficient_u if on_u else coefficient_v
                fixed = coefficient_v if on_u else coefficient_u
                if abs(own) < 1e-12:
                    continue
                if cheapest_only and abs(own) < abs(fixed):
                    continue
                for other in others:
                    offset = fixed * other
                    a, b = (0.0 - offset) / own, (cap - offset) / own
                    lo, hi = (a, b) if own > 0 else (b, a)
                    low, high = max(low, lo), min(high, hi)

            if which == "u1" and high > u0:
                u1 = min(u1, high)
            elif which == "u0" and low < u1:
                u0 = max(u0, low)
            elif which == "v1" and high > v0:
                v1 = min(v1, high)
            elif which == "v0" and low < v1:
                v0 = max(v0, low)
    return u0, u1, v0, v1


def _inside(u0, u1, v0, v1, axes, aspect, tolerance=1e-9):
    (ux, uy), (vx, vy) = axes
    for u in (u0, u1):
        for v in (v0, v1):
            x, y = u * ux + v * vx, u * uy + v * vy
            if not (-tolerance <= x <= aspect + tolerance):
                return False
            if not (-tolerance <= y <= 1.0 + tolerance):
                return False
    return True


def _fit_extents(u0, u1, v0, v1, axes, aspect, *, passes=6):
    """Encoge los extremos de la caja hasta que sus cuatro esquinas caben.

    Con el angulo fijo, cada esquina es una funcion LINEAL de los cuatro
    extremos, asi que cada restriccion del marco (`0 <= x <= aspect`,
    `0 <= y <= 1`) se despeja como una cota sobre el extremo que se ajusta.

    Dos cosas que hacen falta, y las dos salieron de medir, no de razonar
    ------------------------------------------------------------------
    **Cada restriccion va al eje MAS ALINEADO con ella.** Sin eso el ajuste es
    valido y aun asi desastroso. Caso real (`Multiple_Euro_154`): un billete casi
    horizontal que se sale 0.029 por arriba. Como el eje largo tiene
    `uy = 0.0315`, la restriccion `y >= 0` tambien se puede satisfacer encogiendo
    el lado LARGO... de 0.890 a 0.064. Cumplia todo y conservaba el 8% del
    billete.

    **Se prueban varios ordenes y gana el de mayor area.** Con un solo orden, el
    ascenso por coordenadas se atasca en angulos proximos a 45 grados, donde los
    dos ejes pesan casi igual en las dos restricciones, y llega a colapsar una
    dimension a cero. Cuatro ordenes cuestan nada y quitan el problema.

    Sigue sin ser el rectangulo inscrito de area maxima: eso pediria un LP, otra
    dependencia para afinar algo que ya esta dentro del ruido del recorte.
    """
    best = None
    for order in _TRIM_ORDERS:
        candidate = _trim(u0, u1, v0, v1, axes, aspect, order, passes)
        # Un candidato que no cabe no vale por mucha area que tenga: la guarda
        # anti-colapso de `_trim` puede dejar una restriccion sin satisfacer.
        if not _inside(*candidate, axes, aspect):
            continue
        a0, a1, b0, b1 = candidate
        area = (a1 - a0) * (b1 - b0)
        if best is None or area > best[0]:
            best = (area, candidate)
    return best[1] if best is not None else None


def _visible_bounds(points, aspect):
    """Envolvente ALINEADA de la parte visible. El ultimo recurso de `clip_quad`.

    Pierde el angulo, y por eso es el ultimo recurso y no la regla. A cambio es
    rectangulo y cabe en el marco por construccion, siempre.
    """
    from shapely.geometry import Polygon, box

    visible = Polygon(points).intersection(box(0.0, 0.0, aspect, 1.0))
    if visible.is_empty or visible.area <= 0.0:
        # Fuera del todo. Se pinza y que el resto del proyecto decida: el filtro
        # de area o la propia validacion del quad.
        return canonicalize(
            Quad.from_xy(
                [
                    (min(max(x / aspect, 0.0), 1.0), min(max(y, 0.0), 1.0))
                    for x, y in points
                ]
            ),
            aspect=aspect,
        )
    x0, y0, x1, y1 = visible.bounds
    return canonicalize(
        Quad.from_xy(
            [
                (x0 / aspect, y0),
                (x1 / aspect, y0),
                (x1 / aspect, y1),
                (x0 / aspect, y1),
            ]
        ),
        aspect=aspect,
    )


def clip_quad(quad: Quad, *, aspect: float = 1.0) -> Quad:
    """Recorta al marco conservando el ANGULO, y devuelve un RECTANGULO.

    Antes esto pinzaba cada vertice a [0,1] por separado. Es rapido y mantiene
    cuatro vertices, pero un rectangulo GIRADO cortado contra un marco recto no
    da un rectangulo mas pequeno: da un trapecio. Medido sobre los datos reales,
    **82 de 679 anotaciones (12%) dejaban de ser rectangulos**, y el unico run
    registrado entreno asi.

    Eso es un objetivo que el modelo no puede alcanzar: predice
    `(cx, cy, w, h, angulo)`, o sea rectangulos. Un trapecio como verdad de
    referencia solo se puede aprender como el rectangulo que menos mal le pega.

    Que hace ahora, en los ejes de la propia caja (el lado largo y su
    perpendicular):

    1. Parte de la caja original.
    2. Encoge sus cuatro extremos hasta que las cuatro esquinas caben en el
       marco, con el angulo intacto.

    Por que NO se envuelve "lo visible"
    -----------------------------------
    El primer intento fue cruzar el poligono con el marco y tomar su envolvente
    en esos mismos ejes. **No sirve, y de un modo silencioso**: cortar una
    ESQUINA deja intactas las otras tres, que son las que fijan la envolvente.
    Medido, las 99 anotaciones fuera de marco seguian fuera, hasta un 23% del
    lado. Queda escrito porque el fallo no se ve leyendo el codigo, solo
    midiendo.

    Por que se conserva el angulo
    -----------------------------
    El angulo de la anotacion es dato; el de un trapecio recortado es un
    artefacto del corte. Recalcularlo con una caja de area minima meteria ruido
    en la unica magnitud que medimos aparte del IoU.

    El `aspect` no es opcional de verdad: en coordenadas normalizadas un
    rectangulo girado es un paralelogramo, asi que "ser un rectangulo" solo
    significa algo en pixeles.
    """
    points = [(x * aspect, y) for x, y in quad.points]
    (x0, y0), (x1, y1) = points[0], points[1]
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0.0:
        return canonicalize(Quad.from_xy(quad.points), aspect=aspect)

    # El orden canonico ancla p0->p1 en el lado mas largo: esos son los ejes.
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    vx, vy = -uy, ux
    us = [px * ux + py * uy for px, py in points]
    vs = [px * vx + py * vy for px, py in points]

    extents = _fit_extents(
        min(us), max(us), min(vs), max(vs), ((ux, uy), (vx, vy)), aspect
    )
    if extents is None:
        # Ningun rectangulo con ESE angulo cabe en el marco. Pasa cuando el
        # billete esta casi entero fuera, y entonces conservar el angulo ya no
        # es lo importante: se devuelve la envolvente alineada de lo visible,
        # que es rectangulo y cabe por construccion.
        #
        # Estos casos son los que `pad` resuelve bien y `clip` no puede: en los
        # datos reales no aparece ninguno, y el filtro de area relativa se lleva
        # por delante los billetes asi de tapados.
        return _visible_bounds(points, aspect)

    u0, u1, v0, v1 = extents
    rectangle = [
        (u * ux + v * vx, u * uy + v * vy)
        for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
    ]
    return canonicalize(
        Quad.from_xy(
            # El pinzado final es contra el error de redondeo, no contra la
            # geometria: sin el, un 1.0000000002 dispara la validacion del quad.
            [(min(max(x / aspect, 0.0), 1.0), min(max(y, 0.0), 1.0))
             for x, y in rectangle]
        ),
        aspect=aspect,
    )


def pad_geometry(fraction: float) -> tuple[float, float]:
    """Escala y desplazamiento al normalizar sobre la imagen padeada.

    Con un borde de `fraction` en cada lado, el lado total se multiplica por
    `1 + 2f`, y el origen de la imagen original queda en `f` del nuevo.
    """
    scale = 1.0 + 2.0 * fraction
    return 1.0 / scale, fraction / scale


def pad_quad(quad: Quad, fraction: float) -> Quad:
    """Recoloca un quad en coordenadas de la imagen ya padeada."""
    scale, offset = pad_geometry(fraction)
    return canonicalize(
        Quad.from_xy([(x * scale + offset, y * scale + offset) for x, y in quad.points])
    )


def pad_image(source: Path, target: Path, fraction: float) -> None:
    """Borde negro. Constante y no reflejado a proposito: el borde es contenido
    inventado y tiene que PARECERLO, no imitar textura que no se fotografio."""
    import cv2

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"no se pudo leer {source}")
    height, width = image.shape[:2]
    top = bottom = round(height * fraction)
    left = right = round(width * fraction)
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), padded)


def out_of_bounds(quad: Quad) -> bool:
    return any(not (0.0 <= v <= 1.0) for v in quad.flat())


@dataclass(frozen=True, slots=True)
class PreparedSample:
    """Una muestra lista para escribir, en el formato que sea."""

    sample: Sample
    #: Tamano de la imagen QUE SE ESCRIBE. Con `pad` es el padeado, no el
    #: original: los formatos en pixeles tienen que coincidir con la imagen.
    size: ImageSize
    quads: tuple[Quad, ...]
    class_ids: tuple[int, ...]
    #: Si la imagen hay que reescribirla (padding) o basta con enlazarla.
    needs_rewrite: bool = False
    #: Cuanto padding lleva. Va AQUI y no como argumento de quien escribe: cada
    #: exportador tendria que acordarse de pasarlo, y olvidarlo con politica
    #: `pad` escribiria la imagen sin borde mientras las etiquetas si lo
    #: asumen. La muestra sabe lo que es; quien la escribe no tiene que saberlo.
    pad_fraction: float = 0.0

    @property
    def sample_id(self) -> str:
        return self.sample.sample_id


@dataclass
class PreparationReport:
    policy: str
    pad_fraction: float = 0.0
    #: Anotaciones que el filtro de area dejo fuera.
    dropped: int = 0
    #: Anotaciones que cruzaban el borde.
    adjusted: int = 0
    #: Solo con `pad`: las que seguian fuera despues de padear.
    clipped_after_pad: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        extra = ""
        if self.policy == OutOfBoundsPolicy.PAD.value:
            extra = (
                f", pad={self.pad_fraction:.0%}, "
                f"{self.clipped_after_pad} recortadas aun asi"
            )
        return (
            f"{counts}; {self.dropped} anotaciones filtradas; "
            f"borde={self.policy}, {self.adjusted} fuera del marco{extra}"
        )


def prepare(
    samples_by_split: dict[str, list],
    config: Config | None = None,
) -> tuple[dict[str, list[PreparedSample]], PreparationReport]:
    """Filtro de area + politica de borde, igual para todos los formatos."""
    config = config or Config()
    # Coercion deliberada: `Config.model_copy(update=...)` NO valida, asi que un
    # `out_of_bounds="clip"` puesto por ahi llega como str y revienta mas tarde
    # con un AttributeError sin relacion aparente.
    policy = OutOfBoundsPolicy(config.detector.out_of_bounds)
    fraction = config.detector.pad_fraction
    report = PreparationReport(
        policy=policy.value,
        pad_fraction=fraction if policy is OutOfBoundsPolicy.PAD else 0.0,
    )

    out: dict[str, list[PreparedSample]] = {}
    for split, samples in samples_by_split.items():
        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        loaded, filter_report = load_samples(
            samples,
            sizes=sizes,
            min_relative_area=config.annotation_policy.min_relative_area,
        )
        report.dropped += len(filter_report.dropped)

        by_id = {s.sample_id: s for s in samples}
        prepared: list[PreparedSample] = []
        for item in loaded:
            width, height = sizes.size(item.sample_id)
            # El aspecto va a `clip_quad`: en normalizadas un rectangulo girado
            # es un paralelogramo, y "recortar a un rectangulo" sin esto lo
            # haria en el espacio equivocado.
            aspect = width / height
            quads: list[Quad] = []
            for annotation in item.annotations:
                quad = annotation.quad
                if out_of_bounds(quad):
                    report.adjusted += 1
                    if policy is OutOfBoundsPolicy.CLIP:
                        quad = clip_quad(quad, aspect=aspect)
                if policy is OutOfBoundsPolicy.PAD:
                    # TODOS se recolocan, se salieran o no: la imagen cambio de
                    # tamano y el sistema de coordenadas con ella.
                    quad = pad_quad(quad, fraction)
                    if out_of_bounds(quad):
                        # El borde no daba para tanto. Se recorta en vez de
                        # dejarlo fuera, porque dejarlo fuera hace que el
                        # entrenador descarte la imagen ENTERA.
                        report.clipped_after_pad += 1
                        # Con padding la imagen ya es otra: el aspecto no cambia
                        # (el borde es proporcional en los dos lados), pero se
                        # pasa explicito para que no dependa de recordarlo.
                        quad = clip_quad(quad, aspect=aspect)
                quads.append(quad)

            if policy is OutOfBoundsPolicy.PAD:
                width = width + 2 * round(width * fraction)
                height = height + 2 * round(height * fraction)

            prepared.append(
                PreparedSample(
                    sample=by_id[item.sample_id],
                    size=ImageSize(int(width), int(height)),
                    quads=tuple(quads),
                    class_ids=tuple(a.class_id for a in item.annotations),
                    needs_rewrite=policy is OutOfBoundsPolicy.PAD,
                    pad_fraction=fraction if policy is OutOfBoundsPolicy.PAD else 0.0,
                )
            )
        out[split] = prepared
        report.counts[split] = len(prepared)
    return out, report


def place_image(item: PreparedSample, target: Path) -> None:
    """Padea si la muestra lo pide; si no, enlace duro y si no se puede, copia.

    El enlace simbolico seria mejor pero en Windows necesita permisos que no
    siempre hay, y fallar por eso al empezar a entrenar seria absurdo.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if item.needs_rewrite:
        pad_image(item.sample.image_path, target, item.pad_fraction)
        return
    if target.exists():
        return
    try:
        os.link(item.sample.image_path, target)
    except OSError:
        # Volumen distinto, sistema de ficheros sin enlaces duros, o permisos.
        shutil.copy2(item.sample.image_path, target)
