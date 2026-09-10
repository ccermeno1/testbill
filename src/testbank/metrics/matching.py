"""Emparejamiento greedy por confianza descendente, con IoU rotado.

El punto delicado no es el greedy, es que hay TRES destinos posibles para una
deteccion, no dos.

Las anotaciones que no hay que detectar ni penalizar
----------------------------------------------------
Dos mecanismos de este proyecto dejan billetes REALES fuera de la verdad:

1. La politica de visibilidad al 25%. Un billete que asoma una franja no se
   anota. Tu propia especificacion ya avisaba: *"en imagenes con abanico el
   modelo puede detectar correctamente billetes que no estan anotados y
   contaran como falsos positivos"*.

2. El filtro de area relativa. Descarta anotaciones que SI existen en el
   fichero de origen.

En los dos casos, si el detector encuentra ese billete, acierta. Contarlo como
falso positivo castiga al detector por hacer bien su trabajo y hunde la
precision justo en las imagenes con abanico, que son las que importan.

Del caso 1 no se puede hacer nada: si nadie lo anoto, no hay nada contra lo que
comparar. Del caso 2 si, porque el quad descartado lo tenemos. Entra como
`ignored`: una deteccion que cae mayoritariamente sobre uno de esos quads no
suma acierto ni fallo, simplemente sale del recuento.

El criterio es IoA, no IoU: `area(pred ∩ ignorado) / area(pred)`. Con IoU, una
deteccion del billete ENTERO contra una franja anotada de ese mismo billete
daria IoU baja y se colaria como falso positivo. Lo que se quiere preguntar es
"¿esta deteccion esta explicada por un billete real que decidimos no usar?", y
eso es una fraccion de la deteccion, no una interseccion simetrica.

`evaluate` reporta las cifras CON y SIN este descarte, porque el tamano del
efecto es en si mismo un dato sobre la calidad de la anotacion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from testbank.metrics.core import ImageEval, PolygonCache, iou

DEFAULT_MATCH_IOU = 0.5


class Outcome(str, Enum):
    TRUE_POSITIVE = "tp"
    FALSE_POSITIVE = "fp"
    IGNORED = "ignored"


@dataclass(frozen=True, slots=True)
class MatchedPair:
    prediction_index: int
    truth_index: int
    iou: float
    score: float


@dataclass(frozen=True, slots=True)
class ImageMatching:
    sample_id: str
    pairs: tuple[MatchedPair, ...]
    #: (indice de prediccion, resultado, score) para TODAS las predicciones.
    outcomes: tuple[tuple[int, Outcome, float], ...]
    #: Indices de verdades que nadie detecto. Cuentan como fallo, no se excluyen.
    missed: tuple[int, ...]

    @property
    def n_truths(self) -> int:
        return len(self.pairs) + len(self.missed)

    @property
    def detected(self) -> int:
        return len(self.pairs)

    def truth_to_prediction(self) -> dict[int, MatchedPair]:
        return {pair.truth_index: pair for pair in self.pairs}


def match_image(
    item: ImageEval,
    cache: PolygonCache | None = None,
    *,
    match_iou: float = DEFAULT_MATCH_IOU,
    use_ignored: bool = True,
) -> ImageMatching:
    """Greedy por confianza descendente. Una verdad se empareja como mucho una vez.

    Greedy y no asignacion optima a proposito: es lo que hacen las metricas de
    deteccion al uso (COCO incluido), y cambiarlo haria que nuestros numeros no
    se pudieran comparar con los publicados.
    """
    cache = cache or PolygonCache.build(item)
    order = sorted(
        range(len(item.predictions)),
        key=lambda i: item.predictions[i].score,
        reverse=True,
    )

    taken: set[int] = set()
    pairs: list[MatchedPair] = []
    outcomes: list[tuple[int, Outcome, float]] = []

    for pred_index in order:
        prediction = cache.predictions[pred_index]
        score = item.predictions[pred_index].score

        best_truth, best_iou = None, 0.0
        for truth_index, truth in enumerate(cache.truths):
            if truth_index in taken:
                continue
            value = iou(prediction, truth)
            if value > best_iou:
                best_truth, best_iou = truth_index, value

        if best_truth is not None and best_iou >= match_iou:
            taken.add(best_truth)
            pairs.append(
                MatchedPair(pred_index, best_truth, best_iou, score)
            )
            outcomes.append((pred_index, Outcome.TRUE_POSITIVE, score))
            continue

        if use_ignored and _falls_on_ignored(
            prediction, cache, threshold=match_iou
        ):
            outcomes.append((pred_index, Outcome.IGNORED, score))
            continue

        outcomes.append((pred_index, Outcome.FALSE_POSITIVE, score))

    missed = tuple(i for i in range(len(cache.truths)) if i not in taken)
    return ImageMatching(
        sample_id=item.sample_id,
        pairs=tuple(pairs),
        outcomes=tuple(outcomes),
        missed=missed,
    )


def _falls_on_ignored(prediction, cache: PolygonCache, *, threshold: float) -> bool:
    """IoA sobre la union de los quads ignorados.

    Sobre la UNION y no por pares: una deteccion repartida entre dos franjas
    contiguas, cada una al 30%, esta explicada al 60% por anotaciones reales y
    no deberia contar como falso positivo por no llegar al umbral con ninguna.
    """
    if not cache.ignored or prediction.area <= 0:
        return False
    covered = 0.0
    from shapely.ops import unary_union

    merged = unary_union(cache.ignored)
    if merged.is_empty:
        return False
    covered = prediction.intersection(merged).area
    return covered / prediction.area >= threshold


def match_all(
    items,
    caches=None,
    *,
    match_iou: float = DEFAULT_MATCH_IOU,
    use_ignored: bool = True,
) -> list[ImageMatching]:
    items = list(items)
    caches = caches or [PolygonCache.build(i) for i in items]
    return [
        match_image(item, cache, match_iou=match_iou, use_ignored=use_ignored)
        for item, cache in zip(items, caches)
    ]
