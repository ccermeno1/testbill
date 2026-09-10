"""mAP50 y mAP50-95 con IoU rotado, acumulando sobre TODAS las imagenes.

Interpolacion de 101 puntos, como COCO. Es una eleccion, no un detalle: la
alternativa (todos los puntos, VOC 2010+) da numeros ligeramente distintos, y si
vamos a comparar contra el mAP publicado de RTMDet-R o de Ultralytics, tenemos
que calcularlo como ellos.

El ranking es GLOBAL por confianza, no por imagen. Promediar el AP de cada
imagen es una metrica distinta y da otro numero: una imagen con un solo billete
facil pesaria lo mismo que un abanico de seis.

Por que hay una fase de precomputo
----------------------------------
El bootstrap remuestrea IMAGENES 2000 veces, y el mAP50-95 recorre 10 umbrales.
Emparejar dentro del bucle serian 2000 x 10 x 101 emparejamientos con shapely:
medido, no termina en un tiempo util.

Pero el emparejamiento de una imagen NO depende de que otras imagenes hayan
salido en el remuestreo -- es local a la imagen. Solo el ranking global y el
recuento de verdades dependen del conjunto. Asi que se empareja una vez por
imagen y umbral, se guarda `(score, acierto)` por deteccion, y cada replica del
bootstrap se reduce a concatenar y ordenar. Mismo numero, tres ordenes de
magnitud mas rapido.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from testbank.metrics.core import PolygonCache
from testbank.metrics.matching import DEFAULT_MATCH_IOU, Outcome, match_all

#: Los diez umbrales de COCO: 0.50, 0.55, ..., 0.95.
COCO_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))

#: Puntos de recall donde se interpola la precision.
_RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True, slots=True)
class ImageDetectionStats:
    """Emparejamiento de UNA imagen a UN umbral, ya reducido a lo que usa el AP.

    `scored` lleva `(score, acierto)` por deteccion, sin las ignoradas: ni suman
    acierto ni fallo, asi que salen del recuento por completo.
    """

    sample_id: str
    n_truths: int
    scored: tuple[tuple[float, int], ...]
    true_positives: int
    false_positives: int
    ignored: int


@dataclass(frozen=True, slots=True)
class DetectionCounts:
    true_positives: int
    false_positives: int
    ignored: int
    n_truths: int

    @property
    def detection_rate(self) -> float:
        return self.true_positives / self.n_truths if self.n_truths else 0.0


def image_stats(
    items,
    caches=None,
    *,
    iou_threshold: float = DEFAULT_MATCH_IOU,
    use_ignored: bool = True,
) -> list[ImageDetectionStats]:
    """Empareja una vez y guarda lo justo. Es lo que el bootstrap remuestrea."""
    items = list(items)
    caches = caches or [PolygonCache.build(i) for i in items]
    matchings = match_all(
        items, caches, match_iou=iou_threshold, use_ignored=use_ignored
    )
    out = []
    for matching in matchings:
        scored, tp, fp, ign = [], 0, 0, 0
        for _, outcome, score in matching.outcomes:
            if outcome is Outcome.IGNORED:
                ign += 1
                continue
            hit = 1 if outcome is Outcome.TRUE_POSITIVE else 0
            tp += hit
            fp += 1 - hit
            scored.append((score, hit))
        out.append(
            ImageDetectionStats(
                sample_id=matching.sample_id,
                n_truths=matching.n_truths,
                scored=tuple(scored),
                true_positives=tp,
                false_positives=fp,
                ignored=ign,
            )
        )
    return out


def ap_from_stats(stats) -> float:
    """AP sobre un conjunto ya emparejado. Solo concatena, ordena y acumula."""
    stats = list(stats)
    n_truths = sum(s.n_truths for s in stats)
    if n_truths == 0:
        return float("nan")

    scored = [pair for s in stats for pair in s.scored]
    if not scored:
        return 0.0

    scored.sort(key=lambda pair: pair[0], reverse=True)
    hits = np.array([hit for _, hit in scored], dtype=np.float64)
    tp = np.cumsum(hits)
    fp = np.cumsum(1.0 - hits)
    recall = tp / n_truths
    precision = tp / np.maximum(tp + fp, 1e-12)

    # Envolvente monotona decreciente: la precision en un recall dado es la
    # mejor alcanzable a ese recall o mas alla.
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    # `searchsorted`, no `np.interp`. Con falsos positivos DESPUES del ultimo
    # acierto -- que es lo normal con la confianza de inferencia baja -- el
    # recall se queda clavado y el vector tiene valores repetidos. `np.interp`
    # con x duplicadas devuelve la ULTIMA, que es la precision mas baja del
    # tramo, y hunde el AP. Aqui se toma la PRIMERA posicion con recall >= r,
    # que junto con la envolvente da el maximo de la cola, que es la definicion.
    positions = np.searchsorted(recall, _RECALL_POINTS, side="left")
    interpolated = np.where(
        positions < precision.size, precision[np.minimum(positions, precision.size - 1)], 0.0
    )
    return float(interpolated.mean())


def counts_from_stats(stats) -> DetectionCounts:
    stats = list(stats)
    return DetectionCounts(
        true_positives=sum(s.true_positives for s in stats),
        false_positives=sum(s.false_positives for s in stats),
        ignored=sum(s.ignored for s in stats),
        n_truths=sum(s.n_truths for s in stats),
    )


# --- fachadas de conveniencia ---------------------------------------------


def average_precision(
    items,
    caches=None,
    *,
    iou_threshold: float = DEFAULT_MATCH_IOU,
    use_ignored: bool = True,
) -> float:
    return ap_from_stats(
        image_stats(
            items, caches, iou_threshold=iou_threshold, use_ignored=use_ignored
        )
    )


def mean_average_precision(
    items,
    caches=None,
    *,
    thresholds=COCO_THRESHOLDS,
    use_ignored: bool = True,
) -> float:
    values = [
        average_precision(items, caches, iou_threshold=t, use_ignored=use_ignored)
        for t in thresholds
    ]
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def counts(
    items,
    caches=None,
    *,
    iou_threshold: float = DEFAULT_MATCH_IOU,
    use_ignored: bool = True,
) -> DetectionCounts:
    return counts_from_stats(
        image_stats(
            items, caches, iou_threshold=iou_threshold, use_ignored=use_ignored
        )
    )
