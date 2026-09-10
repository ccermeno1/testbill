"""Las dos metricas que deciden: cobertura y contaminacion.

Miden si el RECORTE sirve, que es lo unico que le importa al clasificador de
manchas aguas abajo. El mAP puede ser excelente y el recorte inservible.

Cobertura
    Fraccion del billete real que queda dentro del recorte predicho con margen.
    Un recorte que corta media mancha arruina el clasificador. Objetivo: >= 0.98
    en el percentil 5.

Contaminacion
    Fraccion del recorte que pertenece a OTRO billete. El fondo es ruido inocuo;
    un trozo del billete vecino puede meter una mancha ajena y provocar un falso
    positivo.

Un billete no detectado cuenta como cobertura 0, no se excluye. Excluirlo haria
que un detector que solo encuentra los casos faciles saliera mejor, que es la
conclusion invertida. Aun asi se reportan las tres cifras por separado -- tasa
de deteccion, condicionada a deteccion y agregada -- porque mezclarlas oculta
cual de los dos problemas tiene un candidato.

Que cuenta como "otro billete"
------------------------------
Tambien los quads que el filtro de area dejo fuera. Para puntuar la DETECCION se
ignoran, porque penalizar al detector por encontrarlos seria injusto. Para la
contaminacion no: son billetes fisicos de verdad, y si un trozo de uno entra en
el recorte, la mancha ajena entra con el. Como llevemos la contabilidad de las
anotaciones no cambia lo que hay en los pixeles.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from testbank.metrics.core import ImageEval, PolygonCache, expand, union_of
from testbank.metrics.matching import DEFAULT_MATCH_IOU, match_image

DEFAULT_MARGIN = 0.05


class SceneType(str, Enum):
    """Una imagen con un solo billete o con varios.

    La contaminacion se comporta de forma completamente distinta en cada caso, y
    resumirlas juntas oculta las dos. Ver `ContaminationConfig`.
    """

    SINGLE = "single"
    FAN = "fan"


def scene_type(item: ImageEval) -> SceneType:
    """Cuenta tambien los filtrados por area.

    Una imagen con una anotacion viva y un vecino descartado ES un abanico a
    efectos de contaminacion: el vecino sigue estando en los pixeles y sigue
    ensuciando el recorte. Clasificarla como "un billete" la mediria contra el
    umbral estricto por un billete que decidimos no usar.
    """
    return (
        SceneType.FAN
        if len(item.truths) + len(item.ignored) > 1
        else SceneType.SINGLE
    )


@dataclass(frozen=True, slots=True)
class CropSample:
    """Una verdad y lo que le paso. `detected=False` implica cobertura 0."""

    sample_id: str
    truth_index: int
    detected: bool
    coverage: float
    contamination: float | None
    scene: SceneType = SceneType.SINGLE


def crop_samples(
    item: ImageEval,
    cache: PolygonCache | None = None,
    *,
    margin: float = DEFAULT_MARGIN,
    match_iou: float = DEFAULT_MATCH_IOU,
    contaminate_with_ignored: bool = True,
) -> list[CropSample]:
    cache = cache or PolygonCache.build(item)
    matching = match_image(item, cache, match_iou=match_iou)
    by_truth = matching.truth_to_prediction()

    extra = list(cache.ignored) if contaminate_with_ignored else []
    scene = scene_type(item)

    out: list[CropSample] = []
    for truth_index, truth in enumerate(cache.truths):
        pair = by_truth.get(truth_index)
        if pair is None:
            out.append(
                CropSample(item.sample_id, truth_index, False, 0.0, None, scene)
            )
            continue

        crop = expand(cache.predictions[pair.prediction_index], margin)
        coverage = (
            truth.intersection(crop).area / truth.area if truth.area > 0 else 0.0
        )

        others = [p for i, p in enumerate(cache.truths) if i != truth_index]
        rest = union_of(others + extra)
        contamination = (
            crop.intersection(rest).area / crop.area
            if rest is not None and crop.area > 0
            else 0.0
        )
        out.append(
            CropSample(
                item.sample_id,
                truth_index,
                True,
                min(coverage, 1.0),
                min(contamination, 1.0),
                scene,
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class CropReport:
    margin: float
    n_truths: int
    detected: int
    #: Percentil bajo de cobertura sobre TODAS las verdades (no detectada = 0).
    coverage_p5_all: float
    #: El mismo percentil solo sobre las detectadas. Separado a proposito.
    coverage_p5_detected: float
    coverage_median: float
    contamination_p95: float
    contamination_median: float

    @property
    def detection_rate(self) -> float:
        return self.detected / self.n_truths if self.n_truths else 0.0

    def meets(self, target: float) -> bool:
        return self.coverage_p5_all >= target

    def to_dict(self) -> dict:
        return {
            "margin": self.margin,
            "n_truths": self.n_truths,
            "detected": self.detected,
            "detection_rate": self.detection_rate,
            "coverage_p5_all": self.coverage_p5_all,
            "coverage_p5_detected": self.coverage_p5_detected,
            "coverage_median": self.coverage_median,
            "contamination_p95": self.contamination_p95,
            "contamination_median": self.contamination_median,
        }


def summarize(
    samples: list[CropSample], *, margin: float, percentile: float = 5.0
) -> CropReport:
    if not samples:
        return CropReport(margin, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    all_coverage = np.array([s.coverage for s in samples])
    detected = [s for s in samples if s.detected]
    det_coverage = (
        np.array([s.coverage for s in detected]) if detected else np.array([0.0])
    )
    contamination = (
        np.array([s.contamination for s in detected if s.contamination is not None])
        if detected
        else np.array([0.0])
    )
    return CropReport(
        margin=margin,
        n_truths=len(samples),
        detected=len(detected),
        coverage_p5_all=float(np.percentile(all_coverage, percentile)),
        coverage_p5_detected=float(np.percentile(det_coverage, percentile)),
        coverage_median=float(np.median(all_coverage)),
        contamination_p95=float(np.percentile(contamination, 95)),
        contamination_median=float(np.median(contamination)),
    )


@dataclass(frozen=True, slots=True)
class SceneContamination:
    """Contaminacion de un tipo de escena, contra su propio umbral."""

    scene: SceneType
    n: int
    median: float
    p95: float
    threshold: float

    @property
    def passes(self) -> bool:
        """Sin muestras no se puede suspender: no hay evidencia, no hay veredicto."""
        return self.n == 0 or self.p95 <= self.threshold

    def to_dict(self) -> dict:
        return {
            "scene": self.scene.value,
            "n": self.n,
            "median": self.median,
            "p95": self.p95,
            "threshold": self.threshold,
            "passes": self.passes,
        }


def contamination_by_scene(
    samples: list[CropSample],
    *,
    single_max: float,
    fan_max: float,
    percentile: float = 95.0,
) -> dict[str, SceneContamination]:
    """Un billete y abanicos por separado, cada uno contra su umbral.

    Se mide solo sobre recortes que EXISTEN: una verdad sin detectar no produce
    recorte, asi que no tiene contaminacion que medir. Su coste ya lo paga la
    cobertura, que la cuenta como 0.
    """
    out: dict[str, SceneContamination] = {}
    for scene, threshold in (
        (SceneType.SINGLE, single_max),
        (SceneType.FAN, fan_max),
    ):
        values = np.array(
            [
                s.contamination
                for s in samples
                if s.scene is scene and s.detected and s.contamination is not None
            ]
        )
        out[scene.value] = SceneContamination(
            scene=scene,
            n=int(values.size),
            median=float(np.median(values)) if values.size else 0.0,
            p95=float(np.percentile(values, percentile)) if values.size else 0.0,
            threshold=threshold,
        )
    return out


def contamination_floor(
    items,
    *,
    margin: float = DEFAULT_MARGIN,
    percentile: float = 95.0,
) -> dict[str, float]:
    """Suelo alcanzable: contaminacion de un detector PERFECTO.

    Predice exactamente la verdad, asi que lo que quede es lo que impone la
    geometria de la anotacion y ningun candidato puede bajar de ahi. De aqui
    salen los valores por defecto de `ContaminationConfig`; vuelve a ejecutarlo
    cuando cambie el export, porque el suelo cambia con los datos.

    `items` se usa solo por su verdad y sus ignorados: las predicciones que
    traiga se descartan.
    """
    from testbank.metrics.core import ImageEval, Prediction

    collected: list[CropSample] = []
    for item in items:
        perfect = ImageEval(
            sample_id=item.sample_id,
            size=item.size,
            truths=item.truths,
            predictions=tuple(Prediction(q, 1.0) for q in item.truths),
            ignored=item.ignored,
        )
        collected.extend(crop_samples(perfect, margin=margin))

    by_scene = contamination_by_scene(
        collected, single_max=1.0, fan_max=1.0, percentile=percentile
    )
    return {name: entry.p95 for name, entry in by_scene.items()}
