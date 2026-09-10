"""Ensambla el `metrics.json` de una ejecucion.

Orden de lectura del informe, que es el orden en que se decide:

1. `crop` -- cobertura y contaminacion. Deciden si el recorte sirve.
2. `detection` -- mAP y tasa de deteccion. Dicen si encuentra los billetes.
3. `geometry` -- angulo y vertices. Diagnostico, nunca criterio de exito.

Todo lo que decide lleva intervalo. Lo que es diagnostico no, para no dar a
entender que se esta comparando con ello.

Se empareja UNA vez por imagen y umbral, y el bootstrap remuestrea indices sobre
lo ya calculado. Emparejar dentro del bucle serian 2000 replicas x 10 umbrales x
101 imagenes de shapely, que medido no termina. Ver `detection.py`.
"""

from __future__ import annotations

import numpy as np

from testbank.config import Config
from testbank.metrics.bootstrap import bootstrap_images
from testbank.metrics.core import (
    PolygonCache,
    angle_error_deg,
    longest_side_px,
    vertex_distances_px,
)
from testbank.metrics.crop import (
    contamination_by_scene,
    crop_samples,
    summarize,
)
from testbank.metrics.detection import (
    COCO_THRESHOLDS,
    ap_from_stats,
    counts_from_stats,
    image_stats,
)
from testbank.metrics.matching import match_image


def _percentile(values, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def geometry_diagnostics(items, caches, *, match_iou: float) -> dict:
    """Angulo y distancia por vertice, solo sobre pares emparejados.

    Diagnostico, no criterio de exito: la especificacion dice que la precision
    geometrica exacta no es el objetivo, asi que estos numeros van sin intervalo
    para que nadie los use para elegir candidato.
    """
    angles: list[float] = []
    vertex_px: list[float] = []
    vertex_frac: list[float] = []

    for item, cache in zip(items, caches):
        matching = match_image(item, cache, match_iou=match_iou)
        for pair in matching.pairs:
            truth = item.truths[pair.truth_index]
            predicted = item.predictions[pair.prediction_index].quad
            angles.append(angle_error_deg(truth, predicted, item.size))
            side = longest_side_px(truth, item.size)
            for distance in vertex_distances_px(truth, predicted, item.size):
                vertex_px.append(distance)
                if side > 0:
                    vertex_frac.append(distance / side)

    return {
        "n_matched_pairs": len(angles),
        "angle_error_deg": {
            "median": _percentile(angles, 50),
            "p95": _percentile(angles, 95),
        },
        "vertex_distance_px": {
            "median": _percentile(vertex_px, 50),
            "p95": _percentile(vertex_px, 95),
        },
        "vertex_distance_fraction_of_longest_side": {
            "median": _percentile(vertex_frac, 50),
            "p95": _percentile(vertex_frac, 95),
        },
        "note": (
            "Diagnostico. Un rectangulo aproximado es la politica de anotacion, "
            "asi que la desviacion por vertice no es un fallo del detector."
        ),
    }


def evaluate(items, config: Config | None = None, *, split: str = "valid") -> dict:
    """Metricas completas de un conjunto de imagenes ya predichas."""
    config = config or Config()
    items = list(items)
    caches = [PolygonCache.build(i) for i in items]

    match_iou = config.metrics.match_iou
    samples = config.metrics.bootstrap_samples
    seed = config.metrics.seed
    indices = list(range(len(items)))

    # --- precomputo, una sola vez -----------------------------------------
    thresholds = tuple(sorted({match_iou, *COCO_THRESHOLDS}))
    stats = {
        t: image_stats(items, caches, iou_threshold=t) for t in thresholds
    }
    stats_no_ignore = image_stats(
        items, caches, iou_threshold=match_iou, use_ignored=False
    )

    crops = {}
    for margin in config.crop.margin_sweep:
        crops[margin] = [
            crop_samples(item, cache, margin=margin, match_iou=match_iou)
            for item, cache in zip(items, caches)
        ]

    # --- agregacion y bootstrap sobre indices ------------------------------
    map50 = bootstrap_images(
        indices,
        lambda picked: ap_from_stats([stats[match_iou][i] for i in picked]),
        samples=samples,
        seed=seed,
    )

    def map50_95(picked) -> float:
        values = [
            ap_from_stats([stats[t][i] for i in picked]) for t in COCO_THRESHOLDS
        ]
        finite = [v for v in values if not np.isnan(v)]
        return float(np.mean(finite)) if finite else float("nan")

    map50_95_interval = bootstrap_images(
        indices, map50_95, samples=samples, seed=seed
    )

    with_ignored = counts_from_stats(stats[match_iou])
    without_ignored = counts_from_stats(stats_no_ignore)

    crop_by_margin = {}
    for margin, per_image in crops.items():
        flat = [s for group in per_image for s in group]
        report = summarize(
            flat, margin=margin, percentile=config.metrics.coverage_percentile
        )

        def coverage_p5(picked, per_image=per_image, margin=margin) -> float:
            collected = [s for i in picked for s in per_image[i]]
            return summarize(
                collected,
                margin=margin,
                percentile=config.metrics.coverage_percentile,
            ).coverage_p5_all

        def contamination_p95(picked, per_image=per_image, margin=margin) -> float:
            collected = [s for i in picked for s in per_image[i]]
            return summarize(
                collected,
                margin=margin,
                percentile=config.metrics.coverage_percentile,
            ).contamination_p95

        by_scene = contamination_by_scene(
            flat,
            single_max=config.metrics.contamination.single_max,
            fan_max=config.metrics.contamination.fan_max,
            percentile=config.metrics.contamination.percentile,
        )

        entry = report.to_dict()
        entry["coverage_p5_all_ci"] = bootstrap_images(
            indices, coverage_p5, samples=samples, seed=seed
        ).to_dict()
        entry["contamination_p95_ci"] = bootstrap_images(
            indices, contamination_p95, samples=samples, seed=seed
        ).to_dict()
        entry["contamination_by_scene"] = {
            name: value.to_dict() for name, value in by_scene.items()
        }
        entry["meets_target"] = report.meets(config.metrics.coverage_target)
        crop_by_margin[f"{margin:.2f}"] = entry

    chosen = crop_by_margin[f"{config.crop.margin:.2f}"]
    return {
        "split": split,
        "n_images": len(items),
        "n_truths": with_ignored.n_truths,
        "crop": {
            "margin_used": config.crop.margin,
            "coverage_target": config.metrics.coverage_target,
            "coverage_percentile": config.metrics.coverage_percentile,
            "by_margin": crop_by_margin,
        },
        "coverage_p5": chosen["coverage_p5_all_ci"],
        #: Agregado sobre las dos escenas. Se conserva por continuidad, pero el
        #: veredicto son las dos entradas de abajo: mezclarlas oculta las dos.
        "contamination_p95": chosen["contamination_p95_ci"],
        "contamination": {
            "by_scene": chosen["contamination_by_scene"],
            "passes": all(
                v["passes"] for v in chosen["contamination_by_scene"].values()
            ),
            "note": (
                "Dos umbrales porque la distribucion es bimodal. En imagenes de "
                "un solo billete el suelo es cero exacto y cualquier "
                "contaminacion es un error real. En abanicos ni un detector "
                "perfecto baja del 0.89: si un billete esta parcialmente "
                "tapado, su caja contiene por fuerza pixeles del que lo tapa. "
                "Para comparar candidatos en abanicos mira la mediana, no el "
                "p95, que discrimina poco por estar casi saturado."
            ),
        },
        "detection": {
            "match_iou": match_iou,
            "with_ignored": {
                "true_positives": with_ignored.true_positives,
                "false_positives": with_ignored.false_positives,
                "ignored": with_ignored.ignored,
                "detection_rate": with_ignored.detection_rate,
            },
            "without_ignored": {
                "true_positives": without_ignored.true_positives,
                "false_positives": without_ignored.false_positives,
                "detection_rate": without_ignored.detection_rate,
            },
            "note": (
                "La diferencia de false_positives entre las dos entradas es el "
                "coste de nuestra propia politica de anotacion: detecciones "
                "correctas sobre billetes que el filtro de area dejo fuera. Un "
                "salto grande dice que hay que revisar la anotacion, no el "
                "detector."
            ),
        },
        "map50": map50.to_dict(),
        "map50_95": map50_95_interval.to_dict(),
        "geometry": geometry_diagnostics(items, caches, match_iou=match_iou),
        "bootstrap": {
            "unit": config.metrics.bootstrap_unit,
            "samples": samples,
            "seed": seed,
        },
    }
