"""Metricas: IoU rotado, emparejamiento, cobertura, contaminacion e intervalos."""

from __future__ import annotations

import math

import pytest
from conftest import rotated_rect_points

from testbank.config import Config, MetricsConfig
from testbank.dataio.formats import ImageSize
from testbank.geometry.quad import Quad, canonicalize
from testbank.metrics.bootstrap import bootstrap_images
from testbank.metrics.core import (
    ImageEval,
    Prediction,
    angle_error_deg,
    expand,
    iou,
    to_polygon,
)
from testbank.metrics.crop import crop_samples, summarize
from testbank.metrics.detection import (
    average_precision,
    counts,
    mean_average_precision,
)
from testbank.metrics.evaluate import evaluate
from testbank.metrics.matching import Outcome, match_image

SIZE = ImageSize(640, 480)


def quad(cx=0.5, cy=0.5, half_long=0.15, ratio=2.0, theta=0.0) -> Quad:
    """Rectangulo girado construido en PIXELES, como los billetes reales."""
    points = rotated_rect_points(
        cx * SIZE.width, cy * SIZE.height, half_long * SIZE.width, ratio, theta
    )
    return canonicalize(
        Quad.from_xy([(x / SIZE.width, y / SIZE.height) for x, y in points]),
        aspect=SIZE.aspect,
    )


def image(truths=(), predictions=(), ignored=(), sample_id="img") -> ImageEval:
    return ImageEval(sample_id, SIZE, tuple(truths), tuple(predictions), tuple(ignored))


def perfect(truths, score=0.9):
    return [Prediction(q, score) for q in truths]


# --- angulo ---------------------------------------------------------------


def test_179_grados_contra_1_grado_da_2():
    """El ejemplo textual de la especificacion."""
    a = quad(theta=math.radians(1))
    b = quad(theta=math.radians(179))
    assert angle_error_deg(a, b, SIZE) == pytest.approx(2.0, abs=0.01)


def test_el_error_de_angulo_nunca_pasa_de_90():
    for degrees in range(0, 360, 7):
        error = angle_error_deg(quad(), quad(theta=math.radians(degrees)), SIZE)
        assert 0.0 <= error <= 90.0 + 1e-9


# --- IoU rotado -----------------------------------------------------------


def test_el_mismo_quad_da_iou_1():
    polygon = to_polygon(quad(), SIZE)
    assert iou(polygon, polygon) == pytest.approx(1.0)


def test_quads_separados_dan_iou_0():
    a = to_polygon(quad(cx=0.15, cy=0.2), SIZE)
    b = to_polygon(quad(cx=0.85, cy=0.8), SIZE)
    assert iou(a, b) == 0.0


def test_iou_conocido_de_dos_cuadrados_desplazados():
    a = Quad.from_xy([(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)])
    b = Quad.from_xy([(0.1, 0.0), (0.3, 0.0), (0.3, 0.2), (0.1, 0.2)])
    # Solapan la mitad: interseccion 1/2, union 3/2 -> IoU = 1/3.
    assert iou(to_polygon(a, SIZE), to_polygon(b, SIZE)) == pytest.approx(1 / 3)


# --- emparejamiento -------------------------------------------------------


def test_una_verdad_se_empareja_como_mucho_una_vez():
    truth = quad()
    item = image([truth], [Prediction(truth, 0.9), Prediction(truth, 0.8)])
    matching = match_image(item)
    assert len(matching.pairs) == 1
    results = [o for _, o, _ in matching.outcomes]
    assert results.count(Outcome.TRUE_POSITIVE) == 1
    assert results.count(Outcome.FALSE_POSITIVE) == 1


def test_la_de_mas_confianza_elige_primero():
    truth = quad()
    peor = Prediction(truth, 0.5)
    mejor = Prediction(quad(cx=0.51), 0.95)
    matching = match_image(image([truth], [peor, mejor]))
    assert matching.pairs[0].prediction_index == 1


def test_una_verdad_sin_detectar_cuenta_como_fallo():
    matching = match_image(image([quad(), quad(cx=0.85, cy=0.8)], []))
    assert matching.missed == (0, 1)
    assert matching.detected == 0
    assert matching.n_truths == 2


# --- las anotaciones ignoradas -------------------------------------------


def test_detectar_un_billete_filtrado_no_es_falso_positivo():
    """El filtro de area quito ese billete; encontrarlo es acertar, no fallar."""
    truth = quad(cx=0.25)
    filtrado = quad(cx=0.75)
    item = image([truth], perfect([truth, filtrado]), ignored=[filtrado])
    results = [o for _, o, _ in match_image(item).outcomes]
    assert results.count(Outcome.IGNORED) == 1
    assert results.count(Outcome.FALSE_POSITIVE) == 0


def test_sin_el_descarte_esa_misma_deteccion_es_falso_positivo():
    """Reportar las dos cifras es lo que mide el coste de nuestra politica."""
    truth = quad(cx=0.25)
    filtrado = quad(cx=0.75)
    item = image([truth], perfect([truth, filtrado]), ignored=[filtrado])
    results = [o for _, o, _ in match_image(item, use_ignored=False).outcomes]
    assert results.count(Outcome.FALSE_POSITIVE) == 1


def test_una_deteccion_en_el_fondo_sigue_siendo_falso_positivo():
    """Ignorar no es un salvoconducto: solo cubre los quads declarados."""
    truth = quad(cx=0.2, cy=0.2)
    item = image([truth], perfect([truth, quad(cx=0.8, cy=0.8)]), ignored=[])
    results = [o for _, o, _ in match_image(item).outcomes]
    assert results.count(Outcome.FALSE_POSITIVE) == 1


# --- cobertura y contaminacion -------------------------------------------


def test_una_prediccion_perfecta_cubre_del_todo():
    truth = quad()
    sample = crop_samples(image([truth], perfect([truth])), margin=0.0)[0]
    assert sample.detected
    assert sample.coverage == pytest.approx(1.0, abs=1e-6)


def test_un_billete_no_detectado_cuenta_como_cobertura_cero():
    samples = crop_samples(image([quad()], []))
    assert samples[0].detected is False
    assert samples[0].coverage == 0.0


def test_encontrar_solo_lo_facil_no_puede_salir_mejor():
    """Excluir lo no detectado invertiria la conclusion. Aqui no se excluye.

    El cauto detecta un billete de dos, perfecto. El completo detecta los dos,
    uno de ellos algo peor. Agregado, el completo tiene que ganar.
    """
    facil, dificil = quad(cx=0.25), quad(cx=0.75)
    cauto = image([facil, dificil], perfect([facil]))
    completo = image(
        [facil, dificil], [Prediction(facil, 0.9), Prediction(quad(cx=0.76), 0.7)]
    )
    resumen_cauto = summarize(crop_samples(cauto), margin=0.05)
    resumen_completo = summarize(crop_samples(completo), margin=0.05)

    assert resumen_cauto.coverage_p5_detected >= resumen_completo.coverage_p5_detected
    assert resumen_completo.coverage_p5_all > resumen_cauto.coverage_p5_all
    assert resumen_completo.detection_rate > resumen_cauto.detection_rate


def test_un_billete_aislado_no_se_contamina():
    truth = quad(cx=0.5, cy=0.5, half_long=0.1)
    sample = crop_samples(image([truth], perfect([truth])), margin=0.0)[0]
    assert sample.contamination == pytest.approx(0.0, abs=1e-9)


#: Vecino a la derecha que NO solapa con el billete de 0.35, y prediccion algo
#: mayor que si lo alcanza. Es el caso realista: la caja se pasa de ancha y se
#: come el borde del billete de al lado.
_A = {"cx": 0.35, "cy": 0.5, "half_long": 0.12}
_VECINO = {"cx": 0.61, "cy": 0.5, "half_long": 0.12}
_PREDICCION_ANCHA = {"cx": 0.35, "cy": 0.5, "half_long": 0.155}


def test_el_billete_vecino_contamina_el_recorte():
    a = quad(**_A)
    b = quad(**_VECINO)
    ancha = Prediction(quad(**_PREDICCION_ANCHA), 0.9)
    samples = crop_samples(image([a, b], [ancha]), margin=0.0)

    del_a = next(s for s in samples if s.truth_index == 0)
    assert del_a.detected, "la prediccion tiene que emparejar para poder contaminar"
    assert del_a.contamination > 0.0


def test_un_billete_filtrado_si_contamina():
    """Para puntuar deteccion se ignora; para contaminar no: son pixeles reales."""
    truth = quad(**_A)
    filtrado = quad(**_VECINO)
    ancha = Prediction(quad(**_PREDICCION_ANCHA), 0.9)

    con = crop_samples(
        image([truth], [ancha], ignored=[filtrado]),
        margin=0.0,
        contaminate_with_ignored=True,
    )[0]
    sin = crop_samples(
        image([truth], [ancha], ignored=[filtrado]),
        margin=0.0,
        contaminate_with_ignored=False,
    )[0]
    assert con.detected and sin.detected
    assert con.contamination > 0.0
    assert sin.contamination == pytest.approx(0.0, abs=1e-9)


def test_el_margen_intercambia_cobertura_por_contaminacion():
    """Es el motivo de que el margen vaya en la config y se barra."""
    a = quad(cx=0.40, cy=0.5, half_long=0.12)
    b = quad(cx=0.60, cy=0.5, half_long=0.12)
    item = image([a, b], perfect([a, b]))

    estrecho = summarize(crop_samples(item, margin=0.0), margin=0.0)
    ancho = summarize(crop_samples(item, margin=0.30), margin=0.30)
    assert ancho.contamination_p95 > estrecho.contamination_p95


def test_expandir_conserva_el_cuadrilatero():
    """Se escala, no se dilata: `buffer` redondearia las esquinas."""
    polygon = to_polygon(quad(), SIZE)
    grown = expand(polygon, 0.1)
    assert len(grown.exterior.coords) == len(polygon.exterior.coords)
    assert grown.area > polygon.area


def test_margen_negativo_es_error():
    with pytest.raises(ValueError, match="margen"):
        expand(to_polygon(quad(), SIZE), -0.1)


# --- AP -------------------------------------------------------------------


def test_prediccion_perfecta_da_ap_1():
    items = [image([quad(cx=0.3)], perfect([quad(cx=0.3)]), sample_id="a")]
    assert average_precision(items) == pytest.approx(1.0, abs=1e-6)


def test_sin_predicciones_el_ap_es_cero():
    assert average_precision([image([quad()], [])]) == 0.0


def test_map50_95_no_supera_a_map50():
    truth = quad()
    items = [image([truth], [Prediction(quad(cx=0.505), 0.9)])]
    assert mean_average_precision(items) <= average_precision(items) + 1e-9


def test_el_ranking_es_global_no_por_imagen():
    """Una imagen facil no debe pesar lo mismo que un abanico de seis."""
    facil = image([quad()], perfect([quad()]), sample_id="facil")
    abanico_truths = [quad(cx=0.2 + 0.12 * i, half_long=0.05) for i in range(6)]
    abanico = image(abanico_truths, perfect(abanico_truths[:1]), sample_id="abanico")
    counted = counts([facil, abanico])
    assert counted.n_truths == 7
    assert counted.true_positives == 2
    assert counted.detection_rate == pytest.approx(2 / 7)


# --- intervalos -----------------------------------------------------------


def test_el_bootstrap_es_determinista():
    units = list(range(30))
    stat = lambda xs: sum(xs) / len(xs)
    a = bootstrap_images(units, stat, samples=200, seed=7)
    b = bootstrap_images(units, stat, samples=200, seed=7)
    assert (a.value, a.ci_low, a.ci_high) == (b.value, b.ci_low, b.ci_high)


def test_semillas_distintas_dan_intervalos_distintos():
    units = list(range(30))
    stat = lambda xs: sum(xs) / len(xs)
    a = bootstrap_images(units, stat, samples=200, seed=1)
    b = bootstrap_images(units, stat, samples=200, seed=2)
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


def test_con_una_sola_imagen_el_intervalo_es_nan_no_cero():
    """Un intervalo de anchura cero seria mentir sobre la incertidumbre."""
    result = bootstrap_images([1.0], lambda xs: xs[0], samples=100, seed=1)
    assert result.n == 1
    assert math.isnan(result.ci_low) and math.isnan(result.ci_high)


def test_el_intervalo_contiene_al_punto():
    units = [float(i) for i in range(40)]
    stat = lambda xs: sum(xs) / len(xs)
    result = bootstrap_images(units, stat, samples=500, seed=3)
    assert result.ci_low <= result.value <= result.ci_high


def test_la_unidad_de_bootstrap_solo_puede_ser_imagen():
    """Remuestrear detecciones estrecha los intervalos artificialmente."""
    with pytest.raises(ValueError, match="bootstrap"):
        MetricsConfig(bootstrap_unit="detection")


# --- el informe completo --------------------------------------------------


def _fast_config() -> Config:
    base = Config()
    return base.model_copy(
        update={"metrics": base.metrics.model_copy(update={"bootstrap_samples": 50})}
    )


def test_el_informe_tiene_lo_que_decide_y_lo_que_diagnostica():
    truths = [quad(cx=0.3), quad(cx=0.7)]
    items = [
        image(truths, perfect(truths), sample_id=f"img{i}") for i in range(4)
    ]
    report = evaluate(items, _fast_config())

    assert report["n_images"] == 4 and report["n_truths"] == 8
    assert report["map50"]["value"] == pytest.approx(1.0, abs=1e-6)
    assert report["coverage_p5"]["n"] == 4
    assert "0.05" in report["crop"]["by_margin"]
    assert report["crop"]["by_margin"]["0.05"]["meets_target"] is True
    assert report["geometry"]["n_matched_pairs"] == 8
    assert report["bootstrap"]["unit"] == "image"


def test_el_informe_separa_los_falsos_positivos_con_y_sin_descarte():
    truth = quad(cx=0.25)
    filtrado = quad(cx=0.75)
    items = [
        image([truth], perfect([truth, filtrado]), ignored=[filtrado], sample_id=f"i{i}")
        for i in range(3)
    ]
    report = evaluate(items, _fast_config())
    detection = report["detection"]
    assert detection["with_ignored"]["false_positives"] == 0
    assert detection["with_ignored"]["ignored"] == 3
    assert detection["without_ignored"]["false_positives"] == 3


# --- el precomputo no puede cambiar el numero -----------------------------


def test_el_ap_precomputado_coincide_con_el_directo():
    """El bootstrap remuestrea sobre `image_stats`, no vuelve a emparejar.

    Es valido porque emparejar es LOCAL a cada imagen: que otras imagenes
    entren o no en la replica no cambia el resultado de esta. Si eso dejara de
    ser cierto, este test lo caza.
    """
    from testbank.metrics.detection import ap_from_stats, image_stats

    items = [
        image(
            [quad(cx=0.3), quad(cx=0.7)],
            [Prediction(quad(cx=0.31), 0.9), Prediction(quad(cx=0.85, cy=0.8), 0.4)],
            sample_id=f"img{i}",
        )
        for i in range(5)
    ]
    stats = image_stats(items)

    # Un subconjunto cualquiera, como el que produciria una replica bootstrap.
    picked = [0, 2, 2, 4]
    directo = average_precision([items[i] for i in picked])
    precomputado = ap_from_stats([stats[i] for i in picked])
    assert precomputado == pytest.approx(directo)


def test_el_orden_de_las_imagenes_no_cambia_el_ap():
    """El ranking es global por confianza, asi que barajar imagenes da igual."""
    from testbank.metrics.detection import ap_from_stats, image_stats

    items = [
        image([quad(cx=0.3)], [Prediction(quad(cx=0.3 + 0.01 * i), 0.5 + 0.08 * i)],
              sample_id=f"img{i}")
        for i in range(5)
    ]
    stats = image_stats(items)
    assert ap_from_stats(stats) == pytest.approx(ap_from_stats(list(reversed(stats))))


# --- los dos umbrales de contaminacion ------------------------------------


def test_una_imagen_de_un_billete_es_escena_single():
    from testbank.metrics.crop import SceneType, scene_type

    assert scene_type(image([quad()], [])) is SceneType.SINGLE


def test_un_vecino_filtrado_ya_convierte_la_escena_en_abanico():
    """Sigue en los pixeles y sigue ensuciando, aunque no sea verdad viva."""
    from testbank.metrics.crop import SceneType, scene_type

    item = image([quad(cx=0.3)], [], ignored=[quad(cx=0.7)])
    assert scene_type(item) is SceneType.FAN


def test_cada_escena_se_mide_contra_su_umbral():
    from testbank.metrics.crop import contamination_by_scene

    limpia = image([quad(cx=0.3)], perfect([quad(cx=0.3)]), sample_id="sola")
    a, b = quad(**_A), quad(**_VECINO)
    sucia = image([a, b], [Prediction(quad(**_PREDICCION_ANCHA), 0.9)], sample_id="fan")

    muestras = crop_samples(limpia, margin=0.05) + crop_samples(sucia, margin=0.05)
    by_scene = contamination_by_scene(muestras, single_max=0.01, fan_max=0.92)

    assert by_scene["single"].threshold == 0.01
    assert by_scene["fan"].threshold == 0.92
    assert by_scene["single"].p95 == pytest.approx(0.0, abs=1e-9)
    assert by_scene["single"].passes


def test_sin_muestras_de_una_escena_no_se_suspende():
    """Sin evidencia no hay veredicto; suspender por defecto seria inventar."""
    from testbank.metrics.crop import contamination_by_scene

    muestras = crop_samples(image([quad()], perfect([quad()])), margin=0.0)
    by_scene = contamination_by_scene(muestras, single_max=0.01, fan_max=0.92)
    assert by_scene["fan"].n == 0
    assert by_scene["fan"].passes is True


def test_el_umbral_de_abanico_no_puede_ser_mas_estricto_que_el_de_un_billete():
    from testbank.config import ContaminationConfig

    with pytest.raises(ValueError, match="fan_max"):
        ContaminationConfig(single_max=0.5, fan_max=0.1)


def test_el_suelo_se_puede_rederivar_de_los_datos():
    """De aqui salieron los valores por defecto; hay que repetirlo si cambia el export."""
    from testbank.metrics.crop import contamination_floor

    a, b = quad(**_A), quad(**_VECINO)
    items = [
        image([quad(cx=0.5)], [], sample_id="sola"),
        image([a, b], [], sample_id="fan"),
    ]
    floor = contamination_floor(items, margin=0.05)
    assert floor["single"] == pytest.approx(0.0, abs=1e-9)
    assert floor["fan"] >= 0.0


def test_el_informe_lleva_las_dos_escenas_con_su_veredicto():
    truths = [quad(cx=0.3), quad(cx=0.7)]
    items = [image(truths, perfect(truths), sample_id=f"i{i}") for i in range(3)]
    report = evaluate(items, _fast_config())
    by_scene = report["contamination"]["by_scene"]
    assert set(by_scene) == {"single", "fan"}
    assert by_scene["fan"]["threshold"] == 0.92
    assert by_scene["single"]["threshold"] == 0.01
    assert "passes" in report["contamination"]


# --- los dos umbrales de confianza ----------------------------------------


def test_los_recuentos_se_dan_a_las_dos_confianzas():
    """A 0.01 el recuento mide cuantas cajas dejo pasar el NMS, no calidad.

    La cola de baja confianza es imprescindible para el AP y ruinosa como
    titular, asi que se reportan las dos cifras etiquetadas.
    """
    truth = quad(cx=0.3)
    items = [
        image(
            [truth],
            [
                Prediction(truth, 0.9),                    # acierto
                Prediction(quad(cx=0.8, cy=0.8), 0.02),    # cola de ruido
                Prediction(quad(cx=0.8, cy=0.2), 0.03),    # cola de ruido
            ],
            sample_id=f"i{i}",
        )
        for i in range(3)
    ]
    report = evaluate(items, _fast_config())
    detection = report["detection"]

    assert detection["decision_confidence"] == 0.25
    assert detection["with_ignored"]["false_positives"] == 6
    assert detection["at_decision_confidence"]["false_positives"] == 0
    assert detection["at_decision_confidence"]["true_positives"] == 3
    # El AP se calcula con la cola entera y no se ve afectado por el reporte.
    assert report["map50"]["value"] == pytest.approx(1.0, abs=1e-6)


def test_los_falsos_positivos_de_la_cola_no_hunden_el_ap():
    """Regresion: con `np.interp` y recall repetido, el AP salia hundido.

    Tres aciertos perfectos alcanzan recall 1.0 con precision 1.0. Los falsos
    positivos posteriores no pueden bajar el AP: la precision en un recall dado
    es la MEJOR alcanzable a ese recall o mas alla, y ese maximo ya se logro.
    """
    truth = quad(cx=0.3)
    limpio = [image([truth], [Prediction(truth, 0.9)], sample_id=f"i{i}") for i in range(3)]
    con_cola = [
        image(
            [truth],
            [Prediction(truth, 0.9)] + [
                Prediction(quad(cx=0.8, cy=0.2 + 0.15 * k), 0.02) for k in range(4)
            ],
            sample_id=f"i{i}",
        )
        for i in range(3)
    ]
    assert average_precision(limpio) == pytest.approx(1.0, abs=1e-9)
    assert average_precision(con_cola) == pytest.approx(1.0, abs=1e-9)
