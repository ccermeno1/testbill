"""Filtro de area relativa: conserva el billete de delante de cada imagen."""

from __future__ import annotations

import hashlib

import pytest
from conftest import rotated_rect_points

from testbank.checks.visibility import check_samples
from testbank.data.discover import Sample
from testbank.dataio.obb_yolo import Annotation
from testbank.dataio.prepare import (
    DEFAULT_MIN_RELATIVE_AREA,
    filter_by_relative_area,
    load_sample,
    load_samples,
)
from testbank.geometry.quad import Quad, canonicalize


def quad(cx, cy, half_long=0.2, ratio=2.0, theta=0.0) -> Quad:
    return canonicalize(Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta)))


def ann(*args, **kwargs) -> Annotation:
    return Annotation(class_id=0, quad=quad(*args, **kwargs))


def write_sample(tmp_path, quads, sample_id="img") -> Sample:
    """Escribe un .txt obb_yolo y una imagen ficticia emparejada."""
    label = tmp_path / f"{sample_id}.txt"
    lines = [
        "0 " + " ".join(f"{v:.17g}" for v in q.flat()) for q in quads
    ]
    label.write_text("\n".join(lines) + "\n", encoding="utf-8")
    image = tmp_path / f"{sample_id}.jpg"
    image.write_bytes(b"")
    return Sample(sample_id=sample_id, image_path=image, label_path=label)


# --- el criterio ----------------------------------------------------------


def test_una_sola_anotacion_siempre_se_conserva():
    kept, dropped = filter_by_relative_area([ann(0.5, 0.5)])
    assert len(kept) == 1 and dropped == ()


def test_sin_anotaciones_no_falla():
    assert filter_by_relative_area([]) == ((), ())


def test_billetes_de_tamano_similar_se_conservan_todos():
    """Foto de dos o tres billetes juntos: ninguno es una franja tapada."""
    anns = [ann(0.25, 0.5, half_long=0.20), ann(0.75, 0.5, half_long=0.19)]
    kept, dropped = filter_by_relative_area(anns)
    assert len(kept) == 2 and dropped == ()


def test_franja_de_billete_tapado_cae():
    """Abanico: la franja visible del de detras es mucho menor que el de delante."""
    frente = ann(0.5, 0.5, half_long=0.30, ratio=2.0)
    franja = ann(0.5, 0.5, half_long=0.30, ratio=20.0)  # misma longitud, muy fina
    kept, dropped = filter_by_relative_area([frente, franja])
    assert len(kept) == 1
    assert [index for index, _ in dropped] == [1]
    assert dropped[0][1] == pytest.approx(0.1, abs=0.01)


def test_la_mayor_nunca_se_descarta_ni_con_umbral_uno():
    anns = [ann(0.3, 0.3, half_long=0.10), ann(0.7, 0.7, half_long=0.30)]
    kept, dropped = filter_by_relative_area(anns, min_relative_area=1.0)
    assert len(kept) == 1
    assert kept[0][0] == 1  # la mayor, con su indice original
    assert [index for index, _ in dropped] == [0]


def test_umbral_cero_no_descarta_nada():
    anns = [ann(0.5, 0.5, half_long=0.30), ann(0.5, 0.5, half_long=0.30, ratio=40.0)]
    kept, dropped = filter_by_relative_area(anns, min_relative_area=0.0)
    assert len(kept) == 2 and dropped == ()


def test_umbral_fuera_de_rango_es_error():
    with pytest.raises(ValueError, match="min_relative_area"):
        filter_by_relative_area([ann(0.5, 0.5)], min_relative_area=1.5)


def test_el_umbral_por_defecto_es_el_de_la_politica():
    assert DEFAULT_MIN_RELATIVE_AREA == 0.25


def test_el_umbral_separa_por_encima_y_por_debajo():
    """A cada lado del umbral la decision es la esperada.

    No se comprueba la igualdad EXACTA con el umbral, y no por comodidad: los
    vertices se ajustan a la rejilla diadica de `quad.py`, asi que un area
    construida para dar 0.25 clavado da 0.2499999998. Igual que con la
    involucion del volteo, el borde exacto no es representable. Tampoco
    importa: un cociente de areas a 1e-9 del umbral es arbitrario en cualquier
    caso, porque la anotacion de origen es un "rectangulo aproximado".
    """
    grande = ann(0.5, 0.5, half_long=0.20, ratio=2.0)
    # El area es proporcional a half_long^2 / ratio.
    encima = ann(0.5, 0.5, half_long=0.11, ratio=2.0)  # ~0.30 relativo
    kept, dropped = filter_by_relative_area([grande, encima], min_relative_area=0.25)
    assert len(kept) == 2 and dropped == ()

    debajo = ann(0.5, 0.5, half_long=0.09, ratio=2.0)  # ~0.20 relativo
    kept, dropped = filter_by_relative_area([grande, debajo], min_relative_area=0.25)
    assert [index for index, _ in dropped] == [1]


# --- los indices apuntan al fichero, no a la lista filtrada ----------------


def test_los_indices_conservados_son_los_del_fichero(tmp_path):
    """Si cae la anotacion 0, la siguiente sigue siendo la #1, no la #0.

    Un informe que renumerase mandaria a abrir la anotacion equivocada en
    Roboflow, que es justo para lo que sirve el informe.
    """
    sample = write_sample(
        tmp_path,
        [
            quad(0.5, 0.5, half_long=0.30, ratio=30.0),  # franja, cae
            quad(0.25, 0.5, half_long=0.20),
            quad(0.75, 0.5, half_long=0.20),
        ],
    )
    loaded = load_sample(sample)
    assert loaded.kept_indices == (1, 2)
    assert [d.index for d in loaded.dropped] == [0]


def test_el_informe_lleva_imagen_indice_y_area_relativa(tmp_path):
    sample = write_sample(
        tmp_path,
        [
            quad(0.5, 0.5, half_long=0.30, ratio=2.0),
            quad(0.5, 0.5, half_long=0.30, ratio=20.0),
        ],
        sample_id="abanico",
    )
    loaded = load_sample(sample)
    assert len(loaded.dropped) == 1
    dropped = loaded.dropped[0]
    assert dropped.sample_id == "abanico"
    assert dropped.index == 1
    assert 0.0 < dropped.relative_area < 0.25
    assert "abanico" in dropped.describe() and "#1" in dropped.describe()


# --- filtro, no borrado ---------------------------------------------------


def test_el_fichero_de_origen_no_se_toca(tmp_path):
    sample = write_sample(
        tmp_path,
        [
            quad(0.5, 0.5, half_long=0.30, ratio=2.0),
            quad(0.5, 0.5, half_long=0.30, ratio=20.0),
        ],
    )
    before = hashlib.sha256(sample.label_path.read_bytes()).hexdigest()
    load_sample(sample, min_relative_area=0.9)
    after = hashlib.sha256(sample.label_path.read_bytes()).hexdigest()
    assert before == after


def test_subir_el_umbral_no_requiere_reexportar(tmp_path):
    """El mismo fichero da conjuntos distintos segun el parametro."""
    sample = write_sample(
        tmp_path,
        [
            quad(0.25, 0.5, half_long=0.20),
            quad(0.75, 0.5, half_long=0.12),
        ],
    )
    assert len(load_sample(sample, min_relative_area=0.0).annotations) == 2
    assert len(load_sample(sample, min_relative_area=0.9).annotations) == 1


# --- agregado -------------------------------------------------------------


def test_el_informe_agregado_cuadra(tmp_path):
    a = write_sample(
        tmp_path,
        [quad(0.5, 0.5, half_long=0.30), quad(0.5, 0.5, half_long=0.30, ratio=25.0)],
        sample_id="a",
    )
    b = write_sample(
        tmp_path, [quad(0.5, 0.5, half_long=0.20)], sample_id="b"
    )
    loaded, report = load_samples([a, b])
    assert report.images_loaded == 2
    assert report.annotations_read == 3
    assert report.annotations_kept == 2
    assert report.images_affected == 1
    assert len(loaded) == 2

    payload = report.to_dict()
    assert payload["annotations_dropped"] == 1
    assert payload["dropped"][0]["sample_id"] == "a"
    assert payload["dropped"][0]["index"] == 1
    assert "no se ha borrado nada" in payload["note"]


# --- integracion con el chequeo de visibilidad ----------------------------


def test_check_samples_devuelve_los_dos_informes(tmp_path):
    sample = write_sample(
        tmp_path,
        [quad(0.5, 0.5, half_long=0.30), quad(0.5, 0.5, half_long=0.30, ratio=25.0)],
    )
    report, filter_report = check_samples([sample])
    assert filter_report.annotations_read == 2
    assert filter_report.annotations_kept == 1
    # Con una sola anotacion viva no queda nada que pueda taparse.
    assert report.annotations_checked == 1
    assert report.findings == []


def test_los_hallazgos_de_visibilidad_numeran_sobre_el_fichero(tmp_path):
    """La #0 la filtra el area; las dos que quedan siguen siendo #1 y #2.

    Sin traducir indices, el hallazgo diria #0 y mandaria a revisar justo la
    anotacion que el filtro ya habia descartado.
    """
    sample = write_sample(
        tmp_path,
        [
            quad(0.5, 0.1, half_long=0.30, ratio=25.0),  # franja suelta, cae
            # Area relativa 0.44: sobrevive al filtro, pero queda dentro de la
            # siguiente, asi que el chequeo de visibilidad si la marca.
            quad(0.5, 0.6, half_long=0.20, ratio=2.0),
            quad(0.5, 0.6, half_long=0.30, ratio=2.0),
        ],
    )
    report, filter_report = check_samples([sample], min_relative_area=0.25)
    assert [d.index for d in filter_report.dropped] == [0]
    assert [f.annotation_index for f in report.findings] == [1]
    assert report.findings[0].dominant_index == 2
