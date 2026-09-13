"""Bucle de entrenamiento y adaptador del candidato propio."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="el candidato propio necesita torch")

from conftest import rotated_rect_points

from testbank.config import Config
from testbank.data.discover import Sample
from testbank.dataio.formats import get as get_format
from testbank.detectors import get as get_detector
from testbank.geometry.quad import Quad, canonicalize
from testbank.models.data import build_datasets, collate, quad_to_box
from testbank.models.train import fit, learning_rate_at, load_model

SIDE = 128


def _quad(cx, cy, half_long=0.2, ratio=2.0, theta=0.0) -> Quad:
    return canonicalize(Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta)))


def _write(directory: Path, sample_id: str, quads) -> Sample:
    images = directory / "images"
    labels = directory / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    path = images / f"{sample_id}.jpg"
    Image.new("RGB", (SIDE, SIDE), (60, 60, 60)).save(path)
    writer = get_format("obb_yolo")
    label = labels / f"{sample_id}.txt"
    label.write_text(
        "\n".join(f"0 {writer.from_quad(q).payload}" for q in quads) + "\n",
        encoding="utf-8",
    )
    return Sample(sample_id=sample_id, image_path=path, label_path=label)


@pytest.fixture()
def setup(tmp_path):
    quads = (_quad(0.35, 0.5, theta=0.3), _quad(0.7, 0.5))
    samples = {"train": [_write(tmp_path / "src", f"s{i}", quads) for i in range(4)]}
    base = Config()
    config = base.model_copy(
        update={
            "data": base.data.model_copy(update={"derived_dir": tmp_path / "d"}),
            "detector": base.detector.model_copy(
                update={"epochs": 2, "image_size": SIDE, "batch_size": 2}
            ),
        }
    )
    return samples, config


# --- el dataset -----------------------------------------------------------


def test_el_quad_canonico_da_el_lado_largo_como_w():
    """No es casualidad que aprovecho: es la razon de que exista el orden
    canonico. `p0->p1` ancla el lado mas largo, asi que `w` sale siendo ese."""
    quad = _quad(0.5, 0.5, half_long=0.25, ratio=3.0, theta=0.4)
    _, _, w, h, theta = quad_to_box(quad, 400, 400)
    assert w > h
    assert w / h == pytest.approx(3.0, rel=0.02)
    assert 0.0 <= theta < math.pi


def test_el_centro_de_la_caja_es_el_centro_del_quad():
    quad = _quad(0.3, 0.7, half_long=0.15)
    cx, cy = quad_to_box(quad, 200, 200)[:2]
    assert cx == pytest.approx(0.3 * 200, abs=1.0)
    assert cy == pytest.approx(0.7 * 200, abs=1.0)


def test_el_dataset_entrega_imagen_y_cajas(setup):
    samples, config = setup
    dataset = build_datasets(samples, config)["train"]
    image, boxes, classes, sample_id = dataset[0]
    assert image.shape == (3, SIDE, SIDE)
    # BGR crudo en 0-255, la convencion de YOLOX: es lo que esperan los pesos
    # preentrenados (COCO de Megvii, DOTA de DDGRCF). Antes iba en [0, 1] y el
    # preentreno llegaba destrozado; ver `image_to_input`.
    assert image.min() >= 0.0 and image.max() <= 255.0 and image.max() > 1.0
    assert boxes.shape == (2, 5)
    assert classes.shape == (2,)
    assert sample_id == "s0"


def test_el_dataset_pasa_por_el_filtro_de_area(tmp_path):
    """Si cargara los ficheros por su cuenta, este candidato entrenaria con una
    verdad distinta de la de Ultralytics."""
    grande = _quad(0.5, 0.5, half_long=0.30)
    franja = _quad(0.5, 0.5, half_long=0.30, ratio=30.0)
    sample = _write(tmp_path / "src", "a", [grande, franja])
    base = Config()
    config = base.model_copy(
        update={
            "data": base.data.model_copy(update={"derived_dir": tmp_path / "d"}),
            "detector": base.detector.model_copy(update={"image_size": SIDE}),
        }
    )
    dataset = build_datasets({"train": [sample]}, config)["train"]
    _, boxes, _, _ = dataset[0]
    assert boxes.shape[0] == 1, "la franja filtrada no debe llegar al entrenador"


def test_el_lote_no_apila_las_cajas(setup):
    """Cada imagen tiene un numero distinto; apilarlas exigiria rellenar con
    basura que luego hay que acordarse de ignorar."""
    samples, config = setup
    dataset = build_datasets(samples, config)["train"]
    batch = collate([dataset[0], dataset[1]])
    assert batch.images.shape == (2, 3, SIDE, SIDE)
    assert isinstance(batch.boxes, list) and len(batch.boxes) == 2


# --- el planificador ------------------------------------------------------


def test_el_calentamiento_arranca_bajo():
    """Sin el, las primeras iteraciones con la cabeza recien inicializada dan
    gradientes enormes que desestabilizan la BatchNorm."""
    assert learning_rate_at(0, 1000, 1.0) < 0.1


def test_el_coseno_termina_casi_en_cero():
    assert learning_rate_at(999, 1000, 1.0) < 0.01


def test_el_maximo_esta_en_medio_del_arranque():
    valores = [learning_rate_at(s, 1000, 1.0) for s in range(1000)]
    assert max(valores) == pytest.approx(1.0, abs=0.01)
    assert valores.index(max(valores)) < 100


# --- entrenar de verdad ---------------------------------------------------


def test_entrenar_deja_pesos_y_historial(setup, tmp_path):
    samples, config = setup
    dataset = build_datasets(samples, config)["train"]
    weights, history = fit(dataset, config, output_dir=tmp_path / "out")

    assert weights.exists()
    assert len(history.epochs) == 2
    for entry in history.epochs:
        assert {"box", "angle", "objectness", "classes", "total"} <= set(entry)


def test_la_perdida_baja(setup, tmp_path):
    """La prueba minima de que el bucle aprende algo en vez de dar vueltas."""
    samples, config = setup
    config = config.model_copy(
        update={"detector": config.detector.model_copy(update={"epochs": 6})}
    )
    dataset = build_datasets(samples, config)["train"]
    _, history = fit(dataset, config, output_dir=tmp_path / "out")
    primera = history.epochs[0]["total"]
    ultima = history.epochs[-1]["total"]
    assert ultima < primera, f"no bajo: {primera:.4f} -> {ultima:.4f}"


def test_los_pesos_recuerdan_su_variante(setup, tmp_path):
    """Cargar unos pesos de nano en un tiny fallaria con un error de formas
    incomprensible; el fichero es el unico sitio que no se desincroniza."""
    samples, config = setup
    dataset = build_datasets(samples, config)["train"]
    weights, _ = fit(dataset, config, output_dir=tmp_path / "out")
    model = load_model(weights)
    assert model.variant == config.detector.variant


def test_entrenar_es_determinista(setup, tmp_path):
    """Misma semilla, misma perdida. Sin esto no hay comparacion posible."""
    samples, config = setup
    dataset = build_datasets(samples, config)["train"]
    _, a = fit(dataset, config, output_dir=tmp_path / "a")
    _, b = fit(dataset, config, output_dir=tmp_path / "b")
    assert a.epochs[-1]["total"] == pytest.approx(b.epochs[-1]["total"], rel=1e-6)


# --- el adaptador ---------------------------------------------------------


def test_esta_registrado_y_es_apto_para_produccion():
    detector = get_detector("yolox-obb-nano")
    assert detector.license == "Apache-2.0"
    assert detector.production_ready is True
    assert not detector.component().blockers("el detector")


def test_aparece_entre_los_candidatos_de_produccion():
    from testbank.detectors import production_candidates

    assert "yolox-obb-nano" in production_candidates()


def test_el_recorrido_completo_del_adaptador(setup, tmp_path):
    samples, config = setup
    detector = get_detector("yolox-obb-nano")
    result = detector.train(samples, config, output_dir=tmp_path / "train")
    assert result.weights.exists()

    predictions = detector.predict(
        samples["train"], weights=result.weights, config=config
    )
    assert set(predictions) == {s.sample_id for s in samples["train"]}
    for found in predictions.values():
        for prediction in found:
            assert 0.0 <= prediction.score <= 1.0
            assert len(prediction.quad.points) == 4


# --- una variante por candidato -------------------------------------------


def test_hay_un_candidato_por_variante():
    from testbank.detectors import detectors
    from testbank.models.yolox_obb import VARIANTS

    registrados = set(detectors())
    for variant in VARIANTS:
        assert f"yolox-obb-{variant}" in registrados


def test_el_nombre_registrado_coincide_con_la_variante():
    """Regresion: el nombre estaba fijo en "nano" mientras la variante venia de
    la config, asi que un tiny de 4.37M se registraba como el nano de 857k."""
    from testbank.detectors import get as get_detector
    from testbank.models.yolox_obb import VARIANTS

    for variant in VARIANTS:
        detector = get_detector(f"yolox-obb-{variant}")
        assert detector.variant == variant


def test_la_variante_del_candidato_manda_sobre_la_config(setup, tmp_path):
    """Y se ESCRIBE en la config, para que el config.yaml congelado no mienta."""
    samples, config = setup
    from testbank.detectors import get as get_detector
    from testbank.models.train import load_model

    detector = get_detector("yolox-obb-tiny")
    assert config.detector.variant == "nano", "la config dice otra cosa a proposito"

    result = detector.train(samples, config, output_dir=tmp_path / "t")
    assert load_model(result.weights).variant == "tiny"


def test_cada_variante_declara_sus_parametros():
    from testbank.detectors import get as get_detector

    nota = " ".join(get_detector("yolox-obb-tiny").notes)
    assert "tiny" in nota
    assert "4,366,808" in nota
