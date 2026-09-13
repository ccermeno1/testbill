"""Casi-duplicados: agrupacion, informe y manifiesto."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from testbank.data.discover import Sample
from testbank.data.duplicates import (
    DEFAULT_THRESHOLD,
    find_duplicates,
    signature,
    write_manifest,
)


def _sample(directory: Path, sample_id: str, array: np.ndarray) -> Sample:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sample_id}.png"  # PNG: sin perdida, ruido controlado
    Image.fromarray(array.astype(np.uint8)).save(path)
    return Sample(
        sample_id=sample_id,
        image_path=path,
        label_path=directory / f"{sample_id}.txt",
    )


def _noise(seed: int, size: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (size, size, 3))


def _jitter(array: np.ndarray, amount: int, seed: int) -> np.ndarray:
    """Misma imagen con una perturbacion pequena, como dos tomas seguidas."""
    delta = np.random.default_rng(seed).integers(-amount, amount + 1, array.shape)
    return np.clip(array.astype(int) + delta, 0, 255)


# --- la firma -------------------------------------------------------------


def test_la_firma_esta_normalizada(tmp_path):
    s = _sample(tmp_path, "a", _noise(1))
    vector = signature(s.image_path)
    assert vector.size == 16 * 16 * 3  # color: tres canales
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert vector.mean() == pytest.approx(0.0, abs=1e-9)


def test_una_imagen_de_un_solo_tono_no_revienta(tmp_path):
    """Norma cero: dividir daria NaN y contaminaria toda la matriz."""
    s = _sample(tmp_path, "plana", np.full((64, 64, 3), 128))
    vector = signature(s.image_path)
    assert np.all(np.isfinite(vector))
    assert np.linalg.norm(vector) == pytest.approx(0.0)


# --- la agrupacion --------------------------------------------------------


def test_dos_tomas_casi_iguales_van_al_mismo_grupo(tmp_path):
    base = _noise(1)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=2))
    report = find_duplicates([a, b])
    assert len(report.pairs) == 1
    assert report.groups["a"] == report.groups["b"]
    assert report.n_groups == 1


def test_imagenes_distintas_no_se_agrupan(tmp_path):
    samples = [_sample(tmp_path, f"s{i}", _noise(i)) for i in range(4)]
    report = find_duplicates(samples)
    assert report.pairs == []
    assert report.n_groups == 4


def test_la_agrupacion_es_transitiva(tmp_path):
    """A~B y B~C ponen a los tres juntos aunque A y C no se parezcan.

    Repartirlos dejaria la fuga a medias, que es casi lo mismo que no hacer nada.
    """
    base = _noise(7)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 30, seed=1))
    c = _sample(tmp_path, "c", _jitter(base, 60, seed=2))
    report = find_duplicates([a, b, c], threshold=0.80)
    assert report.n_groups == 1
    assert len({report.groups[i] for i in ("a", "b", "c")}) == 1


def test_la_clave_del_grupo_no_depende_del_orden(tmp_path):
    """El manifiesto tiene que salir igual se pasen como se pasen."""
    base = _noise(3)
    a = _sample(tmp_path, "aaa", base)
    b = _sample(tmp_path, "bbb", _jitter(base, 4, seed=9))
    directo = find_duplicates([a, b]).groups
    inverso = find_duplicates([b, a]).groups
    assert directo == inverso
    assert set(directo.values()) == {"aaa"}


def test_un_umbral_mas_alto_agrupa_menos(tmp_path):
    base = _noise(5)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 90, seed=4))
    assert find_duplicates([a, b], threshold=0.70).n_groups == 1
    assert find_duplicates([a, b], threshold=0.999).n_groups == 2


# --- el informe -----------------------------------------------------------


def test_se_marcan_los_pares_que_cruzan_particiones(tmp_path):
    base = _noise(11)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=1))
    c = _sample(tmp_path, "c", _noise(12))

    report = find_duplicates(
        [a, b, c], split_of={"a": "train", "b": "test", "c": "train"}
    )
    assert len(report.crossing) == 1
    assert report.crossing[0].crosses_splits
    assert report.affected == {"a", "b"}
    assert "train <-> test" in report.crossing[0].describe()


def test_un_par_dentro_de_la_misma_particion_no_cruza(tmp_path):
    base = _noise(13)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=1))
    report = find_duplicates([a, b], split_of={"a": "train", "b": "train"})
    assert len(report.pairs) == 1
    assert report.crossing == []


def test_sin_muestras_no_falla():
    report = find_duplicates([])
    assert report.n_samples == 0 and report.pairs == []


def test_el_informe_dice_que_no_reparticiona(tmp_path):
    """Quien lea el JSON tiene que saber que esto NO ha tocado la particion."""
    note = find_duplicates([_sample(tmp_path, "a", _noise(1))]).to_dict()["note"]
    assert "No re-particiona" in note
    assert "la decide el export" in note
    assert "validacion cruzada" in note


# --- el manifiesto --------------------------------------------------------


def test_el_manifiesto_tiene_el_formato_de_group_manifest(tmp_path):
    base = _noise(21)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=1))
    c = _sample(tmp_path, "c", _noise(22))

    path = write_manifest(find_duplicates([a, b, c]), tmp_path / "out" / "g.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"a", "b", "c"}
    assert data["a"] == data["b"] != data["c"]
    assert all(isinstance(v, str) for v in data.values())


def test_el_umbral_por_defecto_es_el_medido():
    assert DEFAULT_THRESHOLD == 0.90


def test_mismo_encuadre_con_distinto_color_no_es_duplicado(tmp_path):
    """REGRESION: en gris, dos fotos de stock con el mismo encuadre y billetes
    de distinto valor (un 50 naranja, un 500 morado) salian como la misma toma.
    18 de 93 pares reales eran eso. Aqui: mismo dibujo, distinto tinte."""
    base = _noise(51).astype(float)
    naranja = np.clip(base * [1.0, 0.6, 0.2], 0, 255)
    morado = np.clip(base * [0.6, 0.2, 1.0], 0, 255)
    a = _sample(tmp_path, "a", naranja)
    b = _sample(tmp_path, "b", morado)
    assert find_duplicates([a, b]).pairs == []


def test_el_desglose_por_particion_destaca_test(tmp_path):
    """El numero que mas importa es cuanto test esta contaminado.

    Regresion: sin este desglose habia que contarlo a mano sobre la lista
    truncada de pares, y asi se colo un 4 donde eran 21 en un informe.
    """
    base = _noise(31)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=1))
    c = _sample(tmp_path, "c", _noise(32))

    report = find_duplicates(
        [a, b, c], split_of={"a": "train", "b": "test", "c": "train"}
    )
    assert len(report.touching("test")) == 1
    assert report.affected_in("test") == {"b"}
    assert report.affected_in("train") == {"a"}
    assert report.touching("valid") == []

    lineas = "\n".join(report.summary_lines({"train": 2, "test": 1}))
    assert "test" in lineas and "SELLADO" in lineas
    assert lineas.index("test") < lineas.index("train"), "test va primero"


def test_el_desglose_va_tambien_al_json(tmp_path):
    base = _noise(41)
    a = _sample(tmp_path, "a", base)
    b = _sample(tmp_path, "b", _jitter(base, 4, seed=1))
    payload = find_duplicates(
        [a, b], split_of={"a": "train", "b": "test"}
    ).to_dict()
    assert payload["by_split"]["test"]["images"] == 1
    assert payload["by_split"]["valid"]["pairs"] == 0
