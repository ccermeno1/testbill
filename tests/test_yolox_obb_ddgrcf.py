"""Adaptador de `DDGRCF/YOLOX_OBB`: guardas, exp generado y aislamiento.

Como con el otro fork, aqui no se prueba el fork -- necesita un clon compilado
con MSVC y una GPU -- sino lo nuestro: que cada muro se detecte ANTES de hacer
nada y con instrucciones, y que el `Exp` generado lleve lo que tiene que llevar.
"""

from __future__ import annotations

import ast

import pytest

from testbank.config import Config
from testbank.detectors.base import REGISTRY, DetectorError
from testbank.detectors.yolox_obb_ddgrcf import (
    ENV_VAR,
    WEIGHTS_ENV_VAR,
    YoloxObbDdgrcfDetector,
    find_fork,
    pretrained_weights,
)

NAME = "yolox-obb-ddgrcf-small"


def test_esta_registrado_y_dice_lo_que_es():
    d = REGISTRY[NAME]
    assert d.license == "Apache-2.0" and d.production_ready is False
    assert any("DOTA" in n for n in d.notes)
    assert any("PolyIoU" in n for n in d.notes)
    assert any("compilad" in n for n in d.notes)


def test_sin_clon_el_error_trae_la_receta_completa(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(DetectorError) as exc:
        find_fork(tmp_path / "no")
    mensaje = str(exc.value)
    for pieza in ("git clone", "BboxToolkit", "setup.py develop", "Build Tools"):
        assert pieza in mensaje


def test_un_directorio_sin_configs_no_pasa_por_clon(tmp_path):
    (tmp_path / "yolox").mkdir()  # sin configs/: no es este fork
    with pytest.raises(DetectorError, match="no hay un clon"):
        find_fork(tmp_path)


def test_sin_pesos_de_dota_no_hay_preentreno_y_se_sabe(monkeypatch):
    monkeypatch.delenv(WEIGHTS_ENV_VAR, raising=False)
    assert pretrained_weights() is None


def test_una_ruta_de_pesos_que_no_existe_es_error(monkeypatch, tmp_path):
    monkeypatch.setenv(WEIGHTS_ENV_VAR, str(tmp_path / "no.pth"))
    with pytest.raises(DetectorError, match="no es un fichero"):
        pretrained_weights()


def test_el_exp_generado_es_valido_y_lleva_una_clase(tmp_path):
    config = Config().model_copy(
        update={"detector": Config().detector.model_copy(update={"epochs": 2, "image_size": 416})}
    )
    (tmp_path / "configs" / "modules").mkdir(parents=True)
    (tmp_path / "configs" / "losses").mkdir(parents=True)
    path = YoloxObbDdgrcfDetector()._write_exp(
        config, root=tmp_path, output_dir=tmp_path / "out", data_dir=tmp_path / "data"
    )
    texto = path.read_text(encoding="utf-8")
    ast.parse(texto)
    assert "num_classes=1" in texto
    assert 'class_names=["euro_banknote"]' in texto
    assert "yolox_losses_obb.yaml" in texto  # SU receta: PolyIoU, no la KLD
    assert "self.max_epoch = 2" in texto
    # Su exp de referencia trae 15 clases de DOTA por yaml; aqui va en linea.
    assert "dota10.yaml" not in texto


def test_el_exp_no_evalua_durante_el_entrenamiento():
    from testbank.detectors.yolox_obb_ddgrcf import _EXP_TEMPLATE

    assert "self.no_eval = True" in _EXP_TEMPLATE
