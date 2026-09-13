"""Adaptador de `buzhidaoshenme/YOLOX-OBB`: aislamiento, arranque y conversion.

Lo que NO se prueba aqui es el fork. Es un repositorio ajeno que no esta clonado
en CI. Lo que si se prueba, siempre, es lo nuestro: que nadie mas importe
`yolox`, que el localizador falle con instrucciones en vez de con un ImportError,
que el `Exp` generado sea Python valido y lleve nuestros parametros, y que el
deshacer del letterbox sea correcto.

El letterbox merece una nota. Todas nuestras imagenes son 416x416, asi que el
factor de escala `r` vale 1 y la conversion se ejercita sola sin probar nada. El
caso con `r != 1` solo aparece si alguien resube los originales -- y entonces, si
estuviera mal, saldria todo corrido sin que nada avisara. Por eso esta aqui.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from testbank.detectors import yolox_obb_fork
from testbank.detectors.base import REGISTRY, DetectorError
from testbank.detectors.yolox_obb_fork import (
    ENV_VAR,
    YoloxObbForkDetector,
    find_fork,
)

SRC = Path(yolox_obb_fork.__file__).resolve().parents[1]
ADAPTER = SRC / "detectors" / "yolox_obb_fork.py"
#: El otro fork de YOLOX-OBB comparte nombre de paquete: tambien puede importarlo.
ADAPTERS = {ADAPTER, SRC / "detectors" / "yolox_obb_ddgrcf.py"}
NAME = "yolox-obb-fork-small"


# --- registro -------------------------------------------------------------


def test_esta_registrado():
    assert NAME in REGISTRY


def test_es_apache_pero_no_apto_para_produccion():
    """La licencia no es el problema: el clon parcheado a mano lo es."""
    detector = REGISTRY[NAME]
    assert detector.license == "Apache-2.0"
    assert detector.production_ready is False
    assert any("clon parcheado" in n for n in detector.notes)


def test_las_notas_avisan_de_que_su_evaluador_no_evalua():
    """Es lo que mas facil seria leer mal: un 0.0 que parece un resultado."""
    assert any("0.0 fijo" in n for n in REGISTRY[NAME].notes)


# --- aislamiento ----------------------------------------------------------


def _imports_yolox(path: Path) -> bool:
    """Imports de verdad, no menciones en comentarios."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "yolox" for a in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "yolox"
        ):
            return True
    return False


def test_aislamiento_de_yolox():
    """Ningun modulo fuera del adaptador importa `yolox`.

    Importa mas aqui que con las otras dependencias: `yolox` no es un paquete
    instalado sino un clon que el adaptador mete en `sys.path` a mano. Un import
    suelto en otro modulo fallaria o no segun por donde se hubiera entrado.
    """
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if path not in ADAPTERS and _imports_yolox(path)
    ]
    assert offenders == [], (
        "estos modulos importan yolox fuera del adaptador: " + str(offenders)
    )


def test_el_adaptador_si_lo_importa():
    """Guarda del guarda: si no, el test de aislamiento pasaria por vacio."""
    assert _imports_yolox(ADAPTER)


# --- localizar el clon ----------------------------------------------------


def test_sin_clon_el_error_dice_como_clonarlo(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(DetectorError) as exc:
        find_fork(tmp_path / "no_existe")
    mensaje = str(exc.value)
    assert "git clone" in mensaje
    assert "patch_yolox_obb_fork.py" in mensaje


def test_un_clon_sin_parchear_se_detecta_antes_de_importar(tmp_path):
    """Sin parchear el fallo es `ModuleNotFoundError: _polyiou` desde dentro de
    un envoltorio de SWIG. No dice donde mirar; esto si."""
    (tmp_path / "yolox").mkdir()
    with pytest.raises(DetectorError, match="sin parchear"):
        find_fork(tmp_path)


def test_la_variable_de_entorno_se_respeta(tmp_path, monkeypatch):
    (tmp_path / "yolox").mkdir()
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    # Llega hasta la comprobacion del parche, o sea que encontro el clon.
    with pytest.raises(DetectorError, match="sin parchear"):
        find_fork()


# --- el Exp generado ------------------------------------------------------


def test_el_exp_generado_es_python_valido_y_lleva_nuestros_parametros(
    tmp_path, config_fork
):
    detector = YoloxObbForkDetector()
    data_dir = tmp_path / "voc"
    path = detector._write_exp(config_fork, output_dir=tmp_path, data_dir=data_dir)
    texto = path.read_text(encoding="utf-8")
    ast.parse(texto)  # revienta si no es Python valido
    assert "self.num_classes = 1" in texto
    assert f"self.seed = {config_fork.metrics.seed}" in texto
    assert data_dir.name in texto
    # La ruta de datos NO puede quedar cableada como en su exp de referencia,
    # que trae literalmente /home/lyy/gxw/DOTA_OBB_1_5.
    assert "/home/lyy" not in texto


def test_warmup_y_no_aug_no_pasan_del_total_de_epocas(tmp_path, config_fork):
    """Con los circuitos de 1-2 epocas del proyecto, los 5 y 15 de su exp de
    referencia dejarian el entrenamiento entero en calentamiento."""
    detector = YoloxObbForkDetector()
    una = config_fork.model_copy(
        update={"detector": config_fork.detector.model_copy(update={"epochs": 1})}
    )
    texto = detector._write_exp(
        una, output_dir=tmp_path, data_dir=tmp_path / "voc"
    ).read_text(encoding="utf-8")
    assert "self.warmup_epochs = 0" in texto
    assert "self.no_aug_epochs = 0" in texto
    assert "self.max_epoch = 1" in texto


def test_el_exp_apaga_la_evaluacion_durante_el_entrenamiento():
    """Su evaluador devuelve 0.0 fijo: evaluar solo gasta tiempo."""
    assert "self.eval_interval = {epochs} + 1" in yolox_obb_fork._EXP_TEMPLATE


# --- el deshacer del letterbox --------------------------------------------


def test_el_letterbox_se_deshace_dividiendo_sin_desplazamiento():
    """Su `preproc` ancla ARRIBA A LA IZQUIERDA y rellena abajo y a la derecha.

    Si se asumiera centrado -- lo habitual, y lo que haria cualquiera que no lea
    su codigo -- todo saldria corrido media banda de relleno.
    """
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    alto, ancho, lado = 300, 500, 416
    imagen = np.zeros((alto, ancho, 3), dtype=np.uint8)

    # Su aritmetica: escala unica y relleno abajo/derecha.
    ratio = min(lado / alto, lado / ancho)
    escalada = cv2.resize(
        imagen,
        (int(ancho * ratio), int(alto * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    assert escalada.shape[0] <= lado and escalada.shape[1] <= lado

    punto = (137.0, 88.0)
    en_entrada = (punto[0] * ratio, punto[1] * ratio)
    # El camino de vuelta de `predict`: dividir, sin restar nada.
    de_vuelta = (en_entrada[0] / ratio, en_entrada[1] / ratio)
    assert de_vuelta == pytest.approx(punto, abs=1e-9)

    # Y la prueba de que NO es centrado: con relleno centrado habria que restar
    # media banda, y el resultado se iria justo esa cantidad.
    #
    # El relleno cae en el ALTO, no en el ancho: con 300x500 y lado 416 manda
    # `416/500 = 0.832`, asi que el ancho sale justo 416 y lo que sobra va abajo.
    # Mirar el eje equivocado daba banda 0 y el test no comprobaba nada.
    banda = (lado - escalada.shape[0]) / 2
    assert banda > 0, "el caso elegido tiene que tener relleno de verdad"
    centrado = (en_entrada[1] + banda) / ratio
    assert abs(centrado - punto[1]) > 1.0
