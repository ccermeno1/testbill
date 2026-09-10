"""Protocolo Detector, registro, aislamiento de Ultralytics y vista derivada."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import rotated_rect_points
from PIL import Image

from testbank.config import Config
from testbank.data.discover import Sample
from testbank.dataio.formats import get as get_format
from testbank.detectors import base as detector_base
from testbank.detectors import dataset as dataset_view
from testbank.detectors.base import (
    BaseDetector,
    Detector,
    DetectorError,
    build_image_evals,
    detectors,
    get,
    production_candidates,
    register,
)
from testbank.geometry.quad import Quad, canonicalize
from testbank.metrics.core import Prediction

SRC = Path(__file__).resolve().parents[1] / "src" / "testbank"
ADAPTER = SRC / "detectors" / "ultralytics_obb.py"

ULTRALYTICS_NAME = "ultralytics-yolo-obb"


# --- el registro ----------------------------------------------------------


def test_ultralytics_esta_registrado():
    assert ULTRALYTICS_NAME in detectors()


def test_ultralytics_es_agpl_y_no_apto_para_produccion():
    detector = get(ULTRALYTICS_NAME)
    assert detector.license == "AGPL-3.0"
    assert detector.production_ready is False


def test_no_aparece_entre_los_candidatos_de_produccion():
    """Sirve de referencia de rendimiento, nunca de candidato."""
    assert ULTRALYTICS_NAME not in production_candidates()


def test_cumple_el_protocolo():
    assert isinstance(get(ULTRALYTICS_NAME), Detector)


def test_el_componente_lleva_la_licencia_a_la_ejecucion():
    component = get(ULTRALYTICS_NAME).component()
    assert component.license == "AGPL-3.0"
    assert component.blockers("el detector")


def test_detector_desconocido_dice_cuales_hay():
    with pytest.raises(DetectorError, match="desconocido"):
        get("no_existe")


def test_registrar_dos_veces_el_mismo_nombre_es_error():
    with pytest.raises(DetectorError, match="duplicado"):

        @register
        class Otro(BaseDetector):
            name = ULTRALYTICS_NAME


def test_un_detector_sin_nombre_no_se_registra():
    with pytest.raises(DetectorError, match="name"):

        @register
        class SinNombre(BaseDetector):
            pass


# --- aislamiento ----------------------------------------------------------


def _imports_ultralytics(path: Path) -> bool:
    """Busca imports de verdad, no menciones.

    Un grep marcaria los comentarios que explican por que `val` se llama asi o
    el nombre del directorio derivado, y el test dejaria de significar nada.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "ultralytics" for a in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "ultralytics"
        ):
            return True
    return False


def test_aislamiento_de_ultralytics():
    """Ningun modulo fuera del adaptador importa `ultralytics`."""
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if path != ADAPTER and _imports_ultralytics(path)
    ]
    assert offenders == [], (
        "estos modulos importan ultralytics fuera del adaptador: " + str(offenders)
    )


def test_el_adaptador_si_lo_importa():
    """Guarda del guarda: si el adaptador dejara de importarlo, el test de
    aislamiento pasaria por vacio y no estaria comprobando nada."""
    assert _imports_ultralytics(ADAPTER)


def test_importar_testbank_no_arrastra_ultralytics():
    """El import es perezoso: el paquete es opcional y AGPL."""
    code = (
        "import sys; import testbank.detectors; "
        "print('ultralytics' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is not None,
    reason="ultralytics esta instalado en este entorno",
)
def test_sin_ultralytics_el_error_explica_por_que_es_opcional():
    """`find_spec`, no `sys.modules`: lo que importa es si esta INSTALADO.

    Mirar `sys.modules` comprobaba si estaba importado, que con el import
    perezoso es siempre falso, asi que el test se ejecutaba tambien con el
    paquete instalado y fallaba.
    """
    from testbank.detectors.ultralytics_obb import _import_ultralytics

    with pytest.raises(DetectorError, match="AGPL-3.0"):
        _import_ultralytics()


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None,
    reason="ultralytics no esta instalado",
)
def test_con_ultralytics_instalado_el_import_perezoso_funciona():
    """La cara opuesta del test de arriba: uno de los dos corre siempre."""
    from testbank.detectors.ultralytics_obb import _import_ultralytics

    assert _import_ultralytics() is not None


# --- la vista derivada ----------------------------------------------------


def _write_sample(directory: Path, sample_id: str, quads, size=(160, 120)) -> Sample:
    images = directory / "images"
    labels = directory / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    image_path = images / f"{sample_id}.jpg"
    Image.new("RGB", size, (30, 30, 30)).save(image_path)
    writer = get_format("obb_yolo")
    label_path = labels / f"{sample_id}.txt"
    label_path.write_text(
        "\n".join(f"0 {writer.from_quad(q).payload}" for q in quads) + "\n",
        encoding="utf-8",
    )
    return Sample(sample_id=sample_id, image_path=image_path, label_path=label_path)


def _quad(cx, cy, half_long=0.2, ratio=2.0, theta=0.0) -> Quad:
    return canonicalize(Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta)))


@pytest.fixture()
def config(tmp_path) -> Config:
    base = Config()
    return base.model_copy(
        update={"data": base.data.model_copy(update={"derived_dir": tmp_path / "derived"})}
    )


def test_la_vista_derivada_escribe_las_etiquetas_ya_filtradas(tmp_path, config):
    """Entrenar con la verdad sin filtrar y medir contra la filtrada no vale."""
    grande = _quad(0.5, 0.5, half_long=0.30)
    franja = _quad(0.5, 0.5, half_long=0.30, ratio=30.0)
    sample = _write_sample(tmp_path / "src", "a", [grande, franja])

    view = dataset_view.materialize({"train": [sample]}, config, out_dir=tmp_path / "out")

    escritas = (view.root / "train" / "labels" / "a.txt").read_text().strip().splitlines()
    assert len(escritas) == 1, "la franja filtrada no debe llegar al entrenador"
    assert view.dropped == 1
    assert view.counts == {"train": 1}


def test_la_vista_derivada_no_toca_el_origen(tmp_path, config):
    grande = _quad(0.5, 0.5, half_long=0.30)
    franja = _quad(0.5, 0.5, half_long=0.30, ratio=30.0)
    sample = _write_sample(tmp_path / "src", "a", [grande, franja])
    antes = hashlib.sha256(sample.label_path.read_bytes()).hexdigest()

    dataset_view.materialize({"train": [sample]}, config, out_dir=tmp_path / "out")

    assert hashlib.sha256(sample.label_path.read_bytes()).hexdigest() == antes


def test_el_data_yaml_usa_val_apuntando_a_valid(tmp_path, config):
    import yaml

    sample = _write_sample(tmp_path / "src", "a", [_quad(0.5, 0.5)])
    view = dataset_view.materialize(
        {"train": [sample], "valid": [sample]}, config, out_dir=tmp_path / "out"
    )
    data = yaml.safe_load(view.data_yaml.read_text(encoding="utf-8"))
    assert data["val"] == "valid/images"
    assert data["names"] == {0: "euro_banknote"}


def test_la_imagen_llega_a_la_vista(tmp_path, config):
    sample = _write_sample(tmp_path / "src", "a", [_quad(0.5, 0.5)])
    view = dataset_view.materialize({"train": [sample]}, config, out_dir=tmp_path / "out")
    assert (view.root / "train" / "images" / "a.jpg").exists()


# --- evaluate es comun a todos los candidatos -----------------------------


class _PerfectDetector(BaseDetector):
    """Predice exactamente la verdad ya filtrada. Solo para probar `evaluate`."""

    name = "_perfecto_de_prueba"
    license = "Apache-2.0"
    production_ready = True

    def __init__(self, truths: dict) -> None:
        self._truths = truths

    def predict(self, samples, *, weights, config):
        return {
            s.sample_id: [Prediction(q, 0.9) for q in self._truths[s.sample_id]]
            for s in samples
        }


def test_evaluate_lo_pone_la_base_y_usa_nuestras_metricas(tmp_path, config):
    """Si cada adaptador trajera el suyo, las filas no serian comparables."""
    quads = [_quad(0.3, 0.5, half_long=0.12), _quad(0.7, 0.5, half_long=0.12)]
    samples = [_write_sample(tmp_path / "src", f"s{i}", quads) for i in range(3)]

    fast = config.model_copy(
        update={"metrics": config.metrics.model_copy(update={"bootstrap_samples": 40})}
    )
    detector = _PerfectDetector({s.sample_id: quads for s in samples})
    report = detector.evaluate(samples, fast, weights=tmp_path / "no-se-usa.pt")

    assert report["map50"]["value"] == pytest.approx(1.0, abs=1e-6)
    assert report["n_images"] == 3
    assert report["bootstrap"]["unit"] == "image"


def test_la_verdad_de_evaluate_pasa_por_el_filtro(tmp_path, config):
    """La franja filtrada no es verdad que haya que detectar, pero tampoco fondo."""
    grande = _quad(0.5, 0.5, half_long=0.30)
    franja = _quad(0.5, 0.5, half_long=0.30, ratio=30.0)
    sample = _write_sample(tmp_path / "src", "a", [grande, franja])

    items = build_image_evals([sample], {"a": []}, config=config)
    assert len(items[0].truths) == 1
    assert len(items[0].ignored) == 1


def test_el_registro_es_lo_unico_que_decide():
    """`get` no conoce ningun nombre: todo sale del diccionario."""
    assert set(detectors()) == set(detector_base.REGISTRY)


# --- billetes que cruzan el borde -----------------------------------------


def _cfg(config, **detector):
    return config.model_copy(
        update={"detector": config.detector.model_copy(update=detector)}
    )


def _fuera_del_marco() -> Quad:
    """Billete que se sale por la izquierda: vertices en x negativo."""
    return canonicalize(Quad.from_xy([(-0.15, 0.3), (0.5, 0.3), (0.5, 0.7), (-0.15, 0.7)]))


def _coords(path: Path) -> list[float]:
    return [float(t) for t in path.read_text().split()[1:]]


def test_clip_pega_el_quad_al_marco(tmp_path, config):
    sample = _write_sample(tmp_path / "src", "a", [_fuera_del_marco()])
    view = dataset_view.materialize(
        {"train": [sample]}, _cfg(config, out_of_bounds="clip"), out_dir=tmp_path / "out"
    )
    coords = _coords(view.root / "train" / "labels" / "a.txt")
    assert min(coords) >= 0.0 and max(coords) <= 1.0
    assert view.adjusted == 1
    assert view.out_of_bounds == "clip"


def test_keep_no_toca_nada_pero_lo_cuenta(tmp_path, config):
    """Con `keep` Ultralytics descartara la imagen; al menos queda registrado."""
    sample = _write_sample(tmp_path / "src", "a", [_fuera_del_marco()])
    view = dataset_view.materialize(
        {"train": [sample]}, _cfg(config, out_of_bounds="keep"), out_dir=tmp_path / "out"
    )
    assert min(_coords(view.root / "train" / "labels" / "a.txt")) < 0.0
    assert view.adjusted == 1


def test_pad_mete_el_quad_dentro_y_agranda_la_imagen(tmp_path, config):
    sample = _write_sample(tmp_path / "src", "a", [_fuera_del_marco()], size=(160, 120))
    view = dataset_view.materialize(
        {"train": [sample]},
        _cfg(config, out_of_bounds="pad", pad_fraction=0.25),
        out_dir=tmp_path / "out",
    )
    coords = _coords(view.root / "train" / "labels" / "a.txt")
    assert min(coords) >= 0.0 and max(coords) <= 1.0

    with Image.open(view.root / "train" / "images" / "a.jpg") as im:
        assert im.size == (160 + 2 * 40, 120 + 2 * 30)


def test_pad_y_su_inversa_se_cancelan():
    """Si no fueran inversas exactas, las predicciones saldrian desplazadas."""
    from testbank.detectors.dataset import pad_quad
    from testbank.detectors.ultralytics_obb import _unpad

    original = _quad(0.4, 0.5, half_long=0.2, ratio=2.0, theta=0.6)
    ida = pad_quad(original, 0.25)
    vuelta = _unpad(Prediction(ida, 0.9), 0.25).quad
    for (x0, y0), (x1, y1) in zip(original.points, vuelta.points):
        assert x1 == pytest.approx(x0, abs=1e-6)
        assert y1 == pytest.approx(y0, abs=1e-6)


def test_la_inversa_conserva_lo_que_sale_del_marco():
    """Es el unico motivo de existir de `pad`: si se recortara al volver, daria
    igual que clip y el coste del padding no compraria nada."""
    from testbank.detectors.dataset import pad_quad
    from testbank.detectors.ultralytics_obb import _unpad

    fuera = _fuera_del_marco()
    vuelta = _unpad(Prediction(pad_quad(fuera, 0.25), 0.9), 0.25).quad
    assert min(vuelta.flat()) < 0.0


def test_la_politica_por_defecto_es_clip(config):
    assert config.detector.out_of_bounds.value == "clip"


def test_pad_recorta_lo_que_el_borde_no_alcanza(tmp_path, config):
    """Sin este recorte, `pad` con fraccion pequena perderia la imagen entera.

    Es lo que permite padear POCO: el borde cubre el caso comun y el recorte se
    ocupa del residuo, en vez de tener que padear para el peor caso.
    """
    muy_fuera = canonicalize(
        Quad.from_xy([(-0.40, 0.3), (0.5, 0.3), (0.5, 0.7), (-0.40, 0.7)])
    )
    sample = _write_sample(tmp_path / "src", "a", [muy_fuera])
    view = dataset_view.materialize(
        {"train": [sample]},
        _cfg(config, out_of_bounds="pad", pad_fraction=0.05),
        out_dir=tmp_path / "out",
    )
    coords = _coords(view.root / "train" / "labels" / "a.txt")
    assert min(coords) >= 0.0 and max(coords) <= 1.0
    assert view.clipped_after_pad == 1


def test_con_padding_de_sobra_no_hace_falta_recortar(tmp_path, config):
    sample = _write_sample(tmp_path / "src", "a", [_fuera_del_marco()])
    view = dataset_view.materialize(
        {"train": [sample]},
        _cfg(config, out_of_bounds="pad", pad_fraction=0.25),
        out_dir=tmp_path / "out",
    )
    assert view.clipped_after_pad == 0
