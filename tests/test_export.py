"""Exportadores: arbol de ficheros, contenido y compatibilidad de formatos."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from conftest import rotated_rect_points
from PIL import Image

from testbank.config import Config, OutOfBoundsPolicy
from testbank.data.discover import Sample
from testbank.dataio import export as export_mod
from testbank.dataio.export import (
    Exporter,
    ExportError,
    export,
    exporters,
    get,
    register,
)
from testbank.dataio.formats import ImageSize, Record
from testbank.dataio.formats import get as get_format
from testbank.geometry.quad import Quad, canonicalize

SIZE = (200, 160)


def _quad(cx=0.5, cy=0.5, half_long=0.15, ratio=2.0, theta=0.0) -> Quad:
    """Rectangulo girado construido en PIXELES, como los billetes reales."""
    points = rotated_rect_points(
        cx * SIZE[0], cy * SIZE[1], half_long * SIZE[0], ratio, theta
    )
    return canonicalize(
        Quad.from_xy([(x / SIZE[0], y / SIZE[1]) for x, y in points]),
        aspect=SIZE[0] / SIZE[1],
    )


def _write(directory: Path, sample_id: str, quads) -> Sample:
    images = directory / "images"
    labels = directory / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    image_path = images / f"{sample_id}.jpg"
    Image.new("RGB", SIZE, (20, 20, 20)).save(image_path)
    writer = get_format("obb_yolo")
    label_path = labels / f"{sample_id}.txt"
    label_path.write_text(
        "\n".join(f"0 {writer.from_quad(q).payload}" for q in quads) + "\n",
        encoding="utf-8",
    )
    return Sample(sample_id=sample_id, image_path=image_path, label_path=label_path)


@pytest.fixture()
def config(tmp_path) -> Config:
    base = Config()
    return base.model_copy(
        update={"data": base.data.model_copy(update={"derived_dir": tmp_path / "d"})}
    )


@pytest.fixture()
def samples(tmp_path):
    quads = (_quad(0.3, 0.5, theta=0.4), _quad(0.7, 0.5))
    return {
        "train": [_write(tmp_path / "src", f"t{i}", quads) for i in range(3)],
        "valid": [_write(tmp_path / "src", f"v{i}", quads) for i in range(2)],
    }


# --- el registro ----------------------------------------------------------


def test_estan_los_tres_exportadores():
    assert exporters() == ["coco", "dota", "voc_xml"]


@pytest.mark.parametrize("name", ["coco", "dota", "voc_xml"])
def test_cumplen_el_protocolo(name):
    assert isinstance(get(name), Exporter)


def test_exportador_desconocido_dice_cuales_hay():
    with pytest.raises(ExportError, match="desconocido"):
        get("no_existe")


def test_registrar_un_nombre_repetido_es_error():
    with pytest.raises(ExportError, match="duplicado"):

        @register
        class Otro:
            name = "dota"
            annotation_format = "dota"
            requires_rectangles = False


# --- DOTA -----------------------------------------------------------------


def test_dota_escribe_el_arbol_que_espera(tmp_path, config, samples):
    result = export("dota", samples, config, out_dir=tmp_path / "out")
    for split, count in (("train", 3), ("valid", 2)):
        assert len(list((result.root / split / "images").glob("*.jpg"))) == count
        assert len(list((result.root / split / "labelTxt").glob("*.txt"))) == count


def test_dota_escribe_pixeles_y_categoria(tmp_path, config, samples):
    result = export("dota", samples, config, out_dir=tmp_path / "out")
    line = (result.root / "train" / "labelTxt" / "t0.txt").read_text().splitlines()[0]
    fields = line.split()
    assert len(fields) == 10
    assert fields[8] == "euro_banknote"
    assert fields[9] == "0"
    # Pixeles, no normalizado: con 200x160 los valores pasan de 1 holgadamente.
    assert max(float(v) for v in fields[:8]) > 2.0


def test_dota_va_y_vuelve(tmp_path, config, samples):
    """Sin perdida: lo escrito tiene que reconstruir el quad original."""
    original = _quad(0.3, 0.5, theta=0.4)
    result = export("dota", samples, config, out_dir=tmp_path / "out")
    line = (result.root / "train" / "labelTxt" / "t0.txt").read_text().splitlines()[0]

    size = ImageSize(*SIZE)
    vuelta = get_format("dota").to_quad(Record(0, line), size=size)
    error = max(
        math.dist((a[0] * SIZE[0], a[1] * SIZE[1]), (b[0] * SIZE[0], b[1] * SIZE[1]))
        for a, b in zip(original.points, vuelta.points)
    )
    assert error < 1e-3


# --- VOC XML --------------------------------------------------------------


def _voc_config(config):
    return config.model_copy(
        update={
            "detector": config.detector.model_copy(
                update={"out_of_bounds": OutOfBoundsPolicy.KEEP}
            )
        }
    )


def test_voc_escribe_el_arbol_que_espera(tmp_path, config, samples):
    result = export("voc_xml", samples, _voc_config(config), out_dir=tmp_path / "out")
    assert len(list((result.root / "JPEGImages").glob("*.jpg"))) == 5
    assert len(list((result.root / "Annotations").glob("*.xml"))) == 5
    # VOC llama `val` a lo que Roboflow llama `valid`.
    assert (result.root / "ImageSets" / "Main" / "train.txt").exists()
    assert (result.root / "ImageSets" / "Main" / "val.txt").exists()


def test_voc_lleva_tamano_y_robndbox(tmp_path, config, samples):
    result = export("voc_xml", samples, _voc_config(config), out_dir=tmp_path / "out")
    tree = ET.parse(result.root / "Annotations" / "t0.xml")
    assert tree.find("size/width").text == str(SIZE[0])
    assert tree.find("size/height").text == str(SIZE[1])
    objects = tree.findall("object")
    assert len(objects) == 2
    assert objects[0].find("name").text == "euro_banknote"
    assert objects[0].find("robndbox/angle") is not None


def test_voc_con_clip_se_niega_antes_de_escribir_nada(tmp_path, config, samples):
    """Recortar un rectangulo girado deja un cuadrilatero, y robndbox no lo es.

    El conversor tambien lo detecta, pero lo haria a mitad del volcado y tras
    dejar cientos de ficheros a medias.
    """
    out = tmp_path / "out"
    with pytest.raises(ExportError, match="solo representa rectangulos"):
        export("voc_xml", samples, config, out_dir=out)
    assert not any(out.rglob("*.xml"))


def test_el_error_de_voc_dice_como_arreglarlo(tmp_path, config, samples):
    with pytest.raises(ExportError, match="--out-of-bounds keep"):
        export("voc_xml", samples, config, out_dir=tmp_path / "out")


# --- COCO -----------------------------------------------------------------


def test_coco_escribe_un_json_por_particion(tmp_path, config, samples):
    result = export("coco", samples, config, out_dir=tmp_path / "out")
    for split, count in (("train", 3), ("valid", 2)):
        data = json.loads((result.root / split / "annotations.json").read_text())
        assert len(data["images"]) == count
        assert len(data["annotations"]) == count * 2
        assert data["categories"] == [{"id": 0, "name": "euro_banknote"}]
    assert result.entry_point.name == "annotations.json"


def test_coco_numera_sin_repetir(tmp_path, config, samples):
    data = json.loads(
        (
            export("coco", samples, config, out_dir=tmp_path / "out").root
            / "train"
            / "annotations.json"
        ).read_text()
    )
    ids = [a["id"] for a in data["annotations"]]
    assert len(ids) == len(set(ids))
    assert {a["image_id"] for a in data["annotations"]} == {1, 2, 3}


def test_coco_avisa_de_que_pierde_la_orientacion(tmp_path, config, samples):
    data = json.loads(
        (
            export("coco", samples, config, out_dir=tmp_path / "out").root
            / "train"
            / "annotations.json"
        ).read_text()
    )
    assert "pierde la orientacion" in data["info"]["note"]
    assert get_format("bbox_coco").lossy is True


# --- lo comun -------------------------------------------------------------


@pytest.mark.parametrize("name", ["dota", "coco"])
def test_todos_respetan_el_filtro_de_area(tmp_path, name):
    """Si un exportador se saltara el filtro, ese candidato entrenaria con una
    verdad distinta de la de los demas y la tabla los compararia como iguales."""
    grande = _quad(0.5, 0.5, half_long=0.30)
    franja = _quad(0.5, 0.5, half_long=0.30, ratio=30.0)
    sample = _write(tmp_path / "src", "a", [grande, franja])

    base = Config()
    config = base.model_copy(
        update={"data": base.data.model_copy(update={"derived_dir": tmp_path / "d"})}
    )
    result = export(name, {"train": [sample]}, config, out_dir=tmp_path / "out")
    assert result.report.dropped == 1

    if name == "dota":
        lines = (result.root / "train" / "labelTxt" / "a.txt").read_text().splitlines()
        assert len(lines) == 1
    else:
        data = json.loads((result.root / "train" / "annotations.json").read_text())
        assert len(data["annotations"]) == 1


def test_el_informe_dice_que_politica_de_borde_se_uso(tmp_path, config, samples):
    result = export("dota", samples, config, out_dir=tmp_path / "out")
    assert "borde=clip" in result.report.describe()
    assert result.report.counts == {"train": 3, "valid": 2}


def test_exportar_dos_veces_no_acumula(tmp_path, config, samples):
    export("dota", samples, config, out_dir=tmp_path / "out")
    result = export("dota", samples, config, out_dir=tmp_path / "out")
    assert len(list((result.root / "train" / "labelTxt").glob("*.txt"))) == 3


def test_anadir_un_exportador_no_toca_export(tmp_path, config, samples):
    """El registro es lo unico que decide: `export` no conoce ningun nombre."""

    @register
    class Contador:
        name = "_contador_de_prueba"
        annotation_format = "obb_yolo"
        requires_rectangles = False

        def write(self, prepared, root, *, class_names):
            target = root / "cuenta.txt"
            target.write_text(
                str(sum(len(v) for v in prepared.values())), encoding="utf-8"
            )
            return target

    try:
        result = export(
            "_contador_de_prueba", samples, config, out_dir=tmp_path / "out"
        )
        assert result.entry_point.read_text() == "5"
    finally:
        del export_mod.REGISTRY["_contador_de_prueba"]
