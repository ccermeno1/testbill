"""Conversores de formato: registro, ida y vuelta, y perdida donde toca."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET

import pytest
from conftest import rotated_rect_points

from testbank.dataio import formats
from testbank.dataio.formats import (
    DEFAULT_CLASS_NAMES,
    Converter,
    FormatError,
    ImageSize,
    Record,
    convert,
    get,
    register,
)
from testbank.geometry.quad import Quad, QuadShapeWarning, canonicalize

SIZE = ImageSize(width=640, height=480)

LOSSLESS = ("obb_yolo", "dota", "voc_xml")
LOSSY = ("bbox_yolo", "bbox_coco")


def quad(cx=0.5, cy=0.5, half_long=0.2, ratio=2.0, theta=0.0) -> Quad:
    """Rectangulo girado en PIXELES, devuelto en normalizado.

    Construirlo en normalizado seria un error sutil: normalizar divide x por el
    ancho e y por el alto, que es un escalado ANISOTROPO, asi que un rectangulo
    girado en normalizado es un PARALELOGRAMO en pixeles y viceversa. Los
    billetes son rectangulos en la imagen, que es el espacio de pixeles, y ahi
    los medimos: las 762 anotaciones del export dan 90.0000 grados en pixeles.
    """
    half_long_px = half_long * SIZE.width
    points_px = rotated_rect_points(
        cx * SIZE.width, cy * SIZE.height, half_long_px, ratio, theta
    )
    return canonicalize(
        Quad.from_xy(
            [(x / SIZE.width, y / SIZE.height) for x, y in points_px]
        ),
        aspect=SIZE.aspect,
    )


def max_pixel_error(a: Quad, b: Quad) -> float:
    return max(
        math.dist(
            (pa[0] * SIZE.width, pa[1] * SIZE.height),
            (pb[0] * SIZE.width, pb[1] * SIZE.height),
        )
        for pa, pb in zip(a.points, b.points)
    )


# --- el registro ----------------------------------------------------------


def test_estan_todos_los_formatos():
    assert formats.formats() == [
        "bbox_coco",
        "bbox_yolo",
        "dota",
        "obb_yolo",
        "voc_xml",
        "yolox_obb_voc",
    ]


@pytest.mark.parametrize("name", LOSSLESS + LOSSY)
def test_todos_cumplen_el_protocolo(name):
    assert isinstance(get(name), Converter)


@pytest.mark.parametrize(
    "name,lossy,needs_size",
    [
        ("obb_yolo", False, False),
        ("dota", False, True),
        ("voc_xml", False, True),
        ("bbox_yolo", True, False),
        ("bbox_coco", True, True),
    ],
)
def test_metadatos_declarados(name, lossy, needs_size):
    converter = get(name)
    assert converter.lossy is lossy
    assert converter.requires_image_size is needs_size


def test_formato_desconocido_dice_cuales_hay():
    with pytest.raises(FormatError, match="formato desconocido"):
        get("no_existe")


def test_registrar_un_nombre_repetido_es_error():
    with pytest.raises(FormatError, match="duplicado"):

        @register
        class Otro:
            name = "obb_yolo"
            lossy = False
            requires_image_size = False


def test_anadir_un_formato_no_toca_convert():
    """El registro es lo unico que decide: `convert` no conoce ningun nombre."""

    @register
    class Doble:
        name = "_doble_de_prueba"
        lossy = False
        requires_image_size = False

        def to_quad(self, record, *, size=None):
            tokens = [float(t) / 2 for t in str(record.payload).split()]
            return canonicalize(Quad.from_xy(tokens))

        def from_quad(self, q, *, class_id=0, size=None, class_names=DEFAULT_CLASS_NAMES):
            return Record(class_id, " ".join(f"{v * 2:.17g}" for v in q.flat()))

    try:
        source = get("obb_yolo").from_quad(quad())
        out = convert(source, source="obb_yolo", target="_doble_de_prueba")
        back = convert(out, source="_doble_de_prueba", target="obb_yolo")
        assert get("obb_yolo").to_quad(back).points == quad().points
    finally:
        del formats.REGISTRY["_doble_de_prueba"]


# --- ida y vuelta ---------------------------------------------------------


@pytest.mark.parametrize("name", LOSSLESS)
@pytest.mark.parametrize("theta", [0.0, 0.3, 1.2, 2.4, 3.0])
def test_los_formatos_sin_perdida_van_y_vuelven(name, theta):
    original = quad(theta=theta)
    record = get("obb_yolo").from_quad(original)
    other = convert(record, source="obb_yolo", target=name, size=SIZE)
    back = convert(other, source=name, target="obb_yolo", size=SIZE)
    assert max_pixel_error(original, get("obb_yolo").to_quad(back)) < 1e-3


def test_obb_yolo_va_y_vuelve_exacto():
    """Sin cambio de unidades no hay redondeo: la igualdad es exacta."""
    original = quad(theta=0.7)
    record = get("obb_yolo").from_quad(original)
    assert get("obb_yolo").to_quad(record).points == original.points


def test_la_clase_sobrevive_a_la_conversion():
    record = Record(class_id=0, payload=get("obb_yolo").from_quad(quad()).payload)
    for name in LOSSLESS + LOSSY:
        out = convert(record, source="obb_yolo", target=name, size=SIZE)
        assert out.class_id == 0


# --- perdida donde toca ---------------------------------------------------


@pytest.mark.parametrize("name", LOSSY)
def test_los_formatos_con_perdida_pierden_la_orientacion(name):
    """Un quad girado no vuelve igual: es el punto de que sean lossy."""
    original = quad(theta=0.6)
    record = get("obb_yolo").from_quad(original)
    other = convert(record, source="obb_yolo", target=name, size=SIZE)
    back = convert(other, source=name, target="obb_yolo", size=SIZE)
    assert max_pixel_error(original, get("obb_yolo").to_quad(back)) > 1.0


def test_a_45_grados_la_envolvente_alineada_es_un_cuadrado():
    """El peor caso de bbox_coco, y avisa por si solo.

    La envolvente alineada de un rectangulo 2:1 girado 45 grados tiene los dos
    lados iguales, asi que su ancla canonica es inestable y salta
    `QuadShapeWarning`. No es un fallo del conversor: es exactamente el
    diagnostico para el que existen los formatos alineados. Que el aviso llegue
    hasta aqui confirma que no se traga por el camino.
    """
    original = quad(theta=math.pi / 4)
    record = convert(
        get("obb_yolo").from_quad(original),
        source="obb_yolo",
        target="bbox_coco",
        size=SIZE,
    )
    with pytest.warns(QuadShapeWarning, match="ancla del lado mas largo"):
        convert(record, source="bbox_coco", target="obb_yolo", size=SIZE)


@pytest.mark.parametrize("name", LOSSY)
def test_una_caja_ya_alineada_sobrevive_a_los_lossy(name):
    """Sin orientacion que perder, no se pierde nada."""
    original = quad(theta=0.0)
    record = get("obb_yolo").from_quad(original)
    other = convert(record, source="obb_yolo", target=name, size=SIZE)
    back = convert(other, source=name, target="obb_yolo", size=SIZE)
    assert max_pixel_error(original, get("obb_yolo").to_quad(back)) < 1e-3


def test_bbox_yolo_da_la_envolvente_alineada():
    original = quad(theta=math.pi / 4, half_long=0.2, ratio=2.0)
    record = get("bbox_yolo").from_quad(original)
    cx, cy, w, h = (float(t) for t in record.payload.split())
    xs = [p[0] for p in original.points]
    ys = [p[1] for p in original.points]
    assert cx == pytest.approx((min(xs) + max(xs)) / 2)
    assert cy == pytest.approx((min(ys) + max(ys)) / 2)
    assert w == pytest.approx(max(xs) - min(xs))
    assert h == pytest.approx(max(ys) - min(ys))


# --- el tamano de la imagen -----------------------------------------------


@pytest.mark.parametrize("name", ["dota", "voc_xml", "bbox_coco"])
def test_sin_tamano_los_formatos_en_pixeles_fallan_claro(name):
    record = get("obb_yolo").from_quad(quad())
    with pytest.raises(FormatError, match="tamano de la imagen"):
        convert(record, source="obb_yolo", target=name)


@pytest.mark.parametrize("name", ["obb_yolo", "bbox_yolo"])
def test_los_formatos_normalizados_no_piden_tamano(name):
    record = get("obb_yolo").from_quad(quad())
    assert convert(record, source="obb_yolo", target=name) is not None


def test_el_tamano_correcto_cambia_el_resultado():
    """Si el tamano se ignorase, dos aspectos distintos darian lo mismo."""
    q = quad(theta=0.5)
    ancha = get("dota").from_quad(q, size=ImageSize(1000, 200))
    alta = get("dota").from_quad(q, size=ImageSize(200, 1000))
    assert ancha.payload != alta.payload


# --- voc_xml y los rectangulos --------------------------------------------


def test_voc_xml_escribe_robndbox_con_el_lado_largo_en_w():
    q = quad(theta=0.4, half_long=0.2, ratio=2.5)
    record = get("voc_xml").from_quad(q, size=SIZE)
    box = record.payload.find("robndbox")
    w = float(box.find("w").text)
    h = float(box.find("h").text)
    assert w > h
    angle = float(box.find("angle").text)
    assert 0.0 <= angle < math.pi


def test_voc_xml_rechaza_un_quad_que_no_es_rectangulo():
    """Perder informacion en silencio seria peor que fallar."""
    torcido = canonicalize(
        Quad.from_xy([(0.1, 0.1), (0.6, 0.12), (0.55, 0.4), (0.12, 0.3)])
    )
    with pytest.raises(FormatError, match="rectangulos"):
        get("voc_xml").from_quad(torcido, size=SIZE)


def test_voc_xml_sin_robndbox_es_error():
    element = ET.Element("object")
    ET.SubElement(element, "name").text = "euro_banknote"
    with pytest.raises(FormatError, match="robndbox"):
        get("voc_xml").to_quad(Record(0, element), size=SIZE)


# --- las anotaciones de origen son de solo lectura ------------------------


def test_las_anotaciones_originales_no_cambian(tmp_path):
    """Inmutabilidad por hash tras ejecutar TODAS las conversiones."""
    paths = []
    for i, theta in enumerate((0.0, 0.6, 1.9)):
        q = quad(theta=theta)
        path = tmp_path / f"label_{i}.txt"
        path.write_text(
            "0 " + get("obb_yolo").from_quad(q).payload + "\n", encoding="utf-8"
        )
        paths.append(path)

    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    for path in paths:
        payload = path.read_text(encoding="utf-8").split(maxsplit=1)[1].strip()
        record = Record(class_id=0, payload=payload)
        for target in formats.formats():
            other = convert(record, source="obb_yolo", target=target, size=SIZE)
            convert(other, source=target, target="obb_yolo", size=SIZE)

    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before == after
