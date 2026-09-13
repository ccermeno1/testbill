"""Exportadores de dataset, registrados por decorador.

Los conversores de `formats.py` traducen UNA anotacion. Esto escribe el arbol de
ficheros que cada entrenador espera encontrar: donde van las imagenes, donde las
etiquetas, y con que nombres.

Que lee cada quien:

    dota            RTMDet-R: un .txt por imagen en labelTxt/
    yolox_obb_voc   buzhidaoshenme/YOLOX-OBB: VOCdevkit con <angle> en el bndbox
    voc_xml         VOC con <robndbox> de roLabelImg: hoy no lo consume nadie
    coco            diagnostico y terceros: un solo annotations.json

Todos pasan por `dataio/view.prepare`, asi que comparten filtro de area y
politica de borde con la vista de Ultralytics. Es lo que evita que dos
candidatos entrenen con verdades distintas y la tabla los compare como iguales.

Por que hay DOS exportadores VOC y no uno
-----------------------------------------
Parecen el mismo formato y no lo son. `voc_xml` escribe el `<robndbox>` de
roLabelImg -- cx/cy/w/h/angle en elementos propios -- que es el convenio VOC-OBB
mas extendido. El fork de YOLOX-OBB no lee eso: mete un `<angle>` DENTRO de un
`<bndbox>` normal cuyos xmin/xmax son en realidad w y h colocados alrededor del
centro. Unificarlos habria significado escribir una mentira para uno de los dos.

`voc_xml` **no lo consume ningun candidato actual**. Se escribio contra una
suposicion sobre el fork que resulto falsa, y se conserva solo porque el formato
es comun y el conversor esta probado. Antes de enchufarlo a una herramienta
concreta hay que leer SU parser: es exactamente la leccion que costo escribirlo.
"""

from __future__ import annotations

import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from testbank.config import Config
from testbank.dataio.formats import DEFAULT_CLASS_NAMES, ImageSize
from testbank.dataio.formats import get as get_format
from testbank.dataio.view import (
    PreparationReport,
    PreparedSample,
    pad_image,
    place_image,
    prepare,
)


class ExportError(ValueError):
    """No se pudo escribir la vista pedida."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    name: str
    root: Path
    report: PreparationReport
    #: Lo que hay que pasarle al entrenador: un yaml, un json o un directorio.
    entry_point: Path

    def describe(self) -> str:
        return f"{self.name} -> {self.entry_point} ({self.report.describe()})"


@runtime_checkable
class Exporter(Protocol):
    name: ClassVar[str]
    #: Formato de `formats.py` con el que se serializa cada anotacion.
    annotation_format: ClassVar[str]
    #: Si el formato solo sabe representar rectangulos. Ver `export()`.
    requires_rectangles: ClassVar[bool]

    def write(
        self,
        prepared: dict[str, list[PreparedSample]],
        root: Path,
        *,
        class_names: tuple[str, ...],
    ) -> Path: ...


REGISTRY: dict[str, Exporter] = {}


def register(cls: type) -> type:
    if cls.name in REGISTRY:
        raise ExportError(f"exportador duplicado en el registro: {cls.name!r}")
    REGISTRY[cls.name] = cls()
    return cls


def get(name: str) -> Exporter:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ExportError(
            f"exportador desconocido: {name!r}; registrados: {sorted(REGISTRY)}"
        ) from None


def exporters() -> list[str]:
    return sorted(REGISTRY)


# --- formatos --------------------------------------------------------------


@register
class DotaExporter:
    """`images/` + `labelTxt/`, un .txt por imagen. El convenio de DOTA."""

    name = "dota"
    annotation_format = "dota"
    #: DOTA guarda los cuatro vertices sueltos: acepta cualquier cuadrilatero.
    requires_rectangles = False

    def write(self, prepared, root, *, class_names) -> Path:
        writer = get_format(self.annotation_format)
        for split, items in prepared.items():
            images = root / split / "images"
            labels = root / split / "labelTxt"
            labels.mkdir(parents=True, exist_ok=True)
            for item in items:
                place_image(item, images / item.sample.image_path.name)
                lines = [
                    writer.from_quad(
                        quad, class_id=class_id, size=item.size, class_names=class_names
                    ).payload
                    for quad, class_id in zip(item.quads, item.class_ids)
                ]
                (labels / f"{item.sample_id}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
        return root


@register
class VocXmlExporter:
    """`JPEGImages/` + `Annotations/` + `ImageSets/Main/`. El convenio de VOC."""

    name = "voc_xml"
    annotation_format = "voc_xml"
    #: `<robndbox>` es cx/cy/w/h/angle: cinco grados de libertad, solo
    #: rectangulos. Ver el aviso en `export()`.
    requires_rectangles = True

    def write(self, prepared, root, *, class_names) -> Path:
        writer = get_format(self.annotation_format)
        images_dir = root / "JPEGImages"
        annotations_dir = root / "Annotations"
        sets_dir = root / "ImageSets" / "Main"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        sets_dir.mkdir(parents=True, exist_ok=True)

        for split, items in prepared.items():
            # VOC llama `val` a lo que Roboflow llama `valid`. El fichero de
            # conjunto usa el nombre de VOC porque lo lee su dataloader.
            set_name = "val" if split == "valid" else split
            ids = []
            for item in items:
                place_image(item, images_dir / item.sample.image_path.name)
                ids.append(item.sample_id)
                root_element = ET.Element("annotation")
                ET.SubElement(root_element, "folder").text = "JPEGImages"
                ET.SubElement(root_element, "filename").text = (
                    item.sample.image_path.name
                )
                size = ET.SubElement(root_element, "size")
                ET.SubElement(size, "width").text = str(item.size.width)
                ET.SubElement(size, "height").text = str(item.size.height)
                ET.SubElement(size, "depth").text = "3"
                for quad, class_id in zip(item.quads, item.class_ids):
                    record = writer.from_quad(
                        quad, class_id=class_id, size=item.size, class_names=class_names
                    )
                    root_element.append(record.payload)
                ET.ElementTree(root_element).write(
                    annotations_dir / f"{item.sample_id}.xml", encoding="utf-8"
                )
            (sets_dir / f"{set_name}.txt").write_text(
                "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8"
            )
        return sets_dir


@register
class CocoExporter:
    """Un `annotations.json` por particion. LOSSY: pierde la orientacion.

    Se exporta igualmente porque es lo que comen muchas herramientas de
    inspeccion, pero un candidato entrenado desde aqui no puede predecir cajas
    giradas: `bbox_coco` esta marcado `lossy` en el registro de formatos por
    este motivo exacto.
    """

    name = "coco"
    annotation_format = "bbox_coco"
    #: La envolvente alineada al eje siempre es un rectangulo, se le de lo que
    #: se le de. Pierde la orientacion, pero no se atraganta.
    requires_rectangles = False

    def write(self, prepared, root, *, class_names) -> Path:
        writer = get_format(self.annotation_format)
        entry_point = root
        for split, items in prepared.items():
            images_dir = root / split / "images"
            payload = {
                "info": {
                    "description": "Exportado por testbank",
                    "note": (
                        "bbox_coco pierde la orientacion: las cajas son la "
                        "envolvente alineada al eje del billete."
                    ),
                },
                "categories": [
                    {"id": i, "name": name} for i, name in enumerate(class_names)
                ],
                "images": [],
                "annotations": [],
            }
            annotation_id = 1
            for image_id, item in enumerate(items, start=1):
                place_image(item, images_dir / item.sample.image_path.name)
                payload["images"].append(
                    {
                        "id": image_id,
                        "file_name": item.sample.image_path.name,
                        "width": item.size.width,
                        "height": item.size.height,
                    }
                )
                for quad, class_id in zip(item.quads, item.class_ids):
                    record = writer.from_quad(
                        quad, class_id=class_id, size=item.size, class_names=class_names
                    )
                    entry = dict(record.payload)
                    entry["id"] = annotation_id
                    entry["image_id"] = image_id
                    payload["annotations"].append(entry)
                    annotation_id += 1

            target = root / split / "annotations.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            entry_point = target
        return entry_point



@register
class YoloxObbVocExporter:
    """El arbol VOC que espera `buzhidaoshenme/YOLOX-OBB`. Leido, no supuesto.

    Estructura, sacada de su `dota_obb.py`::

        <root>/VOC2007/Annotations/<id>.xml
                      /JPEGImages/<id>.png          <- trainval
                      /JPEGImages-val/<id>.png      <- val
                      /JPEGImages-test/<id>.png     <- test
                      /ImageSets/Main/{trainval,val,test}.txt

    Cuatro trampas que solo se ven leyendo su codigo, no su README:

    1. **La extension .png esta fija**: `_imgpath = "%s/JPEGImages/%s.png"`. Las
       nuestras son JPEG, asi que hay que RECODIFICARLAS. No vale un enlace con
       otro nombre: `cv2.imread` mira el contenido, no la extension, pero el
       resto de la cadena no. Cuesta una segunda copia entera del dataset.
    2. **Validacion va en `JPEGImages-val/`**, un directorio aparte. Y el path se
       elige con `image_sets[0][1]`, o sea con la PRIMERA particion de la lista:
       un mismo objeto no puede servir train y val a la vez.
    3. **El `VOC2007/` intermedio** sale de `os.path.join(root, "VOC" + year)`.
    4. **Las clases son las 15 de DOTA**, cableadas en `dota_classes.py`, y el
       parser hace `self.class_to_ind[name]`. Con nuestro `euro_banknote` da
       KeyError salvo que se edite ESE fichero al vendorizar el fork. El
       exportador no puede arreglarlo; queda anotado en el README.

    La base 1 de las coordenadas y el `<difficult>` obligatorio los resuelve el
    conversor `yolox_obb_voc`, que explica ambos.
    """

    name = "yolox_obb_voc"
    annotation_format = "yolox_obb_voc"
    #: Su `(cx, cy, w, h, angulo)` solo representa rectangulos.
    requires_rectangles = True

    #: Como llama el fork a cada particion nuestra.
    SET_NAMES: ClassVar[dict[str, str]] = {"train": "trainval", "valid": "val"}
    #: El `VOC<year>` intermedio. Su valor por defecto en `image_sets`.
    YEAR: ClassVar[str] = "2007"

    def _images_dir(self, base: Path, set_name: str) -> Path:
        return base / (
            "JPEGImages" if set_name == "trainval" else f"JPEGImages-{set_name}"
        )

    def write(self, prepared, root, *, class_names) -> Path:
        import cv2

        writer = get_format(self.annotation_format)
        base = root / f"VOC{self.YEAR}"
        annotations_dir = base / "Annotations"
        sets_dir = base / "ImageSets" / "Main"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        sets_dir.mkdir(parents=True, exist_ok=True)

        for split, items in prepared.items():
            set_name = self.SET_NAMES.get(split, split)
            images_dir = self._images_dir(base, set_name)
            images_dir.mkdir(parents=True, exist_ok=True)
            ids = []
            for item in items:
                target = images_dir / f"{item.sample_id}.png"
                if item.needs_rewrite:
                    # `pad_image` escribe con cv2, que elige el codec por la
                    # extension: al apuntar a .png ya sale recodificada.
                    pad_image(item.sample.image_path, target, item.pad_fraction)
                else:
                    image = cv2.imread(str(item.sample.image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ExportError(f"no se pudo leer {item.sample.image_path}")
                    cv2.imwrite(str(target), image)

                ids.append(item.sample_id)
                element = ET.Element("annotation")
                ET.SubElement(element, "folder").text = images_dir.name
                ET.SubElement(element, "filename").text = target.name
                size = ET.SubElement(element, "size")
                ET.SubElement(size, "width").text = str(item.size.width)
                ET.SubElement(size, "height").text = str(item.size.height)
                ET.SubElement(size, "depth").text = "3"
                for quad, class_id in zip(item.quads, item.class_ids):
                    element.append(
                        writer.from_quad(
                            quad,
                            class_id=class_id,
                            size=item.size,
                            class_names=class_names,
                        ).payload
                    )
                ET.ElementTree(element).write(
                    annotations_dir / f"{item.sample_id}.xml", encoding="utf-8"
                )
            (sets_dir / f"{set_name}.txt").write_text(
                "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8"
            )
        return sets_dir


def _first_non_rectangle(prepared, *, tolerance: float = 1e-3):
    """El primer cuadrilatero con los lados opuestos desiguales, si lo hay.

    La tolerancia es RELATIVA al lado mayor. Un rectangulo perfecto reconstruido
    desde coordenadas normalizadas difiere en la ultima cifra, y un umbral
    absoluto sobre una imagen de 4000 px no significa lo mismo que sobre una de
    400.
    """
    for items in prepared.values():
        for item in items:
            for quad in item.quads:
                points = [
                    (x * item.size.width, y * item.size.height)
                    for x, y in quad.points
                ]
                sides = [math.dist(points[i], points[(i + 1) % 4]) for i in range(4)]
                longest = max(sides) or 1.0
                if (
                    abs(sides[0] - sides[2]) > tolerance * longest
                    or abs(sides[1] - sides[3]) > tolerance * longest
                ):
                    return item.sample_id, sides
    return None


def export(
    name: str,
    samples_by_split: dict[str, list],
    config: Config | None = None,
    *,
    out_dir: Path | None = None,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    overwrite: bool = True,
) -> ExportResult:
    """Escribe la vista completa en el formato pedido."""
    config = config or Config()
    exporter = get(name)
    root = Path(out_dir or (config.data.derived_dir / name))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    prepared, report = prepare(samples_by_split, config)
    # `clip` ya NO veta a los formatos de rectangulo.
    #
    # Lo vetaba porque recortaba pinzando cada vertice, y eso convierte un
    # rectangulo girado en un trapecio -- 82 de 679 anotaciones reales. Desde
    # que `clip_quad` conserva el angulo y devuelve un rectangulo, la
    # incompatibilidad no existe.
    #
    # Se COMPRUEBA en vez de suponerse: si entra un cuadrilatero irregular por
    # otra via, salta aqui y antes de escribir nada, no a mitad del volcado y
    # con cientos de ficheros a medias.
    if exporter.requires_rectangles:
        offender = _first_non_rectangle(prepared)
        if offender is not None:
            sample_id, sides = offender
            raise ExportError(
                f"el formato {name!r} solo representa rectangulos y"
                f" {sample_id} trae un cuadrilatero irregular: lados "
                + " x ".join(f"{side:.1f}" for side in sides)
                + " px, con los opuestos desiguales."
                "  Con --out-of-bounds keep se exporta sin tocar la geometria."
            )

    entry_point = exporter.write(prepared, root, class_names=class_names)
    return ExportResult(
        name=name, root=root, report=report, entry_point=entry_point
    )


__all__ = [
    "REGISTRY",
    "ExportError",
    "ExportResult",
    "Exporter",
    "ImageSize",
    "export",
    "exporters",
    "get",
    "register",
]
