"""Exportadores de dataset, registrados por decorador.

Los conversores de `formats.py` traducen UNA anotacion. Esto escribe el arbol de
ficheros que cada entrenador espera encontrar: donde van las imagenes, donde las
etiquetas, y con que nombres.

Que lee cada quien:

    dota      RTMDet-R: un .txt por imagen en labelTxt/
    voc_xml   VOC con <robndbox> de roLabelImg: un .xml por imagen
    coco      diagnostico y herramientas de terceros: un solo annotations.json

Todos pasan por `dataio/view.prepare`, asi que comparten filtro de area y
politica de borde con la vista de Ultralytics. Es lo que evita que dos
candidatos entrenen con verdades distintas y la tabla los compare como iguales.

Aviso sobre voc_xml: hoy no lo consume nadie
--------------------------------------------
Se escribio pensando en `buzhidaoshenme/YOLOX-OBB`. Verificado despues contra su
`dota_obb.py`, ese fork NO lee `<robndbox>`: espera un `<bndbox>` de VOC con un
`<angle>` DENTRO y coordenadas en base 1. Asi que este exportador no le vale, y
el fork quedo descartado por otros motivos (ver README).

Se conserva porque `<robndbox>` de roLabelImg es el convenio VOC-OBB mas comun y
el conversor esta probado, pero **ningun candidato actual lo consume**. Antes de
usarlo con una herramienta concreta, hay que leer SU parser -- que es la leccion
de haberlo escrito contra una suposicion.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from testbank.config import Config, OutOfBoundsPolicy
from testbank.dataio.formats import DEFAULT_CLASS_NAMES, ImageSize
from testbank.dataio.formats import get as get_format
from testbank.dataio.view import (
    PreparationReport,
    PreparedSample,
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

    policy = OutOfBoundsPolicy(config.detector.out_of_bounds)
    if exporter.requires_rectangles and policy is OutOfBoundsPolicy.CLIP:
        # Se comprueba ANTES de escribir. La conversion tambien lo detecta, pero
        # lo haria a mitad del volcado y despues de dejar cientos de ficheros a
        # medias, con un mensaje que habla de un quad concreto y no de la causa.
        raise ExportError(
            f"el formato {name!r} solo representa rectangulos, y la politica "
            "de borde 'clip' no los conserva: recortar los vertices de un "
            "rectangulo girado contra el marco da un cuadrilatero con los lados "
            "opuestos desiguales.\n"
            "  Usa --out-of-bounds keep, que deja los vertices fuera de [0,1] "
            "tal cual. VOC no valida los limites como hace Ultralytics, asi que "
            f"los acepta: medido, {name!r} exporta las 452 imagenes con 'keep'."
        )

    prepared, report = prepare(samples_by_split, config)
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
