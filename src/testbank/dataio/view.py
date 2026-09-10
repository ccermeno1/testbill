"""Preparacion comun de las muestras antes de escribirlas en cualquier formato.

Entre el fichero de origen y lo que ve un entrenador hay tres pasos que TIENEN
que ser los mismos para todos los formatos:

1. El filtro de area relativa (`dataio/loader.py`).
2. La politica de billetes que cruzan el borde (`clip`, `pad` o `keep`).
3. El tamano de imagen, que los formatos en pixeles necesitan.

Estaban dentro de `detectors/dataset.py`, que escribe la vista de Ultralytics.
Al anadir los exportadores de DOTA, VOC y COCO habria hecho falta repetirlos, y
una divergencia ahi seria silenciosa: dos candidatos entrenando con verdades
distintas y una tabla comparandolos como si fueran lo mismo. Asi que viven aqui
y los dos sitios llaman a lo mismo.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from testbank.config import Config, OutOfBoundsPolicy
from testbank.data.discover import Sample
from testbank.dataio.formats import ImageSize
from testbank.dataio.image_sizes import SizeIndex
from testbank.dataio.loader import load_samples
from testbank.geometry.quad import Quad, canonicalize


def clip_quad(quad: Quad) -> Quad:
    """Recorta los vertices al marco de la imagen.

    Recorta los VERTICES, no el poligono. Recortar el poligono contra el marco
    daria un pentagono o un hexagono cuando el billete sale por una esquina, y
    el resto del proyecto asume cuadrilateros. Mover los vertices conserva
    cuatro y deja la caja pegada al borde, que es donde acaban los pixeles.
    """
    return canonicalize(
        Quad.from_xy(
            [(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)) for x, y in quad.points]
        )
    )


def pad_geometry(fraction: float) -> tuple[float, float]:
    """Escala y desplazamiento al normalizar sobre la imagen padeada.

    Con un borde de `fraction` en cada lado, el lado total se multiplica por
    `1 + 2f`, y el origen de la imagen original queda en `f` del nuevo.
    """
    scale = 1.0 + 2.0 * fraction
    return 1.0 / scale, fraction / scale


def pad_quad(quad: Quad, fraction: float) -> Quad:
    """Recoloca un quad en coordenadas de la imagen ya padeada."""
    scale, offset = pad_geometry(fraction)
    return canonicalize(
        Quad.from_xy([(x * scale + offset, y * scale + offset) for x, y in quad.points])
    )


def pad_image(source: Path, target: Path, fraction: float) -> None:
    """Borde negro. Constante y no reflejado a proposito: el borde es contenido
    inventado y tiene que PARECERLO, no imitar textura que no se fotografio."""
    import cv2

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"no se pudo leer {source}")
    height, width = image.shape[:2]
    top = bottom = round(height * fraction)
    left = right = round(width * fraction)
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), padded)


def out_of_bounds(quad: Quad) -> bool:
    return any(not (0.0 <= v <= 1.0) for v in quad.flat())


@dataclass(frozen=True, slots=True)
class PreparedSample:
    """Una muestra lista para escribir, en el formato que sea."""

    sample: Sample
    #: Tamano de la imagen QUE SE ESCRIBE. Con `pad` es el padeado, no el
    #: original: los formatos en pixeles tienen que coincidir con la imagen.
    size: ImageSize
    quads: tuple[Quad, ...]
    class_ids: tuple[int, ...]
    #: Si la imagen hay que reescribirla (padding) o basta con enlazarla.
    needs_rewrite: bool = False
    #: Cuanto padding lleva. Va AQUI y no como argumento de quien escribe: cada
    #: exportador tendria que acordarse de pasarlo, y olvidarlo con politica
    #: `pad` escribiria la imagen sin borde mientras las etiquetas si lo
    #: asumen. La muestra sabe lo que es; quien la escribe no tiene que saberlo.
    pad_fraction: float = 0.0

    @property
    def sample_id(self) -> str:
        return self.sample.sample_id


@dataclass
class PreparationReport:
    policy: str
    pad_fraction: float = 0.0
    #: Anotaciones que el filtro de area dejo fuera.
    dropped: int = 0
    #: Anotaciones que cruzaban el borde.
    adjusted: int = 0
    #: Solo con `pad`: las que seguian fuera despues de padear.
    clipped_after_pad: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        extra = ""
        if self.policy == OutOfBoundsPolicy.PAD.value:
            extra = (
                f", pad={self.pad_fraction:.0%}, "
                f"{self.clipped_after_pad} recortadas aun asi"
            )
        return (
            f"{counts}; {self.dropped} anotaciones filtradas; "
            f"borde={self.policy}, {self.adjusted} fuera del marco{extra}"
        )


def prepare(
    samples_by_split: dict[str, list],
    config: Config | None = None,
) -> tuple[dict[str, list[PreparedSample]], PreparationReport]:
    """Filtro de area + politica de borde, igual para todos los formatos."""
    config = config or Config()
    # Coercion deliberada: `Config.model_copy(update=...)` NO valida, asi que un
    # `out_of_bounds="clip"` puesto por ahi llega como str y revienta mas tarde
    # con un AttributeError sin relacion aparente.
    policy = OutOfBoundsPolicy(config.detector.out_of_bounds)
    fraction = config.detector.pad_fraction
    report = PreparationReport(
        policy=policy.value,
        pad_fraction=fraction if policy is OutOfBoundsPolicy.PAD else 0.0,
    )

    out: dict[str, list[PreparedSample]] = {}
    for split, samples in samples_by_split.items():
        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        loaded, filter_report = load_samples(
            samples,
            sizes=sizes,
            min_relative_area=config.annotation_policy.min_relative_area,
        )
        report.dropped += len(filter_report.dropped)

        by_id = {s.sample_id: s for s in samples}
        prepared: list[PreparedSample] = []
        for item in loaded:
            width, height = sizes.size(item.sample_id)
            quads: list[Quad] = []
            for annotation in item.annotations:
                quad = annotation.quad
                if out_of_bounds(quad):
                    report.adjusted += 1
                    if policy is OutOfBoundsPolicy.CLIP:
                        quad = clip_quad(quad)
                if policy is OutOfBoundsPolicy.PAD:
                    # TODOS se recolocan, se salieran o no: la imagen cambio de
                    # tamano y el sistema de coordenadas con ella.
                    quad = pad_quad(quad, fraction)
                    if out_of_bounds(quad):
                        # El borde no daba para tanto. Se recorta en vez de
                        # dejarlo fuera, porque dejarlo fuera hace que el
                        # entrenador descarte la imagen ENTERA.
                        report.clipped_after_pad += 1
                        quad = clip_quad(quad)
                quads.append(quad)

            if policy is OutOfBoundsPolicy.PAD:
                width = width + 2 * round(width * fraction)
                height = height + 2 * round(height * fraction)

            prepared.append(
                PreparedSample(
                    sample=by_id[item.sample_id],
                    size=ImageSize(int(width), int(height)),
                    quads=tuple(quads),
                    class_ids=tuple(a.class_id for a in item.annotations),
                    needs_rewrite=policy is OutOfBoundsPolicy.PAD,
                    pad_fraction=fraction if policy is OutOfBoundsPolicy.PAD else 0.0,
                )
            )
        out[split] = prepared
        report.counts[split] = len(prepared)
    return out, report


def place_image(item: PreparedSample, target: Path) -> None:
    """Padea si la muestra lo pide; si no, enlace duro y si no se puede, copia.

    El enlace simbolico seria mejor pero en Windows necesita permisos que no
    siempre hay, y fallar por eso al empezar a entrenar seria absurdo.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if item.needs_rewrite:
        pad_image(item.sample.image_path, target, item.pad_fraction)
        return
    if target.exists():
        return
    try:
        os.link(item.sample.image_path, target)
    except OSError:
        # Volumen distinto, sistema de ficheros sin enlaces duros, o permisos.
        shutil.copy2(item.sample.image_path, target)
