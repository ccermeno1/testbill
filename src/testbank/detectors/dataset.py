"""Materializa una vista del dataset para entrenadores que leen del disco.

Ultralytics, MMDetection y los forks de YOLOX no saben nada de `SplitLoader` ni
del filtro de area: leen ficheros de un arbol de directorios. Si les dejaramos el
directorio de datos tal cual, entrenarian con DOS incoherencias:

1. **La particion equivocada.** La nuestra esta congelada en `splits/*.txt`, y el
   arbol de Roboflow puede cambiar. Todo lee de los txt, nunca del arbol.

2. **Anotaciones que ya no estan vigentes.** El filtro de area relativa se aplica
   en carga, asi que las metricas evaluan contra la verdad filtrada, pero el
   entrenador leeria los ficheros de origen sin filtrar. Entrenar con una verdad
   y medir contra otra hace que las cifras no signifiquen nada.

Asi que se escribe una vista derivada bajo `data/derived/`, con las etiquetas ya
filtradas. **Los ficheros de origen no se tocan**, como en todo el proyecto: esto
es una copia de trabajo, regenerable y desechable.

Las imagenes se enlazan en duro cuando el sistema deja; si no, se copian. Un
enlace simbolico seria mejor pero en Windows necesita permisos que no siempre
hay, y fallar por eso al empezar a entrenar seria absurdo.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from testbank.config import Config, OutOfBoundsPolicy
from testbank.dataio.formats import DEFAULT_CLASS_NAMES
from testbank.dataio.formats import get as get_format
from testbank.dataio.image_sizes import SizeIndex
from testbank.dataio.loader import load_samples
from testbank.geometry.quad import Quad, canonicalize

DERIVED_DIRNAME = "ultralytics"


@dataclass(frozen=True, slots=True)
class MaterializedDataset:
    root: Path
    data_yaml: Path
    counts: dict[str, int]
    #: Anotaciones que el filtro dejo fuera y que por tanto NO se escribieron.
    dropped: int
    #: Politica aplicada a los billetes que cruzan el borde, y cuantos la
    #: necesitaron. Con `keep` este numero es el de imagenes que Ultralytics
    #: descartara, asi que la perdida queda registrada en vez de ser invisible.
    out_of_bounds: str = OutOfBoundsPolicy.CLIP.value
    adjusted: int = 0
    #: Solo con `pad`: quads que seguian fuera DESPUES de padear y hubo que
    #: recortar. Un numero alto dice que `pad_fraction` se queda corto.
    clipped_after_pad: int = 0
    pad_fraction: float = 0.0

    def describe(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        extra = ""
        if self.out_of_bounds == OutOfBoundsPolicy.PAD.value:
            extra = (
                f", pad={self.pad_fraction:.0%}, "
                f"{self.clipped_after_pad} recortadas aun asi"
            )
        return (
            f"{self.root} ({counts}; {self.dropped} anotaciones filtradas; "
            f"borde={self.out_of_bounds}, {self.adjusted} anotaciones fuera "
            f"del marco{extra})"
        )


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
    """Factor de escala y desplazamiento al normalizar sobre la imagen padeada.

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


def _out_of_bounds(quad: Quad) -> bool:
    return any(not (0.0 <= v <= 1.0) for v in quad.flat())


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        # Volumen distinto, sistema de ficheros sin enlaces duros, o permisos.
        shutil.copy2(source, target)


def materialize(
    samples_by_split: dict[str, list],
    config: Config | None = None,
    *,
    out_dir: Path | None = None,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    overwrite: bool = True,
) -> MaterializedDataset:
    """Escribe `images/` y `labels/` por particion, con la verdad ya filtrada."""
    config = config or Config()
    # Coercion deliberada: `Config.model_copy(update=...)` NO valida, asi que un
    # `out_of_bounds="clip"` puesto por ahi llega como str y revienta mas tarde
    # con un AttributeError sin relacion aparente. Normalizar aqui vale mas que
    # confiar en que todo el mundo construya el enum.
    policy = OutOfBoundsPolicy(config.detector.out_of_bounds)
    pad_fraction = config.detector.pad_fraction
    adjusted = 0
    clipped_after_pad = 0
    root = Path(out_dir or (config.data.derived_dir / DERIVED_DIRNAME))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    writer = get_format("obb_yolo")
    counts: dict[str, int] = {}
    dropped_total = 0

    for split, samples in samples_by_split.items():
        samples = list(samples)
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        loaded, report = load_samples(
            samples,
            sizes=sizes,
            min_relative_area=config.annotation_policy.min_relative_area,
        )
        dropped_total += len(report.dropped)

        by_id = {s.sample_id: s for s in samples}
        for item in loaded:
            sample = by_id[item.sample_id]
            target = images_dir / sample.image_path.name

            if policy is OutOfBoundsPolicy.PAD:
                pad_image(sample.image_path, target, pad_fraction)
            else:
                _link_or_copy(sample.image_path, target)

            lines = []
            for annotation in item.annotations:
                quad = annotation.quad
                if _out_of_bounds(quad):
                    adjusted += 1
                    if policy is OutOfBoundsPolicy.CLIP:
                        quad = clip_quad(quad)
                if policy is OutOfBoundsPolicy.PAD:
                    # TODOS los quads se recolocan, se salieran o no: la imagen
                    # ha cambiado de tamano y el sistema de coordenadas con ella.
                    quad = pad_quad(quad, pad_fraction)
                    if _out_of_bounds(quad):
                        # Sigue saliendose: el borde no daba para tanto. Se
                        # recorta en vez de dejarlo fuera, porque dejarlo fuera
                        # hace que el entrenador descarte la imagen ENTERA -- se
                        # perderia el billete bueno junto al que se sale. Esto
                        # es lo que permite padear poco: cubrir el caso comun
                        # con el borde y el residuo con el recorte.
                        clipped_after_pad += 1
                        quad = clip_quad(quad)
                lines.append(f"{annotation.class_id} " + writer.from_quad(quad).payload)

            (labels_dir / f"{item.sample_id}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        counts[split] = len(loaded)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": "train/images",
                # Ultralytics usa la clave `val`; el directorio se llama `valid`
                # porque es el nombre que exporta Roboflow. Son cosas distintas.
                "val": "valid/images",
                "names": dict(enumerate(class_names)),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return MaterializedDataset(
        root=root,
        data_yaml=data_yaml,
        counts=counts,
        dropped=dropped_total,
        out_of_bounds=policy.value,
        adjusted=adjusted,
        clipped_after_pad=clipped_after_pad,
        pad_fraction=pad_fraction if policy is OutOfBoundsPolicy.PAD else 0.0,
    )
