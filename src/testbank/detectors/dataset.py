"""Vista del dataset en el formato que lee Ultralytics.

Ultralytics, MMDetection y los forks de YOLOX no saben nada de `SplitLoader` ni
del filtro de area: leen ficheros de un arbol de directorios. Si les dejaramos el
directorio de datos tal cual, entrenarian con DOS incoherencias:

1. **La particion equivocada.** La nuestra esta congelada en `splits/*.txt`, y el
   arbol de Roboflow puede cambiar. Todo lee de los txt, nunca del arbol.

2. **Anotaciones que ya no estan vigentes.** El filtro de area relativa se aplica
   en carga, asi que las metricas evaluan contra la verdad filtrada, pero el
   entrenador leeria los ficheros de origen sin filtrar. Entrenar con una verdad
   y medir contra otra hace que las cifras no signifiquen nada.

Asi que se escribe una vista derivada bajo `data/derived/`. **Los ficheros de
origen no se tocan**: esto es una copia de trabajo, regenerable y desechable.

El filtro y la politica de borde viven en `dataio/prepare.py`, compartidos con los
exportadores de DOTA, VOC y COCO. Repetirlos aqui habria dejado que derivaran, y
entonces dos candidatos entrenarian con verdades distintas mientras la tabla los
compara como si fueran lo mismo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from testbank.config import Config, OutOfBoundsPolicy
from testbank.dataio.formats import DEFAULT_CLASS_NAMES
from testbank.dataio.formats import get as get_format
from testbank.dataio.prepare import (
    PreparationReport,
    clip_quad,
    pad_geometry,
    pad_image,
    pad_quad,
    place_image,
    prepare,
)

DERIVED_DIRNAME = "ultralytics"

__all__ = [
    "DERIVED_DIRNAME",
    "MaterializedDataset",
    "clip_quad",
    "materialize",
    "pad_geometry",
    "pad_image",
    "pad_quad",
]


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

    @classmethod
    def from_report(
        cls, root: Path, data_yaml: Path, report: PreparationReport
    ) -> MaterializedDataset:
        return cls(
            root=root,
            data_yaml=data_yaml,
            counts=dict(report.counts),
            dropped=report.dropped,
            out_of_bounds=report.policy,
            adjusted=report.adjusted,
            clipped_after_pad=report.clipped_after_pad,
            pad_fraction=report.pad_fraction,
        )

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


def materialize(
    samples_by_split: dict[str, list],
    config: Config | None = None,
    *,
    out_dir: Path | None = None,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    overwrite: bool = True,
) -> MaterializedDataset:
    """Escribe `images/` y `labels/` por particion, con la verdad ya filtrada."""
    import shutil

    config = config or Config()
    root = Path(out_dir or (config.data.derived_dir / DERIVED_DIRNAME))
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    prepared, report = prepare(samples_by_split, config)
    writer = get_format("obb_yolo")

    for split, items in prepared.items():
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            place_image(item, images_dir / item.sample.image_path.name)
            lines = [
                f"{class_id} " + writer.from_quad(quad).payload
                for quad, class_id in zip(item.quads, item.class_ids)
            ]
            (labels_dir / f"{item.sample_id}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

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
    return MaterializedDataset.from_report(root, data_yaml, report)
