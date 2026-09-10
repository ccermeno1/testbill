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

from testbank.config import Config
from testbank.dataio.formats import DEFAULT_CLASS_NAMES
from testbank.dataio.formats import get as get_format
from testbank.dataio.image_sizes import SizeIndex
from testbank.dataio.loader import load_samples

DERIVED_DIRNAME = "ultralytics"


@dataclass(frozen=True, slots=True)
class MaterializedDataset:
    root: Path
    data_yaml: Path
    counts: dict[str, int]
    #: Anotaciones que el filtro dejo fuera y que por tanto NO se escribieron.
    dropped: int

    def describe(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        return f"{self.root} ({counts}; {self.dropped} anotaciones filtradas)"


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
            _link_or_copy(
                sample.image_path, images_dir / sample.image_path.name
            )
            lines = [
                f"{annotation.class_id} "
                + writer.from_quad(annotation.quad).payload
                for annotation in item.annotations
            ]
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
        root=root, data_yaml=data_yaml, counts=counts, dropped=dropped_total
    )
