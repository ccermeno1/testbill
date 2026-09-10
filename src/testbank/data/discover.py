"""Deteccion de la estructura del directorio de datos.

Modo 1 (adoptar): ya existe train/valid/test con images/ y labels/. Es lo que
exporta Roboflow y la particion se adopta tal cual.
Modo 2 (crear): no existe esa estructura, hay que generar la particion.

Este modulo solo DESCUBRE. La asignacion a particiones es competencia exclusiva
de splits.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

CANONICAL_SPLITS = ("train", "valid", "test")

# Roboflow usa "valid". "val" es el nombre de Ultralytics y aparece en exports
# retocados a mano: se acepta como alias pero se deja constancia.
_VALID_ALIASES = ("valid", "val")


class LayoutMode(str, Enum):
    ADOPT = "adopt"
    CREATE = "create"


class LayoutError(RuntimeError):
    """La estructura del directorio de datos no es utilizable."""


@dataclass(frozen=True, slots=True)
class Sample:
    """Una imagen y su fichero de etiquetas. `sample_id` es el nombre sin extension."""

    sample_id: str
    image_path: Path
    label_path: Path

    @property
    def directory_key(self) -> str:
        return self.image_path.parent.name


@dataclass(frozen=True, slots=True)
class Layout:
    root: Path
    mode: LayoutMode
    #: En modo adoptar: particion -> muestras. En modo crear: {"__all__": muestras}.
    groups: dict[str, tuple[Sample, ...]]
    notes: tuple[str, ...] = ()

    @property
    def all_samples(self) -> tuple[Sample, ...]:
        out: list[Sample] = []
        for key in sorted(self.groups):
            out.extend(self.groups[key])
        return tuple(out)


def _pair_directory(images_dir: Path, labels_dir: Path) -> tuple[Sample, ...]:
    """Empareja imagenes con etiquetas. Cualquier huerfano es fatal.

    Una etiqueta que falta NO se trata como "imagen sin billetes": es asi como se
    envenena un entrenamiento en silencio. Roboflow escribe un .txt vacio cuando
    no hay objetos, asi que la ausencia del fichero es una anomalia real.
    """
    images = {
        p.stem: p
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {
        p.stem: p for p in sorted(labels_dir.iterdir()) if p.is_file() and p.suffix == ".txt"
    }

    missing_labels = sorted(set(images) - set(labels))
    if missing_labels:
        raise LayoutError(
            f"{len(missing_labels)} imagenes sin fichero de etiquetas en "
            f"{labels_dir}: {missing_labels[:10]}"
            + (" ..." if len(missing_labels) > 10 else "")
        )
    orphan_labels = sorted(set(labels) - set(images))
    if orphan_labels:
        raise LayoutError(
            f"{len(orphan_labels)} etiquetas sin imagen en {images_dir}: "
            f"{orphan_labels[:10]}" + (" ..." if len(orphan_labels) > 10 else "")
        )

    return tuple(
        Sample(sample_id=stem, image_path=images[stem], label_path=labels[stem])
        for stem in sorted(images)
    )


def _resolve_split_dir(root: Path, split: str) -> tuple[Path | None, str | None]:
    """Devuelve el directorio de la particion y una nota si se uso un alias."""
    names = _VALID_ALIASES if split == "valid" else (split,)
    found = [name for name in names if (root / name).is_dir()]
    if len(found) > 1:
        raise LayoutError(
            f"existen a la vez {found} en {root}; ambiguo. Roboflow usa 'valid'; "
            "elimina o renombra el otro antes de continuar"
        )
    if not found:
        return None, None
    name = found[0]
    note = None
    if name != split:
        note = (
            f"se encontro '{name}/' y se adopta como particion '{split}'. "
            "Roboflow exporta 'valid'; revisa que el directorio sea el esperado"
        )
    return root / name, note


def detect_layout(root: str | Path) -> Layout:
    """Decide entre modo adoptar y modo crear inspeccionando el directorio."""
    root = Path(root)
    if not root.is_dir():
        raise LayoutError(f"el directorio de datos no existe: {root}")

    notes: list[str] = []
    split_dirs: dict[str, Path] = {}
    for split in CANONICAL_SPLITS:
        path, note = _resolve_split_dir(root, split)
        if path is not None and (path / "images").is_dir() and (path / "labels").is_dir():
            split_dirs[split] = path
            if note:
                notes.append(note)

    if "train" in split_dirs and "valid" in split_dirs:
        if "test" not in split_dirs:
            notes.append(
                "no hay particion 'test' en el export; se adopta train/valid y test "
                "queda vacia. El comando evaluate-test no tendra nada que evaluar"
            )
        groups = {
            split: _pair_directory(path / "images", path / "labels")
            for split, path in split_dirs.items()
        }
        return Layout(
            root=root, mode=LayoutMode.ADOPT, groups=groups, notes=tuple(notes)
        )

    if split_dirs:
        raise LayoutError(
            f"estructura a medias en {root}: se encontro {sorted(split_dirs)} pero "
            "el modo adoptar exige al menos 'train' y 'valid', cada una con "
            "images/ y labels/. Corrige la estructura o retirala del todo para "
            "que se genere la particion en modo crear"
        )

    samples = _discover_flat(root)
    return Layout(
        root=root,
        mode=LayoutMode.CREATE,
        groups={"__all__": samples},
        notes=tuple(notes),
    )


def _discover_flat(root: Path) -> tuple[Sample, ...]:
    """Modo crear: images/ + labels/ en la raiz, o imagenes y .txt conviviendo."""
    if (root / "images").is_dir() and (root / "labels").is_dir():
        return _pair_directory(root / "images", root / "labels")

    images = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise LayoutError(
            f"no se encontro ninguna imagen bajo {root}. Se esperaba o bien "
            "train/valid/test con images/ y labels/, o bien images/ y labels/, "
            "o imagenes con su .txt al lado"
        )

    by_stem: dict[str, Path] = {}
    for path in images:
        if path.stem in by_stem:
            raise LayoutError(
                f"nombre de imagen duplicado '{path.stem}': {by_stem[path.stem]} y "
                f"{path}. El identificador de muestra es el nombre sin extension y "
                "tiene que ser unico"
            )
        by_stem[path.stem] = path

    samples: list[Sample] = []
    missing: list[str] = []
    for stem, image_path in sorted(by_stem.items()):
        label_path = image_path.with_suffix(".txt")
        if not label_path.is_file():
            missing.append(stem)
            continue
        samples.append(
            Sample(sample_id=stem, image_path=image_path, label_path=label_path)
        )
    if missing:
        raise LayoutError(
            f"{len(missing)} imagenes sin .txt al lado: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    return tuple(samples)
