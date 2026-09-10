"""Unica puerta de acceso a las anotaciones de una muestra.

`read_label_file` lee un fichero y valida su formato. Aqui se compone con el
aspecto de la imagen y con el filtro de area relativa, y es lo que consumen la
visualizacion, los chequeos y todo lo que venga despues. Un solo sitio decide
que anotaciones estan vigentes, igual que `splits.py` es el unico que decide a
que particion pertenece una muestra.

Filtro de area relativa
-----------------------
En cada imagen se conserva el billete de delante: se descarta toda anotacion
cuya area sea menor que `min_relative_area` veces el area de la mayor anotacion
de esa misma imagen.

Cubre las dos situaciones sin tener que distinguirlas. En un abanico, la franja
visible de un billete tapado es mucho menor que el billete de delante y cae. En
una foto de dos o tres billetes juntos, todos tienen un tamano parecido y se
conservan todos.

Es un FILTRO EN CARGA, no un borrado: los ficheros de anotacion no se tocan
nunca, y lo descartado se reporta con imagen, indice y porcentaje de area
relativa para poder revisarlo y corregirlo en el origen.

Por que no hace falta el tamano de la imagen
--------------------------------------------
El criterio es un cociente de areas dentro de una misma imagen. Pasar de
coordenadas normalizadas a pixeles multiplica ambas areas por el mismo W*H, asi
que el cociente no cambia. El aspecto se sigue pidiendo, pero solo para que el
orden canonico de los quads sea el correcto, no para el filtro.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from testbank.data.discover import Sample
from testbank.dataio.obb_yolo import Annotation, LabelFormatError, read_label_file
from testbank.geometry.quad import DEFAULT_ASPECT, Quad

#: Fraccion del area de la mayor anotacion por debajo de la cual se descarta.
#: Aproximacion a la politica de visibilidad del 25%, no una medida de oclusion.
DEFAULT_MIN_RELATIVE_AREA = 0.25


@dataclass(frozen=True, slots=True)
class DroppedAnnotation:
    """Una anotacion que el filtro deja fuera. Sigue en el fichero de origen."""

    sample_id: str
    #: Indice de la anotacion dentro de su imagen, contando desde 0, en el mismo
    #: orden en que aparece en el fichero. Es el indice que usa el informe de
    #: visibilidad, para poder cruzar los dos informes.
    index: int
    #: area / area de la mayor anotacion de la imagen, en [0, 1).
    relative_area: float
    label_path: Path
    quad: Quad

    def describe(self) -> str:
        return (
            f"{self.sample_id}: anotacion #{self.index} descartada, "
            f"area relativa {self.relative_area:.1%}"
        )


@dataclass(frozen=True, slots=True)
class LoadedSample:
    """Anotaciones vigentes de una muestra, mas lo que se dejo fuera y por que."""

    sample_id: str
    label_path: Path
    aspect: float
    annotations: tuple[Annotation, ...]
    #: Posicion original de cada anotacion conservada dentro de su fichero,
    #: contando desde 0. Filtrar renumera, y un informe que dijera "anotacion
    #: #2" refiriendose a la posicion ya filtrada apuntaria a la anotacion
    #: equivocada al abrir el fichero en Roboflow. Todo informe aguas abajo usa
    #: estos indices, de modo que el del filtro y el de visibilidad se cruzan.
    kept_indices: tuple[int, ...] = ()
    dropped: tuple[DroppedAnnotation, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def quads(self) -> tuple[Quad, ...]:
        return tuple(a.quad for a in self.annotations)


@dataclass
class FilterReport:
    """Recuento de lo que el filtro deja fuera, para revisarlo en Roboflow."""

    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA
    images_loaded: int = 0
    annotations_read: int = 0
    dropped: list[DroppedAnnotation] = field(default_factory=list)

    @property
    def annotations_kept(self) -> int:
        return self.annotations_read - len(self.dropped)

    @property
    def images_affected(self) -> int:
        return len({d.sample_id for d in self.dropped})

    def to_dict(self) -> dict:
        return {
            "min_relative_area": self.min_relative_area,
            "images_loaded": self.images_loaded,
            "annotations_read": self.annotations_read,
            "annotations_kept": self.annotations_kept,
            "annotations_dropped": len(self.dropped),
            "images_affected": self.images_affected,
            "note": (
                "Filtro en carga, aproximacion a la politica de visibilidad del "
                "25%: no mide oclusion. Las anotaciones listadas siguen en el "
                "dataset de origen, no se ha borrado nada."
            ),
            "dropped": [
                {
                    "sample_id": d.sample_id,
                    "index": d.index,
                    "relative_area": round(d.relative_area, 6),
                    "label_path": str(d.label_path),
                }
                for d in sorted(self.dropped, key=lambda d: (d.sample_id, d.index))
            ],
        }

    def summary_lines(self) -> list[str]:
        umbral = (
            f"Filtro de area relativa: umbral {self.min_relative_area:.0%} "
            f"del billete mas grande de cada imagen"
        )
        descartadas = (
            f"Anotaciones descartadas: {len(self.dropped)} "
            f"en {self.images_affected} imagenes"
        )
        return [
            umbral,
            f"Imagenes cargadas:       {self.images_loaded}",
            f"Anotaciones leidas:      {self.annotations_read}",
            f"Anotaciones conservadas: {self.annotations_kept}",
            descartadas,
        ]


def filter_by_relative_area(
    annotations: tuple[Annotation, ...] | list[Annotation],
    *,
    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA,
) -> tuple[tuple[tuple[int, Annotation], ...], tuple[tuple[int, float], ...]]:
    """Conserva el billete de delante y lo que se le parezca en tamano.

    Devuelve `(conservadas, descartadas)`, donde `conservadas` son pares
    `(indice original, anotacion)` y `descartadas` son pares `(indice original,
    area relativa)`. Los indices son siempre la posicion dentro del fichero, no
    la posicion tras filtrar, para que cualquier informe apunte a la anotacion
    que de verdad hay que abrir en Roboflow.

    La mayor anotacion tiene area relativa 1.0 y por tanto nunca se descarta,
    ni siquiera con `min_relative_area = 1.0`: la comparacion es estricta, de
    modo que el umbral 1.0 conserva la mayor y sus empates exactos.
    """
    if not 0.0 <= min_relative_area <= 1.0:
        raise ValueError(
            f"min_relative_area debe estar en [0, 1], se recibio {min_relative_area}"
        )
    annotations = tuple(annotations)
    if not annotations:
        return (), ()

    areas = [abs(a.quad.signed_area()) for a in annotations]
    largest = max(areas)
    if largest <= 0.0:  # pragma: no cover - el lector ya rechaza area nula
        return tuple(enumerate(annotations)), ()

    kept: list[tuple[int, Annotation]] = []
    dropped: list[tuple[int, float]] = []
    for index, (annotation, area) in enumerate(zip(annotations, areas)):
        relative = area / largest
        if relative < min_relative_area:
            dropped.append((index, relative))
        else:
            kept.append((index, annotation))
    return tuple(kept), tuple(dropped)


def load_sample(
    sample: Sample,
    *,
    aspect: float = DEFAULT_ASPECT,
    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA,
) -> LoadedSample:
    """Lee las anotaciones de una muestra y les aplica el filtro de area."""
    label = read_label_file(sample.label_path, aspect=aspect)
    kept, dropped = filter_by_relative_area(
        label.annotations, min_relative_area=min_relative_area
    )
    return LoadedSample(
        sample_id=sample.sample_id,
        label_path=sample.label_path,
        aspect=aspect,
        annotations=tuple(a for _, a in kept),
        kept_indices=tuple(i for i, _ in kept),
        dropped=tuple(
            DroppedAnnotation(
                sample_id=sample.sample_id,
                index=index,
                relative_area=relative,
                label_path=sample.label_path,
                quad=label.annotations[index].quad,
            )
            for index, relative in dropped
        ),
        warnings=label.warnings,
    )


def load_samples(
    samples,
    *,
    sizes=None,
    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA,
) -> tuple[tuple[LoadedSample, ...], FilterReport]:
    """Carga un conjunto de muestras y acumula el informe del filtro.

    `sizes` es un `SizeIndex`; sin el, el orden canonico se calcula en el
    espacio normalizado, que no es el geometrico. Pasalo siempre que lo tengas.
    """
    report = FilterReport(min_relative_area=min_relative_area)
    loaded: list[LoadedSample] = []
    problems: list[str] = []
    for sample in samples:
        aspect = sizes.aspect(sample.sample_id) if sizes else DEFAULT_ASPECT
        try:
            item = load_sample(
                sample, aspect=aspect, min_relative_area=min_relative_area
            )
        except LabelFormatError as exc:
            # Igual que dentro de un fichero: se recorren todos y se falla al
            # final con la lista completa. Con 500 etiquetas, enterarse de un
            # error por ejecucion convierte la limpieza en un bucle.
            problems.extend(exc.problems)
            continue
        loaded.append(item)
        report.images_loaded += 1
        report.annotations_read += len(item.annotations) + len(item.dropped)
        report.dropped.extend(item.dropped)

    if problems:
        raise LabelFormatError.combine(
            problems, "anotaciones invalidas en el dataset"
        )

    return tuple(loaded), report
