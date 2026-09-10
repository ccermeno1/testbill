"""Registro de datasets, con licencia y procedencia.

Por que se declara aqui y no se lee del README del export
---------------------------------------------------------
`README.dataset.txt` trae la licencia, pero parsearlo convertiria una afirmacion
legal en una heuristica sobre texto libre. Se declara en codigo, se revisa en el
diff y se versiona. Equivocarse aqui es un problema legal, no un bug.

Por que la version del export tambien esta aqui
-----------------------------------------------
Las imagenes NO estan en el repositorio -- son 16 MB cuya fuente de verdad es
Roboflow -- asi que el hash de git fija el codigo pero no los datos. Registrar
proyecto, version y fecha de export en cada ejecucion es lo que cierra ese hueco:
sin ello, dos ejecuciones con el mismo commit podrian haber corrido sobre
anotaciones distintas y nada lo delataria.

`sources` lleva la licencia de CADA fuente, porque la licencia de un agregado no
anula la de sus fuentes. `compare` recorre la cadena entera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from testbank.experiment.run import COPYLEFT, ComponentInfo


class DatasetError(ValueError):
    """El dataset no esta registrado o su declaracion es incoherente."""


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    name: str
    root: Path
    license: str
    production_ready: bool
    #: Licencia de cada fuente del agregado. Un dataset de una sola fuente lleva
    #: la suya repetida, para que el campo nunca este vacio por descuido.
    sources: tuple[str, ...]
    #: Identidad del export. Fija los DATOS, que el commit de git no fija.
    origin: str = ""
    version: str = ""
    exported: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.sources:
            raise DatasetError(
                f"{self.name}: `sources` no puede estar vacio; si el dataset "
                "tiene una sola fuente, repite su licencia"
            )
        contaminated = [s for s in self.sources if s in COPYLEFT]
        if self.production_ready and contaminated:
            raise DatasetError(
                f"{self.name}: declarado production_ready=True pero agrega "
                f"fuentes con licencia {contaminated}; la licencia del agregado "
                "no anula la de sus fuentes"
            )

    def component(self) -> ComponentInfo:
        return ComponentInfo(
            name=self.name,
            license=self.license,
            production_ready=self.production_ready,
            sources=self.sources,
        )

    def provenance(self) -> dict:
        return {
            "name": self.name,
            "origin": self.origin,
            "version": self.version,
            "exported": self.exported,
            "license": self.license,
            "sources": list(self.sources),
        }


REGISTRY: dict[str, DatasetInfo] = {}


def register(info: DatasetInfo) -> DatasetInfo:
    if info.name in REGISTRY:
        raise DatasetError(f"dataset duplicado en el registro: {info.name!r}")
    REGISTRY[info.name] = info
    return info


def get(name: str) -> DatasetInfo:
    try:
        return REGISTRY[name]
    except KeyError:
        raise DatasetError(
            f"dataset desconocido: {name!r}; registrados: {sorted(REGISTRY)}"
        ) from None


def datasets() -> list[str]:
    return sorted(REGISTRY)


# --- lo que hay hoy --------------------------------------------------------

_ATTRIBUTION = (
    "CC BY 4.0 permite uso comercial, pero EXIGE atribucion: hay que citar la "
    "fuente alli donde se distribuya el modelo o los datos."
)
_RESIZED = (
    "489 de 502 imagenes son 416x416, ya redimensionadas en origen: el aspecto "
    "real de los billetes esta distorsionado y no se recupera."
)

ANNOTATED_BANKNOTES_2 = register(
    DatasetInfo(
        name="annotated-banknotes-2",
        root=Path("data/Annotated banknotes 2.yolov8-obb"),
        license="CC-BY-4.0",
        production_ready=True,
        sources=("CC-BY-4.0",),
        origin="https://universe.roboflow.com/clara-cermeno/annotated-banknotes-2-rwayh",
        version="v1",
        exported="2026-09-10",
        notes=(_ATTRIBUTION, _RESIZED),
    )
)

DEFAULT_DATASET = ANNOTATED_BANKNOTES_2.name
