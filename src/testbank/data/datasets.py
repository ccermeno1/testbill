"""Dataset registry, with license and provenance.

Why it is declared here and not read from the export's README
-------------------------------------------------------------
`README.dataset.txt` carries the license, but parsing it would turn a legal
statement into a heuristic over free text. It is declared in code, reviewed in
the diff and versioned. Getting this wrong is a legal problem, not a bug.

Why the export version is here too
----------------------------------
The images are NOT in the repository -- they are 16 MB whose source of truth
is Roboflow -- so the git hash pins the code but not the data. Recording
project, version and export date in every run is what closes that gap:
without it, two runs on the same commit could have run on different
annotations and nothing would give it away.

`sources` carries the license of EACH source, because the license of an
aggregate does not override those of its sources. `compare` walks the whole
chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from testbank.experiment.run import COPYLEFT, ComponentInfo


class DatasetError(ValueError):
    """The dataset is not registered or its declaration is inconsistent."""


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    name: str
    root: Path
    license: str
    production_ready: bool
    #: License of each source of the aggregate. A single-source dataset carries
    #: its own repeated, so the field is never empty by oversight.
    sources: tuple[str, ...]
    #: Identity of the export. Pins the DATA, which the git commit does not.
    origin: str = ""
    version: str = ""
    exported: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.sources:
            raise DatasetError(
                f"{self.name}: `sources` cannot be empty; if the dataset has a "
                "single source, repeat its license"
            )
        contaminated = [s for s in self.sources if s in COPYLEFT]
        if self.production_ready and contaminated:
            raise DatasetError(
                f"{self.name}: declared production_ready=True but aggregates "
                f"sources with license {contaminated}; the aggregate's license "
                "does not override that of its sources"
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
        raise DatasetError(f"duplicate dataset in the registry: {info.name!r}")
    REGISTRY[info.name] = info
    return info


def get(name: str) -> DatasetInfo:
    try:
        return REGISTRY[name]
    except KeyError:
        raise DatasetError(
            f"unknown dataset: {name!r}; registered: {sorted(REGISTRY)}"
        ) from None


def datasets() -> list[str]:
    return sorted(REGISTRY)


# --- what exists today -----------------------------------------------------

_ATTRIBUTION = (
    "CC BY 4.0 allows commercial use but REQUIRES attribution: the source must "
    "be cited wherever the model or the data are distributed."
)
_RESIZED = (
    "489 of 502 images are 416x416, already resized at the source: the real "
    "aspect of the banknotes is distorted and cannot be recovered."
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
