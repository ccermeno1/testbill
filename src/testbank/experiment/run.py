"""Una ejecucion es un directorio en disco. Nada de MLflow ni W&B.

`runs/{timestamp}_{nombre}/` con todo lo necesario para saber que se ejecuto,
con que codigo, sobre que datos y con que licencia:

    config.yaml     la config RESUELTA, no la que se paso por linea de comandos
    run.json        procedencia, componentes, veredicto de aptitud
    metrics.json    los numeros
    weights/        los pesos
    viz/            visualizaciones sobre la muestra fija de validacion

El veredicto de aptitud se calcula aqui y no en `compare`, porque depende de
cosas que solo se conocen en el momento de ejecutar: la licencia del detector,
las de los datasets de la cadena y el estado del arbol de codigo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from testbank.config import Config
from testbank.experiment.provenance import Provenance, collect

RUN_RECORD = "run.json"
CONFIG_SNAPSHOT = "config.yaml"
METRICS = "metrics.json"

#: Licencias que contaminan un producto cerrado. La comprobacion es por lista
#: explicita y no por heuristica sobre el texto: equivocarse aqui es un problema
#: legal, no un bug.
COPYLEFT = ("AGPL-3.0", "GPL-3.0", "GPL-2.0", "SSPL-1.0")


@dataclass(frozen=True, slots=True)
class ComponentInfo:
    """Algo con licencia que entra en una ejecucion: un dataset o un detector."""

    name: str
    license: str
    production_ready: bool
    #: Solo para datasets agregados: la licencia de CADA fuente. La licencia de
    #: un agregado no anula la de sus fuentes, asi que se comprueban todas.
    sources: tuple[str, ...] = ()

    def blockers(self, role: str) -> list[str]:
        found: list[str] = []
        if not self.production_ready:
            found.append(
                f"{role} {self.name!r} esta marcado production_ready=False "
                f"(licencia {self.license})"
            )
        if self.license in COPYLEFT:
            found.append(f"{role} {self.name!r} tiene licencia {self.license}")
        for source in self.sources:
            if source in COPYLEFT:
                found.append(
                    f"{role} {self.name!r} agrega una fuente con licencia "
                    f"{source}, que la licencia del agregado no anula"
                )
        return found


@dataclass
class RunRecord:
    name: str
    timestamp: str
    provenance: Provenance
    config: Config
    detector: ComponentInfo | None = None
    datasets: tuple[ComponentInfo, ...] = ()
    metrics: dict = field(default_factory=dict)
    #: Identidad del export de datos: proyecto, version y fecha. El hash de git
    #: fija el CODIGO; las imagenes no estan en el repositorio, asi que sin esto
    #: dos ejecuciones del mismo commit podrian haber corrido sobre anotaciones
    #: distintas y nada lo delataria.
    dataset_provenance: tuple[dict, ...] = ()
    notes: tuple[str, ...] = ()

    def license_blockers(self) -> list[str]:
        """Lo que impide llevar esta ejecucion a produccion.

        Separado a proposito de `provenance_blockers`: no poder reproducir una
        ejecucion y no poder desplegarla son dos problemas distintos, con
        arreglos distintos, y mezclarlos en una sola lista hace que el lector
        no sepa cual esta leyendo.
        """
        found: list[str] = []
        if self.detector is None:
            found.append("no se declaro detector, asi que no hay licencia que revisar")
        else:
            found.extend(self.detector.blockers("el detector"))
        for dataset in self.datasets:
            found.extend(dataset.blockers("el dataset"))
        return found

    def provenance_blockers(self) -> list[str]:
        return list(self.provenance.blockers())

    @property
    def caveats(self) -> list[str]:
        """Cosas que cambian COMO HAY QUE LEER estos numeros.

        No impiden desplegar ni repetir la ejecucion, asi que no son bloqueos.
        Pero sin ellas delante, las cifras se interpretan mal, y eso es peor que
        no tenerlas.
        """
        found: list[str] = []
        detector = self.config.detector
        if (
            str(getattr(detector.out_of_bounds, "value", detector.out_of_bounds)) == "pad"
            and not detector.pad_at_inference
        ):
            found.append(
                "entrenado con padding pero inferido SIN el: estos numeros miden "
                "el pipeline desajustado. Si salen mal no prueban que el padding "
                "no sirva, solo que entrenar y predecir con encuadres distintos "
                "no funciona. Para juzgar el padding, pad_at_inference=True."
            )
        return found

    def blockers(self) -> list[str]:
        return self.license_blockers() + self.provenance_blockers()

    @property
    def production_ready(self) -> bool:
        """Apto solo si NADA de la cadena lo impide.

        Un detector Apache sobre un dataset con una fuente AGPL no es apto. Es
        el punto de que la comprobacion recorra la cadena entera.
        """
        if self.detector is None:
            return False
        if not self.detector.production_ready:
            return False
        return not any(
            not d.production_ready or d.license in COPYLEFT or
            any(s in COPYLEFT for s in d.sources)
            for d in self.datasets
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "provenance": self.provenance.to_dict(),
            "detector": asdict(self.detector) if self.detector else None,
            "datasets": [asdict(d) for d in self.datasets],
            "dataset_provenance": list(self.dataset_provenance),
            "production_ready": self.production_ready,
            "reproducible": self.provenance.reproducible,
            "caveats": self.caveats,
            "license_blockers": self.license_blockers(),
            "provenance_blockers": self.provenance_blockers(),
            "notes": list(self.notes),
        }


def _timestamp(moment: datetime | None = None) -> str:
    moment = moment or datetime.now(UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Directorio de una ejecucion, ya creado en disco."""

    directory: Path
    record: RunRecord

    @property
    def weights_dir(self) -> Path:
        return self.directory / "weights"

    @property
    def viz_dir(self) -> Path:
        return self.directory / "viz"

    @classmethod
    def create(
        cls,
        name: str,
        config: Config,
        *,
        detector: ComponentInfo | None = None,
        datasets: tuple[ComponentInfo, ...] = (),
        dataset_provenance: tuple[dict, ...] = (),
        runs_dir: Path | None = None,
        seed: int | None = None,
        moment: datetime | None = None,
        notes: tuple[str, ...] = (),
    ) -> ExperimentRun:
        """Crea el directorio y congela config y procedencia ANTES de ejecutar.

        Escribir la config despues de entrenar permitiria que una ejecucion que
        peto por la mitad no dejara rastro de con que se lanzo, que es justo
        cuando mas falta hace.
        """
        stamp = _timestamp(moment)
        base = Path(runs_dir or config.runs_dir)
        directory = base / f"{stamp}_{name}"
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "weights").mkdir()
        (directory / "viz").mkdir()

        record = RunRecord(
            name=name,
            timestamp=stamp,
            provenance=collect(seed if seed is not None else config.metrics.seed),
            config=config,
            detector=detector,
            datasets=datasets,
            dataset_provenance=dataset_provenance,
            notes=notes,
        )
        (directory / CONFIG_SNAPSHOT).write_text(config.dump_yaml(), encoding="utf-8")
        run = cls(directory=directory, record=record)
        run._write_record()
        return run

    def _write_record(self) -> None:
        (self.directory / RUN_RECORD).write_text(
            json.dumps(self.record.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def write_metrics(self, metrics: dict) -> Path:
        self.record.metrics = metrics
        path = self.directory / METRICS
        path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._write_record()
        return path

    def add_note(self, note: str) -> None:
        self.record.notes = (*self.record.notes, note)
        self._write_record()


def load_run(directory: str | Path) -> dict:
    """Lee una ejecucion de disco. Devuelve el registro crudo mas las metricas.

    Deliberadamente no reconstruye `RunRecord`: una ejecucion antigua pudo
    escribirse con otra version del esquema, y `compare` tiene que poder
    listarla igualmente en vez de reventar.
    """
    directory = Path(directory)
    record_path = directory / RUN_RECORD
    if not record_path.exists():
        raise FileNotFoundError(f"{directory}: no hay {RUN_RECORD}")
    data = json.loads(record_path.read_text(encoding="utf-8"))
    metrics_path = directory / METRICS
    data["metrics"] = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )
    data["directory"] = str(directory)
    return data


def discover_runs(runs_dir: str | Path) -> list[dict]:
    """Todas las ejecuciones de `runs/`, de la mas reciente a la mas antigua."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    found = []
    for child in sorted(runs_dir.iterdir(), reverse=True):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / RUN_RECORD).exists():
            found.append(load_run(child))
    return found
