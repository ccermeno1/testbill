"""A run is a directory on disk. No MLflow or W&B.

`runs/{timestamp}_{name}/` with everything needed to know what ran, with which
code, on which data and under which license:

    config.yaml     the RESOLVED config, not the one passed on the command line
    run.json        provenance, components, fitness verdict
    metrics.json    the numbers
    weights/        the weights
    viz/            visualizations over the fixed validation sample

The fitness verdict is computed here and not in `compare`, because it depends
on things only known at run time: the detector's license, those of the
datasets in the chain and the state of the code tree.
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

#: Licenses that contaminate a closed product. The check is by explicit list
#: and not by heuristic over the text: getting this wrong is a legal problem,
#: not a bug.
COPYLEFT = ("AGPL-3.0", "GPL-3.0", "GPL-2.0", "SSPL-1.0")


@dataclass(frozen=True, slots=True)
class ComponentInfo:
    """Something with a license that enters a run: a dataset or a detector."""

    name: str
    license: str
    production_ready: bool
    #: Only for aggregated datasets: the license of EACH source. The license
    #: of an aggregate does not override that of its sources, so all are
    #: checked.
    sources: tuple[str, ...] = ()

    def blockers(self, role: str) -> list[str]:
        found: list[str] = []
        if not self.production_ready:
            found.append(
                f"{role} {self.name!r} is marked production_ready=False "
                f"(license {self.license})"
            )
        if self.license in COPYLEFT:
            found.append(f"{role} {self.name!r} has license {self.license}")
        for source in self.sources:
            if source in COPYLEFT:
                found.append(
                    f"{role} {self.name!r} aggregates a source with license "
                    f"{source}, which the aggregate's license does not override"
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
    #: Identity of the data export: project, version and date. The git hash
    #: pins the CODE; the images are not in the repository, so without this
    #: two runs of the same commit could have run on different annotations and
    #: nothing would give it away.
    dataset_provenance: tuple[dict, ...] = ()
    #: Which split version the run used (`SplitLoader.describe()`): version,
    #: mode and digest. Two runs on different versions are not comparable,
    #: and `compare` shows the version so that is visible.
    split: dict = field(default_factory=dict)
    #: Offline augmentation added to the train split (`augmented.describe`):
    #: version, digest, copies, recipe. Empty when trained on originals only.
    augmented: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def license_blockers(self) -> list[str]:
        """What prevents taking this run to production.

        Deliberately separate from `provenance_blockers`: not being able to
        reproduce a run and not being able to deploy it are two different
        problems, with different fixes, and mixing them in a single list
        leaves the reader not knowing which one they are reading.
        """
        found: list[str] = []
        if self.detector is None:
            found.append("no detector was declared, so there is no license to review")
        else:
            found.extend(self.detector.blockers("the detector"))
        for dataset in self.datasets:
            found.extend(dataset.blockers("the dataset"))
        return found

    def provenance_blockers(self) -> list[str]:
        return list(self.provenance.blockers())

    @property
    def caveats(self) -> list[str]:
        """Things that change HOW these numbers must be read.

        They do not prevent deploying or repeating the run, so they are not
        blockers. But without them up front, the figures are misread, and that
        is worse than not having them.
        """
        found: list[str] = []
        detector = self.config.detector
        if (
            str(getattr(detector.out_of_bounds, "value", detector.out_of_bounds)) == "pad"
            and not detector.pad_at_inference
        ):
            found.append(
                "trained with padding but inferred WITHOUT it: these numbers "
                "measure the mismatched pipeline. If they come out bad they do "
                "not prove padding is useless, only that training and "
                "predicting with different framings does not work. To judge "
                "padding, pad_at_inference=True."
            )
        return found

    def blockers(self) -> list[str]:
        return self.license_blockers() + self.provenance_blockers()

    @property
    def production_ready(self) -> bool:
        """Fit only if NOTHING in the chain prevents it.

        An Apache detector on a dataset with an AGPL source is not fit. That
        is the point of the check walking the whole chain.
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
            "split": dict(self.split),
            "augmented": dict(self.augmented),
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
    """Directory of a run, already created on disk."""

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
        split: dict | None = None,
        augmented: dict | None = None,
        runs_dir: Path | None = None,
        seed: int | None = None,
        moment: datetime | None = None,
        notes: tuple[str, ...] = (),
    ) -> ExperimentRun:
        """Creates the directory and freezes config and provenance BEFORE running.

        Writing the config after training would let a run that crashed
        halfway leave no trace of what it was launched with, which is exactly
        when it is most needed.
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
            split=dict(split or {}),
            augmented=dict(augmented or {}),
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
    """Reads a run from disk. Returns the raw record plus the metrics.

    Deliberately does not rebuild `RunRecord`: an old run may have been
    written with another version of the schema, and `compare` has to be able
    to list it anyway instead of blowing up.
    """
    directory = Path(directory)
    record_path = directory / RUN_RECORD
    if not record_path.exists():
        raise FileNotFoundError(f"{directory}: no {RUN_RECORD}")
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
    """All runs in `runs/`, from most recent to oldest."""
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
