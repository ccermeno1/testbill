"""Lanza un candidato y deja la ejecucion entera en disco.

Es el pegamento entre las tres piezas que ya existen: `SplitLoader` decide sobre
que datos, el detector entrena y predice, y `metrics` puntua. Aqui no hay logica
propia salvo el orden y lo que se registra.

Tres cosas que se hacen a proposito
-----------------------------------
1. **Test no se toca.** Se carga `train` y `valid` y nada mas. `SplitLoader.load`
   exige `allow_test=True` y aqui no se pasa nunca: el sello se rompe solo con
   `evaluate-test`, que lleva su propio registro de accesos.

2. **La config y la procedencia se congelan ANTES de entrenar.** Un entrenamiento
   que revienta a las tres horas tiene que dejar dicho con que se lanzo.

3. **Los pesos se copian dentro de la ejecucion.** El entrenador los deja donde
   le apetece; si la ejecucion no los tiene consigo, dentro de dos semanas nadie
   sabe que pesos produjeron ese `metrics.json`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from testbank.config import Config
from testbank.data.datasets import DEFAULT_DATASET
from testbank.data.datasets import get as get_dataset
from testbank.data.splits import (
    SplitError,
    SplitLoader,
    read_test_accesses,
    record_test_access,
)
from testbank.detectors.base import BaseDetector
from testbank.experiment.run import ExperimentRun
from testbank.viz.inspect import fixed_validation_sample, inspect_samples

TRAIN_SPLITS = ("train", "valid")


@dataclass(frozen=True, slots=True)
class RunOutcome:
    run: ExperimentRun
    weights: Path
    metrics: dict

    def summary(self) -> str:
        metrics = self.metrics
        contamination = (metrics.get("contamination") or {}).get("by_scene", {})
        lines = [
            f"Ejecucion: {self.run.directory}",
            f"  imagenes {metrics.get('n_images')}  anotaciones {metrics.get('n_truths')}",
            f"  mAP50        {_interval(metrics.get('map50'))}",
            f"  mAP50-95     {_interval(metrics.get('map50_95'))}",
            f"  cobertura p5 {_interval(metrics.get('coverage_p5'))}",
        ]
        decided = (metrics.get("detection") or {}).get("at_decision_confidence")
        if decided:
            lines.append(
                f"  a confianza {metrics['detection']['decision_confidence']:.2f}: "
                f"{decided['true_positives']} aciertos, "
                f"{decided['false_positives']} falsos positivos, "
                f"deteccion {decided['detection_rate']:.3f}"
            )
        for scene in ("single", "fan"):
            entry = contamination.get(scene)
            if entry and entry.get("n"):
                mark = "" if entry["passes"] else "  <- SUPERA EL UMBRAL"
                lines.append(
                    f"  contaminacion {scene:6s} p95 {entry['p95']:.3f} "
                    f"/ umbral {entry['threshold']:.2f}  (mediana "
                    f"{entry['median']:.3f}, n={entry['n']}){mark}"
                )
        for warning in self.run.record.caveats:
            lines.append(f"  AVISO: {warning}")
        if not self.run.record.production_ready:
            lines.append("  NO APTO para produccion:")
            for blocker in self.run.record.license_blockers():
                lines.append(f"    - {blocker}")
        for blocker in self.run.record.provenance_blockers():
            lines.append(f"  NO REPRODUCIBLE: {blocker}")
        return "\n".join(lines)


def _interval(entry) -> str:
    if not isinstance(entry, dict) or entry.get("value") is None:
        return "-"
    low, high = entry.get("ci_low"), entry.get("ci_high")
    if low is None or high is None or low != low:  # noqa: PLR0124 - NaN
        return f"{entry['value']:.3f} (sin IC, n={entry.get('n')})"
    return f"{entry['value']:.3f} [{low:.3f}, {high:.3f}]  n={entry.get('n')}"


def run_candidate(
    detector: BaseDetector,
    config: Config,
    *,
    splits_dir: Path,
    data_root: Path | None = None,
    dataset_name: str = DEFAULT_DATASET,
    run_name: str | None = None,
    weights: Path | None = None,
) -> RunOutcome:
    """Entrena (o reutiliza pesos), evalua sobre `valid` y lo deja todo escrito."""
    loader = SplitLoader(splits_dir, data_root)
    samples = {split: list(loader.load(split)) for split in TRAIN_SPLITS}

    dataset = get_dataset(dataset_name)
    run = ExperimentRun.create(
        run_name or detector.name,
        config,
        detector=detector.component(),
        datasets=(dataset.component(),),
        dataset_provenance=(dataset.provenance(),),
        notes=(*detector.notes, *dataset.notes),
    )

    if weights is None:
        result = detector.train(samples, config, output_dir=run.directory / "_train")
        weights = result.weights
        for note in result.notes:
            run.add_note(note)

    stored = run.weights_dir / Path(weights).name
    if Path(weights).resolve() != stored.resolve():
        shutil.copy2(weights, stored)

    metrics = detector.evaluate(samples["valid"], config, weights=stored)
    run.write_metrics(metrics)

    shown = fixed_validation_sample(
        samples["valid"], count=config.viz.sample_count, seed=config.viz.seed
    )
    predictions = detector.predict(shown, weights=stored, config=config)
    predictions = {
        sample_id: [
            p for p in found if p.score >= config.viz.confidence_threshold
        ]
        for sample_id, found in predictions.items()
    }
    inspect_samples(
        shown,
        run.viz_dir,
        visibility_threshold=config.annotation_policy.visibility_threshold,
        min_relative_area=config.annotation_policy.min_relative_area,
        predictions=predictions,
    )
    return RunOutcome(run=run, weights=stored, metrics=metrics)


# --- el conjunto sellado ---------------------------------------------------

TEST_METRICS = "metrics_test.json"


@dataclass(frozen=True, slots=True)
class TestOutcome:
    run_directory: Path
    metrics: dict
    #: Accesos al test ANTERIORES a este. Si no es cero, hay que explicarlo.
    previous_accesses: list[dict]
    entry: dict

    def summary(self) -> str:
        metrics = self.metrics
        lines = [
            f"Evaluacion sobre TEST de {self.run_directory.name}",
            f"  imagenes {metrics.get('n_images')}  anotaciones {metrics.get('n_truths')}",
            f"  mAP50        {_interval(metrics.get('map50'))}",
            f"  mAP50-95     {_interval(metrics.get('map50_95'))}",
            f"  cobertura p5 {_interval(metrics.get('coverage_p5'))}",
        ]
        decided = (metrics.get("detection") or {}).get("at_decision_confidence")
        if decided:
            lines.append(
                f"  a confianza {metrics['detection']['decision_confidence']:.2f}: "
                f"{decided['true_positives']} aciertos, "
                f"{decided['false_positives']} falsos positivos, "
                f"deteccion {decided['detection_rate']:.3f}"
            )
        return "\n".join(lines)


def evaluate_on_test(
    detector: BaseDetector,
    config: Config,
    *,
    run_directory: Path,
    weights: Path,
    reason: str,
    splits_dir: Path,
    data_root: Path | None = None,
    log_path: Path | None = None,
) -> TestOutcome:
    """Rompe el sello, deja constancia y evalua. En ese orden.

    Se registra ANTES de evaluar. Si se registrara despues, una evaluacion que
    se interrumpe al ver un numero malo no dejaria rastro, y el registro dejaria
    de servir para lo unico que sirve: saber cuantas veces se ha mirado.

    `reason` es obligatorio y va al registro. No es burocracia: la unica defensa
    real contra ajustar al test es que cada acceso tenga que justificarse por
    escrito y quede a la vista del siguiente que mire.
    """
    if not reason.strip():
        raise SplitError("evaluate-test exige una razon; el registro sin motivo no sirve")

    previous = read_test_accesses(log_path)
    entry = record_test_access(
        reason.strip(),
        run_directory.name,
        log_path,
        weights=str(weights),
        access_number=len(previous) + 1,
    )

    loader = SplitLoader(splits_dir, data_root)
    # El unico sitio del proyecto que pasa allow_test=True.
    samples = list(loader.load("test", allow_test=True))
    metrics = detector.evaluate(samples, config, weights=weights)
    metrics["split"] = "test"
    metrics["test_access_number"] = entry["access_number"]

    (run_directory / TEST_METRICS).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return TestOutcome(
        run_directory=run_directory,
        metrics=metrics,
        previous_accesses=previous,
        entry=entry,
    )
