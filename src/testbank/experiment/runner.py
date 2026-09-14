"""Launches a candidate and leaves the whole run on disk.

It is the glue between the three pieces that already exist: `SplitLoader`
decides on which data, the detector trains and predicts, and `metrics`
scores. There is no logic of its own here beyond the order and what is
recorded.

Three things done on purpose
----------------------------
1. **Test is not touched.** `train` and `valid` are loaded and nothing else.
   `SplitLoader.load` requires `allow_test=True` and it is never passed here:
   the seal is broken only with `evaluate-test`, which keeps its own access
   log.

2. **Config and provenance are frozen BEFORE training.** A training that
   blows up after three hours has to leave a record of what it was launched
   with.

3. **The weights are copied inside the run.** The trainer leaves them
   wherever it likes; if the run does not carry them, in two weeks nobody
   knows which weights produced that `metrics.json`.
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
        split = self.run.record.split
        lines = [
            f"Run: {self.run.directory}",
            f"  split {split.get('version', '?')} ({split.get('mode', '?')})",
            f"  images {metrics.get('n_images')}  annotations {metrics.get('n_truths')}",
            f"  mAP50        {_interval(metrics.get('map50'))}",
            f"  mAP50-95     {_interval(metrics.get('map50_95'))}",
            f"  coverage p5  {_interval(metrics.get('coverage_p5'))}",
        ]
        decided = (metrics.get("detection") or {}).get("at_decision_confidence")
        if decided:
            lines.append(
                f"  at confidence {metrics['detection']['decision_confidence']:.2f}: "
                f"{decided['true_positives']} hits, "
                f"{decided['false_positives']} false positives, "
                f"detection {decided['detection_rate']:.3f}"
            )
        for scene in ("single", "fan"):
            entry = contamination.get(scene)
            if entry and entry.get("n"):
                mark = "" if entry["passes"] else "  <- EXCEEDS THE THRESHOLD"
                lines.append(
                    f"  contamination {scene:6s} p95 {entry['p95']:.3f} "
                    f"/ threshold {entry['threshold']:.2f}  (median "
                    f"{entry['median']:.3f}, n={entry['n']}){mark}"
                )
        for warning in self.run.record.caveats:
            lines.append(f"  WARNING: {warning}")
        if not self.run.record.production_ready:
            lines.append("  NOT READY for production:")
            for blocker in self.run.record.license_blockers():
                lines.append(f"    - {blocker}")
        for blocker in self.run.record.provenance_blockers():
            lines.append(f"  NOT REPRODUCIBLE: {blocker}")
        return "\n".join(lines)


def _interval(entry) -> str:
    if not isinstance(entry, dict) or entry.get("value") is None:
        return "-"
    low, high = entry.get("ci_low"), entry.get("ci_high")
    if low is None or high is None or low != low:  # noqa: PLR0124 - NaN
        return f"{entry['value']:.3f} (no CI, n={entry.get('n')})"
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
    split_version: str | None = None,
    augmented: Path | None = None,
) -> RunOutcome:
    """Trains (or reuses weights), evaluates on `valid` and leaves everything written.

    `split_version` picks the split (`None` = latest). Whichever is resolved
    is recorded in the run, with its digest. `augmented` is a version of
    offline copies (`data/augmented/vN`) to ADD to the train split; it must
    have been made from the same split version, and it is recorded too.
    """
    loader = SplitLoader(splits_dir, data_root, version=split_version)
    samples = {split: list(loader.load(split)) for split in TRAIN_SPLITS}
    split = loader.describe()

    augmented_record: dict = {}
    notes: tuple[str, ...] = ()
    if augmented is not None:
        from testbank.data.augmented import describe, load_augmented

        copies, manifest = load_augmented(augmented, split=split)
        samples["train"] = samples["train"] + copies
        augmented_record = describe(manifest)
        recipe = manifest["recipe"]
        note = (
            f"offline augmentation {manifest['version']}: {manifest['copies']} copies of "
            f"{manifest['sources']} training photos added (rotation +-{recipe['degrees']}, "
            f"shear {recipe['shear']}, perspective {recipe['perspective']})"
        )
        notes = (note,)

    dataset = get_dataset(dataset_name)
    run = ExperimentRun.create(
        run_name or detector.name,
        config,
        detector=detector.component(),
        datasets=(dataset.component(),),
        dataset_provenance=(dataset.provenance(),),
        split=split,
        augmented=augmented_record,
        notes=(*detector.notes, *dataset.notes, *notes),
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


# --- the sealed set --------------------------------------------------------

TEST_METRICS = "metrics_test.json"


@dataclass(frozen=True, slots=True)
class TestOutcome:
    run_directory: Path
    metrics: dict
    #: Accesses to the test PRIOR to this one. If not zero, it must be explained.
    previous_accesses: list[dict]
    entry: dict

    def summary(self) -> str:
        metrics = self.metrics
        lines = [
            f"Evaluation on TEST of {self.run_directory.name}",
            f"  images {metrics.get('n_images')}  annotations {metrics.get('n_truths')}",
            f"  mAP50        {_interval(metrics.get('map50'))}",
            f"  mAP50-95     {_interval(metrics.get('map50_95'))}",
            f"  coverage p5  {_interval(metrics.get('coverage_p5'))}",
        ]
        decided = (metrics.get("detection") or {}).get("at_decision_confidence")
        if decided:
            lines.append(
                f"  at confidence {metrics['detection']['decision_confidence']:.2f}: "
                f"{decided['true_positives']} hits, "
                f"{decided['false_positives']} false positives, "
                f"detection {decided['detection_rate']:.3f}"
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
    split_version: str | None = None,
) -> TestOutcome:
    """Breaks the seal, leaves a record and evaluates. In that order.

    It is recorded BEFORE evaluating. If it were recorded afterwards, an
    evaluation interrupted upon seeing a bad number would leave no trace, and
    the log would stop serving the only thing it serves: knowing how many
    times it has been looked at.

    `reason` is mandatory and goes to the log. It is not bureaucracy: the only
    real defence against tuning to the test is that every access has to be
    justified in writing and stays visible to the next person who looks.
    """
    if not reason.strip():
        raise SplitError("evaluate-test requires a reason; a log without a motive is useless")

    # The split is resolved BEFORE logging, so the log says which version's
    # test was opened: a test of v1 and a test of v2 are different sets.
    loader = SplitLoader(splits_dir, data_root, version=split_version)
    previous = read_test_accesses(log_path)
    entry = record_test_access(
        reason.strip(),
        run_directory.name,
        log_path,
        weights=str(weights),
        access_number=len(previous) + 1,
        split=loader.describe(),
    )

    # The only place in the project that passes allow_test=True.
    samples = list(loader.load("test", allow_test=True))
    metrics = detector.evaluate(samples, config, weights=weights)
    metrics["split"] = "test"
    metrics["split_version"] = loader.describe()
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
