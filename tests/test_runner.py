"""The full path: split -> training -> metrics -> disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import rotated_rect_points
from PIL import Image

from testbank.config import Config, OutOfBoundsPolicy
from testbank.data import datasets as datasets_mod
from testbank.data.datasets import DatasetError, DatasetInfo
from testbank.data.splits import materialize_splits
from testbank.dataio.formats import get as get_format
from testbank.detectors import base as detector_base
from testbank.detectors.base import BaseDetector, TrainResult, register
from testbank.experiment.runner import run_candidate
from testbank.geometry.quad import Quad, canonicalize
from testbank.metrics.core import Prediction

FAKE_DETECTOR = "_test_fake"
FAKE_DATASET = "_test_dataset"


def _quad(cx, cy, half_long=0.12, ratio=2.0, theta=0.0) -> Quad:
    return canonicalize(
        Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta))
    )


QUADS = (_quad(0.3, 0.5), _quad(0.7, 0.5))


def _build_dataset(root: Path, counts: dict[str, int]) -> None:
    """Minimal Roboflow tree: train/valid/test with images/ and labels/."""
    writer = get_format("obb_yolo")
    payload = "\n".join(f"0 {writer.from_quad(q).payload}" for q in QUADS) + "\n"
    for split, count in counts.items():
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            sample_id = f"{split}_{i:03d}"
            Image.new("RGB", (160, 120), (40, 40, 40)).save(images / f"{sample_id}.jpg")
            (labels / f"{sample_id}.txt").write_text(payload, encoding="utf-8")


class _FakeDetector(BaseDetector):
    """Trains for pretend and predicts the truth. Exercises the path, not the model."""

    name = FAKE_DETECTOR
    license = "Apache-2.0"
    production_ready = True
    notes = ("test detector",)

    #: Splits that `train` received. It is how it is checked that test is not touched.
    seen_splits: tuple[str, ...] = ()

    def train(self, samples_by_split, config, *, output_dir: Path) -> TrainResult:
        type(self).seen_splits = tuple(sorted(samples_by_split))
        output_dir.mkdir(parents=True, exist_ok=True)
        weights = output_dir / "best.pt"
        weights.write_bytes(b"fake weights")
        return TrainResult(weights=weights, epochs=1, notes=("fake training",))

    def predict(self, samples, *, weights, config):
        return {s.sample_id: [Prediction(q, 0.9) for q in QUADS] for s in samples}


@pytest.fixture()
def registered():
    register(_FakeDetector)
    info = DatasetInfo(
        name=FAKE_DATASET,
        root=Path("."),
        license="CC-BY-4.0",
        production_ready=True,
        sources=("CC-BY-4.0",),
        origin="https://example/project",
        version="v9",
        exported="2026-09-10",
    )
    datasets_mod.register(info)
    try:
        yield detector_base.REGISTRY[FAKE_DETECTOR]
    finally:
        del detector_base.REGISTRY[FAKE_DETECTOR]
        del datasets_mod.REGISTRY[FAKE_DATASET]


@pytest.fixture()
def prepared(tmp_path):
    root = tmp_path / "data"
    _build_dataset(root, {"train": 6, "valid": 4, "test": 3})
    splits_dir = tmp_path / "splits"
    materialize_splits(root, splits_dir)
    base = Config()
    config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={"derived_dir": tmp_path / "derived", "splits_dir": splits_dir}
            ),
            "metrics": base.metrics.model_copy(update={"bootstrap_samples": 30}),
            "viz": base.viz.model_copy(update={"sample_count": 2}),
            "runs_dir": tmp_path / "runs",
        }
    )
    return root, splits_dir, config


def test_the_run_leaves_everything_on_disk(registered, prepared):
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )

    directory = outcome.run.directory
    assert (directory / "config.yaml").exists()
    assert (directory / "run.json").exists()
    assert (directory / "metrics.json").exists()
    assert outcome.weights.exists()
    assert outcome.weights.parent == outcome.run.weights_dir
    assert list(outcome.run.viz_dir.glob("*.png"))


def test_test_is_never_loaded(registered, prepared):
    """The seal is broken only by `evaluate-test`, which keeps its own log."""
    root, splits_dir, config = prepared
    run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    assert _FakeDetector.seen_splits == ("train", "valid")


def test_evaluation_is_on_valid_not_on_train(registered, prepared):
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    assert outcome.metrics["n_images"] == 4


def test_the_split_version_is_recorded_in_the_run(registered, prepared):
    """Two runs on different split versions were evaluated on different
    images; without this the table would compare them as equals."""
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    record = json.loads((outcome.run.directory / "run.json").read_text(encoding="utf-8"))
    assert record["split"]["version"] == "v1"
    assert record["split"]["mode"] == "adopt"
    assert len(record["split"]["digest"]) == 64
    assert "split v1" in outcome.summary()


def test_a_pinned_split_version_is_the_one_used(registered, prepared):
    from testbank.data.splits import GroupConfig, materialize_splits

    root, splits_dir, config = prepared
    materialize_splits(
        root, splits_dir, repartition=True, ratios=(0.5, 0.25, 0.25), seed=1,
        groups=GroupConfig(independence_confirmed=True),
    )
    latest = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    pinned = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET, split_version="v1", run_name="pinned",
    )
    assert latest.run.record.split["version"] == "v2"
    assert pinned.run.record.split["version"] == "v1"
    assert pinned.metrics["n_images"] == 4  # v1's valid: the export's 4


def test_the_dataset_provenance_is_recorded(registered, prepared):
    """The commit pins the code; the images are not in git, so without this
    two runs of the same commit could be on different data."""
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    record = json.loads((outcome.run.directory / "run.json").read_text(encoding="utf-8"))
    provenance = record["dataset_provenance"][0]
    assert provenance["version"] == "v9"
    assert provenance["exported"] == "2026-09-10"
    assert provenance["origin"] == "https://example/project"


def test_the_weights_are_copied_inside_the_run(registered, prepared):
    """If they live outside, in two weeks nobody knows which weights gave that metrics.json."""
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    assert outcome.weights.read_bytes() == b"fake weights"
    assert outcome.run.directory in outcome.weights.parents


def test_weights_can_be_reused_without_training(registered, prepared, tmp_path):
    root, splits_dir, config = prepared
    weights = tmp_path / "previous.pt"
    weights.write_bytes(b"previous")
    _FakeDetector.seen_splits = ()

    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET, weights=weights,
    )
    assert _FakeDetector.seen_splits == (), "it should not have trained"
    assert outcome.weights.read_bytes() == b"previous"


def test_the_summary_says_what_decides(registered, prepared):
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    summary = outcome.summary()
    assert "mAP50" in summary and "coverage p5" in summary
    assert "contamination" in summary


# --- consistency of the dataset registry ----------------------------------


def test_a_fit_dataset_cannot_aggregate_copyleft_sources():
    """The license of an aggregate does not override that of its sources."""
    with pytest.raises(DatasetError, match="does not override"):
        DatasetInfo(
            name="inconsistent",
            root=Path("."),
            license="CC-BY-4.0",
            production_ready=True,
            sources=("CC-BY-4.0", "AGPL-3.0"),
        )


def test_sources_cannot_be_empty():
    """Empty by oversight would pass the license check without looking at anything."""
    with pytest.raises(DatasetError, match="sources"):
        DatasetInfo(
            name="no_sources",
            root=Path("."),
            license="CC-BY-4.0",
            production_ready=True,
            sources=(),
        )


def test_the_real_dataset_declares_attribution():
    real = datasets_mod.get("annotated-banknotes-2")
    assert real.license == "CC-BY-4.0"
    assert any("attribution" in n.lower() for n in real.notes)
    assert real.origin.startswith("https://universe.roboflow.com/")


# --- reading caveats ------------------------------------------------------


def test_training_with_padding_and_inferring_without_it_warns(registered, prepared):
    """A silent mismatch would make the result be misread: without the
    warning, a bad number would be attributed to the padding and not to the
    mismatch."""
    root, splits_dir, config = prepared
    config = config.model_copy(
        update={
            "detector": config.detector.model_copy(
                update={
                    "out_of_bounds": OutOfBoundsPolicy.PAD,
                    "pad_at_inference": False,
                }
            )
        }
    )
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    assert any("mismatched" in c for c in outcome.run.record.caveats)
    assert "WARNING" in outcome.summary()
    record = json.loads((outcome.run.directory / "run.json").read_text(encoding="utf-8"))
    assert record["caveats"]


def test_without_padding_there_is_no_warning(registered, prepared):
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    assert outcome.run.record.caveats == []


# --- the sealed set -------------------------------------------------------


def _prepare_test_run(registered, prepared):
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    return root, splits_dir, config, outcome


def test_evaluating_on_test_leaves_a_record(registered, prepared, tmp_path):
    from testbank.data.splits import read_test_accesses
    from testbank.experiment.runner import evaluate_on_test

    root, splits_dir, config, outcome = _prepare_test_run(registered, prepared)
    log = tmp_path / "test_evaluations.jsonl"

    result = evaluate_on_test(
        registered, config,
        run_directory=outcome.run.directory,
        weights=outcome.weights,
        reason="definitive baseline",
        splits_dir=splits_dir,
        data_root=root,
        log_path=log,
    )
    entries = read_test_accesses(log)
    assert len(entries) == 1
    assert entries[0]["reason"] == "definitive baseline"
    assert entries[0]["access_number"] == 1
    assert entries[0]["split"]["version"] == "v1", "the log says WHICH test was opened"
    assert result.metrics["split"] == "test"
    assert result.metrics["split_version"]["version"] == "v1"
    assert (outcome.run.directory / "metrics_test.json").exists()


def test_the_second_access_sees_the_first(registered, prepared, tmp_path):
    """It is what makes the log useful: the next one who looks sees it was already looked at."""
    from testbank.experiment.runner import evaluate_on_test

    root, splits_dir, config, outcome = _prepare_test_run(registered, prepared)
    log = tmp_path / "log.jsonl"
    common = {
        "run_directory": outcome.run.directory,
        "weights": outcome.weights,
        "splits_dir": splits_dir,
        "data_root": root,
        "log_path": log,
    }
    evaluate_on_test(registered, config, reason="first", **common)
    second = evaluate_on_test(registered, config, reason="second", **common)

    assert len(second.previous_accesses) == 1
    assert second.previous_accesses[0]["reason"] == "first"
    assert second.entry["access_number"] == 2


def test_without_a_reason_the_test_is_not_opened(registered, prepared, tmp_path):
    from testbank.data.splits import SplitError
    from testbank.experiment.runner import evaluate_on_test

    root, splits_dir, config, outcome = _prepare_test_run(registered, prepared)
    log = tmp_path / "log.jsonl"
    with pytest.raises(SplitError, match="reason"):
        evaluate_on_test(
            registered, config, run_directory=outcome.run.directory,
            weights=outcome.weights, reason="   ",
            splits_dir=splits_dir, data_root=root, log_path=log,
        )
    assert not log.exists(), "a rejected attempt must not leave an entry"


def test_it_is_recorded_before_evaluating(registered, prepared, tmp_path):
    """If the evaluation failed, the access has to be on record anyway.

    Recording afterwards would let an evaluation abandoned upon seeing a bad
    number leave no trace, and the count would stop meaning anything.
    """
    from testbank.data.splits import read_test_accesses
    from testbank.experiment.runner import evaluate_on_test

    root, splits_dir, config, outcome = _prepare_test_run(registered, prepared)
    log = tmp_path / "log.jsonl"

    class _BlowsUp(type(registered)):
        def predict(self, samples, *, weights, config):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        evaluate_on_test(
            _BlowsUp(), config, run_directory=outcome.run.directory,
            weights=outcome.weights, reason="failed attempt",
            splits_dir=splits_dir, data_root=root, log_path=log,
        )
    assert len(read_test_accesses(log)) == 1


def test_the_normal_runner_never_opens_the_test(registered, prepared, tmp_path):
    from testbank.data.splits import read_test_accesses

    _prepare_test_run(registered, prepared)
    assert read_test_accesses(tmp_path / "log.jsonl") == []
    assert _FakeDetector.seen_splits == ("train", "valid")
