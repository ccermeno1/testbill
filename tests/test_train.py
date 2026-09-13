"""Training loop and adapter of the own candidate."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="the own candidate needs torch")

from conftest import rotated_rect_points

from testbank.config import Config
from testbank.data.discover import Sample
from testbank.dataio.formats import get as get_format
from testbank.detectors import get as get_detector
from testbank.geometry.quad import Quad, canonicalize
from testbank.models.data import build_datasets, collate, quad_to_box
from testbank.models.train import fit, learning_rate_at, load_model

SIDE = 128


def _quad(cx, cy, half_long=0.2, ratio=2.0, theta=0.0) -> Quad:
    return canonicalize(Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta)))


def _write(directory: Path, sample_id: str, quads) -> Sample:
    images = directory / "images"
    labels = directory / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    path = images / f"{sample_id}.jpg"
    Image.new("RGB", (SIDE, SIDE), (60, 60, 60)).save(path)
    writer = get_format("obb_yolo")
    label = labels / f"{sample_id}.txt"
    label.write_text(
        "\n".join(f"0 {writer.from_quad(q).payload}" for q in quads) + "\n",
        encoding="utf-8",
    )
    return Sample(sample_id=sample_id, image_path=path, label_path=label)


@pytest.fixture()
def setup(tmp_path):
    quads = (_quad(0.35, 0.5, theta=0.3), _quad(0.7, 0.5))
    samples = {"train": [_write(tmp_path / "src", f"s{i}", quads) for i in range(4)]}
    base = Config()
    config = base.model_copy(
        update={
            "data": base.data.model_copy(update={"derived_dir": tmp_path / "d"}),
            "detector": base.detector.model_copy(
                update={"epochs": 2, "image_size": SIDE, "batch_size": 2}
            ),
        }
    )
    return samples, config


# --- the dataset ----------------------------------------------------------


def test_the_canonical_quad_gives_the_long_side_as_w():
    """It is not a coincidence being exploited: it is the reason the canonical
    order exists. `p0->p1` anchors the longest side, so `w` comes out as that."""
    quad = _quad(0.5, 0.5, half_long=0.25, ratio=3.0, theta=0.4)
    _, _, w, h, theta = quad_to_box(quad, 400, 400)
    assert w > h
    assert w / h == pytest.approx(3.0, rel=0.02)
    assert 0.0 <= theta < math.pi


def test_the_box_center_is_the_quad_center():
    quad = _quad(0.3, 0.7, half_long=0.15)
    cx, cy = quad_to_box(quad, 200, 200)[:2]
    assert cx == pytest.approx(0.3 * 200, abs=1.0)
    assert cy == pytest.approx(0.7 * 200, abs=1.0)


def test_the_dataset_delivers_image_and_boxes(setup):
    samples, config = setup
    dataset = build_datasets(samples, config)["train"]
    image, boxes, classes, sample_id = dataset[0]
    assert image.shape == (3, SIDE, SIDE)
    # Raw BGR in 0-255, the YOLOX convention: it is what the pretrained
    # weights expect (Megvii's COCO, DDGRCF's DOTA). Before it went in [0, 1]
    # and the pretraining arrived wrecked; see `image_to_input`.
    assert image.min() >= 0.0 and image.max() <= 255.0 and image.max() > 1.0
    assert boxes.shape == (2, 5)
    assert classes.shape == (2,)
    assert sample_id == "s0"


def test_the_dataset_goes_through_the_area_filter(tmp_path):
    """If it loaded the files on its own, this candidate would train on a
    different truth than Ultralytics."""
    large = _quad(0.5, 0.5, half_long=0.30)
    strip = _quad(0.5, 0.5, half_long=0.30, ratio=30.0)
    sample = _write(tmp_path / "src", "a", [large, strip])
    base = Config()
    config = base.model_copy(
        update={
            "data": base.data.model_copy(update={"derived_dir": tmp_path / "d"}),
            "detector": base.detector.model_copy(update={"image_size": SIDE}),
        }
    )
    dataset = build_datasets({"train": [sample]}, config)["train"]
    _, boxes, _, _ = dataset[0]
    assert boxes.shape[0] == 1, "the filtered strip must not reach the trainer"


def test_the_batch_does_not_stack_the_boxes(setup):
    """Every image has a different number; stacking would require padding
    with garbage that one must then remember to ignore."""
    samples, config = setup
    dataset = build_datasets(samples, config)["train"]
    batch = collate([dataset[0], dataset[1]])
    assert batch.images.shape == (2, 3, SIDE, SIDE)
    assert isinstance(batch.boxes, list) and len(batch.boxes) == 2


# --- the scheduler --------------------------------------------------------


def test_the_warmup_starts_low():
    """Without it, the first iterations with the freshly initialized head give
    huge gradients that destabilize the BatchNorm."""
    assert learning_rate_at(0, 1000, 1.0) < 0.1


def test_the_cosine_ends_almost_at_zero():
    assert learning_rate_at(999, 1000, 1.0) < 0.01


def test_the_maximum_is_in_the_middle_of_the_start():
    values = [learning_rate_at(s, 1000, 1.0) for s in range(1000)]
    assert max(values) == pytest.approx(1.0, abs=0.01)
    assert values.index(max(values)) < 100


# --- training for real ----------------------------------------------------


def test_training_leaves_weights_and_history(setup, tmp_path):
    samples, config = setup
    dataset = build_datasets(samples, config)["train"]
    weights, history = fit(dataset, config, output_dir=tmp_path / "out")

    assert weights.exists()
    assert len(history.epochs) == 2
    for entry in history.epochs:
        assert {"box", "angle", "objectness", "classes", "total"} <= set(entry)


def test_the_loss_goes_down(setup, tmp_path):
    """The minimal proof that the loop learns something instead of going in circles."""
    samples, config = setup
    config = config.model_copy(
        update={"detector": config.detector.model_copy(update={"epochs": 6})}
    )
    dataset = build_datasets(samples, config)["train"]
    _, history = fit(dataset, config, output_dir=tmp_path / "out")
    first = history.epochs[0]["total"]
    last = history.epochs[-1]["total"]
    assert last < first, f"it did not go down: {first:.4f} -> {last:.4f}"


def test_the_weights_remember_their_variant(setup, tmp_path):
    """Loading nano weights into a tiny would fail with an incomprehensible
    shape error; the file is the only place that does not drift out of sync."""
    samples, config = setup
    dataset = build_datasets(samples, config)["train"]
    weights, _ = fit(dataset, config, output_dir=tmp_path / "out")
    model = load_model(weights)
    assert model.variant == config.detector.variant


def test_training_is_deterministic(setup, tmp_path):
    """Same seed, same loss. Without this there is no possible comparison."""
    samples, config = setup
    dataset = build_datasets(samples, config)["train"]
    _, a = fit(dataset, config, output_dir=tmp_path / "a")
    _, b = fit(dataset, config, output_dir=tmp_path / "b")
    assert a.epochs[-1]["total"] == pytest.approx(b.epochs[-1]["total"], rel=1e-6)


# --- the adapter ----------------------------------------------------------


def test_it_is_registered_and_fit_for_production():
    detector = get_detector("yolox-obb-nano")
    assert detector.license == "Apache-2.0"
    assert detector.production_ready is True
    assert not detector.component().blockers("the detector")


def test_it_appears_among_the_production_candidates():
    from testbank.detectors import production_candidates

    assert "yolox-obb-nano" in production_candidates()


def test_the_full_path_of_the_adapter(setup, tmp_path):
    samples, config = setup
    detector = get_detector("yolox-obb-nano")
    result = detector.train(samples, config, output_dir=tmp_path / "train")
    assert result.weights.exists()

    predictions = detector.predict(
        samples["train"], weights=result.weights, config=config
    )
    assert set(predictions) == {s.sample_id for s in samples["train"]}
    for found in predictions.values():
        for prediction in found:
            assert 0.0 <= prediction.score <= 1.0
            assert len(prediction.quad.points) == 4


# --- one candidate per variant --------------------------------------------


def test_there_is_one_candidate_per_variant():
    from testbank.detectors import detectors
    from testbank.models.yolox_obb import VARIANTS

    registered = set(detectors())
    for variant in VARIANTS:
        assert f"yolox-obb-{variant}" in registered


def test_the_registered_name_matches_the_variant():
    """Regression: the name was fixed at "nano" while the variant came from
    the config, so a 4.37M tiny was registered as the 857k nano."""
    from testbank.detectors import get as get_detector
    from testbank.models.yolox_obb import VARIANTS

    for variant in VARIANTS:
        detector = get_detector(f"yolox-obb-{variant}")
        assert detector.variant == variant


def test_the_candidate_variant_overrides_the_config(setup, tmp_path):
    """And it is WRITTEN into the config, so the frozen config.yaml does not lie."""
    samples, config = setup
    from testbank.detectors import get as get_detector
    from testbank.models.train import load_model

    detector = get_detector("yolox-obb-tiny")
    assert config.detector.variant == "nano", "the config says otherwise on purpose"

    result = detector.train(samples, config, output_dir=tmp_path / "t")
    assert load_model(result.weights).variant == "tiny"


def test_each_variant_declares_its_parameters():
    from testbank.detectors import get as get_detector

    note = " ".join(get_detector("yolox-obb-tiny").notes)
    assert "tiny" in note
    assert "4,366,808" in note


# --- checkpoint selection and the live history ----------------------------


def _config_with(config, **detector):
    return config.model_copy(
        update={"detector": config.detector.model_copy(update=detector)}
    )


def test_without_validation_best_is_the_last_epoch_and_it_is_said(setup, tmp_path):
    samples, config = setup
    dataset = build_datasets(samples, config)["train"]
    weights, history = fit(dataset, config, output_dir=tmp_path / "out", log=None)
    assert weights.name == "best.pt" and (tmp_path / "out" / "last.pt").exists()
    assert history.best is None
    assert any("best.pt = last epoch" in n for n in history.notes)


def test_validation_keeps_the_best_checkpoint_not_the_last(setup, tmp_path):
    """A fake validator whose score PEAKS in the middle: best.pt must be that
    epoch, last.pt the final one, and the history must say which."""
    import torch as torch_

    from testbank.models.train import load_model

    samples, config = setup
    config = _config_with(config, epochs=4, eval_every=1)
    dataset = build_datasets(samples, config)["train"]
    scores = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.4}
    snapshots = {}

    def validate(model, epoch):
        snapshots[epoch] = {k: v.clone() for k, v in model.state_dict().items()}
        return {"map50": scores[epoch], "coverage_p5": 1.0}

    lines = []
    weights, history = fit(
        dataset, config, output_dir=tmp_path / "out", validate=validate, log=lines.append
    )
    assert history.best["epoch"] == 1 and history.best["map50"] == 0.9
    assert [v["epoch"] for v in history.validation] == [0, 1, 2, 3]
    best = load_model(weights).state_dict()
    assert all(torch_.equal(best[k], snapshots[1][k]) for k in best)
    last = torch_.load(tmp_path / "out" / "last.pt", weights_only=False)
    assert last["epoch"] == 3
    assert sum("<- best" in line for line in lines) == 2  # epochs 1 and 2 (1-based)
    assert any("best.pt = epoch 2 of 4" in n for n in history.notes)


def test_eval_every_spaces_the_evaluations_and_always_includes_the_last(setup, tmp_path):
    samples, config = setup
    config = _config_with(config, epochs=5, eval_every=2)
    dataset = build_datasets(samples, config)["train"]
    seen = []

    def validate(model, epoch):
        seen.append(epoch)
        return {"map50": 0.5, "coverage_p5": 1.0}

    fit(dataset, config, output_dir=tmp_path / "out", validate=validate, log=None)
    assert seen == [1, 3, 4]


def test_the_history_is_written_after_every_epoch(setup, tmp_path):
    """A run that dies halfway has to leave its curve on disk."""
    import json

    from testbank.models.train import HISTORY_FILE

    samples, config = setup
    config = _config_with(config, epochs=3, eval_every=1)
    dataset = build_datasets(samples, config)["train"]
    lengths = []

    def validate(model, epoch):
        lengths.append(len(json.loads((tmp_path / "out" / HISTORY_FILE).read_text())["epochs"])
                       if (tmp_path / "out" / HISTORY_FILE).exists() else 0)
        return {"map50": 0.5, "coverage_p5": 1.0}

    fit(dataset, config, output_dir=tmp_path / "out", validate=validate, log=None)
    # At the validation of epoch k the file already had k epochs written.
    assert lengths == [0, 1, 2]
    written = json.loads((tmp_path / "out" / HISTORY_FILE).read_text(encoding="utf-8"))
    assert len(written["epochs"]) == 3 and len(written["validation"]) == 3
    assert written["best"]["epoch"] == 0


def test_the_adapter_validates_with_our_metrics_and_records_it(setup, tmp_path):
    samples, config = setup
    samples = {**samples, "valid": samples["train"][:2]}
    config = _config_with(config, epochs=2, eval_every=1)
    detector = get_detector("yolox-obb-nano")
    result = detector.train(samples, config, output_dir=tmp_path / "t")
    assert result.weights.name == "best.pt"
    assert not (tmp_path / "t" / "_eval.pt").exists()
    assert any("2 validation images" in n for n in result.notes)
    assert any(n.startswith("best.pt = epoch") for n in result.notes)


def test_validating_mid_training_does_not_change_the_trajectory(setup, tmp_path):
    """Evaluating must not consume the training RNG: with and without
    validation the losses have to match exactly."""
    samples, config = setup
    config = _config_with(config, epochs=3, eval_every=1)
    dataset = build_datasets(samples, config)["train"]
    _, plain = fit(dataset, config, output_dir=tmp_path / "a", log=None)
    _, validated = fit(
        dataset, config, output_dir=tmp_path / "b",
        validate=lambda m, e: {"map50": 0.0, "coverage_p5": 0.0}, log=None,
    )
    assert [e["total"] for e in plain.epochs] == pytest.approx(
        [e["total"] for e in validated.epochs], rel=1e-6
    )


# --- the curves -----------------------------------------------------------


def test_plot_training_renders_losses_and_validation(setup, tmp_path):
    pytest.importorskip("matplotlib", reason="plots live in the dev group")
    from testbank.viz.curves import plot_run

    samples, config = setup
    config = _config_with(config, epochs=3, eval_every=1)
    dataset = build_datasets(samples, config)["train"]
    fit(
        dataset, config, output_dir=tmp_path / "run" / "_train",
        validate=lambda m, e: {"map50": 0.2 * (e + 1), "coverage_p5": 0.9}, log=None,
    )
    path = plot_run(tmp_path / "run")
    assert path == tmp_path / "run" / "viz" / "training.png"
    assert path.stat().st_size > 1000


def test_plot_training_says_when_there_is_nothing_to_plot(tmp_path):
    from testbank.viz.curves import find_history

    with pytest.raises(FileNotFoundError, match="training.json"):
        find_history(tmp_path)
