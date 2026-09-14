"""Inference on loose images with the runs in `runs/` (what the app uses)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import rotated_rect_points

from testbank.config import Config
from testbank.geometry.quad import Quad, canonicalize
from testbank.prediction import Prediction
from testbank.serve import (
    TrainedModel,
    crop,
    crops,
    decode_image,
    discover_models,
    draw,
    expand_quad,
    load_model,
)

SIDE = 160


def _quad(cx, cy, half_long=0.25, ratio=2.0, theta=0.4) -> Quad:
    return canonicalize(Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta)))


def _run(root, name, *, weights=("best.pt",), detector="yolox-obb-nano", metrics=True):
    d = root / name
    (d / "weights").mkdir(parents=True)
    for w in weights:
        (d / "weights" / w).write_bytes(b"x")
    record = {"name": name.split("_", 1)[-1], "detector": {"name": detector},
              "split": {"version": "v1", "mode": "adopt"}}
    (d / "run.json").write_text(json.dumps(record), encoding="utf-8")
    (d / "config.yaml").write_text(Config().dump_yaml(), encoding="utf-8")
    if metrics:
        (d / "metrics.json").write_text(json.dumps({
            "map50": {"value": 0.91}, "map50_95": {"value": 0.7}, "coverage_p5": {"value": 0.95},
        }), encoding="utf-8")
    return d


# --- discovery ------------------------------------------------------------


def test_discovery_lists_runs_with_weights_most_recent_first(tmp_path):
    _run(tmp_path, "20260101T000000Z_old")
    _run(tmp_path, "20260102T000000Z_new")
    _run(tmp_path, "20260103T000000Z_no_weights", weights=())
    (tmp_path / "_inspection").mkdir()
    (tmp_path / "test_evaluations.jsonl").write_text("{}\n")
    models = discover_models(tmp_path)
    assert [m.name for m in models] == ["new", "old"]
    assert models[0].detector == "yolox-obb-nano"
    assert models[0].split == "v1 (adopt)"
    assert models[0].config.detector.image_size == Config().detector.image_size


def test_best_pt_wins_over_last_and_a_foreign_run_gives_its_highest_epoch(tmp_path):
    own = load_model(_run(tmp_path, "a_own", weights=("last.pt", "best.pt")))
    assert own.weights.name == "best.pt"
    foreign = load_model(_run(tmp_path, "b_rtmdet", weights=("epoch_9.pth", "epoch_100.pth"),
                              detector="rtmdet-r-tiny"))
    assert foreign.weights.name == "epoch_100.pth"


def test_the_label_carries_the_headline_metrics_and_survives_without_them(tmp_path):
    scored = load_model(_run(tmp_path, "a_scored"))
    assert scored.summary() == {"mAP50": 0.91, "mAP50-95": 0.7, "cov p5": 0.95}
    assert "mAP50 0.910" in scored.label and "yolox-obb-nano" in scored.label
    bare = load_model(_run(tmp_path, "b_bare", metrics=False))
    assert bare.summary() == {} and bare.label == "bare  |  yolox-obb-nano"


def test_a_missing_runs_dir_is_just_empty(tmp_path):
    assert discover_models(tmp_path / "nowhere") == []


def test_an_adapter_without_load_support_is_served_by_loading_each_time(tmp_path):
    from testbank.detectors.base import BaseDetector, register
    from testbank.serve import inference_config, load_weights

    assert BaseDetector().load(Path("w.pt"), Config()) is None
    @register
    class NoLoad(BaseDetector):
        name = "test-no-load"

    model = load_model(_run(tmp_path, "a_noload", detector="test-no-load"))
    assert load_weights(model, confidence=0.5) is None
    cfg = inference_config(model, confidence=0.5, nms_iou=None)
    assert cfg.detector.confidence_threshold == 0.5
    assert cfg.detector.nms_iou == model.config.detector.nms_iou


# --- drawing and crops ----------------------------------------------------


def _painted(quad: Quad, side: int = SIDE) -> np.ndarray:
    image = np.zeros((side, side, 3), dtype=np.uint8)
    pts = np.array([[x * side, y * side] for x, y in quad.points], dtype=np.int32)
    cv2.fillPoly(image, [pts], (255, 255, 255))
    return image


def test_the_crop_rectifies_the_banknote_landscape_and_full_of_it():
    quad = _quad(0.5, 0.5, half_long=0.3, ratio=2.0, theta=0.6)
    cut = crop(_painted(quad), quad)
    h, w = cut.shape[:2]
    assert w > h and w / h == pytest.approx(2.0, rel=0.05)
    assert w == pytest.approx(0.6 * SIDE, abs=2)
    assert (cut[..., 0] > 127).mean() > 0.97, "the rectified crop is the banknote"


def test_the_margin_adds_black_around_the_banknote():
    quad = _quad(0.5, 0.5, half_long=0.25, ratio=2.0, theta=0.6)
    tight = crop(_painted(quad), quad, margin=0.0)
    loose = crop(_painted(quad), quad, margin=0.1)
    assert loose.shape[1] == pytest.approx(tight.shape[1] * 1.2, abs=2)
    inner = 1.0 / 1.2**2
    assert (loose[..., 0] > 127).mean() == pytest.approx(inner, abs=0.05)
    with pytest.raises(ValueError, match="negative"):
        expand_quad(np.zeros((4, 2), np.float32), -0.1)


def test_crops_skip_predictions_without_geometry_and_draw_keeps_the_size():
    quad = _quad(0.5, 0.5)
    image = _painted(quad)
    predictions = [Prediction(quad=quad, score=0.9), Prediction(quad=None, score=0.4)]
    assert len(crops(image, predictions)) == 1
    canvas = draw(image, predictions)
    assert canvas.shape == image.shape and not np.array_equal(canvas, image)


def test_decode_image_round_trips_and_rejects_garbage():
    image = np.random.default_rng(0).integers(0, 255, (32, 48, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".png", image)
    assert np.array_equal(decode_image(buffer.tobytes()), image)
    with pytest.raises(ValueError, match="decode"):
        decode_image(b"not an image")


# --- end to end with a trained nano ---------------------------------------


def test_predict_image_runs_the_adapter_and_leaves_nothing_behind(tmp_path):
    torch = pytest.importorskip("torch")
    from testbank.models.yolox_obb import YoloxObb
    from testbank.serve import load_weights, predict_image

    # A checkpoint as `main` writes them: the network and its head travel
    # with the weights. Untrained, so it detects nothing at 0.25; the point
    # is the plumbing, checked at a threshold where it emits something.
    quad = _quad(0.5, 0.5, half_long=0.25)
    network = YoloxObb("nano", num_classes=1)
    weights = tmp_path / "t" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    torch.save({
        "model": network.state_dict(), "variant": "nano", "head": network.head_spec.to_dict(),
        "arch": "yolox_obb", "num_classes": 1, "image_size": 64, "epoch": 0,
    }, weights)
    base = Config()
    config = base.model_copy(update={
        "data": base.data.model_copy(update={"derived_dir": tmp_path / "derived"}),
        "detector": base.detector.model_copy(update={"image_size": 64}),
    })
    model = TrainedModel(
        directory=tmp_path / "t", name="t", detector="yolox-obb-nano", weights=weights, config=config,
    )
    predictions = predict_image(model, _painted(quad, 96), confidence=1e-6, nms_iou=0.5)
    assert predictions, "an untrained head still emits something at a near-zero threshold"
    assert all(isinstance(p, Prediction) for p in predictions)
    assert [p.score for p in predictions] == sorted((p.score for p in predictions), reverse=True)
    assert not (tmp_path / "derived").exists(), "the loose image must not touch a size cache"
    # Loading once and predicting with the loaded model (what the app caches)
    # gives exactly the same answer as loading inside `predict`.
    loaded = load_weights(model, confidence=1e-6, nms_iou=0.5)
    assert loaded is not None
    again = predict_image(model, _painted(quad, 96), confidence=1e-6, nms_iou=0.5, loaded=loaded)
    assert [(p.score, p.quad) for p in again] == [(p.score, p.quad) for p in predictions]


# --- the Streamlit page, headless -----------------------------------------


def test_the_app_lists_the_runs_and_the_sliders_take_the_runs_defaults(tmp_path, monkeypatch):
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from testbank import app

    _run(tmp_path, "20260102T000000Z_new")
    _run(tmp_path, "20260101T000000Z_old", metrics=False)
    monkeypatch.setattr("sys.argv", ["app.py", "--runs-dir", str(tmp_path)])
    page = AppTest.from_file(app.__file__, default_timeout=30).run()
    assert not page.exception, page.exception
    assert page.title[0].value.startswith("testbank")
    box = page.sidebar.selectbox[0]
    assert len(box.options) == 2
    sliders = {s.label: s.value for s in page.sidebar.slider}
    cfg = Config()
    assert sliders == {
        "Confidence": cfg.metrics.report_confidence,
        "NMS IoU": cfg.detector.nms_iou,
        "Crop margin": cfg.crop.margin,
    }
    assert streamlit.__version__


def test_the_app_says_so_when_there_is_nothing_to_serve(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from testbank import app

    monkeypatch.setattr("sys.argv", ["app.py", "--runs-dir", str(tmp_path / "empty")])
    page = AppTest.from_file(app.__file__, default_timeout=30).run()
    assert not page.exception
    assert "No run with weights" in page.warning[0].value


def test_the_cli_has_an_app_command():
    from testbank.cli import build_parser

    args = build_parser().parse_args(["--runs-dir", "x", "app", "--port", "8600"])
    assert (args.command, args.runs_dir, args.port) == ("app", "x", 8600)
    args = build_parser().parse_args(["predict", "a.jpg", "b.jpg", "--run", "r", "--margin", "0.1"])
    assert (args.command, [p.name for p in args.images], args.run, args.margin) == (
        "predict", ["a.jpg", "b.jpg"], "r", 0.1,
    )
