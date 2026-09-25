import pytest

torch = pytest.importorskip("torch")

from rtmdet_obb.evaluation import summarize  # noqa: E402


def test_synthetic_summary_perfect_detection():
    box = torch.tensor([[10., 10., 4., 6., 0., 0.95]])
    detections = [[box]]
    annotations = [dict(boxes=box[:, :5], labels=torch.tensor([0]))]
    result = summarize(detections, annotations, 1)
    assert result["mAP@0.50"] == pytest.approx(1.0)
    assert result["mAP@.5:.95"] == pytest.approx(1.0)
    assert result["recall@0.50"] == pytest.approx(1.0)


def test_fitness_formula_matches_checkpoint_selection_weights():
    metrics = {"mAP@0.50": 0.8, "mAP@.5:.95": 0.6}
    fitness = 0.9 * metrics["mAP@.5:.95"] + 0.1 * metrics["mAP@0.50"]
    assert fitness == pytest.approx(0.62)


def test_synthetic_summary_empty_class_is_finite():
    detections = [[torch.empty((0, 6))]]
    annotations = [dict(boxes=torch.empty((0, 5)), labels=torch.empty((0,), dtype=torch.long))]
    result = summarize(detections, annotations, 1)
    assert result["mAP@0.50"] == 0.0
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
