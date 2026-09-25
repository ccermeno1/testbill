import pytest

torch = pytest.importorskip("torch")

from yolox_obb.assigner import SimOTAAssigner  # noqa: E402


def _priors(size=8, stride=8):
    ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32),
                            indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1), torch.full((size * size,), float(stride))], 1)


def test_simota_assigns_anchors_inside_the_gt_only():
    priors = _priors()
    n = priors.shape[0]
    gt = torch.tensor([[20., 20., 24., 16., 0.4]])
    decoded = gt.expand(n, -1).clone()  # every prediction is perfect: IoU 1
    res = SimOTAAssigner().assign(torch.zeros(n, 1), torch.zeros(n), decoded, priors, gt, torch.tensor([0]))
    fg = res["fg_mask"]
    assert 1 <= int(fg.sum()) <= 10
    centers = (priors[fg, :2] + 0.5) * 8
    assert ((centers - gt[:, :2]).abs() < 2.5 * 8).all()
    assert torch.allclose(res["matched_ious"], torch.ones_like(res["matched_ious"]), atol=1e-4)
    assert (res["matched_gt_inds"] == 0).all()


def test_simota_without_gt_is_all_background():
    priors = _priors()
    n = priors.shape[0]
    res = SimOTAAssigner().assign(torch.zeros(n, 1), torch.zeros(n), torch.rand(n, 5) * 10, priors,
                                  torch.zeros((0, 5)), torch.zeros((0,), dtype=torch.long))
    assert not res["fg_mask"].any()
