from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from rtmdet_obb import RTMDetR, load_pretrained  # noqa: E402


def test_rtmdet_tiny_forward_has_three_finite_output_levels():
    model = RTMDetR(num_classes=1, size="tiny").eval()
    with torch.no_grad():
        cls, reg, angle = model(torch.zeros(1, 3, 64, 64))
    assert [x.shape[-2:] for x in cls] == [(8, 8), (4, 4), (2, 2)]
    assert [x.shape[1] for x in reg] == [4, 4, 4]
    assert [x.shape[1] for x in angle] == [1, 1, 1]
    assert all(torch.isfinite(x).all() for group in (cls, reg, angle) for x in group)


def test_local_dota_checkpoint_loads_when_present():
    checkpoint = Path(__file__).parents[1] / "models/rtmdet/checkpoints/rotated_rtmdet_tiny-3x-dota-9d821076.pth"
    if not checkpoint.exists():
        pytest.skip("local DOTA checkpoint is not available")
    model = RTMDetR(num_classes=1, size="tiny")
    mismatched, missing = load_pretrained(model, str(checkpoint), verbose=False)
    # The DOTA checkpoint has 15 classes: weight and bias differ at each level.
    assert len(mismatched) == 6
    assert missing
