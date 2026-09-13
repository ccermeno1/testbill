"""Attenuator of the angle loss on nearly square boxes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the own candidate needs torch")

from testbank.config import AngleWeightConfig
from testbank.models.losses import (
    DECAYS,
    angle_weight,
    side_ratio_px,
)


def w(ratio, **kwargs):
    return angle_weight(torch.tensor(ratio, dtype=torch.float32), **kwargs)


# --- the behaviour sought -------------------------------------------------


def test_a_well_defined_box_weighs_one():
    """A 2:1 banknote has a perfectly defined angle: no discount."""
    assert float(w([2.0])) == pytest.approx(1.0)


def test_a_perfect_square_weighs_nothing():
    """Its angle is undefined: punishing the model for missing it is noise."""
    assert float(w([1.0])) == pytest.approx(0.0)


def test_above_the_threshold_always_weighs_one():
    assert torch.allclose(w([1.1, 1.5, 3.0, 10.0]), torch.ones(4), atol=1e-6)


def test_the_weight_grows_with_the_ratio():
    """The better defined the angle, the more getting it right counts."""
    values = w([1.0, 1.02, 1.05, 1.08, 1.1])
    assert torch.all(values[1:] >= values[:-1])


def test_the_weight_stays_in_its_range():
    values = w(torch.linspace(0.5, 5.0, 50).tolist())
    assert torch.all(values >= 0.0)
    assert torch.all(values <= 1.0)


# --- the control branch ---------------------------------------------------


def test_disabled_returns_ones():
    """It is the control arm of the experiment: without attenuation, everything weighs the same."""
    values = w([1.0, 1.05, 2.0], enabled=False)
    assert torch.allclose(values, torch.ones(3))


def test_the_floor_is_respected():
    """`min_weight` allows attenuating without cancelling, in case zero turns out to be too much."""
    assert float(w([1.0], min_weight=0.3)) == pytest.approx(0.3)
    assert float(w([2.0], min_weight=0.3)) == pytest.approx(1.0)


# --- the decay shapes -----------------------------------------------------


@pytest.mark.parametrize("decay", sorted(DECAYS))
def test_every_shape_goes_from_zero_to_one(decay):
    assert float(w([1.0], decay=decay)) == pytest.approx(0.0, abs=1e-6)
    assert float(w([1.1], decay=decay)) == pytest.approx(1.0, abs=1e-6)


def test_smoothstep_starts_slower_than_linear():
    """It is its reason to exist: leave the most ambiguous band almost without weight."""
    halfway = 1.05  # exactly halfway with threshold 1.1
    assert float(w([halfway], decay="smoothstep")) == pytest.approx(0.5, abs=0.01)
    quarter = 1.025
    assert float(w([quarter], decay="smoothstep")) < float(w([quarter], decay="linear"))


def test_step_cuts_hard():
    """Reference to measure whether the smooth transition contributes anything."""
    assert float(w([1.09], decay="step")) == 0.0
    assert float(w([1.10], decay="step")) == 1.0


def test_unknown_shape_says_which_exist():
    with pytest.raises(ValueError, match="unknown decay shape"):
        w([2.0], decay="exponential")


def test_invalid_threshold_is_an_error():
    with pytest.raises(ValueError, match="ratio_threshold"):
        w([2.0], ratio_threshold=1.0)


# --- the ratio ------------------------------------------------------------


def test_the_ratio_does_not_depend_on_side_order():
    """The order of w and h is precisely what the ambiguity makes arbitrary."""
    a = side_ratio_px(torch.tensor([100.0]), torch.tensor([50.0]))
    b = side_ratio_px(torch.tensor([50.0]), torch.tensor([100.0]))
    assert torch.allclose(a, b)
    assert float(a) == pytest.approx(2.0)


def test_a_zero_side_does_not_blow_up():
    ratio = side_ratio_px(torch.tensor([100.0]), torch.tensor([0.0]))
    assert torch.isfinite(ratio).all()


def test_an_inverted_ratio_gives_the_minimum_weight_not_a_negative_one():
    """If someone swaps the sides, the weight comes out minimal -- not absurd."""
    assert float(w([0.5])) == pytest.approx(0.0)


# --- the config -----------------------------------------------------------


def test_the_config_rejects_a_shape_that_does_not_exist():
    with pytest.raises(ValueError, match="unknown decay shape"):
        AngleWeightConfig(decay="none")


def test_the_default_values_are_the_documented_ones():
    cfg = AngleWeightConfig()
    assert cfg.enabled is True
    assert cfg.ratio_threshold == 1.1  # the same threshold as the anchor warning
    assert cfg.min_weight == 0.0
    assert cfg.decay == "smoothstep"


def test_it_can_be_switched_off_from_the_config():
    cfg = AngleWeightConfig(enabled=False)
    values = angle_weight(
        torch.tensor([1.0, 2.0]),
        enabled=cfg.enabled,
        ratio_threshold=cfg.ratio_threshold,
        min_weight=cfg.min_weight,
        decay=cfg.decay,
    )
    assert torch.allclose(values, torch.ones(2))


def test_the_attenuator_is_recorded_in_the_run():
    """Without this it would not be a comparable experiment: two runs with
    different attenuators would look the same in the table."""
    from testbank.config import Config

    yaml_text = Config().dump_yaml()
    assert "angle_weight" in yaml_text
    assert "ratio_threshold" in yaml_text
    assert "smoothstep" in yaml_text
