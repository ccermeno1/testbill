"""Detector protocol, registry, Ultralytics isolation and derived view."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import rotated_rect_points
from PIL import Image

from testbank.config import Config
from testbank.data.discover import Sample
from testbank.dataio.formats import get as get_format
from testbank.detectors import base as detector_base
from testbank.detectors import dataset as dataset_view
from testbank.detectors.base import (
    BaseDetector,
    Detector,
    DetectorError,
    build_image_evals,
    detectors,
    get,
    production_candidates,
    register,
)
from testbank.geometry.quad import Quad, canonicalize
from testbank.metrics.core import Prediction

SRC = Path(__file__).resolve().parents[1] / "src" / "testbank"
ADAPTER = SRC / "detectors" / "ultralytics_obb.py"

ULTRALYTICS_NAME = "ultralytics-yolo-obb"


# --- the registry ---------------------------------------------------------


def test_ultralytics_is_registered():
    assert ULTRALYTICS_NAME in detectors()


def test_ultralytics_is_agpl_and_not_fit_for_production():
    detector = get(ULTRALYTICS_NAME)
    assert detector.license == "AGPL-3.0"
    assert detector.production_ready is False


def test_it_does_not_appear_among_the_production_candidates():
    """It serves as a performance reference, never as a candidate."""
    assert ULTRALYTICS_NAME not in production_candidates()


def test_it_satisfies_the_protocol():
    assert isinstance(get(ULTRALYTICS_NAME), Detector)


def test_the_component_carries_the_license_into_the_run():
    component = get(ULTRALYTICS_NAME).component()
    assert component.license == "AGPL-3.0"
    assert component.blockers("the detector")


def test_unknown_detector_says_which_exist():
    with pytest.raises(DetectorError, match="unknown"):
        get("does_not_exist")


def test_registering_the_same_name_twice_is_an_error():
    with pytest.raises(DetectorError, match="duplicate"):

        @register
        class Other(BaseDetector):
            name = ULTRALYTICS_NAME


def test_a_detector_without_a_name_is_not_registered():
    with pytest.raises(DetectorError, match="name"):

        @register
        class Nameless(BaseDetector):
            pass


# --- isolation ------------------------------------------------------------


def _imports_ultralytics(path: Path) -> bool:
    """Looks for real imports, not mentions.

    A grep would flag the comments explaining why `val` is called that or the
    name of the derived directory, and the test would stop meaning anything.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "ultralytics" for a in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "ultralytics"
        ):
            return True
    return False


def test_ultralytics_isolation():
    """No module outside the adapter imports `ultralytics`."""
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if path != ADAPTER and _imports_ultralytics(path)
    ]
    assert offenders == [], (
        "these modules import ultralytics outside the adapter: " + str(offenders)
    )


FRAMEWORK_ADAPTERS = {
    "ultralytics": SRC / "detectors" / "ultralytics_obb.py",
    "mmrotate": SRC / "detectors" / "rtmdet_r.py",
    "mmdet": SRC / "detectors" / "rtmdet_r.py",
    "mmengine": SRC / "detectors" / "rtmdet_r.py",
    "paddle": SRC / "detectors" / "ppyoloe_r.py",
    "oriented_det": SRC / "detectors" / "oriented_det.py",
    "ppdet": SRC / "detectors" / "ppyoloe_r.py",
}


def _imports_package(path: Path, package: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == package for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == package:
            return True
    return False


@pytest.mark.parametrize("package", sorted(FRAMEWORK_ADAPTERS))
def test_every_foreign_framework_stays_behind_its_adapter(package):
    """Same rule as for ultralytics: torch is the base; every other framework
    is imported by exactly one adapter and nothing else."""
    adapter = FRAMEWORK_ADAPTERS[package]
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if path != adapter and _imports_package(path, package)
    ]
    assert offenders == [], f"{package} imported outside {adapter.name}: {offenders}"
    assert _imports_package(adapter, package), "guard of the guard"


def test_importing_the_registry_does_not_need_torch():
    """The Paddle environment has no torch. Registering the own variants used
    to build three networks at import time; now it reads a table."""
    code = (
        "import sys; sys.modules['torch'] = None; "
        "import testbank.detectors as d; print(len(d.detectors()))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr[-800:]
    assert int(out.stdout.strip()) >= 7


def test_the_adapter_does_import_it():
    """Guard of the guard: if the adapter stopped importing it, the isolation
    test would pass vacuously and would not be checking anything."""
    assert _imports_ultralytics(ADAPTER)


def test_importing_testbank_does_not_drag_in_ultralytics():
    """The import is lazy: the package is optional and AGPL."""
    code = (
        "import sys; import testbank.detectors; "
        "print('ultralytics' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is not None,
    reason="ultralytics is installed in this environment",
)
def test_without_ultralytics_the_error_explains_why_it_is_optional():
    """`find_spec`, not `sys.modules`: what matters is whether it is INSTALLED.

    Looking at `sys.modules` checked whether it was imported, which with the
    lazy import is always false, so the test also ran with the package
    installed and failed.
    """
    from testbank.detectors.ultralytics_obb import _import_ultralytics

    with pytest.raises(DetectorError, match="AGPL-3.0"):
        _import_ultralytics()


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None,
    reason="ultralytics is not installed",
)
def test_with_ultralytics_installed_the_lazy_import_works():
    """The opposite face of the test above: one of the two always runs."""
    from testbank.detectors.ultralytics_obb import _import_ultralytics

    assert _import_ultralytics() is not None


# --- the derived view -----------------------------------------------------


def _write_sample(directory: Path, sample_id: str, quads, size=(160, 120)) -> Sample:
    images = directory / "images"
    labels = directory / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    image_path = images / f"{sample_id}.jpg"
    Image.new("RGB", size, (30, 30, 30)).save(image_path)
    writer = get_format("obb_yolo")
    label_path = labels / f"{sample_id}.txt"
    label_path.write_text(
        "\n".join(f"0 {writer.from_quad(q).payload}" for q in quads) + "\n",
        encoding="utf-8",
    )
    return Sample(sample_id=sample_id, image_path=image_path, label_path=label_path)


def _quad(cx, cy, half_long=0.2, ratio=2.0, theta=0.0) -> Quad:
    return canonicalize(Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta)))


@pytest.fixture()
def config(tmp_path) -> Config:
    base = Config()
    return base.model_copy(
        update={"data": base.data.model_copy(update={"derived_dir": tmp_path / "derived"})}
    )


def test_the_derived_view_writes_the_labels_already_filtered(tmp_path, config):
    """Training with the unfiltered truth and measuring against the filtered one is no good."""
    large = _quad(0.5, 0.5, half_long=0.30)
    strip = _quad(0.5, 0.5, half_long=0.30, ratio=30.0)
    sample = _write_sample(tmp_path / "src", "a", [large, strip])

    view = dataset_view.materialize({"train": [sample]}, config, out_dir=tmp_path / "out")

    written = (view.root / "train" / "labels" / "a.txt").read_text().strip().splitlines()
    assert len(written) == 1, "the filtered strip must not reach the trainer"
    assert view.dropped == 1
    assert view.counts == {"train": 1}


def test_the_derived_view_does_not_touch_the_source(tmp_path, config):
    large = _quad(0.5, 0.5, half_long=0.30)
    strip = _quad(0.5, 0.5, half_long=0.30, ratio=30.0)
    sample = _write_sample(tmp_path / "src", "a", [large, strip])
    before = hashlib.sha256(sample.label_path.read_bytes()).hexdigest()

    dataset_view.materialize({"train": [sample]}, config, out_dir=tmp_path / "out")

    assert hashlib.sha256(sample.label_path.read_bytes()).hexdigest() == before


def test_the_data_yaml_uses_val_pointing_to_valid(tmp_path, config):
    import yaml

    sample = _write_sample(tmp_path / "src", "a", [_quad(0.5, 0.5)])
    view = dataset_view.materialize(
        {"train": [sample], "valid": [sample]}, config, out_dir=tmp_path / "out"
    )
    data = yaml.safe_load(view.data_yaml.read_text(encoding="utf-8"))
    assert data["val"] == "valid/images"
    assert data["names"] == {0: "euro_banknote"}


def test_the_image_reaches_the_view(tmp_path, config):
    sample = _write_sample(tmp_path / "src", "a", [_quad(0.5, 0.5)])
    view = dataset_view.materialize({"train": [sample]}, config, out_dir=tmp_path / "out")
    assert (view.root / "train" / "images" / "a.jpg").exists()


# --- evaluate is common to all candidates ---------------------------------


class _PerfectDetector(BaseDetector):
    """Predicts exactly the already filtered truth. Only to test `evaluate`."""

    name = "_perfect_for_testing"
    license = "Apache-2.0"
    production_ready = True

    def __init__(self, truths: dict) -> None:
        self._truths = truths

    def predict(self, samples, *, weights, config):
        return {
            s.sample_id: [Prediction(q, 0.9) for q in self._truths[s.sample_id]]
            for s in samples
        }


def test_evaluate_is_provided_by_the_base_and_uses_our_metrics(tmp_path, config):
    """If each adapter brought its own, the rows would not be comparable."""
    quads = [_quad(0.3, 0.5, half_long=0.12), _quad(0.7, 0.5, half_long=0.12)]
    samples = [_write_sample(tmp_path / "src", f"s{i}", quads) for i in range(3)]

    fast = config.model_copy(
        update={"metrics": config.metrics.model_copy(update={"bootstrap_samples": 40})}
    )
    detector = _PerfectDetector({s.sample_id: quads for s in samples})
    report = detector.evaluate(samples, fast, weights=tmp_path / "unused.pt")

    assert report["map50"]["value"] == pytest.approx(1.0, abs=1e-6)
    assert report["n_images"] == 3
    assert report["bootstrap"]["unit"] == "image"


def test_the_truth_of_evaluate_goes_through_the_filter(tmp_path, config):
    """The filtered strip is not truth to detect, but not background either."""
    large = _quad(0.5, 0.5, half_long=0.30)
    strip = _quad(0.5, 0.5, half_long=0.30, ratio=30.0)
    sample = _write_sample(tmp_path / "src", "a", [large, strip])

    items = build_image_evals([sample], {"a": []}, config=config)
    assert len(items[0].truths) == 1
    assert len(items[0].ignored) == 1


def test_the_registry_is_the_only_thing_that_decides():
    """`get` knows no name: everything comes from the dictionary."""
    assert set(detectors()) == set(detector_base.REGISTRY)


# --- banknotes crossing the border ----------------------------------------


def _cfg(config, **detector):
    return config.model_copy(
        update={"detector": config.detector.model_copy(update=detector)}
    )


def _outside_the_frame() -> Quad:
    """Banknote sticking out on the left: vertices at negative x."""
    return canonicalize(Quad.from_xy([(-0.15, 0.3), (0.5, 0.3), (0.5, 0.7), (-0.15, 0.7)]))


def _coords(path: Path) -> list[float]:
    return [float(t) for t in path.read_text().split()[1:]]


def test_clip_pins_the_quad_to_the_frame(tmp_path, config):
    sample = _write_sample(tmp_path / "src", "a", [_outside_the_frame()])
    view = dataset_view.materialize(
        {"train": [sample]}, _cfg(config, out_of_bounds="clip"), out_dir=tmp_path / "out"
    )
    coords = _coords(view.root / "train" / "labels" / "a.txt")
    assert min(coords) >= 0.0 and max(coords) <= 1.0
    assert view.adjusted == 1
    assert view.out_of_bounds == "clip"


def test_keep_touches_nothing_but_counts_it(tmp_path, config):
    """With `keep` Ultralytics will discard the image; at least it is recorded."""
    sample = _write_sample(tmp_path / "src", "a", [_outside_the_frame()])
    view = dataset_view.materialize(
        {"train": [sample]}, _cfg(config, out_of_bounds="keep"), out_dir=tmp_path / "out"
    )
    assert min(_coords(view.root / "train" / "labels" / "a.txt")) < 0.0
    assert view.adjusted == 1


def test_pad_puts_the_quad_inside_and_enlarges_the_image(tmp_path, config):
    sample = _write_sample(tmp_path / "src", "a", [_outside_the_frame()], size=(160, 120))
    view = dataset_view.materialize(
        {"train": [sample]},
        _cfg(config, out_of_bounds="pad", pad_fraction=0.25),
        out_dir=tmp_path / "out",
    )
    coords = _coords(view.root / "train" / "labels" / "a.txt")
    assert min(coords) >= 0.0 and max(coords) <= 1.0

    with Image.open(view.root / "train" / "images" / "a.jpg") as im:
        assert im.size == (160 + 2 * 40, 120 + 2 * 30)


def test_pad_and_its_inverse_cancel_out():
    """If they were not exact inverses, predictions would come out shifted."""
    from testbank.detectors.dataset import pad_quad
    from testbank.detectors.ultralytics_obb import _unpad

    original = _quad(0.4, 0.5, half_long=0.2, ratio=2.0, theta=0.6)
    forward = pad_quad(original, 0.25)
    back = _unpad(Prediction(forward, 0.9), 0.25).quad
    for (x0, y0), (x1, y1) in zip(original.points, back.points):
        assert x1 == pytest.approx(x0, abs=1e-6)
        assert y1 == pytest.approx(y0, abs=1e-6)


def test_the_inverse_preserves_what_sticks_out_of_the_frame():
    """It is the only reason for `pad` to exist: if it clipped on the way back,
    it would equal clip and the cost of padding would buy nothing."""
    from testbank.detectors.dataset import pad_quad
    from testbank.detectors.ultralytics_obb import _unpad

    outside = _outside_the_frame()
    back = _unpad(Prediction(pad_quad(outside, 0.25), 0.9), 0.25).quad
    assert min(back.flat()) < 0.0


def test_the_default_policy_is_clip(config):
    assert config.detector.out_of_bounds.value == "clip"


def test_pad_clips_what_the_border_does_not_reach(tmp_path, config):
    """Without this clipping, `pad` with a small fraction would lose the whole image.

    It is what allows padding LITTLE: the border covers the common case and
    the clipping handles the residue, instead of having to pad for the worst
    case.
    """
    far_outside = canonicalize(
        Quad.from_xy([(-0.40, 0.3), (0.5, 0.3), (0.5, 0.7), (-0.40, 0.7)])
    )
    sample = _write_sample(tmp_path / "src", "a", [far_outside])
    view = dataset_view.materialize(
        {"train": [sample]},
        _cfg(config, out_of_bounds="pad", pad_fraction=0.05),
        out_dir=tmp_path / "out",
    )
    coords = _coords(view.root / "train" / "labels" / "a.txt")
    assert min(coords) >= 0.0 and max(coords) <= 1.0
    assert view.clipped_after_pad == 1


def test_with_plenty_of_padding_no_clipping_is_needed(tmp_path, config):
    sample = _write_sample(tmp_path / "src", "a", [_outside_the_frame()])
    view = dataset_view.materialize(
        {"train": [sample]},
        _cfg(config, out_of_bounds="pad", pad_fraction=0.25),
        out_dir=tmp_path / "out",
    )
    assert view.clipped_after_pad == 0
