"""The registry, the `load`/`predict` contract, and framework isolation."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from testbank.config import Config
from testbank.detectors import base as detector_base
from testbank.detectors import detectors, get
from testbank.detectors.base import BaseDetector, Detector, DetectorError, register
from testbank.detectors.yolox_obb import PARAMETER_COUNTS

SRC = Path(__file__).resolve().parents[1] / "src" / "testbank"

# --- registry -------------------------------------------------------------


def test_the_runs_of_main_find_their_adapter_by_name():
    """`run.json` names the adapter `main` trained with; these names must
    not change or the run cannot be served."""
    assert set(detectors()) >= {
        "yolox-obb-nano", "yolox-obb-tiny", "yolox-obb-small",
        "yolox-obb-ddgrcf-port", "rotated-fcos-r50", "rotated-fcos-r18",
    }
    for name in detectors():
        assert isinstance(get(name), Detector)
        assert get(name).license == "Apache-2.0"


def test_unknown_detector_says_which_exist():
    with pytest.raises(DetectorError, match="yolox-obb-nano"):
        get("nope")


def test_registering_the_same_name_twice_is_an_error():
    with pytest.raises(DetectorError, match="duplicate"):

        @register
        class Twice(BaseDetector):
            name = "yolox-obb-nano"


def test_a_detector_without_a_name_is_not_registered():
    with pytest.raises(DetectorError, match="name"):

        @register
        class Nameless(BaseDetector):
            pass


def test_the_base_has_no_loaded_model_and_no_predict():
    base = BaseDetector()
    assert base.load(Path("w.pt"), Config()) is None
    with pytest.raises(NotImplementedError):
        base.predict([], weights=Path("w.pt"), config=Config())


def test_the_parameter_table_matches_the_real_networks():
    pytest.importorskip("torch")
    from testbank.models.yolox_obb import YoloxObb

    for variant, params in PARAMETER_COUNTS.items():
        assert YoloxObb(variant, num_classes=1).parameter_count() == params


# --- isolation ------------------------------------------------------------

FRAMEWORK_ADAPTERS = {
    "oriented_det": SRC / "detectors" / "oriented_det.py",
    "streamlit": SRC / "app.py",
}


def _imports_package(path: Path, package: str) -> bool:
    """Looks for real imports, not mentions in comments or strings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == package for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == package:
            return True
    return False


@pytest.mark.parametrize("package", sorted(FRAMEWORK_ADAPTERS))
def test_every_foreign_package_stays_behind_one_module(package):
    """torch is the base; oriented-det is imported by exactly one adapter,
    Streamlit by the page and nothing else, so the library and the CLI work
    without them."""
    owner = FRAMEWORK_ADAPTERS[package]
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if path != owner and _imports_package(path, package)
    ]
    assert offenders == [], f"{package} imported outside {owner.name}: {offenders}"
    assert _imports_package(owner, package), "guard of the guard"


def test_importing_the_registry_does_not_need_torch():
    code = (
        "import sys; sys.modules['torch'] = None; "
        "import testbank.detectors as d, testbank.serve, testbank.cli; print(len(d.detectors()))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr[-800:]
    assert int(out.stdout.strip()) == 6


def test_nothing_from_the_training_side_survived():
    """The branch serves; it does not train, split or score."""
    modules = {p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")}
    for gone in ("models/train.py", "models/losses.py", "models/augment.py",
                 "metrics/evaluate.py", "data/splits.py", "experiment/runner.py"):
        assert gone not in modules
    assert not hasattr(detector_base.BaseDetector, "train")
    assert not hasattr(detector_base.BaseDetector, "evaluate")
