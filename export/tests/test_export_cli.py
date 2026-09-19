"""CLI wiring smoke (no Makefile)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_cli_export_commands_map_to_python_modules() -> None:
    """Export lives in this repo; former names must not be odet subcommands."""
    from export import cli

    try:
        from oriented_det import cli as odet_cli
    except ImportError:
        odet_cli = None
    if odet_cli is not None:
        for name in ("export-tf", "export-onnx", "export-detect", "export-savedmodel", "export-preds"):
            assert name not in odet_cli._COMMANDS
    for name, module in (
        ("onnx", "export.scripts.export_onnx"),
        ("infer", "export.scripts.infer_onnx"),
        ("demo", "export.scripts.demo"),
        ("preds", "export.scripts.save_predictions_onnx"),
    ):
        assert cli._COMMANDS[name][0] == module
    assert "tf" not in cli._COMMANDS
    assert "detect" not in cli._COMMANDS
    assert "savedmodel" not in cli._COMMANDS


@pytest.mark.parametrize(
    "argv",
    [
        ["onnx", "--help"],
        ["infer", "--help"],
        ["demo", "--help"],
        ["preds", "--help"],
        ["export-onnx", "--help"],
    ],
)
def test_export_subcommand_help(argv: list[str], capsys) -> None:
    from export.cli import main

    with pytest.raises(SystemExit) as ei:
        main(argv)
    assert ei.value.code in (0, None)
    out = capsys.readouterr().out
    assert "usage:" in out.lower() or "Usage:" in out


def test_unknown_tf_command_exits(capsys) -> None:
    from export.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["tf"])
    assert ei.value.code == 2
    err = capsys.readouterr()
    assert "Unknown command" in err.err


def test_copy_runtime_sidecars_writes_consumer_stack(tmp_path: Path) -> None:
    from export.runtime import copy_runtime_sidecars

    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"stub")
    written = copy_runtime_sidecars(onnx)
    rel = {str(p.relative_to(tmp_path)) for p in written}
    assert rel == {
        "nms.py",
        "preprocess.py",
        "postprocess.py",
        "runtime.py",
        "ort_runtime.py",
        "requirements-runtime.txt",
        "demo.py",
        "infer_onnx.py",
        "demo/planes_pleiades_neo.jpg",
    }
    src = Path(__file__).resolve().parents[1]
    assert (tmp_path / "nms.py").read_bytes() == (src / "nms.py").read_bytes()
    assert (tmp_path / "runtime.py").read_bytes() == (src / "runtime.py").read_bytes()
    assert (tmp_path / "demo.py").read_bytes() == (src / "scripts" / "demo.py").read_bytes()
    assert (tmp_path / "requirements-runtime.txt").read_text(encoding="utf-8") == (
        src / "requirements-runtime.txt"
    ).read_text(encoding="utf-8")


def test_zip_bundle_skips_pycache(tmp_path: Path) -> None:
    import zipfile

    from export.scripts.zip_bundle import zip_bundle

    bundle = tmp_path / "onnx_export"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"stub")
    (bundle / "runtime.py").write_text("ok\n", encoding="utf-8")
    cache = bundle / "__pycache__"
    cache.mkdir()
    (cache / "runtime.cpython-312.pyc").write_bytes(b"cache")
    (bundle / "nms.pyc").write_bytes(b"cache")
    out = tmp_path / "onnx_export.zip"
    zip_bundle(bundle, out)
    names = zipfile.ZipFile(out).namelist()
    assert "onnx_export/model.onnx" in names
    assert "onnx_export/runtime.py" in names
    assert all("__pycache__" not in n and not n.endswith(".pyc") for n in names)
