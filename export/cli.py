"""ONNX export CLI (``odet export`` / ``python -m export``)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Maps subcommand name -> (module path, argv[0] for the target script)
_COMMANDS: Dict[str, Tuple[str, str]] = {
    "onnx": ("export.scripts.export_onnx", "export-onnx"),
    "infer": ("export.scripts.infer_onnx", "export-infer"),
    "demo": ("export.scripts.demo", "export-demo"),
    "preds": ("export.scripts.save_predictions_onnx", "export-preds"),
}

# Legacy python -m export names. Top-level odet export-* (TF / SavedModel) stay removed.
_ALIASES = {
    "export-onnx": "onnx",
    "export-infer": "infer",
    "export-demo": "demo",
    "export-preds": "preds",
}


def install_hint() -> str:
    root = Path(__file__).resolve().parent
    runtime = root / "requirements-runtime.txt"
    export = root / "requirements-export.txt"
    return (
        f"Infer/demo (no oriented-det): pip install -r {runtime}\n"
        f"Producer (odet export): uv pip install -e \".[export]\" "
        f"or pip install -r {export} (plus oriented-det)"
    )


def _print_help() -> None:
    print("Usage: odet export <command> [options]")
    print("       python -m export <command> [options]")
    print("ONNX only (no TensorFlow / SavedModel). demo / infer do not need PyTorch.")
    print("")
    print("Commands:")
    for name in sorted(_COMMANDS):
        print(f"  {name}")
    print("")
    print("Examples:")
    print(
        "  odet export onnx --config path/to/config.json "
        "--checkpoint path/to/model.pth --output ./onnx_export/model.onnx"
    )
    print(
        "  odet export infer --onnx ./onnx_export/model.onnx "
        "--images ./tiles --output ./onnx_export/predictions"
    )
    print("  odet export demo")
    print("")
    print(install_hint())


def _ensure_sys_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _invoke(module_path: str, prog: str, args: List[str]) -> None:
    _ensure_sys_path()
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise SystemExit(f"Failed to import {module_path}: {e}\n{install_hint()}") from e
    if not hasattr(mod, "main"):
        raise SystemExit(f"{module_path} has no main()")
    sys.argv = [prog] + args
    mod.main()


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return
    cmd, rest = argv[0], argv[1:]
    cmd = _ALIASES.get(cmd, cmd)
    if cmd not in _COMMANDS:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        _print_help()
        raise SystemExit(2)
    module_path, prog = _COMMANDS[cmd]
    _invoke(module_path, prog, rest)


if __name__ == "__main__":
    main()
