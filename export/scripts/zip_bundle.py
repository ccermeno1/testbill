#!/usr/bin/env python3
"""Zip an onnx_export bundle, excluding Python cache files."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

_SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache"}
_SKIP_SUFFIXES = {".pyc", ".pyo"}


def zip_bundle(src: Path, dest: Path) -> Path:
    src = src.resolve()
    dest = dest.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Missing bundle dir: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    archive_root = src.name
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            if path.suffix in _SKIP_SUFFIXES:
                continue
            if path.resolve() == dest:
                continue
            arcname = Path(archive_root) / path.relative_to(src)
            zf.write(path, arcname.as_posix())
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description="Zip onnx_export without Python cache.")
    p.add_argument("--dir", type=Path, required=True, help="Bundle directory (e.g. onnx_export).")
    p.add_argument("--output", type=Path, required=True, help="Zip path (e.g. onnx_export.zip).")
    args = p.parse_args()
    out = zip_bundle(args.dir, args.output)
    n = zipfile.ZipFile(out).namelist()
    print(f"Wrote {out} ({out.stat().st_size / (1024 * 1024):.1f} MiB, {len(n)} files)")


if __name__ == "__main__":
    main()
