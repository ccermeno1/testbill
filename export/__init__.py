"""ONNX export package for OrientedDet. CLI: ``odet export`` / ``python -m export``.

Inference (``demo`` / ``infer``) needs only numpy, Pillow, and onnxruntime —
see ``requirements-runtime.txt``. The same consumer modules are copied into
``onnx_export/`` at export. Exporting a checkpoint still needs oriented-det
plus ``requirements-export.txt``.
"""

from __future__ import annotations

from .cli import install_hint

__all__ = ["install_hint"]
