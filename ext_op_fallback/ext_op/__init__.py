"""
Sustituto en Paddle puro de las ops custom de PaddleDetection (`ppdet/ext_op`),
que en Windows exigen Visual Studio + CUDA toolkit para compilarse.

PP-YOLOE-R necesita `rbox_iou` en el asignador (entrenamiento) y en la metrica RBOX
(evaluacion). `matched_rbox_iou` solo la usa FCOSRAssigner. `nms_rotated` no la usa
PP-YOLOE-R (su NMS es el `multiclass_nms` nativo de Paddle con poligonos).

Si en sys.path hay una extension `ext_op` compilada (instalada con
`python ppdet/ext_op/setup.py install`), se delega en ella automaticamente.
"""
import importlib.machinery
import importlib.util
import os
import sys

__all__ = ["rbox_iou", "matched_rbox_iou", "nms_rotated", "IS_FALLBACK"]

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_PKG_DIR)


def _find_compiled():
    """Busca otro modulo llamado `ext_op` fuera de este paquete (la op compilada)."""
    paths = []
    for p in sys.path:
        ap = os.path.abspath(p or os.getcwd())
        if ap != _PARENT:
            paths.append(ap)
    try:
        spec = importlib.machinery.PathFinder.find_spec("ext_op", paths)
    except Exception:
        return None
    if spec is None or not spec.origin:
        return None
    if os.path.abspath(spec.origin).startswith(_PKG_DIR):
        return None
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "rbox_iou"):
            return mod
    except Exception as e:  # la compilada existe pero no carga: usamos el fallback
        sys.stderr.write(f"[ext_op fallback] no se pudo cargar la ext_op compilada ({e}); "
                         "se usa la implementacion en Paddle puro\n")
    return None


_compiled = _find_compiled()

if _compiled is not None:
    IS_FALLBACK = False
    rbox_iou = _compiled.rbox_iou
    matched_rbox_iou = _compiled.matched_rbox_iou
    nms_rotated = getattr(_compiled, "nms_rotated", None)
else:
    IS_FALLBACK = True
    from ._rbox_iou import rbox_iou, matched_rbox_iou  # noqa: E402

    def nms_rotated(*args, **kwargs):
        raise NotImplementedError(
            "nms_rotated no esta implementada en el fallback de Paddle puro; "
            "PP-YOLOE-R no la necesita. Compila ppdet/ext_op si usas otro modelo rotado.")
