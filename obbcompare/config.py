"""Uniform description of an exported model (``model.json``) and discovery of ``models/``.

Every model lives in its own folder under ``models/``. The folder holds the ``.onnx`` and
its contract, in one of two forms:

* ``model.json``: the uniform format of this app (see README). ``rtmdet_export_onnx.py``
  writes it directly; it can also be written by hand for any detector that returns
  ``boxes (1, N, 5)`` + ``scores (1, N, C)`` or ``(1, C, N)``.
* ``<name>.metadata.json`` next to ``<name>.onnx``: what the export scripts of the
  YOLOX-OBB and PP-YOLOE-R branches write. It is translated to the uniform format on load,
  so those exports can be copied as they are.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    name: str
    onnx_path: Path
    family: str = "custom"
    class_names: list[str] = field(default_factory=lambda: ["object"])
    # preprocessing: letterbox (long side = size), pasted top-left, then optional mean/std
    input_name: str | None = None      # None: the graph's only input
    size: int = 640                    # long side after resizing
    color: str = "BGR"                 # channel order the graph expects
    pad_value: float = 114.0
    pad_multiple: int = 32             # only used when the graph input has dynamic H, W
    mean: list[float] | None = None    # 0..255 units; None when normalisation is inside the graph
    std: list[float] | None = None
    # outputs
    boxes_output: str | None = None    # None: the output whose last dim is 5
    scores_output: str | None = None   # None: the other one
    scores_layout: str | None = None   # "NC" (1, N, C) or "CN" (1, C, N); None: from the shapes
    # raw head maps instead of decoded boxes (RTMDet-R export of feature/rt_refactor):
    # output names in (cls, reg, angle) order per level, and the stride of each level
    raw_outputs: list[str] = field(default_factory=list)
    strides: list[int] = field(default_factory=lambda: [8, 16, 32])
    # recommended post-processing
    score_thr: float = 0.5
    nms_iou: float = 0.3
    max_det: int = 100
    nms_pre: int = 1000
    # Grad-CAM / EigenCAM: names of the feature maps the head reads; empty = auto-detect
    feature_tensors: list[str] = field(default_factory=list)
    source: str = "model.json"         # which file the config came from (shown in the app)

    @property
    def key(self) -> str:
        return str(self.onnx_path)


def _from_uniform(path: Path) -> ModelConfig:
    d = json.loads(path.read_text(encoding="utf-8"))
    inp, out, post = d.get("input", {}), d.get("outputs", {}), d.get("postprocess", {})
    norm = inp.get("normalize") or {}
    onnx_name = d.get("onnx") or next(path.parent.glob("*.onnx")).name
    return ModelConfig(
        name=d.get("name", path.parent.name),
        onnx_path=path.parent / onnx_name,
        family=d.get("family", "custom"),
        class_names=d.get("class_names", ["object"]),
        input_name=inp.get("name"),
        size=int(inp.get("size", 640)),
        color=inp.get("color", "BGR").upper(),
        pad_value=float(inp.get("pad_value", 114)),
        pad_multiple=int(inp.get("pad_multiple", 32)),
        mean=norm.get("mean"),
        std=norm.get("std"),
        boxes_output=out.get("boxes"),
        scores_output=out.get("scores"),
        scores_layout=out.get("scores_layout"),
        raw_outputs=list(out.get("raw", [])),
        strides=list(out.get("strides", [8, 16, 32])),
        score_thr=float(post.get("score_thr", 0.5)),
        nms_iou=float(post.get("nms_iou", 0.3)),
        max_det=int(post.get("max_det", 100)),
        nms_pre=int(post.get("nms_pre", 1000)),
        feature_tensors=list(d.get("explain", {}).get("feature_tensors", [])),
        source=path.name,
    )


def _from_native_metadata(onnx_path: Path, meta_path: Path) -> ModelConfig:
    """``.metadata.json`` of the YOLOX-OBB / PP-YOLOE-R export scripts -> uniform config."""
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    inp = m.get("input", {})
    shape = inp.get("shape", [1, 3, 640, 640])
    size = int(max(shape[-2:])) if all(isinstance(s, int) for s in shape[-2:]) else 640
    pad = inp.get("padding", {})
    model_name = str(m.get("model", onnx_path.stem))
    rec = m.get("recommended", {})
    low = model_name.lower().replace("-", "")
    family = "ppyoloe_r" if "ppyoloe" in low else "rtmdet_r" if "rtmdet" in low else "yolox_obb"
    # RTMDet-R (feature/rt_refactor) exports the raw maps: cls/reg/angle per level
    raw = m["outputs"] if isinstance(m.get("outputs"), list) else []
    return ModelConfig(
        name=f"{model_name} ({onnx_path.stem})",
        onnx_path=onnx_path,
        family=family,
        class_names=m.get("class_names") or m.get("classes") or ["object"],
        input_name=inp.get("name"),
        size=size,
        color=str(inp.get("color", "BGR")).upper(),
        # YOLOX pastes on 114; PP-YOLOE-R declares its padding (0), RTMDet-R as square_value
        pad_value=float(pad.get("value", pad.get("square_value", 114))),
        pad_multiple=int(pad.get("to_multiple_of", 32)),
        # the three exports normalise inside the graph (RTMDet-R lists its mean/std but
        # applies them in the graph too)
        mean=None if inp.get("normalization_inside_graph", True) else inp.get("mean_0_255"),
        std=None if inp.get("normalization_inside_graph", True) else inp.get("std_0_255"),
        boxes_output=None if raw else "boxes",
        scores_output=None if raw else "scores",
        raw_outputs=raw,
        strides=list(m.get("output_semantics", {}).get("strides", [8, 16, 32])),
        score_thr=float(rec.get("score_thr", 0.5)),
        nms_iou=float(rec.get("nms_iou", 0.3)),
        max_det=int(rec.get("max_detections", 100)),
        source=meta_path.name,
    )


def discover(models_dir: Path) -> tuple[list[ModelConfig], list[str]]:
    """All models under ``models_dir`` (one folder per model, nested folders allowed).
    Returns the configs and a list of warnings for folders that could not be read."""
    configs, warnings = [], []
    seen: set[Path] = set()
    for uniform in sorted(models_dir.rglob("model.json")):
        try:
            cfg = _from_uniform(uniform)
            if not cfg.onnx_path.exists():
                raise FileNotFoundError(cfg.onnx_path.name)
            configs.append(cfg)
            seen.add(cfg.onnx_path.resolve())
        except Exception as e:  # noqa: BLE001 - one bad folder must not hide the others
            warnings.append(f"{uniform}: {e}")
    for onnx_path in sorted(models_dir.rglob("*.onnx")):
        if onnx_path.resolve() in seen:
            continue
        meta = onnx_path.with_suffix(".metadata.json")
        if meta.exists():
            try:
                configs.append(_from_native_metadata(onnx_path, meta))
            except Exception as e:  # noqa: BLE001
                warnings.append(f"{meta}: {e}")
        else:
            warnings.append(f"{onnx_path}: sin model.json ni {meta.name}, se ignora")
    return configs, warnings
