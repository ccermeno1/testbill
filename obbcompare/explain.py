"""Class-activation maps for ONNX detectors: which parts of the photo raise a detection's score.

Both methods work on the feature maps the detection head reads (one per level, strides
8 / 16 / 32), so the heatmap has the resolution of those strides: blobs, not edges.

* **Grad-CAM** in its element-wise form (HiResCAM): gradient of the score of ONE detection
  w.r.t. each feature map, multiplied element-wise with the map, summed over channels and
  rectified. Plain Grad-CAM averages the gradient over space first; for a detector the
  score of one box has gradient on a handful of cells and the average dilutes it to noise.
  ONNX Runtime has no autograd, so the graph is converted to PyTorch with ``onnx2torch``
  (checked numerically against ONNX Runtime before being trusted).
* **EigenCAM**: no gradient and no target, the projection of the activations on their
  first principal component: what the network looks at in general. Runs on ONNX Runtime
  only, so it works even when the conversion to PyTorch does not.

The feature maps are found automatically: for every head level (a resolution at which the
graph has tensors that feed only the scores or only the boxes, i.e. the separate cls / reg
branches) the deepest tensor that feeds BOTH outputs is the input of the head. They can be
set by name in ``model.json`` -> ``explain.feature_tensors``.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime as ort

from .detector import OnnxDetector

# ---------------------------------------------------------------------- graph analysis


def _static_model(det: OnnxDetector) -> onnx.ModelProto:
    model = onnx.load(str(det.cfg.onnx_path))
    # fix batch (and H, W when dynamic) so that shape inference gives numbers
    hw = det.static_hw or (det.size, det.size)
    for inp in model.graph.input:
        if inp.name == det.input_name:
            dims = inp.type.tensor_type.shape.dim
            dims[0].dim_value = 1
            dims[2].dim_value, dims[3].dim_value = hw
    for out in model.graph.output:  # stale symbolic dims would clash with the inferred ones
        out.type.tensor_type.ClearField("shape")
    return onnx.shape_inference.infer_shapes(model)


def _ancestors(model: onnx.ModelProto, *outputs: str) -> set[str]:
    producer = {o: n for n in model.graph.node for o in n.output}
    seen, stack = set(), list(outputs)
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        node = producer.get(t)
        if node is not None:
            stack.extend(i for i in node.input if i)
    return seen


def find_feature_tensors(det: OnnxDetector) -> list[str]:
    """Names of the tensors the head reads, one per level (see the module docstring).

    A head level is a resolution with "branch" tensors: activations that feed only the
    scores or only the boxes. Its input is the deepest tensor at that resolution that
    feeds both outputs AND every branch tensor (the point where cls and reg split; in
    YOLOX the objectness makes the reg branch feed the scores too, hence the last rule).
    """
    model = _static_model(det)
    order = {o: i for i, n in enumerate(model.graph.node) for o in n.output}
    shapes = {}
    for vi in list(model.graph.value_info) + list(model.graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        # activations only (not weights) with a spatial extent (not pooled)
        if vi.name in order and len(dims) == 4 and all(d > 0 for d in dims) and dims[2] > 1 and dims[3] > 1:
            shapes[vi.name] = dims
    to_scores = _ancestors(model, *det.score_names)
    to_boxes = _ancestors(model, *det.box_names)

    branches = defaultdict(list)
    for t, dims in shapes.items():
        if (t in to_scores) != (t in to_boxes):
            branches[tuple(dims[2:])].append(t)
    features = []
    for hw in sorted(branches, reverse=True):  # stride 8 first
        feeds_all = set.intersection(*(_ancestors(model, t) for t in branches[hw]))
        cands = [t for t, dims in shapes.items()
                 if tuple(dims[2:]) == hw and dims[1] >= 8 and t in to_scores and t in to_boxes
                 and t in feeds_all and t not in branches[hw]]
        if cands:
            features.append(max(cands, key=order.__getitem__))
    return features


def _with_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    """Copy of the graph that also returns the tensors ``names``."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    info = {vi.name: vi for vi in m.graph.value_info}
    for n in names:
        if n not in have:
            m.graph.output.append(info[n] if n in info else onnx.helper.make_empty_tensor_value_info(n))
    return m


# ---------------------------------------------------------------------- maps


def _normalize(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    peak = float(cam.max())
    return cam / peak if peak > 0 else cam


def _to_image(maps: list[np.ndarray], input_hw: tuple[int, int], scale: float,
              image_hw: tuple[int, int]) -> np.ndarray:
    """Maps over the network input -> one map over the original photo, in [0, 1].

    Every level is resized to the network input and summed, then the padding is cropped
    and the rest resized to the photo."""
    in_h, in_w = input_hw
    total = np.zeros((in_h, in_w), np.float32)
    for m in maps:
        total += cv2.resize(m.astype(np.float32), (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    h0, w0 = image_hw
    rh, rw = max(1, round(h0 * scale)), max(1, round(w0 * scale))
    return _normalize(cv2.resize(total[:rh, :rw], (w0, h0), interpolation=cv2.INTER_LINEAR))


def _eigen(feat: np.ndarray) -> np.ndarray:
    """(C, H, W) -> (H, W): projection on the first right singular vector. Its sign is
    arbitrary; it is chosen so that the cell with the most activation energy is positive."""
    c, h, w = feat.shape
    flat = feat.reshape(c, h * w).astype(np.float64)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    comp = vt[0]
    if comp[int(np.argmax(np.linalg.norm(flat, axis=0)))] < 0:
        comp = -comp
    return comp.reshape(h, w)


class Explainer:
    """CAM for one model. EigenCAM always; Grad-CAM when the conversion to torch works."""

    def __init__(self, det: OnnxDetector):
        self.det = det
        self.features = det.cfg.feature_tensors or find_feature_tensors(det)
        if not self.features:
            raise RuntimeError("no se encontraron los mapas de características de la cabeza")
        base = onnx.load(str(det.cfg.onnx_path))
        self.model = _with_outputs(onnx.shape_inference.infer_shapes(base), self.features)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(self.model.SerializeToString(), opts,
                                            providers=["CPUExecutionProvider"])
        self._torch = None
        self.gradcam_error: str | None = None

    # ---- EigenCAM on ONNX Runtime
    def eigencam(self, bgr: np.ndarray) -> np.ndarray:
        x, scale = self.det.preprocess(bgr)
        feats = self.session.run(self.features, {self.det.input_name: x})
        return _to_image([_eigen(f[0]) for f in feats], x.shape[-2:], scale, bgr.shape[:2])

    # ---- Grad-CAM (HiResCAM) through onnx2torch
    def _torch_model(self):
        if self._torch is None and self.gradcam_error is None:
            try:
                self._torch = _convert_and_check(self.model, self.session, self.det)
            except Exception as e:  # noqa: BLE001 - shown in the app, EigenCAM keeps working
                self.gradcam_error = f"{type(e).__name__}: {e}"
        return self._torch

    @property
    def gradcam_available(self) -> bool:
        return self._torch_model() is not None

    def gradcam(self, bgr: np.ndarray, targets: list[tuple[int, int]]) -> np.ndarray:
        """``targets``: (prior index, class) of the detections whose summed score is explained."""
        import torch

        tm = self._torch_model()
        if tm is None:
            raise RuntimeError(self.gradcam_error)
        x, scale = self.det.preprocess(bgr)
        # weights are frozen: the graph is built from the input
        outs = tm(torch.from_numpy(x).requires_grad_(True))
        names = [o.name for o in self.model.graph.output]
        by_name = dict(zip(names, outs))
        if self.det.levels:  # raw maps: same flattening as detector.decode_raw
            scores = torch.cat([by_name[n].sigmoid().flatten(2).transpose(1, 2) for n in self.det.score_names], 1)
        else:
            scores = by_name[self.det.score_names[0]]
            if self.det.cfg.scores_layout == "CN":
                scores = scores.transpose(1, 2)
        target = sum(scores[0, p, c] for p, c in targets)
        feats = [by_name[n] for n in self.features]
        grads = torch.autograd.grad(target, feats, allow_unused=True)
        maps = []
        for f, g in zip(feats, grads):
            if g is not None:
                maps.append(torch.relu((g[0] * f[0]).sum(0)).detach().numpy())
        if not maps:
            return np.zeros(bgr.shape[:2], np.float32)
        return _to_image(maps, x.shape[-2:], scale, bgr.shape[:2])


# ---------------------------------------------------------------------- onnx -> torch


def _register_newer_opsets() -> None:
    """onnx2torch lags behind the ONNX opsets (e.g. no ``Resize`` v18, which torch writes at
    opset 18). Missing versions fall back to the newest older converter; the result is
    checked numerically afterwards, so a converter that does not fit is caught."""
    from onnx2torch.node_converters import registry

    reg = registry._CONVERTER_REGISTRY
    by_op = defaultdict(dict)
    for desc, conv in reg.items():
        by_op[(desc.domain, desc.operation_type)][desc.version] = conv
    for (domain, op), versions in by_op.items():
        for v in range(min(versions), 25):
            if v not in versions:
                versions[v] = versions[v - 1]
                reg[registry.OperationDescription(domain=domain, operation_type=op, version=v)] = versions[v]


# since opset 18 these take ``axes`` as an input instead of an attribute
_REDUCE_AXES_INPUT_18 = {"ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2",
                         "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare"}


def _prepare_for_onnx2torch(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrites what onnx2torch does not read yet into an equivalent older form:

    * ``Reshape(allowzero=1)`` (torch dynamo exporter): with a constant shape without zeros
      the attribute changes nothing, so it is dropped.
    * opset-18 ``Reduce*`` with constant ``axes`` input: the axes go back to an attribute,
      which the opset-13 converter (see ``_register_newer_opsets``) understands.
    """
    consts = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = onnx.numpy_helper.to_array(a.t)
    m = onnx.ModelProto()
    m.CopyFrom(model)
    for n in m.graph.node:
        if n.op_type == "Reshape":
            shape = consts.get(n.input[1])
            for a in list(n.attribute):
                if a.name == "allowzero" and a.i == 1 and shape is not None and not (shape == 0).any():
                    n.attribute.remove(a)
        elif n.op_type in _REDUCE_AXES_INPUT_18 and len(n.input) > 1 and n.input[1] in consts:
            axes = [int(v) for v in consts[n.input[1]].reshape(-1)]
            del n.input[1:]
            n.attribute.append(onnx.helper.make_attribute("axes", axes))
    return m


_registered = False


def _convert_and_check(model: onnx.ModelProto, session: ort.InferenceSession, det: OnnxDetector):
    global _registered
    import torch
    from onnx2torch import convert

    if not _registered:
        _register_newer_opsets()
        _registered = True
    tm = convert(_prepare_for_onnx2torch(model)).eval()
    for p in tm.parameters():
        p.requires_grad_(False)
    hw = det.static_hw or (det.size, det.size)
    x = np.random.default_rng(0).uniform(0, 255, (1, 3, *hw)).astype(np.float32)
    if det.cfg.mean is not None:
        x = (x - 127.5) / 64.0
    ref = session.run(None, {det.input_name: x})
    with torch.no_grad():
        got = tm(torch.from_numpy(x))
    got = got if isinstance(got, (list, tuple)) else [got]
    names = [o.name for o in model.graph.output]
    for name, r, g in zip(names, ref, got):
        if name in det.box_names:
            continue  # coordinates in pixels and angles wrap at +-pi/2: compared via the scores
        err = float(np.abs(g.numpy() - r).max()) / (float(np.abs(r).max()) + 1e-6)
        if err > 1e-2:
            raise RuntimeError(f"la conversión a PyTorch no reproduce ONNX Runtime en '{name}' (error rel. {err:.1e})")
    return tm
