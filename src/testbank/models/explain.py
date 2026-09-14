"""Class-activation maps for a detection: which parts of the photo push the
score of ONE box up.

Two methods, both on the feature maps the detection head reads (the FPN
levels, strides 8/16/32 and up), so the resolution is the stride's: blobs,
not edges.

* **Grad-CAM**, in its element-wise form (HiResCAM, Draelos & Carin 2020):
  the gradient of the chosen detection's score w.r.t. each feature map,
  multiplied element-wise with the map and summed over channels, rectified.
  The original Grad-CAM averages the gradient over space first; for a
  detector that is fatal, because the score of ONE box has gradients on a
  handful of cells and the average dilutes them into the noise (measured
  here: an all-zero map on every level). The element-wise form is also the
  one proven to reflect the model's computation. It answers "what raises
  this score", per detection; it says nothing about the angle or the
  tightness of the box: only the score is differentiated.
* **EigenCAM** (Muhammad & Yeasin, 2020): no gradients, no target -- the
  projection of the activations on their first principal component. It
  answers "what the network is looking at at all", which is what one wants
  when nothing was detected and there is no score to differentiate.

The maps of every level are resized to the input and summed, then scaled to
[0, 1]. Torch and OpenCV only; the hooks are removed after every call and
the model is left as it was (eval mode, no grads stored).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from torch import nn

METHODS = ("gradcam", "eigencam")


class Captured:
    """What a forward hook on the head saw: its input feature maps (the
    tensors the CAM is computed on) and its raw outputs."""

    def __init__(self) -> None:
        self.features: list[torch.Tensor] = []
        self.outputs = None


@contextmanager
def capture_head(head: nn.Module, *, retain_grad: bool):
    """Hooks `head` for one forward pass. The feature maps are the positional
    tensor arguments of the head (a list/tuple of levels, or several
    tensors); with `retain_grad` they keep their gradient after `backward`."""
    seen = Captured()

    def hook(module, args, kwargs, output):
        features: list[torch.Tensor] = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                features.append(arg)
            elif isinstance(arg, (list, tuple)):
                features.extend(t for t in arg if isinstance(t, torch.Tensor))
        if retain_grad:
            for t in features:
                if t.requires_grad:
                    t.retain_grad()
        seen.features = features
        seen.outputs = output

    handle = head.register_forward_hook(hook, with_kwargs=True)
    try:
        yield seen
    finally:
        handle.remove()


def _resize_sum(maps: Sequence[np.ndarray], out_hw: tuple[int, int]) -> np.ndarray:
    height, width = out_hw
    total = np.zeros((height, width), dtype=np.float32)
    for m in maps:
        total += cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return total


def normalize(cam: np.ndarray) -> np.ndarray:
    """Rectify and scale to [0, 1]; an all-zero map stays all zero."""
    cam = np.maximum(cam, 0.0)
    peak = float(cam.max())
    return cam / peak if peak > 0 else cam


def gradcam_maps(
    features: Sequence[torch.Tensor], out_hw: tuple[int, int]
) -> np.ndarray:
    """After `backward` on the target: one map per level from the retained
    gradients (element-wise, see the module docstring), resized to `out_hw`
    and summed. Levels without a gradient (the target did not depend on
    them) contribute nothing."""
    maps = []
    for f in features:
        if f.grad is None:
            continue
        cam = torch.relu((f.grad[0] * f[0]).sum(dim=0))  # (H, W)
        maps.append(cam.detach().cpu().numpy())
    if not maps:
        return np.zeros(out_hw, dtype=np.float32)
    return normalize(_resize_sum(maps, out_hw))


def eigencam_maps(
    features: Sequence[torch.Tensor], out_hw: tuple[int, int]
) -> np.ndarray:
    """Projection of each level's activations on their first singular
    vector (no centring: the dominant direction of the raw activations, as
    in the paper). A singular vector's sign is arbitrary, so it is chosen to
    make the cell with the most activation energy positive."""
    maps = []
    for f in features:
        a = f[0].detach().float()  # (C, H, W)
        c, h, w = a.shape
        flat = a.reshape(c, h * w)
        try:
            _, _, vh = torch.linalg.svd(flat, full_matrices=False)
        except RuntimeError:
            continue
        component = vh[0]  # (HW,)
        strongest = int(torch.argmax(flat.norm(dim=0)))
        if component[strongest] < 0:
            component = -component
        maps.append(component.reshape(h, w).cpu().numpy())
    if not maps:
        return np.zeros(out_hw, dtype=np.float32)
    return normalize(_resize_sum(maps, out_hw))


def nearest_cell_score(
    scores: torch.Tensor,
    centres: torch.Tensor,
    target_xy: tuple[float, float],
    radius: float,
) -> torch.Tensor:
    """The target scalar for Grad-CAM on a detection: the highest score
    among the cells whose centre lies within `radius` pixels of the
    detection's centre (the cell the detection came out of is one of them);
    the nearest cell if none is that close. Differentiable through `scores`.
    """
    target = torch.tensor(target_xy, dtype=centres.dtype, device=centres.device)
    distances = torch.linalg.norm(centres - target, dim=1)
    close = distances <= radius
    if bool(close.any()):
        masked = torch.where(close, scores, torch.full_like(scores, -1.0))
        return scores[int(torch.argmax(masked))]
    return scores[int(torch.argmin(distances))]


def explain(
    model: nn.Module,
    head: nn.Module,
    tensor: torch.Tensor,
    *,
    method: str,
    target: Callable[[Captured], torch.Tensor] | None = None,
) -> np.ndarray:
    """One CAM for `tensor` (1, 3, H, W), in [0, 1] at the input size.

    `target(captured) -> scalar` is required for Grad-CAM: it builds the
    detection's score from what the head produced (`captured.outputs`).
    The model is run in eval mode; gradients are enabled only for the pass
    and nothing is left on the parameters.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    out_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))
    # Per module, not the top flag: a wrapper around the real network has
    # its own flag, and restoring that one would flip the network's.
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        if method == "eigencam":
            with torch.no_grad(), capture_head(head, retain_grad=False) as seen:
                model(tensor)
            return eigencam_maps(seen.features, out_hw)
        if target is None:
            raise ValueError("gradcam needs a target: the detection whose score to explain")
        with torch.enable_grad(), capture_head(head, retain_grad=True) as seen:
            x = tensor.detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            model(x)
            scalar = target(seen)
            if not scalar.requires_grad:
                return np.zeros(out_hw, dtype=np.float32)
            scalar.backward()
            cam = gradcam_maps(seen.features, out_hw)
        model.zero_grad(set_to_none=True)
        return cam
    finally:
        for module, training in modes:
            module.training = training


def overlay(image_bgr: np.ndarray, cam: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    """The heat map (any size, [0, 1]) blended over the image, JET colours:
    red is what raised the score, blue is indifferent."""
    height, width = image_bgr.shape[:2]
    resized = cv2.resize(cam.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    colour = cv2.applyColorMap((np.clip(resized, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colour, alpha, 0.0)


__all__ = [
    "METHODS",
    "Captured",
    "capture_head",
    "eigencam_maps",
    "explain",
    "gradcam_maps",
    "nearest_cell_score",
    "normalize",
    "overlay",
]
