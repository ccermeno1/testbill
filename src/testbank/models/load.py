"""Rebuilding a trained network from a run's weights, and the input
conversion those weights expect."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from testbank.models.yolox_obb import HeadSpec, YoloxObb


def image_to_input(image_bgr: np.ndarray) -> torch.Tensor:
    """OpenCV image (BGR, uint8) -> `(3, H, W)` tensor fed to the network.

    **Raw BGR in 0-255, not normalized.** It is the convention of YOLOX
    (Megvii) and of DDGRCF, and therefore the one the trained weights expect;
    training on `main` uses this same function. Changing it here alone would
    invalidate every saved weight.
    """
    return torch.from_numpy(np.ascontiguousarray(image_bgr.transpose(2, 0, 1))).float()


def pick_device() -> torch.device:
    """`cuda` if available, else `mps` (Apple's GPU), else `cpu`. Can be
    forced with `TESTBANK_DEVICE=cpu`: MPS and CUDA do not guarantee the same
    arithmetic as the CPU."""
    import os

    forced = os.environ.get("TESTBANK_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights: Path, device: torch.device | None = None) -> YoloxObb:
    """Rebuild the model from the weights file.

    The variant, the head and the architecture are stored WITH the weights
    (`main`'s `checkpoint_payload`): loading nano weights into a tiny would
    fail with an incomprehensible shape error, and the file is the only
    place where that information cannot drift out of sync.
    """
    device = device or pick_device()
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    if payload.get("arch") == "ddgrcf":
        from testbank.models.ddgrcf import DdgrcfYoloxObb

        model = DdgrcfYoloxObb(num_classes=payload["num_classes"])
    else:
        model = YoloxObb(
            payload["variant"],
            num_classes=payload["num_classes"],
            head=HeadSpec.from_dict(payload.get("head")),
        )
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


__all__ = ["image_to_input", "load_model", "pick_device"]
