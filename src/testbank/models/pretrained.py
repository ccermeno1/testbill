"""Loading foreign weights into the own models. With explicit mapping, no magic.

Two cases, and in both what does not fit is said, not hidden:

- **Megvii's YOLOX (COCO) into the own head.** `yolox_s.pth.tar`, Apache-2.0.
  Our `CSPDarknet` + `PAFPN` is the same network as their `YOLOPAFPN`:
  measured, 354 tensors and 7,066,683 parameters in both, identical shapes.
  Only the names change: they wrap the backbone as `backbone.backbone.*` and
  call the neck modules `lateral_conv0`, `C3_p4`, `reduce_conv1`...; here they
  are `backbone.*`, `neck.lateral_c5`, `neck.p4`, `neck.lateral_c4`... The
  table below is that mapping, and it is checked shape by shape on load.

  Their head (`head.*`) is NOT loaded: it is the COCO one, 80 classes and no
  angle. What is inherited is "knowing how to see", not "knowing where the
  banknote is".

- **DDGRCF (DOTA) into the port.** Same names by construction; see
  `ddgrcf.load_pretrained`. Here it is only dispatched.
"""

from __future__ import annotations

from pathlib import Path

import torch

#: Megvii neck module -> ours. Paired by function: the 1x1 that reduces C5,
#: the C3 after concatenating with C4, etc. Verified with the shapes: the
#: eight pairs have identical tensors.
_MEGVII_NECK = {
    "lateral_conv0": "lateral_c5",
    "C3_p4": "p4",
    "reduce_conv1": "lateral_c4",
    "C3_p3": "p3",
    "bu_conv2": "down_p3",
    "C3_n3": "n4",
    "bu_conv1": "down_n4",
    "C3_n4": "n5",
}


def remap_megvii_key(key: str) -> str | None:
    """Megvii checkpoint key -> our key, or None if it is their head."""
    if key.startswith("backbone.backbone."):
        return "backbone." + key[len("backbone.backbone.") :]
    if key.startswith("backbone."):
        module, _, tail = key[len("backbone.") :].partition(".")
        if module not in _MEGVII_NECK:
            raise KeyError(
                f"unknown neck module in the checkpoint: {module!r}. "
                "Either it is not a Megvii YOLOX or its PAFPN has changed"
            )
        return f"neck.{_MEGVII_NECK[module]}.{tail}"
    return None  # head.*: the COCO head, which is not wanted


def load_megvii_yolox(model, path: str | Path) -> dict:
    """Load backbone and neck from a Megvii `yolox_*.pth.tar`.

    Returns `{"loaded": n, "skipped_head": n, "mismatched": [...]}`. It is an
    error, not a warning, for a backbone or neck tensor to remain uncovered:
    it would mean the variant does not match (loading `yolox_s` into `nano`,
    for example), and training "half pretrained" without knowing is worse
    than training from scratch.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    own = model.state_dict()

    mapped, skipped_head, mismatched = {}, 0, []
    for key, value in state.items():
        target = remap_megvii_key(key)
        if target is None:
            skipped_head += 1
            continue
        if target not in own:
            mismatched.append(f"{key} -> {target}: does not exist")
        elif own[target].shape != value.shape:
            mismatched.append(
                f"{key} -> {target}: shape {tuple(value.shape)} != {tuple(own[target].shape)}"
            )
        else:
            mapped[target] = value

    expected = [k for k in own if k.startswith(("backbone.", "neck."))]
    uncovered = [k for k in expected if k not in mapped]
    if mismatched or uncovered:
        raise RuntimeError(
            f"{Path(path).name} does not fit {model.variant!r}: "
            f"{len(mismatched)} conflicting tensors and {len(uncovered)} uncovered. "
            f"First ones: {(mismatched + uncovered)[:3]}. Check that the checkpoint "
            "variant (nano/tiny/s) is the model's"
        )
    model.load_state_dict(mapped, strict=False)
    return {"loaded": len(mapped), "skipped_head": skipped_head, "mismatched": mismatched}


def load_pretrained(model, path: str | Path) -> str:
    """Dispatch by architecture and return a note for the log."""
    from testbank.models.ddgrcf import DdgrcfYoloxObb
    from testbank.models.ddgrcf import load_pretrained as load_ddgrcf

    if isinstance(model, DdgrcfYoloxObb):
        skipped = load_ddgrcf(model, path)
        return (
            f"DOTA pretraining loaded from {Path(path).name}; "
            f"{len(skipped)} tensors skipped (class layer): {sorted(skipped)[:2]}"
        )
    report = load_megvii_yolox(model, path)
    return (
        f"COCO pretraining (Megvii) loaded from {Path(path).name}: "
        f"{report['loaded']} backbone+neck tensors; their head "
        f"({report['skipped_head']} tensors) discarded"
    )


__all__ = ["load_megvii_yolox", "load_pretrained", "remap_megvii_key"]
