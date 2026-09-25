"""Rotated NMS and post-processing helpers, plain PyTorch (CPU / CUDA / MPS)."""
from __future__ import annotations

import torch

from .boxes import rotated_iou

__all__ = ["rotated_nms", "batched_postprocess", "Detections"]


class Detections(dict):
    """Detections for one image: rboxes (N, 5), polys (N, 8), scores (N,), labels (N,)."""

    @property
    def rboxes(self) -> torch.Tensor:
        return self["rboxes"]

    @property
    def polys(self) -> torch.Tensor:
        return self["polys"]

    @property
    def scores(self) -> torch.Tensor:
        return self["scores"]

    @property
    def labels(self) -> torch.Tensor:
        return self["labels"]

    def __len__(self) -> int:
        return int(self["scores"].numel())


def rotated_nms(rboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Greedy NMS over rotated boxes. Returns the kept indices, highest score first."""
    if rboxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=rboxes.device)
    order = torch.argsort(scores, descending=True)
    keep: list[int] = []
    # the loop runs over survivors, not over every box: with nms_top_k capped (2000 by
    # default) that is few iterations, each one a vectorised call
    while order.numel() > 0:
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break
        ious = rotated_iou(rboxes[i].unsqueeze(0), rboxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return torch.as_tensor(keep, dtype=torch.long, device=rboxes.device)


def batched_postprocess(
    scores: torch.Tensor,
    rboxes: torch.Tensor,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.1,
    nms_top_k: int = 2000,      # candidates before NMS (nms_pre)
    keep_top_k: int = 2000,     # detections kept after NMS (max_per_img)
    scale_factor: torch.Tensor | None = None,
) -> list[Detections]:
    """Score threshold plus rotated NMS, per class and per image.

    scores: (B, C, L), sigmoid already applied. rboxes: (B, L, 5) in INPUT pixels.
    scale_factor: (B, 2) holding (scale_y, scale_x), to map back to the original image.
    Leave it as None to keep input-pixel coordinates.
    """
    from .boxes import rbox2poly

    batch: list[Detections] = []
    b, c, _ = scores.shape
    for bi in range(b):
        boxes_i = rboxes[bi]
        keep_rboxes, keep_scores, keep_labels = [], [], []
        for ci in range(c):
            sc = scores[bi, ci]
            mask = sc >= score_threshold
            if not bool(mask.any()):
                continue
            sc_sel = sc[mask]
            bx_sel = boxes_i[mask]
            if nms_top_k > 0 and sc_sel.numel() > nms_top_k:
                top = torch.topk(sc_sel, nms_top_k).indices
                sc_sel, bx_sel = sc_sel[top], bx_sel[top]
            idx = rotated_nms(bx_sel, sc_sel, nms_threshold)
            keep_rboxes.append(bx_sel[idx])
            keep_scores.append(sc_sel[idx])
            keep_labels.append(torch.full((idx.numel(),), ci, dtype=torch.long, device=sc.device))

        if keep_scores:
            rb = torch.cat(keep_rboxes)
            sc = torch.cat(keep_scores)
            lb = torch.cat(keep_labels)
            order = torch.argsort(sc, descending=True)
            if keep_top_k > 0:
                order = order[:keep_top_k]
            rb, sc, lb = rb[order], sc[order], lb[order]
        else:
            rb = rboxes.new_zeros((0, 5))
            sc = rboxes.new_zeros((0,))
            lb = torch.zeros(0, dtype=torch.long, device=rboxes.device)

        polys = rbox2poly(rb)
        if scale_factor is not None and rb.numel():
            sy, sx = scale_factor[bi][0], scale_factor[bi][1]
            div = torch.stack([sx, sy, sx, sy, sx, sy, sx, sy]).to(polys.dtype)
            polys = polys / div
            rb = torch.cat(
                [rb[:, 0:1] / sx, rb[:, 1:2] / sy, rb[:, 2:3] / sx, rb[:, 3:4] / sy, rb[:, 4:5]], dim=-1
            )
        batch.append(Detections(rboxes=rb, polys=polys, scores=sc, labels=lb))
    return batch
