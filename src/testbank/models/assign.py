"""The anchor grid: one point per cell of every level, in image pixels.
What decoding needs to turn the head's offsets into boxes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AnchorGrid:
    """Cell centers of every level, already in image pixels."""

    #: (N, 2) -- center of each cell.
    centers: torch.Tensor
    #: (N,) -- spatial stride of the level each cell belongs to.
    strides: torch.Tensor

    def __len__(self) -> int:
        return self.centers.shape[0]


def build_anchor_grid(
    sizes: list[tuple[int, int]],
    strides: tuple[int, ...],
    device: torch.device | None = None,
) -> AnchorGrid:
    """One point per cell, at the CENTER of the cell, not its corner.

    The half pixel matters: without it every box comes out biased half a cell
    up and to the left, which at stride 32 is 16 pixels.
    """
    centers, all_strides = [], []
    for (height, width), stride in zip(sizes, strides):
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        grid = torch.stack(((xs + 0.5) * stride, (ys + 0.5) * stride), dim=-1)
        centers.append(grid.reshape(-1, 2))
        all_strides.append(
            torch.full((height * width,), float(stride), device=device)
        )
    return AnchorGrid(torch.cat(centers), torch.cat(all_strides))


__all__ = ["AnchorGrid", "build_anchor_grid"]
