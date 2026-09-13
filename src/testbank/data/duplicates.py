"""Near-duplicate detection and group manifest.

The problem, measured on the current export
-------------------------------------------
27 pairs of nearly identical images cross splits (measured in color; see below
why not in grayscale), with consecutive indices: `Multiple_Euro_090` in test
and `_091` in train. They are the same shot spread across splits, so
validation and test come out optimistic: 10 of the 50 test images have their
twin in train or valid.

Why this does NOT re-split on its own
-------------------------------------
The specification is explicit: in adopt mode the export decides the split and
**it is neither recomputed nor "improved"**. So this does not touch
`splits/{train,valid,test}.txt`. It does two things:

1. **Reports** the leak that already exists, so the numbers are read knowing it.
2. **Writes a group manifest** for whoever DOES split: `make-splits
   --repartition`, the explicit, opt-in exception to the rule above (see
   `splits.py`).

Why a manifest and not a new strategy
-------------------------------------
`GroupResolver.key()` is per sample and stateless. Putting the reading of 500
images in there would make it expensive and unpredictable, and the grouping
would stop being auditable. Writing a JSON `{identifier: group}` and passing
it with the existing `--group-key manifest` reuses the mechanism, freezes the
result and allows reviewing it by hand before trusting it.

How they are compared
---------------------
COLOR, 16x16, zero mean and unit norm; correlation by dot product. No
perceptual-hash library, to avoid adding a dependency for four lines of numpy,
and because a correlation is easier to reason about than a hash: the threshold
can be raised or lowered by looking at the pairs that fall near it.

Why in color and not in grayscale, which came first
---------------------------------------------------
In grayscale, 18 of the 93 "nearly identical" pairs were banknotes of a
DIFFERENT value: a 50 with a 500, a 20 with a 200. Looked at, they are stock
photos with the same framing -- white background and the image bank's banner in
the same place -- and a 16x16 grayscale thumbnail sees "white with a dark bar"
in both. Resolution does not fix it (16, 32 and 64 give the same); color does,
because a 50 is orange and a 500 is purple:

    pair                         gray    color
    100_035 ~ 100_037 (real)     0.988   0.979
    050_232 ~ 500_437 (false)    0.948   0.519
    020_284 ~ 200_096 (false)    0.896   0.455

It came to light when stratifying the split by banknote type: a group of
"duplicates" crossing types is impossible by definition, and the split refused.
Consequence: the contamination figure this project reported for a while (42
pairs, 36% of test) was INFLATED by those false positives: in color it is 27
pairs and 20% of test.

Grouping is by connected components (union-find), not by pairs: if A looks
like B and B like C, all three go to the same group even if A and C do not
look alike. Splitting them would leave the leak half open.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

#: Correlation above which two images count as the same shot. 0.90 comes from
#: looking at the real pairs IN COLOR: above 0.95 there are 48 pairs and none
#: crosses banknote type; between 0.90 and 0.95 sit the consecutive shots
#: (005_111 ~ 005_112, 010_360 ~ 010_361) with a single type crossing; below
#: 0.90 identical framings with different banknotes start to dominate. It is
#: deliberately on the low side: for SPLITTING, over-grouping is cheap and
#: under-grouping is the leak we want to avoid.
DEFAULT_THRESHOLD = 0.90

#: Thumbnail side. 16x16 is enough to tell shots apart and is immune to the
#: JPEG compression differences that would trip an exact hash.
THUMBNAIL = 16


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    a: str
    b: str
    correlation: float
    split_a: str = ""
    split_b: str = ""

    @property
    def crosses_splits(self) -> bool:
        return bool(self.split_a) and bool(self.split_b) and self.split_a != self.split_b

    def describe(self) -> str:
        where = (
            f"  {self.split_a} <-> {self.split_b}"
            if self.split_a or self.split_b
            else ""
        )
        return f"{self.correlation:.3f}  {self.a} ~ {self.b}{where}"


@dataclass
class DuplicateReport:
    threshold: float
    n_samples: int
    pairs: list[DuplicatePair] = field(default_factory=list)
    #: identifier -> group key, with the connected components already resolved.
    groups: dict[str, str] = field(default_factory=dict)

    @property
    def crossing(self) -> list[DuplicatePair]:
        return [p for p in self.pairs if p.crosses_splits]

    @property
    def affected(self) -> set[str]:
        return {s for p in self.crossing for s in (p.a, p.b)}

    @property
    def n_groups(self) -> int:
        return len(set(self.groups.values()))

    def touching(self, split: str) -> list[DuplicatePair]:
        """Crossing pairs that have `split` on one side."""
        return [p for p in self.crossing if split in (p.split_a, p.split_b)]

    def affected_in(self, split: str) -> set[str]:
        """Images OF `split` that have a near-duplicate in another split."""
        out = set()
        for pair in self.touching(split):
            if pair.split_a == split:
                out.add(pair.a)
            if pair.split_b == split:
                out.add(pair.b)
        return out

    def summary_lines(self, sizes: dict[str, int] | None = None) -> list[str]:
        share = 100 * len(self.affected) / self.n_samples if self.n_samples else 0.0
        lines = [
            f"Near-duplicates: correlation threshold {self.threshold:.2f}",
            f"Images analysed:      {self.n_samples}",
            f"Pairs detected:       {len(self.pairs)}",
            f"Pairs that CROSS:     {len(self.crossing)}",
            f"Images involved:      {len(self.affected)} ({share:.1f}%)",
            f"Resulting groups:     {self.n_groups}",
        ]
        # The per-split breakdown ALWAYS goes in, and `test` first. Without it
        # one has to count by hand over a truncated list, and counting by hand
        # over a truncated list is exactly how a wrong number slips into a
        # report: it happened, and that is why this is here.
        for split in ("test", "valid", "train"):
            affected = self.affected_in(split)
            if not self.touching(split):
                continue
            total = (sizes or {}).get(split)
            share_of_split = (
                f" of {total} ({100 * len(affected) / total:.0f}%)" if total else ""
            )
            mark = "  <- the SEALED set" if split == "test" else ""
            lines.append(
                f"  {split:5s}: {len(self.touching(split))} pairs, "
                f"{len(affected)} images{share_of_split}{mark}"
            )
        return lines

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "n_pairs": len(self.pairs),
            "n_crossing": len(self.crossing),
            "n_affected": len(self.affected),
            "by_split": {
                split: {
                    "pairs": len(self.touching(split)),
                    "images": len(self.affected_in(split)),
                }
                for split in ("train", "valid", "test")
            },
            "n_groups": self.n_groups,
            "note": (
                "Does not re-split: in adopt mode the export decides the "
                "split. This reports the leak and provides a group manifest "
                "for `make-splits --repartition`, which does split."
            ),
            "crossing_pairs": [
                {
                    "a": p.a,
                    "b": p.b,
                    "correlation": round(p.correlation, 6),
                    "split_a": p.split_a,
                    "split_b": p.split_b,
                }
                for p in sorted(self.crossing, key=lambda p: -p.correlation)
            ],
        }


def signature(path: Path) -> np.ndarray:
    """COLOR thumbnail, centered and normalized. Ready for a dot product.

    In color and not grayscale: see the module. The mean is subtracted over the
    three channels together, not per channel, so that two photos with the same
    framing and a different banknote do not become alike again once their
    average color is removed.
    """
    with Image.open(path) as image:
        thumb = image.convert("RGB").resize((THUMBNAIL, THUMBNAIL), Image.BILINEAR)
    values = np.asarray(thumb, dtype=np.float64).ravel()
    values -= values.mean()
    norm = np.linalg.norm(values)
    # A single-tone image has zero norm: it resembles nothing by correlation,
    # so it is left as zeros instead of dividing by zero.
    return values / norm if norm > 0 else values


def _union_find(n: int, edges) -> list[int]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def find_duplicates(
    samples,
    *,
    split_of: dict[str, str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> DuplicateReport:
    """Group by connected components of "nearly identical"."""
    samples = list(samples)
    split_of = split_of or {}
    if not samples:
        return DuplicateReport(threshold=threshold, n_samples=0)

    matrix = np.stack([signature(s.image_path) for s in samples])
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -1.0)

    pairs: list[DuplicatePair] = []
    edges: list[tuple[int, int]] = []
    for i, j in zip(*np.where(np.triu(similarity, 1) > threshold)):
        i, j = int(i), int(j)
        edges.append((i, j))
        pairs.append(
            DuplicatePair(
                a=samples[i].sample_id,
                b=samples[j].sample_id,
                correlation=float(similarity[i, j]),
                split_a=split_of.get(samples[i].sample_id, ""),
                split_b=split_of.get(samples[j].sample_id, ""),
            )
        )

    roots = _union_find(len(samples), edges)
    members: dict[int, list[str]] = defaultdict(list)
    for index, root in enumerate(roots):
        members[root].append(samples[index].sample_id)

    # The group key is the smallest identifier in the component: stable
    # against input order, so the manifest is reproducible.
    groups = {
        sample_id: min(ids)
        for ids in members.values()
        for sample_id in ids
    }

    return DuplicateReport(
        threshold=threshold,
        n_samples=len(samples),
        pairs=sorted(pairs, key=lambda p: -p.correlation),
        groups=groups,
    )


def write_manifest(report: DuplicateReport, path: str | Path) -> Path:
    """JSON `{identifier: group_key}`, the `--group-manifest` format."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(sorted(report.groups.items())), indent=2) + "\n",
        encoding="utf-8",
    )
    return path
