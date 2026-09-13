"""Consistency check of the visibility policy.

The policy says: a banknote covered by another is annotated only if at least
25% of it is visible. The code cannot enforce it directly -- if a banknote was
not annotated, there is nothing to measure. What it can do is detect the
opposite direction: an ANNOTATED quad that is covered below the threshold,
i.e. an annotation that according to the policy should not exist.

Two warnings on how to read the result:

1. The depth order is not annotated. That A overlaps B does not say which one
   is on top. A heavily overlapped quad may perfectly well be the top one,
   which is not covered at all. That is why this is a list of candidates for
   manual review and not a fatal error.

2. Occlusion is computed against the UNION of the other quads, not pairwise.
   A banknote covered 40% by one neighbour and 40% by another is 20% visible
   and violates the policy, and no pair would detect it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Polygon
from shapely.ops import unary_union

from testbank.dataio.prepare import (
    DEFAULT_MIN_RELATIVE_AREA,
    FilterReport,
    load_samples,
)
from testbank.geometry.quad import Quad

DEFAULT_VISIBILITY_THRESHOLD = 0.25


def quad_to_polygon(quad: Quad) -> Polygon:
    return Polygon(quad.points)


@dataclass(frozen=True, slots=True)
class Finding:
    sample_id: str
    annotation_index: int
    visible_fraction: float
    #: Neighbour that covers the most, to direct the manual review.
    dominant_index: int | None
    dominant_fraction: float

    def describe(self) -> str:
        dominant = (
            f", the one covering most is #{self.dominant_index} "
            f"({self.dominant_fraction:.0%})"
            if self.dominant_index is not None
            else ""
        )
        return (
            f"{self.sample_id}: annotation #{self.annotation_index} is visible at "
            f"{self.visible_fraction:.1%}{dominant}"
        )


@dataclass
class VisibilityReport:
    visibility_threshold: float
    images_checked: int = 0
    annotations_checked: int = 0
    findings: list[Finding] = field(default_factory=list)
    read_warnings: list[str] = field(default_factory=list)
    #: Upper bound of real violations, discounting the top banknote of each
    #: pile, whose mark is necessarily a false positive.
    upper_bound: int = 0

    @property
    def flagged_images(self) -> list[str]:
        return sorted({f.sample_id for f in self.findings})

    def summary(self) -> str:
        lines = [
            f"Visibility policy: threshold {self.visibility_threshold:.0%}",
            f"Images checked:        {self.images_checked}",
            f"Annotations checked:   {self.annotations_checked}",
            (
                f"Annotations flagged:   {len(self.findings)} "
                f"in {len(self.flagged_images)} images"
            ),
            f"Real violations:       at most {self.upper_bound}",
        ]
        if self.findings:
            lines.append("")
            lines.append("Candidates for manual review (not an error):")
            for finding in sorted(self.findings, key=lambda f: f.visible_fraction):
                lines.append(f"  - {finding.describe()}")
            lines.append("")
            lines.append(
                "  The mark measures geometric overlap, not occlusion: the depth"
            )
            lines.append(
                "  order is not annotated. In every pile there is a banknote on"
            )
            lines.append(
                "  the very top that nobody covers, so its mark is a sure false"
            )
            lines.append(
                "  positive. Hence the bound. Look at the visualization before"
            )
            lines.append("  touching any annotation.")
        if self.read_warnings:
            lines.append("")
            lines.append(f"Read warnings ({len(self.read_warnings)}):")
            for message in self.read_warnings[:20]:
                lines.append(f"  - {message}")
            if len(self.read_warnings) > 20:
                lines.append(f"  ... and {len(self.read_warnings) - 20} more")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "visibility_threshold": self.visibility_threshold,
            "images_checked": self.images_checked,
            "annotations_checked": self.annotations_checked,
            "upper_bound": self.upper_bound,
            "findings": [
                {
                    "sample_id": f.sample_id,
                    "annotation_index": f.annotation_index,
                    "visible_fraction": f.visible_fraction,
                    "dominant_index": f.dominant_index,
                    "dominant_fraction": f.dominant_fraction,
                }
                for f in self.findings
            ],
            "read_warnings": self.read_warnings,
        }


def _overlap_components(polygons: list[Polygon]) -> list[set[int]]:
    """Connected components by overlap, over ALL the quads of the image."""
    parent = list(range(len(polygons)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and polygons[i].intersection(
                polygons[j]
            ).area > 0:
                parent[find(i)] = find(j)

    groups: dict[int, set[int]] = {}
    for i in range(len(polygons)):
        groups.setdefault(find(i), set()).add(i)
    return list(groups.values())


#: Above this component size the exact computation is given up.
_EXACT_COMPONENT_LIMIT = 9


def _max_occluded_in_component(
    polygons: list[Polygon], component: set[int], visibility_threshold: float
) -> int:
    """Maximum number of quads that can be covered at once under SOME order.

    A banknote is covered only by the ones ABOVE it. With the depth order
    fixed, the quad at position k is covered by the union of the k-1 previous
    ones, and the top one is covered by nobody. Walking the m! orders is
    unnecessary: a DP over subsets suffices, where the state is the set
    already placed (top to bottom) and the transition adds the next one.

    The result is exact, not a loose bound: it is the largest number of
    annotations in that component that could be real violations
    simultaneously.
    """
    order = sorted(component)
    size = len(order)
    if size > _EXACT_COMPONENT_LIMIT:
        return size - 1

    cache: dict[tuple[int, int], float] = {}

    def visible_under(position: int, mask: int) -> float:
        key = (position, mask)
        if key not in cache:
            above = [polygons[order[k]] for k in range(size) if mask >> k & 1]
            polygon = polygons[order[position]]
            covered = (
                polygon.intersection(unary_union(above)).area / polygon.area
                if above
                else 0.0
            )
            cache[key] = max(0.0, 1.0 - covered)
        return cache[key]

    best = [-1] * (1 << size)
    best[0] = 0
    for mask in range(1 << size):
        if best[mask] < 0:
            continue
        for position in range(size):
            if mask >> position & 1:
                continue
            gain = 1 if visible_under(position, mask) < visibility_threshold else 0
            nxt = mask | (1 << position)
            best[nxt] = max(best[nxt], best[mask] + gain)
    return best[(1 << size) - 1]


def max_real_violations(
    polygons: list[Polygon],
    flagged: set[int],
    *,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> int:
    """How many of the marks in an image can be real violations.

    The union mark is a superset: if a quad is not marked, not even the union
    of ALL the others covers it enough, so neither do the ones above it. Real
    violations are necessarily among the marked ones, and this number says
    how many at most.
    """
    if not flagged:
        return 0
    return sum(
        _max_occluded_in_component(polygons, component, visibility_threshold)
        for component in _overlap_components(polygons)
    )


def check_quads(
    quads: list[Quad],
    sample_id: str,
    *,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    indices: tuple[int, ...] | None = None,
) -> list[Finding]:
    """`indices` translates each position of `quads` to its original position
    in the file. Needed when what arrives is already filtered: without it the
    report would number over the filtered list and point to another
    annotation when opening the file."""
    if len(quads) < 2:
        return []

    def original(position: int) -> int:
        return indices[position] if indices is not None else position

    polygons = [quad_to_polygon(q).buffer(0) for q in quads]
    findings: list[Finding] = []

    for index, polygon in enumerate(polygons):
        area = polygon.area
        if area <= 0:
            continue
        others = [p for i, p in enumerate(polygons) if i != index]
        occluded = polygon.intersection(unary_union(others)).area / area
        visible = max(0.0, 1.0 - occluded)
        if visible >= visibility_threshold:
            continue

        dominant_index, dominant_fraction = None, 0.0
        for other_index, other in enumerate(polygons):
            if other_index == index:
                continue
            fraction = polygon.intersection(other).area / area
            if fraction > dominant_fraction:
                dominant_index, dominant_fraction = original(other_index), fraction

        findings.append(
            Finding(
                sample_id=sample_id,
                annotation_index=original(index),
                visible_fraction=visible,
                dominant_index=dominant_index,
                dominant_fraction=dominant_fraction,
            )
        )
    return findings


def check_samples(
    samples,
    *,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    sizes=None,
    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA,
) -> tuple[VisibilityReport, FilterReport]:
    """Reviews the annotations that remain ALIVE after the relative area filter.

    The two things are complementary and that is why both are returned: the
    filter report lists what was left out, and this one reviews what remains,
    which is what the detector will really see. Reviewing the filtered ones
    too would return precisely the strips the filter just discarded.

    Area ratios are affine invariants, so the aspect changes no number here.
    It is passed to the reader only so the canonical order of the quads is
    the right one for the report.
    """
    report = VisibilityReport(visibility_threshold=visibility_threshold)
    loaded, filter_report = load_samples(
        samples, sizes=sizes, min_relative_area=min_relative_area
    )
    for item in loaded:
        quads = list(item.quads)
        report.images_checked += 1
        report.annotations_checked += len(quads)
        report.read_warnings.extend(item.warnings)
        findings = check_quads(
            quads,
            item.sample_id,
            visibility_threshold=visibility_threshold,
            indices=item.kept_indices,
        )
        report.findings.extend(findings)
        if findings:
            polygons = [quad_to_polygon(q).buffer(0) for q in quads]
            # `findings` numbers over the file; `polygons` over the already
            # filtered list. The bound is computed over the latter, so it has
            # to be translated back.
            position_of = {
                original: position
                for position, original in enumerate(item.kept_indices)
            }
            report.upper_bound += max_real_violations(
                polygons,
                {position_of[f.annotation_index] for f in findings},
                visibility_threshold=visibility_threshold,
            )
    return report, filter_report
