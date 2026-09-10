"""Chequeo de consistencia de la politica de visibilidad.

La politica dice: un billete tapado por otro se anota solo si se ve al menos el
25%. El codigo no puede hacerla cumplir directamente -- si un billete no se
anoto, no hay nada que medir. Lo que si puede hacer es detectar la direccion
contraria: un quad ANOTADO que queda tapado por debajo del umbral, es decir, una
anotacion que segun la politica no deberia existir.

Dos avisos sobre como leer el resultado:

1. El orden de profundidad no esta anotado. Que A solape a B no dice cual esta
   encima. Un quad muy solapado puede ser perfectamente el de arriba, que no
   esta tapado en absoluto. Por eso esto es una lista de candidatos para
   revision manual y no un error fatal.

2. La oclusion se calcula contra la UNION de los demas quads, no por pares. Un
   billete tapado al 40% por un vecino y al 40% por otro queda al 20% visible e
   incumple la politica, y ningun par lo detectaria.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Polygon
from shapely.ops import unary_union

from testbank.dataio.loader import (
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
    #: Vecino que mas tapa, para dirigir la revision manual.
    dominant_index: int | None
    dominant_fraction: float

    def describe(self) -> str:
        dominant = (
            f", el que mas tapa es el #{self.dominant_index} "
            f"({self.dominant_fraction:.0%})"
            if self.dominant_index is not None
            else ""
        )
        return (
            f"{self.sample_id}: anotacion #{self.annotation_index} queda visible al "
            f"{self.visible_fraction:.1%}{dominant}"
        )


@dataclass
class VisibilityReport:
    visibility_threshold: float
    images_checked: int = 0
    annotations_checked: int = 0
    findings: list[Finding] = field(default_factory=list)
    read_warnings: list[str] = field(default_factory=list)
    #: Cota superior de incumplimientos reales, descontando el billete de encima
    #: de cada monton, cuya marca es necesariamente un falso positivo.
    upper_bound: int = 0

    @property
    def flagged_images(self) -> list[str]:
        return sorted({f.sample_id for f in self.findings})

    def summary(self) -> str:
        lines = [
            f"Politica de visibilidad: umbral {self.visibility_threshold:.0%}",
            f"Imagenes revisadas:      {self.images_checked}",
            f"Anotaciones revisadas:   {self.annotations_checked}",
            f"Anotaciones marcadas:    {len(self.findings)} "
            f"en {len(self.flagged_images)} imagenes",
            f"Incumplimientos reales:  como mucho {self.upper_bound}",
        ]
        if self.findings:
            lines.append("")
            lines.append("Candidatos para revision manual (no es un error):")
            for finding in sorted(self.findings, key=lambda f: f.visible_fraction):
                lines.append(f"  - {finding.describe()}")
            lines.append("")
            lines.append(
                "  La marca mide solapamiento geometrico, no oclusion: el orden de"
            )
            lines.append(
                "  profundidad no esta anotado. En cada monton hay un billete arriba"
            )
            lines.append(
                "  del todo que no esta tapado por nadie, asi que su marca es un falso"
            )
            lines.append(
                "  positivo seguro. De ahi la cota. Mira la visualizacion antes de"
            )
            lines.append("  tocar ninguna anotacion.")
        if self.read_warnings:
            lines.append("")
            lines.append(f"Avisos de lectura ({len(self.read_warnings)}):")
            for message in self.read_warnings[:20]:
                lines.append(f"  - {message}")
            if len(self.read_warnings) > 20:
                lines.append(f"  ... y {len(self.read_warnings) - 20} mas")
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
    """Componentes conexas por solapamiento, sobre TODOS los quads de la imagen."""
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


#: Por encima de este tamano de componente se renuncia al calculo exacto.
_EXACT_COMPONENT_LIMIT = 9


def _max_occluded_in_component(
    polygons: list[Polygon], component: set[int], visibility_threshold: float
) -> int:
    """Maximo de quads que pueden estar tapados a la vez bajo ALGUN orden.

    Un billete solo lo tapan los que estan por ENCIMA de el. Con el orden de
    profundidad fijado, el quad en la posicion k esta tapado por la union de los
    k-1 anteriores, y el de arriba del todo no lo tapa nadie. Recorrer los m!
    ordenes es innecesario: basta un DP sobre subconjuntos, donde el estado es el
    conjunto ya colocado (de arriba abajo) y la transicion anade el siguiente.

    El resultado es exacto, no una cota floja: es el mayor numero de anotaciones
    de esa componente que podrian ser incumplimientos reales simultaneamente.
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
    """Cuantas de las marcas de una imagen pueden ser incumplimientos de verdad.

    La marca por union es un sobreconjunto: si un quad no se marca es que ni
    siquiera la union de TODOS los demas lo tapa lo bastante, asi que tampoco lo
    tapan los de encima. Los incumplimientos reales estan por fuerza entre los
    marcados, y este numero dice cuantos como mucho.
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
    """`indices` traduce cada posicion de `quads` a su posicion original en el
    fichero. Hace falta cuando lo que llega ya viene filtrado: sin ello el
    informe numeraria sobre la lista filtrada y apuntaria a otra anotacion al
    abrir el fichero."""
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
    """Revisa las anotaciones que quedan VIVAS tras el filtro de area relativa.

    Las dos cosas son complementarias y por eso se devuelven las dos: el informe
    del filtro lista lo que se dejo fuera, y este revisa lo que queda, que es lo
    que de verdad vera el detector. Revisar tambien lo filtrado devolveria
    justo las franjas que el filtro acaba de descartar.

    Las razones de area son invariantes afines, asi que el aspecto no cambia
    ningun numero de aqui. Se pasa al lector solo para que el orden canonico de
    los quads sea el correcto de cara al informe.
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
            # `findings` numera sobre el fichero; `polygons` sobre la lista ya
            # filtrada. La cota se calcula sobre esta ultima, asi que hay que
            # volver a traducir.
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
