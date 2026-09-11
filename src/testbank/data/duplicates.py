"""Deteccion de casi-duplicados y manifiesto de grupos.

El problema, medido sobre el export actual
------------------------------------------
42 pares de imagenes casi identicas cruzan particiones -- el 14.7% del dataset --
con correlaciones de hasta 0.999 e indices consecutivos: `Multiple_Euro_090` en
test y `_091` en train. Son la misma toma repartida entre particiones, asi que
validacion y test salen optimistas.

Por que esto NO re-particiona
-----------------------------
La especificacion es explicita: en modo adoptar, la particion la decide el export
y **no se recalcula ni se "mejora"**. Asi que esto no toca `splits/{train,valid,
test}.txt`. Hace dos cosas:

1. **Informa** de la fuga que ya existe, para que los numeros se lean sabiendola.
2. **Escribe un manifiesto de grupos** que si sirve para lo que SI generamos
   nosotros: los pliegues de validacion cruzada. Un pliegue con la misma toma a
   los dos lados no mide generalizacion, y ese si esta en nuestra mano.

Por que un manifiesto y no una estrategia nueva
-----------------------------------------------
`GroupResolver.key()` es por muestra y sin estado. Meter ahi la lectura de 500
imagenes lo convertiria en algo caro e impredecible, y la agrupacion dejaria de
ser auditable. Escribir un JSON `{identificador: grupo}` y pasarlo con el
`--group-key manifest` que ya existe reutiliza el mecanismo, congela el
resultado y permite revisarlo a mano antes de fiarse.

Como se comparan
----------------
Gris, 16x16, media cero y norma uno; correlacion por producto escalar. No se usa
una libreria de hashes perceptuales para no anadir dependencia por algo que son
cuatro lineas de numpy, y porque una correlacion se razona mejor que un hash: el
umbral se puede subir o bajar mirando los pares que caen cerca.

La agrupacion es por componentes conexas (union-find), no por pares: si A se
parece a B y B a C, los tres van al mismo grupo aunque A y C no se parezcan.
Repartirlos seria dejar la fuga a medias.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

#: Correlacion por encima de la cual dos imagenes se consideran la misma toma.
#: 0.92 sale de mirar los pares reales: por encima estan las tomas consecutivas
#: del mismo billete, por debajo empiezan los billetes distintos del mismo valor.
DEFAULT_THRESHOLD = 0.92

#: Lado de la miniatura. 16x16 basta para distinguir tomas y es inmune a
#: diferencias de compresion JPEG que dispararian un hash exacto.
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
    #: identificador -> clave de grupo, con las componentes conexas ya resueltas.
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
        """Pares que cruzan y tienen `split` a un lado."""
        return [p for p in self.crossing if split in (p.split_a, p.split_b)]

    def affected_in(self, split: str) -> set[str]:
        """Imagenes DE `split` que tienen un casi-duplicado en otra particion."""
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
            f"Casi-duplicados: umbral de correlacion {self.threshold:.2f}",
            f"Imagenes analizadas:     {self.n_samples}",
            f"Pares detectados:        {len(self.pairs)}",
            f"Pares que CRUZAN:        {len(self.crossing)}",
            f"Imagenes implicadas:     {len(self.affected)} ({share:.1f}%)",
            f"Grupos resultantes:      {self.n_groups}",
        ]
        # El desglose por particion va SIEMPRE, y `test` el primero. Sin el, hay
        # que contar a mano sobre una lista truncada, y contar a mano sobre una
        # lista truncada es exactamente como se cuela un numero equivocado en un
        # informe: paso, y por eso esta aqui.
        for split in ("test", "valid", "train"):
            afectadas = self.affected_in(split)
            if not self.touching(split):
                continue
            total = (sizes or {}).get(split)
            proporcion = f" de {total} ({100*len(afectadas)/total:.0f}%)" if total else ""
            marca = "  <- el conjunto SELLADO" if split == "test" else ""
            lines.append(
                f"  {split:5s}: {len(self.touching(split))} pares, "
                f"{len(afectadas)} imagenes{proporcion}{marca}"
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
                "No re-particiona: en modo adoptar la particion la decide el "
                "export. Esto informa de la fuga y da un manifiesto de grupos "
                "para los pliegues de validacion cruzada, que si generamos "
                "nosotros."
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
    """Miniatura en gris, centrada y normalizada. Lista para producto escalar."""
    with Image.open(path) as image:
        thumb = image.convert("L").resize((THUMBNAIL, THUMBNAIL), Image.BILINEAR)
    values = np.asarray(thumb, dtype=np.float64).ravel()
    values -= values.mean()
    norm = np.linalg.norm(values)
    # Una imagen de un solo tono da norma cero: no se parece a nada por
    # correlacion, asi que se deja en ceros en vez de dividir por cero.
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
    """Agrupa por componentes conexas de "casi identicas"."""
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

    # La clave del grupo es el identificador menor de la componente: estable
    # frente al orden de entrada, asi que el manifiesto es reproducible.
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
    """JSON `{identificador: clave_de_grupo}`, el formato de `--group-manifest`."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(sorted(report.groups.items())), indent=2) + "\n",
        encoding="utf-8",
    )
    return path
