"""Unica puerta de acceso a la asignacion de particiones.

Ningun otro modulo calcula a que particion pertenece una muestra. Todo lee de
splits/{train,valid,test}.txt, nunca del arbol de directorios, para que la
particion quede congelada y versionada aunque el directorio de datos cambie.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from testbank.data.discover import Layout, LayoutMode, Sample, detect_layout

SPLIT_NAMES = ("train", "valid", "test")
MANIFEST_NAME = "manifest.json"
FOLDS_DIRNAME = "folds"
TEST_LOG_PATH = Path("runs") / "test_evaluations.jsonl"

_HEADER = (
    "# Generado por testbank. Particion congelada: no editar a mano.\n"
    "# Un identificador por linea (nombre de fichero sin extension).\n"
)


class GroupStrategy(str, Enum):
    FILENAME_PREFIX = "filename-prefix"
    DIRECTORY = "directory"
    MANIFEST = "manifest"
    NONE = "none"


class SplitError(RuntimeError):
    """Fallo de integridad o de uso de las particiones."""


class SealedTestSetError(SplitError):
    """Se intento acceder a test sin autorizacion explicita."""


@dataclass(frozen=True, slots=True)
class GroupConfig:
    """Como se agrupan las muestras que no pueden repartirse entre particiones.

    `none` declara que cada imagen es independiente. Es una afirmacion sobre el
    mundo, no un valor por defecto comodo: si existen varias tomas del mismo
    billete fisico y caen en particiones distintas, las metricas salen infladas.
    Por eso exige confirmacion explicita.
    """

    strategy: GroupStrategy = GroupStrategy.NONE
    regex: str | None = None
    manifest_path: Path | None = None
    independence_confirmed: bool = False

    def validate(self, *, require_independence_confirmation: bool = True) -> None:
        """`require_independence_confirmation` solo aplica al CREAR la particion.

        En modo adoptar la particion ya viene decidida por el export: congelarla
        no es afirmar nada sobre la independencia de las muestras, asi que no se
        exige la confirmacion. La comprobacion de grupos repartidos si se hace
        igualmente cuando hay clave de grupo, porque ahi si se puede detectar la
        fuga aunque no la hayamos causado nosotros.
        """
        if (
            require_independence_confirmation
            and self.strategy is GroupStrategy.NONE
            and not self.independence_confirmed
        ):
            raise SplitError(
                "--group-key none afirma que cada imagen es independiente. Si hay "
                "varias tomas del mismo billete fisico repartidas entre "
                "particiones, las metricas salen infladas. Pasa "
                "--i-confirm-independence para dejar constancia de que es una "
                "decision y no un descuido"
            )
        if self.strategy is GroupStrategy.FILENAME_PREFIX and not self.regex:
            raise SplitError(
                "--group-key filename-prefix necesita --group-regex con exactamente "
                "un grupo de captura que extraiga la clave del nombre de fichero. "
                "No se adivina un separador por defecto: depende de como nombre el "
                "export las tomas del mismo billete"
            )
        if self.strategy is GroupStrategy.MANIFEST and not self.manifest_path:
            raise SplitError(
                "--group-key manifest necesita --group-manifest con un JSON "
                "{identificador: clave_de_grupo}"
            )

    def resolver(self, *, require_independence_confirmation: bool = True) -> "GroupResolver":
        self.validate(
            require_independence_confirmation=require_independence_confirmation
        )
        return GroupResolver(self)


class GroupResolver:
    def __init__(self, config: GroupConfig) -> None:
        self.config = config
        self._pattern = re.compile(config.regex) if config.regex else None
        self._table: dict[str, str] = {}
        if config.manifest_path is not None:
            raw = json.loads(Path(config.manifest_path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise SplitError(
                    f"{config.manifest_path}: se esperaba un objeto JSON "
                    "{identificador: clave_de_grupo}"
                )
            self._table = {str(k): str(v) for k, v in raw.items()}

    def key(self, sample: Sample) -> str:
        strategy = self.config.strategy
        if strategy is GroupStrategy.NONE:
            return sample.sample_id
        if strategy is GroupStrategy.DIRECTORY:
            return sample.directory_key
        if strategy is GroupStrategy.MANIFEST:
            try:
                return self._table[sample.sample_id]
            except KeyError:
                raise SplitError(
                    f"la muestra '{sample.sample_id}' no aparece en el manifiesto "
                    f"de grupos {self.config.manifest_path}"
                ) from None
        assert self._pattern is not None
        match = self._pattern.search(sample.sample_id)
        if match is None or not match.groups():
            raise SplitError(
                f"el regex de grupo {self.config.regex!r} no captura nada en "
                f"'{sample.sample_id}'"
            )
        return match.group(1)


# -- escritura --------------------------------------------------------------


def _write_split_file(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + "".join(f"{i}\n" for i in ids), encoding="utf-8")


def _read_split_file(path: Path) -> list[str]:
    if not path.is_file():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _digest(ids_by_split: dict[str, list[str]]) -> str:
    payload = json.dumps(ids_by_split, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def materialize_splits(
    data_root: str | Path,
    splits_dir: str | Path,
    *,
    groups: GroupConfig | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 20260910,
    overwrite: bool = False,
) -> dict:
    """Materializa la particion en modo adoptar o crear, segun lo que haya en disco.

    Modo adoptar: la particion del export ES la particion. No se recalcula ni se
    "mejora"; solo se congela en ficheros.
    """
    data_root = Path(data_root)
    splits_dir = Path(splits_dir)
    group_config = groups or GroupConfig()

    existing = [splits_dir / f"{name}.txt" for name in SPLIT_NAMES]
    if any(p.exists() for p in existing) and not overwrite:
        raise SplitError(
            f"ya hay una particion materializada en {splits_dir}. Se niega a "
            "sobrescribirla: eso invalidaria toda comparacion anterior. Usa "
            "extend-splits para anadir muestras, o --overwrite si de verdad "
            "quieres empezar de cero"
        )

    layout = detect_layout(data_root)
    # La confirmacion de independencia solo se exige si somos nosotros quienes
    # decidimos el reparto.
    resolver = group_config.resolver(
        require_independence_confirmation=layout.mode is LayoutMode.CREATE
    )

    if layout.mode is LayoutMode.ADOPT:
        assignment = {
            name: sorted(s.sample_id for s in layout.groups.get(name, ()))
            for name in SPLIT_NAMES
        }
    else:
        assignment = _random_assignment(
            layout.all_samples, resolver, ratios=ratios, seed=seed
        )

    _check_disjoint(assignment)
    _check_groups_disjoint(assignment, layout, resolver)

    for name in SPLIT_NAMES:
        _write_split_file(splits_dir / f"{name}.txt", assignment[name])

    manifest = {
        "created_utc": _now(),
        "mode": layout.mode.value,
        "data_root": str(data_root.resolve()),
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios)) if layout.mode is LayoutMode.CREATE else None,
        "group_strategy": group_config.strategy.value,
        "group_regex": group_config.regex,
        "group_manifest": str(group_config.manifest_path)
        if group_config.manifest_path
        else None,
        "independence_confirmed": group_config.independence_confirmed,
        "counts": {name: len(assignment[name]) for name in SPLIT_NAMES},
        "digest": _digest(assignment),
        "notes": list(layout.notes),
        "history": [{"action": "materialize", "utc": _now()}],
    }
    (splits_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _random_assignment(
    samples: tuple[Sample, ...],
    resolver: GroupResolver,
    *,
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[str]]:
    """Reparto aleatorio por grupo, con semilla fijada.

    Se reparten grupos, no ficheros: con `none` cada fichero es su propio grupo y
    el resultado es el split aleatorio por fichero.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise SplitError(f"los porcentajes deben sumar 1, suman {sum(ratios)}")

    by_group: dict[str, list[str]] = {}
    for sample in samples:
        by_group.setdefault(resolver.key(sample), []).append(sample.sample_id)

    keys = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(keys)

    total = len(keys)
    n_train = int(round(total * ratios[0]))
    n_valid = int(round(total * ratios[1]))
    n_train = min(n_train, total)
    n_valid = min(n_valid, total - n_train)
    chunks = {
        "train": keys[:n_train],
        "valid": keys[n_train : n_train + n_valid],
        "test": keys[n_train + n_valid :],
    }
    return {
        name: sorted(sid for key in chunk for sid in by_group[key])
        for name, chunk in chunks.items()
    }


def _check_disjoint(assignment: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    clashes: list[str] = []
    for name in SPLIT_NAMES:
        for sample_id in assignment[name]:
            if sample_id in seen:
                clashes.append(f"'{sample_id}' esta en '{seen[sample_id]}' y en '{name}'")
            else:
                seen[sample_id] = name
    if clashes:
        raise SplitError(
            "una muestra no puede estar en dos particiones:\n  "
            + "\n  ".join(clashes[:20])
            + (f"\n  ... y {len(clashes) - 20} mas" if len(clashes) > 20 else "")
        )


def _check_groups_disjoint(
    assignment: dict[str, list[str]], layout: Layout, resolver: GroupResolver
) -> None:
    if resolver.config.strategy is GroupStrategy.NONE:
        return
    by_id = {s.sample_id: s for s in layout.all_samples}
    group_split: dict[str, str] = {}
    clashes: list[str] = []
    for name in SPLIT_NAMES:
        for sample_id in assignment[name]:
            sample = by_id.get(sample_id)
            if sample is None:
                continue
            key = resolver.key(sample)
            previous = group_split.get(key)
            if previous is None:
                group_split[key] = name
            elif previous != name:
                clashes.append(
                    f"grupo '{key}' aparece en '{previous}' y en '{name}' "
                    f"(por ejemplo '{sample_id}')"
                )
    if clashes:
        raise SplitError(
            "un grupo no puede repartirse entre particiones:\n  "
            + "\n  ".join(sorted(set(clashes))[:20])
        )


# -- lectura ----------------------------------------------------------------


class SplitLoader:
    """Acceso de solo lectura a la particion congelada.

    Comprueba integridad al cargar. Test queda sellado: `load("test")` exige
    allow_test=True y el runner de experimentos nunca lo pasa.
    """

    def __init__(self, splits_dir: str | Path, data_root: str | Path | None = None):
        self.splits_dir = Path(splits_dir)
        manifest_path = self.splits_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise SplitError(
                f"no hay particion materializada en {self.splits_dir}. Ejecuta "
                "`testbank make-splits` primero"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.data_root = Path(data_root or self.manifest["data_root"])
        self._assignment = {
            name: _read_split_file(self.splits_dir / f"{name}.txt")
            for name in SPLIT_NAMES
        }
        self._layout = detect_layout(self.data_root)
        self._by_id = {s.sample_id: s for s in self._layout.all_samples}
        self._verify()

    def _verify(self) -> None:
        _check_disjoint(self._assignment)
        strategy = GroupStrategy(self.manifest["group_strategy"])
        if strategy is not GroupStrategy.NONE:
            config = GroupConfig(
                strategy=strategy,
                regex=self.manifest.get("group_regex"),
                manifest_path=(
                    Path(self.manifest["group_manifest"])
                    if self.manifest.get("group_manifest")
                    else None
                ),
                independence_confirmed=True,
            )
            _check_groups_disjoint(
                self._assignment,
                self._layout,
                config.resolver(require_independence_confirmation=False),
            )

        listed = {sid for ids in self._assignment.values() for sid in ids}
        missing = sorted(listed - set(self._by_id))
        if missing:
            raise SplitError(
                f"{len(missing)} muestras de la particion no estan en "
                f"{self.data_root}: {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
                + ". La particion esta congelada; si el directorio de datos "
                "cambio, arregla el directorio, no la particion"
            )

    def unassigned(self) -> tuple[str, ...]:
        listed = {sid for ids in self._assignment.values() for sid in ids}
        return tuple(sorted(set(self._by_id) - listed))

    def load(self, split: str, *, allow_test: bool = False) -> tuple[Sample, ...]:
        if split not in SPLIT_NAMES:
            raise SplitError(f"particion desconocida '{split}'; validas: {SPLIT_NAMES}")
        if split == "test" and not allow_test:
            raise SealedTestSetError(
                "test esta sellado. El runner de experimentos no puede leerlo: "
                "mirar test durante el desarrollo lo convierte en un segundo "
                "conjunto de validacion. Usa el comando `evaluate-test`, que "
                "registra cada acceso en runs/test_evaluations.jsonl"
            )
        return tuple(self._by_id[sid] for sid in self._assignment[split])

    def ids(self, split: str) -> tuple[str, ...]:
        return tuple(self._assignment[split])


# -- extension --------------------------------------------------------------


def extend_splits(
    splits_dir: str | Path,
    *,
    data_root: str | Path | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int | None = None,
) -> dict:
    """Anexa muestras nuevas sin reordenar las existentes.

    Una muestra nueva de un grupo ya existente hereda su particion sin opcion.
    """
    splits_dir = Path(splits_dir)
    manifest_path = splits_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(data_root or manifest["data_root"])

    layout = detect_layout(root)
    assignment = {
        name: _read_split_file(splits_dir / f"{name}.txt") for name in SPLIT_NAMES
    }
    known = {sid for ids in assignment.values() for sid in ids}
    new_samples = [s for s in layout.all_samples if s.sample_id not in known]
    if not new_samples:
        return {"added": {name: 0 for name in SPLIT_NAMES}, "manifest": manifest}

    strategy = GroupStrategy(manifest["group_strategy"])
    config = GroupConfig(
        strategy=strategy,
        regex=manifest.get("group_regex"),
        manifest_path=(
            Path(manifest["group_manifest"]) if manifest.get("group_manifest") else None
        ),
        independence_confirmed=True,
    )
    resolver = config.resolver(require_independence_confirmation=False)

    group_split: dict[str, str] = {}
    by_id = {s.sample_id: s for s in layout.all_samples}
    for name in SPLIT_NAMES:
        for sid in assignment[name]:
            sample = by_id.get(sid)
            if sample is not None:
                group_split.setdefault(resolver.key(sample), name)

    if layout.mode is LayoutMode.ADOPT:
        where = {
            s.sample_id: split
            for split, samples in layout.groups.items()
            for s in samples
        }
    else:
        where = {}

    inherited: list[str] = []
    fresh: dict[str, list[str]] = {}
    for sample in new_samples:
        key = resolver.key(sample)
        if key in group_split:
            assignment[group_split[key]].append(sample.sample_id)
            inherited.append(sample.sample_id)
        elif sample.sample_id in where:
            assignment[where[sample.sample_id]].append(sample.sample_id)
        else:
            fresh.setdefault(key, []).append(sample.sample_id)

    added = {name: 0 for name in SPLIT_NAMES}
    if fresh:
        rng = random.Random(seed if seed is not None else manifest["seed"] + 1)
        keys = sorted(fresh)
        rng.shuffle(keys)
        total = len(keys)
        n_train = int(round(total * ratios[0]))
        n_valid = min(int(round(total * ratios[1])), total - n_train)
        chunks = {
            "train": keys[:n_train],
            "valid": keys[n_train : n_train + n_valid],
            "test": keys[n_train + n_valid :],
        }
        for name, chunk in chunks.items():
            for key in chunk:
                assignment[name].extend(sorted(fresh[key]))
                added[name] += len(fresh[key])

    _check_disjoint(assignment)
    _check_groups_disjoint(assignment, layout, resolver)
    for name in SPLIT_NAMES:
        _write_split_file(splits_dir / f"{name}.txt", assignment[name])

    manifest["counts"] = {name: len(assignment[name]) for name in SPLIT_NAMES}
    manifest["digest"] = _digest(assignment)
    manifest.setdefault("history", []).append(
        {
            "action": "extend",
            "utc": _now(),
            "new_samples": len(new_samples),
            "inherited_by_group": len(inherited),
            "added_fresh": added,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"added": added, "inherited": len(inherited), "manifest": manifest}


# -- pliegues ---------------------------------------------------------------


def make_folds(
    splits_dir: str | Path,
    *,
    k: int = 5,
    seed: int = 20260910,
    data_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Pliegues agrupados sobre train+valid, materializados y congelados.

    Test se queda fuera, sellado, igual que siempre.
    """
    loader = SplitLoader(splits_dir, data_root)
    folds_dir = Path(splits_dir) / FOLDS_DIRNAME
    if folds_dir.exists() and any(folds_dir.iterdir()) and not overwrite:
        raise SplitError(
            f"ya hay pliegues congelados en {folds_dir}; usa --overwrite para rehacerlos"
        )

    pool = list(loader.load("train")) + list(loader.load("valid"))
    strategy = GroupStrategy(loader.manifest["group_strategy"])
    config = GroupConfig(
        strategy=strategy,
        regex=loader.manifest.get("group_regex"),
        manifest_path=(
            Path(loader.manifest["group_manifest"])
            if loader.manifest.get("group_manifest")
            else None
        ),
        independence_confirmed=True,
    )
    resolver = config.resolver(require_independence_confirmation=False)

    by_group: dict[str, list[str]] = {}
    for sample in pool:
        by_group.setdefault(resolver.key(sample), []).append(sample.sample_id)

    keys = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(keys)
    buckets: list[list[str]] = [[] for _ in range(k)]
    # Reparto por tamano descendente para que los pliegues queden equilibrados
    # aunque los grupos sean de tamanos muy distintos.
    for key in sorted(keys, key=lambda g: (-len(by_group[g]), g)):
        target = min(range(k), key=lambda i: (len(buckets[i]), i))
        buckets[target].extend(sorted(by_group[key]))

    folds_dir.mkdir(parents=True, exist_ok=True)
    for index, bucket in enumerate(buckets):
        _write_split_file(folds_dir / f"fold_{index}.txt", sorted(bucket))

    summary = {
        "k": k,
        "seed": seed,
        "group_strategy": strategy.value,
        "pool_size": len(pool),
        "fold_sizes": [len(b) for b in buckets],
        "created_utc": _now(),
    }
    (folds_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def record_test_access(reason: str, run_name: str, log_path: Path | None = None) -> None:
    """Deja constancia de cada acceso al conjunto sellado."""
    path = Path(log_path or TEST_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"utc": _now(), "run": run_name, "reason": reason}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
