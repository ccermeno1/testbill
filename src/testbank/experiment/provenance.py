"""De donde salio una ejecucion: codigo, semillas y entorno.

Sin esto una fila de la tabla comparativa no es un resultado, es un rumor. La
regla de todo el modulo es que **la ausencia de un dato se registra como
ausencia**, nunca como un valor por defecto que parezca informacion.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Semilla unica de la que cuelga todo lo demas. Registrada en cada ejecucion.
DEFAULT_SEED = 20260910


@dataclass(frozen=True, slots=True)
class GitInfo:
    """Estado del arbol de codigo.

    `available = False` no es un detalle menor: significa que la ejecucion **no
    es reproducible**, porque no hay forma de recuperar el codigo que la produjo.
    `compare` lo lista aparte de la aptitud para produccion: no poder repetir un
    numero y no poder desplegarlo son problemas distintos.
    """

    available: bool
    commit: str | None = None
    dirty: bool | None = None
    branch: str | None = None
    reason: str | None = None

    def describe(self) -> str:
        if not self.available:
            return f"sin git ({self.reason})"
        state = "sucio" if self.dirty else "limpio"
        return f"{self.commit[:12]} ({state})"


def _git(args: list[str], repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def collect_git(repo: Path | None = None) -> GitInfo:
    repo = Path(repo or Path.cwd())
    inside = _git(["rev-parse", "--is-inside-work-tree"], repo)
    if inside != "true":
        return GitInfo(
            available=False,
            reason=(
                "el proyecto no esta en un repositorio git, asi que esta "
                "ejecucion no se puede reproducir: no hay forma de recuperar el "
                "codigo que la genero. Ejecuta `git init` y haz un commit."
            ),
        )
    commit = _git(["rev-parse", "HEAD"], repo)
    if commit is None:
        return GitInfo(
            available=False,
            reason="repositorio git sin ningun commit todavia",
        )
    status = _git(["status", "--porcelain"], repo)
    return GitInfo(
        available=True,
        commit=commit,
        dirty=bool(status),
        branch=_git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
    )


@dataclass(frozen=True, slots=True)
class Seeds:
    """Semillas efectivamente aplicadas, no las que se pidieron."""

    base: int
    python: int
    numpy: int
    torch: int | None = None
    #: `PYTHONHASHSEED` solo tiene efecto si estaba puesta ANTES de arrancar el
    #: proceso. Fijarla aqui no haria nada, asi que se registra lo que hay.
    pythonhashseed: str | None = None
    cudnn_deterministic: bool | None = None


def seed_everything(seed: int = DEFAULT_SEED) -> Seeds:
    """Fija las semillas y devuelve lo que de verdad quedo fijado.

    torch es opcional (grupo `torch` del pyproject). Si no esta instalado, su
    semilla se registra como None en lugar de fingir que se fijo.
    """
    random.seed(seed)

    numpy_seed = seed
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy es dependencia base
        numpy_seed = seed

    torch_seed: int | None = None
    cudnn: bool | None = None
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch_seed = seed
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            cudnn = True
        except Exception:  # noqa: BLE001 - la build de torch decide que soporta
            # Un torch sin algoritmos deterministas no debe tumbar la ejecucion:
            # se registra `cudnn_deterministic=False` y se sigue, que es
            # informacion util en vez de un fallo.
            cudnn = False

    return Seeds(
        base=seed,
        python=seed,
        numpy=numpy_seed,
        torch=torch_seed,
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        cudnn_deterministic=cudnn,
    )


@dataclass(frozen=True, slots=True)
class Environment:
    python: str
    platform: str
    packages: dict[str, str] = field(default_factory=dict)


#: Lo que puede mover un numero entre ejecuciones. No es un `pip freeze`: una
#: lista larga se vuelve ruido y nadie la lee.
#:
#: `opencv-python` esta aqui aunque el proyecto pida la variante `headless`:
#: ultralytics arrastra la normal, las dos ocupan el mismo espacio de nombres
#: `cv2` y gana la que se instalara ultima. Registrar las dos hace visible en
#: cada ejecucion si esa colision estaba presente.
_TRACKED = (
    "numpy",
    "shapely",
    "opencv-python",
    "opencv-python-headless",
    "torch",
    "ultralytics",
)


def collect_environment() -> Environment:
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str] = {}
    for name in _TRACKED:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    return Environment(
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages=packages,
    )


@dataclass(frozen=True, slots=True)
class Provenance:
    git: GitInfo
    seeds: Seeds
    environment: Environment

    @property
    def reproducible(self) -> bool:
        """Reproducible pide codigo recuperable Y arbol limpio."""
        return self.git.available and self.git.dirty is False

    def blockers(self) -> list[str]:
        """Motivos por los que esta ejecucion no vale como referencia estable."""
        if not self.git.available:
            return [f"sin procedencia de codigo: {self.git.reason}"]
        if self.git.dirty:
            sucio = (
                "el arbol de trabajo tenia cambios sin commitear, asi que el "
                f"commit {self.git.commit[:12]} no describe el codigo ejecutado"
            )
            return [sucio]
        return []

    def to_dict(self) -> dict:
        return {
            "git": asdict(self.git),
            "seeds": asdict(self.seeds),
            "environment": asdict(self.environment),
            "reproducible": self.reproducible,
            "blockers": self.blockers(),
        }


def collect(seed: int = DEFAULT_SEED, repo: Path | None = None) -> Provenance:
    return Provenance(
        git=collect_git(repo),
        seeds=seed_everything(seed),
        environment=collect_environment(),
    )
