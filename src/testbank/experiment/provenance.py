"""Where a run came from: code, seeds and environment.

Without this a row of the comparison table is not a result, it is a rumour.
The rule of the whole module is that **the absence of a datum is recorded as
absence**, never as a default value that looks like information.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Single seed everything else hangs from. Recorded in every run.
DEFAULT_SEED = 20260910


@dataclass(frozen=True, slots=True)
class GitInfo:
    """State of the code tree.

    `available = False` is not a minor detail: it means the run **is not
    reproducible**, because there is no way to recover the code that produced
    it. `compare` lists it apart from production fitness: not being able to
    repeat a number and not being able to deploy it are different problems.
    """

    available: bool
    commit: str | None = None
    dirty: bool | None = None
    branch: str | None = None
    reason: str | None = None

    def describe(self) -> str:
        if not self.available:
            return f"no git ({self.reason})"
        state = "dirty" if self.dirty else "clean"
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
                "the project is not in a git repository, so this run cannot "
                "be reproduced: there is no way to recover the code that "
                "generated it. Run `git init` and make a commit."
            ),
        )
    commit = _git(["rev-parse", "HEAD"], repo)
    if commit is None:
        return GitInfo(
            available=False,
            reason="git repository without any commit yet",
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
    """Seeds actually applied, not the ones requested."""

    base: int
    python: int
    numpy: int
    torch: int | None = None
    #: `PYTHONHASHSEED` only takes effect if it was set BEFORE the process
    #: started. Setting it here would do nothing, so what is there is recorded.
    pythonhashseed: str | None = None
    cudnn_deterministic: bool | None = None


def seed_everything(seed: int = DEFAULT_SEED) -> Seeds:
    """Fixes the seeds and returns what was really fixed.

    torch is optional (`torch` group of the pyproject). If it is not
    installed, its seed is recorded as None instead of pretending it was set.
    """
    random.seed(seed)

    numpy_seed = seed
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a base dependency
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
        except Exception:  # noqa: BLE001 - the torch build decides what it supports
            # A torch without deterministic algorithms must not bring the run
            # down: `cudnn_deterministic=False` is recorded and it goes on,
            # which is useful information instead of a failure.
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


#: What can move a number between runs. It is not a `pip freeze`: a long list
#: becomes noise and nobody reads it.
#:
#: `opencv-python` is here even though the project asks for the `headless`
#: variant: ultralytics drags in the normal one, both occupy the same `cv2`
#: namespace and whichever was installed last wins. Recording both makes it
#: visible in every run whether that collision was present.
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
        """Reproducible requires recoverable code AND a clean tree."""
        return self.git.available and self.git.dirty is False

    def blockers(self) -> list[str]:
        """Reasons why this run is not valid as a stable reference."""
        if not self.git.available:
            return [f"no code provenance: {self.git.reason}"]
        if self.git.dirty:
            dirty = (
                "the working tree had uncommitted changes, so commit "
                f"{self.git.commit[:12]} does not describe the code that ran"
            )
            return [dirty]
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
