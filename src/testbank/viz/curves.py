"""Training curves from `training.json`: losses per epoch and validation metrics.

Two panels, deliberately plain: the left one says whether the loop learns,
the right one says where it stops improving on `valid` and which epoch won.
It reads the file the loop rewrites after every epoch, so it works on a run
that is still training or one that died halfway.

`matplotlib` is imported lazily: it lives in the `dev` group and nothing in
production needs it.
"""

from __future__ import annotations

import json
from pathlib import Path

from testbank.models.train import HISTORY_FILE

LOSS_TERMS = ("total", "box", "angle", "objectness", "classes", "dfl", "l1")


def find_history(run_directory: str | Path) -> Path:
    """`training.json` of a run: the runner puts it under `_train/`, `fit`
    puts it wherever it was told to write."""
    run_directory = Path(run_directory)
    for candidate in (run_directory / "_train" / HISTORY_FILE, run_directory / HISTORY_FILE):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{run_directory}: no {HISTORY_FILE}. Only the own candidates write it "
        "(Ultralytics and RTMDet-R keep their own logs)"
    )


def plot_training(history: dict, output: str | Path, *, title: str = "") -> Path:
    """Renders the two panels to `output` (PNG). Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history.get("epochs", [])
    validation = history.get("validation", [])
    best = history.get("best")
    if not epochs:
        raise ValueError("the history has no epochs yet")

    x = [e["epoch"] + 1 for e in epochs]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))

    for term in LOSS_TERMS:
        values = [e.get(term, 0.0) for e in epochs]
        if any(values):
            left.plot(x, values, label=term, linewidth=2 if term == "total" else 1)
    left.set_xlabel("epoch")
    left.set_ylabel("loss")
    left.set_title("training losses")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    if validation:
        vx = [v["epoch"] + 1 for v in validation]
        for metric in sorted(k for k in validation[0] if k != "epoch"):
            right.plot(vx, [v[metric] for v in validation], marker="o", label=metric)
        if best is not None:
            right.axvline(best["epoch"] + 1, color="gray", linestyle="--", linewidth=1)
            right.annotate(
                f"best.pt (epoch {best['epoch'] + 1})",
                xy=(best["epoch"] + 1, 0.02),
                xycoords=("data", "axes fraction"),
                fontsize=8,
                rotation=90,
                va="bottom",
                ha="right",
            )
        right.set_ylim(0.0, 1.02)
        right.legend(fontsize=8)
    else:
        right.text(0.5, 0.5, "no validation during training", ha="center", va="center")
    right.set_xlabel("epoch")
    right.set_title("validation")
    right.grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output


def plot_run(run_directory: str | Path, output: str | Path | None = None) -> Path:
    run_directory = Path(run_directory)
    history = json.loads(find_history(run_directory).read_text(encoding="utf-8"))
    target = Path(output) if output else run_directory / "viz" / "training.png"
    return plot_training(history, target, title=run_directory.name)


__all__ = ["LOSS_TERMS", "find_history", "plot_run", "plot_training"]
