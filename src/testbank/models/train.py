"""Training loop of the own candidate.

Deliberately plain. No mosaic, no mixup, no EMA, no multi-scale. All of that
helps on COCO with 118,000 images; here there are 351, and every added trick
is one more hyperparameter to tune blindly on a validation set of 101 images
where the differences fall within the noise.

The baseline has to be interpretable before it is good. If it needs to go up,
ONE thing is added and measured -- which is what the comparison table exists
for.

Determinism
-----------
Seed fixed and logged, `DataLoader` without workers and with its own
generator. With `num_workers > 0` the batch order depends on the operating
system's scheduling and two runs with the same seed stop matching. With 351
images of 416x416 loading is not the bottleneck.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from testbank.config import Config
from testbank.experiment.provenance import seed_everything
from testbank.models.data import BanknoteDataset, Batch, collate
from testbank.models.losses import architecture_of, build_model, losses_for_image
from testbank.models.yolox_obb import HeadSpec, YoloxObb

#: Fraction of training spent raising the learning rate from almost zero.
#: Without warmup, the first iterations with the freshly initialized head give
#: huge gradients that destabilize the BatchNorm.
WARMUP_FRACTION = 0.05


@dataclass
class TrainingHistory:
    """What happened in each epoch. Written to `training.json` after EVERY
    epoch, so a run that dies at hour three still leaves its curve."""

    epochs: list[dict] = field(default_factory=list)
    #: Validation metrics at the epochs where `valid` was evaluated.
    validation: list[dict] = field(default_factory=list)
    #: The epoch that produced `best.pt` and its selection metric.
    best: dict | None = None
    #: Things that happened once and are worth leaving written: e.g. which
    #: pretraining was loaded and which tensors were skipped.
    notes: list[str] = field(default_factory=list)

    def record(self, epoch: int, terms: dict, learning_rate: float) -> None:
        self.epochs.append({"epoch": epoch, "lr": learning_rate, **terms})

    def to_list(self) -> list[dict]:
        return list(self.epochs)

    def to_dict(self) -> dict:
        return {
            "epochs": list(self.epochs),
            "validation": list(self.validation),
            "best": self.best,
            "notes": list(self.notes),
        }

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


HISTORY_FILE = "training.json"

#: Given the model and the epoch, returns the validation metrics as plain
#: numbers (`{"map50": ..., "coverage_p5": ...}`). The adapter builds it: the
#: loop must not know how a candidate predicts.
Validator = Callable[["torch.nn.Module", int], dict]


def _format_terms(terms: dict) -> str:
    keys = ("total", "box", "angle", "objectness", "classes", "dfl", "l1")
    return "  ".join(f"{k} {terms[k]:.4f}" for k in keys if terms.get(k))


def learning_rate_at(step: int, total: int, base: float) -> float:
    """Linear warmup and then cosine down to almost zero."""
    warmup = max(1, int(total * WARMUP_FRACTION))
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def _batch_targets(batch: Batch, index: int, device: torch.device):
    return (
        batch.boxes[index].to(device),
        batch.classes[index].to(device),
    )


def train_one_epoch(
    model: YoloxObb,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Config,
    *,
    device: torch.device,
    step: int,
    total_steps: int,
    base_lr: float,
    epoch: int = 0,
) -> tuple[dict, int]:
    """One epoch. Returns the averaged losses and the step reached.

    `epoch` is passed because one recipe (the fork's) changes shape in the
    last epochs: it switches on an L1. The loss has to know which one it is in.
    """
    model.train()
    totals: dict[str, float] = {}
    batches = 0

    for batch in loader:
        learning_rate = learning_rate_at(step, total_steps, base_lr)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        images = batch.images.to(device)
        outputs = model(images)

        # Assignment is PER IMAGE: mixing the boxes of the whole batch would let
        # a cell of image 3 be assigned to a banknote of image 1.
        loss = torch.zeros((), device=device)
        accumulated: dict[str, float] = {}
        for index in range(len(batch)):
            single = [
                type(o)(
                    distances=o.distances[index : index + 1],
                    angle=o.angle[index : index + 1],
                    objectness=(
                        o.objectness[index : index + 1]
                        if o.objectness is not None
                        else None
                    ),
                    classes=o.classes[index : index + 1],
                    stride=o.stride,
                    distribution=(
                        o.distribution[index : index + 1]
                        if o.distribution is not None
                        else None
                    ),
                    regression=o.regression,
                    angle_mode=o.angle_mode,
                )
                for o in outputs
            ]
            boxes, classes = _batch_targets(batch, index, device)
            terms = losses_for_image(
                single,
                boxes,
                classes,
                config,
                device,
                epoch=epoch,
                total_epochs=config.detector.epochs,
            )
            loss = loss + terms.total
            for key, value in terms.to_dict().items():
                accumulated[key] = accumulated.get(key, 0.0) + value

        loss = loss / max(1, len(batch))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping: with batches of 8 and one hard image inside, a
        # stray gradient can throw the weights into a region they do not
        # come back from.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for key, value in accumulated.items():
            totals[key] = totals.get(key, 0.0) + value / max(1, len(batch))
        batches += 1
        step += 1

    return {k: v / max(1, batches) for k, v in totals.items()}, step


def checkpoint_payload(model, config: Config, *, epoch: int | None = None) -> dict:
    """What `load_model` needs to rebuild the network: the architecture and the
    head travel with the weights, so a checkpoint never depends on the config
    of the moment."""
    return {
        "model": model.state_dict(),
        "variant": config.detector.variant,
        "head": model.head_spec.to_dict(),
        "arch": architecture_of(model),
        "num_classes": 1,
        "image_size": config.detector.image_size,
        "epoch": epoch,
    }


def pick_device() -> torch.device:
    """`cuda` if available, else `mps` (Apple's GPU), else `cpu`.

    Everything that trains with this loop is pure torch, so it runs on all
    three. Before it was pinned to `cpu` and on a Mac with a GPU nobody
    noticed. It can be forced with `TESTBANK_DEVICE=cpu`, which is useful to
    reproduce an exact number: MPS and CUDA do not guarantee the same
    arithmetic as the CPU.
    """
    import os

    forced = os.environ.get("TESTBANK_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit(
    dataset: BanknoteDataset,
    config: Config,
    *,
    output_dir: Path,
    device: torch.device | None = None,
    base_lr: float = 1e-3,
    pretrained: Path | None = None,
    validate: Validator | None = None,
    log: Callable[[str], None] | None = print,
) -> tuple[Path, TrainingHistory]:
    """Train and leave the weights. Returns `(path to best.pt, history)`.

    `pretrained`: a foreign checkpoint to start from (Megvii's COCO on the own
    head, DDGRCF's DOTA on its port). What does not fit is skipped and noted.

    `validate`: evaluates the model on `valid` every `config.detector.eval_every`
    epochs and on the last one; `best.pt` is the checkpoint with the best
    `selection_metric`, `last.pt` the final one. Without it (or with
    `eval_every = 0`) `best.pt` is simply the last epoch, and the history says
    so. One line per epoch goes to `log`, and `training.json` is rewritten
    after every epoch.
    """
    device = device or pick_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = TrainingHistory()
    history.notes.append(f"device: {device}")
    say = log or (lambda _: None)
    # EVERYTHING is seeded before building the model, not only the DataLoader.
    # Weights are initialized at random from torch's global generator: seeding
    # only the loader left two runs with the same seed starting from different
    # networks. Measured: 8.518 versus 8.292 in the same configuration.
    seed_everything(config.metrics.seed)
    generator = torch.Generator().manual_seed(config.metrics.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.detector.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )

    model = build_model(config).to(device)
    pretrained = pretrained or config.detector.pretrained
    if pretrained is not None:
        from testbank.models.pretrained import load_pretrained

        history.notes.append(load_pretrained(model, pretrained))
    else:
        history.notes.append("no pretraining: random initial weights")
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-4)

    total_epochs = config.detector.epochs
    eval_every = config.detector.eval_every if validate is not None else 0
    metric_name = config.detector.selection_metric
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    total_steps = max(1, len(loader) * total_epochs)
    step = 0
    for epoch in range(total_epochs):
        terms, step = train_one_epoch(
            model, loader, optimizer, config,
            device=device, step=step, total_steps=total_steps, base_lr=base_lr,
            epoch=epoch,
        )
        lr = learning_rate_at(step, total_steps, base_lr)
        history.record(epoch, terms, lr)
        say(f"epoch {epoch + 1}/{total_epochs}  lr {lr:.2e}  {_format_terms(terms)}")

        is_last = epoch == total_epochs - 1
        if eval_every and (is_last or (epoch + 1) % eval_every == 0):
            model.eval()
            metrics = validate(model, epoch)
            model.train()
            history.validation.append({"epoch": epoch, **metrics})
            value = metrics[metric_name]
            improved = history.best is None or value > history.best[metric_name]
            say(
                f"  valid @ {epoch + 1}: "
                + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items())
                + ("  <- best" if improved else "")
            )
            if improved:
                history.best = {"epoch": epoch, **metrics}
                torch.save(checkpoint_payload(model, config, epoch=epoch), best_path)
        history.write(output_dir / HISTORY_FILE)

    torch.save(checkpoint_payload(model, config, epoch=total_epochs - 1), last_path)
    if history.best is None:
        # No validation: `best.pt` is the last epoch, and it is said, so
        # nobody reads the name as a claim.
        torch.save(checkpoint_payload(model, config, epoch=total_epochs - 1), best_path)
        history.notes.append(
            "best.pt = last epoch: no validation during training "
            "(eval_every = 0 or no valid split)"
        )
    else:
        history.notes.append(
            f"best.pt = epoch {history.best['epoch'] + 1} of {total_epochs} by "
            f"{metric_name} {history.best[metric_name]:.4f}; last.pt = epoch {total_epochs}"
        )
    history.write(output_dir / HISTORY_FILE)
    return best_path, history


def load_model(weights: Path, device: torch.device | None = None) -> YoloxObb:
    """Rebuild the model from the weights file.

    The variant is stored WITH the weights: loading nano weights into a tiny
    would fail with an incomprehensible shape error, and the file is the only
    place where that information cannot drift out of sync.
    """
    device = device or pick_device()
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    if payload.get("arch") == "ddgrcf":
        from testbank.models.ddgrcf import DdgrcfYoloxObb

        model = DdgrcfYoloxObb(num_classes=payload["num_classes"])
    else:
        model = YoloxObb(
            payload["variant"],
            num_classes=payload["num_classes"],
            head=HeadSpec.from_dict(payload.get("head")),
        )
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


__all__ = [
    "HISTORY_FILE",
    "WARMUP_FRACTION",
    "TrainingHistory",
    "Validator",
    "checkpoint_payload",
    "fit",
    "learning_rate_at",
    "load_model",
    "train_one_epoch",
]
