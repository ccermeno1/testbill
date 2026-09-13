"""Bucle de entrenamiento del candidato propio.

Deliberadamente sobrio. Sin mosaico, sin mixup, sin EMA, sin escalado
multiescala. Todo eso ayuda en COCO con 118.000 imagenes; aqui hay 351, y cada
truco anadido es un hiperparametro mas que ajustar a ciegas sobre un conjunto de
validacion de 101 imagenes donde las diferencias caen dentro del ruido.

La linea base tiene que ser interpretable antes que buena. Si hace falta subir,
se anade UNA cosa y se mide -- que es para lo que existe la tabla comparativa.

Determinismo
------------
Semilla fijada y registrada, `DataLoader` sin workers y con generador propio.
Con `num_workers > 0` el orden de los lotes depende de la planificacion del
sistema operativo y dos ejecuciones con la misma semilla dejan de coincidir.
Con 351 imagenes de 416x416 la carga no es el cuello de botella.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from testbank.config import Config
from testbank.experiment.provenance import seed_everything
from testbank.models.data import BanknoteDataset, Batch, collate
from testbank.models.recipes import architecture_of, build_model, losses_for_image
from testbank.models.yolox_obb import HeadSpec, YoloxObb

#: Fraccion del entrenamiento dedicada a subir el learning rate desde casi cero.
#: Sin calentamiento, las primeras iteraciones con la cabeza recien inicializada
#: dan gradientes enormes que desestabilizan la BatchNorm.
WARMUP_FRACTION = 0.05


@dataclass
class TrainingHistory:
    """Lo que paso en cada epoca. Va al `run.json` para poder mirarlo despues."""

    epochs: list[dict] = field(default_factory=list)
    #: Cosas que pasaron una vez y conviene dejar escritas: p. ej. que
    #: preentreno se cargo y que tensores se saltaron.
    notes: list[str] = field(default_factory=list)

    def record(self, epoch: int, terms: dict, learning_rate: float) -> None:
        self.epochs.append({"epoch": epoch, "lr": learning_rate, **terms})

    def to_list(self) -> list[dict]:
        return list(self.epochs)


def learning_rate_at(step: int, total: int, base: float) -> float:
    """Calentamiento lineal y despues coseno hasta casi cero."""
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
    """Una epoca. Devuelve las perdidas promediadas y el paso alcanzado.

    `epoch` se pasa porque una receta (la del fork) cambia de forma en las
    ultimas epocas: enciende una L1. La perdida tiene que saber en cual va.
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

        # La asignacion es POR IMAGEN: mezclar las cajas de todo el lote haria
        # que una celda de la imagen 3 pudiera asignarse a un billete de la 1.
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
        # Recorte de gradiente: con lotes de 8 y una imagen dificil dentro, un
        # gradiente suelto puede tirar los pesos a una zona de la que no vuelven.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for key, value in accumulated.items():
            totals[key] = totals.get(key, 0.0) + value / max(1, len(batch))
        batches += 1
        step += 1

    return {k: v / max(1, batches) for k, v in totals.items()}, step



def fit(
    dataset: BanknoteDataset,
    config: Config,
    *,
    output_dir: Path,
    device: torch.device | None = None,
    base_lr: float = 1e-3,
    pretrained: Path | None = None,
) -> tuple[Path, TrainingHistory]:
    """Entrena y deja los pesos. Devuelve `(ruta, historial)`.

    `pretrained`: un checkpoint ajeno con el que arrancar (hoy solo el de DOTA
    de DDGRCF sobre su port). Lo que no encaje se salta y queda anotado.
    """
    device = device or torch.device("cpu")
    history = TrainingHistory()
    # Se siembra TODO antes de construir el modelo, no solo el DataLoader. Los
    # pesos se inicializan al azar desde el generador global de torch: sembrar
    # solo el cargador dejaba dos ejecuciones con la misma semilla partiendo de
    # redes distintas. Medido: 8.518 frente a 8.292 en la misma configuracion.
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
        history.notes.append("sin preentreno: pesos iniciales aleatorios")
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-4)

    total_steps = max(1, len(loader) * config.detector.epochs)
    step = 0
    for epoch in range(config.detector.epochs):
        terms, step = train_one_epoch(
            model, loader, optimizer, config,
            device=device, step=step, total_steps=total_steps, base_lr=base_lr,
            epoch=epoch,
        )
        history.record(epoch, terms, learning_rate_at(step, total_steps, base_lr))

    output_dir.mkdir(parents=True, exist_ok=True)
    weights = output_dir / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "variant": config.detector.variant,
            "head": model.head_spec.to_dict(),
            "arch": architecture_of(model),
            "num_classes": 1,
            "image_size": config.detector.image_size,
        },
        weights,
    )
    return weights, history


def load_model(weights: Path, device: torch.device | None = None) -> YoloxObb:
    """Reconstruye el modelo desde el fichero de pesos.

    La variante se guarda CON los pesos: cargar unos pesos de nano en un tiny
    fallaria con un error de formas incomprensible, y el fichero es el unico
    sitio donde esa informacion no se puede desincronizar.
    """
    payload = torch.load(weights, map_location=device or "cpu", weights_only=False)
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
    return model.to(device or torch.device("cpu")).eval()


__all__ = [
    "WARMUP_FRACTION",
    "TrainingHistory",
    "fit",
    "learning_rate_at",
    "load_model",
    "train_one_epoch",
]
