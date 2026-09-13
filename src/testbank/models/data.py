"""Dataset de entrenamiento del candidato propio.

Lee por la misma puerta que todo lo demas: `dataio/view.prepare`, que aplica el
filtro de area relativa y la politica de billetes que cruzan el borde. Si esto
cargara los ficheros por su cuenta, el candidato propio entrenaria con una
verdad distinta de la de Ultralytics y la tabla los compararia como iguales.

Sobre el redimensionado
-----------------------
Se reescala directo a `image_size x image_size`, sin letterbox. Podria parecer
descuidado, pero es lo coherente con estos datos: 489 de las 502 imagenes YA
llegan a 416x416 desde el origen, deformadas al cuadrado por el export. El
aspecto real de los billetes se perdio antes de que nosotros toquemos nada, asi
que anadir letterbox ahora no recupera nada -- solo mete franjas negras y una
segunda geometria que explicar. Ver la nota del README sobre el 19% de cajas
casi cuadradas.

Si en algun momento se resuben los originales sin redimensionar, esto hay que
revisarlo: ahi el letterbox si valdria la pena.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from testbank.config import Config
from testbank.dataio.view import PreparedSample, prepare
from testbank.geometry.quad import Quad


def image_to_input(image_bgr: np.ndarray) -> torch.Tensor:
    """Imagen de OpenCV (BGR, uint8) -> tensor `(3, H, W)` de entrada a la red.

    **BGR crudo en 0-255, sin normalizar.** Es la convencion de YOLOX (Megvii)
    y de DDGRCF, y por tanto la que esperan los pesos preentrenados que se
    cargan: el COCO de Megvii en la cabeza propia y el DOTA de DDGRCF en su
    port. Antes se daba RGB en [0, 1] -- canales cambiados y escala 255 veces
    menor -- y el preentreno llegaba destrozado a la primera capa: medido, el
    port con DOTA daba mAP 0.000 y puntuaciones maximas de 0.004 tras una epoca.

    Para entrenar de cero da igual (la BatchNorm absorbe la escala), asi que la
    convencion se fija aqui, en un solo sitio, y la usan entrenamiento e
    inferencia. Cambiarla en uno solo invalidaria todos los pesos guardados.
    """
    return torch.from_numpy(np.ascontiguousarray(image_bgr.transpose(2, 0, 1))).float()


def quad_to_box(quad: Quad, width: int, height: int) -> tuple[float, ...]:
    """Quad normalizado -> `cx, cy, w, h, theta` en PIXELES.

    El quad canonico ancla `p0->p1` en el lado mas largo, asi que `w` sale
    siempre siendo el lado largo y `theta` su orientacion. Eso no es casualidad
    que aproveche: es la razon de que exista el orden canonico.
    """
    points = [(x * width, y * height) for x, y in quad.points]
    cx = sum(p[0] for p in points) / 4
    cy = sum(p[1] for p in points) / 4
    long_side = math.dist(points[0], points[1])
    short_side = math.dist(points[1], points[2])
    theta = math.atan2(
        points[1][1] - points[0][1], points[1][0] - points[0][0]
    ) % math.pi
    return cx, cy, long_side, short_side, theta


@dataclass(frozen=True, slots=True)
class Batch:
    """Un lote. Las cajas van en lista porque cada imagen tiene un numero
    distinto y apilarlas exigiria rellenar con basura que luego hay que
    acordarse de ignorar."""

    images: torch.Tensor
    boxes: list[torch.Tensor]
    classes: list[torch.Tensor]
    sample_ids: list[str]

    def __len__(self) -> int:
        return self.images.shape[0]


class BanknoteDataset(Dataset):
    """Imagenes y cajas orientadas, ya filtradas y en el tamano de entrada."""

    def __init__(self, prepared: list[PreparedSample], image_size: int) -> None:
        self.items = list(prepared)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = cv2.imread(str(item.sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"no se pudo leer {item.sample.image_path}")
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        tensor = image_to_input(image)

        # Los quads son normalizados, asi que las cajas se calculan ya en la
        # escala de entrada: no hay que reescalarlas despues.
        boxes = [
            quad_to_box(quad, self.image_size, self.image_size)
            for quad in item.quads
        ]
        return (
            tensor,
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5),
            torch.zeros(len(boxes), dtype=torch.long),
            item.sample_id,
        )


def collate(entries) -> Batch:
    images, boxes, classes, ids = zip(*entries)
    return Batch(
        images=torch.stack(images),
        boxes=list(boxes),
        classes=list(classes),
        sample_ids=list(ids),
    )


def build_datasets(
    samples_by_split: dict[str, list], config: Config
) -> dict[str, BanknoteDataset]:
    """Un dataset por particion, compartiendo filtro y politica de borde."""
    prepared, _ = prepare(samples_by_split, config)
    return {
        split: BanknoteDataset(items, config.detector.image_size)
        for split, items in prepared.items()
    }


__all__ = ["BanknoteDataset", "Batch", "build_datasets", "collate", "image_to_input", "quad_to_box"]
