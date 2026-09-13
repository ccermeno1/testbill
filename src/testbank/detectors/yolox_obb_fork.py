"""Adaptador de `buzhidaoshenme/YOLOX-OBB`. Apache-2.0, repositorio ajeno.

ESTE ES EL UNICO MODULO DEL PROYECTO QUE PUEDE IMPORTAR `yolox`.
`tests/test_detectors.py` lo verifica sobre el arbol de fuentes, igual que con
`ultralytics` y con `mmrotate`.

No es un paquete: es un clon
----------------------------
No esta en PyPI y su ultimo commit es de noviembre de 2021. Hay que clonarlo y
parchearlo, y el parche es nuestro para siempre::

    git clone --depth 1 https://github.com/buzhidaoshenme/YOLOX-OBB.git fork
    python tools/patch_yolox_obb_fork.py fork

El parche sustituye `polyiou` -- una extension C++ con SWIG sin wheel, de la que
cuelga el `import yolox` entero -- por shapely, y pone nuestra clase en su
`dota_classes.py`, que trae cableadas las 15 de DOTA. Ver el README.

La ruta del clon se busca, por orden: el argumento del constructor, la variable
de entorno `TESTBANK_YOLOX_OBB_FORK`, y `./YOLOX-OBB`.

Lo que hay que saber antes de leer sus numeros
----------------------------------------------
**Su evaluador no evalua.** `DOTAOBBDetection.evaluate_detections` hace
`return 0.0, 0.0` antes de calcular nada (`dota_obb.py:190`, el `#add` es suyo):
todo el mAP que viene debajo es codigo muerto. Da igual, porque `evaluate` lo
pone `BaseDetector` y puntua con NUESTRAS metricas -- pero significa que el mAP
que imprima su entrenamiento durante la validacion es un cero literal, no un
resultado malo.

**La perdida es KLD**, la formulacion gaussiana que este proyecto descarto a
proposito por alejarse del IoU rotado con el que medimos. Su resultado no es
separable entre backbone y perdida.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from testbank.config import Config
from testbank.detectors.base import (
    BaseDetector,
    DetectorError,
    TrainResult,
    register,
)

#: Donde se busca el clon si no se dice otra cosa.
ENV_VAR = "TESTBANK_YOLOX_OBB_FORK"
DEFAULT_DIRECTORY = "YOLOX-OBB"

#: Lo que el parche tiene que haber dejado puesto. Se comprueba antes de
#: importar nada: el fallo sin parchear es un `ModuleNotFoundError: _polyiou`
#: desde dentro de un envoltorio de SWIG, que no dice donde mirar.
_PATCH_MARKER = "testbank"
_PATCHED_FILES = (
    "DOTA_devkit_YOLO/polyiou.py",
    "yolox/data/datasets/dota_classes.py",
)

_SETUP_HINT = (
    "YOLOX-OBB no es un paquete: hay que clonarlo y parchearlo.\n"
    "  git clone --depth 1 https://github.com/buzhidaoshenme/YOLOX-OBB.git "
    f"{DEFAULT_DIRECTORY}\n"
    f"  python tools/patch_yolox_obb_fork.py {DEFAULT_DIRECTORY}\n"
    f"Y si va a otro sitio, {ENV_VAR}=<ruta>. Ver el README."
)


def find_fork(explicit: Path | None = None) -> Path:
    """La raiz del clon, ya comprobada. Falla con instrucciones, no con ImportError."""
    candidate = explicit or os.environ.get(ENV_VAR) or DEFAULT_DIRECTORY
    root = Path(candidate).expanduser().resolve()
    if not (root / "yolox").is_dir():
        raise DetectorError(f"no hay un clon de YOLOX-OBB en {root}.\n{_SETUP_HINT}")
    for relative in _PATCHED_FILES:
        target = root / relative
        if not target.exists() or _PATCH_MARKER not in target.read_text(
            encoding="utf-8", errors="ignore"
        ):
            raise DetectorError(
                f"{root} esta sin parchear ({relative}).\n"
                f"  python tools/patch_yolox_obb_fork.py {root}"
            )
    return root


def _import_fork(root: Path):
    """Mete el clon en `sys.path` e importa lo suyo. Perezoso a proposito."""
    if str(root) not in sys.path:
        # Al final: un repositorio ajeno no debe ganarle a nuestros modulos si
        # coincide algun nombre.
        sys.path.append(str(root))
    try:
        from yolox.data.data_augment_obb import preproc
        from yolox.models.yolo_head_obb_kld import YOLOXHeadOBB_KLD
        from yolox.models.yolo_pafpn import YOLOPAFPN
        from yolox.models.yolox_obb_kld import YOLOXOBB_KLD
        from yolox.utils.boxes import postprocessobb_kld
    except ImportError as exc:
        raise DetectorError(f"no se pudo importar yolox desde {root}: {exc}") from exc
    return {
        "preproc": preproc,
        "YOLOXHeadOBB_KLD": YOLOXHeadOBB_KLD,
        "YOLOPAFPN": YOLOPAFPN,
        "YOLOXOBB_KLD": YOLOXOBB_KLD,
        "postprocessobb_kld": postprocessobb_kld,
    }


#: Su `preproc` normaliza con estas constantes y su exp de DOTA tambien. Si se
#: cambian en el entrenamiento hay que cambiarlas aqui: no las lee de ningun
#: sitio comun, van escritas a mano en los dos.
RGB_MEANS = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


#: La plantilla del fichero `Exp` que come su `Trainer`.
#:
#: Se GENERA en vez de llevar uno fijo en el repositorio porque su propio exp de
#: referencia trae la ruta de datos cableada -- literalmente
#: `/home/lyy/gxw/DOTA_OBB_1_5` -- y porque asi el exp de cada ejecucion queda
#: junto a sus pesos, diciendo exactamente con que se entreno.
_EXP_TEMPLATE = '''# Generado por testbank. No editar a mano: se reescribe en cada ejecucion.
import os

import torch

from yolox.exp import ExpOBB_KLD as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = {num_classes}
        self.depth = {depth}
        self.width = {width}
        self.input_size = ({side}, {side})
        self.test_size = ({side}, {side})
        # Sin redimensionado aleatorio: este proyecto entrena de forma
        # determinista y `random_size` mete una fuente de ruido por lote.
        self.random_size = None
        self.seed = {seed}
        self.exp_name = "{exp_name}"
        self.output_dir = r"{output_dir}"
        self.data_num_workers = 0

        self.degrees = 0.0
        self.translate = 0.1
        self.scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = False

        self.warmup_epochs = {warmup_epochs}
        self.max_epoch = {epochs}
        self.no_aug_epochs = {no_aug_epochs}
        self.basic_lr_per_img = 0.0025 / 16.0
        self.scheduler = "yoloxwarmcos"
        self.min_lr_ratio = 0.05
        self.ema = True
        self.save_interval = 1
        self.print_interval = 5
        # Su evaluador devuelve 0.0 fijo, asi que evaluar durante el
        # entrenamiento solo gasta tiempo. Se puntua despues con `evaluate`.
        self.eval_interval = {epochs} + 1

        self.test_conf = {confidence}
        self.nmsthre = {nms_iou}

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            DataLoader,
            DOTAOBBDetection,
            InfiniteSampler,
            MosaicDetectionOBB,
            TrainTransformOBB,
            YoloBatchSampler,
        )

        dataset = DOTAOBBDetection(
            data_dir=r"{data_dir}",
            image_sets=[("{year}", "trainval")],
            img_size=self.input_size,
            preproc=TrainTransformOBB(
                rgb_means={rgb_means}, std={rgb_std}, max_labels=50
            ),
        )
        dataset = MosaicDetectionOBB(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransformOBB(
                rgb_means={rgb_means}, std={rgb_std}, max_labels=100
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )
        self.dataset = dataset
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed or 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )
        return DataLoader(
            self.dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import DOTAOBBDetection, ValTransformOBB

        dataset = DOTAOBBDetection(
            data_dir=r"{data_dir}",
            image_sets=[("{year}", "val")],
            img_size=self.test_size,
            preproc=ValTransformOBB(rgb_means={rgb_means}, std={rgb_std}),
        )
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            sampler=torch.utils.data.SequentialSampler(dataset),
            batch_size=batch_size,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import DOTAEvaluator

        return DOTAEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
'''


def _require_cuda() -> None:
    """Su entrenamiento NO corre en CPU, y conviene decirlo antes de empezar.

    No es una limitacion de este adaptador: es como esta escrito el fork.

        yolox/core/trainer.py:47    self.device = "cuda:{}".format(self.local_rank)
        yolox/core/trainer.py:161   torch.cuda.set_device(self.local_rank)
        yolox/data/data_prefetcher.py:23  self.stream = torch.cuda.Stream()

    Las dos primeras se parchearian en dos lineas. La tercera no: su
    `DataPrefetcher` esta construido entero sobre streams de CUDA, y reescribirlo
    ya no es un parche quirurgico sino mantener su bucle de datos.

    Sin esta comprobacion, el fallo es un `AttributeError: module 'torch._C' has
    no attribute '_cuda_setDevice'` seguido de `lost sys.stderr`, despues de
    haber volcado el dataset entero a disco.

    La INFERENCIA si funciona en CPU: `predict` no pasa por nada de esto.
    """
    import torch

    if not torch.cuda.is_available():
        raise DetectorError(
            "YOLOX-OBB solo entrena con CUDA, y aqui torch es "
            f"{torch.__version__} sin GPU disponible.\n"
            "  No es cosa del adaptador: su `DataPrefetcher` esta construido "
            "sobre `torch.cuda.Stream` (data_prefetcher.py:23), y eso no se "
            "parchea en dos lineas.\n"
            "  La inferencia SI corre en CPU: `evaluate` con unos pesos ya "
            "entrenados funciona sin GPU."
        )


class _Args:
    """Lo que su `Trainer` lee de `args`. Son siete campos, comprobados uno a uno.

    Se construye a mano en vez de con `argparse` porque pasar por su CLI
    significaria pasar tambien por su `launch()`, que monta el entorno
    distribuido aunque haya una sola GPU.
    """

    def __init__(self, *, experiment_name, batch_size, checkpoint=None):
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.fp16 = False
        # `args.fp` (sin el 16) es un tercer nombre que su trainer tambien lee.
        self.fp = False
        self.occupy = False
        self.resume = False
        self.ckpt = str(checkpoint) if checkpoint else None
        self.start_epoch = None


@register
class YoloxObbForkDetector(BaseDetector):
    """YOLOX-OBB con perdida KLD, del fork de buzhidaoshenme.

    Variante `small` (depth 0.33, width 0.50), que es la unica para la que su
    repositorio trae un exp de referencia. Medido: 8,938,069 parametros, diez
    veces la cabeza propia.
    """

    name = "yolox-obb-fork-small"
    license = "Apache-2.0"
    production_ready = False
    DEPTH = 0.33
    WIDTH = 0.50
    YEAR = "2007"

    notes = (
        (
            "Repositorio ajeno y abandonado (ultimo commit en 2021): se clona y "
            "se parchea, y el parche es nuestro para siempre."
        ),
        (
            "Perdida KLD (gaussiana), distinta del IoU rotado con el que "
            "medimos: su resultado no es separable entre backbone y perdida."
        ),
        "Su evaluador devuelve 0.0 fijo; se puntua con las metricas de testbank.",
        (
            "production_ready=False: no por licencia (es Apache-2.0) sino "
            "porque depende de un clon parcheado a mano de un repo muerto."
        ),
    )

    def __init__(self, fork_root: Path | None = None) -> None:
        self._fork_root = fork_root

    # --- entrenamiento ----------------------------------------------------

    def train(self, samples_by_split, config: Config, *, output_dir: Path) -> TrainResult:
        root = find_fork(self._fork_root)
        _import_fork(root)
        _require_cuda()
        from yolox.core import Trainer
        from yolox.exp import get_exp

        from testbank.dataio.export import export

        output_dir = Path(output_dir)
        data_root = output_dir / "voc"
        result = export(
            "yolox_obb_voc", samples_by_split, config, out_dir=data_root
        )
        exp_path = self._write_exp(config, output_dir=output_dir, data_dir=data_root)

        exp = get_exp(str(exp_path), None)
        args = _Args(
            experiment_name=exp.exp_name, batch_size=config.detector.batch_size
        )
        Trainer(exp, args).train()

        weights = output_dir / exp.exp_name / "best_ckpt.pth"
        if not weights.exists():
            weights = output_dir / exp.exp_name / "latest_ckpt.pth"
        if not weights.exists():
            raise DetectorError(
                f"el entrenamiento termino sin dejar pesos en {weights.parent}"
            )
        return TrainResult(
            weights=weights,
            epochs=config.detector.epochs,
            notes=(
                f"exportado a VOC en {result.entry_point.parent.parent}",
                f"exp generado en {exp_path.name}",
                (
                    "el mAP que imprimio la validacion es un 0.0 literal: su "
                    "evaluador no evalua. Puntuar con `evaluate`."
                ),
            ),
        )

    def _write_exp(self, config: Config, *, output_dir: Path, data_dir: Path) -> Path:
        epochs = config.detector.epochs
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "exp_yolox_obb_fork.py"
        path.write_text(
            _EXP_TEMPLATE.format(
                num_classes=1,
                depth=self.DEPTH,
                width=self.WIDTH,
                side=config.detector.image_size,
                seed=config.metrics.seed,
                exp_name=self.name,
                output_dir=str(output_dir),
                # `warmup` y `no_aug` no pueden pasar de `epochs`: con los
                # circuitos de 1-2 epocas del proyecto, los 5 y 15 de su exp de
                # referencia dejarian el entrenamiento entero en calentamiento.
                warmup_epochs=min(5, max(0, epochs - 1)),
                no_aug_epochs=min(15, max(0, epochs - 1)),
                epochs=epochs,
                confidence=config.detector.confidence_threshold,
                nms_iou=config.detector.nms_iou,
                data_dir=str(data_dir),
                year=self.YEAR,
                rgb_means=RGB_MEANS,
                rgb_std=RGB_STD,
            ),
            encoding="utf-8",
        )
        return path

    # --- inferencia -------------------------------------------------------

    def predict(self, samples, *, weights: Path, config: Config) -> dict:
        """`sample_id -> [Prediction]` en coordenadas NORMALIZADAS.

        Aqui esta el trabajo que su repositorio no permite hacer desde dentro:
        su `evaluate_detections` devuelve 0.0 fijo, asi que la unica forma de
        puntuarlo con las metricas del proyecto es sacarle las cajas a mano.

        Tres detalles suyos que hay que respetar o los numeros salen mal:

        1. **`preproc` hace letterbox anclado ARRIBA A LA IZQUIERDA**, con relleno
           114 y `r = min(alto_entrada/alto, ancho_entrada/ancho)`. Como no
           centra, deshacerlo es dividir por `r` y ya: sin desplazamiento. Si se
           asumiera centrado, todo saldria corrido media banda de relleno.
        2. **`postprocessobb_kld` ya devuelve los CUATRO VERTICES** en pixeles de
           la imagen de entrada, no `(cx, cy, w, h, angulo)`. Y los redondea con
           `np.int0`, que es truncado: se pierde precision subpixel en inferencia
           igual que el formato la pierde en disco.
        3. **La columna 8 es `class_conf * obj_conf`** y la 9 la clase.
        """
        import numpy as np
        import torch

        root = find_fork(self._fork_root)
        modules = _import_fork(root)
        import cv2

        from testbank.dataio.image_sizes import SizeIndex
        from testbank.geometry.quad import Quad, canonicalize
        from testbank.metrics.core import Prediction

        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        side = config.detector.image_size
        model = self._load(modules, weights, side=side)

        out: dict[str, list] = {}
        with torch.no_grad():
            for sample in samples:
                image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise DetectorError(f"no se pudo leer {sample.image_path}")
                tensor, ratio = modules["preproc"](
                    image, (side, side), RGB_MEANS, RGB_STD
                )
                outputs = model(torch.from_numpy(tensor).unsqueeze(0))
                detections = modules["postprocessobb_kld"](
                    outputs.cpu(),
                    1,
                    conf_thre=config.detector.confidence_threshold,
                    nms_thre=config.detector.nms_iou,
                )[0]

                width, height = sizes.size(sample.sample_id)
                predictions = []
                if detections is not None:
                    for row in detections.numpy():
                        corners = np.asarray(row[:8], dtype=float).reshape(4, 2)
                        corners /= ratio  # deshace el letterbox: sin offset
                        quad = Quad.from_xy(
                            [(x / width, y / height) for x, y in corners]
                        )
                        predictions.append(
                            Prediction(
                                quad=canonicalize(quad, aspect=width / height),
                                score=float(row[8]),
                                class_id=int(row[9]),
                            )
                        )
                out[sample.sample_id] = predictions
        return out

    def _load(self, modules, weights: Path, *, side: int):
        """El modelo con los pesos puestos, en evaluacion.

        Prefiere los pesos EMA (`model` es la clave que guarda su trainer) y
        acepta tanto el checkpoint suyo como un `state_dict` pelado.
        """
        import torch

        backbone = modules["YOLOPAFPN"](self.DEPTH, self.WIDTH)
        head = modules["YOLOXHeadOBB_KLD"](1, self.WIDTH)
        model = modules["YOLOXOBB_KLD"](backbone, head)

        checkpoint = torch.load(str(weights), map_location="cpu")
        state = checkpoint.get("model", checkpoint) if isinstance(
            checkpoint, dict
        ) else checkpoint
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            raise DetectorError(
                f"{weights} no encaja con el modelo: faltan {len(missing)} "
                f"tensores, el primero {missing[0]!r}"
            )
        model.eval()
        # `head.decode_outputs` necesita saber el tamano de entrada, y su trainer
        # lo fija por la config del exp. En inferencia suelta hay que decirlo.
        model.head.decode_in_inference = True
        return model


__all__ = ["ENV_VAR", "YoloxObbForkDetector", "find_fork"]
