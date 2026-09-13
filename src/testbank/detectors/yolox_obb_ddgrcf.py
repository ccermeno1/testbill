"""Adaptador de `DDGRCF/YOLOX_OBB`. Apache-2.0, repositorio ajeno.

ESTE ES EL UNICO MODULO DEL PROYECTO QUE PUEDE IMPORTAR `yolox` DE ESTE CLON.
Comparte nombre de paquete con el fork de buzhidaoshenme, asi que los dos NO
pueden convivir en el mismo `sys.path`: este va en su propio entorno
(`.venv-yolox-ddgrcf`), y el test de aislamiento vigila que ningun otro modulo
importe `yolox`.

Por que existe, habiendo ya un fork de YOLOX-OBB
------------------------------------------------
Este publica **pesos entrenados en DOTA** (`YOLOX_s_dota1_0`, 70.82 mAP@0.5) y
el otro no: el otro solo trae el COCO de Megvii. Es el unico YOLOX con una cabeza
OBB ya entrenada que se ha encontrado. El precio es alto y se detalla abajo.

Tres muros, y que hace cada uno al chocar
-----------------------------------------
1. **Operadores compilados en el bucle de entrenamiento.** Su SimOTA calcula el
   solape con `box_iou_rotated` (C++/CUDA propio), y usa `nms_rotated` y
   `convex`. No hay wheels: `python setup.py develop` con MSVC. Sin compilador,
   `import yolox` falla. Aqui se detecta y se dice que instalar.
2. **CUDA.** Su `DataPrefetcher` esta construido sobre `torch.cuda.Stream`,
   como el del otro fork. `train` lo comprueba antes de exportar nada.
3. **Los pesos estan en Baidu Pan** con codigo de acceso. No se pueden bajar
   desde aqui: se apunta la ruta por variable de entorno y, si no esta, se
   avisa de que se entrena de cero, que no es lo que se queria.

Lo que si esta verificado hoy, sin compilador: el camino de datos entero.
`export dota` -> `img_split.py` de BboxToolkit -> los `.pkl` que lee su
`DOTADataset`, con 1 parche = 1 imagen y todas las cajas conservadas
(351 -> 351, 537 -> 537 en train).

Su receta, para la tabla
------------------------
`configs/losses/yolox_losses_obb.yaml`: caja **PolyIoU** (IoU de poligonos,
exacta, via operador compilado) x5, obj BCE, cls BCE, L1 extra. NO es la KLD,
aunque el repositorio la trae: el modelo del zoo se entreno con PolyIoU. Y su
modelo no es el CSPDarknet estandar sino uno definido en yaml con bloques C3 y
ReLU: sus pesos no sirven para la cabeza propia.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from testbank.config import Config
from testbank.detectors.base import (
    BaseDetector,
    DetectorError,
    TrainResult,
    register,
)

ENV_VAR = "TESTBANK_YOLOX_OBB_DDGRCF"
WEIGHTS_ENV_VAR = "TESTBANK_YOLOX_OBB_DDGRCF_WEIGHTS"
DEFAULT_DIRECTORY = "YOLOX_OBB"

_SETUP_HINT = (
    "DDGRCF/YOLOX_OBB no es un paquete: hay que clonarlo y COMPILARLO en su "
    "propio entorno.\n"
    "  git clone --depth 1 https://github.com/DDGRCF/YOLOX_OBB.git "
    f"{DEFAULT_DIRECTORY}\n"
    "  git clone --depth 1 https://github.com/jbwang1997/BboxToolkit.git\n"
    "  uv venv .venv-yolox-ddgrcf --python 3.11\n"
    "  VIRTUAL_ENV=.venv-yolox-ddgrcf uv pip install -e BboxToolkit -e .\n"
    f"  cd {DEFAULT_DIRECTORY} && ../.venv-yolox-ddgrcf/Scripts/python setup.py develop\n"
    "El ultimo paso exige Microsoft C++ Build Tools (MSVC 14+). Ver el README."
)


def find_fork(explicit: Path | None = None) -> Path:
    candidate = explicit or os.environ.get(ENV_VAR) or DEFAULT_DIRECTORY
    root = Path(candidate).expanduser().resolve()
    if not (root / "yolox").is_dir() or not (root / "configs").is_dir():
        raise DetectorError(f"no hay un clon de DDGRCF/YOLOX_OBB en {root}.\n{_SETUP_HINT}")
    return root


def _import_fork(root: Path):
    """Importa lo suyo. El fallo tipico es la extension sin compilar."""
    if str(root) not in sys.path:
        sys.path.append(str(root))
    try:
        import yolox.ops  # noqa: F401  -- es lo que revienta sin compilar
        from yolox.data.data_augment import preproc
        from yolox.exp import get_exp
        from yolox.utils import obbpostprocess
    except ImportError as exc:
        raise DetectorError(
            f"no se pudo importar yolox desde {root}: {exc}\n"
            "  Casi seguro que sus operadores no estan compilados. "
            f"`python setup.py develop` en {root} exige MSVC 14+:\n"
            "  https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        ) from exc
    return {"preproc": preproc, "get_exp": get_exp, "obbpostprocess": obbpostprocess}


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise DetectorError(
            "DDGRCF/YOLOX_OBB solo entrena con CUDA, y aqui torch es "
            f"{torch.__version__} sin GPU disponible.\n"
            "  Su DataPrefetcher esta construido sobre torch.cuda.Stream "
            "(yolox/data/data_prefetcher.py), igual que el otro fork.\n"
            "  La inferencia SI corre en CPU con unos pesos ya entrenados."
        )


def pretrained_weights() -> Path | None:
    """Los pesos de DOTA, si el usuario los ha bajado. Estan en Baidu Pan
    (`MODEL_ZOO` en su README, codigo `tdm6`): no se pueden traer desde aqui."""
    value = os.environ.get(WEIGHTS_ENV_VAR)
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise DetectorError(f"{WEIGHTS_ENV_VAR}={value} no es un fichero")
    return path


#: Su `Exp` de referencia (`exps/example/yolox_obb/yolox_s_dota1_0.py`) con lo
#: nuestro encima. Se GENERA: el suyo trae las rutas de datos y las 15 clases
#: de DOTA por un yaml, y aqui `dataset_config` va como diccionario en linea,
#: que su `_get_data_info` acepta igual.
_EXP_TEMPLATE = '''# Generado por testbank. No editar a mano: se reescribe en cada ejecucion.
import os

from yolox.exp import OBBExp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "{exp_name}"
        self.output_dir = r"{output_dir}"
        self.modules_config = r"{modules_config}"
        self.losses_config = r"{losses_config}"
        self.dataset_config = dict(
            data_dir=r"{data_dir}",
            train_ann="train",
            val_ann="valid",
            test_ann="valid",
            num_classes=1,
            class_names=["euro_banknote"],
        )
        self.input_size = ({side}, {side})
        self.test_size = ({side}, {side})
        self.multiscale_range = 0
        self.data_num_workers = 0
        self.seed = {seed}

        self.max_epoch = {epochs}
        self.warmup_epochs = {warmup_epochs}
        self.no_aug_epochs = {no_aug_epochs}
        self.no_eval = True
        self.eval_interval = {epochs} + 1
        self.mosaic_prob = 1.0
        self.copy_paste_prob = 0.0
        self.mixup_prob = 0.0
        self.enable_resample = False
        self.aug_ignore = None
        self.empty_ignore = True
        self.postprocess_cfg = dict(conf_thre={confidence}, nms_thre={nms_iou})
        self.evaluate_cfg = dict(is_submiss=False, is_merge=False, nproc=1)
'''


class _Args:
    """Los ocho campos que su `Trainer` lee de `args`."""

    def __init__(self, *, experiment_name, batch_size, checkpoint=None):
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.fp16 = False
        self.fp = False
        self.occupy = False
        self.cache = False
        self.resume = False
        self.ckpt = str(checkpoint) if checkpoint else None
        self.start_epoch = None


@register
class YoloxObbDdgrcfDetector(BaseDetector):
    """YOLOX-OBB de DDGRCF: el unico con cabeza OBB ya entrenada (DOTA)."""

    name = "yolox-obb-ddgrcf-small"
    license = "Apache-2.0"
    production_ready = False
    notes = (
        (
            "Clon compilado a mano en su propio entorno: operadores C++/CUDA "
            "propios (box_iou_rotated, nms_rotated, convex) sin wheels."
        ),
        (
            "Pesos preentrenados en DOTA (70.82 mAP@0.5) publicados en Baidu "
            "Pan; se apuntan por TESTBANK_YOLOX_OBB_DDGRCF_WEIGHTS."
        ),
        "Receta: PolyIoU x5 (IoU de poligonos exacta) + obj + cls + L1.",
        (
            "production_ready=False: no por licencia (Apache-2.0) sino por "
            "depender de extensiones compiladas y de un clon ajeno."
        ),
    )

    def __init__(self, fork_root: Path | None = None) -> None:
        self._fork_root = fork_root

    # --- datos ----------------------------------------------------------

    def prepare_data(self, samples_by_split, config: Config, *, output_dir: Path) -> Path:
        """`export dota` + `img_split.py` de BboxToolkit -> lo que lee su dataset.

        Verificado sin compilador: es Python puro. `--sizes 1024` es mayor que
        cualquier imagen nuestra, asi que sale 1 parche = 1 imagen, y las
        predicciones vuelven a nuestros ids sin merge. Con 416 se partian 7
        imagenes en dos parches con la caja duplicada.
        """
        from testbank.dataio.export import export

        dota_root = output_dir / "dota"
        export("dota", samples_by_split, config, out_dir=dota_root)
        data_root = output_dir / "bboxtoolkit"
        classes = output_dir / "classes.txt"
        classes.write_text("euro_banknote\n", encoding="utf-8")
        split_tool = _find_img_split()
        for split in samples_by_split:
            subprocess.run(
                [
                    sys.executable, str(split_tool),
                    "--load_type", "dota",
                    "--img_dirs", str(dota_root / split / "images"),
                    "--ann_dirs", str(dota_root / split / "labelTxt"),
                    "--classes", str(classes),
                    "--sizes", "1024", "--gaps", "0", "--no_padding",
                    "--nproc", "1",
                    "--save_dir", str(data_root / split),
                    "--save_ext", ".png",
                ],
                check=True,
                capture_output=True,
            )
        return data_root

    # --- entrenamiento ----------------------------------------------------

    def train(self, samples_by_split, config: Config, *, output_dir: Path) -> TrainResult:
        root = find_fork(self._fork_root)
        modules = _import_fork(root)
        _require_cuda()
        from yolox.core import Trainer

        output_dir = Path(output_dir)
        data_root = self.prepare_data(samples_by_split, config, output_dir=output_dir)
        exp_path = self._write_exp(config, root=root, output_dir=output_dir, data_dir=data_root)
        exp = modules["get_exp"](str(exp_path), None)

        weights_in = pretrained_weights()
        notes = ["receta del fork: PolyIoU x5 + obj + cls + L1 (su yaml de losses)"]
        if weights_in is None:
            notes.append(
                "SIN preentreno: TESTBANK_YOLOX_OBB_DDGRCF_WEIGHTS no esta. Se "
                "entrena de cero, que NO es el motivo por el que este candidato "
                "esta en la lista"
            )
        else:
            notes.append(f"preentreno DOTA desde {weights_in.name}")

        args = _Args(
            experiment_name=exp.exp_name,
            batch_size=config.detector.batch_size,
            checkpoint=weights_in,
        )
        Trainer(exp, args).train()

        weights = output_dir / exp.exp_name / "latest_ckpt.pth"
        if not weights.exists():
            raise DetectorError(f"el entrenamiento termino sin dejar pesos en {weights.parent}")
        return TrainResult(weights=weights, epochs=config.detector.epochs, notes=tuple(notes))

    def _write_exp(self, config: Config, *, root: Path, output_dir: Path, data_dir: Path) -> Path:
        epochs = config.detector.epochs
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "exp_yolox_obb_ddgrcf.py"
        path.write_text(
            _EXP_TEMPLATE.format(
                exp_name=self.name,
                output_dir=str(output_dir),
                modules_config=str(root / "configs" / "modules" / "yoloxs_obb.yaml"),
                losses_config=str(root / "configs" / "losses" / "yolox_losses_obb.yaml"),
                data_dir=str(data_dir),
                side=config.detector.image_size,
                seed=config.metrics.seed,
                epochs=epochs,
                warmup_epochs=min(1, max(0, epochs - 1)),
                no_aug_epochs=min(3, max(0, epochs - 1)),
                confidence=config.detector.confidence_threshold,
                nms_iou=config.detector.nms_iou,
            ),
            encoding="utf-8",
        )
        return path

    # --- inferencia -------------------------------------------------------

    def predict(self, samples, *, weights: Path, config: Config) -> dict:
        """`sample_id -> [Prediction]` normalizadas.

        Su `obbpostprocess` devuelve `[x1 y1 ... x4 y4, obj, cls_conf, cls]` en
        pixeles de la entrada, con el mismo letterbox arriba-izquierda que
        YOLOX: se deshace dividiendo por `r`, sin desplazamiento.
        """
        import cv2
        import numpy as np
        import torch

        root = find_fork(self._fork_root)
        modules = _import_fork(root)
        from testbank.dataio.image_sizes import SizeIndex
        from testbank.geometry.quad import Quad, canonicalize
        from testbank.metrics.core import Prediction

        samples = list(samples)
        sizes = SizeIndex.for_samples(samples, cache_path=config.data.derived_dir / "image_sizes.json")
        side = config.detector.image_size
        model = self._load(modules, weights, root=root, config=config)

        out: dict[str, list] = {}
        with torch.no_grad():
            for sample in samples:
                image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise DetectorError(f"no se pudo leer {sample.image_path}")
                tensor, ratio = modules["preproc"](image, (side, side))
                outputs = model(torch.from_numpy(tensor).unsqueeze(0))
                detections = modules["obbpostprocess"](
                    outputs, 1, conf_thre=config.detector.confidence_threshold,
                    nms_thre=config.detector.nms_iou,
                )[0]
                width, height = sizes.size(sample.sample_id)
                predictions = []
                if detections is not None:
                    for row in detections.cpu().numpy():
                        corners = np.asarray(row[:8], dtype=float).reshape(4, 2) / ratio
                        quad = Quad.from_xy([(x / width, y / height) for x, y in corners])
                        predictions.append(
                            Prediction(
                                quad=canonicalize(quad, aspect=width / height),
                                score=float(row[8] * row[9]),
                                class_id=int(row[10]),
                            )
                        )
                out[sample.sample_id] = predictions
        return out

    def _load(self, modules, weights: Path, *, root: Path, config: Config):
        import torch

        exp_path = self._write_exp(
            config, root=root, output_dir=Path(weights).parent, data_dir=Path(weights).parent
        )
        exp = modules["get_exp"](str(exp_path), None)
        model = exp.get_model()
        checkpoint = torch.load(str(weights), map_location="cpu")
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state, strict=False)
        model.eval()
        return model


def _find_img_split() -> Path:
    """`tools/img_split.py` de BboxToolkit: no se instala con el paquete."""
    try:
        import BboxToolkit
    except ImportError as exc:
        raise DetectorError(
            "BboxToolkit no esta instalado en este entorno. Es Python puro: "
            "`uv pip install -e BboxToolkit` sobre el clon de jbwang1997/BboxToolkit"
        ) from exc
    candidate = Path(BboxToolkit.__file__).resolve().parents[1] / "tools" / "img_split.py"
    if not candidate.is_file():
        raise DetectorError(f"no se encuentra tools/img_split.py junto a BboxToolkit ({candidate})")
    return candidate


__all__ = ["ENV_VAR", "WEIGHTS_ENV_VAR", "YoloxObbDdgrcfDetector", "find_fork"]
