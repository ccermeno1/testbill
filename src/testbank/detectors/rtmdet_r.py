"""Adaptador de RTMDet-R (MMRotate). Apache-2.0 y apto para produccion.

ESTE ES EL UNICO MODULO DEL PROYECTO QUE PUEDE IMPORTAR `mmrotate`, `mmdet` o
`mmcv`. `tests/test_detectors.py` lo verifica sobre el arbol de fuentes, igual
que con `ultralytics`.

No corre en el entorno principal
--------------------------------
Necesita torch 2.0 y una cadena de versiones muy estrecha que ningun resolutor
declara. La receta verificada esta en el README; el grupo opcional `rtmdet` del
pyproject la lleva clavada. Aqui el import es PEREZOSO y el mensaje de error
manda al sitio correcto en vez de dejar un ImportError pelado.

Punto de partida
----------------
`rotated_rtmdet_tiny-3x-dota`: un detector rotado YA ENTRENADO en DOTA, no un
preentreno de clasificacion. La especificacion pedia preentreno COCO, que para
la variante tiny no esta publicado -- ver la desviacion documentada en el README.
"""

from __future__ import annotations

from pathlib import Path

from testbank.config import Config
from testbank.dataio.formats import ImageSize
from testbank.dataio.image_sizes import SizeIndex
from testbank.detectors.base import (
    BaseDetector,
    DetectorError,
    TrainResult,
    register,
)

#: Config de MMRotate del que se parte. `tiny` por el despliegue movil.
DEFAULT_CONFIG = "rotated_rtmdet_tiny-3x-dota"

#: Pesos publicados: detector rotado entrenado en DOTA, mAP 75.60.
DEFAULT_CHECKPOINT = (
    "https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/"
    "rotated_rtmdet_tiny-3x-dota/rotated_rtmdet_tiny-3x-dota-9d821076.pth"
)

_INSTALL_HINT = (
    "RTMDet-R necesita mmrotate, mmdet y mmcv, que NO caben en el entorno "
    "principal: exigen torch 2.0 y una cadena de versiones muy estrecha "
    "(mmrotate 1.x -> mmdet <3.2 -> mmcv <2.1 -> indice de torch 2.0). "
    "La receta verificada esta en el README, seccion 'Entorno para RTMDet-R'."
)


def _import_mmrotate():
    """Importa y REGISTRA los modulos de MMRotate.

    `register_all_modules(init_default_scope=True)` no es opcional, y el flag
    tampoco. El modelo se declara como `mmdet.RTMDet`, asi que al construirlo
    mmengine salta al registro de mmdet -- donde `RotatedRTMDetSepBNHead` no
    esta, porque vive en mmrotate. Con `init_default_scope=False` falla con
    `RotatedRTMDetSepBNHead is not in the mmdet::model registry`, que no dice
    nada del ambito y manda a buscar en el sitio equivocado.

    El Runner lo hace por su cuenta desde `default_scope` de la config; cualquier
    uso manual del registro, no.
    """
    try:
        from mmdet.apis import inference_detector, init_detector
        from mmrotate.utils import register_all_modules
    except ImportError as exc:
        raise DetectorError(f"{_INSTALL_HINT} ({exc})") from exc
    register_all_modules(init_default_scope=True)
    return init_detector, inference_detector


@register
class RtmdetRDetector(BaseDetector):
    name = "rtmdet-r-tiny"
    license = "Apache-2.0"
    production_ready = True
    _NOTE = (
        "Parte de rotated_rtmdet_tiny-3x-dota: detector rotado ya entrenado en "
        "DOTA. El preentreno COCO que pedia la especificacion no esta publicado "
        "para tiny; ver la desviacion en el README."
    )
    _ENV_NOTE = (
        "Corre en un entorno APARTE con torch 2.0. Sus numeros no son "
        "estrictamente comparables con los de candidatos en torch 2.14."
    )
    notes = (_NOTE, _ENV_NOTE)

    def __init__(
        self,
        config_name: str = DEFAULT_CONFIG,
        checkpoint: str = DEFAULT_CHECKPOINT,
    ) -> None:
        self.config_name = config_name
        self.checkpoint = checkpoint

    # --- entrenamiento ----------------------------------------------------

    def train(self, samples_by_split, config: Config, *, output_dir: Path) -> TrainResult:
        """Ajusta sobre la vista DOTA que exportamos.

        MMRotate lee un arbol `images/` + `labelTxt/`, que es exactamente lo que
        escribe nuestro exportador `dota`. Asi comparte filtro de area y politica
        de borde con los demas candidatos.
        """
        _import_mmrotate()
        from mmengine.runner import Runner

        from testbank.dataio.export import export

        view = export("dota", samples_by_split, config, out_dir=output_dir / "dota")
        cfg = build_train_config(view.root, config, output_dir=output_dir)
        Runner.from_cfg(cfg).train()

        weights = output_dir / "work" / "last_checkpoint"
        resolved = _resolve_checkpoint(output_dir / "work")
        if resolved is None:
            raise DetectorError(
                f"el entrenamiento no dejo pesos en {weights.parent}"
            )
        return TrainResult(
            weights=resolved,
            epochs=config.detector.epochs,
            notes=(view.describe(), f"partiendo de {self.checkpoint.rsplit('/', 1)[-1]}"),
        )

    # --- inferencia -------------------------------------------------------

    def predict(self, samples, *, weights: Path, config: Config) -> dict:
        """`sample_id -> [Prediction]` en coordenadas NORMALIZADAS.

        MMRotate devuelve `cx, cy, w, h, theta` en pixeles de la imagen, asi que
        la conversion pasa por el mismo `boxes_to_quads` que el candidato propio:
        una sola implementacion del paso a quad canonico para los dos.
        """
        init_detector, inference_detector = _import_mmrotate()
        import torch

        from testbank.metrics.core import Prediction
        from testbank.models.decode import boxes_to_quads

        samples = list(samples)
        sizes = SizeIndex.for_samples(
            samples, cache_path=config.data.derived_dir / "image_sizes.json"
        )
        model = init_detector(self.config_name, str(weights), device="cpu")

        out: dict[str, list] = {}
        for sample in samples:
            result = inference_detector(model, str(sample.image_path))
            instances = result.pred_instances
            keep = instances.scores >= config.detector.confidence_threshold
            boxes = instances.bboxes[keep].cpu()
            scores = instances.scores[keep].cpu()

            width, height = sizes.size(sample.sample_id)
            size = ImageSize(int(width), int(height))
            if boxes.numel() == 0:
                out[sample.sample_id] = []
                continue
            quads = boxes_to_quads(torch.as_tensor(boxes, dtype=torch.float32), size)
            out[sample.sample_id] = [
                Prediction(quad=quad, score=float(score), class_id=0)
                for quad, score in zip(quads, scores)
            ]
        return out


__all__ = ["DEFAULT_CHECKPOINT", "DEFAULT_CONFIG", "RtmdetRDetector"]


# --- config de MMRotate ----------------------------------------------------


def _resolve_checkpoint(work_dir: Path) -> Path | None:
    """El ultimo checkpoint que dejo el Runner.

    Se busca en vez de reconstruir el nombre: MMEngine lo decide segun el
    planificador y el intervalo de guardado, y adivinarlo ya fallo una vez con
    Ultralytics. `last_checkpoint` es un fichero de texto con la ruta dentro.
    """
    pointer = work_dir / "last_checkpoint"
    if pointer.is_file():
        candidate = Path(pointer.read_text(encoding="utf-8").strip())
        if candidate.exists():
            return candidate
    found = sorted(work_dir.glob("*.pth"))
    return found[-1] if found else None


def build_train_config(dota_root: Path, config: Config, *, output_dir: Path):
    """Config de MMRotate para UNA clase sobre nuestra vista DOTA.

    Se parte de la config publicada y solo se sobrescribe lo que cambia. Copiar
    la config entera aqui la dejaria desincronizada del paquete en cuanto este
    se actualice, y ademas son 200 lineas que no aportan nada.

    Tres cosas que hay que tocar y no son evidentes:

    1. `ann_file` apunta a `labelTxt/`. MMRotate usa `annfiles/` por defecto,
       pero `labelTxt/` es el nombre del convenio DOTA y es el que escribe
       nuestro exportador. Cambiar el exportador para complacer a MMRotate
       romperia a cualquier otro consumidor de DOTA.
    2. `metainfo` con una sola clase. Sin esto, `DOTADataset` espera las 15
       clases de DOTA y las anotaciones de `euro_banknote` no casan con ninguna.
    3. `load_from`, no `init_cfg`. Queremos partir del DETECTOR entrenado en
       DOTA, no del backbone preentrenado en ImageNet que trae la config.
    """
    from mmengine.config import Config as MMConfig
    from mmengine.utils import get_installed_path

    package = Path(get_installed_path("mmrotate"))
    base = package / ".mim" / "configs" / "rotated_rtmdet"
    if not base.exists():  # instalado desde git sin .mim
        base = package.parent / "configs" / "rotated_rtmdet"
    cfg = MMConfig.fromfile(str(base / f"{DEFAULT_CONFIG}.py"))

    classes = ("euro_banknote",)
    cfg.work_dir = str((output_dir / "work").resolve())
    cfg.load_from = DEFAULT_CHECKPOINT
    cfg.resume = False
    cfg.randomness = {"seed": config.metrics.seed, "deterministic": True}

    for loader, split in (("train_dataloader", "train"), ("val_dataloader", "valid")):
        section = cfg.get(loader)
        if section is None:
            continue
        section["batch_size"] = config.detector.batch_size
        # Sin workers: con `num_workers > 0` el orden de los lotes depende de la
        # planificacion del sistema y dos ejecuciones con la misma semilla dejan
        # de coincidir. Es la misma decision que en el candidato propio.
        section["num_workers"] = 0
        dataset = section["dataset"]
        dataset["data_root"] = str((dota_root / split).resolve()) + "/"
        dataset["ann_file"] = "labelTxt/"
        dataset["data_prefix"] = {"img_path": "images/"}
        dataset["metainfo"] = {"classes": classes}

    if cfg.get("test_dataloader") is not None:
        # El test esta SELLADO. Se apunta a valid para que MMRotate no intente
        # abrir un directorio que no existe, pero nada de aqui lo evalua: eso
        # solo lo hace `evaluate-test`, con su registro de accesos.
        cfg.test_dataloader = cfg.val_dataloader

    cfg.model["bbox_head"]["num_classes"] = len(classes)
    cfg.train_cfg = {
        "type": "EpochBasedTrainLoop",
        "max_epochs": config.detector.epochs,
        "val_interval": max(1, config.detector.epochs),
    }
    return cfg
