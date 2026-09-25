# YOLOX-OBB-s Banknote Detector

Implementación en PyTorch puro de **YOLOX-OBB-s** ([DDGRCF/YOLOX_OBB](https://github.com/DDGRCF/YOLOX_OBB))
para detección orientada de billetes. Usa la arquitectura y la pérdida de YOLOX_OBB con el
runtime de entrenamiento del repo RTMDet-R (`testbill`, rama `feature/rt_refactor`). Funciona
en CPU, CUDA y Apple MPS, sin operadores compilados, BboxToolkit, mmcv ni YAML.

## Qué se ha tomado de cada repo

| Pieza | Origen |
| --- | --- |
| Red (backbone YOLOv5-s con ReLU, PAN, head desacoplado) | `configs/modules/yoloxs_obb.yaml` |
| Asignación de etiquetas SimOTA (centro en caja rotada + radio 2.5, dynamic-k) | `OBBDetectX.get_assignments` |
| Pérdidas: BCE obj + BCE cls (IoU como objetivo) + 5 x PolyIoU + L1 extra | `configs/losses/yolox_losses_obb.yaml` |
| Optimizador SGD nesterov, `yoloxwarmcos`, EMA 0.9998, lr 0.01/64 por imagen | `yolox_obb_base.py`, `trainer.py` |
| Post-proceso: score = obj x cls, NMS rotado por clase | `obbpostprocess` |
| Dataset YOLOv8-OBB, aumentaciones (mosaic, mixup, rotación, HSV, flip, stage 2) | testbill |
| Bucle de entrenamiento, `log.jsonl`, TensorBoard, `best_epoch_N.pth` por fitness | testbill |
| Evaluación mAP rotado, inferencia, exportación ONNX, tests | testbill |
| IoU rotado diferenciable, IoU por pares y NMS en PyTorch puro | testbill |

YOLOX_OBB solo publica la variante **s** para OBB (8.06 M parámetros), así que es la única que
se incluye. Los nombres de los pesos son los mismos que en el repo original (`model.0.conv.weight`
... `model.33.cls_preds.2.bias`), y el checkpoint de DOTA carga directamente.

Las últimas `--stage2-epochs` épocas son la fase "no aug" de YOLOX: se activa la pérdida L1,
la LR se queda en su mínimo y, con `--strong-aug`, se cambia al pipeline ligero de testbill.

Diferencias deliberadas con el YOLOX_OBB original:

* Las aumentaciones son las de testbill, no las de YOLOX (copy-paste y afinidad aleatoria).
  Tampoco hay entrenamiento multiescala.
* El ángulo de las cajas va en sentido horario, como en testbill. La red conserva la semántica
  original (antihoraria) y el signo se invierte al decodificar, así que los pesos de DOTA siguen valiendo.
* En el original, el objetivo L1 del ángulo se queda a 0 (solo se rellenan x, y, w, h). Aquí
  se usa el ángulo real del GT.

## Arquitecturas (`--model`)

| `--model` | Red | Parámetros | Pesos iniciales (`--init` por defecto) |
| --- | --- | --- | --- |
| `ddgrcf_s` (defecto) | YOLOX-OBB-s de DDGRCF (YOLOv5-s, ReLU) | 8,06 M | `yolox_s_dota1_0.pth` (DOTA, OBB) |
| `yolox_nano` | YOLOX oficial nano (CSPDarknet depthwise, SiLU) | 0,90 M | `yolox_nano.pth` (COCO) |
| `yolox_tiny` | YOLOX oficial tiny | 5,1 M | `yolox_tiny.pth` (COCO) |
| `yolox_s` | YOLOX oficial s | 9,0 M | `yolox_s.pth` (COCO) |

Las variantes oficiales (`yolox_obb/official.py`) son CSPDarknet + PAFPN + `YOLOXHead` de
[Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) con los mismos
nombres de pesos. El único cambio es `reg_preds`, que tiene 5 canales (x, y, w, h, ángulo): al
cargar los pesos de COCO se conservan los 4 canales xywh preentrenados y el del ángulo empieza
en 0 (cajas horizontales). Los pesos COCO de la release `0.1.1rc0` (entrada BGR 0..255 sin
normalizar) se descargan de GitHub:

```bash
curl -L -o models/yolox_obb/checkpoints/yolox_nano.pth \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
```

Todas comparten entrenamiento, pérdidas, SimOTA y post-proceso (`OBBDetector` en `model.py`).
La arquitectura se guarda en cada checkpoint, así que `evaluate.py`, `infer.py` y
`export_onnx.py` la detectan solas (los checkpoints antiguos sin ese dato son `ddgrcf_s`).

## Estructura

```text
data/
  dataset/                  export YOLOv8-OBB con train/, valid/, test/
  augmented/                aumentaciones offline (opcional)
  external_dataset/         evaluación externa: images/ + labels/

models/yolox_obb/
  checkpoints/              pesos iniciales (p. ej. yolox_s_dota1_0.pth)
  experiments/              un directorio por ejecución

src/yolox_mps/              scripts de entrenamiento, evaluación, inferencia y exportación
  yolox_obb/                modelo, SimOTA, pérdidas, datos, motor de entrenamiento
tests/                      tests de runtime
```

## Entorno

El entorno se gestiona con `uv` (Python 3.11):

```bash
uv sync --extra export --extra test
# con TensorBoard:
uv sync --extra export --extra test --extra monitoring
```

En macOS se usa Pillow para redimensionar (`YOLOX_RESIZE_BACKEND=pil`, el valor por defecto
allí) y conviene `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Pesos iniciales

El model zoo de YOLOX_OBB está en Baidu Pan: <https://pan.baidu.com/s/1k1k1JCq56Z-g9NrRtHNWhQ>
(código `tdm6`). Descarga `YOLOX_s_dota1_0` y guárdalo como:

```text
models/yolox_obb/checkpoints/yolox_s_dota1_0.pth
```

Al cargarlo con 1 clase solo se descartan las 6 capas `cls_preds` (15 clases en DOTA). El
resto de la red, incluida la regresión, se reutiliza. Sin esos pesos, `--init none` entrena
desde cero, que con pocos cientos de imágenes da resultados claramente peores.

## Entrenamiento

`--device auto` elige CUDA, luego MPS y luego CPU. `train.py` usa `data/dataset` por defecto,
resuelve `--init` en `models/yolox_obb/checkpoints` y escribe en `models/yolox_obb/experiments/<work-dir>`.

### Export de Roboflow + split congelado `v1`

Con el export completo y el split congelado (`train.txt`, `valid.txt`, `test.txt` con un id
por línea; el hash `.rf.XXX` se ignora al comparar):

```powershell
uv run python src/yolox_mps/train.py `
  --data "C:\Users\Clara\OneDrive\Escritorio\paddletest\Annotated banknotes 2.yolov8-obb" `
  --split-dir "C:\Users\Clara\OneDrive\Escritorio\paddletest\v1" `
  --extra-train "C:\Users\Clara\OneDrive\Escritorio\paddletest\augmented" `
  --init yolox_s_dota1_0.pth --work-dir v1_200 `
  --img-size 640 --batch 8 --epochs 200 --stage2-epochs 15 --strong-aug --val-interval 5
```

Resultado: 355 + 710 imágenes de train (las 710 de `augmented` salen solo de train de `v1`;
comprobado con su `manifest.json`), 100 de valid y 47 de test.

Validación/test con el mismo split, y generalización en el conjunto externo `billetesprueba 2`
(143 fotos, 59 en HEIC, que se leen con `pillow-heif`):

```powershell
$P = "C:\Users\Clara\OneDrive\Escritorio\paddletest"
uv run python src/yolox_mps/evaluate.py v1_200/best_epoch_200.pth `
  --data "$P\Annotated banknotes 2.yolov8-obb" --split-dir "$P\v1" --split test --img-size 640
uv run python src/yolox_mps/evaluate.py v1_200/best_epoch_200.pth `
  --data "$P\billetesprueba 2.yolov8-obb" --split train --img-size 800 `
  --tensorboard-logdir models/yolox_obb/experiments/v1_200/tensorboard --tensorboard-tag external
```

### CUDA (Windows / Linux)

```powershell
uv run python src/yolox_mps/train.py `
  --init yolox_s_dota1_0.pth `
  --work-dir dota_200 `
  --extra-train augmented `
  --img-size 640 `
  --batch 8 `
  --epochs 200 `
  --stage2-epochs 15 `
  --strong-aug `
  --mosaic-prob 1 `
  --mixup-prob 0.5 `
  --device cuda `
  --workers 4 `
  --val-interval 5 `
  --tensorboard
```

### Apple MPS

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python src/yolox_mps/train.py \
  --init yolox_s_dota1_0.pth --work-dir dota_200 --extra-train augmented \
  --img-size 640 --batch 4 --epochs 200 --stage2-epochs 15 \
  --strong-aug --mosaic-prob 1 --mixup-prob 0.5 \
  --device mps --workers 0 --val-interval 5 --seed 0
```

En macOS hay que usar `--workers 0` con este pipeline de aumentación.

### CPU

```bash
uv run python src/yolox_mps/train.py \
  --init yolox_s_dota1_0.pth --work-dir cpu_run \
  --img-size 512 --batch 2 --epochs 100 --stage2-epochs 10 --strong-aug \
  --device cpu --workers 0
```

La LR por defecto es `0.01 / 64 * batch * accumulate` (la `basic_lr_per_img` de YOLOX). Con
`--lr` se fija a mano. Cada ejecución escribe `args.json`, `log.jsonl`, `epoch_N.pth`,
`latest.pth` y `best_epoch_N.pth`. El mejor checkpoint se elige en validación con:

```text
fitness = 0.9 * mAP@.5:.95 + 0.1 * mAP@0.50
```

Para reanudar, repite el comando con `--resume dota_200/latest.pth`. TensorBoard:

```bash
uv run --extra monitoring tensorboard --logdir models/yolox_obb/experiments
```

## Evaluación

Protocolo de testbill: AP con score >= 0.05 y NMS rotado con IoU 0.1. Precisión, recall y F1
se miden en el umbral operativo 0.5.

```bash
uv run python src/yolox_mps/evaluate.py dota_200/best_epoch_200.pth \
  --data dataset --split test --img-size 640 --batch 2 --workers 0 \
  --score-thr 0.05 --report-score-thr 0.5 --nms-iou 0.1
```

Para un conjunto externo plano (`images/` + `labels/`): `--data external_dataset --split train`.
Con `--tensorboard-logdir models/yolox_obb/experiments/dota_200/tensorboard --tensorboard-tag external`
las métricas se añaden al mismo panel.

## Inferencia

Genera una imagen anotada por entrada y `predictions.json`. El GT se dibuja en rojo y las
predicciones en verde.

```bash
uv run python src/yolox_mps/infer.py dota_200/best_epoch_200.pth external_dataset/images \
  --out inference_800 --img-size 800 --score-thr 0.5 --nms-iou 0.3 \
  --gt external_dataset/labels
```

Acepta JPG, PNG y HEIC (con `pillow-heif`).

## Exportación

```bash
# solo pesos de inferencia (EMA) + metadatos
uv run python src/yolox_mps/export_checkpoint.py \
  models/yolox_obb/experiments/v1_gpu100/best_epoch_100.pth models/yolox_obb/checkpoints/banknotes_obb.pth

# ONNX (un solo archivo) + contrato .metadata.json, junto al checkpoint
uv run python src/yolox_mps/export_onnx.py v1_gpu100/best_epoch_100.pth --out model_640.onnx --img-size 640
```

La decodificación va dentro del grafo, así que el cliente solo hace lo que queda fuera:

| Paso | Qué hace |
| --- | --- |
| Preprocesado | Foto con orientación EXIF aplicada → `scale = S / max(h, w)` → resize a `(round(w·scale), round(h·scale))` (INTER_AREA al reducir) → pegar arriba a la izquierda en un lienzo `S×S` de 114 → NCHW float32 BGR **0..255, sin normalizar** |
| Red (`images` → `boxes`, `scores`) | `boxes (1, N, 5)`: cx, cy, w, h en píxeles de la entrada, ángulo en radianes (horario). `scores (1, N, C)`: sigmoid(obj) · sigmoid(cls) |
| Postprocesado | `label = argmax`, `score = max` por fila → `score > score_thr` → NMS rotado por clase (IoU > `nms_iou`) → `cx, cy, w, h /= scale` → esquinas `c ± (w/2)(cos a, sin a) ± (h/2)(−sin a, cos a)` |

`N = (S/8)² + (S/16)² + (S/32)²` (8400 a 640 px). El `.metadata.json` recoge este contrato,
las clases y los umbrales recomendados (0,5 / 0,3).

`src/yolox_mps/onnx_example.py` es la implementación de referencia para la app: usa solo
`onnxruntime`, `numpy` y OpenCV (NMS rotado con `cv2.rotatedRectangleIntersection`), lee JPG y
HEIC y dibuja los resultados. Con `--check` compara cada foto con el modelo de PyTorch:

```bash
uv run python src/yolox_mps/onnx_example.py models/yolox_obb/experiments/v1_gpu100/model_640.onnx \
  fotos/ --out onnx_vis --check models/yolox_obb/experiments/v1_gpu100/best_epoch_100.pth
```

## Tests

```bash
uv run pytest -q
```

Incluyen forward y shapes, que las claves y shapes de los pesos sigan el layout de
`yoloxs_obb.yaml`, pérdida y backward, SimOTA, mAP, datos y exportación ONNX. Si
`models/yolox_obb/checkpoints/yolox_s_dota1_0.pth` existe, también se prueba su carga.
