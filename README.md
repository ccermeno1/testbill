# PP-YOLOE-R Banknote Detector (PyTorch puro)

Detección de billetes de euro con *oriented bounding boxes* usando **PP-YOLOE-R**, reimplementado
en **PyTorch puro**. El mismo código corre en **CPU, CUDA y Apple MPS**: no depende de Paddle,
de PaddleDetection ni de ninguna operación compilada.

Es el port de la implementación con Paddle que vive en la rama `feature/paddle-ppyoloe-r`.
La equivalencia está verificada numéricamente (ver *Paridad con Paddle*).

## Estructura

```text
data/                                   datasets (no versionados)
  banknotes_obb/                          images/ + annotations/*.json (COCO con polígonos)
  billetesprueba/                         test externo de fotos reales
models/ppyoloe_r/                       checkpoints .pt
src/ppyoloer_mps/
  ppyoloe_obb/
    boxes.py        geometría OBB: rbox<->polígono, ProbIoU, IoU rotada exacta
    ops.py          NMS rotado y post-proceso
    model.py        CSPResNet + CustomCSPPAN + PPYOLOERHead
    assigner.py     RotatedTaskAlignedAssigner
    losses.py       VariFocal + ProbIoU + DFL
    data.py         dataset COCO-polígonos, preproceso y augmentación
    engine.py       dispositivo, EMA, bucles de train/eval
    evaluation.py   mAP50/75/50-95, P/R/F1
    checkpoint.py   conversión .pdparams -> .pt, guardado y carga
  train.py  evaluate.py  infer.py
  convert_paddle.py                     .pdparams de PaddleDetection -> .pt
  export_onnx.py  onnx_example.py       export a ONNX y pre/post-proceso del cliente
  data_prep/                            preparacion de datasets (sin Paddle)
tests/                                  tests de runtime (pytest)
```

## Entorno

```bash
uv venv .venv --python 3.11
uv sync --extra test --extra export      # instala desde uv.lock
# o, sin lock:  uv pip install -e ".[test,export]"
# PyTorch según plataforma:
#   macOS (MPS) y CPU:  el wheel por defecto ya vale
#   CUDA:               uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
uv run pytest
```

`--device auto` elige CUDA, luego MPS, luego CPU. En macOS usa `--workers 0` (los procesos de
OpenCV no son estables en los workers) y, si alguna operación no está implementada en MPS,
`export PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Uso

```bash
# predicción sobre fotos sueltas o una carpeta
uv run python src/ppyoloer_mps/infer.py -w models/ppyoloe_r/ppyoloe_r_s_banknotes_extra.pt \
    --images fotos/ --out-dir runs/pred --vis --device auto

# evaluación (mAP50/75/50-95 + P/R/F1)
uv run python src/ppyoloer_mps/evaluate.py -w models/ppyoloe_r/ppyoloe_r_s_banknotes_extra.pt \
    --data data/banknotes_obb --split valid test --device auto

# entrenamiento (la receta del modelo `extra`)
uv run python src/ppyoloer_mps/train.py \
    --data data/banknotes_obb --train-split train_plus_extra --val-split valid \
    --work-dir runs/extra --epochs 60 --mosaic-epochs 50 --batch 4 --device auto --workers 0
```

Opciones útiles de `train.py`: `--init` (pesos de partida, p. ej. el checkpoint de DOTA convertido;
las capas de clasificación con otro número de clases se descartan automáticamente), `--resume`,
`--img-size`, `--lr`, `--no-ema`, `--val-interval`, `--nms-iou`.

## Datos

El formato es COCO con un polígono de 4 puntos por caja en `segmentation`, idéntico al de la rama
de Paddle, así que los JSON de `data_manifests/annotations/` sirven tal cual:

```text
data/banknotes_obb/
  images/<stem>.jpg
  annotations/train.json  valid.json  test.json  train_plus_extra.json  ...
```

## Checkpoints

| fichero | origen |
|---|---|
| `ppyoloe_r_s_banknotes_torch.pt` | **entrenado con este repo** (60 epocas, 95.78 / 81.19 / 69.53) |
| `ppyoloe_r_s_banknotes_extra.pt` | el modelo `extra` entrenado con Paddle, convertido |
| `ppyoloe_r_s_dota.pt` | checkpoint oficial de DOTA, punto de partida para entrenar |

Para convertir otro checkpoint de PaddleDetection:

```bash
uv run python src/ppyoloer_mps/convert_paddle.py \
    --src model_final.pdparams --out models/ppyoloe_r/otro.pt --num-classes 1 --classes euro_banknote
```

Solo ese script necesita Paddle instalado (`uv pip install -e ".[convert]"`), y únicamente si se
le pasa un `.pdparams`; también acepta un `.npz` exportado en otra máquina.

## Paridad con Paddle

### Inferencia

Mismo checkpoint y mismas imagenes, en CPU:

| comprobacion | resultado |
|---|---|
| carga del state_dict | 461 tensores, 0 faltantes, 0 sobrantes |
| mapas del cuello | error max. 6e-5 (escala ~12) |
| scores de la cabeza | error max. 3.8e-6 |
| cajas decodificadas | error max. 3e-3 px sobre ~480 px |
| perdida de entrenamiento (cls/iou/dfl) | diferencia relativa < 8e-6, tambien con lotes de mosaico |
| 20 pasos de SGD sobre el mismo lote | trayectorias identicas (2.4799 -> 0.6123 vs 0.6234) |

### Entrenamiento desde cero

60 epocas desde el checkpoint de DOTA, misma receta y misma metrica, evaluado en valid:

| | port PyTorch | PaddleDetection |
|---|---|---|
| mAP50 | **95.78** | 95.10 |
| mAP75 | **81.19** | 80.94 |
| mAP50-95 | 69.53 | 69.53 |
| P / R / F1 @.5 | 0.969 / 0.874 / 0.919 | 0.969 / 0.874 / 0.919 |

Llegar hasta aqui exigio seis correcciones de fidelidad respecto a ppdet, todas con test de
regresion. Se documentan porque son justo las que no se ven comparando modulos por separado:

| # | diferencia | sintoma si no se corrige |
|---|---|---|
| 1 | `RRotate` usa `auto_bound`: encoge la imagen en vez de recortar esquinas | objetos cortados con su caja entera como objetivo |
| 2 | `Poly2RBox` descarta cajas con lado menor < 2 px | ProbIoU hace `log(0)` y la perdida se va a **NaN** |
| 3 | en `RandomDistort` el parametro `prob` es la probabilidad de **saltar** la operacion, y usa PIL `ImageEnhance` | imagenes lavadas (media 127 vs 107, desviacion 54 vs 74) |
| 4 | las rotaciones rellenan con negro, no con gris 114 | mismas estadisticas de imagen desviadas |
| 5 | `ModelEMA` usa `ema_decay_type='threshold'`, no el ramp exponencial | decay 0.34 en vez de 0.99: el EMA copia el modelo en vez de promediarlo |
| 6 | **`RResize` recorta los vertices del poligono al lienzo** | se entrena con la extension no visible del objeto: **mAP75 hundido** (10 vs 40) |

La numero 6 era la principal. Se localizo midiendo el pipeline de Paddle etapa por etapa
(mosaico 0.2348 -> +rotaciones 0.1717 -> +RResize 0.1202 de area mediana por imagen).

### Umbrales, alineados con mmrotate

Los valores por defecto son los del `test_cfg` oficial de mmrotate (`rotated_retinanet`), para
poder comparar con otros detectores rotados en igualdad de condiciones:

| parametro | valor | donde |
|---|---|---|
| `score_thr` (antes del NMS) | 0.05 | `--score-threshold` |
| `nms_iou` | 0.1 | `--nms-iou` |
| `nms_pre` / `max_per_img` | 2000 | `ops.batched_postprocess` |
| umbral para P/R/F1 y produccion | 0.5 | `--conf` |
| seleccion del mejor checkpoint | `0.9*mAP50-95 + 0.1*mAP50` | `--save-best` |

Con billetes apilados conviene subir el NMS a 0.5: el 17 % de las cajas de valid solapan con un
vecino por encima de IoU 0.1. Medido sobre el modelo entrenado con este repo:

| dataset | score 0.05 / NMS 0.1 (mmrotate) | score 0.01 / NMS 0.5 |
|---|---|---|
| valid | 90.54 / 80.99 / 67.76 | **95.78 / 81.19 / 69.53** |
| test | 100.00 / 87.48 / 77.46 | **100.00 / 89.42 / 78.12** |
| billetesprueba (externo) | 98.34 / 86.02 / 68.64 | **98.69 / 86.00 / 68.78** |

(mAP50 / mAP75 / mAP50-95.) Los valores por defecto son los de mmrotate para poder comparar con
otros detectores; para desplegar, `--nms-iou 0.5`.

## Export a ONNX

```bash
uv pip install -e ".[export]"
uv run python src/ppyoloer_mps/export_onnx.py models/ppyoloe_r/ppyoloe_r_s_banknotes_torch.pt     --out models/ppyoloe_r/ppyoloe_r_s_banknotes_640.onnx --img-size 640
```

El grafo acepta **RGB float32 en 0..255** ya redimensionado y rellenado (la normalizacion va
dentro) y devuelve `scores` (B, C, L) con sigmoid aplicada y `boxes` (B, L, 5) ya decodificadas
en pixeles de la entrada. Al cliente solo le quedan tres pasos: umbral, NMS rotado y deshacer la
escala. Junto al `.onnx` se escribe un `.metadata.json` con todo el contrato.

`onnx_example.py` implementa esos tres pasos en numpy y los contrasta con el modelo PyTorch:

```bash
uv run python src/ppyoloer_mps/onnx_example.py foto.jpg     --onnx models/ppyoloe_r/ppyoloe_r_s_banknotes_640.onnx     --checkpoint models/ppyoloe_r/ppyoloe_r_s_banknotes_torch.pt
# ONNX: 2 detecciones (conf >= 0.5, NMS IoU 0.1)
# PyTorch: 2 detecciones
#   diferencia maxima en las cajas: 0.0002 px
#   diferencia maxima en los scores: 0.000000
```

Cuidado con la convencion de esquinas al reimplementar `rbox2poly` fuera de Python: el signo del
termino del angulo da cajas plausibles pero mal orientadas. Hay un test que lo fija.

## Rendimiento y dispositivos

- **macOS / MPS**: es el objetivo de este port; entrenar e inferir funcionan sin Paddle.
- **CUDA**: requiere un driver reciente (los wheels de PyTorch 2.14 piden driver >= 525). En la
  máquina de desarrollo, con el driver 461.33, PyTorch solo ve CPU aunque Paddle sí use la GPU;
  ahí conviene entrenar con la rama de Paddle o actualizar el driver.
- **CPU**: válido para predecir y evaluar (~1 s/imagen a 640 px). Entrenar en CPU es lento
  (~2 s/iteración con batch 2).
