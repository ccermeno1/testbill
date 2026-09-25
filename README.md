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
  train.py  evaluate.py  infer.py  convert_paddle.py
tests/                                  tests de runtime (pytest)
```

## Entorno

```bash
uv venv .venv --python 3.11
uv pip install -e ".[test]"
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

`models/ppyoloe_r/ppyoloe_r_s_banknotes_extra.pt` es el modelo recomendado (el `extra` de la rama
de Paddle), ya convertido. Para convertir otro checkpoint de PaddleDetection:

```bash
uv run python src/ppyoloer_mps/convert_paddle.py \
    --src model_final.pdparams --out models/ppyoloe_r/otro.pt --num-classes 1 --classes euro_banknote
```

Solo ese script necesita Paddle instalado (`uv pip install -e ".[convert]"`), y únicamente si se
le pasa un `.pdparams`; también acepta un `.npz` exportado en otra máquina.

## Paridad con Paddle

Comprobado en CPU con el mismo checkpoint y las mismas imágenes:

| comprobación | resultado |
|---|---|
| carga del state_dict | 461 tensores, 0 faltantes, 0 sobrantes |
| mapas del cuello | error máx. 6e-5 (escala ~12) |
| scores de la cabeza | error máx. 3.8e-6 |
| cajas decodificadas | error máx. 3e-3 px sobre ~480 px |
| pérdida de entrenamiento (cls/iou/dfl) | diferencia relativa < 8e-6 |

Y las métricas extremo a extremo coinciden con las de Paddle hasta el segundo decimal:

| dataset | mAP50 | mAP75 | mAP50-95 | P@.5 | R@.5 | F1@.5 |
|---|---|---|---|---|---|---|
| valid | 95.10 | 80.94 | 69.53 | 0.969 | 0.874 | 0.919 |
| test | 100.00 | 91.53 | 77.61 | 1.000 | 0.984 | 0.992 |
| billetesprueba | 98.65 | 84.17 | 66.50 | 0.935 | 0.971 | 0.953 |

### Detalles del port

- Los submódulos se llaman igual que en Paddle, así que los pesos mapean 1:1: solo cambian
  `bn._mean`/`bn._variance` por `running_mean`/`running_var`. No hay tensores 2D, luego no hay
  ninguna matriz que transponer.
- `swish` de Paddle == `SiLU`; `hardsigmoid` de Paddle (slope 1/6, offset 0.5) == `F.hardsigmoid`.
- El `Resize` de inferencia usa `INTER_AREA`, como la versión final de la rama de Paddle: al reducir
  fotos grandes evita el *aliasing* de la interpolación cúbica.
- `F.binary_cross_entropy` de PyTorch no admite un `weight` con gradiente, pero el de Paddle sí lo
  propaga y la VariFocal calcula el peso desde la propia predicción: por eso la BCE está escrita a
  mano en `losses.py` (verificado contra Paddle).
- La IoU rotada exacta está en PyTorch puro (`boxes.rotated_iou`), así que no hace falta compilar
  las ops custom de PaddleDetection. El NMS rotado la usa directamente.

## Rendimiento y dispositivos

- **macOS / MPS**: es el objetivo de este port; entrenar e inferir funcionan sin Paddle.
- **CUDA**: requiere un driver reciente (los wheels de PyTorch 2.14 piden driver >= 525). En la
  máquina de desarrollo, con el driver 461.33, PyTorch solo ve CPU aunque Paddle sí use la GPU;
  ahí conviene entrenar con la rama de Paddle o actualizar el driver.
- **CPU**: válido para predecir y evaluar (~1 s/imagen a 640 px). Entrenar en CPU es lento
  (~2 s/iteración con batch 2).
