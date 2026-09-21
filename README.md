# Detección de billetes de euro con OBB — PP-YOLOE-R-s (PaddleDetection)

Fine-tuning del modelo preentrenado **PP-YOLOE-R-s** (checkpoint oficial entrenado en DOTA)
para detectar billetes de euro y predecir sus *oriented bounding boxes*.

## Estructura

```
paddletest/
├── Annotated banknotes 2.yolov8-obb/   export de Roboflow (YOLOv8-OBB), sin tocar
├── v1/                                 splits congelados (train/valid/test.txt + manifest.json)
├── dataset/banknotes_obb/              GENERADO por scripts/prepare_dataset.py
│   ├── images/<stem>.jpg               502 imágenes con nombre limpio (sin hash de Roboflow)
│   ├── annotations/{train,valid,test}.json   COCO con polígonos (formato de PP-YOLOE-R)
│   ├── labelTxt/<stem>.txt             formato DOTA (por si se usa con otras herramientas)
│   └── splits/, classes.txt
├── configs/
│   ├── ppyoloe_r_crn_s_banknotes.yml   config principal (hereda las bases oficiales)
│   └── _base_/                         dataset, reader (640 px, batch 4), optimizador (80 épocas)
├── scripts/
│   ├── prepare_dataset.py              YOLOv8-OBB + splits v1 → COCO/DOTA
│   └── predict_obb.py                  inferencia → txt por imagen (px y YOLO-OBB), JSON, visualización
├── ext_op_fallback/                    rbox_iou / matched_rbox_iou en Paddle puro (ver abajo)
├── PaddleDetection/                    clone release/2.8
└── .venv/                              entorno (uv, Python 3.10, paddlepaddle-gpu 3.0.0 cu118)
```

## Datos y splits

- Los ids de `v1/*.txt` y los ficheros del export llevan hashes de Roboflow **distintos**
  (`005_Euro_013_jpg.rf.<HASH>`), así que el emparejamiento se hace **solo por el nombre base**
  (`005_Euro_013`). Las carpetas train/valid/test del export se ignoran (233 de las 502 imágenes
  cambian de split respecto a Roboflow).
- Resultado: **train 355 / valid 100 / test 47** imágenes (556 / 143 / 63 cajas), 1 clase
  (`euro_banknote`), imágenes 416×416 salvo 13.
- 65 polígonos sobresalen >1 % del borde de la imagen (hasta un 23 %). Se conservan íntegros
  para no deformar el rectángulo rotado; PaddleDetection recorta él mismo el `bbox` axis-aligned.
  Si prefieres recortarlos: `--clip`.

```powershell
.venv\Scripts\python scripts\prepare_dataset.py        # regenera dataset/banknotes_obb
```

## Entrenar / evaluar / predecir

Todo se lanza desde la raíz del proyecto con el Python del entorno (`.venv\Scripts\python`).

```powershell
# entrenamiento (80 épocas, evalúa en valid cada 5, guarda best_model y model_final en output/)
.venv\Scripts\python PaddleDetection\tools\train.py -c configs\ppyoloe_r_crn_s_banknotes.yml --eval

# evaluación en valid / test (mAP@0.5 con IoU rotada)
.venv\Scripts\python PaddleDetection\tools\eval.py -c configs\ppyoloe_r_crn_s_banknotes.yml -o weights=output\model_final.pdparams
.venv\Scripts\python PaddleDetection\tools\eval.py -c configs\ppyoloe_r_crn_s_banknotes_evaltest.yml -o weights=output\model_final.pdparams

# predicción de OBB (split test por defecto; también --images carpeta_o_imagen)
.venv\Scripts\python scripts\predict_obb.py -c configs\ppyoloe_r_crn_s_banknotes.yml -w output\model_final.pdparams --split test --vis
#   -> output/predictions/<stem>.txt        clase score x1 y1 x2 y2 x3 y3 x4 y4 (píxeles)
#   -> output/predictions/yolo_obb/<stem>.txt  formato YOLOv8-OBB normalizado
#   -> output/predictions/predictions.json, output/predictions/vis/*.jpg

# alternativa oficial (dibuja polígonos y guarda bbox.json)
.venv\Scripts\python PaddleDetection\tools\infer.py -c configs\ppyoloe_r_crn_s_banknotes.yml -o weights=output\model_final.pdparams --infer_dir=dataset\banknotes_obb\images --output_dir=output\infer --save_results=True

# exportar a modelo estático para despliegue (Paddle Inference / ONNX)
.venv\Scripts\python PaddleDetection\tools\export_model.py -c configs\ppyoloe_r_crn_s_banknotes.yml -o weights=output\model_final.pdparams
```

Para reanudar: `-r output\<epoch>`. Cambios rápidos con `-o`, p. ej.
`-o TrainReader.batch_size=2 LearningRate.base_lr=0.002` si hubiera OOM.

### Decisiones de la config

| Parámetro | DOTA (oficial) | Aquí | Motivo |
|---|---|---|---|
| `pretrain_weights` | backbone ImageNet | `ppyoloe_r_crn_s_3x_dota.pdparams` | fine-tuning del detector completo; solo `pred_cls` (15→1 clases) se reinicializa |
| resolución | 1024 | 640 | imágenes de 416 px; 1024 no aporta y no cabe en 4 GB |
| batch × GPUs | 2 × 4 = 8 | 4 × 1 | GTX 1650 (4 GB): usa ~1.9 GB |
| `base_lr` | 0.008 | 0.004 | escalado lineal con el batch total |
| épocas | 36 | 80 (coseno hasta 96) | dataset pequeño; ~88 iter/época, ≈40 min en total |
| `worker_num` / shared memory | 4 / sí | 2 / no | Windows |
| augmentaciones | flip + rotaciones 90° + 30/60° | iguales | mismas que el preentrenado |

## Resultados (run del 2026-09-21, 80 épocas, ~1 h en la GTX 1650)

| Pesos | NMS 0.1 (oficial) | NMS 0.5 (config actual) |
|---|---|---|
| valid, `best_model` (época 15) | 81.8 % | 89.4 % |
| valid, `model_final` | 81.7 % | **90.6 %** |
| test, `model_final` | 90.9 % | **90.9 %** |

mAP@0.5 con IoU rotada, 11 puntos. En test con score ≥ 0.5: precisión 0.97, recall 0.95,
IoU media de los aciertos 0.90 (60 TP / 2 FP / 3 FN; los fallos son un billete cortado por el
borde y fajos de billetes).

- El mAP en valid se quedó clavado en ~81.7 % desde la época 10 por el `nms_threshold: 0.1`
  heredado de DOTA: el 17 % de los billetes de valid solapan con un vecino con IoU > 0.1 y el
  NMS eliminaba uno de cada par. Con 0.5 se recupera (+9 puntos); 0.6/0.7 no mejoran más.
- `best_model` se eligió con el NMS antiguo, así que **usa `model_final`** (ya es el `weights`
  por defecto de la config). Si reentrenas, la selección del best se hará ya con NMS 0.5.
- Predicciones de test en `output/predictions/` (txt, YOLO-OBB, JSON, `vis/`).
- Log completo: `logs/train.log`.

### Métricas completas (`scripts/evaluate_obb.py`)

IoU rotada exacta (Shapely), AP COCO-style (101 puntos, score ≥ 0.01), P/R/F1 con score ≥ 0.5.
`model_final`, NMS 0.5, inferencia a 640:

| dataset | imgs | GT | mAP50 | mAP75 | mAP50-95 | P@.5 | R@.5 | F1@.5 | P@.75 | R@.75 | F1@.75 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| valid | 100 | 143 | 95.0 | 79.3 | 69.6 | 0.977 | 0.895 | 0.934 | 0.863 | 0.790 | 0.825 |
| test | 47 | 63 | 97.1 | 96.0 | 79.1 | 0.968 | 0.952 | 0.960 | 0.968 | 0.952 | 0.960 |
| billetesprueba (143 fotos nuevas, 208 cajas) | 143 | 208 | 85.7 | 44.3 | 45.4 | 0.966 | 0.683 | 0.800 | 0.701 | 0.495 | 0.580 |
| billetesprueba, inferencia a 1024 | 143 | 208 | 88.4 | 63.3 | 54.4 | 0.803 | 0.841 | 0.822 | 0.638 | 0.668 | 0.653 |

(El mAP50 de ppdet — 90.6/90.9 — es más bajo porque usa 11 puntos y score ≥ 0.1.)

`billetesprueba 2.yolov8-obb` (fotos de móvil a resolución completa, 59 en HEIC) se convierte con
`scripts/convert_yolo_obb.py --src "billetesprueba 2.yolov8-obb" --out dataset/billetesprueba` y se
evalúa con `scripts/evaluate_obb.py --dataset_dir dataset/billetesprueba --split all [--img_size 1024]`.
Ahí el modelo falla sobre todo con **billetes pequeños en el encuadre** (fotos lejanas, varios billetes
sobre una mesa): el dataset de entrenamiento solo tiene billetes que llenan la imagen. Inferir a 1024
recupera recall (0.68 → 0.84) a costa de precisión; la solución real es reentrenar con augmentación de
escala (zoom-out / recops) o añadir fotos de ese tipo al train.

## Hallazgo: interpolación al reducir fotos grandes (`interp: 3`)

El `Resize` de inferencia venía con interpolación cúbica (`interp: 2`); al reducir fotos de 4000 px a 640
produce *aliasing* y hunde las métricas en billetesprueba. Con `INTER_AREA` (`interp: 3`, ya fijado en
`TestReader`/`EvalReader` de ambos readers) valid/test no cambian (±0.5) y billetesprueba sube en los dos
modelos (mAP50-95: base 45.4 → 52.2, aug 47.1 → 59.8). Reducir además a 416 px antes (`--pre_size 416`,
la resolución de origen del train) da un poco más (aug: 60.8 / mAP75 75.8). Para desplegar con fotos de
móvil: reducir siempre con INTER_AREA (o a 416 px de lado mayor primero).

## Variante con train ampliado + augmentación tipo Ultralytics (entrenada el 2026-09-21)

- **Train ampliado**: `augmented/` contiene 710 copias offline (2 por imagen de train, rotación ±180°,
  shear, perspectiva, traslación) de las 355 imágenes de train de v1. `scripts/add_augmented.py` las
  integra en `dataset/banknotes_obb/images/<stem>__augN.jpg`, comprueba que ninguna procede de valid/test
  y genera `annotations/train_plus_aug.json` (1065 imgs / 1668 cajas). Valid y test no cambian.
- **Augmentación online** (`configs/_base_/ppyoloe_r_reader_banknotes_aug.yml`), aplicada en vuelo a las
  1065 imágenes: mosaico 2×2 con escala ±20 % y traslación ±5 % hasta la época 34 (`mosaic_epoch`), después
  escala/traslación sobre imagen suelta (`RRandomAffine start_epoch: 34`), jitter HSV
(`RandomDistort`), flip y rotaciones como antes. Los operadores OBB (`RMosaic`, `RRandomAffine`)
están en `scripts/obb_aug.py` porque los de ppdet no propagan polígonos rotados; se registran al
lanzar con el wrapper. `configs/ppyoloe_r_crn_s_banknotes_aug.yml`: 40 épocas (266 iter/época → ~10.6k
iteraciones, frente a 7k del run base), coseno hasta 48, checkpoint/eval cada 4 épocas.

```powershell
.venv\Scripts\python scripts	rain.py -c configs\ppyoloe_r_crn_s_banknotes_aug.yml --eval -o save_dir=output_aug
.venv\Scripts\python scripts\evaluate_obb.py -c configs\ppyoloe_r_crn_s_banknotes_aug.yml -w output_aug\model_final.pdparams --split valid test
.venv\Scripts\python scripts\evaluate_obb.py -c configs\ppyoloe_r_crn_s_banknotes_aug.yml -w output_aug\model_final.pdparams --dataset_dir datasetilletesprueba --split all --output_dir output_aug\metrics_billetesprueba
```

Motivación: en billetesprueba los billetes ocupan ~10 % de la foto (38 % en train) y el reader base
no tiene ninguna augmentación de escala. Si se quiere cubrir mejor esa escala, subir `scale` de
`RMosaic` a 0.4.

### Resultados base vs aug (`model_final` de cada uno, NMS 0.5, inferencia 640 con `interp: 3`)

| dataset | modelo | mAP50 | mAP75 | mAP50-95 | P@.5 | R@.5 | F1@.5 |
|---|---|---|---|---|---|---|---|
| valid | base (`output/`) | 94.9 | **80.4** | **70.1** | 0.977 | 0.895 | 0.934 |
| valid | aug (`output_aug/`) | **96.4** | 75.0 | 64.0 | 0.969 | 0.888 | 0.927 |
| test | base | 96.9 | **96.0** | **78.7** | 0.984 | 0.952 | 0.968 |
| test | aug | **99.9** | 88.3 | 72.5 | 0.984 | 0.952 | 0.968 |
| billetesprueba | base | 93.2 | 50.7 | 52.2 | 0.970 | 0.784 | 0.867 |
| billetesprueba | aug | **95.4** | **69.9** | **59.8** | 0.962 | **0.851** | **0.903** |
| billetesprueba, `--pre_size 416` | aug | 95.3 | 75.8 | 60.8 | 0.962 | 0.851 | 0.903 |

- El modelo aug generaliza claramente mejor a fotos reales (billetesprueba) pero localiza peor en
  valid/test (mAP75 −5/−8). Hipótesis: las copias offline con *shear*/perspectiva ya no son rectángulos
  (desviación mediana 2°, p90 4°; IoU con su `minAreaRect` 0.977, p10 0.956) → etiquetas rbox algo
  imprecisas en 2/3 del train; también influye que fueron 40 épocas con mosaico frente a 80 sin él.
- mAP@0.5 de ppdet en valid durante el run: 74.9 (ép. 4) → 84.0 → 82.6 → 86.1 → 86.2 → 87.8 → 88.8 →
  88.8 → 89.0 (ép. 36, sin mosaico) → **90.1** (ép. 40). Log: `logs/train_aug.log`.
- Siguiente experimento razonable: solo augmentación online (sin copias offline con shear/perspectiva),
  80 épocas, para ver si se recupera mAP75 manteniendo la generalización.

## Sobre `ext_op_fallback` (ops custom sin compilar)

Los modelos rotados de PaddleDetection usan la op custom `rbox_iou` (IoU entre rectángulos
rotados) en el **asignador de entrenamiento** y en la **métrica RBOX**. Compilarla en Windows
exige Visual Studio + CUDA toolkit, que no están instalados. `ext_op_fallback/` es un paquete
`ext_op` en Paddle puro con el mismo algoritmo (detectron2: intersección de aristas + vértices
interiores + shoelace), vectorizado en GPU:

- validado contra Shapely: error máx. 8.6e-7 (`python ext_op_fallback\test_rbox_iou.py`)
- ~44 ms por llamada [9 gt × 8400 anchors]; sobrecoste ≈ 0.2 s/iteración con batch 4
- si más adelante compilas la op oficial (`python PaddleDetection\ppdet\ext_op\setup.py install`
  desde una consola *x64 Native Tools* de VS), el paquete la detecta y delega en ella.
- `nms_rotated` no está implementada: PP-YOLOE-R no la usa (su NMS es el `multiclass_nms`
  nativo de Paddle con polígonos), así que exportación e inferencia tampoco la necesitan.

## Dataset externo "Euro Banknote Detection" → modelo `extra` (RECOMENDADO)

551 fotos (640×640, 12 clases de billete + `hand`) → **304 incluidas** en train tras: quitar 142
duplicados/casi duplicados con nuestros datasets (pHash/dHash ≤ 8; 20 coincidían con nuestro valid/test),
96 primeros planos (1 billete > 50 % de la imagen) y 9 fotos sin billete; clases de billete fusionadas en
`euro_banknote`, `hand` descartada. Resultado: `annotations/train_plus_extra.json` = 355 + 304 = 659 imgs /
1918 cajas (y `train_plus_aug_extra.json` con las copias offline además).

- **Manifiesto de la selección** (todas las fotos con su estado y motivo, versionado en git):
  `data_manifests/eurobanknotes_extra_selection.{md,csv,json}` (`scripts/extra_manifest.py`).
- Selección limpia en formato YOLOv8-OBB para subir a Roboflow: `export/eurobanknotes_extra_yolov8obb/`
  (`scripts/export_yolo_obb.py`).
- Config del run: `configs/ppyoloe_r_crn_s_banknotes_extra.yml` (train 659, augmentación online, sin copias
  offline, 60 épocas, mosaico hasta la 50, ~1 h 30): `python scripts/train.py -c configs/ppyoloe_r_crn_s_banknotes_extra.yml --eval -o save_dir=output_extra`.
  mAP@0.5 ppdet en valid: 81.7 (ép. 5) → 89.0 (ép. 30) → 89.4 (ép. 60). Log: `logs/train_extra.log`.

### Comparativa final de los tres modelos (NMS 0.5, inferencia 640 con `interp: 3`, P/R/F1 a score ≥ 0.5)

| dataset | modelo | mAP50 | mAP75 | mAP50-95 | P@.5 | R@.5 | F1@.5 |
|---|---|---|---|---|---|---|---|
| valid | base | 94.9 | 80.4 | **70.1** | 0.977 | 0.895 | 0.934 |
| valid | aug | 96.4 | 75.0 | 64.0 | 0.969 | 0.888 | 0.927 |
| valid | **extra** | 95.1 | **80.9** | 69.5 | 0.969 | 0.874 | 0.919 |
| test | base | 96.9 | **96.0** | **78.7** | 0.984 | 0.952 | 0.968 |
| test | aug | 99.9 | 88.3 | 72.5 | 0.984 | 0.952 | 0.968 |
| test | **extra** | **100.0** | 91.5 | 77.6 | **1.000** | **0.984** | **0.992** |
| billetesprueba | base | 93.2 | 50.7 | 52.2 | 0.970 | 0.784 | 0.867 |
| billetesprueba | aug | 95.4 | 69.9 | 59.8 | 0.962 | 0.851 | 0.903 |
| billetesprueba | **extra** | **98.7** | **84.2** | **66.5** | 0.935 | **0.971** | **0.953** |

**`output_extra/model_final.pdparams` es el modelo recomendado**: en fotos reales sube el recall de 0.78 a 0.97 y el
mAP75 de 51 a 84, manteniendo la localización fina del base en valid/test. Conclusión de la ablación: lo que
faltaba eran datos del tipo correcto (billetes pequeños, varios por foto, manos, fondos reales); 304 fotos así
aportaron más que 710 copias offline y sin la penalización en mAP75 de estas. billetesprueba sigue siendo una
prueba independiente (ninguna de sus fotos coincide con las 304 añadidas).

## Repositorio: qué está y qué no

En git van: código (`scripts/`, `ext_op_fallback/`), `configs/`, `v1/` (splits), `data_manifests/`, `README.md`,
`requirements.txt`, `logs/`, las métricas JSON (`output*/metrics*/`) y los **pesos finales**
`output/model_final.pdparams` (base) y `output_aug/model_final.pdparams` (aug), 31 MB cada uno.
Fuera (`.gitignore`): `.venv/`, el clone de `PaddleDetection/`, los exports de Roboflow, `augmented/`,
`dataset/`, `export/` y los checkpoints intermedios. Si vas a subir pesos con frecuencia, mejor Git LFS
(`git lfs track "*.pdparams"`) para no engordar el historial.

## Puesta en marcha en otra máquina (macOS / Linux / Windows)

```bash
git clone <este-repo> paddletest && cd paddletest
git clone --depth 1 --branch release/2.8 https://github.com/PaddlePaddle/PaddleDetection.git   # probado con commit 7a4fc25
uv venv .venv --python 3.10
# PaddlePaddle segun plataforma:
uv pip install --python .venv/bin/python paddlepaddle==3.0.0                                     # macOS (Apple Silicon o Intel) / CPU
# uv pip install --python .venv/Scripts/python.exe "paddlepaddle-gpu==3.0.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/   # Windows/Linux GPU
uv pip install --python .venv/bin/python -r requirements.txt      # incluye ext_op_fallback (ops rotadas en Paddle puro, sin compilar nada)
.venv/bin/python ext_op_fallback/test_rbox_iou.py                 # comprueba la IoU rotada contra Shapely
```

Datos: copia a la raíz del repo, con estos nombres exactos, las carpetas `Annotated banknotes 2.yolov8-obb`,
`billetesprueba 2.yolov8-obb`, `Euro Banknote Detection.yolov8-obb` y `augmented` (o simplemente copia la
carpeta `dataset/` ya generada desde otra máquina) y regenera `dataset/` en este orden:

```bash
P=.venv/bin/python   # en Windows: .venv\Scripts\python
$P scripts/prepare_dataset.py                                                      # principal con splits v1 -> dataset/banknotes_obb
$P scripts/convert_yolo_obb.py --src "billetesprueba 2.yolov8-obb" --out dataset/billetesprueba
$P scripts/add_augmented.py --src augmented --dataset dataset/banknotes_obb        # -> train_plus_aug.json
$P scripts/convert_yolo_obb.py --src "Euro Banknote Detection.yolov8-obb" --out dataset/eurobanknotes_extra     --merge_class euro_banknote --drop_classes hand --exclude data_manifests/eurobanknotes_extra_exclude.json     --max_single_area 0.5 --skip_empty
$P scripts/merge_extra.py --extra dataset/eurobanknotes_extra --out_name train_plus_extra                       # -> train_plus_extra.json
$P scripts/merge_extra.py --base train_plus_aug --extra dataset/eurobanknotes_extra --out_name train_plus_aug_extra
```

(`data_manifests/eurobanknotes_extra_exclude.json` es la lista de duplicados calculada con
`scripts/find_duplicates.py`; se versiona para no depender de recalcular los hashes.)

En **Mac no hay GPU NVIDIA**, así que todo va en CPU: predecir y evaluar funciona bien (~1 s/imagen a
640 px; el fallback `rbox_iou` tarda ~5 s por imagen en la métrica, asumible para cientos de fotos),
pero **entrenar en CPU no es práctico** (horas por época). Comandos (en Mac/Linux usa `.venv/bin/python`
y barras `/`; en las herramientas de ppdet añade `-o use_gpu=false`):

```bash
.venv/bin/python scripts/predict_obb.py -w output_aug/model_final.pdparams --images /ruta/a/fotos --vis --cpu
.venv/bin/python scripts/evaluate_obb.py -w output_aug/model_final.pdparams --split valid test --cpu
.venv/bin/python PaddleDetection/tools/infer.py -c configs/ppyoloe_r_crn_s_banknotes.yml -o weights=output_aug/model_final.pdparams use_gpu=false --infer_img=foto.jpg
```

Notas: las configs referencian `../PaddleDetection/configs/...`, así que el clone debe estar en la raíz
del proyecto con ese nombre; `worker_num: 2` y `use_shared_memory: false` funcionan igual en macOS.

## Entorno (máquina de desarrollo, Windows)

- Python 3.10 (gestionado por `uv`), `paddlepaddle-gpu==3.0.0` (cu118, cuDNN incluido).
- El driver NVIDIA es 461.33 (CUDA 11.2); Paddle avisa de la diferencia con el runtime 11.8
  pero `paddle.utils.run_check()` y el entrenamiento funcionan (compatibilidad de versiones
  menores de CUDA 11.x). Si en algún momento diera problemas, actualizar el driver es la solución.
- `setuptools<80` (ppdet usa `pkg_resources`), `scikit-learn` en lugar del roto `sklearn==0.0`.

Reproducir el entorno desde cero:

```powershell
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe "paddlepaddle-gpu==3.0.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
uv pip install --python .venv\Scripts\python.exe "setuptools<80" wheel "numpy<2" tqdm typeguard "visualdl>=2.2.0" "opencv-python<=4.6.0" PyYAML shapely scipy terminaltables Cython pycocotools Pillow "imgaug>=0.4.0" pyclipper lapx motmetrics scikit-learn
uv pip install --python .venv\Scripts\python.exe -e .\ext_op_fallback
git clone --depth 1 --branch release/2.8 https://github.com/PaddlePaddle/PaddleDetection.git
```
