# Comparador de detectores OBB

Herramientas locales para comparar los detectores de billetes con cajas orientadas (OBB)
que se entrenan en las ramas de este repo, una vez exportados a ONNX:

- **App Streamlit** (`app.py`): sube una foto o hazla con la webcam y mira, lado a lado, las
  cajas orientadas y la probabilidad de cada modelo, con sliders de *confidence threshold*
  y *NMS threshold* y mapas de explicabilidad (Grad-CAM / EigenCAM).
- **Notebook de métricas** (`evaluacion_modelos.ipynb`): precision, recall, F1, mAP50,
  mAP75, mAP50-95 y estadísticas de la confianza sobre un dataset externo, con gráficas y
  estudios de umbral, NMS, rotación, resolución y tamaño del billete.

| rama | modelo | script de export |
|---|---|---|
| `feature/yolox_obb` | YOLOX-OBB | `src/yolox_mps/export_onnx.py` |
| `ppyoloe-r-torch-clean` | PP-YOLOE-R | `src/ppyoloer_mps/export_onnx.py` |
| `feature/rt_refactor` | RTMDet-R | `src/rtmdet_mps/export_onnx.py` |

No hay que tocar esas ramas: se copia lo que sale de cada `export_onnx.py` y la app y el
notebook lo leen tal cual.

---

## 1. Instalación

Requisitos: [uv](https://docs.astral.sh/uv/getting-started/installation/) (instala él
mismo Python 3.12). Todo corre en CPU, en Windows, macOS o Linux.

```bash
git clone https://github.com/ccermeno1/testbill.git
cd testbill
git checkout feature/model-comparison-app
uv sync
```

`uv sync` crea `.venv/` con las versiones exactas de `uv.lock`. En Linux, `torch` se
instala desde el índice CPU de PyTorch (sin los ~2 GB de CUDA); en Windows y macOS la de
PyPI ya es CPU.

Los **modelos** y el **dataset** no están en git (pesan ~50 MB y ~260 MB). Después de
clonar hay que copiarlos:

- `models/` → los ONNX exportados (ver [§3](#3-añadir-un-modelo) y `models/README.md`).
- `external_dataset/` → el dataset externo, solo para el notebook (ver
  `external_dataset/README.md`).

## 2. La app

```bash
uv run streamlit run app.py
```

Se abre en <http://localhost:8501>.

### Uso

1. **Fuente de la imagen** (arriba): *📁 Subir foto* (JPG, PNG, BMP, WEBP, HEIC) o
   *📷 Webcam* (el navegador pide permiso; pulsa *Take photo*). La orientación EXIF de
   las fotos de móvil se aplica sola.
2. **Resultados**: una tarjeta por modelo con
   - la foto con cada caja orientada, su número (`#1`, `#2`…), clase y probabilidad; el
     punto relleno marca la primera esquina de la caja, para ver su orientación;
   - familia, tamaño de entrada, orden de color, tiempo de inferencia y nº de detecciones;
   - desplegable **Detecciones**: tabla con probabilidad, cx, cy, w, h y ángulo (px de la
     foto original, grados).

   Debajo, una tabla **Resumen** compara todos los modelos, y el desplegable **Contrato de
   pre/post-proceso** muestra qué ha leído la app de cada modelo (color, relleno,
   normalización, clases).

Barra lateral:

| control | efecto |
|---|---|
| **🔄 Recargar carpeta models/** | vuelve a leer `models/` (tras copiar o borrar un modelo) |
| **Modelos a comparar** | qué modelos se muestran |
| **Confidence threshold** | descarta las cajas con probabilidad menor |
| **NMS IoU threshold** | dos cajas de la misma clase con IoU rotado mayor se fusionan (queda la de más probabilidad). Bajo = más agresivo; alto = respeta billetes superpuestos |
| **Máx. detecciones por imagen** | tope tras el NMS |
| **Mostrar mapa de activación** | añade a cada tarjeta el mapa de calor |
| **Método** | *Grad-CAM (HiResCAM)*: qué zonas suben la probabilidad de una detección. *EigenCAM*: en qué se fija la red en general, sin objetivo |
| **Detección a explicar** | la de mayor probabilidad, *Todas* (suma) o *Elegir nº* (el `#` de la tabla); la elegida se resalta en amarillo |
| **Opacidad del mapa** | mezcla del mapa de calor con la foto |

Los sliders son instantáneos: la red se ejecuta una vez por foto y al moverlos solo se
repite el post-proceso. Si no hay detecciones, Grad-CAM explica la predicción de mayor
score aunque quede bajo el umbral (útil para ver qué "casi" detectó).

### Explicabilidad: cómo funciona

Los dos métodos trabajan sobre los mapas de características que lee la cabeza del detector
(uno por nivel, strides 8/16/32), así que el mapa tiene esa resolución: manchas, no bordes.

- **Grad-CAM (HiResCAM)**: gradiente de la probabilidad de la detección elegida respecto a
  cada mapa, multiplicado elemento a elemento por el mapa. No explica el ángulo ni el
  ajuste de la caja, solo la probabilidad. ONNX Runtime no calcula gradientes, así que el
  ONNX se convierte a PyTorch con `onnx2torch` y se comprueba contra ONNX Runtime antes de
  usarlo. Si la conversión falla, la app lo avisa y muestra EigenCAM.
- **EigenCAM**: primera componente principal de las activaciones; funciona solo con ONNX
  Runtime.

Los mapas de la cabeza se localizan solos en el grafo: en cada nivel, el último tensor del
que salen tanto la rama de clasificación como la de regresión. Si en algún modelo fallara,
se pueden fijar por nombre en `explain.feature_tensors` de un `model.json` (§5).

## 3. Añadir un modelo

1. En la rama del modelo, exporta a ONNX:

   ```bash
   # YOLOX-OBB (feature/yolox_obb)
   uv run python src/yolox_mps/export_onnx.py <checkpoint.pth> --out model_640.onnx --img-size 640

   # PP-YOLOE-R (ppyoloe-r-torch-clean)
   uv run python src/ppyoloer_mps/export_onnx.py <checkpoint.pt> --out ppyoloe_r_s_640.onnx --img-size 640

   # RTMDet-R (feature/rt_refactor)
   uv run python src/rtmdet_mps/export_onnx.py <checkpoint.pth> --out model_800.onnx --img-size 800
   ```

   En Windows, si el export falla con `UnicodeEncodeError` (el exportador de torch imprime
   un emoji), ejecútalo con `PYTHONIOENCODING=utf-8` (PowerShell:
   `$env:PYTHONIOENCODING="utf-8"`).

2. Crea una carpeta con el nombre que quieras dentro de `models/` y copia **todos** los
   archivos que genera el export:

   ```
   models/
     yolox_nano_v1/
       model_640.onnx
       model_640.metadata.json
     ppyoloe_r_s/
       ppyoloe_r_s_640.onnx
       ppyoloe_r_s_640.metadata.json
     rtmdet_r_tiny_C/
       model_800.onnx
       model_800.onnx.data         <- los pesos de RTMDet-R: sin él no carga
       model_800.metadata.json
   ```

3. En la app, **🔄 Recargar carpeta models/**. En el notebook se evalúa en la siguiente
   ejecución.

Detalles:

- El `.metadata.json` de RTMDet-R no trae los nombres de clase: la app pone `object`.
  Añade `"class_names": ["euro_banknote"]` al JSON para que salga el nombre.
- El tamaño de entrada de la red va fijo en el ONNX (`--img-size`). Para comparar tamaños,
  exporta el mismo checkpoint varias veces en carpetas distintas: salen como modelos
  separados.
- Si una carpeta no se puede leer, la barra lateral de la app lo avisa y el resto de
  modelos sigue funcionando.

## 4. Notebook de métricas

```bash
uv run jupyter lab evaluacion_modelos.ipynb
```

(o ábrelo en VS Code y elige como kernel el Python de `.venv`). Revisa la celda de
configuración y ejecuta todo (*Run → Run All Cells*). El propio notebook explica cada
sección y cada variable; lo principal:

| variable | por defecto | qué hace |
|---|---|---|
| `DATASET` | `external_dataset/billetesprueba 2.yolov8-obb` | dataset YOLOv8-OBB a evaluar |
| `AP_SCORE_THR` | 0.05 | confianza mínima de las detecciones que entran en el mAP |
| `OPERATING_THR` | 0.5 | umbral de trabajo para precision / recall / F1 y la confianza |
| `NMS_IOU` | 0.3 | NMS, igual para todos los modelos |
| `RUN_STUDIES` | `True` | `False` = solo métricas principales (mucho más rápido) |
| `QUICK` | `False` | `True` = estudios con 1 de cada 3 fotos |

Qué calcula: tabla principal (precision, recall, F1, mAP50/75/50-95, media / mediana /
desviación de la confianza, TP/FP/FN, ms por imagen), curvas PR, P/R/F1 frente al umbral
con el umbral óptimo de cada modelo, histogramas de confianza de aciertos frente a falsos
positivos, IoU y error de ángulo de los aciertos, recall por tamaño del billete, estudios
de NMS, rotación (0-180°) y resolución (320-1600 px), latencia y las fotos donde más falla
cada modelo.

**Umbrales** (explicado con detalle en el notebook): el mAP se calcula con todas las
detecciones de confianza ≥ 0.05, igual que los `evaluate.py` de los repos; precision,
recall y F1 se dan con un umbral común de 0.5 (la comparación principal, justa y la que usa
producción) y con el umbral que maximiza el F1 de cada modelo (su techo, optimista porque se
elige sobre el propio test; lo correcto es fijarlo en validación).

El AP es el de los repos (port de `eval_rbbox_map` de mmrotate, VOC "area"), así que los
números son comparables con sus READMEs: RTMDet-R C @800 da aquí 0.989 / 0.938 / 0.766
(mAP50 / 75 / 50-95) frente a 0.992 / 0.957 / 0.773 en su README, que usó su propia
conversión de las anotaciones de estas mismas fotos.

La inferencia se guarda en `eval_cache/` (no se sube a git): volver a ejecutar es
instantáneo, y si cambian los modelos, el dataset o los estudios se rehace sola. Con dos
modelos y los estudios activados tarda ~10 min en CPU.

## 5. Contrato de pre/post-proceso

La app y el notebook leen cada export tal como sale de su rama y lo llevan a la misma
forma:

- **Entrada**: `(1, 3, S, S)` float32 en 0..255. La foto se redimensiona manteniendo el
  aspecto (lado largo = S; `INTER_AREA` al reducir, `INTER_LINEAR` al ampliar) y se pega
  arriba a la izquierda de un lienzo relleno. La normalización mean/std va dentro del
  grafo en los tres.
- **Salidas**: YOLOX-OBB y PP-YOLOE-R dan cajas ya decodificadas, `boxes (1, N, 5)` (cx, cy,
  w, h en px de la entrada, ángulo en radianes, horario con y hacia abajo) y `scores`
  `(1, N, C)` o `(1, C, N)` (se detecta por la forma). RTMDet-R da los 9 mapas crudos de la
  cabeza (cls / reg / angle × strides 8, 16, 32) y se decodifican como en su
  `onnx_example.py` (comprobado contra `RTMDetR.predict`: mismas cajas y scores).
- **Post-proceso**: clase de mayor score por prior, umbral, NMS rotado por clase, dividir
  cx, cy, w, h por la escala del redimensionado.

| | YOLOX-OBB | PP-YOLOE-R | RTMDet-R |
|---|---|---|---|
| color | BGR | RGB | BGR |
| relleno | 114 | 0 | 114 |
| salidas | decodificadas | decodificadas | mapas crudos |
| S habitual | 640 | 640 | 800 |

Todo esto sale del `.metadata.json`. Si un modelo necesita otra cosa (u otro nombre en la
app), se puede poner un `model.json` en su carpeta, que tiene prioridad (todos los campos
salvo `onnx` son opcionales):

```json
{
  "name": "Mi modelo",
  "family": "custom",
  "onnx": "model.onnx",
  "class_names": ["euro_banknote"],
  "input": {"name": "images", "size": 640, "color": "BGR", "pad_value": 114,
            "pad_multiple": 32, "normalize": null},
  "outputs": {"boxes": "boxes", "scores": "scores", "scores_layout": "NC",
              "raw": [], "strides": [8, 16, 32]},
  "postprocess": {"score_thr": 0.5, "nms_iou": 0.3, "max_det": 100, "nms_pre": 1000},
  "explain": {"feature_tensors": []}
}
```

- `normalize`: `{"mean": [...], "std": [...]}` en unidades 0..255, solo si el grafo **no**
  normaliza por dentro.
- `raw`: salidas crudas estilo RTMDet-R, nombres en orden cls, reg, angle por nivel
  (entonces `boxes` y `scores` no se usan).
- `pad_multiple`: solo si el grafo tiene alto y ancho dinámicos; si son fijos, se rellena
  hasta el tamaño del grafo.

## 6. Problemas frecuentes

| síntoma | causa / solución |
|---|---|
| Un modelo no aparece | falta el `.metadata.json` (o `model.json`) junto al `.onnx`; la barra lateral lo avisa. Pulsa **🔄 Recargar** |
| RTMDet-R falla al cargar | falta el `.onnx.data` en la carpeta |
| Sale `object` en vez del nombre de la clase | añade `"class_names"` al `.metadata.json` |
| "Grad-CAM no disponible" | la conversión a PyTorch no reproduce el ONNX para ese modelo; se muestra EigenCAM |
| La webcam no aparece | abre la app en `localhost` (el navegador bloquea la cámara en `http://` remoto) y concede el permiso |
| Export en Windows: `UnicodeEncodeError` | `PYTHONIOENCODING=utf-8` antes de exportar |
| El notebook no ve el dataset | revisa `DATASET` en la celda de configuración |

## 7. Estructura

```
app.py                     app Streamlit
evaluacion_modelos.ipynb   notebook de métricas y estudios
obbcompare/
  config.py                lectura de models/ y del contrato (.metadata.json / model.json)
  detector.py              pre-proceso, ONNX Runtime, decodificación, NMS rotado
  explain.py               Grad-CAM (onnx2torch) y EigenCAM
  draw.py                  dibujo de cajas orientadas y mapas de calor
  evaluation.py            dataset YOLOv8-OBB, rotación/resolución, métricas (AP de mmrotate)
models/                    ONNX a comparar (no versionado, ver models/README.md)
external_dataset/          datasets de test (no versionado, ver external_dataset/README.md)
pyproject.toml, uv.lock    dependencias (uv)
```
