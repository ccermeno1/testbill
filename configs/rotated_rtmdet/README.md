# Rotated RTMDet (tiny) — PyTorch puro, CUDA / MPS / CPU

Reimplementación de `RotatedRTMDetSepBNHead` de MMRotate sin mmcv:

| Pieza | Archivo |
|---|---|
| CSPNeXt + CSPNeXtPAFPN (nombres de módulos idénticos a MMDet) | `oriented_det/models/backbones/cspnext.py` |
| Cabeza SepBN, DynamicSoftLabelAssigner rotado, QFL + rIoU, NMS | `oriented_det/models/rotated_rtmdet.py` |
| Conversor YOLO-OBB → DOTA | `tools/yolo_obb_to_dota.py` |
| Config para billetes | `configs/rotated_rtmdet/billetes_tiny_le90.json` |

Paridad verificada: el checkpoint oficial `rotated_rtmdet_tiny-3x-dota` de MMRotate
carga **482/482 tensores sin huecos**, y con él el modelo detecta 5/5 aviones de
`demo/dota_1` (IoU mediana 0,85).

## Pesos iniciales (`model.rtmdet_pretrained_weights`)

| Valor | Qué carga | Cuándo usarlo |
|---|---|---|
| `"coco"` (por defecto) | RTMDet-tiny COCO: backbone + neck + torres de la cabeza + `rtm_reg` | Fotos normales (billetes). Recomendado. |
| `"imagenet"` | Solo el backbone CSPNeXt | Si quieres partir de menos sesgo |
| `"dota"` | Rotated RTMDet-tiny DOTA (incluye rama de ángulo) | Imágenes aéreas |
| ruta / URL a un `.pth` de MMDet o MMRotate | Lo que coincida en nombre y forma | Checkpoints propios |
| `null` | Nada (desde cero) | — |

Se descargan una vez en la caché de `torch.hub` (`~/.cache/torch/hub/checkpoints`).
No hace falta mmcv/mmengine: los objetos de mmengine del checkpoint se sustituyen por stubs.
`rtmdet_input_bgr: null` elige automáticamente BGR para `coco`/`dota` (convención MMDet).

## Flujo con un dataset YOLO-OBB

```bash
# 1) Convertir (una clase "banknote"; reduce fotos grandes para acelerar la carga)
python tools/yolo_obb_to_dota.py --src /ruta/dataset_yolo --dst /ruta/billetes_dota \
    --single-class banknote --max-side 1280

# 2) Editar configs/rotated_rtmdet/billetes_tiny_le90.json → dataset.data_root / train_tiles_dir / val_tiles_dir

# 3) Entrenar (en Mac elige MPS automáticamente; AMP se desactiva solo en MPS)
PYTORCH_ENABLE_MPS_FALLBACK=1 python tools/train.py --config configs/rotated_rtmdet/billetes_tiny_le90.json

# 4) Inferencia
python tools/image_demo.py fotos/ runs/rotated_rtmdet/<fecha>/config.json \
    runs/rotated_rtmdet/<fecha>/checkpoints/best_mAP_*.pth --device mps --out-dir salida/
```

`image_demo.py` reescala cada foto entera al canvas del modelo, igual que en el
entrenamiento (`resize_mode: pad`). Antes procesaba las imágenes mayores que el canvas
en ventanas a resolución nativa, así que los billetes llegaban a otra escala.

El conversor acepta `images/<split>` + `labels/<split>` (Ultralytics) o `<split>/images` +
`<split>/labels` (Roboflow; `valid` → `val`), aplica la orientación EXIF a los píxeles (el
repo lee con PIL sin EXIF) y pone las extensiones en minúscula (el loader busca `*.jpg`
y en macOS importa la mayúscula). Las imágenes sin etiquetas quedan como negativos.

## Receta y ajustes

- AdamW, lr `2.5e-4` para batch 8 (receta MMRotate), weight decay `0.05` sin decay en
  normas ni bias, cosine + warmup. Si cambias el batch, escala el lr proporcionalmente.
- `final_nms_iou_threshold: 0.3`. Es **el** parámetro a ajustar si tus billetes se
  solapan mucho (abanico/montón): súbelo (0.4–0.5) si se pierden billetes solapados;
  bájalo si aparecen duplicados.
- `target_size: [640, 640]` con `resize_mode: pad`. Súbelo si los billetes salen muy pequeños en la foto.
- Rotación aleatoria ±180° y flips activados: los billetes aparecen en cualquier orientación.
- No incluye EMA ni Mosaic/MixUp de la receta original de RTMDet.
