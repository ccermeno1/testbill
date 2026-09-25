# Preparación de datos

Scripts sin dependencias de Paddle que convierten los exports de Roboflow (YOLOv8-OBB) al
formato COCO con polígonos que consume este repo. Son los mismos que en la rama de Paddle.

| script | qué hace |
|---|---|
| `prepare_dataset.py` | export principal + splits congelados `v1` -> `data/banknotes_obb` |
| `convert_yolo_obb.py` | cualquier export YOLOv8-OBB -> dataset COCO (fusiona/descarta clases, excluye duplicados, filtra primeros planos; lee HEIC) |
| `find_duplicates.py` | duplicados y casi duplicados por hash perceptual entre datasets |
| `merge_extra.py` | añade un dataset convertido al train del principal |
| `add_augmented.py` | integra copias aumentadas offline en el train |
| `extra_manifest.py` | manifiesto de qué fotos entraron y por qué se excluyeron las demás |
| `export_yolo_obb.py` | dataset COCO -> YOLOv8-OBB (para subir a Roboflow) |
| `visualize_obb.py` | dibuja ground truth y predicciones, con hojas de contacto |

Los manifiestos y las anotaciones ya generadas están en `data_manifests/`, así que no hace falta
recalcular hashes ni filtros: basta con regenerar las imágenes y copiar esos JSON.

Necesitan `opencv-python-headless`, `pillow`, `pillow-heif`, `numpy`, `shapely` e `imagehash`
(este último solo `find_duplicates.py`).
