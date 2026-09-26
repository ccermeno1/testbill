# external_dataset/

Datasets externos para `evaluacion_modelos.ipynb`, en formato **YOLOv8-OBB** (export de
Roboflow): `<split>/images/` + `<split>/labels/` con líneas `clase x1 y1 x2 y2 x3 y3 x4 y4`
normalizadas. Se leen todos los splits que haya dentro. Admite JPG, PNG y HEIC.

Esta carpeta no se sube a git (salvo este README). El dataset usado hasta ahora es
`billetesprueba 2.yolov8-obb` (143 fotos de móvil, 208 billetes):
<https://universe.roboflow.com/clara-cermeno/billetesprueba-2> → Download → formato
"YOLOv8 Oriented Bounding Boxes" → descomprimir aquí.

```
external_dataset/
  billetesprueba 2.yolov8-obb/
    train/images/
    train/labels/
```

Para usar otro, cambia `DATASET` en la celda de configuración del notebook.
