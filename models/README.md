# models/

Una carpeta por modelo con lo que escribe el `export_onnx.py` de su rama, tal cual.
Esta carpeta no se sube a git (salvo este README): copia aquí los modelos a mano.

```
models/
  yolox_nano_v1/              # feature/yolox_obb
    model_640.onnx
    model_640.metadata.json
  ppyoloe_r_s/                # ppyoloe-r-torch-clean
    ppyoloe_r_s_banknotes_640.onnx
    ppyoloe_r_s_banknotes_640.metadata.json
  rtmdet_r_tiny_C/            # feature/rt_refactor
    rtmdet_r_tiny_C_800.onnx
    rtmdet_r_tiny_C_800.onnx.data      <- pesos, imprescindible
    rtmdet_r_tiny_C_800.metadata.json
```

El nombre de las carpetas es libre. Ver el README principal, sección "Añadir un modelo".
