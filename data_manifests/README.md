# Manifiestos de datos

Las **imágenes** no se versionan (ver `.gitignore`): se descargan de Roboflow y se regeneran con los
scripts. Aquí está todo lo que fija *qué* datos se usaron, para que el entrenamiento sea reproducible.

| fichero | qué es |
|---|---|
| `eurobanknotes_extra_selection.{md,csv,json}` | las 551 fotos del dataset externo *Euro Banknote Detection* con su estado: 304 incluidas en train (y con qué nombre) y 247 excluidas con el motivo (duplicado + de qué foto y a qué distancia, primer plano, sin billete) |
| `eurobanknotes_extra_exclude.json` | rutas de los 142 duplicados detectados (`scripts/find_duplicates.py`), consumido por `scripts/convert_yolo_obb.py --exclude` |
| `annotations/*.json` | anotaciones COCO ya generadas (polígonos de 4 puntos), copia de las que produce el pipeline |

## `annotations/`

| fichero | imgs / cajas | uso |
|---|---|---|
| `train.json` | 355 / 556 | train de v1 |
| `valid.json` | 100 / 143 | validación (fijo en todos los runs) |
| `test.json` | 47 / 63 | test (fijo en todos los runs) |
| `train_plus_extra.json` | 659 / 1918 | **train del modelo `extra`** (recomendado) = train + 304 fotos externas |
| `train_plus_aug.json` | 1065 / 1668 | train del modelo `aug` = train + 710 copias offline (`augmented/`) |
| `train_plus_aug_extra.json` | 1369 / 3030 | ambas cosas (sin usar aún) |
| `eurobanknotes_extra_all.json` | 304 / 1362 | solo la selección externa (`dataset/eurobanknotes_extra`) |
| `billetesprueba_all.json` | 143 / 208 | test externo de fotos reales |

Los `file_name` apuntan a `dataset/banknotes_obb/images/` (las externas con prefijo `ebd_`), así que
tras regenerar las imágenes con los scripts del README se pueden copiar a
`dataset/banknotes_obb/annotations/` y usar directamente, sin recalcular hashes ni filtros.
