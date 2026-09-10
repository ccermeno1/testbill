# testbank — localización de billetes

Etapa de localización de un pipeline de inspección. Recorta cada billete de una
foto y se lo pasa al clasificador de manchas, que ya existe y queda fuera de este
alcance.

Arquitectura: **detector OBB de una etapa**. Entrada imagen, salida rectángulos
orientados, uno por billete. El recorte se hace con margen configurable sobre
cada caja y se rectifica por homografía.

Las alternativas de dos etapas están descartadas y no deben implementarse. La
segmentación clásica no separa billetes que se solapan o se tocan —son del mismo
color y textura, no hay borde entre ellos—, y el cuadrilátero libre no aporta
nada porque el ground truth son rectángulos girados.

Caja orientada y no alineada al eje porque los billetes aparecen en abanico o
adyacentes en ángulos distintos: las cajas alineadas de objetos alargados en
ángulos distintos se solapan casi por completo y el NMS estándar suprime
detecciones verdaderas. El NMS rotado lo resuelve.

## Guía de anotación

**Rectángulo orientado aproximado.** No se persigue exactitud geométrica. Un
billete arrugado o con bordes ondulados se anota con el rectángulo que lo
envuelva razonablemente.

**Umbral de visibilidad: 25%.** Un billete tapado por otro se anota solo si se ve
al menos ese porcentaje. Los que asoman una franja quedan sin anotar y son fondo
a efectos de entrenamiento.

Esta regla se aplica de verdad **aquí, al anotar**. El código no puede hacerla
cumplir: si un billete no se anotó, no hay nada que medir. Lo único que se puede
comprobar es la dirección contraria —una anotación que la contradice— y para eso
está `check-visibility`, que es un aviso para revisión manual, nunca un error.

Lee la sección siguiente antes de fiarte de su salida.

### Filtro de área relativa

**En cada imagen se conserva el billete de delante.** Al cargar, se descarta toda
anotación cuya área sea menor que `min_relative_area` (por defecto **0.25**) veces
el área de la mayor anotación de esa misma imagen.

Cubre las dos situaciones sin tener que distinguirlas:

- En un **abanico**, la franja visible de un billete tapado es mucho menor que el
  billete de delante, y cae.
- En una foto de **dos o tres billetes juntos**, todos tienen un tamaño parecido y
  se conservan todos.

Tres cosas que conviene tener claras:

**Es una aproximación a la política de visibilidad del 25%, no una medida de
oclusión.** El área relativa y la fracción visible son cosas distintas: una franja
larga y estrecha puede superar el 25% de área y estar tapada del todo, y un
billete pequeño pero entero puede quedar por debajo del umbral sin que nada lo
tape. El filtro y `check-visibility` son complementarios, no redundantes — el
primero quita fragmentos pequeños, el segundo marca lo grande pero tapado.

**Es un filtro en carga, no un borrado.** Los ficheros de anotación no se tocan
nunca. Las anotaciones filtradas **siguen en el dataset de origen**, y el informe
las lista con imagen, índice y porcentaje de área relativa para poder revisarlas y
corregirlas en Roboflow. Los índices son siempre la posición dentro del fichero,
nunca la posición tras filtrar, para que lleven a la anotación correcta.

**Es un parámetro, no una constante.** Está en `annotation_policy.min_relative_area`
y se puede subir o bajar sin reexportar el dataset. `--min-relative-area 0`
desactiva el filtro y muestra las anotaciones tal cual están en el fichero.

En la visualización, lo filtrado se dibuja en **gris discontinuo** con su índice y
su porcentaje, sin vértices numerados ni flecha de ancla: está ahí para poder
localizarlo, no como parte del conjunto canónico.

Medido sobre el export actual (502 imágenes, 693 anotaciones en `train`+`valid`),
el filtro por defecto descarta **14 anotaciones en 10 imágenes**, un 2%.

### Cómo leer `check-visibility`

El chequeo mide **solapamiento geométrico, no oclusión**, porque el orden de
profundidad no está anotado. Que A solape a B no dice cuál está encima.
Consecuencias medidas sobre datos sintéticos con 3 incumplimientos plantados
entre 108 imágenes y 301 anotaciones:

| | |
|---|---|
| Anotaciones marcadas | 28, en 20 imágenes |
| Incumplimientos reales | 3 |
| Encontrados por el chequeo | 2 de 3 |

Los **falsos positivos** salen del desconocimiento de la profundidad: un quad muy
solapado puede ser perfectamente el de arriba, que no está tapado en absoluto. La
línea `Incumplimientos reales: como mucho N` acota esto de forma exacta — recorre
todos los órdenes de profundidad posibles con un DP sobre subconjuntos y devuelve
el máximo número de anotaciones que podrían estar tapadas a la vez.

El **falso negativo** tiene una causa distinta y no tiene arreglo en código: si el
billete que tapa **no está anotado**, el chequeo no puede verlo. Es el mismo punto
ciego de la política, con la vuelta de tuerca de que puede ocultar justo el
incumplimiento que busca.

Usa el informe como lista de candidatos ordenada, y mira siempre la
visualización antes de tocar una anotación.

## Coordenadas y orden canónico

El pivote de todas las conversiones es el **quad canónico**: 4 vértices
normalizados. N formatos son 2N conversores, no N².

**Orden canónico.** Horario respecto al centroide, arrancando en el vértice que
abre el lado más largo. Desempate por menor `x+y`, luego menor `x`, luego menor
`y` — la cadena entera hace falta porque `x+y` empata en rectángulos cuya
diagonal es perpendicular a `(1,1)`.

No se usa «la esquina más cercana al origen»: es discontinua cerca de 45° y un
jitter de 2 px rota las etiquetas 90°.

**Aspecto de la imagen.** Las coordenadas normalizadas dividen `x` por el ancho e
`y` por el alto, que es un escalado **anisótropo**. En ese espacio el lado más
largo, el ángulo y el ratio no son los geométricos: un billete 2:1 tumbado en una
imagen 16:9 tiene ratio normalizado 1.13, y en 20:9 baja de 1 y el ancla salta al
lado corto. Por eso toda comparación de longitudes admite el aspecto, y el lector
lo recibe desde `data/derived/image_sizes.json`.

**Rejilla diádica.** Las coordenadas se ajustan a múltiplos de `2^-30`. Es lo que
hace que `flip(flip(q)) == q` se cumpla de forma **exacta**: en float64
`1 - (1 - 0.1)` da `0.09999999999999998`, así que `x -> 1-x` no es una involución.
Sobre la rejilla sí lo es, bit a bit. El error introducido es `2e-6 px` en una
imagen de 4000 px. Los ficheros de anotación no se modifican.

**Rango tolerante** `[-0.5, 1.5]`, con aviso fuera de `[0,1]` pero sin fallo: hay
billetes que cruzan el borde de la imagen y sus vértices caen fuera
legítimamente.

**Aviso de ancla inestable** si el ratio de lados baja de 1.1.

## Particiones

`splits.py` es la **única puerta de acceso** a la asignación. Ningún otro módulo
la calcula. Todo lee de `splits/{train,valid,test}.txt`, nunca del árbol de
directorios, para que la partición quede congelada aunque el directorio cambie.

**Modo adoptar (por defecto).** Si el directorio ya trae `train/`, `valid/` y
`test/` con `images/` y `labels/` —la estructura que exporta Roboflow— esa es la
partición. No se recalcula ni se «mejora»: solo se congela. Roboflow usa `valid`,
no `val`; `val` se acepta como alias dejando constancia, y tener los dos a la vez
es error.

**Modo crear.** Si no existe esa estructura, se genera con porcentajes
configurables y semilla registrada en `splits/manifest.json`.

`--group-key` acepta `filename-prefix`, `directory`, `manifest` y `none`. El
valor `none` afirma que cada imagen es independiente, y exige
`--i-confirm-independence` para que sea una afirmación explícita y no un
descuido: si existen varias tomas del mismo billete físico repartidas entre
particiones, las métricas salen infladas. La confirmación **solo se exige al
crear**; en modo adoptar no elegimos el reparto, únicamente lo congelamos.

**Integridad al cargar.** Una muestra en dos particiones es error fatal con
mensaje que la identifica. Con clave de grupo, también un grupo repartido.

**`extend-splits`** anexa muestras nuevas sin reordenar las existentes. Una
muestra nueva de un grupo ya existente hereda su partición sin opción.

**Test sellado.** `SplitLoader.load("test")` exige `allow_test=True`. El runner
nunca lo pasa. El comando aparte `evaluate-test` registra cada acceso en
`runs/test_evaluations.jsonl`.

**Pliegues.** `make-folds` materializa 5 pliegues agrupados sobre train+valid en
`splits/folds/fold_{k}.txt`, congelados igual que las particiones. Test queda
fuera.

## Métricas

Las dos que deciden, porque miden si el recorte sirve:

- **Cobertura** — fracción del billete real dentro del recorte predicho con
  margen. Un recorte que corta media mancha arruina el clasificador de aguas
  abajo. Objetivo ≥ 0.98 en el percentil 5.
- **Contaminación** — fracción del recorte que pertenece a otro billete. El
  fondo es ruido inocuo; un trozo del vecino puede meter una mancha ajena y
  provocar un falso positivo. **Dos umbrales**, ver abajo.

### Los dos umbrales de contaminación

La distribución es **bimodal, no continua**, así que un umbral único sería inútil
en los dos sentidos a la vez. Medido con un detector perfecto (predicción =
verdad) sobre train+valid del export actual, a margen 0.05:

| escena | n | mediana | p95 | umbral |
|---|---|---|---|---|
| un billete | 301 | 0.0000 | **0.0000** | 0.01 |
| abanico | 378 | 0.0231 | **0.8906** | 0.92 |

En **imágenes de un solo billete** el suelo es cero exacto: no hay nada más en la
imagen que pueda ensuciar el recorte, así que cualquier contaminación es un error
real del detector. El umbral 0.01 es holgado respecto al suelo y estricto en
términos absolutos, que es lo que se quiere.

En **abanicos** ni un detector perfecto baja del 0.89, y no por ser malo: si un
billete está parcialmente tapado, su caja contiene por fuerza píxeles del que lo
tapa. Es geometría, no error. El umbral 0.92 deja ~3 puntos de holgura, un cuarto
del recorrido que queda hasta 1.0.

**Aviso al leer el umbral de abanico:** entre el suelo (0.89) y el techo (1.0)
quedan 11 puntos, así que discrimina poco — solo caza detectores bastante peores
que el perfecto. Para comparar candidatos en abanicos mira la **mediana**, que el
detector perfecto deja en 0.023 y tiene recorrido de sobra.

Una imagen cuenta como abanico si tiene **más de un billete presente**, contando
también los que el filtro de área descartó: un vecino filtrado sigue en los
píxeles y sigue ensuciando.

Los dos números salen de los datos, así que hay que rederivarlos cuando cambie el
export: `contamination_floor()` en `metrics/crop.py` los recalcula.

### [PENDIENTE] Enmascarado por polígono en el recorte

Mejora pendiente para la etapa de recorte, **fuera del alcance actual**.

Hoy el recorte es la caja predicha con margen, así que en abanico arrastra
inevitablemente píxeles del billete de delante. Pero en inferencia se tienen
todas las detecciones de la imagen, no solo una: se pueden **enmascarar los
polígonos de los demás billetes** dentro del recorte antes de pasárselo al
clasificador de manchas. La contaminación pasaría de ser píxeles de otro billete
—que pueden meter una mancha ajena— a ser fondo, que es ruido inocuo.

Bajaría el suelo de 0.89 y haría el umbral de abanico mucho más discriminante.
Toca la etapa que alimenta al clasificador, que queda fuera de este alcance.

Ambas dependen del **margen de recorte**, que por eso vive en la config
(`crop.margin`), se registra en `metrics.json` y se reporta barrido en
`{0.00, 0.05, 0.10}`: el margen intercambia una métrica por la otra y una sola
fila oculta el intercambio. La contaminación se reporta junto a la del **quad
anotado** con el mismo margen: en abanico el propio ground truth ya contiene
trozos del vecino, y sin ese suelo no se puede separar el error del modelo de la
geometría irreducible.

Detección: mAP50 y mAP50-95 con IoU rotado vía shapely. Ángulo: error módulo
180°, como `min(|Δ|, 180-|Δ|)`. Vértices: distancia mediana y p95, en píxeles y
como fracción del lado mayor — diagnóstico, no criterio de éxito.

Emparejamiento por IoU rotado, umbral 0.5, asignación greedy por confianza
descendente. **Los billetes no detectados cuentan como fallo**, no se excluyen:
excluirlos hace que un detector que solo encuentra los casos fáciles salga mejor,
que es la conclusión invertida. Se reportan por separado tasa de detección, error
condicionado a detección, y agregado.

**Intervalos.** Toda fila de la tabla comparativa lleva `n` e intervalo por
bootstrap, sin excepción. La **unidad de remuestreo es la imagen**, no la
detección: varios billetes de una misma imagen están correlacionados y
remuestrear detecciones estrecha los intervalos artificialmente.

La tabla que decide se calcula sobre el `valid` adoptado. La validación cruzada
de 5 pliegues queda como comprobación del candidato ganador.

**Al interpretar:** por la política de visibilidad, en imágenes con abanico el
modelo puede detectar correctamente billetes que no están anotados y contarán
como falsos positivos. Si ves precisión hundida en esas imágenes, mira las
visualizaciones antes de concluir que el modelo falla.

## Ángulo y cajas casi cuadradas

La cabeza OBB propia predice el ángulo como **`(sin 2θ, cos 2θ)`**, no como un
escalar en radianes. Un rectángulo girado θ y otro girado θ+180° son el mismo
rectángulo; regresar θ directamente castigaría al modelo por acertar —predecir
179° con verdad 1° daría un error enorme siendo 2° de error real—. Con el ángulo
doblado, los dos caen en el mismo punto del círculo y la ambigüedad desaparece
por construcción.

**Eso resuelve la periodicidad de 180°, pero no el intercambio de lados.** Cuando
`w ≈ h`, la caja `(w, h, θ)` y la caja `(h, w, θ+90°)` describen el mismo
rectángulo y la codificación las manda a puntos opuestos: dos objetivos
contradictorios para la misma caja.

Por eso la pérdida de ángulo se **atenúa** cuando el ratio de la caja verdadera
es bajo. Si el rectángulo es casi cuadrado, el ángulo apenas cambia el recorte
—la política de anotación ya dice que un rectángulo aproximado basta—, así que no
tiene sentido gastar capacidad castigando algo que ni está bien definido ni
altera el resultado.

Todo es parametrizable en `detector.loss.angle_weight`: `enabled`,
`ratio_threshold`, `min_weight` y `decay` (`linear`, `smoothstep`, `quadratic`,
`step`). Va en la config y no clavado en el código **para poder medirlo**: la
pregunta «cuánto aporta esto» se responde entrenando con y sin, y ambas
ejecuciones quedan registradas con su config.

Se descartó la representación gaussiana (GWD/KLD), que absorbería la ambigüedad
de forma natural, porque se aleja del IoU rotado por shapely con el que medimos.
Esa trazabilidad pesa más que la elegancia de la formulación.

### [SOSPECHA] Ese 19% probablemente es un artefacto

El atenuador existe porque **145 de 762 anotaciones (19%)** tienen ratio < 1.1 —
el mismo 19% que dispara el aviso de ancla inestable. Pero hay motivos para creer
que ese número **no describe el dominio, sino el export**:

| | |
|---|---|
| Ratio real de un billete de euro | **~1.95:1** en todas las denominaciones |
| Mediana de ratio medida | **1.45** |
| Imágenes a 416×416 | 489 de 502 |

Un billete es 1.95:1 en el mundo. Que la mediana salga en 1.45 apunta a la
deformación del redimensionado a cuadrado, no a que los billetes aparezcan
escorzados. Las imágenes llegaron ya redimensionadas en origen, así que la
distorsión no se puede deshacer desde aquí.

**Si en algún momento se resuben los originales sin redimensionar**, hay que
volver a medir la distribución de ratios. Es bastante probable que el 19% se
desplome y que el atenuador deje de hacer falta — en cuyo caso `enabled=False`
y a otra cosa. Medirlo antes de quitarlo, no al revés.

## [DESCARTADO] buzhidaoshenme/YOLOX-OBB

Estaba en la lista de candidatos. Se deja fuera, y estos son los motivos, todos
verificados contra el repositorio y no supuestos.

**Su esquema XML no es el que se asumía.** Durante un tiempo este proyecto dio
por hecho que leía `<robndbox>` de roLabelImg. Leído su `dota_obb.py`, lo que
espera es un `<bndbox>` de VOC con un `<angle>` **dentro**, y las coordenadas
parseadas con `int(...) - 1`:

```xml
<object>
  <name>...</name><difficult>0</difficult>
  <bndbox>
    <xmin>..</xmin><ymin>..</ymin><xmax>..</xmax><ymax>..</ymax>
    <angle>..</angle>
  </bndbox>
</object>
```

Eso no es un rectángulo girado: es una envolvente alineada con un ángulo pegado.
Reconstruir el quad desde ahí depende de un convenio que habría que deducir
leyendo su código de decodificación.

**Usa pérdida KLD**, la formulación gaussiana que este proyecto descartó a
propósito por alejarse del IoU rotado de shapely con el que medimos. Meterlo
daría un candidato optimizando algo distinto de lo que se mide, y sin forma de
separar si su resultado viene del backbone o de la pérdida.

**Está abandonado.** Último commit en noviembre de 2021, 14 issues abiertas. Es
un fork de YOLOX de la época de PyTorch 1.9.

**Y no compensa.** 9.0M parámetros frente a los 857k de la cabeza propia, que
además controlamos entera. Su única ventaja sobre ella era el preentreno, y no
justifica el resto.

Si alguna vez se retoma, el trabajo pendiente es un exportador para ese esquema
concreto — no el `voc_xml` que hay, que escribe `<robndbox>` y **no le sirve**.

## Entorno para RTMDet-R

RTMDet-R es Apache 2.0 y apto para producción, pero **no entra en el entorno
principal**. MMCV lleva operadores compilados —NMS rotado, IoU rotado— enlazados
contra la ABI binaria de una versión concreta de PyTorch, y de ahí sale una
cadena de restricciones que empuja el proyecto entero dos años atrás.

### La cadena, y por qué acaba en torch 2.0

```
mmrotate 1.0.0rc1   ->  mmdet >=3.0.0rc6, <3.2.0
mmdet 3.1.0         ->  mmcv  >=2.0.0rc4, <2.1.0
mmcv 2.0.x          ->  solo existe en el índice de torch 2.0
```

Cada eslabón empuja al siguiente. Y **ninguna de estas incompatibilidades la ve
el resolutor de dependencias**: son `assert` dentro del `__init__.py` de cada
paquete, que solo saltan al importar.

Ruedas `cp311-win_amd64` disponibles, medido:

| índice | mmcv disponibles |
|---|---|
| `torch2.14` | el índice no existe |
| `torch2.4` | solo `manylinux` |
| `torch2.1` | 2.1.0, 2.2.0 — **las dos por encima del tope de mmdet** |
| **`torch2.0`** | **2.0.0, 2.0.1** — las únicas que sirven |

Y `mmrotate` **no está publicado en PyPI en su línea 1.x**: en PyPI solo hay
0.3.4, que va con `mmcv-full` 1.x y `mmdet <3`. Hay que instalarlo desde la rama
`dev-1.x` de GitHub.

### Receta verificada

```bash
uv venv .venv-rtmdet --python 3.11
VIRTUAL_ENV=.venv-rtmdet uv pip install torch==2.0.0 torchvision==0.15.1   "numpy==1.26.4" packaging "setuptools<81"
VIRTUAL_ENV=.venv-rtmdet uv pip install --only-binary=:all:   --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html   mmcv==2.0.1
VIRTUAL_ENV=.venv-rtmdet uv pip install mmdet==3.1.0   "git+https://github.com/open-mmlab/mmrotate@dev-1.x" "numpy==1.26.4"
```

Comprobado de punta a punta: los cuatro paquetes importan, `box_iou_rotated`
calcula (0.6337 en el caso de prueba), `RotatedRTMDetHead` está en el registro, y
Ultralytics sigue cargando YOLO26-obb en ese mismo entorno.

**`numpy<2` es la parte frágil.** Torch 2.0 es anterior a NumPy 2 y no lo
restringe; `mmcv` y `mmengine` piden `numpy` sin tope. El resolutor lo sube solo
y `torch.from_numpy` empieza a fallar con *"Numpy is not available"*. Cualquier
instalación posterior en ese entorno puede volver a romperlo **en silencio**:
hay que repetir el pin en cada `uv pip install`.

### [DESVIACIÓN] El preentreno COCO no existe para tiny

La especificación pedía *"la config con preentreno COCO, no ImageNet"*. Para la
variante `tiny` —la que encaja con el despliegue móvil— **esa config no está
publicada**:

```python
# rotated_rtmdet_tiny-3x-dota.py
checkpoint = '.../cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
```

El único `coco_pretrain` es el de la variante `l`, ~52M parámetros: diez veces
el presupuesto móvil.

**Lo que se usa en su lugar**, y por qué es mejor que ambas opciones: los
checkpoints publicados no son preentrenos de backbone, son **detectores rotados
ya entrenados en DOTA**.

| checkpoint | mAP en DOTA |
|---|---|
| `rotated_rtmdet_tiny-3x-dota` | 75.60 |
| `rotated_rtmdet_tiny-3x-dota_ms` | 79.82 |

Eso es más que «aprendió a localizar en COCO»: es un modelo que ya predice
**cajas orientadas**, que es exactamente nuestra tarea. Partir de ahí y ajustar
sobre 351 imágenes es mejor punto de partida que cualquier preentreno de
clasificación. La cadena real es ImageNet → DOTA, y el destino es lo que importa.

### Al comparar

Un candidato entrenado aquí corre con **torch 2.0** mientras el candidato propio
y Ultralytics corren con **2.14**. El `run.json` registra la versión, así que la
diferencia es visible en `compare`, pero sigue siendo una comparación entre
entornos separados por dos años de PyTorch. Conviene decirlo al leer los números.

## Licencias

Sin código AGPL ni GPL en producción. El registro distingue **dos ejes
independientes**:

- `production_ready` — licencia del **código**. Ultralytics es AGPL-3.0, así que
  queda en `False`: solo referencia de rendimiento, aislado tras la interfaz
  común y en grupo opcional de dependencias. Un test comprueba que ningún módulo
  fuera de su adaptador importa `ultralytics`.
- `restricted_pretrain` — licencia de los **datos de preentreno**. DOTA-v1.0 se
  distribuye solo para uso académico y eso alcanza a los pesos derivados.

Los dos ejes se separan a propósito: cada candidato apto se instancia en dos
variantes, con y sin preentreno DOTA, y `compare` las muestra como filas hermanas
con la columna marcada. Nada queda descartado de antemano.

Los datasets llevan `license`, `production_ready` y `sources: list[str]` con la
licencia de cada fuente, porque la licencia de un agregado no anula la de sus
fuentes. `compare` marca una ejecución como no apta si cualquier dataset de su
cadena lo es.

## Uso

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"

# datos sintéticos mientras no llega el export real
python tools/make_synthetic.py --out data/synth_roboflow --count 120 --layout roboflow

python -m testbank.cli --data-root data/synth_roboflow detect
python -m testbank.cli --data-root data/synth_roboflow --splits-dir splits make-splits
python -m testbank.cli --splits-dir splits make-folds
python -m testbank.cli --splits-dir splits check-visibility --json-out runs/_inspection/visibility.json
python -m testbank.cli --splits-dir splits inspect --split valid --count 9

pytest -q
```

## Estado

Hecho: estructura y `pyproject`, quad canónico con las cinco invariantes del
volteo, lector `obb_yolo` con validación estricta, detección de estructura y
materialización de splits en ambos modos, chequeo de visibilidad, visualización
de inspección.

Pendiente: conversores de formato, motor de experimentos, y solo entonces los
candidatos. El export real aún no ha llegado; todo está construido contra datos
sintéticos generados por `tools/make_synthetic.py`.
