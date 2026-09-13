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

## Salvedades

Todo lo que hay que saber antes de creerse un número de este proyecto. Cada
entrada dice dónde está tratada; ninguna es un descuido pendiente de descubrir.

### Sobre los datos

**El aspecto de los billetes está deformado y no se recupera.** 489 de 502
imágenes llegan ya a 416×416 desde el export, aplastadas al cuadrado. Un billete
real es ~1.95:1 y la mediana medida es 1.45. De ahí sale el 19% de cajas casi
cuadradas que motiva el atenuador de ángulo. → *[SOSPECHA] Ese 19%…*

**Hay fuga entre particiones en el export de Roboflow, y ahora sí se puede
arreglar.** Imágenes casi idénticas —la misma toma, dos disparos seguidos— caen
en particiones distintas. Medido en color:

| | |
|---|---|
| Pares que cruzan | 27 |
| De ellos, con `test` a un lado | **12** |
| Imágenes de `test` implicadas | **10 de 50 (20%)** |

**Corrección: este README dijo 36% durante un tiempo, y era falso.** La primera
medida se hizo con miniaturas en gris, y en gris 18 de los 93 pares «idénticos»
eran billetes de *distinto valor* con el mismo encuadre de foto de stock: un 50
naranja y un 500 morado son la misma silueta en gris. El color los separa (el
duplicado real se queda en 0.98; los falsos caen a 0.45–0.52). Salió a la luz al
estratificar la partición por tipo de billete, que se negó a repartir un grupo
que cruzaba tipos. → *Por qué en color* en `data/duplicates.py`

Un quinto del test sigue contaminado, y `evaluate-test` sobre la partición del
export **saldrá inflado**. Pero la partición ya no es intocable:

```bash
testbank find-duplicates --manifest-out runs/_inspection/dups.json
testbank make-splits --repartition \
    --group-key manifest --group-manifest runs/_inspection/dups.json \
    --stratify-regex '^(\d+|Multiple)_'
```

Los ratios y la semilla se eligen (`--ratios 0.8,0.1,0.1 --seed 1`) y quedan en
el manifiesto junto a un digest de la partición: misma semilla, mismos datos,
mismos ficheros byte a byte — comprobado sobre el export real, y con test.

`--repartition` es la excepción explícita a «en modo adoptar manda el export»:
junta todas las muestras y reparte **por grupos** (las dos tomas de la misma foto
van juntas) y **por estratos** (cada partición recibe su parte de cada tipo de
billete). Medido sobre el export real: 355 / 76 / 71, los ocho tipos en cada
lado, y **cero pares cruzando**. El manifiesto lo deja escrito como
`mode: repartition` para que nadie lo confunda con la partición original.

→ `testbank find-duplicates`, `testbank make-splits --repartition`,
`data/duplicates.py`, `data/splits.py`

**El 64.6% de las anotaciones sigue alineado al eje.** Tras la recorrección son
cajas correctas —muchos billetes se fotografían rectos— pero significa que la
ventaja de OBB sobre una caja alineada es menor de lo que sugiere la premisa del
proyecto. Medido: un detector alineado con localización perfecta llega a mAP50
0.939 frente a 1.000. La mediana de error de los formatos *lossy* es de 2 px.

**La contaminación tiene un suelo de 0.89 en abanicos.** Ni un detector perfecto
baja de ahí: si un billete está parcialmente tapado, su caja contiene por fuerza
píxeles del que lo tapa. → *Los dos umbrales de contaminación*

**`check-visibility` puede dar 0 y no significar nada.** Solo compara quads
anotados entre sí, así que una oclusión causada por un billete sin anotar le es
invisible. → *Cómo leer `check-visibility`*

**Queda una anotación degenerada en Roboflow.** Se borró en local
(`005_Euro_328`, un clic suelto), pero el próximo export la trae de vuelta.

**El dataset es CC BY 4.0: exige atribución** allí donde se distribuya el modelo
o los datos. → `data/datasets.py`

### Aproximaciones asumidas

**Se entrena con un IoU aproximado y se mide con el bueno.** El IoU rotado por
shapely no cabe en el bucle de entrenamiento, así que la asignación usa
envolventes alineadas y la pérdida se descompone en forma + giro. La evaluación
sigue usando shapely, de modo que si la aproximación reparte mal el compromiso,
el número final lo delata. → `models/assign.py`, `models/losses.py`

**Se entrena con la verdad recortada y se mide con la sin recortar.** La política
`clip` ajusta la caja al marco; las métricas evalúan contra el quad original.
Medido: el percentil 5 de cobertura no se mueve (1.0000 en validación), solo la
cola (p1 = 0.936). El margen de recorte absorbe casi todo.

**`clip` devuelve rectángulos, y antes no.** Hasta ahora pinzaba cada vértice a
[0,1] por separado. Es lo obvio y está mal: un rectángulo **girado** cortado
contra un marco recto da un trapecio, no un rectángulo más pequeño. Medido sobre
los datos reales, **82 de 679 anotaciones (12%) dejaban de ser rectángulos**, y
el único run registrado hasta entonces entrenó así.

**Lo que el cambio NO es: una mejora de calidad.** El argumento inicial era que
un trapecio es un objetivo inalcanzable para un modelo que predice
`(cx, cy, w, h, ángulo)`. Es verdad a medias, y la mitad que falta importa:
`quad_to_box` ya rectangularizaba cualquier quad, tomando dos de sus cuatro
lados. O sea que la red nunca vio un trapecio — vio una caja deducida de él en
silencio. Medido el objetivo real contra el billete visible:

| IoU del objetivo que ve la red | mediana | p05 | mín |
|---|---|---|---|
| Recorte viejo | 0.9828 | 0.8361 | 0.7541 |
| Recorte nuevo | 0.9852 | 0.7933 | 0.6371 |

Delta mediano **+0.0002**: mejora en 57 casos de 99 y empeora en 42, y la cola
queda algo peor. Es un empate.

**Lo que sí es:** coherencia. Lo que se escribe en disco ahora es lo que dice ser
—un rectángulo—, la conversión implícita de `quad_to_box` deja de estar oculta, y
`clip` deja de estar vetado en los formatos que solo representan rectángulos. Los
casos de la cola, billetes muy salidos del marco, son precisamente los que `pad`
resuelve bien y `clip` no puede.

Ahora se encogen los extremos de la caja **en sus propios ejes** hasta que las
cuatro esquinas caben, con el ángulo intacto. Resultado sobre los datos reales:

| | antes | ahora |
|---|---|---|
| No rectángulos | 82 / 679 | **0 / 679** |
| Fuera de [0,1] | 0 | **0** |
| Cobertura de lo visible | — | mediana 0.985, p05 0.793, mín 0.637 |
| Ángulo conservado | — | 96 de 99 exacto |

Los 3 que cambian de ángulo lo hacen 90° justos y todos tienen ratio < 1.1: es el
intercambio de lados de las cajas casi cuadradas que ya documenta § *Ángulo y
cajas casi cuadradas*, no un error de geometría — el rectángulo es el mismo, solo
cambia qué lado se etiqueta como largo.

Dos cosas salieron de medir y no se ven leyendo el código, así que están fijadas
con tests de regresión en `tests/test_clip_rectangulo.py`:

- **Envolver "lo visible" no sirve.** Cortar una **esquina** deja intactas las
  otras tres, que son las que fijan la envolvente: las 99 anotaciones fuera de
  marco seguían fuera, hasta un 23% del lado.
- **Cada restricción va al eje más alineado con ella.** Sin eso el ajuste es
  válido y aun así desastroso: en `Multiple_Euro_154`, un billete casi horizontal
  que se sale 0.029 por arriba, la restricción `y >= 0` se «arreglaba» encogiendo
  el lado **largo** de 0.890 a 0.064. Cumplía todo y conservaba el 8% del billete.

Consecuencia práctica: los formatos que solo representan rectángulos dejaron
de necesitar un veto a `clip`. Esos formatos (`voc_xml`, `yolox_obb_voc`) se
quitaron después en el refactor por falta de consumidor, y con ellos el veto:
`dota` y `coco` aceptan cualquier cuadrilátero.

**El atenuador de ángulo puede sobrar.** Existe por el 19% de cajas casi
cuadradas, que probablemente es artefacto del redimensionado. Si se resuben los
originales, hay que volver a medir **antes** de quitarlo. → `enabled=False`

**Sin letterbox al redimensionar**, porque el aspecto ya se perdió en origen y
añadirlo ahora no recupera nada. → `models/data.py`

**Con `out_of_bounds=pad` e inferencia sin padding, los números miden el pipeline
desajustado.** Sale un `AVISO` en el `run.json` y en el resumen. →
`pad_at_inference`

### Fragilidades de entorno

**`opencv-python` y `opencv-python-headless` están los dos instalados.**
Ultralytics arrastra el primero; ambos ocupan el espacio de nombres `cv2` y gana
el último instalado. Se registran las dos versiones en cada ejecución para que la
colisión sea visible. → `experiment/provenance.py`

**El entorno de RTMDet-R se rompe solo si alguien instala algo.** `numpy<2` no lo
declara nadie y el resolutor lo sube sin avisar. → *Entorno para RTMDet-R*

**RTMDet-R usa `mmrotate` desde una rama de desarrollo sin publicar.**

**Los candidatos corren en entornos distintos** — torch 2.0 para RTMDet-R, 2.14
para los demás. El `run.json` registra la versión, pero sigue siendo una
comparación entre entornos separados por dos años de PyTorch.

**El torch del entorno principal es CPU.** Vale para los circuitos cortos y
para un Mac (MPS); para líneas base largas en PC hay que instalar el de CUDA en
ese mismo entorno. → *Uso*

### Sin probar todavía

- **RTMDet-R no ha entrenado nunca.** Verificado que la config se construye y el
  modelo se instancia (4.873.470 parámetros); no se ha lanzado un entrenamiento.
- **La partición re-hecha no se ha usado para entrenar.** `--repartition` está
  medido sobre el export real (cero pares cruzando, los ocho tipos en cada lado)
  pero la partición activa en `splits/` sigue siendo la del export. Cambiarla es
  una decisión, y anula toda comparación con ejecuciones anteriores.
- **`evaluate-test` no se ha ejecutado.** Deliberado: gastaría un acceso al
  conjunto sellado sobre un modelo que no significa nada.
- **Ninguna línea base larga se ha lanzado.** Todo lo medido son pruebas de
  circuito de 1 o 2 épocas, cuyos números no significan nada.

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

**`--repartition`: la excepción explícita.** Ignora la partición del export y
reparte de cero como en modo crear. Existe porque la de Roboflow separa tomas
casi idénticas de la misma foto. Es opt-in, y el manifiesto lo escribe como
`mode: repartition` con el modo del export al lado, para que nadie confunda esta
partición con la original. Si se intenta *adoptar* con un manifiesto de grupos
que demuestra que el export reparte alguno, se niega y el mensaje apunta aquí:
la fuga no se congela en silencio.

**Se reparten grupos, no fotos.** Con `--group-key manifest` y el JSON que
escribe `find-duplicates`, las dos tomas de la misma foto caen en el mismo lado.
Consecuencia: los ratios se aplican sobre el número de grupos, así que 0.8 sobre
441 grupos da 394 fotos y no 402. Es lo correcto —partir un grupo para cuadrar
un número sería reintroducir la fuga— pero el recuento exacto no sale redondo.

**`--stratify-regex`: representativos de cada tipo en cada lado.** Una regex
con un grupo de captura sobre el nombre de la muestra; para el export de
Roboflow, `'^(\d+|Multiple)_'` saca el tipo de billete. El reparto se hace
dentro de cada estrato y luego se junta, así que cada partición recibe su
proporción de cada tipo. El comando imprime la tabla estrato × partición y el
manifiesto la guarda. Dos límites honestos: con pocos grupos por tipo el redondeo
puede dejar la partición pequeña sin alguno (`round(2 × 0.1) = 0`), y un grupo
que cruce tipos —dos fotos «iguales» con billetes de distinto valor— se reparte
igualmente y se avisa, porque agrupar de más es barato y bloquear no.

**`--ratios` y `--seed`.** `--ratios 0.8,0.1,0.1` (deben sumar 1); `--seed N`.
Misma semilla, mismos datos, mismos ficheros byte a byte: comprobado sobre el
export real y anclado con test. El manifiesto guarda ratios, semilla y un
digest de la partición para verificarlo después.

```bash
testbank find-duplicates --manifest-out runs/_inspection/dups.json
testbank make-splits --repartition --ratios 0.8,0.1,0.1 --seed 1 \
    --group-key manifest --group-manifest runs/_inspection/dups.json \
    --stratify-regex '^(\d+|Multiple)_'
```

`--group-key` acepta `filename-prefix`, `directory`, `manifest` y `none`. El
valor `none` afirma que cada imagen es independiente, y exige
`--i-confirm-independence` para que sea una afirmación explícita y no un
descuido: si existen varias tomas del mismo billete físico repartidas entre
particiones, las métricas salen infladas. La confirmación **solo se exige al
crear**; en modo adoptar no elegimos el reparto, únicamente lo congelamos.

**Integridad al cargar.** Una muestra en dos particiones es error fatal con
mensaje que la identifica. Con clave de grupo, también un grupo repartido.

**Sin `extend-splits`.** Hubo un comando para anexar muestras nuevas a una
partición ya materializada; se quitó en el refactor. Si llega un export nuevo,
o se adopta su partición o se re-parte con `--repartition`: dos caminos que
hacen cosas distintas con los datos nuevos era una ambigüedad de más.

**Test sellado.** `SplitLoader.load("test")` exige `allow_test=True`. El runner
nunca lo pasa. El comando aparte `evaluate-test` registra cada acceso en
`runs/test_evaluations.jsonl`.

**Sin validación cruzada.** Hubo un `make-folds`; se quitó en el refactor
porque la validación cruzada quedó descartada y el runner de pliegues nunca se
escribió. El manifiesto de casi-duplicados sirve ahora para `--repartition`.

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

## Preentreno: qué carga cada candidato, y el port que lo hace posible en CPU

Hasta aquí ningún candidato propio leía un checkpoint: la cabeza propia
entrenaba de cero y el port de DDGRCF no existía. Las dos cosas han cambiado.

### La cabeza propia carga el COCO de Megvii

`yolox_s.pth.tar` —el YOLOX-S oficial, Apache-2.0, el mismo fichero que trae el
fork de buzhidaoshenme— **encaja entero en nuestro backbone y cuello de
`small`**. Medido: 354 tensores y 7.066.683 parámetros en los dos, formas
idénticas. Sólo cambian los nombres: ellos envuelven el backbone como
`backbone.backbone.*` y llaman a los módulos del cuello `lateral_conv0`,
`C3_p4`, `reduce_conv1`…; aquí son `backbone.*` y `neck.lateral_c5`, `neck.p4`,
`neck.lateral_c4`… El mapeo está en `models/pretrained.py`, emparejado por
función y comprobado forma a forma al cargar. Su cabeza (80 clases, sin ángulo)
se descarta: lo que se hereda es *saber ver*, no *saber dónde está el billete*.

```bash
testbank train yolox-obb-small --pretrained weights/yolox_s.pth.tar
```

`--pretrained` va a la config y queda congelado en la ejecución: dos runs con y
sin preentreno no son comparables, y el `config.yaml` lo tiene que decir.
Cargar la variante equivocada (`yolox_s` en `nano`) es un **error**, no un
aviso: entrenar «preentrenado a medias» sin saberlo es peor que entrenar de
cero. Megvii publica también `yolox_tiny.pth` y `yolox_nano.pth` en las releases
de GitHub —descarga directa, sin Baidu— y **los tres cargan enteros en su
variante**: 354 tensores en `small` y `tiny`, 462 en `nano` (las convoluciones
*depthwise* se parten en dos). Las tres arquitecturas propias son idénticas a
las suyas; medido, no supuesto. Van en `weights/`, ignorada por git.

Efecto en una época de circuito, para calibrar: `classes 0.24` frente a `1.87`
de cero, `box 0.57` frente a `0.86`. No es un resultado, es la señal de que el
mapeo carga algo que sirve.

### El port de DDGRCF/YOLOX_OBB: entrena en CPU y en MPS

`yolox-obb-ddgrcf-port` es su red reescrita en torch puro (`models/ddgrcf.py`),
capa a capa siguiendo su yaml y **con sus mismos nombres de parámetro**, para
que su checkpoint de DOTA cargue con `strict=True` sin ningún mapeo. Verificado
contra su modelo construido de verdad (con sus operadores sustituidos por un
stub, que sólo se llaman al entrenar): **426 claves y formas idénticas,
8.051.797 parámetros**, y con sus pesos cargados la salida coincide con la suya
—obj, cls y θ exactos; cajas decodificadas a 3e-5 px—. Eso valida a la vez el
port y nuestro decodificador para su regresión `(dx, dy, log w, log h)`.

Un detalle que costó un intento: su yaml pone `n=2` en los tallos de las
cabezas, pero su parser aplica el multiplicador de profundidad, `round(2 ×
0.33) = 1`, y queda **una** conv sin `Sequential`. Con dos, el port tenía 8,94M
parámetros y las claves `model.27.0.*` en vez de `model.27.*`.

Lo que sustituye a sus operadores compilados:

| Suyo (C++/CUDA) | Aquí |
|---|---|
| `box_iou_rotated` en el SimOTA | `models/overlap.py`: IoU exacto de polígonos en torch, 18,6 ms por imagen |
| PolyIoU en la pérdida | El mismo, diferenciable; coincide con shapely a 2,5e-6 en 2.000 pares |
| `nms_rotated` | Nuestro `rotated_nms` |
| Su `Trainer` y `DataPrefetcher` (CUDA) | El bucle de testbank |

La receta `ddgrcf` reproduce su `get_losses`: PolyIoU ×5 + obj + cls por IoU +
L1 tardía, todo `Σ / num_fg`, SimOTA con `−log IoU`. Una desviación, dicha: su
L1 tardía deja el objetivo del **ángulo a cero** (`get_reg_l1_target` rellena 4
de 5 componentes), casi seguro un descuido; aquí va el ángulo real. Y lo que el
port **no** reproduce: su *dataloader* (mosaico, mixup, resampling), su
optimizador y su EMA.

Es **apto para producción** —nada compilado, ningún clon— y es hoy el único
candidato con preentreno DOTA que puede entrenar en un M4. Sus pesos están en
Baidu Pan (hace falta su cliente): `--pretrained weights/yolox_s_dota1_0.pth`.

### La entrada iba en el formato equivocado, y con preentreno se notaba

El port con los pesos de DOTA dio **mAP 0,000** tras una época, peor que `small`
con sólo COCO. Puntuación máxima 0,004, y todas las cajas en la misma esquina
en todas las imágenes: la red no localizaba nada. Con backbone *y* cabeza de
regresión entrenados en DOTA, eso sólo pasa si la imagen llega en otro formato.

Y llegaba: nuestro bucle daba **RGB en [0, 1]**; YOLOX —Megvii y DDGRCF por
igual— espera **BGR crudo en 0–255**, sin normalizar. Canales cambiados y escala
255 veces menor en la primera capa. Entrenando de cero no importa (la BatchNorm
absorbe la escala); con preentreno lo destroza. `models/data.py:image_to_input`
fija ahora la convención de YOLOX en un solo sitio, para entrenamiento e
inferencia.

Mismo circuito de una época, antes y después:

| | RGB [0, 1] | BGR 0–255 |
|---|---|---|
| Port + pesos DOTA | 0,000 | **0,105** [0,075–0,153] |

No es una línea base —es una época— pero es la primera vez que un circuito da
un número que no es cero por una razón que se entiende.

### Lo que salió al probarlo

Con el backbone COCO recién cargado y la cabeza sin entrenar, la red predijo un
vértice en −0,54 normalizado y `Quad` lo rechazó con un `QuadError`: **la
evaluación entera se cayó**. Habría tumbado una línea base real a la primera.
Y cuando el port con DOTA produjo por fin detecciones, la visualización de la
ejecución cayó por lo mismo (`'NoneType' object has no attribute 'points'`):
ningún circuito anterior había producido una predicción así. Arreglado en los
dos consumidores que quedaban, con test.

La primera corrección descartaba esas predicciones, y eso era un regalo a la
métrica: una caja que el modelo predijo mayormente fuera de la imagen es un
falso positivo que cometió. Ahora la predicción se emite **sin geometría**
(`Prediction.quad = None`), no empareja con nada y **cuenta como falso positivo
a su puntuación**. Es lo que hacen en la práctica los frameworks de referencia,
que no descartan. Con test de regresión sobre el emparejamiento.

## Recetas de pérdida: tres redes sobre un mismo backbone

La cabeza propia puede entrenarse con **tres recetas de pérdida completas**,
cada una la de una red concreta y sin mezclar componentes entre ellas:

```bash
testbank train yolox-obb-nano --loss-recipe own
testbank train yolox-obb-nano --loss-recipe yolox_obb_fork
testbank train yolox-obb-nano --loss-recipe ultralytics_obb
```

| Receta | Caja | Otros términos | Asignador | Cabeza |
|---|---|---|---|---|
| `own` | `1 − IoU` alineada | ángulo (coseno sobre `sin 2θ, cos 2θ`, atenuado), obj BCE, cls BCE | SimOTA, coste `−log IoU` | directa |
| `yolox_obb_fork` | KLD ×5 | obj BCE, cls BCE con objetivo `one-hot × solape`, L1 tardía; todo `Σ / num_fg` | SimOTA, coste = KLD | directa |
| `ultralytics_obb` | `1 − ProbIoU` ×7.5 | DFL ×1.5, cls BCE ×0.5 con objetivo **suave**; sin objectness; todo `Σ / Σ objetivos` | TAL (`topk=10, α=0.5, β=6`) | **DFL** (16 bins/lado), ángulo escalar, sin obj |

Una receta no es una pérdida de caja: es asignador + objetivos + términos +
normalización + ganancias. Por eso `ultralytics_obb` **cambia la cabeza**: su DFL
exige que cada distancia se prediga como una distribución, y su BCE de clase sin
objectness necesita los objetivos suaves de TAL. Entrenar «su pérdida» sobre
nuestra regresión directa habría sido comparar otra cosa con su nombre puesto.
La receta fija la cabeza (`HeadSpec`), y la cabeza viaja **dentro del
checkpoint**: cargar unos pesos reconstruye la que los produjo.

### Licencias, y qué es fiel y qué no

**El fork es Apache-2.0 y se leyó entero.** Se reproduce su `get_losses`:
mismos términos, mismas ganancias (`reg_weight = 5.0`, `τ = 1.0`), misma
normalización, mismo objetivo de clase, mismo SimOTA con la KLD como coste. La
KLD propia está anclada numéricamente contra su fórmula en
`tests/test_overlap.py` (500 pares aleatorios, `atol 1e-5`). Dos desviaciones,
las dos de cabeza: ellos regresan el ángulo en grados directamente —con el salto
en ±90°— y aquí se mantiene `(sin 2θ, cos 2θ)`; y su L1 tardía va sobre
`(dx, dy, log w, log h)` mientras aquí va sobre las cuatro distancias, que es la
regresión cruda de *esta* cabeza. Con circuitos cortos la L1 está encendida
desde la primera época, igual que en el fork cuando `no_aug_epochs ≥ max_epoch`.

**Ultralytics es AGPL y no se ha leído ni copiado una línea.** La *composición*
—qué términos, qué ganancias, qué asignador— sale de su documentación pública.
Las *fórmulas* salen de los papers: ProbIoU (Llerena et al., 2021), DFL (Li et
al., 2020), TAL (Feng et al., "TOOD", 2021). La ProbIoU se verifica contra una
implementación matricial independiente de la distancia de Bhattacharyya; el
test de aislamiento sigue garantizando que nada fuera del adaptador importa
`ultralytics`. Es una reproducción de la receta descrita, no una copia: si su
código tuviera un detalle no documentado, aquí no está.

### Dos convenciones de covarianza que no se pueden mezclar

Los dos papers convierten `(w, h)` en varianzas de forma distinta —`w²/4` la
KLD, `w²/12` la ProbIoU— y mezclarlas cambia los números sin cambiar el nombre.
`box_to_gaussian` exige el divisor explícito para que no se pueda llamar «a
secas». → `models/overlap.py`

### Lo que está medido

Las tres recetas completan el circuito de 1 época sobre los datos reales, con
pérdidas finitas, gradientes finitos y cada una reportando **sus** términos (la
propia tiene ángulo; el fork ángulo cero y L1; Ultralytics DFL y sin
objectness). El mAP tras una época en CPU es 0 en las tres, como toca: es un
circuito, no un resultado. Las tres son comparables porque comparten backbone,
datos, partición y métrica; lo único que cambia es la receta.

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

## Descartados en el refactor: los dos clones de YOLOX-OBB

Hubo dos adaptadores que envolvían repositorios ajenos clonados y parcheados.
Se quitaron en `feature/refactor_code` porque lo que aportaban ya está en casa,
y los dos exigían CUDA. Se deja lo aprendido, que costó medirlo.

**`buzhidaoshenme/YOLOX-OBB`** (Apache-2.0, abandonado en 2021). Aportaba su
receta KLD y el COCO de Megvii (`yolox_s.pth.tar`, backbone sin rama de
ángulo). Hoy la receta es `--loss-recipe yolox_obb_fork` sobre la cabeza propia
—anclada numéricamente contra su fórmula— y el COCO lo carga `--pretrained` en
las tres variantes. Lo que costó hacerlo arrancar, por si alguien vuelve:
`polyiou` es una extensión C++ con SWIG sin wheel de la que cuelga `import
yolox` entero (se sustituía por shapely: dos nombres, `VectorDouble` e
`iou_poly`); `apex` importado sin condición; `np.int0` y `np.bool` eliminados
en NumPy 2.0; su lector VOC resta 1 a las coordenadas mientras su generador
escribe en base 0; su `evaluate_detections` devuelve `0.0, 0.0` fijo; y su
`DataPrefetcher` está construido sobre `torch.cuda.Stream`. Su README dice
0.712 mAP en DOTA pero no publica ese checkpoint.

**`DDGRCF/YOLOX_OBB`** (Apache-2.0, 2022). Aportaba el único YOLOX con cabeza
OBB **entrenada en DOTA** publicada (`YOLOX_s_dota1_0`, 70.82 mAP@0.5, en Baidu
Pan). Hoy eso es `yolox-obb-ddgrcf-port`: su red portada tensor a tensor y su
receta con el IoU exacto en torch (§ *Preentreno*). El clon exigía compilar sus
operadores C++/CUDA (`box_iou_rotated`, `nms_rotated`, `convex`) con MSVC —aquí
no lo hay: `Microsoft Visual C++ 14.0 or greater is required`— y BboxToolkit,
que sí es Python puro. El camino de datos (`export dota` → `img_split.py`
`--sizes 1024` → sus `.pkl`, 1 parche = 1 imagen) se verificó entero. El
operador compilado estaba *dentro* de su SimOTA, así que no admitía el truco
de shapely del otro fork; por eso el port reimplementa el IoU de polígonos.

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

## Uso: entornos y cómo lanzar cada candidato

Hay **tres entornos**, no uno, y la razón es siempre la misma: dependencias que
no caben juntas. Cada candidato dice cuál necesita.

### Entorno principal — `.venv`

Todo lo que no es un modelo, más la cabeza propia y Ultralytics.

```bash
uv venv --python 3.11
uv pip install -e ".[dev,torch]"          # base + torch (CPU)
uv pip install -e ".[ultralytics]"        # opcional: la referencia AGPL
pytest -q                                 # 495 tests
```

El `torch` que instala es **CPU**. En un Mac con M-series el bucle propio usa
MPS solo; en un PC con NVIDIA hace falta el torch con CUDA de
<https://pytorch.org/get-started/locally/> instalado *en este mismo entorno*.

| Candidato | Entorno | Comando |
|---|---|---|
| `yolox-obb-nano` (857k) | principal | `testbank train yolox-obb-nano --name nano_base` |
| `yolox-obb-tiny` (4.37M) | principal | `testbank train yolox-obb-tiny --name tiny_base` |
| `yolox-obb-small` (7.75M) | principal | `testbank train yolox-obb-small --name small_base` |
| `ultralytics-yolo-obb` | principal + `[ultralytics]` | `testbank train ultralytics-yolo-obb --name ul_ref` |
| `yolox-obb-ddgrcf-port` (8.05M) | principal | `testbank train yolox-obb-ddgrcf-port --pretrained <DOTA.pth>` |
| `rtmdet-r-tiny` | `.venv-rtmdet` aparte | ver abajo |

`train` entrena, evalúa sobre `valid` con las métricas de testbank y deja la
ejecución en `runs/<fecha>_<nombre>/` con la config congelada. **El test no se
toca**: ver *Test sellado* en § *Particiones*.

Opciones que valen para todos:

```bash
--epochs N                  # sustituye al de la config; queda registrado
--out-of-bounds clip|pad|keep   # política de borde; clip por defecto
--loss-recipe own|yolox_obb_fork|ultralytics_obb   # solo cabeza propia; § Recetas
--pretrained ruta.pth       # checkpoint ajeno con el que ARRANCAR; queda en la config
--weights ruta.pt           # reevalúa unos pesos en vez de entrenar
```

Y después de varias ejecuciones:

```bash
testbank compare --csv-out runs/tabla.csv
```

### RTMDet-R — `.venv-rtmdet`, aparte

Exige torch 2.0 y una cadena de versiones que no convive con el entorno
principal. La receta completa y verificada está en § *Entorno para RTMDet-R*;
resumida:

```bash
uv venv .venv-rtmdet --python 3.11
VIRTUAL_ENV=.venv-rtmdet uv pip install torch==2.0.0 torchvision==0.15.1 "numpy==1.26.4" packaging "setuptools<81"
VIRTUAL_ENV=.venv-rtmdet uv pip install --only-binary=:all: --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html mmcv==2.0.1
VIRTUAL_ENV=.venv-rtmdet uv pip install mmdet==3.1.0 "git+https://github.com/open-mmlab/mmrotate@dev-1.x" "numpy==1.26.4"
VIRTUAL_ENV=.venv-rtmdet uv pip install -e .
.venv-rtmdet/Scripts/testbank train rtmdet-r-tiny --name rtmdet_base
```

Sus números salen de torch 2.0 y los demás de torch 2.14: la tabla lo anota.

### Antes de las líneas base

- **`clip` es la política por defecto y devuelve rectángulos** — con el ángulo
  original, dentro del marco. `pad` y `keep` siguen disponibles por flag.
- Lo que hay en `runs/` hasta hoy son circuitos de 1–2 épocas. **Ninguno es una
  medida de calidad.** Conviene vaciarlo, o nombrar las líneas base de forma que
  `compare` no los mezcle.
- Las tres variantes propias, el port de DDGRCF y Ultralytics se pueden lanzar
  hoy, y usan la GPU si la hay (`cuda` → `mps` → `cpu`; `TESTBANK_DEVICE` lo
  fuerza). RTMDet-R necesita su entorno.
- **Decidir la partición.** La activa es la del export, con un 20% del test
  contaminado por casi-duplicados. `make-splits --repartition` la rehace sin
  fugas y estratificada (§ *Particiones*); hacerlo invalida la comparación con
  cualquier ejecución anterior, así que es antes de las líneas base o nunca.

## Estado

**Hecho.** Quad canónico con las cinco invariantes del volteo. Lector `obb_yolo`
estricto, que acumula todos los errores de un fichero en vez de morir en la
primera línea. Detección de estructura y materialización de splits en los dos
modos, chequeo de visibilidad, filtro por área relativa, política de borde
(`clip` / `pad` / `keep`, con `clip` devolviendo rectángulos con el ángulo
original), detección de casi-duplicados en color, y re-partición opt-in por
grupos y estratos con ratios y semilla (`--repartition`). Seis conversores de
formato y cuatro exportadores. Métricas con AP a 101 puntos, IoU rotado por
shapely y bootstrap por imagen. Motor de experimentos con la config congelada en
cada ejecución. Cabeza OBB propia sobre YOLOX en tres variantes (857k / 4.37M /
7.75M parámetros), entrenamiento determinista, atenuador de ángulo por ratio.
Adaptadores: Ultralytics (referencia, `production_ready=False`), RTMDet-R, el
fork de YOLOX-OBB (`predict` verificado; `train` exige CUDA) y las tres variantes
propias, y DDGRCF/YOLOX_OBB **como port en torch puro** (`yolox-obb-ddgrcf-port`,
entrena en CPU/MPS, carga sus pesos de DOTA). Seis candidatos registrados; los
dos clones de YOLOX-OBB se quitaron en el refactor (§ *Descartados*). La cabeza propia carga el **COCO de Megvii**
(`--pretrained`). La cabeza propia entrena con **tres
recetas de pérdida completas** (`--loss-recipe own | yolox_obb_fork |
ultralytics_obb`), cada una la de una red concreta, sobre el mismo backbone.

**Datos reales dentro.** 502 imágenes de Roboflow, 762 anotaciones. Ya no se
depende de `tools/make_synthetic.py`, que se conserva para los tests.

**Pendiente.**

- Entrenar `buzhidaoshenme/YOLOX-OBB`. El adaptador está hecho y `predict` está
  verificado, pero su `Trainer` exige CUDA y este equipo es CPU
  (§ *Descartados en el refactor*).
- Prueba de humo de RTMDet-R: el entorno se resolvió y el `Runner` se construye,
  pero no se ha ejecutado ni una época.
- Enmascarado por polígono para el recorte, hoy rectangular (§ *Salvedades*).
- Líneas base largas y la tabla comparativa. Todo lo ejecutado hasta ahora son
  circuitos de 1–2 épocas para verificar que la cañería no gotea, **no
  resultados**: ningún número de rendimiento de este repositorio es todavía una
  medida de calidad del detector.
- El conjunto de test sigue **sellado**. `evaluate-test` es el único camino y
  deja constancia. Sobre la partición del export, un 20% del test tiene un
  casi-duplicado en train; `--repartition` lo deja en cero (§ *Salvedades*).
