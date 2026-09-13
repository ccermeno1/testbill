"""Parchea un clon de `buzhidaoshenme/YOLOX-OBB` para que arranque sin compilar.

Por que hace falta
------------------
`import yolox` falla tal cual. La cadena es:

    yolox/__init__.py -> yolox.utils -> yolox/utils/boxes.py:7
                                        `from DOTA_devkit_YOLO import polyiou`

y `polyiou` es una extension C++ envuelta con SWIG 3.0.12 que no trae wheel: se
construye con `swig` mas un compilador de C++, y en Windows eso es una tarde.
Es exactamente la dependencia compilada que la especificacion de este proyecto
manda evitar -- "IoU y NMS rotados con shapely o PyTorch puro" -- asi que la
alternativa no es compilarla, es no necesitarla.

Que se parchea, y por que tan poco
----------------------------------
La superficie de `polyiou` que usa la ruta OBB entera son DOS nombres:

    polyiou.VectorDouble([x1, y1, ..., x4, y4])     una lista de 8 dobles
    polyiou.iou_poly(p, q)                          el IoU de dos poligonos

Los dos se reimplementan con shapely en veinte lineas. Y el cambio no es solo
mas barato: es MEJOR alineado, porque shapely es justamente la referencia contra
la que este proyecto mide. Si se compilase `polyiou`, el fork puntuaria con una
implementacion de IoU distinta de la de la tabla comparativa y no habria forma
de separar esa diferencia del modelo.

El segundo parche son las clases: `dota_classes.py` lleva cableadas las 15 de
DOTA y el parser hace `class_to_ind[name]`, asi que `euro_banknote` es un
KeyError hasta que se sustituyen.

Reversible y ruidoso a proposito
--------------------------------
Cada fichero tocado se guarda con sufijo `.orig` antes de escribirse, y el
script dice exactamente lo que cambio. Un parche silencioso sobre un repo ajeno
es como se acaba depurando durante horas un comportamiento que uno mismo
introdujo.

Uso::

    git clone --depth 1 https://github.com/buzhidaoshenme/YOLOX-OBB.git fork
    python tools/patch_yolox_obb_fork.py fork
    python tools/patch_yolox_obb_fork.py fork --check     # solo comprueba
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: El reemplazo de `DOTA_devkit_YOLO/polyiou.py`. Mismo API, cero compilacion.
POLYIOU_SHIM = '''\
"""Sustituto en Python puro de la extension SWIG `polyiou`, con shapely.

Lo instala `tools/patch_yolox_obb_fork.py` de testbank. El original es C++ con
SWIG y hay que compilarlo; esto expone el mismo API con la misma libreria de
geometria que testbank usa para medir, asi que el fork y la tabla comparativa
puntuan con el MISMO IoU en vez de con dos implementaciones parecidas.

API reproducido, que es todo lo que usa la ruta OBB:

    VectorDouble(seq) -> la secuencia de 8 dobles [x1,y1,...,x4,y4]
    iou_poly(p, q)    -> IoU de los dos cuadrilateros
"""

from shapely.geometry import Polygon


def VectorDouble(values):  # noqa: N802  (el nombre lo fija el API original)
    """El original es un `std::vector<double>` envuelto. Una lista sirve."""
    return [float(v) for v in values]


def _polygon(flat):
    if len(flat) != 8:
        raise ValueError(f"se esperaban 8 coordenadas, llegaron {len(flat)}")
    polygon = Polygon([(flat[i], flat[i + 1]) for i in range(0, 8, 2)])
    # Un cuadrilatero con los lados cruzados es invalido para shapely y su area
    # sale mal. `buffer(0)` lo repara, que es lo mismo que hace testbank.
    return polygon if polygon.is_valid else polygon.buffer(0)


def iou_poly(p, q):
    # Descarte por envolvente ANTES de construir nada. Dos rectangulos no pueden
    # solaparse si sus envolventes alineadas no se tocan, y en un NMS la inmensa
    # mayoria de pares no se tocan. Medido: x15 con resultados identicos. Sin
    # esto, cada par pagaba dos Polygon() y una interseccion para devolver 0.0.
    px, py, qx, qy = p[0::2], p[1::2], q[0::2], q[1::2]
    if (
        max(px) < min(qx) or max(qx) < min(px)
        or max(py) < min(qy) or max(qy) < min(py)
    ):
        return 0.0
    a, b = _polygon(p), _polygon(q)
    intersection = a.intersection(b).area
    if intersection <= 0.0:
        return 0.0
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0
'''

#: Sustituto de `apex`, la libreria de precision mixta de NVIDIA.
APEX_STUB = '''"""Stub de `apex`. Lo instala tools/patch_yolox_obb_fork.py de testbank.

`apex` esta muerta desde que `torch.amp` existe y aun asi hay que compilarla
contra CUDA. El fork la importa SIN CONDICION en dos sitios
(`yolox/core/trainer.py:10` y `yolox/utils/ema.py:15`), asi que sin ella no
arranca ni el entrenamiento en CPU.

Pero no la USA salvo en dos casos que aqui no se dan:

    amp.*                     solo con `args.fp16`, que testbank deja en False
    parallel.Distributed...   solo con entrenamiento distribuido

Asi que esto da lo justo para que los imports pasen, y **falla ruidosamente** si
alguien activa de verdad la precision mixta: un stub silencioso que devolviera
cosas plausibles entrenaria en fp32 creyendo que va en fp16.

Vive dentro del clon y `sys.path` lo lleva al FINAL, asi que un `apex` de verdad
instalado en el entorno gana a este.
"""

_MESSAGE = (
    "apex no esta instalado: este es el stub de testbank. Solo hace falta de "
    "verdad para precision mixta (fp16) o entrenamiento distribuido, y ninguna "
    "de las dos esta soportada por el adaptador."
)


class _Unavailable:
    def __getattr__(self, name):
        raise RuntimeError(f"apex.{name}: {_MESSAGE}")


amp = _Unavailable()


class _NeverAnInstance:
    """Para el `isinstance` de `yolox/utils/ema.py:is_parallel`.

    Tiene que ser una CLASE de verdad -- isinstance no acepta otra cosa -- y no
    ser nunca el tipo de nada, que es justo lo correcto: sin apex no hay ningun
    modelo envuelto por apex.
    """


class _Distributed:
    DistributedDataParallel = _NeverAnInstance


class parallel:  # noqa: N801  (el nombre lo fija el API original)
    distributed = _Distributed
    DistributedDataParallel = _NeverAnInstance
'''


CLASSES = '''\
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Sustituido por tools/patch_yolox_obb_fork.py de testbank.
# El original trae las 15 clases de DOTA; el parser hace `class_to_ind[name]`,
# asi que cualquier nombre que no este aqui es un KeyError al cargar.

VOC_CLASSES = (
    "euro_banknote",
)
'''

#: Sustituciones QUIRURGICAS, no ficheros enteros.
#:
#: Los alias `np.int0` y `np.bool` desaparecieron en NumPy 2.0 y el fork es de
#: 2021. Reemplazar `yolox/utils/boxes.py` entero por una copia nuestra seria
#: absurdo -- son 250 lineas de las que cambian dos tokens -- y ademas congelaria
#: el resto de ese fichero contra cualquier actualizacion suya.
#:
#: `np.int0` es exactamente `np.intp`: era un alias, no un tipo propio.
#:
#: **Esto solo salta cuando hay cajas de verdad.** Con pesos sin entrenar y un
#: umbral normal no se produce ninguna deteccion, `postprocessobb_kld` sale por
#: el camino corto y la incompatibilidad pasa desapercibida. Se encontro bajando
#: el umbral a proposito para ejercitar el camino que si las produce.
SUBSTITUTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "yolox/utils/boxes.py": (("np.int0(", "np.intp("),),
    "yolox/evaluators/voc_eval.py": ((".astype(np.bool)", ".astype(bool)"),),
}


PATCHES: dict[str, str] = {
    "DOTA_devkit_YOLO/polyiou.py": POLYIOU_SHIM,
    "yolox/data/datasets/dota_classes.py": CLASSES,
    # Este no sustituye nada suyo: lo AÑADE. Por eso no hay `.orig`.
    "apex.py": APEX_STUB,
}

#: Los que se crean de cero, y por tanto no tienen original que respaldar.
CREATED = frozenset({"apex.py"})


def apply(root: Path, *, check_only: bool = False) -> int:
    if not (root / "yolox").is_dir():
        print(f"error: {root} no parece un clon de YOLOX-OBB", file=sys.stderr)
        return 2

    changed = 0
    for relative, content in PATCHES.items():
        target = root / relative
        if not target.exists() and relative not in CREATED:
            print(f"  AUSENTE  {relative}  <- el fork ha cambiado de forma")
            return 2
        if not target.exists():
            if check_only:
                print(f"  FALTA    {relative}")
                changed += 1
                continue
            target.write_text(content, encoding="utf-8")
            print(f"  creado   {relative}")
            changed += 1
            continue
        if target.read_text(encoding="utf-8") == content:
            print(f"  ya esta  {relative}")
            continue
        if check_only:
            print(f"  FALTA    {relative}")
            changed += 1
            continue
        backup = target.with_suffix(target.suffix + ".orig")
        if not backup.exists():
            backup.write_bytes(target.read_bytes())
        target.write_text(content, encoding="utf-8")
        print(f"  escrito  {relative}   (original en {backup.name})")
        changed += 1

    for relative, pairs in SUBSTITUTIONS.items():
        target = root / relative
        if not target.exists():
            print(f"  AUSENTE  {relative}  <- el fork ha cambiado de forma")
            return 2
        text = original = target.read_text(encoding="utf-8")
        for old_text, new_text in pairs:
            text = text.replace(old_text, new_text)
        if text == original:
            print(f"  ya esta  {relative}")
            continue
        if check_only:
            print(f"  FALTA    {relative}")
            changed += 1
            continue
        backup = target.with_suffix(target.suffix + ".orig")
        if not backup.exists():
            backup.write_bytes(target.read_bytes())
        target.write_text(text, encoding="utf-8")
        cuantas = sum(original.count(a) for a, _ in pairs)
        print(f"  {cuantas} cambio(s) en {relative}   (original en {backup.name})")
        changed += 1

    if check_only:
        return 1 if changed else 0
    print(f"\n{changed} fichero(s) parcheado(s). Comprueba con:")
    print(f"  cd {root} && python -c \"import yolox; print(yolox.__version__)\"")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="el clon del fork")
    parser.add_argument(
        "--check",
        action="store_true",
        help="no escribe nada; sale con 1 si falta algun parche",
    )
    args = parser.parse_args(argv)
    return apply(args.root, check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
