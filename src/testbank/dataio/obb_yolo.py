"""Lector del formato canonico: `class x1 y1 x2 y2 x3 y3 x4 y4` normalizado.

Es el formato del export de Roboflow y el pivote del resto de conversores.
Las anotaciones originales son de solo lectura: aqui no se escribe nunca.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from testbank.geometry.quad import (
    DEFAULT_ASPECT,
    CoordinateRangeWarning,
    Quad,
    QuadError,
    QuadShapeWarning,
    canonicalize,
)

TOKENS_PER_LINE = 9

# Un float64 no puede serializar mas de 17 cifras significativas. Un token con
# mas no viene de un exportador honesto: el fichero paso por un extractor que lo
# corrompio y no es una anotacion valida.
MAX_SIGNIFICANT_DIGITS = 17

_NUMBER = re.compile(
    r"^[+-]?(?:(?P<int>\d+)(?:\.(?P<frac>\d*))?|\.(?P<frac2>\d+))"
    r"(?:[eE](?P<exp>[+-]?\d+))?$"
)


class LabelFormatError(ValueError):
    """El fichero de etiquetas no cumple el formato. Siempre identifica linea.

    Sigue siendo fatal, pero acumula: un fichero con tres lineas malas las
    reporta las tres de una vez. Con 500 ficheros, fallar en la primera obliga a
    un ciclo completo de correccion y reexport por cada error, y no deja ver de
    entrada si el problema es un clic suelto o el export entero.
    """

    def __init__(self, message: str, problems: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        #: Cada problema por separado, ya formateado con su ubicacion.
        self.problems = problems or (message,)

    @classmethod
    def combine(cls, problems: list[str], header: str) -> LabelFormatError:
        """Un solo error con todos los problemas dentro.

        Con uno solo el mensaje es el de siempre, para no envolver de adorno el
        caso corriente ni romper a quien busque un texto concreto.
        """
        if len(problems) == 1:
            return cls(problems[0], (problems[0],))
        listed = "\n".join(f"  - {p}" for p in problems)
        return cls(f"{header} ({len(problems)}):\n{listed}", tuple(problems))


@dataclass(frozen=True, slots=True)
class Annotation:
    class_id: int
    quad: Quad


@dataclass(frozen=True, slots=True)
class LabelFile:
    path: Path
    annotations: tuple[Annotation, ...]
    #: Avisos no fatales, ya formateados con su ubicacion.
    warnings: tuple[str, ...] = ()


def significant_digits(token: str) -> int:
    """Cifras significativas de la mantisa.

    Se ignoran signo, punto, exponente, ceros a la izquierda y ceros a la
    derecha: ninguno de ellos aporta informacion que un float64 no pueda
    producir, y lo que se quiere detectar es precision fabricada.
    """
    match = _NUMBER.match(token)
    if match is None:
        raise ValueError(f"token no numerico: {token!r}")
    digits = (match.group("int") or "") + (
        match.group("frac") or match.group("frac2") or ""
    )
    return len(digits.strip("0"))


def _parse_line(
    line: str, path: Path, lineno: int, aspect: float = DEFAULT_ASPECT
) -> Annotation:
    where = f"{path}:{lineno}"
    tokens = line.split()
    if len(tokens) != TOKENS_PER_LINE:
        raise LabelFormatError(
            f"{where}: una linea obb_yolo tiene {TOKENS_PER_LINE} tokens "
            f"(class + 4 vertices), se encontraron {len(tokens)}: {line.strip()!r}"
        )

    for index, token in enumerate(tokens):
        try:
            count = significant_digits(token)
        except ValueError as exc:
            raise LabelFormatError(f"{where}: token {index}: {exc}") from exc
        if count > MAX_SIGNIFICANT_DIGITS:
            raise LabelFormatError(
                f"{where}: token {index} tiene {count} cifras significativas "
                f"({token!r}); un float64 no puede serializar mas de "
                f"{MAX_SIGNIFICANT_DIGITS}, asi que el fichero paso por un "
                f"extractor que lo corrompio y no es una anotacion valida"
            )

    class_token = tokens[0]
    class_value = float(class_token)
    if class_value != int(class_value) or class_value < 0:
        raise LabelFormatError(
            f"{where}: el id de clase debe ser un entero no negativo, "
            f"se encontro {class_token!r}"
        )

    try:
        quad = Quad.from_xy([float(t) for t in tokens[1:]])
    except QuadError as exc:
        raise LabelFormatError(f"{where}: {exc}") from exc

    return Annotation(
        class_id=int(class_value), quad=canonicalize(quad, aspect=aspect)
    )


def read_label_file(
    path: str | Path, *, aspect: float = DEFAULT_ASPECT
) -> LabelFile:
    """Lee un .txt obb_yolo. Un fichero vacio es una imagen sin billetes, valido.

    `aspect` es ancho/alto en pixeles de la imagen correspondiente. Sin el, el
    orden canonico se calcula en el espacio normalizado, donde el lado mas largo
    puede no ser el lado mas largo real. Pasalo siempre que lo tengas.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise LabelFormatError(f"{path}: no existe el fichero de etiquetas") from exc

    annotations: list[Annotation] = []
    messages: list[str] = []
    problems: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                parsed = _parse_line(line, path, lineno, aspect)
            except LabelFormatError as exc:
                # Se sigue leyendo para reportar TODAS las lineas malas del
                # fichero de una vez; el error se lanza al terminarlo.
                problems.append(str(exc))
                continue
            annotations.append(parsed)
        for entry in caught:
            if issubclass(entry.category, (CoordinateRangeWarning, QuadShapeWarning)):
                messages.append(f"{path}:{lineno}: {entry.message}")

    if problems:
        raise LabelFormatError.combine(problems, f"{path}: lineas invalidas")

    return LabelFile(
        path=path, annotations=tuple(annotations), warnings=tuple(messages)
    )
