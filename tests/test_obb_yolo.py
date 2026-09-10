from __future__ import annotations

import pytest

from testbank.dataio.obb_yolo import (
    LabelFormatError,
    read_label_file,
    significant_digits,
)

VALID = "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"


def write(tmp_path, text, name="img_0001.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# -- cifras significativas --------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("0", 0),
        ("0.5", 1),
        ("0.000123", 3),
        ("123.456", 6),
        ("-0.10", 1),
        ("1e-3", 1),
        ("0.12345678901234567", 17),
        ("0.123456789012345678", 18),
    ],
)
def test_cuenta_de_cifras_significativas(token, expected):
    assert significant_digits(token) == expected


# -- lectura ----------------------------------------------------------------


def test_lee_una_anotacion_valida(tmp_path):
    label = read_label_file(write(tmp_path, VALID))
    assert len(label.annotations) == 1
    assert label.annotations[0].class_id == 0
    assert label.annotations[0].quad.is_clockwise()


def test_fichero_vacio_es_imagen_sin_billetes(tmp_path):
    assert read_label_file(write(tmp_path, "")).annotations == ()


def test_ignora_lineas_en_blanco(tmp_path):
    assert len(read_label_file(write(tmp_path, VALID + "\n\n" + VALID)).annotations) == 2


def test_la_salida_ya_viene_canonica(tmp_path):
    """El lector canonicaliza: nadie aguas abajo tiene que acordarse."""
    shuffled = "0 0.5 0.3 0.1 0.3 0.1 0.1 0.5 0.1\n"
    a = read_label_file(write(tmp_path, VALID, "a.txt")).annotations[0]
    b = read_label_file(write(tmp_path, shuffled, "b.txt")).annotations[0]
    assert a.quad.points == b.quad.points


# -- errores explicitos -----------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1\n",
        "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3 0.9\n",
        "0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n",
    ],
)
def test_numero_de_tokens_incorrecto_es_error_explicito(tmp_path, line):
    path = write(tmp_path, line)
    with pytest.raises(LabelFormatError, match="9 tokens") as exc:
        read_label_file(path)
    assert f"{path}:1" in str(exc.value)


def test_precision_fabricada_es_error(tmp_path):
    """18 cifras: el fichero paso por un extractor que lo corrompio."""
    line = "0 0.100000000000000001 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"
    with pytest.raises(LabelFormatError, match="cifras significativas") as exc:
        read_label_file(write(tmp_path, line))
    assert "token 1" in str(exc.value)


def test_diecisiete_cifras_se_acepta(tmp_path):
    line = "0 0.10000000000000001 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"
    assert len(read_label_file(write(tmp_path, line)).annotations) == 1


def test_token_no_numerico_es_error(tmp_path):
    line = "0 nan 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"
    with pytest.raises(LabelFormatError, match="no numerico"):
        read_label_file(write(tmp_path, line))


def test_clase_no_entera_es_error(tmp_path):
    line = "0.5 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"
    with pytest.raises(LabelFormatError, match="entero no negativo"):
        read_label_file(write(tmp_path, line))


def test_coordenada_fuera_de_rango_es_error(tmp_path):
    line = "0 0.1 0.1 1.9 0.1 1.9 0.3 0.1 0.3\n"
    with pytest.raises(LabelFormatError, match="fuera del rango"):
        read_label_file(write(tmp_path, line))


def test_el_error_identifica_la_linea(tmp_path):
    path = write(tmp_path, VALID + VALID + "0 1 2 3\n")
    with pytest.raises(LabelFormatError) as exc:
        read_label_file(path)
    assert f"{path}:3" in str(exc.value)


# -- avisos no fatales ------------------------------------------------------


def test_vertice_fuera_de_0_1_es_aviso_no_error(tmp_path):
    """Billete que cruza el borde de la imagen."""
    line = "0 -0.2 0.1 0.5 0.1 0.5 0.3 -0.2 0.3\n"
    label = read_label_file(write(tmp_path, line))
    assert len(label.annotations) == 1
    assert any("fuera de [0,1]" in w for w in label.warnings)


# -- los errores se acumulan, no se para en el primero ----------------------


def test_un_fichero_reporta_todas_sus_lineas_malas(tmp_path):
    """Tres lineas malas se ven de una vez, no una por ejecucion."""
    text = "0 1 2 3\n" + VALID + "0 nan 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n" + "0.5 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"
    path = write(tmp_path, text)
    with pytest.raises(LabelFormatError) as exc:
        read_label_file(path)
    assert len(exc.value.problems) == 3
    assert f"{path}:1" in str(exc.value)
    assert f"{path}:3" in str(exc.value)
    assert f"{path}:4" in str(exc.value)
    # La linea 2 es valida y no aparece.
    assert f"{path}:2" not in str(exc.value)


def test_con_una_sola_linea_mala_el_mensaje_no_cambia(tmp_path):
    """No envolver el caso corriente: sigue siendo el error de siempre."""
    path = write(tmp_path, "0 1 2 3\n")
    with pytest.raises(LabelFormatError) as exc:
        read_label_file(path)
    assert exc.value.problems == (str(exc.value),)
    assert "lineas invalidas" not in str(exc.value)


def test_los_errores_se_acumulan_entre_ficheros(tmp_path):
    """Con 500 etiquetas, un error por ejecucion vuelve la limpieza un bucle."""
    from testbank.data.discover import Sample
    from testbank.dataio.loader import load_samples

    samples = []
    for name, text in (("a", "0 1 2 3\n"), ("b", VALID), ("c", "0 nan 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n")):
        label = write(tmp_path, text, f"{name}.txt")
        samples.append(
            Sample(sample_id=name, image_path=tmp_path / f"{name}.jpg", label_path=label)
        )

    with pytest.raises(LabelFormatError) as exc:
        load_samples(samples)
    assert len(exc.value.problems) == 2
    assert "a.txt" in str(exc.value) and "c.txt" in str(exc.value)
    assert "b.txt" not in str(exc.value)
