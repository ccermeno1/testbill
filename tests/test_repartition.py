"""`--repartition` y `--stratify-regex`: rehacer la particion sin fugas y con
todos los tipos de billete en cada lado.

Existe porque la particion de Roboflow venia con casi-duplicados cruzando
particiones. Lo que se comprueba aqui es lo que se pidio, en este orden:

1. Que ningun grupo (misma toma) quede repartido entre particiones.
2. Que cada particion tenga representantes de cada tipo.
3. Que sea opt-in, quede escrito en el manifiesto, y no se confunda con la
   particion del export.
"""

from __future__ import annotations

import json
import re

import pytest
from test_splits import make_sample

from testbank.data.splits import (
    GroupConfig,
    GroupStrategy,
    SplitError,
    _read_split_file,
    materialize_splits,
)

TIPOS = ("005", "010", "020", "050", "100", "200", "500", "Multiple")


def _tipo(sid):
    return re.match(r"^(\d+|Multiple)_", sid).group(1)


def roboflow_con_tipos(tmp_path, *, por_tipo=8, tomas=2):
    """Un export ya partido, con `por_tipo` billetes de cada tipo y `tomas`
    fotos casi iguales de cada uno. Roboflow los reparte a lo tonto: las tomas
    del mismo billete caen en particiones distintas a proposito."""
    root = tmp_path / "data"
    manifest = {}
    splits = ("train", "valid", "test")
    n = 0
    for tipo in TIPOS:
        for billete in range(por_tipo):
            for toma in range(tomas):
                stem = f"{tipo}_Euro_{billete:03d}_t{toma}"
                make_sample(root / splits[n % 3], stem)
                manifest[stem] = f"{tipo}_Euro_{billete:03d}"
                n += 1
    path = tmp_path / "dups.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, path


def _leer(splits_dir):
    # Por el lector oficial: los ficheros llevan una cabecera de comentario.
    return {
        name: _read_split_file(splits_dir / f"{name}.txt")
        for name in ("train", "valid", "test")
    }


# --- 1. sin fugas ----------------------------------------------------------


def test_adoptar_con_un_manifiesto_que_demuestra_fugas_se_niega(tmp_path):
    """En modo adoptar manda el export -- pero si el manifiesto demuestra que
    el export reparte un grupo, no se congela la fuga en silencio: se para y el
    mensaje dice cual es la salida."""
    root, dups = roboflow_con_tipos(tmp_path)
    with pytest.raises(SplitError, match="--repartition"):
        materialize_splits(
            root,
            tmp_path / "s",
            groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
        )
    assert not (tmp_path / "s" / "train.txt").exists(), "no ha escrito a medias"


def test_con_repartition_ningun_grupo_cruza(tmp_path):
    root, dups = roboflow_con_tipos(tmp_path)
    manifest = materialize_splits(
        root,
        tmp_path / "s",
        groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
        repartition=True,
    )
    assert manifest["mode"] == "repartition"
    assert manifest["export_mode"] == "adopt"
    particiones = _leer(tmp_path / "s")
    grupos = json.loads(dups.read_text(encoding="utf-8"))
    hogar_de_grupo: dict[str, set[str]] = {}
    for name, ids in particiones.items():
        for sid in ids:
            hogar_de_grupo.setdefault(grupos[sid], set()).add(name)
    cruzan = {g: h for g, h in hogar_de_grupo.items() if len(h) > 1}
    assert cruzan == {}, f"grupos repartidos entre particiones: {cruzan}"
    assert sum(len(ids) for ids in particiones.values()) == len(grupos)


# --- 2. representativos de cada tipo ---------------------------------------


def test_estratificado_pone_cada_tipo_en_cada_particion(tmp_path):
    root, dups = roboflow_con_tipos(tmp_path, por_tipo=10)
    manifest = materialize_splits(
        root,
        tmp_path / "s",
        groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
        repartition=True,
        stratify_regex=r"^(\d+|Multiple)_",
        ratios=(0.6, 0.2, 0.2),
    )
    tipo = _tipo
    for name, ids in _leer(tmp_path / "s").items():
        presentes = {tipo(sid) for sid in ids}
        assert presentes == set(TIPOS), f"{name} no tiene todos los tipos: {presentes}"
    # Y el manifiesto lo cuenta, por estrato y particion.
    assert set(manifest["strata"]) == set(TIPOS)
    for counts in manifest["strata"].values():
        assert all(counts[n] > 0 for n in ("train", "valid", "test"))
    assert manifest["stratify_regex"] == r"^(\d+|Multiple)_"


def test_sin_estratificar_un_tipo_puede_quedarse_fuera(tmp_path):
    """El motivo de estratificar, demostrado: con pocos de cada tipo y reparto
    ciego, alguna particion se queda sin alguno. Con estratos, nunca.

    Cuatro grupos por tipo y no dos: con dos, `round(2 * 0.25) == 0` y ninguna
    estrategia puede poner uno en cada una de tres particiones. No es un fallo
    del reparto, es aritmetica -- y el manifiesto lo deja ver en su tabla por
    estrato, que para eso esta.
    """
    root, dups = roboflow_con_tipos(tmp_path, por_tipo=4)
    tipo = _tipo

    faltas_ciego = 0
    for seed in range(20):
        materialize_splits(
            root, tmp_path / f"ciego{seed}",
            groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
            repartition=True, ratios=(0.5, 0.25, 0.25), seed=seed,
        )
        for ids in _leer(tmp_path / f"ciego{seed}").values():
            faltas_ciego += len(set(TIPOS) - {tipo(s) for s in ids})
    assert faltas_ciego > 0, "el caso tiene que ser lo bastante escaso para fallar"

    for seed in range(20):
        materialize_splits(
            root, tmp_path / f"estr{seed}",
            groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
            repartition=True, ratios=(0.5, 0.25, 0.25), seed=seed,
            stratify_regex=r"^(\d+|Multiple)_",
        )
        for ids in _leer(tmp_path / f"estr{seed}").values():
            assert set(TIPOS) - {tipo(s) for s in ids} == set()


# --- 3. explicito, escrito, y con sus guardas ------------------------------


def test_estratificar_sin_repartition_es_error(tmp_path):
    root, _ = roboflow_con_tipos(tmp_path)
    with pytest.raises(SplitError, match="--repartition"):
        materialize_splits(root, tmp_path / "s", stratify_regex=r"^(\d+)_")


def test_la_regex_necesita_un_grupo_de_captura(tmp_path):
    root, _ = roboflow_con_tipos(tmp_path)
    with pytest.raises(SplitError, match="un grupo de captura"):
        materialize_splits(
            root, tmp_path / "s", repartition=True, stratify_regex=r"^\d+_",
            groups=GroupConfig(strategy=GroupStrategy.NONE, independence_confirmed=True),
        )


def test_una_muestra_sin_estrato_es_error_y_dice_cual(tmp_path):
    root, _ = roboflow_con_tipos(tmp_path)
    make_sample(root / "train", "sin_tipo_001")
    with pytest.raises(SplitError, match="sin_tipo_001"):
        materialize_splits(
            root, tmp_path / "s", repartition=True, stratify_regex=r"^(\d+|Multiple)_",
            groups=GroupConfig(strategy=GroupStrategy.NONE, independence_confirmed=True),
        )


def test_un_grupo_que_cruza_estratos_avisa_pero_no_bloquea(tmp_path):
    """Dos fotos 'iguales' de billetes distintos: se reparten juntas y se avisa.

    Agrupar de mas es barato; bloquear la particion entera por un par, no.
    """
    root, dups = roboflow_con_tipos(tmp_path)
    grupos = json.loads(dups.read_text(encoding="utf-8"))
    grupos["500_Euro_000_t0"] = grupos["005_Euro_000_t0"]  # un 500 pegado a un 5
    dups.write_text(json.dumps(grupos), encoding="utf-8")
    manifest = materialize_splits(
        root, tmp_path / "s",
        groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
        repartition=True, stratify_regex=r"^(\d+|Multiple)_",
    )
    avisos = [n for n in manifest["notes"] if "cruza estratos" in n]
    assert len(avisos) == 1 and "005" in avisos[0] and "500" in avisos[0]
    particiones = _leer(tmp_path / "s")
    hogares = {n for n, ids in particiones.items() if "500_Euro_000_t0" in ids}
    hogares |= {n for n, ids in particiones.items() if "005_Euro_000_t0" in ids}
    assert len(hogares) == 1, "siguen juntas: es un grupo"


def test_repartition_es_determinista(tmp_path):
    root, dups = roboflow_con_tipos(tmp_path)
    kw = {
        "groups": GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
        "repartition": True,
        "stratify_regex": r"^(\d+|Multiple)_",
        "seed": 7,
    }
    a = materialize_splits(root, tmp_path / "a", **kw)
    b = materialize_splits(root, tmp_path / "b", **kw)
    assert a["digest"] == b["digest"]
    assert _leer(tmp_path / "a") == _leer(tmp_path / "b")
