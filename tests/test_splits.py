from __future__ import annotations

import json

import pytest

from testbank.data.discover import LayoutError, LayoutMode, detect_layout
from testbank.data.splits import (
    GroupConfig,
    GroupStrategy,
    SealedTestSetError,
    SplitError,
    SplitLoader,
    materialize_splits,
    record_test_access,
)

LINE = "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"


def make_sample(directory, stem):
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "labels").mkdir(parents=True, exist_ok=True)
    # JPEG minimo valido no hace falta: discover solo mira la extension.
    (directory / "images" / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (directory / "labels" / f"{stem}.txt").write_text(LINE, encoding="utf-8")


def roboflow_root(tmp_path, counts=(6, 3, 2)):
    root = tmp_path / "data"
    index = 0
    for split, count in zip(("train", "valid", "test"), counts):
        for _ in range(count):
            make_sample(root / split, f"img_{index:03d}")
            index += 1
    return root


def flat_root(tmp_path, count=10):
    root = tmp_path / "flat"
    for i in range(count):
        make_sample(root, f"img_{i:03d}")
    return root


# -- deteccion --------------------------------------------------------------


def test_detecta_modo_adoptar(tmp_path):
    layout = detect_layout(roboflow_root(tmp_path))
    assert layout.mode is LayoutMode.ADOPT
    assert len(layout.groups["train"]) == 6


def test_detecta_modo_crear(tmp_path):
    layout = detect_layout(flat_root(tmp_path))
    assert layout.mode is LayoutMode.CREATE
    assert len(layout.all_samples) == 10


def test_acepta_val_como_alias_pero_deja_constancia(tmp_path):
    """Roboflow usa 'valid'. 'val' aparece en exports retocados a mano."""
    root = roboflow_root(tmp_path)
    (root / "valid").rename(root / "val")
    layout = detect_layout(root)
    assert layout.mode is LayoutMode.ADOPT
    assert any("val" in note for note in layout.notes)


def test_valid_y_val_a_la_vez_es_ambiguo(tmp_path):
    root = roboflow_root(tmp_path)
    make_sample(root / "val", "otro_000")
    with pytest.raises(LayoutError, match="ambiguo"):
        detect_layout(root)


def test_imagen_sin_etiqueta_es_fatal(tmp_path):
    """Tratarla como 'imagen sin billetes' envenena el entrenamiento en silencio."""
    root = roboflow_root(tmp_path)
    (root / "train" / "images" / "huerfana.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    with pytest.raises(LayoutError, match="sin fichero de etiquetas"):
        detect_layout(root)


def test_etiqueta_sin_imagen_es_fatal(tmp_path):
    root = roboflow_root(tmp_path)
    (root / "train" / "labels" / "huerfana.txt").write_text(LINE, encoding="utf-8")
    with pytest.raises(LayoutError, match="sin imagen"):
        detect_layout(root)


def test_estructura_a_medias_es_fatal(tmp_path):
    root = tmp_path / "data"
    make_sample(root / "train", "img_000")
    with pytest.raises(LayoutError, match="a medias"):
        detect_layout(root)


# -- modo adoptar -----------------------------------------------------------


def test_adoptar_congela_la_particion_del_export(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    manifest = materialize_splits(root, splits)
    assert manifest["mode"] == "adopt"
    assert manifest["counts"] == {"train": 6, "valid": 3, "test": 2}
    assert (splits / "valid.txt").is_file()


def test_adoptar_no_exige_confirmar_independencia(tmp_path):
    """En modo adoptar no elegimos el reparto, solo lo congelamos."""
    materialize_splits(roboflow_root(tmp_path), tmp_path / "splits")


def test_se_niega_a_sobrescribir_una_particion_existente(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    materialize_splits(root, splits)
    with pytest.raises(SplitError, match="Se niega a"):
        materialize_splits(root, splits)
    materialize_splits(root, splits, overwrite=True)


# -- modo crear -------------------------------------------------------------


def test_crear_exige_confirmar_independencia(tmp_path):
    with pytest.raises(SplitError, match="i-confirm-independence"):
        materialize_splits(flat_root(tmp_path), tmp_path / "splits")


def test_crear_con_confirmacion(tmp_path):
    manifest = materialize_splits(
        flat_root(tmp_path),
        tmp_path / "splits",
        groups=GroupConfig(independence_confirmed=True),
        ratios=(0.6, 0.2, 0.2),
        seed=7,
    )
    assert manifest["mode"] == "create"
    assert sum(manifest["counts"].values()) == 10


def test_crear_es_determinista_con_la_misma_semilla(tmp_path):
    root = flat_root(tmp_path)
    groups = GroupConfig(independence_confirmed=True)
    a = materialize_splits(root, tmp_path / "a", groups=groups, seed=99)
    b = materialize_splits(root, tmp_path / "b", groups=groups, seed=99)
    c = materialize_splits(root, tmp_path / "c", groups=groups, seed=100)
    assert a["digest"] == b["digest"]
    assert a["digest"] != c["digest"]


def test_filename_prefix_exige_regex(tmp_path):
    with pytest.raises(SplitError, match="group-regex"):
        materialize_splits(
            flat_root(tmp_path),
            tmp_path / "splits",
            groups=GroupConfig(strategy=GroupStrategy.FILENAME_PREFIX),
        )


def test_un_grupo_no_se_reparte_entre_particiones(tmp_path):
    """Varias tomas del mismo billete fisico caen juntas."""
    root = tmp_path / "flat"
    for physical in range(4):
        for take in range(3):
            make_sample(root, f"bill{physical:02d}_take{take}")
    manifest = materialize_splits(
        root,
        tmp_path / "splits",
        groups=GroupConfig(
            strategy=GroupStrategy.FILENAME_PREFIX, regex=r"^(bill\d+)_"
        ),
        ratios=(0.5, 0.25, 0.25),
        seed=3,
    )
    assignment = {
        name: (tmp_path / "splits" / f"{name}.txt").read_text(encoding="utf-8")
        for name in ("train", "valid", "test")
    }
    for physical in range(4):
        homes = [
            name
            for name, text in assignment.items()
            if f"bill{physical:02d}_" in text
        ]
        assert len(homes) == 1, f"el grupo bill{physical:02d} se repartio: {homes}"
    assert manifest["group_strategy"] == "filename-prefix"


# -- integridad y sellado ---------------------------------------------------


def test_una_muestra_en_dos_particiones_es_fatal(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    materialize_splits(root, splits)
    with (splits / "valid.txt").open("a", encoding="utf-8") as handle:
        handle.write("img_000\n")  # ya esta en train
    with pytest.raises(SplitError, match="img_000"):
        SplitLoader(splits, root)


def test_muestra_de_la_particion_que_falta_en_disco_es_fatal(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    materialize_splits(root, splits)
    (root / "train" / "images" / "img_000.jpg").unlink()
    (root / "train" / "labels" / "img_000.txt").unlink()
    with pytest.raises(SplitError, match="no estan en"):
        SplitLoader(splits, root)


def test_test_esta_sellado(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    materialize_splits(root, splits)
    loader = SplitLoader(splits, root)
    with pytest.raises(SealedTestSetError, match="sellado"):
        loader.load("test")
    assert len(loader.load("test", allow_test=True)) == 2


def test_cada_acceso_a_test_queda_registrado(tmp_path):
    log = tmp_path / "test_evaluations.jsonl"
    record_test_access("evaluacion final", "run_a", log_path=log)
    record_test_access("segunda mirada", "run_b", log_path=log)
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [e["run"] for e in entries] == ["run_a", "run_b"]
