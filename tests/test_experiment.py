"""Motor de experimentos: procedencia, directorio de ejecucion y comparativa."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime

import pytest

from testbank.config import Config
from testbank.experiment import compare as compare_mod
from testbank.experiment.provenance import (
    Provenance,
    collect_environment,
    collect_git,
    seed_everything,
)
from testbank.experiment.run import (
    ComponentInfo,
    ExperimentRun,
    discover_runs,
    load_run,
)

MOMENT = datetime(2026, 9, 10, 18, 30, 0, tzinfo=UTC)

APACHE_DETECTOR = ComponentInfo("rtmdet-r-tiny", "Apache-2.0", True)
AGPL_DETECTOR = ComponentInfo("ultralytics-yolo26n", "AGPL-3.0", False)
CLEAN_DATASET = ComponentInfo("banknotes-v2", "CC-BY-4.0", True, sources=("CC-BY-4.0",))
DIRTY_DATASET = ComponentInfo(
    "agregado", "CC-BY-4.0", True, sources=("CC-BY-4.0", "AGPL-3.0")
)


def make_run(tmp_path, **kwargs) -> ExperimentRun:
    config = Config()
    return ExperimentRun.create(
        kwargs.pop("name", "prueba"),
        config,
        runs_dir=tmp_path,
        moment=kwargs.pop("moment", MOMENT),
        **kwargs,
    )


# --- procedencia ----------------------------------------------------------


def test_sin_repositorio_git_se_registra_el_motivo(tmp_path):
    """La ausencia se registra como ausencia, no como un valor por defecto."""
    info = collect_git(tmp_path)
    assert info.available is False
    assert info.commit is None
    assert info.reason and "git init" in info.reason
    assert "sin git" in info.describe()


def test_sin_git_la_ejecucion_no_es_reproducible(tmp_path):
    prov = Provenance(
        git=collect_git(tmp_path),
        seeds=seed_everything(1),
        environment=collect_environment(),
    )
    assert prov.reproducible is False
    assert len(prov.blockers()) == 1
    assert "no se puede reproducir" in prov.blockers()[0]


def test_la_misma_semilla_da_la_misma_secuencia():
    seed_everything(123)
    first = [random.random() for _ in range(5)]
    seed_everything(123)
    assert [random.random() for _ in range(5)] == first


def test_se_registra_lo_que_de_verdad_quedo_fijado():
    """torch es opcional: si no esta, su semilla es None, no un numero falso."""
    seeds = seed_everything(7)
    assert seeds.base == 7 and seeds.python == 7
    try:
        import torch  # noqa: F401
    except ImportError:
        assert seeds.torch is None
    else:
        assert seeds.torch == 7


def test_el_entorno_solo_lista_lo_instalado():
    env = collect_environment()
    assert env.python and env.platform
    assert "numpy" in env.packages and "shapely" in env.packages


# --- el directorio de ejecucion -------------------------------------------


def test_se_crea_la_estructura_completa(tmp_path):
    run = make_run(tmp_path)
    assert run.directory.name == "20260910T183000Z_prueba"
    assert (run.directory / "config.yaml").exists()
    assert (run.directory / "run.json").exists()
    assert run.weights_dir.is_dir() and run.viz_dir.is_dir()


def test_la_config_se_congela_antes_de_ejecutar(tmp_path):
    """Si la ejecucion peta a la mitad, tiene que quedar con que se lanzo."""
    run = make_run(tmp_path)
    text = (run.directory / "config.yaml").read_text(encoding="utf-8")
    assert "min_relative_area" in text
    assert "visibility_threshold" in text


def test_no_se_pisa_una_ejecucion_existente(tmp_path):
    make_run(tmp_path)
    with pytest.raises(FileExistsError):
        make_run(tmp_path)


def test_las_metricas_actualizan_los_dos_ficheros(tmp_path):
    run = make_run(tmp_path, detector=APACHE_DETECTOR)
    run.write_metrics({"map50": 0.81, "n_images": 101})
    assert json.loads((run.directory / "metrics.json").read_text())["map50"] == 0.81
    assert load_run(run.directory)["metrics"]["n_images"] == 101


# --- la cadena de licencias -----------------------------------------------


def test_detector_agpl_no_es_apto(tmp_path):
    run = make_run(tmp_path, detector=AGPL_DETECTOR, datasets=(CLEAN_DATASET,))
    assert run.record.production_ready is False
    assert any("AGPL-3.0" in b for b in run.record.blockers())


def test_una_fuente_contaminada_arrastra_al_agregado(tmp_path):
    """La licencia de un agregado no anula la de sus fuentes."""
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(DIRTY_DATASET,))
    assert run.record.production_ready is False
    assert any("no anula" in b for b in run.record.blockers())


def test_cadena_limpia_es_apta(tmp_path):
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,))
    assert run.record.production_ready is True


def test_sin_detector_declarado_no_es_apto(tmp_path):
    assert make_run(tmp_path).record.production_ready is False


# --- la tabla comparativa -------------------------------------------------


def test_compare_ignora_los_directorios_auxiliares(tmp_path):
    """`runs/_inspection` no es una ejecucion."""
    make_run(tmp_path, detector=APACHE_DETECTOR)
    (tmp_path / "_inspection").mkdir()
    (tmp_path / "_inspection" / "algo.png").write_bytes(b"")
    assert len(discover_runs(tmp_path)) == 1


def test_toda_fila_lleva_n_e_intervalo(tmp_path):
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,))
    run.write_metrics(
        {
            "n_images": 101,
            "map50": {"value": 0.812, "ci_low": 0.74, "ci_high": 0.87},
        }
    )
    rows, table = compare_mod.compare(tmp_path)
    assert rows[0]["n"] == "101"
    assert "0.812 [0.740,0.870]" in table


def test_una_metrica_sin_intervalo_se_dice(tmp_path):
    """Un numero pelado invita a leer como mejora lo que es dispersion."""
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,))
    run.write_metrics({"n_images": 101, "map50": {"value": 0.81}})
    _, table = compare_mod.compare(tmp_path)
    assert "sin IC" in table


def test_la_tabla_marca_lo_no_apto_y_dice_por_que(tmp_path):
    run = make_run(tmp_path, name="base", detector=AGPL_DETECTOR)
    run.write_metrics({"n_images": 101, "map50": {"value": 0.9}})
    _, table = compare_mod.compare(tmp_path)
    assert compare_mod.NOT_READY in table
    assert "AGPL-3.0" in table
    assert "referencia de rendimiento, no como candidato" in table


def test_la_tabla_separa_no_apto_de_no_reproducible(tmp_path):
    """Son dos problemas distintos con arreglos distintos.

    Esta ejecucion es APTA (detector y dataset Apache/CC-BY) pero NO
    reproducible (el proyecto no esta en git). Meterlas en la misma lista
    dejaria al lector sin saber cual de las dos cosas esta leyendo.
    """
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,))
    run.write_metrics({"n_images": 101, "map50": {"value": 0.8}})
    rows, table = compare_mod.compare(tmp_path)
    assert rows[0]["apto"] == "si"
    assert rows[0]["reproducible"] == "NO"
    assert compare_mod.NOT_READY not in table
    assert "NO REPRODUCIBLES" in table
    # El motivo concreto depende del estado del repositorio donde se ejecuten
    # los tests (sin repo, sin commits, o arbol sucio), asi que se comprueba que
    # hay motivo y que va en la lista de procedencia, no su texto exacto.
    assert rows[0]["_provenance_blockers"]


def test_el_csv_lleva_las_mismas_columnas(tmp_path):
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,))
    run.write_metrics({"n_images": 101, "map50": {"value": 0.8, "ci_low": 0.7, "ci_high": 0.9}})
    rows, _ = compare_mod.compare(tmp_path)
    path = compare_mod.write_csv(rows, tmp_path / "out" / "compare.csv")
    text = path.read_text(encoding="utf-8")
    assert "mAP50" in text and "apto" in text and "101" in text


def test_sin_ejecuciones_lo_dice(tmp_path):
    _, table = compare_mod.compare(tmp_path)
    assert "No hay ejecuciones" in table
