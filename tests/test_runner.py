"""El recorrido completo: particion -> entrenamiento -> metricas -> disco."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import rotated_rect_points
from PIL import Image

from testbank.config import Config, OutOfBoundsPolicy
from testbank.data import datasets as datasets_mod
from testbank.data.datasets import DatasetError, DatasetInfo
from testbank.data.splits import materialize_splits
from testbank.dataio.formats import get as get_format
from testbank.detectors import base as detector_base
from testbank.detectors.base import BaseDetector, TrainResult, register
from testbank.experiment.runner import run_candidate
from testbank.geometry.quad import Quad, canonicalize
from testbank.metrics.core import Prediction

FAKE_DETECTOR = "_fake_de_prueba"
FAKE_DATASET = "_dataset_de_prueba"


def _quad(cx, cy, half_long=0.12, ratio=2.0, theta=0.0) -> Quad:
    return canonicalize(
        Quad.from_xy(rotated_rect_points(cx, cy, half_long, ratio, theta))
    )


QUADS = (_quad(0.3, 0.5), _quad(0.7, 0.5))


def _build_dataset(root: Path, counts: dict[str, int]) -> None:
    """Arbol Roboflow minimo: train/valid/test con images/ y labels/."""
    writer = get_format("obb_yolo")
    payload = "\n".join(f"0 {writer.from_quad(q).payload}" for q in QUADS) + "\n"
    for split, count in counts.items():
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            sample_id = f"{split}_{i:03d}"
            Image.new("RGB", (160, 120), (40, 40, 40)).save(images / f"{sample_id}.jpg")
            (labels / f"{sample_id}.txt").write_text(payload, encoding="utf-8")


class _FakeDetector(BaseDetector):
    """Entrena de mentira y predice la verdad. Ejercita el recorrido, no el modelo."""

    name = FAKE_DETECTOR
    license = "Apache-2.0"
    production_ready = True
    notes = ("detector de prueba",)

    #: Particiones que `train` recibio. Es como se comprueba que test no se toca.
    seen_splits: tuple[str, ...] = ()

    def train(self, samples_by_split, config, *, output_dir: Path) -> TrainResult:
        type(self).seen_splits = tuple(sorted(samples_by_split))
        output_dir.mkdir(parents=True, exist_ok=True)
        weights = output_dir / "best.pt"
        weights.write_bytes(b"pesos de mentira")
        return TrainResult(weights=weights, epochs=1, notes=("entrenamiento falso",))

    def predict(self, samples, *, weights, config):
        return {s.sample_id: [Prediction(q, 0.9) for q in QUADS] for s in samples}


@pytest.fixture()
def registered():
    register(_FakeDetector)
    info = DatasetInfo(
        name=FAKE_DATASET,
        root=Path("."),
        license="CC-BY-4.0",
        production_ready=True,
        sources=("CC-BY-4.0",),
        origin="https://ejemplo/proyecto",
        version="v9",
        exported="2026-09-10",
    )
    datasets_mod.register(info)
    try:
        yield detector_base.REGISTRY[FAKE_DETECTOR]
    finally:
        del detector_base.REGISTRY[FAKE_DETECTOR]
        del datasets_mod.REGISTRY[FAKE_DATASET]


@pytest.fixture()
def prepared(tmp_path):
    root = tmp_path / "data"
    _build_dataset(root, {"train": 6, "valid": 4, "test": 3})
    splits_dir = tmp_path / "splits"
    materialize_splits(root, splits_dir)
    base = Config()
    config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={"derived_dir": tmp_path / "derived", "splits_dir": splits_dir}
            ),
            "metrics": base.metrics.model_copy(update={"bootstrap_samples": 30}),
            "viz": base.viz.model_copy(update={"sample_count": 2}),
            "runs_dir": tmp_path / "runs",
        }
    )
    return root, splits_dir, config


def test_la_ejecucion_deja_todo_en_disco(registered, prepared):
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )

    directory = outcome.run.directory
    assert (directory / "config.yaml").exists()
    assert (directory / "run.json").exists()
    assert (directory / "metrics.json").exists()
    assert outcome.weights.exists()
    assert outcome.weights.parent == outcome.run.weights_dir
    assert list(outcome.run.viz_dir.glob("*.png"))


def test_test_nunca_se_carga(registered, prepared):
    """El sello solo lo rompe `evaluate-test`, que lleva su propio registro."""
    root, splits_dir, config = prepared
    run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    assert _FakeDetector.seen_splits == ("train", "valid")


def test_se_evalua_sobre_valid_no_sobre_train(registered, prepared):
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    assert outcome.metrics["n_images"] == 4


def test_la_procedencia_del_dataset_queda_registrada(registered, prepared):
    """El commit fija el codigo; las imagenes no estan en git, asi que sin esto
    dos ejecuciones del mismo commit podrian ser sobre datos distintos."""
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    record = json.loads((outcome.run.directory / "run.json").read_text(encoding="utf-8"))
    provenance = record["dataset_provenance"][0]
    assert provenance["version"] == "v9"
    assert provenance["exported"] == "2026-09-10"
    assert provenance["origin"] == "https://ejemplo/proyecto"


def test_los_pesos_se_copian_dentro_de_la_ejecucion(registered, prepared):
    """Si viven fuera, en dos semanas nadie sabe que pesos dieron ese metrics.json."""
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    assert outcome.weights.read_bytes() == b"pesos de mentira"
    assert outcome.run.directory in outcome.weights.parents


def test_se_pueden_reutilizar_pesos_sin_entrenar(registered, prepared, tmp_path):
    root, splits_dir, config = prepared
    weights = tmp_path / "previos.pt"
    weights.write_bytes(b"previos")
    _FakeDetector.seen_splits = ()

    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET, weights=weights,
    )
    assert _FakeDetector.seen_splits == (), "no debia entrenar"
    assert outcome.weights.read_bytes() == b"previos"


def test_el_resumen_dice_lo_que_decide(registered, prepared):
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    summary = outcome.summary()
    assert "mAP50" in summary and "cobertura p5" in summary
    assert "contaminacion" in summary


# --- coherencia del registro de datasets ----------------------------------


def test_un_dataset_apto_no_puede_agregar_fuentes_copyleft():
    """La licencia de un agregado no anula la de sus fuentes."""
    with pytest.raises(DatasetError, match="no anula"):
        DatasetInfo(
            name="incoherente",
            root=Path("."),
            license="CC-BY-4.0",
            production_ready=True,
            sources=("CC-BY-4.0", "AGPL-3.0"),
        )


def test_sources_no_puede_estar_vacio():
    """Vacio por descuido pasaria la comprobacion de licencias sin mirar nada."""
    with pytest.raises(DatasetError, match="sources"):
        DatasetInfo(
            name="sin_fuentes",
            root=Path("."),
            license="CC-BY-4.0",
            production_ready=True,
            sources=(),
        )


def test_el_dataset_real_declara_atribucion():
    real = datasets_mod.get("annotated-banknotes-2")
    assert real.license == "CC-BY-4.0"
    assert any("atribucion" in n.lower() for n in real.notes)
    assert real.origin.startswith("https://universe.roboflow.com/")


# --- avisos de lectura ----------------------------------------------------


def test_entrenar_con_padding_e_inferir_sin_el_se_avisa(registered, prepared):
    """Un desajuste silencioso haria leer mal el resultado: sin el aviso, un mal
    numero se atribuiria al padding y no al desajuste."""
    root, splits_dir, config = prepared
    config = config.model_copy(
        update={
            "detector": config.detector.model_copy(
                update={
                    "out_of_bounds": OutOfBoundsPolicy.PAD,
                    "pad_at_inference": False,
                }
            )
        }
    )
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    assert any("desajustado" in c for c in outcome.run.record.caveats)
    assert "AVISO" in outcome.summary()
    record = json.loads((outcome.run.directory / "run.json").read_text(encoding="utf-8"))
    assert record["caveats"]


def test_sin_padding_no_hay_aviso(registered, prepared):
    root, splits_dir, config = prepared
    outcome = run_candidate(
        registered, config, splits_dir=splits_dir, data_root=root,
        dataset_name=FAKE_DATASET,
    )
    assert outcome.run.record.caveats == []
