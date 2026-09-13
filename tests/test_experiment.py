"""Experiment engine: provenance, run directory and comparison."""

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
    "aggregate", "CC-BY-4.0", True, sources=("CC-BY-4.0", "AGPL-3.0")
)


def make_run(tmp_path, **kwargs) -> ExperimentRun:
    config = Config()
    return ExperimentRun.create(
        kwargs.pop("name", "trial"),
        config,
        runs_dir=tmp_path,
        moment=kwargs.pop("moment", MOMENT),
        **kwargs,
    )


# --- provenance -----------------------------------------------------------


def test_without_a_git_repository_the_reason_is_recorded(tmp_path):
    """Absence is recorded as absence, not as a default value."""
    info = collect_git(tmp_path)
    assert info.available is False
    assert info.commit is None
    assert info.reason and "git init" in info.reason
    assert "no git" in info.describe()


def test_without_git_the_run_is_not_reproducible(tmp_path):
    prov = Provenance(
        git=collect_git(tmp_path),
        seeds=seed_everything(1),
        environment=collect_environment(),
    )
    assert prov.reproducible is False
    assert len(prov.blockers()) == 1
    assert "cannot be reproduced" in prov.blockers()[0]


def test_the_same_seed_gives_the_same_sequence():
    seed_everything(123)
    first = [random.random() for _ in range(5)]
    seed_everything(123)
    assert [random.random() for _ in range(5)] == first


def test_what_was_really_fixed_is_recorded():
    """torch is optional: if it is not there, its seed is None, not a fake number."""
    seeds = seed_everything(7)
    assert seeds.base == 7 and seeds.python == 7
    try:
        import torch  # noqa: F401
    except ImportError:
        assert seeds.torch is None
    else:
        assert seeds.torch == 7


def test_the_environment_only_lists_what_is_installed():
    env = collect_environment()
    assert env.python and env.platform
    assert "numpy" in env.packages and "shapely" in env.packages


# --- the run directory ----------------------------------------------------


def test_the_full_structure_is_created(tmp_path):
    run = make_run(tmp_path)
    assert run.directory.name == "20260910T183000Z_trial"
    assert (run.directory / "config.yaml").exists()
    assert (run.directory / "run.json").exists()
    assert run.weights_dir.is_dir() and run.viz_dir.is_dir()


def test_the_config_is_frozen_before_running(tmp_path):
    """If the run crashes halfway, what it was launched with has to remain."""
    run = make_run(tmp_path)
    text = (run.directory / "config.yaml").read_text(encoding="utf-8")
    assert "min_relative_area" in text
    assert "visibility_threshold" in text


def test_an_existing_run_is_not_overwritten(tmp_path):
    make_run(tmp_path)
    with pytest.raises(FileExistsError):
        make_run(tmp_path)


def test_metrics_update_both_files(tmp_path):
    run = make_run(tmp_path, detector=APACHE_DETECTOR)
    run.write_metrics({"map50": 0.81, "n_images": 101})
    assert json.loads((run.directory / "metrics.json").read_text())["map50"] == 0.81
    assert load_run(run.directory)["metrics"]["n_images"] == 101


# --- the license chain ----------------------------------------------------


def test_agpl_detector_is_not_fit(tmp_path):
    run = make_run(tmp_path, detector=AGPL_DETECTOR, datasets=(CLEAN_DATASET,))
    assert run.record.production_ready is False
    assert any("AGPL-3.0" in b for b in run.record.blockers())


def test_a_contaminated_source_drags_the_aggregate(tmp_path):
    """The license of an aggregate does not override that of its sources."""
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(DIRTY_DATASET,))
    assert run.record.production_ready is False
    assert any("does not override" in b for b in run.record.blockers())


def test_a_clean_chain_is_fit(tmp_path):
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,))
    assert run.record.production_ready is True


def test_without_a_declared_detector_it_is_not_fit(tmp_path):
    assert make_run(tmp_path).record.production_ready is False


# --- the comparison table -------------------------------------------------


def test_compare_ignores_auxiliary_directories(tmp_path):
    """`runs/_inspection` is not a run."""
    make_run(tmp_path, detector=APACHE_DETECTOR)
    (tmp_path / "_inspection").mkdir()
    (tmp_path / "_inspection" / "something.png").write_bytes(b"")
    assert len(discover_runs(tmp_path)) == 1


def test_every_row_carries_n_and_an_interval(tmp_path):
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


def test_a_metric_without_an_interval_is_said(tmp_path):
    """A bare number invites reading as improvement what is dispersion."""
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,))
    run.write_metrics({"n_images": 101, "map50": {"value": 0.81}})
    _, table = compare_mod.compare(tmp_path)
    assert "no CI" in table


def test_the_table_marks_the_unfit_and_says_why(tmp_path):
    run = make_run(tmp_path, name="base", detector=AGPL_DETECTOR)
    run.write_metrics({"n_images": 101, "map50": {"value": 0.9}})
    _, table = compare_mod.compare(tmp_path)
    assert compare_mod.NOT_READY in table
    assert "AGPL-3.0" in table
    assert "performance reference, not as a candidate" in table


def test_the_table_separates_unfit_from_not_reproducible(tmp_path):
    """They are two different problems with different fixes.

    This run is FIT (Apache/CC-BY detector and dataset) but NOT reproducible
    (the project is not in git). Putting them in the same list would leave
    the reader not knowing which of the two things they are reading.
    """
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,))
    run.write_metrics({"n_images": 101, "map50": {"value": 0.8}})
    rows, table = compare_mod.compare(tmp_path)
    assert rows[0]["ready"] == "yes"
    assert rows[0]["reproducible"] == "NO"
    assert compare_mod.NOT_READY not in table
    assert "NOT REPRODUCIBLE" in table
    # The concrete reason depends on the state of the repository the tests
    # run in (no repo, no commits, or dirty tree), so what is checked is that
    # there is a reason and that it goes in the provenance list, not its
    # exact text.
    assert rows[0]["_provenance_blockers"]


def test_the_table_shows_the_split_version_and_warns_when_they_differ(tmp_path):
    """Rows on different split versions were evaluated on different images."""
    a = make_run(tmp_path, name="a", detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,),
                 split={"version": "v1", "mode": "adopt", "digest": "x"})
    a.write_metrics({"n_images": 101, "map50": {"value": 0.8}})
    b = make_run(tmp_path, name="b", detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,),
                 split={"version": "v2", "mode": "repartition", "digest": "y"},
                 moment=MOMENT.replace(minute=31))
    b.write_metrics({"n_images": 76, "map50": {"value": 0.8}})
    rows, table = compare_mod.compare(tmp_path)
    assert {r["split"] for r in rows} == {"v1", "v2"}
    assert "NOT comparable" in table


def test_an_old_run_without_split_info_still_lists(tmp_path):
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,))
    run.write_metrics({"n_images": 101, "map50": {"value": 0.8}})
    rows, table = compare_mod.compare(tmp_path)
    assert rows[0]["split"] == "-"
    assert "NOT comparable" not in table


def test_the_csv_carries_the_same_columns(tmp_path):
    run = make_run(tmp_path, detector=APACHE_DETECTOR, datasets=(CLEAN_DATASET,))
    run.write_metrics({"n_images": 101, "map50": {"value": 0.8, "ci_low": 0.7, "ci_high": 0.9}})
    rows, _ = compare_mod.compare(tmp_path)
    path = compare_mod.write_csv(rows, tmp_path / "out" / "compare.csv")
    text = path.read_text(encoding="utf-8")
    assert "mAP50" in text and "ready" in text and "split" in text and "101" in text


def test_without_runs_it_says_so(tmp_path):
    _, table = compare_mod.compare(tmp_path)
    assert "No runs" in table
