"""`--repartition` and `--stratify-regex`: redo the split without leaks and
with every banknote type on each side.

It exists because Roboflow's split came with near-duplicates crossing splits.
What is checked here is what was asked for, in this order:

1. That no group (same shot) is spread across splits.
2. That every split has representatives of every type.
3. That it is opt-in, is written in the manifest, and is not confused with
   the export's split.
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

TYPES = ("005", "010", "020", "050", "100", "200", "500", "Multiple")


def _type(sid):
    return re.match(r"^(\d+|Multiple)_", sid).group(1)


def roboflow_with_types(tmp_path, *, per_type=8, shots=2):
    """An already split export, with `per_type` banknotes of each type and
    `shots` nearly identical photos of each. Roboflow splits them blindly:
    shots of the same banknote fall in different splits on purpose."""
    root = tmp_path / "data"
    manifest = {}
    splits = ("train", "valid", "test")
    n = 0
    for kind in TYPES:
        for banknote in range(per_type):
            for shot in range(shots):
                stem = f"{kind}_Euro_{banknote:03d}_t{shot}"
                make_sample(root / splits[n % 3], stem)
                manifest[stem] = f"{kind}_Euro_{banknote:03d}"
                n += 1
    path = tmp_path / "dups.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, path


def _read(splits_dir, version="v1"):
    # Through the official reader: the files carry a comment header.
    return {
        name: _read_split_file(splits_dir / version / f"{name}.txt")
        for name in ("train", "valid", "test")
    }


# --- 1. no leaks -----------------------------------------------------------


def test_adopting_with_a_manifest_that_proves_leaks_refuses(tmp_path):
    """In adopt mode the export rules -- but if the manifest proves the export
    spreads a group, the leak is not frozen silently: it stops and the
    message says what the way out is."""
    root, dups = roboflow_with_types(tmp_path)
    with pytest.raises(SplitError, match="--repartition"):
        materialize_splits(
            root,
            tmp_path / "s",
            groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
        )
    assert not (tmp_path / "s" / "v1" / "train.txt").exists(), "it has not written halfway"


def test_with_repartition_no_group_crosses(tmp_path):
    root, dups = roboflow_with_types(tmp_path)
    manifest = materialize_splits(
        root,
        tmp_path / "s",
        groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
        repartition=True,
    )
    assert manifest["mode"] == "repartition"
    assert manifest["export_mode"] == "adopt"
    splits = _read(tmp_path / "s")
    groups = json.loads(dups.read_text(encoding="utf-8"))
    home_of_group: dict[str, set[str]] = {}
    for name, ids in splits.items():
        for sid in ids:
            home_of_group.setdefault(groups[sid], set()).add(name)
    crossing = {g: h for g, h in home_of_group.items() if len(h) > 1}
    assert crossing == {}, f"groups spread across splits: {crossing}"
    assert sum(len(ids) for ids in splits.values()) == len(groups)


# --- 2. representatives of each type ---------------------------------------


def test_stratified_puts_every_type_in_every_split(tmp_path):
    root, dups = roboflow_with_types(tmp_path, per_type=10)
    manifest = materialize_splits(
        root,
        tmp_path / "s",
        groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
        repartition=True,
        stratify_regex=r"^(\d+|Multiple)_",
        ratios=(0.6, 0.2, 0.2),
    )
    for name, ids in _read(tmp_path / "s").items():
        present = {_type(sid) for sid in ids}
        assert present == set(TYPES), f"{name} does not have every type: {present}"
    # And the manifest reports it, per stratum and split.
    assert set(manifest["strata"]) == set(TYPES)
    for counts in manifest["strata"].values():
        assert all(counts[n] > 0 for n in ("train", "valid", "test"))
    assert manifest["stratify_regex"] == r"^(\d+|Multiple)_"


def test_without_stratifying_a_type_can_be_left_out(tmp_path):
    """The reason for stratifying, demonstrated: with few of each type and a
    blind split, some split ends up without one. With strata, never.

    Four groups per type and not two: with two, `round(2 * 0.25) == 0` and no
    strategy can put one in each of three splits. It is not a failure of the
    split, it is arithmetic -- and the manifest shows it in its per-stratum
    table, which is what it is for.
    """
    root, dups = roboflow_with_types(tmp_path, per_type=4)

    blind_misses = 0
    for seed in range(20):
        materialize_splits(
            root, tmp_path / f"blind{seed}",
            groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
            repartition=True, ratios=(0.5, 0.25, 0.25), seed=seed,
        )
        for ids in _read(tmp_path / f"blind{seed}").values():
            blind_misses += len(set(TYPES) - {_type(s) for s in ids})
    assert blind_misses > 0, "the case has to be scarce enough to fail"

    for seed in range(20):
        materialize_splits(
            root, tmp_path / f"strat{seed}",
            groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
            repartition=True, ratios=(0.5, 0.25, 0.25), seed=seed,
            stratify_regex=r"^(\d+|Multiple)_",
        )
        for ids in _read(tmp_path / f"strat{seed}").values():
            assert set(TYPES) - {_type(s) for s in ids} == set()


# --- 3. explicit, written, and with its guards -----------------------------


def test_stratifying_without_repartition_is_an_error(tmp_path):
    root, _ = roboflow_with_types(tmp_path)
    with pytest.raises(SplitError, match="--repartition"):
        materialize_splits(root, tmp_path / "s", stratify_regex=r"^(\d+)_")


def test_the_regex_needs_one_capture_group(tmp_path):
    root, _ = roboflow_with_types(tmp_path)
    with pytest.raises(SplitError, match="one capture group"):
        materialize_splits(
            root, tmp_path / "s", repartition=True, stratify_regex=r"^\d+_",
            groups=GroupConfig(strategy=GroupStrategy.NONE, independence_confirmed=True),
        )


def test_a_sample_without_a_stratum_is_an_error_and_says_which(tmp_path):
    root, _ = roboflow_with_types(tmp_path)
    make_sample(root / "train", "no_type_001")
    with pytest.raises(SplitError, match="no_type_001"):
        materialize_splits(
            root, tmp_path / "s", repartition=True, stratify_regex=r"^(\d+|Multiple)_",
            groups=GroupConfig(strategy=GroupStrategy.NONE, independence_confirmed=True),
        )


def test_a_group_crossing_strata_warns_but_does_not_block(tmp_path):
    """Two 'equal' photos of different banknotes: they are split together and it warns.

    Over-grouping is cheap; blocking the whole split for one pair is not.
    """
    root, dups = roboflow_with_types(tmp_path)
    groups = json.loads(dups.read_text(encoding="utf-8"))
    groups["500_Euro_000_t0"] = groups["005_Euro_000_t0"]  # a 500 glued to a 5
    dups.write_text(json.dumps(groups), encoding="utf-8")
    manifest = materialize_splits(
        root, tmp_path / "s",
        groups=GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
        repartition=True, stratify_regex=r"^(\d+|Multiple)_",
    )
    warnings = [n for n in manifest["notes"] if "crosses strata" in n]
    assert len(warnings) == 1 and "005" in warnings[0] and "500" in warnings[0]
    splits = _read(tmp_path / "s")
    homes = {n for n, ids in splits.items() if "500_Euro_000_t0" in ids}
    homes |= {n for n, ids in splits.items() if "005_Euro_000_t0" in ids}
    assert len(homes) == 1, "they stay together: it is one group"


def test_repartition_is_deterministic(tmp_path):
    root, dups = roboflow_with_types(tmp_path)
    kw = {
        "groups": GroupConfig(strategy=GroupStrategy.MANIFEST, manifest_path=dups),
        "repartition": True,
        "stratify_regex": r"^(\d+|Multiple)_",
        "seed": 7,
    }
    a = materialize_splits(root, tmp_path / "a", **kw)
    b = materialize_splits(root, tmp_path / "b", **kw)
    assert a["digest"] == b["digest"]
    assert _read(tmp_path / "a") == _read(tmp_path / "b")
