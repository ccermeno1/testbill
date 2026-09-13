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
    list_versions,
    materialize_splits,
    record_test_access,
)

LINE = "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n"


def make_sample(directory, stem):
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "labels").mkdir(parents=True, exist_ok=True)
    # A minimal valid JPEG is not needed: discover only looks at the extension.
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


# -- detection --------------------------------------------------------------


def test_detects_adopt_mode(tmp_path):
    layout = detect_layout(roboflow_root(tmp_path))
    assert layout.mode is LayoutMode.ADOPT
    assert len(layout.groups["train"]) == 6


def test_detects_create_mode(tmp_path):
    layout = detect_layout(flat_root(tmp_path))
    assert layout.mode is LayoutMode.CREATE
    assert len(layout.all_samples) == 10


def test_accepts_val_as_alias_but_records_it(tmp_path):
    """Roboflow uses 'valid'. 'val' appears in hand-edited exports."""
    root = roboflow_root(tmp_path)
    (root / "valid").rename(root / "val")
    layout = detect_layout(root)
    assert layout.mode is LayoutMode.ADOPT
    assert any("val" in note for note in layout.notes)


def test_valid_and_val_at_the_same_time_is_ambiguous(tmp_path):
    root = roboflow_root(tmp_path)
    make_sample(root / "val", "other_000")
    with pytest.raises(LayoutError, match="ambiguous"):
        detect_layout(root)


def test_image_without_label_is_fatal(tmp_path):
    """Treating it as 'image without banknotes' silently poisons training."""
    root = roboflow_root(tmp_path)
    (root / "train" / "images" / "orphan.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    with pytest.raises(LayoutError, match="without a label file"):
        detect_layout(root)


def test_label_without_image_is_fatal(tmp_path):
    root = roboflow_root(tmp_path)
    (root / "train" / "labels" / "orphan.txt").write_text(LINE, encoding="utf-8")
    with pytest.raises(LayoutError, match="without an image"):
        detect_layout(root)


def test_half_built_structure_is_fatal(tmp_path):
    root = tmp_path / "data"
    make_sample(root / "train", "img_000")
    with pytest.raises(LayoutError, match="half-built"):
        detect_layout(root)


# -- adopt mode -------------------------------------------------------------


def test_adopt_freezes_the_export_split(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    manifest = materialize_splits(root, splits)
    assert manifest["mode"] == "adopt"
    assert manifest["counts"] == {"train": 6, "valid": 3, "test": 2}
    assert manifest["version"] == "v1"
    assert (splits / "v1" / "valid.txt").is_file()


def test_adopt_does_not_require_confirming_independence(tmp_path):
    """In adopt mode we do not choose the split, we only freeze it."""
    materialize_splits(roboflow_root(tmp_path), tmp_path / "splits")


def test_each_make_splits_writes_the_next_version_and_keeps_the_previous(tmp_path):
    """Versions are immutable: redoing the split never destroys the export's."""
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    first = materialize_splits(root, splits)
    second = materialize_splits(root, splits)
    assert (first["version"], second["version"]) == ("v1", "v2")
    assert list_versions(splits) == ["v1", "v2"]
    assert (splits / "v1" / "train.txt").is_file() and (splits / "v2" / "train.txt").is_file()


def test_refuses_to_overwrite_an_existing_version(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    materialize_splits(root, splits)
    with pytest.raises(SplitError, match="immutable"):
        materialize_splits(root, splits, version="v1")
    materialize_splits(root, splits, version="v1", overwrite=True)
    with pytest.raises(SplitError, match="named vN"):
        materialize_splits(root, splits, version="latest")


def test_versions_sort_numerically_not_lexicographically(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    for name in ("v1", "v9", "v10"):
        materialize_splits(root, splits, version=name)
    assert list_versions(splits) == ["v1", "v9", "v10"]
    assert SplitLoader(splits, root).version == "v10"


def test_the_loader_reads_the_latest_by_default_and_a_pinned_one_on_request(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    materialize_splits(root, splits)
    materialize_splits(
        root, splits, repartition=True, ratios=(0.5, 0.25, 0.25), seed=3,
        groups=GroupConfig(independence_confirmed=True),
    )
    latest = SplitLoader(splits, root)
    pinned = SplitLoader(splits, root, version="v1")
    assert latest.version == "v2" and latest.describe()["mode"] == "repartition"
    assert pinned.version == "v1" and pinned.describe()["mode"] == "adopt"
    assert latest.describe()["digest"] != pinned.describe()["digest"]
    with pytest.raises(SplitError, match="does not exist"):
        SplitLoader(splits, root, version="v7")


def test_without_any_version_the_loader_says_so(tmp_path):
    with pytest.raises(SplitError, match="make-splits"):
        SplitLoader(tmp_path / "empty", roboflow_root(tmp_path))


# -- create mode ------------------------------------------------------------


def test_create_requires_confirming_independence(tmp_path):
    with pytest.raises(SplitError, match="i-confirm-independence"):
        materialize_splits(flat_root(tmp_path), tmp_path / "splits")


def test_create_with_confirmation(tmp_path):
    manifest = materialize_splits(
        flat_root(tmp_path),
        tmp_path / "splits",
        groups=GroupConfig(independence_confirmed=True),
        ratios=(0.6, 0.2, 0.2),
        seed=7,
    )
    assert manifest["mode"] == "create"
    assert sum(manifest["counts"].values()) == 10


def test_create_is_deterministic_with_the_same_seed(tmp_path):
    root = flat_root(tmp_path)
    groups = GroupConfig(independence_confirmed=True)
    a = materialize_splits(root, tmp_path / "a", groups=groups, seed=99)
    b = materialize_splits(root, tmp_path / "b", groups=groups, seed=99)
    c = materialize_splits(root, tmp_path / "c", groups=groups, seed=100)
    assert a["digest"] == b["digest"]
    assert a["digest"] != c["digest"]


def test_filename_prefix_requires_regex(tmp_path):
    with pytest.raises(SplitError, match="group-regex"):
        materialize_splits(
            flat_root(tmp_path),
            tmp_path / "splits",
            groups=GroupConfig(strategy=GroupStrategy.FILENAME_PREFIX),
        )


def test_a_group_is_not_split_across_partitions(tmp_path):
    """Several shots of the same physical banknote fall together."""
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
        name: (tmp_path / "splits" / "v1" / f"{name}.txt").read_text(encoding="utf-8")
        for name in ("train", "valid", "test")
    }
    for physical in range(4):
        homes = [
            name
            for name, text in assignment.items()
            if f"bill{physical:02d}_" in text
        ]
        assert len(homes) == 1, f"group bill{physical:02d} was split: {homes}"
    assert manifest["group_strategy"] == "filename-prefix"


# -- integrity and sealing --------------------------------------------------


def test_a_sample_in_two_splits_is_fatal(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    materialize_splits(root, splits)
    with (splits / "v1" / "valid.txt").open("a", encoding="utf-8") as handle:
        handle.write("img_000\n")  # already in train
    with pytest.raises(SplitError, match="img_000"):
        SplitLoader(splits, root)


def test_split_sample_missing_from_disk_is_fatal(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    materialize_splits(root, splits)
    (root / "train" / "images" / "img_000.jpg").unlink()
    (root / "train" / "labels" / "img_000.txt").unlink()
    with pytest.raises(SplitError, match="are not in"):
        SplitLoader(splits, root)


def test_test_is_sealed(tmp_path):
    root = roboflow_root(tmp_path)
    splits = tmp_path / "splits"
    materialize_splits(root, splits)
    loader = SplitLoader(splits, root)
    with pytest.raises(SealedTestSetError, match="sealed"):
        loader.load("test")
    assert len(loader.load("test", allow_test=True)) == 2


def test_every_access_to_test_is_recorded(tmp_path):
    log = tmp_path / "test_evaluations.jsonl"
    record_test_access("final evaluation", "run_a", log_path=log)
    record_test_access("second look", "run_b", log_path=log)
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [e["run"] for e in entries] == ["run_a", "run_b"]
