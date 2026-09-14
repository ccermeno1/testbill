"""Command-line interface.

Every read of splits goes through SplitLoader. Test is sealed: the only
command that opens it is `evaluate-test`, which requires a written reason,
logs it in runs/test_evaluations.jsonl and shows the previous accesses before
evaluating.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from testbank.checks.visibility import check_samples
from testbank.config import Config, OutOfBoundsPolicy
from testbank.data.datasets import DEFAULT_DATASET, datasets
from testbank.data.discover import LayoutMode, detect_layout
from testbank.data.duplicates import (
    DEFAULT_THRESHOLD,
    find_duplicates,
    write_manifest,
)
from testbank.data.splits import (
    TEST_LOG_PATH,
    GroupConfig,
    GroupStrategy,
    SplitError,
    SplitLoader,
    materialize_splits,
    read_test_accesses,
)
from testbank.dataio.export import export, exporters
from testbank.dataio.export import get as get_exporter
from testbank.dataio.formats import get as get_format
from testbank.dataio.image_sizes import SizeIndex
from testbank.detectors import detectors
from testbank.detectors import get as get_detector
from testbank.experiment.compare import compare, write_csv
from testbank.experiment.run import load_run
from testbank.experiment.runner import (
    TEST_METRICS,
    evaluate_on_test,
    run_candidate,
)
from testbank.viz.inspect import fixed_validation_sample, inspect_samples

#: How many entries are listed on screen before summarizing. The JSON report
#: carries them all; this is only so the output fits in a terminal.
_MAX_LISTED = 20


def _loader(args, config: Config) -> SplitLoader:
    """Every command reads the split through here: same container, same
    version rule (`--split-version`, then the config, then the latest)."""
    return SplitLoader(
        args.splits_dir or config.data.splits_dir,
        args.data_root,
        version=args.split_version or config.data.split_version,
    )


def _group_config(args, config: Config) -> GroupConfig:
    strategy = GroupStrategy(args.group_key or config.splits.group_key)
    return GroupConfig(
        strategy=strategy,
        regex=args.group_regex or config.splits.group_regex,
        manifest_path=Path(args.group_manifest)
        if args.group_manifest
        else config.splits.group_manifest,
        independence_confirmed=args.i_confirm_independence
        or config.splits.i_confirm_independence,
    )


def cmd_detect(args, config: Config) -> int:
    layout = detect_layout(args.data_root or config.data.root)
    print(f"Mode: {layout.mode.value}")
    if layout.mode is LayoutMode.ADOPT:
        for split in ("train", "valid", "test"):
            print(f"  {split:6s} {len(layout.groups.get(split, ())):5d} samples")
    else:
        print(f"  no previous split: {len(layout.all_samples)} samples")
    for note in layout.notes:
        print(f"  warning: {note}")
    return 0


def _parse_ratios(text: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(x) for x in text.split(","))
    except ValueError:
        raise SystemExit(f"--ratios: {text!r} are not three numbers") from None
    if len(parts) != 3:
        raise SystemExit(f"--ratios: three values are needed (train,valid,test), got {len(parts)}")
    if abs(sum(parts) - 1.0) > 1e-9:
        raise SystemExit(f"--ratios: must add up to 1, they add up to {sum(parts):g}")
    if any(x < 0 for x in parts):
        raise SystemExit("--ratios: no proportion can be negative")
    return parts


def cmd_make_splits(args, config: Config) -> int:
    ratios = _parse_ratios(args.ratios) if args.ratios else config.splits.ratios
    manifest = materialize_splits(
        args.data_root or config.data.root,
        args.splits_dir or config.data.splits_dir,
        groups=_group_config(args, config),
        ratios=ratios,
        seed=args.seed if args.seed is not None else config.splits.seed,
        overwrite=args.overwrite,
        repartition=args.repartition,
        stratify_regex=args.stratify_regex,
        version=args.split_version,
    )
    print(f"Split version: {manifest['version']}  (mode: {manifest['mode']})")
    if manifest["mode"] == "repartition":
        print("  (the export's split has been IGNORED; this one is ours)")
    if manifest.get("ratios"):
        r = manifest["ratios"]
        print(f"  ratios {r['train']:g}/{r['valid']:g}/{r['test']:g}   seed {manifest['seed']}")
    for split, count in manifest["counts"].items():
        print(f"  {split:6s} {count:5d}")
    if manifest.get("strata"):
        # One row per stratum, to see at a glance that no split has been left
        # without a banknote type.
        print("  per stratum:      train  valid   test")
        for stratum, counts in sorted(manifest["strata"].items()):
            print(
                f"    {stratum:12s}  {counts['train']:5d}  {counts['valid']:5d}  "
                f"{counts['test']:5d}"
            )
    print(f"  digest {manifest['digest'][:16]}")
    for note in manifest["notes"]:
        print(f"  warning: {note}")
    print(
        f"  Commands read the latest version by default; "
        f"--split-version {manifest['version']} pins this one."
    )
    return 0


def cmd_check_visibility(args, config: Config) -> int:
    loader = _loader(args, config)
    splits = args.splits or ["train", "valid"]
    samples = [s for split in splits for s in loader.load(split)]
    sizes = SizeIndex.for_samples(
        samples, cache_path=config.data.derived_dir / "image_sizes.json"
    )
    min_relative_area = (
        args.min_relative_area
        if args.min_relative_area is not None
        else config.annotation_policy.min_relative_area
    )
    report, filter_report = check_samples(
        samples,
        visibility_threshold=args.visibility_threshold
        or config.annotation_policy.visibility_threshold,
        sizes=sizes,
        min_relative_area=min_relative_area,
    )
    for line in filter_report.summary_lines():
        print(line)
    if filter_report.dropped:
        shown = sorted(filter_report.dropped, key=lambda d: d.relative_area)
        print("\nDiscarded by the filter (still in the source dataset):")
        for dropped in shown[:_MAX_LISTED]:
            print(f"  - {dropped.describe()}")
        if len(shown) > _MAX_LISTED:
            print(f"  ... and {len(shown) - _MAX_LISTED} more")
    print()
    print(report.summary())
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "visibility": report.to_dict(),
            "relative_area_filter": filter_report.to_dict(),
        }
        out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON report: {args.json_out}")
    # A warning, not a fatal error.
    return 0


def cmd_find_duplicates(args, config: Config) -> int:
    loader = _loader(args, config)
    # All THREE splits are looked at, test included. It is not breaking the
    # seal: no annotations are read and nothing is evaluated, it only checks
    # whether a test shot is also in train. That is precisely what one needs
    # to know.
    samples, split_of = [], {}
    for split in ("train", "valid", "test"):
        for sample in loader.load(split, allow_test=True):
            samples.append(sample)
            split_of[sample.sample_id] = split

    report = find_duplicates(samples, split_of=split_of, threshold=args.threshold)
    sizes: dict[str, int] = {}
    for split in split_of.values():
        sizes[split] = sizes.get(split, 0) + 1
    for line in report.summary_lines(sizes):
        print(line)

    if report.crossing:
        print("\nPairs crossing splits:")
        for pair in report.crossing[:_MAX_LISTED]:
            print(f"  {pair.describe()}")
        if len(report.crossing) > _MAX_LISTED:
            print(f"  ... and {len(report.crossing) - _MAX_LISTED} more")
        print(
            "\n  The adopted split is NOT recomputed: the export decides it. "
            "This is a warning so the numbers are read knowing it."
        )

    if args.manifest_out:
        path = write_manifest(report, args.manifest_out)
        print(f"\nGroup manifest: {path}")
        print(
            "  Use it when splitting:\n"
            f"    testbank make-splits --repartition --group-key manifest --group-manifest {path}"
        )
    return 0


def cmd_export(args, config: Config) -> int:
    if args.out_of_bounds is not None:
        config = config.model_copy(
            update={
                "detector": config.detector.model_copy(
                    # The enum, not the string: `model_copy(update=...)` does NOT validate.
                    update={"out_of_bounds": OutOfBoundsPolicy(args.out_of_bounds)}
                )
            }
        )
    loader = _loader(args, config)
    splits = args.splits or ["train", "valid"]
    samples = {split: list(loader.load(split)) for split in splits}

    result = export(
        args.format,
        samples,
        config,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    print(f"Format:       {result.name}")
    print(f"Root:         {result.root}")
    print(f"Entry point:  {result.entry_point}")
    print(f"  {result.report.describe()}")
    # The reason comes from the format, not from this function: a warning that
    # lies about what is lost is worse than no warning.
    annotation_format = get_format(get_exporter(args.format).annotation_format)
    if annotation_format.lossy:
        print(f"  WARNING: LOSSY format -- {annotation_format.lossy_reason}.")
    return 0


def cmd_evaluate_test(args, config: Config) -> int:
    """The only path to the sealed set. Always leaves a record."""
    run_directory = Path(args.run)
    record = load_run(run_directory)
    weights = Path(args.weights) if args.weights else _only_weights(run_directory)

    previous = read_test_accesses()
    if previous:
        print(f"WARNING: the test has already been opened {len(previous)} time(s):")
        for entry in previous[-_MAX_LISTED:]:
            print(
                f"  {entry.get('utc', '?')}  {entry.get('run', '?')}  "
                f"-- {entry.get('reason', '')}"
            )
        print(
            "  Every additional evaluation spends the set: choosing among "
            "several is tuning to the test even without touching the model.\n"
        )

    outcome = evaluate_on_test(
        get_detector(record["detector"]["name"]),
        config,
        run_directory=run_directory,
        weights=weights,
        reason=args.reason,
        splits_dir=args.splits_dir or config.data.splits_dir,
        data_root=args.data_root,
        split_version=args.split_version or config.data.split_version,
    )
    print(outcome.summary())
    print(
        f"\nAccess number {outcome.entry['access_number']}, "
        f"logged in {TEST_LOG_PATH}"
    )
    print(f"Metrics: {run_directory / TEST_METRICS}")
    return 0


def _only_weights(run_directory: Path) -> Path:
    found = sorted((run_directory / "weights").glob("*"))
    if len(found) != 1:
        raise SplitError(
            f"{run_directory}: a single weights file was expected, there are "
            f"{len(found)}; say which one with --weights"
        )
    return found[0]


def cmd_train(args, config: Config) -> int:
    # Overrides are substituted INSIDE the config, not passed loose to the
    # detector: the resolved config is what gets frozen in the run, and two
    # runs with different epochs or border policy are not comparable. If the
    # override did not get there, the config.yaml would lie about what ran.
    updates = {}
    if args.epochs is not None:
        updates["epochs"] = args.epochs
    if args.image_size is not None:
        updates["image_size"] = args.image_size
    if args.augment:
        updates["augment"] = config.detector.augment.model_copy(update={"enabled": True})
    if args.out_of_bounds is not None:
        updates["out_of_bounds"] = OutOfBoundsPolicy(args.out_of_bounds)
    if args.pretrained is not None:
        updates["pretrained"] = Path(args.pretrained).resolve()
    if args.loss_recipe is not None:
        # Through the constructor, which validates; `model_copy(update=...)` does not.
        updates["loss"] = config.detector.loss.model_validate(
            {**config.detector.loss.model_dump(), "recipe": args.loss_recipe}
        )
    if updates:
        config = config.model_copy(
            update={"detector": config.detector.model_copy(update=updates)}
        )
    outcome = run_candidate(
        get_detector(args.detector),
        config,
        splits_dir=args.splits_dir or config.data.splits_dir,
        data_root=args.data_root,
        dataset_name=args.dataset,
        run_name=args.name,
        weights=Path(args.weights) if args.weights else None,
        split_version=args.split_version or config.data.split_version,
    )
    print(outcome.summary())
    print(f"\nVisualizations: {outcome.run.viz_dir}")
    return 0


def cmd_list(args, config: Config) -> int:
    print("Registered detectors:")
    for name in detectors():
        detector = get_detector(name)
        mark = "" if detector.production_ready else "   NOT READY for production"
        print(f"  {name:24s} {detector.license:12s}{mark}")
    print("\nRegistered datasets:")
    for name in datasets():
        print(f"  {name}")
    return 0


def cmd_compare(args, config: Config) -> int:
    runs_dir = Path(args.runs_dir or config.runs_dir)
    rows, table = compare(runs_dir)
    print(table)
    if args.csv_out and rows:
        path = write_csv(rows, args.csv_out)
        print(f"\nCSV: {path}")
    return 0


def cmd_inspect(args, config: Config) -> int:
    loader = _loader(args, config)
    samples = fixed_validation_sample(
        loader.load(args.split),
        count=args.count or config.viz.sample_count,
        seed=config.viz.seed,
    )
    output_dir = Path(args.output_dir or config.viz.output_dir)
    results = inspect_samples(
        samples,
        output_dir,
        visibility_threshold=config.annotation_policy.visibility_threshold,
        min_relative_area=config.annotation_policy.min_relative_area,
    )
    print(f"Fixed sample of '{args.split}' ({len(results)} images, seed {config.viz.seed})")
    for result in results:
        notes = []
        if result.flagged:
            notes.append(f"{result.flagged} flagged")
        if result.dropped:
            notes.append(f"{result.dropped} filtered")
        suffix = f"  <- {', '.join(notes)}" if notes else ""
        print(f"  {result.sample_id}  {result.annotations} annotations{suffix}")
    print(f"\nOutput: {output_dir}")
    print(f"Contact sheet: {output_dir / '_contact_sheet.png'}")
    return 0


def cmd_plot_training(args, config: Config) -> int:
    from testbank.viz.curves import plot_run

    path = plot_run(args.run, args.output)
    print(f"Training curves: {path}")
    return 0


def cmd_app(args, config: Config) -> int:
    """Starts the Streamlit front end in this interpreter's environment, so
    the runs it can predict with are the ones whose adapter imports here."""
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is not installed: `uv pip install -e .[app]`", file=sys.stderr)
        return 2
    from testbank import app

    runs_dir = Path(args.runs_dir or config.runs_dir)
    command = [sys.executable, "-m", "streamlit", "run", app.__file__]
    if args.port:
        command += ["--server.port", str(args.port)]
    command += ["--", "--runs-dir", str(runs_dir)]
    return subprocess.call(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="testbank", description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--splits-dir", type=Path, default=None,
        help="container of split versions (default: data.splits_dir, `splits/`)",
    )
    parser.add_argument(
        "--split-version", default=None, metavar="vN",
        help=(
            "which split version to read (default: data.split_version, or the "
            "latest). For make-splits: the name of the version to write "
            "(default: the next one)"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("detect", help="inspects the directory structure")

    make = subparsers.add_parser(
        "make-splits",
        help="materializes a NEW split version under splits/ (v1, v2, ...)",
    )
    make.add_argument("--group-key", choices=[s.value for s in GroupStrategy], default=None)
    make.add_argument("--group-regex", default=None)
    make.add_argument("--group-manifest", default=None)
    make.add_argument(
        "--i-confirm-independence",
        action="store_true",
        help="explicitly asserts that every image is independent",
    )
    make.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed of the split; with the same seed and the same data the same "
        "split comes out, and the manifest stores a digest to check it",
    )
    make.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "redo the target version in place (only with --split-version); "
            "never needed to add a new version"
        ),
    )
    make.add_argument(
        "--ratios",
        default=None,
        metavar="TRAIN,VALID,TEST",
        help="proportions of the split, e.g. 0.8,0.1,0.1; must add up to 1. Only "
        "when we do the splitting (create mode or --repartition). By default, "
        "the config's: 0.70,0.15,0.15",
    )
    make.add_argument(
        "--repartition",
        action="store_true",
        help=(
            "ignores the export's split and splits from scratch, by groups. It is "
            "the explicit exception to 'in adopt mode the export rules': it exists "
            "because Roboflow's has 20%% of the test contaminated by "
            "near-duplicates. It is written in the manifest"
        ),
    )
    make.add_argument(
        "--stratify-regex",
        default=None,
        help=(
            "regex with ONE capture group over the sample name; every split "
            "receives its proportion of each captured value. For the Roboflow "
            r"export: '^(\d+|Multiple)_' (the banknote type)"
        ),
    )

    visibility = subparsers.add_parser(
        "check-visibility", help="consistency of the visibility policy"
    )
    visibility.add_argument("--splits", nargs="+", default=None)
    visibility.add_argument("--visibility-threshold", type=float, default=None)
    visibility.add_argument(
        "--min-relative-area",
        type=float,
        default=None,
        help=(
            "threshold of the relative area filter; 0 disables it and shows "
            "all annotations as they are in the file"
        ),
    )
    visibility.add_argument("--json-out", default=None)

    train = subparsers.add_parser(
        "train", help="trains a candidate and leaves the run in runs/"
    )
    train.add_argument("detector", help="registered name; see `list`")
    train.add_argument("--dataset", default=DEFAULT_DATASET)
    train.add_argument("--name", default=None, help="name of the run")
    train.add_argument(
        "--out-of-bounds",
        choices=[p.value for p in OutOfBoundsPolicy],
        default=None,
        help=(
            "banknotes crossing the border: clip trims them to the frame (by "
            "default), pad adds a black border so they fit whole (and pads at "
            "inference too), keep touches nothing and Ultralytics will discard "
            "those images"
        ),
    )
    train.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="overrides detector.epochs; it is recorded in the resolved config",
    )
    train.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=(
            "input side in pixels for every candidate (own head, port, "
            "Ultralytics, RTMDet-R); overrides detector.image_size and is "
            "recorded. The export is 416x416: 416 uses the pixels as they are"
        ),
    )
    train.add_argument(
        "--augment",
        action="store_true",
        help=(
            "training-time augmentation for the candidates trained by this "
            "loop (own variants, DDGRCF port, Rotated FCOS), Ultralytics' "
            "recipe by default: mosaic, scale/translate, HSV, horizontal flip; "
            "details in detector.augment. Off by default; recorded in the run"
        ),
    )
    train.add_argument(
        "--loss-recipe",
        choices=["own", "yolox_obb_fork", "ultralytics_obb", "ddgrcf"],
        default=None,
        help=(
            "only for the own head: which WHOLE loss recipe trains. "
            "own = the own one; yolox_obb_fork = KLD x5 + obj + cls + late L1 with "
            "SimOTA; ultralytics_obb = ProbIoU + DFL + soft cls with TAL, which "
            "also changes the head (DFL regression, scalar angle, no obj); "
            "ddgrcf = exact PolyIoU x5 + obj + cls + late L1 with SimOTA, on the "
            "DDGRCF port (the yolox-obb-ddgrcf-port candidate forces it)"
        ),
    )
    train.add_argument(
        "--pretrained",
        default=None,
        help=(
            "foreign checkpoint to START training from (recorded in the "
            "config): a Megvii yolox_*.pth.tar for the own head, or DDGRCF's "
            "DOTA one for its port. Not to be confused with --weights"
        ),
    )
    train.add_argument(
        "--weights",
        default=None,
        help="reuses these weights instead of training (to re-evaluate)",
    )

    subparsers.add_parser("list", help="registered detectors and datasets")

    duplicates = subparsers.add_parser(
        "find-duplicates", help="near-duplicates and group manifest"
    )
    duplicates.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    duplicates.add_argument(
        "--manifest-out",
        default=None,
        help="writes {identifier: group} for --group-manifest",
    )

    exporter = subparsers.add_parser(
        "export", help="writes the dataset in the format of another trainer"
    )
    exporter.add_argument("format", choices=exporters())
    exporter.add_argument("--splits", nargs="+", default=None)
    exporter.add_argument("--out-dir", default=None)
    exporter.add_argument(
        "--out-of-bounds",
        choices=[p.value for p in OutOfBoundsPolicy],
        default=None,
        help="same as in `train`: border policy applied to the exported labels",
    )

    evaluate = subparsers.add_parser(
        "evaluate-test",
        help="breaks the seal of the test and leaves a record; use once",
    )
    evaluate.add_argument("run", help="run directory in runs/")
    evaluate.add_argument(
        "--reason",
        required=True,
        help="why the test is opened; stays in the log forever",
    )
    evaluate.add_argument("--weights", default=None)

    comparison = subparsers.add_parser(
        "compare", help="comparison table of the runs in runs/"
    )
    comparison.add_argument("--runs-dir", default=None)
    comparison.add_argument("--csv-out", default=None)

    curves = subparsers.add_parser(
        "plot-training",
        help="losses per epoch and validation metrics of a run (own candidates)",
    )
    curves.add_argument("run", help="run directory in runs/ (or the _train/ dir)")
    curves.add_argument(
        "--output", default=None, help="PNG path; default <run>/viz/training.png"
    )

    app = subparsers.add_parser(
        "app", help="Streamlit front end: upload or take photos, pick a run, see crops"
    )
    app.add_argument("--runs-dir", default=None, help="where the trained runs are")
    app.add_argument("--port", type=int, default=None, help="Streamlit's server port")

    inspect = subparsers.add_parser("inspect", help="inspection visualization")
    inspect.add_argument("--split", default="valid")
    inspect.add_argument("--count", type=int, default=None)
    inspect.add_argument("--output-dir", default=None)

    return parser


_COMMANDS = {
    "detect": cmd_detect,
    "make-splits": cmd_make_splits,
    "check-visibility": cmd_check_visibility,
    "train": cmd_train,
    "find-duplicates": cmd_find_duplicates,
    "export": cmd_export,
    "evaluate-test": cmd_evaluate_test,
    "list": cmd_list,
    "compare": cmd_compare,
    "plot-training": cmd_plot_training,
    "app": cmd_app,
    "inspect": cmd_inspect,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config.load(args.config)
    try:
        return _COMMANDS[args.command](args, config)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
