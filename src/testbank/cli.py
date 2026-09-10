"""Interfaz de linea de comandos.

Toda lectura de particiones pasa por SplitLoader. Test esta sellado: ningun
comando de aqui lo abre salvo `evaluate-test`, que llegara con el motor de
experimentos y registrara cada acceso.
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
from testbank.data.splits import (
    GroupConfig,
    GroupStrategy,
    SplitLoader,
    extend_splits,
    make_folds,
    materialize_splits,
)
from testbank.dataio.image_sizes import SizeIndex
from testbank.detectors import detectors
from testbank.detectors import get as get_detector
from testbank.experiment.compare import compare, write_csv
from testbank.experiment.runner import run_candidate
from testbank.viz.inspect import fixed_validation_sample, inspect_samples

#: Cuantas entradas se listan por pantalla antes de resumir. El informe JSON las
#: lleva todas; esto es solo para que la salida quepa en una terminal.
_MAX_LISTED = 20


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
    print(f"Modo: {layout.mode.value}")
    if layout.mode is LayoutMode.ADOPT:
        for split in ("train", "valid", "test"):
            print(f"  {split:6s} {len(layout.groups.get(split, ())):5d} muestras")
    else:
        print(f"  sin particion previa: {len(layout.all_samples)} muestras")
    for note in layout.notes:
        print(f"  aviso: {note}")
    return 0


def cmd_make_splits(args, config: Config) -> int:
    manifest = materialize_splits(
        args.data_root or config.data.root,
        args.splits_dir or config.data.splits_dir,
        groups=_group_config(args, config),
        ratios=config.splits.ratios,
        seed=args.seed if args.seed is not None else config.splits.seed,
        overwrite=args.overwrite,
    )
    print(f"Modo: {manifest['mode']}")
    for split, count in manifest["counts"].items():
        print(f"  {split:6s} {count:5d}")
    print(f"  digest {manifest['digest'][:16]}")
    for note in manifest["notes"]:
        print(f"  aviso: {note}")
    return 0


def cmd_extend_splits(args, config: Config) -> int:
    result = extend_splits(
        args.splits_dir or config.data.splits_dir,
        data_root=args.data_root,
        ratios=config.splits.ratios,
        seed=args.seed,
    )
    print(f"heredadas por grupo: {result.get('inherited', 0)}")
    print(f"nuevas asignadas:    {result['added']}")
    return 0


def cmd_make_folds(args, config: Config) -> int:
    summary = make_folds(
        args.splits_dir or config.data.splits_dir,
        k=args.k or config.splits.folds,
        seed=args.seed if args.seed is not None else config.splits.seed,
        data_root=args.data_root,
        overwrite=args.overwrite,
    )
    print(f"{summary['k']} pliegues sobre {summary['pool_size']} muestras de train+valid")
    print(f"  tamanos: {summary['fold_sizes']}")
    print("  test sigue sellado y fuera de los pliegues")
    return 0


def cmd_check_visibility(args, config: Config) -> int:
    loader = SplitLoader(args.splits_dir or config.data.splits_dir, args.data_root)
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
        print("\nDescartadas por el filtro (siguen en el dataset de origen):")
        for dropped in shown[:_MAX_LISTED]:
            print(f"  - {dropped.describe()}")
        if len(shown) > _MAX_LISTED:
            print(f"  ... y {len(shown) - _MAX_LISTED} mas")
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
        print(f"\nInforme JSON: {args.json_out}")
    # Aviso, no error fatal.
    return 0


def cmd_train(args, config: Config) -> int:
    # Los overrides se sustituyen DENTRO de la config, no se pasan sueltos al
    # detector: la config resuelta es lo que se congela en la ejecucion, y dos
    # runs con epochs o politica de borde distintos no son comparables. Si el
    # override no llegara ahi, el config.yaml mentiria sobre lo que se ejecuto.
    updates = {}
    if args.epochs is not None:
        updates["epochs"] = args.epochs
    if args.out_of_bounds is not None:
        updates["out_of_bounds"] = OutOfBoundsPolicy(args.out_of_bounds)
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
    )
    print(outcome.summary())
    print(f"\nVisualizaciones: {outcome.run.viz_dir}")
    return 0


def cmd_list(args, config: Config) -> int:
    print("Detectores registrados:")
    for name in detectors():
        detector = get_detector(name)
        mark = "" if detector.production_ready else "   NO APTO para produccion"
        print(f"  {name:24s} {detector.license:12s}{mark}")
    print("\nDatasets registrados:")
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
    loader = SplitLoader(args.splits_dir or config.data.splits_dir, args.data_root)
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
    print(f"Muestra fija de '{args.split}' ({len(results)} imagenes, semilla {config.viz.seed})")
    for result in results:
        notes = []
        if result.flagged:
            notes.append(f"{result.flagged} marcadas")
        if result.dropped:
            notes.append(f"{result.dropped} filtradas")
        suffix = f"  <- {', '.join(notes)}" if notes else ""
        print(f"  {result.sample_id}  {result.annotations} anotaciones{suffix}")
    print(f"\nSalida: {output_dir}")
    print(f"Hoja de contacto: {output_dir / '_contact_sheet.png'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="testbank", description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("detect", help="inspecciona la estructura del directorio")

    make = subparsers.add_parser("make-splits", help="materializa la particion")
    make.add_argument("--group-key", choices=[s.value for s in GroupStrategy], default=None)
    make.add_argument("--group-regex", default=None)
    make.add_argument("--group-manifest", default=None)
    make.add_argument(
        "--i-confirm-independence",
        action="store_true",
        help="afirma explicitamente que cada imagen es independiente",
    )
    make.add_argument("--seed", type=int, default=None)
    make.add_argument("--overwrite", action="store_true")

    extend = subparsers.add_parser("extend-splits", help="anexa muestras nuevas")
    extend.add_argument("--seed", type=int, default=None)

    folds = subparsers.add_parser("make-folds", help="pliegues agrupados sobre train+valid")
    folds.add_argument("--k", type=int, default=None)
    folds.add_argument("--seed", type=int, default=None)
    folds.add_argument("--overwrite", action="store_true")

    visibility = subparsers.add_parser(
        "check-visibility", help="consistencia de la politica de visibilidad"
    )
    visibility.add_argument("--splits", nargs="+", default=None)
    visibility.add_argument("--visibility-threshold", type=float, default=None)
    visibility.add_argument(
        "--min-relative-area",
        type=float,
        default=None,
        help=(
            "umbral del filtro de area relativa; 0 lo desactiva y muestra "
            "todas las anotaciones tal cual estan en el fichero"
        ),
    )
    visibility.add_argument("--json-out", default=None)

    train = subparsers.add_parser(
        "train", help="entrena un candidato y deja la ejecucion en runs/"
    )
    train.add_argument("detector", help="nombre registrado; ver `list`")
    train.add_argument("--dataset", default=DEFAULT_DATASET)
    train.add_argument("--name", default=None, help="nombre de la ejecucion")
    train.add_argument(
        "--out-of-bounds",
        choices=[p.value for p in OutOfBoundsPolicy],
        default=None,
        help=(
            "billetes que cruzan el borde: clip los recorta al marco (por "
            "defecto), pad anade borde negro para que quepan enteros (y padea "
            "tambien en inferencia), keep no toca nada y Ultralytics descartara "
            "esas imagenes"
        ),
    )
    train.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="sustituye detector.epochs; queda registrado en la config resuelta",
    )
    train.add_argument(
        "--weights",
        default=None,
        help="reutiliza estos pesos en vez de entrenar (para reevaluar)",
    )

    subparsers.add_parser("list", help="detectores y datasets registrados")

    comparison = subparsers.add_parser(
        "compare", help="tabla comparativa de las ejecuciones de runs/"
    )
    comparison.add_argument("--runs-dir", default=None)
    comparison.add_argument("--csv-out", default=None)

    inspect = subparsers.add_parser("inspect", help="visualizacion de inspeccion")
    inspect.add_argument("--split", default="valid")
    inspect.add_argument("--count", type=int, default=None)
    inspect.add_argument("--output-dir", default=None)

    return parser


_COMMANDS = {
    "detect": cmd_detect,
    "make-splits": cmd_make_splits,
    "extend-splits": cmd_extend_splits,
    "make-folds": cmd_make_folds,
    "check-visibility": cmd_check_visibility,
    "train": cmd_train,
    "list": cmd_list,
    "compare": cmd_compare,
    "inspect": cmd_inspect,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config.load(args.config)
    try:
        return _COMMANDS[args.command](args, config)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
