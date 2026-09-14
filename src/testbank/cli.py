"""Command-line interface of the inference branch.

Three commands: `list` the runs that can be served, `predict` on image files
from the terminal, `app` for the Streamlit page. Training, splits and
metrics live on `main`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from testbank.config import Config
from testbank.detectors.base import DetectorError


def _runs_dir(args, config: Config) -> Path:
    return Path(args.runs_dir or config.runs_dir)


def cmd_list(args, config: Config) -> int:
    from testbank.serve import discover_models

    models = discover_models(_runs_dir(args, config))
    if not models:
        print(f"no run with weights in {_runs_dir(args, config)}")
        return 1
    for model in models:
        line = f"{model.directory.name}\n    {model.label}"
        if model.split:
            line += f"\n    split {model.split}  image size {model.config.detector.image_size}"
        print(line)
    return 0


def _pick_run(args, config: Config):
    from testbank.serve import discover_models, load_model

    if args.run and Path(args.run).is_dir():
        model = load_model(Path(args.run))
        if model is None:
            raise DetectorError(f"{args.run}: not a run with run.json, config.yaml and weights")
        return model
    models = discover_models(_runs_dir(args, config))
    if not models:
        raise DetectorError(f"no run with weights in {_runs_dir(args, config)}")
    if not args.run:
        return models[0]
    # By folder name or by the name in run.json (`20260914T101500Z_nano_aug`
    # or `nano_aug`); the most recent wins if several share the name.
    for model in models:
        if args.run in (model.directory.name, model.name):
            return model
    raise DetectorError(
        f"no run named {args.run!r} in {_runs_dir(args, config)}; "
        f"available: {[m.name for m in models]}"
    )


def cmd_predict(args, config: Config) -> int:
    """Draws the detections on each image and writes the crops next to it."""
    import cv2

    from testbank.serve import crops, draw, load_weights, predict_image

    model = _pick_run(args, config)
    print(f"run: {model.directory.name}  ({model.detector}, {model.weights.name})")
    # The run's own operating point unless told otherwise, like the app.
    confidence = model.config.metrics.report_confidence if args.confidence is None else args.confidence
    margin = model.config.crop.margin if args.margin is None else args.margin
    loaded = load_weights(model, confidence=confidence, nms_iou=args.nms_iou)
    output_dir = Path(args.output_dir) if args.output_dir else None
    for path in args.images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  {path}: not an image, skipped", file=sys.stderr)
            continue
        predictions = predict_image(
            model, image, confidence=confidence, nms_iou=args.nms_iou, loaded=loaded
        )
        target_dir = output_dir or Path(path).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        cv2.imwrite(str(target_dir / f"{stem}_detections.png"), draw(image, predictions))
        for index, cut in enumerate(crops(image, predictions, margin=margin)):
            cv2.imwrite(str(target_dir / f"{stem}_crop{index + 1}.png"), cut)
        scores = "  ".join(f"{p.score:.2f}" for p in predictions) or "-"
        print(f"  {Path(path).name}: {len(predictions)} banknote(s)  scores {scores}")
    return 0


def cmd_app(args, config: Config) -> int:
    """Starts the Streamlit front end in this interpreter's environment, so
    the runs it can predict with are the ones whose adapter imports here."""
    import importlib.util
    import subprocess

    if importlib.util.find_spec("streamlit") is None:
        print("Streamlit is not installed: `uv pip install -e .[app]`", file=sys.stderr)
        return 2
    from testbank import app

    command = [sys.executable, "-m", "streamlit", "run", app.__file__]
    if args.port:
        command += ["--server.port", str(args.port)]
    command += ["--", "--runs-dir", str(_runs_dir(args, config))]
    return subprocess.call(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="testbank", description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--runs-dir", default=None, help="where the trained runs are (runs/)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="runs that can be served, most recent first")

    predict = subparsers.add_parser("predict", help="detections and crops for image files")
    predict.add_argument("images", nargs="+", type=Path)
    predict.add_argument("--run", default=None, help="run directory or name; default: the most recent")
    predict.add_argument("--confidence", type=float, default=None, help="default: the run's report confidence")
    predict.add_argument("--nms-iou", type=float, default=None, help="default: the run's")
    predict.add_argument("--margin", type=float, default=None, help="crop margin per side; default: the run's")
    predict.add_argument("--output-dir", default=None, help="default: next to each image")

    app = subparsers.add_parser(
        "app", help="Streamlit front end: upload or take photos, pick a run, see crops"
    )
    app.add_argument("--port", type=int, default=None, help="Streamlit's server port")
    return parser


_COMMANDS = {"list": cmd_list, "predict": cmd_predict, "app": cmd_app}


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
