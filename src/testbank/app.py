"""Streamlit front end: upload photos or take one with the webcam, pick a
trained run, see the detections and the rectified crops.

Run it with `testbank app` (or `streamlit run src/testbank/app.py`). Only
this module imports Streamlit; everything it computes lives in
`testbank.serve` so it stays testable without a browser.

The defaults of the sliders are the run's own: the confidence at which the
metrics are *read* (`metrics.report_confidence`, 0.25), the NMS the run
trained with, and the crop margin the coverage was scored at. Move them and
you are looking at a different operating point than the table did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import streamlit as st

from testbank.config import Config
from testbank.detectors.base import DetectorError
from testbank.serve import (
    TrainedModel,
    crops,
    decode_image,
    discover_models,
    draw,
    load_weights,
    predict_image,
)


def _runs_dir() -> Path:
    """`streamlit run app.py -- --runs-dir X` passes X after the `--`."""
    argv = sys.argv[1:]
    if "--runs-dir" in argv:
        return Path(argv[argv.index("--runs-dir") + 1])
    return Path(Config.load(None).runs_dir)


@st.cache_resource(show_spinner=False)
def _models(runs_dir: str) -> list[TrainedModel]:
    return discover_models(runs_dir)


@st.cache_resource(show_spinner="Loading the model...", max_entries=3)
def _loaded(directory: str, weights: str, confidence: float, nms: float):
    """One loaded model per (run, thresholds): the sliders reload it, a new
    photo does not. `max_entries` keeps at most three models in memory."""
    model = next(m for m in _models(_runs_key()) if str(m.directory) == directory)
    assert str(model.weights) == weights
    return load_weights(model, confidence=confidence, nms_iou=nms)


def _runs_key() -> str:
    return str(_runs_dir())


def _rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _png(image) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode the crop")
    return buffer.tobytes()


def _sidebar(models: list[TrainedModel]) -> tuple[TrainedModel, float, float, float]:
    st.sidebar.header("Model")
    index = st.sidebar.selectbox(
        "Run",
        range(len(models)),
        format_func=lambda i: models[i].label,
        help="Every run in runs/ with weights, most recent first.",
    )
    model = models[index]
    with st.sidebar.expander("About this run"):
        st.markdown(
            f"**Adapter** `{model.detector}`  \n"
            f"**Weights** `{model.weights.relative_to(model.directory)}`  \n"
            f"**Split** {model.split or 'not recorded'}  \n"
            f"**Image size** {model.config.detector.image_size} px"
        )
        summary = model.summary()
        if summary:
            st.markdown("**On `valid`:** " + ", ".join(f"{k} {v:.3f}" for k, v in summary.items()))
        else:
            st.caption("No metrics.json: this run was not evaluated.")
    st.sidebar.header("Operating point")
    confidence = st.sidebar.slider(
        "Confidence", 0.05, 0.95, float(model.config.metrics.report_confidence), 0.05,
        help="Detections below it are dropped. 0.25 is where the metrics are read.",
    )
    nms = st.sidebar.slider(
        "NMS IoU", 0.1, 0.9, float(model.config.detector.nms_iou), 0.05,
        help="Two boxes overlapping more than this keep only the higher-scored "
        "one. Lower it for isolated notes, raise it for piles.",
    )
    margin = st.sidebar.slider(
        "Crop margin", 0.0, 0.3, float(model.config.crop.margin), 0.01,
        help="Slack added on each border of the crop, as a fraction of the side.",
    )
    return model, confidence, nms, margin


def _show(model, image, name: str, *, confidence, nms, margin) -> None:
    try:
        loaded = _loaded(str(model.directory), str(model.weights), confidence, nms)
        with st.spinner(f"Running {model.detector} on {name}..."):
            predictions = predict_image(
                model, image, confidence=confidence, nms_iou=nms, loaded=loaded
            )
    except DetectorError as exc:
        st.error(
            f"`{model.detector}` cannot run in this environment: {exc}\n\n"
            "Runs trained in a separate environment (RTMDet-R, Rotated FCOS, "
            "PP-YOLOE-R) need the app started from that environment."
        )
        return
    left, right = st.columns([3, 2])
    with left:
        st.image(_rgb(draw(image, predictions)), caption=f"{name}: {len(predictions)} banknote(s)")
    with right:
        if not predictions:
            st.info("No banknote above the confidence threshold.")
        for index, (prediction, cut) in enumerate(zip(predictions, crops(image, predictions, margin=margin))):
            st.image(_rgb(cut), caption=f"#{index + 1}  score {prediction.score:.2f}  {cut.shape[1]}x{cut.shape[0]} px")
            st.download_button(
                "Download crop", _png(cut),
                file_name=f"{Path(name).stem}_{index + 1}.png", mime="image/png",
                key=f"dl-{name}-{index}",
            )


def main() -> None:
    st.set_page_config(page_title="testbank", page_icon="💶", layout="wide")
    st.title("testbank · banknote localization")
    runs_dir = _runs_dir()
    models = _models(_runs_key())
    if not models:
        st.warning(f"No run with weights in `{runs_dir}`. Train one with `testbank train ...`.")
        st.stop()
    model, confidence, nms, margin = _sidebar(models)

    upload_tab, camera_tab = st.tabs(["Upload photos", "Camera"])
    with upload_tab:
        files = st.file_uploader(
            "JPEG or PNG, one or several", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
        )
        for file in files or []:
            _show(model, decode_image(file.getvalue()), file.name,
                  confidence=confidence, nms=nms, margin=margin)
    with camera_tab:
        shot = st.camera_input("Take a photo")
        if shot is not None:
            _show(model, decode_image(shot.getvalue()), "camera.jpg",
                  confidence=confidence, nms=nms, margin=margin)


if __name__ == "__main__" or st.runtime.exists():
    main()
