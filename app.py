"""Comparador de detectores de cajas orientadas (OBB) exportados a ONNX.

    uv run streamlit run app.py

Cada carpeta de ``models/`` es un modelo: el ``.onnx`` + su contrato (``model.json`` o el
``.metadata.json`` que escriben los scripts de export de las ramas). Ver README.md.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from obbcompare.config import ModelConfig, discover
from obbcompare.detector import Detections, OnnxDetector, RawOutput, postprocess
from obbcompare.draw import PALETTE, draw_detections, overlay_heatmap, to_rgb
from obbcompare.explain import Explainer

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

MODELS_DIR = Path(__file__).parent / "models"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp", "heic", "heif"]
GRADCAM, EIGENCAM = "Grad-CAM (HiResCAM)", "EigenCAM"

st.set_page_config(page_title="Comparador OBB", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------- cached work
@st.cache_resource(show_spinner=False)
def model_catalog() -> tuple[dict[str, ModelConfig], list[str]]:
    configs, warnings = discover(MODELS_DIR)
    return {c.key: c for c in configs}, warnings


@st.cache_resource(show_spinner="Cargando modelo…")
def load_detector(key: str) -> OnnxDetector:
    return OnnxDetector(model_catalog()[0][key])


@st.cache_resource(show_spinner="Preparando explicabilidad…")
def load_explainer(key: str) -> Explainer:
    return Explainer(load_detector(key))


@st.cache_data(max_entries=32, show_spinner=False)
def run_raw(key: str, img_hash: str, _bgr: np.ndarray) -> RawOutput:
    """Salida cruda de la red; los sliders solo rehacen el post-proceso."""
    return load_detector(key).infer(_bgr)


@st.cache_data(max_entries=64, show_spinner=False)
def run_cam(key: str, img_hash: str, method: str, targets: tuple, _bgr: np.ndarray) -> np.ndarray:
    ex = load_explainer(key)
    if method == EIGENCAM:
        return ex.eigencam(_bgr)
    return ex.gradcam(_bgr, list(targets))


def decode_image(data: bytes) -> np.ndarray:
    """Bytes of a photo -> BGR uint8 with the EXIF orientation applied (as the 3 repos do)."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------- sidebar
catalog, warnings = model_catalog()

with st.sidebar:
    st.header("Modelos")
    if st.button("🔄 Recargar carpeta models/", width="stretch"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    for w in warnings:
        st.warning(w, icon="⚠️")
    if not catalog:
        st.info(f"No hay modelos en `{MODELS_DIR}`. Copia allí la carpeta exportada (ver README).")
        st.stop()
    keys = list(catalog)
    selected = st.multiselect("Modelos a comparar", keys, default=keys,
                              format_func=lambda k: catalog[k].name)

    st.header("Post-proceso")
    score_thr = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01,
                          help="Se descartan las cajas con probabilidad por debajo de este valor.")
    nms_iou = st.slider("NMS IoU threshold", 0.0, 1.0, 0.3, 0.01,
                        help="Dos cajas de la misma clase con IoU rotado por encima de este valor "
                             "se consideran la misma: se queda la de mayor probabilidad.")
    max_det = st.number_input("Máx. detecciones por imagen", 1, 1000, 100)

    st.header("Explicabilidad")
    explain_on = st.toggle("Mostrar mapa de activación", value=False)
    method = st.radio("Método", [GRADCAM, EIGENCAM], disabled=not explain_on,
                      help="Grad-CAM: qué zonas suben la probabilidad de UNA detección (con gradientes). "
                           "EigenCAM: en qué se fija la red en general (sin gradientes, sin objetivo).")
    target_mode = st.radio("Detección a explicar", ["La de mayor probabilidad", "Todas", "Elegir nº"],
                           disabled=not explain_on or method != GRADCAM)
    target_n = st.number_input("Nº de detección", 1, 1000, 1,
                               disabled=not explain_on or method != GRADCAM or target_mode != "Elegir nº")
    alpha = st.slider("Opacidad del mapa", 0.0, 1.0, 0.5, 0.05, disabled=not explain_on)


# ---------------------------------------------------------------------- image
st.title("🔎 Comparador de detectores OBB")
source = st.segmented_control("Fuente", ["📁 Subir foto", "📷 Webcam"], default="📁 Subir foto",
                              required=True, label_visibility="collapsed")
if source == "📷 Webcam":
    file = st.camera_input("Haz una foto")
else:
    file = st.file_uploader("Sube una foto", type=IMAGE_TYPES)

if file is None:
    st.info("Sube una foto o haz una con la webcam para ver las detecciones de cada modelo.")
    st.stop()
if not selected:
    st.warning("Selecciona al menos un modelo en la barra lateral.")
    st.stop()

data = file.getvalue()
img_hash = hashlib.sha1(data).hexdigest()
try:
    bgr = decode_image(data)
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo leer la imagen: {e}")
    st.stop()
st.caption(f"Imagen {bgr.shape[1]}×{bgr.shape[0]} px")


# ---------------------------------------------------------------------- results
def cam_targets(dets: Detections, raw: RawOutput) -> tuple[tuple[tuple[int, int], ...], int | None, str]:
    """(prior, class) pairs to explain, the detection to highlight and a caption."""
    if not len(dets):
        # nothing above the threshold: explain the best prior anyway, it is informative
        best = raw.scores.max(1)
        p = int(best.argmax())
        return ((p, int(raw.scores[p].argmax())),), None, \
            f"Sin detecciones: se explica la predicción de mayor score ({best[p]:.2f})."
    pairs = [(int(p), int(c)) for p, c in zip(dets.prior_idx, dets.labels)]
    if target_mode == "Todas":
        return tuple(pairs), None, f"Suma de las {len(pairs)} detecciones."
    i = 0
    if target_mode == "Elegir nº":
        i = min(int(target_n), len(pairs)) - 1
    return (pairs[i],), i, f"Detección #{i + 1} ({dets.scores[i]:.2f})."


summary = []
n_cols = min(len(selected), 3)
for row_start in range(0, len(selected), n_cols):
    cols = st.columns(n_cols)
    for col, key in zip(cols, selected[row_start:row_start + n_cols]):
        cfg = catalog[key]
        color = PALETTE[keys.index(key) % len(PALETTE)]
        with col, st.container(border=True):
            st.subheader(cfg.name)
            try:
                det = load_detector(key)
                raw = run_raw(key, img_hash, bgr)
                dets = postprocess(raw, score_thr, nms_iou, int(max_det), cfg.nms_pre)
            except Exception as e:  # noqa: BLE001 - one broken model must not hide the others
                st.error(f"{type(e).__name__}: {e}")
                continue
            st.caption(f"{cfg.family} · entrada {raw.input_hw[1]}×{raw.input_hw[0]} {cfg.color} · "
                       f"{raw.ms:.0f} ms · **{len(dets)} detecciones**")
            st.image(to_rgb(draw_detections(bgr, dets, cfg.class_names, color)), width="stretch")

            if explain_on:
                targets, highlight, caption = cam_targets(dets, raw)
                try:
                    ex = load_explainer(key)
                    if method == GRADCAM and not ex.gradcam_available:
                        st.warning(f"Grad-CAM no disponible para este modelo ({ex.gradcam_error}). "
                                   "Se muestra EigenCAM.")
                        used = EIGENCAM
                    else:
                        used = method
                    with st.spinner(f"Calculando {used}…"):
                        cam = run_cam(key, img_hash, used, targets if used == GRADCAM else (), bgr)
                    vis = overlay_heatmap(bgr, cam, alpha)
                    vis = draw_detections(vis, dets, cfg.class_names, color, highlight=highlight)
                    st.image(to_rgb(vis), width="stretch",
                             caption=f"{used}. " + (caption if used == GRADCAM else "Sin objetivo."))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Explicabilidad: {type(e).__name__}: {e}")

            with st.expander("Detecciones"):
                if len(dets):
                    st.dataframe(pd.DataFrame({
                        "#": np.arange(1, len(dets) + 1),
                        "clase": [cfg.class_names[l] if l < len(cfg.class_names) else l for l in dets.labels],
                        "probabilidad": dets.scores.round(4),
                        "cx": dets.boxes[:, 0].round(1), "cy": dets.boxes[:, 1].round(1),
                        "w": dets.boxes[:, 2].round(1), "h": dets.boxes[:, 3].round(1),
                        "ángulo (°)": np.degrees(dets.boxes[:, 4]).round(1),
                    }), hide_index=True)
                else:
                    st.write("Ninguna caja supera el umbral.")
            summary.append({
                "modelo": cfg.name,
                "detecciones": len(dets),
                "prob. máx.": float(dets.scores.max()) if len(dets) else None,
                "prob. media": float(dets.scores.mean()) if len(dets) else None,
                "inferencia (ms)": round(raw.ms, 1),
                "entrada": f"{raw.input_hw[1]}×{raw.input_hw[0]}",
            })

if summary:
    st.subheader("Resumen")
    st.dataframe(pd.DataFrame(summary), hide_index=True,
                 column_config={"prob. máx.": st.column_config.NumberColumn(format="%.3f"),
                                "prob. media": st.column_config.NumberColumn(format="%.3f")})

with st.expander("Contrato de pre/post-proceso de cada modelo"):
    for key in selected:
        c = catalog[key]
        st.markdown(f"**{c.name}** — `{c.onnx_path.relative_to(MODELS_DIR)}` (leído de `{c.source}`)")
        st.json({"entrada": {"lado largo": c.size, "color": c.color, "relleno": c.pad_value,
                             "normalización": "dentro del grafo" if c.mean is None else {"mean": c.mean, "std": c.std}},
                 "clases": c.class_names,
                 "recomendado": {"score_thr": c.score_thr, "nms_iou": c.nms_iou}}, expanded=False)
