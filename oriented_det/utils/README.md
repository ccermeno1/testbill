# `oriented_det.utils`

- **`progress.tqdm_progress_stream`**: file object for `tqdm(..., file=...)` so progress shows on the real TTY when stdout/stderr are piped to a log (same idea as `progress_stream` in `tools/train.py`).
- **`viz`**: `draw_boxes` / `draw_polygons`. Prediction overlays (`odet image-demo`, Gradio viewer, eval visualizations) use outline width **`DEFAULT_LINE_WIDTH`** (4). Ground-truth overlays stay thinner (width 2) when both are drawn.

See the package `__init__.py` for the public exports.
