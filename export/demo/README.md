# export/demo

Bundled **Pleiades Neo** airfield crop (`planes_pleiades_neo.jpg`) for a one-command ONNX + NMS check.

Needs **no oriented-det**. From a checkout that already has `onnx_export/model.onnx`:

```bash
pip install -r export/requirements-runtime.txt
python -m export demo
# or: odet export demo
# after export, the same demo also runs from the copied bundle:
#   python onnx_export/demo.py
# optional exact polygon IoU:
python -m export demo --nms-backend shapely
# or: make export-demo
```

Writes overlay + JSON under `onnx_export/demo/` (gitignored). The command **exits non-zero** unless:

1. Score-filtered pre-NMS boxes include at least one pair with rotated IoU ≥ the production NMS threshold
2. Post-NMS count is strictly smaller (NMS dropped duplicates)
3. Remaining boxes have pairwise IoU below that threshold
