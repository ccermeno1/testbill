# testbank — inference and app

Banknote localization by oriented bounding boxes, **serving side**. This
branch takes the runs trained on `main` (`runs/<name>/` with the frozen
config and the weights), predicts on loose photos, draws the rotated boxes
with their confidence and cuts each banknote out, rectified. Nothing here
trains, splits or scores: that all lives on `main`, and so does the long
README that explains the metrics, the data and the candidates.

Pure Python, Apache-only, nothing compiled. Runs on CPU, on Apple's GPU
(MPS) and on CUDA without changing anything.

## What a served run is

```
runs/20260914T101500Z_nano_aug/
  run.json        # who trained it: adapter name, split, dataset, notes
  config.yaml     # the frozen config: image_size, NMS, thresholds, crop margin
  metrics.json    # what it scored on valid (optional; shown in the selector)
  weights/best.pt # the checkpoint (Git LFS)
```

`run.json["detector"]["name"]` picks the adapter; the registered names are
the ones `main` trains under and must not change:

| Adapter | Weights | Environment |
|---|---|---|
| `yolox-obb-nano` (857k), `-tiny` (4.37M), `-small` (7.75M) | `best.pt` / `last.pt` | main (`torch`) |
| `yolox-obb-ddgrcf-port` (8.05M) | `best.pt` (`arch: ddgrcf`) | main (`torch`) |
| `rotated-fcos-r50` (36.2M), `rotated-fcos-r18` (19.7M) | `best.pt` (`arch: rotated_fcos`) | separate: `oriented-det` pins `numpy<2` |

`best.pt` is preferred, then `last.pt`, then the highest-numbered epoch.
The checkpoint carries its own architecture (variant, head, backbone), so
loading nano weights into a tiny cannot happen silently.

## Environment

One venv for the own candidates and the app:

```bash
uv venv --python 3.11
uv pip install -e ".[torch,app,dev]"      # base + torch (CPU) + streamlit + pytest
pytest -q                                 # 86 tests, no run needed
```

The `torch` it installs is CPU. On a Mac with M-series the adapters pick
MPS on their own (`TESTBANK_DEVICE=cpu` forces the CPU for an exact
number); on a PC with NVIDIA install the CUDA torch from
<https://pytorch.org/get-started/locally/> in this same environment.

To serve **Rotated FCOS** runs, a second venv, because `oriented-det` pins
`numpy<2` and the main one does not:

```bash
uv venv .venv-orienteddet --python 3.11
VIRTUAL_ENV=.venv-orienteddet uv pip install --only-binary=:all: oriented-det
VIRTUAL_ENV=.venv-orienteddet uv pip install -e ".[app,dev]" "numpy<2"
.venv-orienteddet/bin/testbank app       # Windows: .venv-orienteddet/Scripts/testbank
```

`--only-binary` because a transitive dependency (`stringzilla`) has no wheel
for every build and wants a C compiler otherwise. **Repeat `"numpy<2"` in
every later `uv pip install` in that venv**: the resolver upgrades it
silently and torch then fails with `RuntimeError: Numpy is not available`.
A run whose adapter is not installed in the current venv still lists in the
app and fails to predict with a message saying so, instead of crashing.

## Use

```bash
testbank list                                   # runs that can be served, most recent first
testbank predict photo.jpg other.jpg            # most recent run; writes <name>_detections.png and <name>_cropN.png next to each
testbank predict photo.jpg --run nano_aug --confidence 0.5 --margin 0.1 --output-dir out/
testbank app                                    # http://localhost:8501
testbank --runs-dir /elsewhere app --port 8600
```

The app has two tabs, **Upload photos** (one or several JPEG/PNG) and
**Camera** (the webcam), a run selector labelled with each run's `valid`
numbers, and three sliders that start at the run's own operating point:

* **Confidence** at `metrics.report_confidence` (0.25): where `main` reads
  precision and recall. Below it, detections are dropped.
* **NMS IoU**: the one the run trained with. Lower for isolated notes,
  higher for piles.
* **Crop margin** at `crop.margin` (0.05): slack per border, as a fraction
  of the side, exactly the expansion the coverage metric scores.

Per photo: the image with every detection outlined in amber, numbered and
tagged with its score (a dot marks the canonical anchor vertex), and on
the right each banknote rectified by homography onto an upright rectangle
with the quad's own side lengths, downloadable as PNG. The loaded model is
kept per (run, confidence, NMS), three at most, so only the first photo or
a slider move pays the load.

## Adding a run

On the machine that trained it (`main`), copy the run folder into `runs/`
here without `_train/` and `viz/` (they are ignored anyway), then:

```bash
git lfs install                                  # once per machine
git add runs/<name>
git commit -m "Add run <name>"
```

`.gitattributes` routes `runs/**/weights/*` through Git LFS: the repository
keeps a pointer and the binary goes apart. Cloning needs `git lfs install`
before `git clone` (or `git lfs pull` after) or the weights come down as
pointer files and `load` fails on them.

## How it computes

* **Same door as the metrics.** `serve.predict_image` calls the adapter's
  own `predict`, with the run's frozen config (same `image_size`, same
  input conversion, same NMS): what the app shows is what `main`'s table
  measured, not a second implementation. The image goes through a temporary
  file and a temporary size cache; nothing is written anywhere else.
* **Input convention.** Raw BGR in 0–255, not normalized, resized to the
  square `image_size` without letterbox (`models/load.image_to_input`). It
  is what the weights were trained with; the Rotated FCOS adapter converts
  to RGB ImageNet-normalized inside, as its model expects.
* **Quads.** Every prediction is a `Quad` in normalized coordinates in
  canonical order: the first side is the longest, vertices clockwise, so the
  crop comes out landscape with the anchor top-left. The adapter
  re-canonicalizes with the image's real aspect, because the network worked
  on a square. A box predicted more than half a frame outside the image has
  no geometry (`quad=None`): `main` counts it as a false positive, here it
  is skipped.
* **Crops.** The four corners are scaled about their center by
  `1 + 2·margin` (the metric's `expand`) and mapped by perspective
  transform onto a `w × h` rectangle where `w`, `h` are the quad's side
  lengths. Parts outside the image come out black rather than clipped, so
  the proportions of the banknote are kept.

Code: `src/testbank/serve.py` (everything the app computes, tested without a
browser in `tests/test_serve.py`), `src/testbank/app.py` (the Streamlit
page; the only module that imports Streamlit, and `tests/test_detectors.py`
checks it), `src/testbank/cli.py`, the adapters in `src/testbank/detectors/`
and the networks in `src/testbank/models/`.

## Licenses

No AGPL or GPL code. The own head, its DDGRCF port and `oriented-det` are
Apache-2.0; Streamlit is Apache-2.0. Ultralytics (AGPL) and the mmrotate and
PaddleDetection references that `main` compares against are not in this
branch. The training data (`annotated-banknotes-2`, CC BY 4.0) requires
attribution wherever a model trained on it is distributed; `run.json`
carries the dataset provenance for that.
