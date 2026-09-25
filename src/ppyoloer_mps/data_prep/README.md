# Data preparation

Paddle-free scripts that turn Roboflow YOLOv8-OBB exports into the COCO-with-polygons format
this repo reads.

| script | what it does |
|---|---|
| `prepare_dataset.py` | main export plus the frozen `v1` splits -> `data/banknotes_obb` |
| `convert_yolo_obb.py` | any YOLOv8-OBB export -> COCO dataset (merge or drop classes, skip duplicates, filter close-ups, reads HEIC) |
| `find_duplicates.py` | duplicates and near-duplicates across datasets, by perceptual hash |
| `merge_extra.py` | append a converted dataset to the main train split |
| `add_augmented.py` | fold offline augmented copies into train |
| `extra_manifest.py` | manifest of which photos went in and why the rest were left out |
| `export_yolo_obb.py` | COCO dataset -> YOLOv8-OBB, ready to upload to Roboflow |
| `visualize_obb.py` | draw ground truth and predictions, with contact sheets |

The manifests and the generated annotations already sit in `data_manifests/`, so there is no
need to recompute hashes or filters: rebuild the images and copy those JSONs across.

These need `opencv-python-headless`, `pillow`, `pillow-heif`, `numpy`, `shapely` and
`imagehash` (the last one only for `find_duplicates.py`).
