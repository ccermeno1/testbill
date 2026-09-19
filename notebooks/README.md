# Notebooks

Interactive tutorials for OrientedDet.

| Notebook | Where to run | What it does |
|----------|--------------|--------------|
| [`kaggle_fair1m_tutorial.ipynb`](kaggle_fair1m_tutorial.ipynb) | [Kaggle](https://www.kaggle.com/) | FAIR1M: explore → convert → tile → 1-epoch smoke; documents full 1× Faster R-CNN **36.7%** tiled-val mAP |
| [`ssdd_finetune_tutorial.ipynb`](ssdd_finetune_tutorial.ipynb) | Local / Colab | SSDD SAR ships: discover dump → optional DOTA export → 1-epoch smoke → **12-epoch 1×** Faster R-CNN |

## Kaggle FAIR1M

1. Create a new notebook (GPU + Internet ON).
2. **Add Input** → [`ollypowell/fair1m-satellite-imagery-for-object-detection`](https://www.kaggle.com/datasets/ollypowell/fair1m-satellite-imagery-for-object-detection).
3. Upload `kaggle_fair1m_tutorial.ipynb` (or copy cells).
4. Run top-to-bottom. Defaults use a small image subset so `/kaggle/working` disk lasts.

Dataset license is **CC BY-NC-SA 3.0 IGO**. OrientedDet supports FAIR1M for local train/metrics only — no Hub weights.

The published dump uses `Dataset/Images/{Train,Val}` (`t_N.jpg` / `v_N.jpg`) with XML under `Notebook_Working/{train,val}_labels/N.xml`. The loader maps those stems automatically.

A full 1× Rotated Faster R-CNN finetune from DOTA Hub (`runs/rotated_faster_rcnn/20260910-072116`) reached **36.70%** mAP50 on tiled val — in band for FAIR1M Faster R-CNN (literature ~31–35%), not a failed train. Do not compare to DOTA 70%+. The notebook smoke will not hit that number. Story: [FAIR1M user guide](../docs/user-guide/data.md#local-1x-faster-rcnn).

## SSDD finetune

1. Download SSDD from the [Official-SSDD](https://github.com/TianwenZhang/Official-SSDD) Google Drive (this repo does not vendor chips).
2. Set `DATA_ROOT` / `SSDD_DATA_ROOT` to `/path/to/data/Official-SSDD-OPEN` (the loader walks into `RBox_SSDD/voc_style`). COCO or DOTA layouts also work.
3. Run [`ssdd_finetune_tutorial.ipynb`](ssdd_finetune_tutorial.ipynb) top-to-bottom. Set `RUN_SMOKE = True` for a 1-epoch subset; set `RUN_FULL = True` (or run `odet train --config configs/rotated_faster_rcnn/ssdd_le90_1x.json`) for the **12-epoch** recipe.

Native `dataset.format: ssdd` does not need tiling. Full 1× Faster R-CNN: [`configs/rotated_faster_rcnn/ssdd_le90_1x.json`](../configs/rotated_faster_rcnn/ssdd_le90_1x.json). Held-out test mAP50 is **90.34%** (`runs/rotated_faster_rcnn/20260918-130546`; literature Faster R-CNN ~89%). HRSID is the larger SAR **benchmark** — documented in [Data guide — HRSID](../docs/user-guide/data.md#hrsid), not this notebook. See also [Data guide — SSDD](../docs/user-guide/data.md#ssdd).
