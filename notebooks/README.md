# Notebooks

Interactive tutorials for OrientedDet.

| Notebook | Where to run | What it does |
|----------|--------------|--------------|
| [`kaggle_fair1m_tutorial.ipynb`](kaggle_fair1m_tutorial.ipynb) | [Kaggle](https://www.kaggle.com/) | FAIR1M: explore → convert → tile → 1-epoch smoke; documents full 1× Faster R-CNN **36.7%** tiled-val mAP |

## Kaggle FAIR1M

1. Create a new notebook (GPU + Internet ON).
2. **Add Input** → [`ollypowell/fair1m-satellite-imagery-for-object-detection`](https://www.kaggle.com/datasets/ollypowell/fair1m-satellite-imagery-for-object-detection).
3. Upload `kaggle_fair1m_tutorial.ipynb` (or copy cells).
4. Run top-to-bottom. Defaults use a small image subset so `/kaggle/working` disk lasts.

Dataset license is **CC BY-NC-SA 3.0 IGO**. OrientedDet supports FAIR1M for local train/metrics only — no Hub weights.

The published dump uses `Dataset/Images/{Train,Val}` (`t_N.jpg` / `v_N.jpg`) with XML under `Notebook_Working/{train,val}_labels/N.xml`. The loader maps those stems automatically.

A full 1× Rotated Faster R-CNN finetune from DOTA Hub (`runs/rotated_faster_rcnn/20260910-072116`) reached **36.70%** mAP50 on tiled val — in band for FAIR1M Faster R-CNN (literature ~31–35%), not a failed train. Do not compare to DOTA 70%+. The notebook smoke will not hit that number. Story: [FAIR1M user guide](../docs/user-guide/data.md#local-1x-faster-rcnn).
