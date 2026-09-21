#!/usr/bin/env python
"""
Wrapper de PaddleDetection/tools/train.py que registra los operadores de augmentacion OBB
(scripts/obb_aug.py) antes de construir el reader. Mismos argumentos que tools/train.py:

  python scripts/train.py -c configs/ppyoloe_r_crn_s_banknotes_aug.yml --eval -o save_dir=output_aug
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "PaddleDetection"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import obb_aug  # noqa: E402,F401  (registra RMosaic / RRandomAffine; tambien en los workers)

if __name__ == "__main__":
    runpy.run_path(os.path.join(ROOT, "PaddleDetection", "tools", "train.py"), run_name="__main__")
