# weights/

Foreign checkpoints, downloaded by hand. Versioned through **Git LFS**
(`.gitattributes`): `git clone` brings them along if `git lfs` is installed;
without it you get 130-byte pointers, and `git lfs pull` fetches the real files.

| File | Where from | What for |
|---|---|---|
| `yolox_s.pth`, `yolox_tiny.pth`, `yolox_nano.pth` | Megvii/YOLOX releases on GitHub (Apache-2.0) | `testbank train yolox-obb-<small|tiny|nano> --pretrained weights/yolox_<s|tiny|nano>.pth` |
| `yolox_s_dota1_0.pth` (or whatever the DOTA one is called) | MODEL_ZOO of DDGRCF/YOLOX_OBB (Baidu Pan) | `testbank train yolox-obb-ddgrcf-port --pretrained weights/<file>` |
| `ppyoloe_r_crn_s_3x_dota.pdparams` | PaddleDetection model zoo (direct download, Apache-2.0) | `testbank train ppyoloe-r-s` (default) in `.venv-paddle` |

The variant has to match the checkpoint: loading `yolox_s` into `nano` is an
error, not a warning.
