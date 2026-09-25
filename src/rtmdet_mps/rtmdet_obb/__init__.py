"""Pure-PyTorch RTMDet-R (rotated boxes) for CPU / CUDA / Apple MPS."""
from .model import RTMDetR, PRESETS  # noqa: F401
from .checkpoint import load_pretrained, load_state_dict_file  # noqa: F401
