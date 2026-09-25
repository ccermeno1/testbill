"""Pure-PyTorch YOLOX-OBB detectors (rotated boxes) for CPU / CUDA / Apple MPS."""
from .model import OBBDetector, YOLOXOBB, PRESETS  # noqa: F401
from .official import YOLOXOfficialOBB  # noqa: F401
from .checkpoint import load_pretrained, load_state_dict_file  # noqa: F401

# name -> (constructor, default initial checkpoint in models/yolox_obb/checkpoints)
MODELS = {
    'ddgrcf_s': (lambda num_classes: YOLOXOBB(num_classes), 'yolox_s_dota1_0.pth'),
    'yolox_nano': (lambda num_classes: YOLOXOfficialOBB(num_classes, 'nano'), 'yolox_nano.pth'),
    'yolox_tiny': (lambda num_classes: YOLOXOfficialOBB(num_classes, 'tiny'), 'yolox_tiny.pth'),
    'yolox_s': (lambda num_classes: YOLOXOfficialOBB(num_classes, 's'), 'yolox_s.pth'),
}


def build_model(arch: str = 'ddgrcf_s', num_classes: int = 1) -> OBBDetector:
    return MODELS[arch][0](num_classes)


def checkpoint_arch(path: str, default: str = 'ddgrcf_s') -> str:
    """Architecture recorded in a checkpoint of this repo (older ones are ``ddgrcf_s``)."""
    import torch
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    meta = ckpt.get('meta', {}) if isinstance(ckpt, dict) else {}
    return meta.get('arch', default)
