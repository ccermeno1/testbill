"""Pytest setup for the ``src/yolox_mps`` implementation."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "yolox_mps"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
