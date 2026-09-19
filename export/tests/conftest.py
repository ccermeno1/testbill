# Force CPU for ORT tests unless a test explicitly selects CUDA.
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ORIENTED_DET_ORT_DEVICE", "cpu")
