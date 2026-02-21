"""
Crack detection config — paths and thresholds.

Backend and Drone app use backend.config for overrides (YOLO_CRACK_SEG_ONNX_PATH, etc.).
This module provides defaults when running crack_detection standalone.
"""

from pathlib import Path

# Default: repo root is parent of this package
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent

DEFAULT_MODEL_PATH = str(_REPO_ROOT / "models" / "yolov8_crack_seg.onnx")
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_ENABLED = True


def get_default_config() -> dict:
    """Return default config for standalone use (e.g. run_inference.py)."""
    return {
        "model_path": DEFAULT_MODEL_PATH,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "enabled": DEFAULT_ENABLED,
    }
