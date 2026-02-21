"""
Crack detection module — YOLOv8 crack segmentation for drone environmental detection.

Model: OpenSistemas/YOLOv8-crack-seg (Hugging Face).
Single-class ONNX detector; same API as backend YOLO detectors (bbox + class + confidence).

Usage:
    from crack_detection import YOLOCrackSegDetector, get_default_config
    cfg = get_default_config()
    det = YOLOCrackSegDetector(cfg["model_path"], confidence_threshold=cfg["confidence_threshold"])
    detections = det.detect(frame)
"""

from crack_detection.detector import CRACK_CLASS_NAME, YOLOCrackSegDetector
from crack_detection.config import get_default_config

__all__ = ["YOLOCrackSegDetector", "CRACK_CLASS_NAME", "get_default_config"]
