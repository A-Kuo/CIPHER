"""
Functional tests for crack detection module.

Tests module imports, config, detector logic (preprocessing, parsing, NMS),
and backend integration — all without requiring the real ONNX model file.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))


class TestCrackImports(unittest.TestCase):
    """Verify all public symbols are importable."""

    def test_package_import(self):
        from crack import (
            CRACK_CLASS_NAME,
            CRACK_CONFIDENCE_THRESHOLD,
            CRACK_ENABLED,
            CRACK_ONNX_PATH,
            YOLOCrackSegDetector,
        )
        self.assertEqual(CRACK_CLASS_NAME, "crack")
        self.assertIsInstance(CRACK_CONFIDENCE_THRESHOLD, float)
        self.assertIsInstance(CRACK_ENABLED, bool)
        self.assertIsInstance(CRACK_ONNX_PATH, str)
        self.assertTrue(callable(YOLOCrackSegDetector))

    def test_config_module(self):
        from crack.config import (
            CRACK_CONFIDENCE_THRESHOLD,
            CRACK_ENABLED,
            CRACK_INPUT_SIZE,
            CRACK_NMS_IOU_THRESHOLD,
            CRACK_ONNX_PATH,
            CRACK_VIS_COLOR_BGR,
            HF_REPO,
            VARIANT_FILES,
        )
        self.assertEqual(CRACK_INPUT_SIZE, 640)
        self.assertEqual(HF_REPO, "OpenSistemas/YOLOv8-crack-seg")
        self.assertIn("yolov8n", VARIANT_FILES)
        self.assertEqual(len(CRACK_VIS_COLOR_BGR), 3)

    def test_detector_module(self):
        from crack.detector import CRACK_CLASS_NAME, YOLOCrackSegDetector
        self.assertEqual(CRACK_CLASS_NAME, "crack")


class TestCrackConfig(unittest.TestCase):
    """Verify config defaults and environment overrides."""

    def test_default_threshold(self):
        from crack.config import CRACK_CONFIDENCE_THRESHOLD
        self.assertAlmostEqual(CRACK_CONFIDENCE_THRESHOLD, 0.35)

    def test_default_nms_iou(self):
        from crack.config import CRACK_NMS_IOU_THRESHOLD
        self.assertAlmostEqual(CRACK_NMS_IOU_THRESHOLD, 0.5)

    def test_model_path_is_absolute(self):
        from crack.config import CRACK_ONNX_PATH
        self.assertTrue(Path(CRACK_ONNX_PATH).is_absolute())
        self.assertTrue(CRACK_ONNX_PATH.endswith("yolov8_crack_seg.onnx"))

    def test_enabled_by_default(self):
        from crack.config import CRACK_ENABLED
        self.assertTrue(CRACK_ENABLED)


class TestCrackDetector(unittest.TestCase):
    """Test detector logic with a mocked ONNX session."""

    def _make_detector(self, conf=0.35, nms_iou=0.5):
        """Create a detector with a mocked ONNX session."""
        from crack.detector import YOLOCrackSegDetector

        with patch("crack.detector.YOLOCrackSegDetector._resolve_model_path", return_value="/fake.onnx"):
            with patch("onnxruntime.InferenceSession") as MockSession:
                mock_sess = MagicMock()
                mock_sess.get_providers.return_value = ["CPUExecutionProvider"]
                mock_sess.get_inputs.return_value = [MagicMock(name="images")]
                MockSession.return_value = mock_sess
                det = YOLOCrackSegDetector(
                    "/fake.onnx",
                    confidence_threshold=conf,
                    nms_iou_threshold=nms_iou,
                )
        return det

    def test_instantiation(self):
        det = self._make_detector()
        self.assertEqual(det.get_provider(), "CPUExecutionProvider")

    def test_preprocess_shape(self):
        det = self._make_detector()
        # Simulate a 480x640 BGR frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        blob = det._preprocess(frame)
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertTrue(blob.dtype == np.float32)
        self.assertTrue(blob.max() <= 1.0)

    def test_detect_no_detections(self):
        det = self._make_detector(conf=0.9)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Simulate ONNX output: 1-class seg => (1, 37, 8400), all low confidence
        raw = np.random.uniform(0, 0.1, (1, 37, 8400)).astype(np.float32)
        raw[0, :4, :] = np.random.uniform(0, 640, (4, 8400))  # bbox coords
        det._session.run = MagicMock(return_value=[raw])

        results = det.detect(frame)
        self.assertEqual(results, [])

    def test_detect_single_detection(self):
        det = self._make_detector(conf=0.3)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        # Build a (1, 37, 8400) tensor with one high-confidence anchor
        raw = np.zeros((1, 37, 8400), dtype=np.float32)
        # Anchor 0: center=(320,320), size=(100,50), conf=0.95
        raw[0, 0, 0] = 320.0   # cx
        raw[0, 1, 0] = 320.0   # cy
        raw[0, 2, 0] = 100.0   # w
        raw[0, 3, 0] = 50.0    # h
        raw[0, 4, 0] = 0.95    # class confidence
        # Remaining 32 mask coefficients = 0 (unused)

        det._session.run = MagicMock(return_value=[raw])
        results = det.detect(frame)

        self.assertEqual(len(results), 1)
        d = results[0]
        self.assertEqual(d["class"], "crack")
        self.assertAlmostEqual(d["confidence"], 0.95, places=2)
        self.assertAlmostEqual(d["center"][0], 320.0, places=0)
        self.assertAlmostEqual(d["center"][1], 320.0, places=0)
        # bbox should be [270, 295, 370, 345]
        self.assertAlmostEqual(d["bbox"][0], 270.0, places=0)
        self.assertAlmostEqual(d["bbox"][1], 295.0, places=0)
        self.assertAlmostEqual(d["bbox"][2], 370.0, places=0)
        self.assertAlmostEqual(d["bbox"][3], 345.0, places=0)

    def test_detect_rescales_to_original(self):
        det = self._make_detector(conf=0.3)
        # Original frame is 1080x1920 (different from model's 640x640)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        raw = np.zeros((1, 37, 8400), dtype=np.float32)
        raw[0, 0, 0] = 320.0   # cx (in model coords)
        raw[0, 1, 0] = 320.0   # cy
        raw[0, 2, 0] = 100.0   # w
        raw[0, 3, 0] = 50.0    # h
        raw[0, 4, 0] = 0.8     # conf

        det._session.run = MagicMock(return_value=[raw])
        results = det.detect(frame)

        self.assertEqual(len(results), 1)
        d = results[0]
        # cx should scale: 320 * (1920/640) = 960
        self.assertAlmostEqual(d["center"][0], 960.0, places=0)
        # cy should scale: 320 * (1080/640) = 540
        self.assertAlmostEqual(d["center"][1], 540.0, places=0)

    def test_nms_suppresses_overlapping(self):
        det = self._make_detector(conf=0.3, nms_iou=0.4)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        # Two overlapping anchors at nearly the same spot
        raw = np.zeros((1, 37, 8400), dtype=np.float32)
        for i in range(2):
            raw[0, 0, i] = 320.0 + i * 5   # slightly offset
            raw[0, 1, i] = 320.0
            raw[0, 2, i] = 100.0
            raw[0, 3, i] = 50.0
            raw[0, 4, i] = 0.9 - i * 0.1   # 0.9, 0.8

        det._session.run = MagicMock(return_value=[raw])
        results = det.detect(frame)

        # NMS should suppress the lower-confidence duplicate
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["confidence"], 0.9, places=2)

    def test_nms_keeps_non_overlapping(self):
        det = self._make_detector(conf=0.3, nms_iou=0.4)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        # Two anchors far apart
        raw = np.zeros((1, 37, 8400), dtype=np.float32)
        # Anchor 0: top-left
        raw[0, 0, 0] = 100.0
        raw[0, 1, 0] = 100.0
        raw[0, 2, 0] = 50.0
        raw[0, 3, 0] = 50.0
        raw[0, 4, 0] = 0.8
        # Anchor 1: bottom-right
        raw[0, 0, 1] = 500.0
        raw[0, 1, 1] = 500.0
        raw[0, 2, 1] = 50.0
        raw[0, 3, 1] = 50.0
        raw[0, 4, 1] = 0.7

        det._session.run = MagicMock(return_value=[raw])
        results = det.detect(frame)

        self.assertEqual(len(results), 2)

    def test_latency_tracking(self):
        det = self._make_detector()
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        raw = np.zeros((1, 37, 8400), dtype=np.float32)
        det._session.run = MagicMock(return_value=[raw])
        det.detect(frame)
        self.assertGreater(det.get_last_latency(), 0.0)

    def test_detection_dict_format(self):
        det = self._make_detector(conf=0.3)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        raw = np.zeros((1, 37, 8400), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 100.0
        raw[0, 3, 0] = 50.0
        raw[0, 4, 0] = 0.95

        det._session.run = MagicMock(return_value=[raw])
        results = det.detect(frame)

        d = results[0]
        # Must have all required keys for backend compatibility
        self.assertIn("class", d)
        self.assertIn("confidence", d)
        self.assertIn("bbox", d)
        self.assertIn("center", d)
        # Types
        self.assertIsInstance(d["class"], str)
        self.assertIsInstance(d["confidence"], float)
        self.assertIsInstance(d["bbox"], list)
        self.assertEqual(len(d["bbox"]), 4)
        self.assertIsInstance(d["center"], list)
        self.assertEqual(len(d["center"]), 2)


class TestBackendIntegration(unittest.TestCase):
    """Verify the backend can import crack module correctly."""

    def test_backend_config_crack_settings(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from config import (
            YOLO_CRACK_CONFIDENCE_THRESHOLD,
            YOLO_CRACK_ENABLED,
            YOLO_CRACK_SEG_ONNX_PATH,
        )
        self.assertTrue(YOLO_CRACK_ENABLED)
        self.assertEqual(YOLO_CRACK_SEG_ONNX_PATH, "models/yolov8_crack_seg.onnx")
        self.assertAlmostEqual(YOLO_CRACK_CONFIDENCE_THRESHOLD, 0.35)

    def test_crack_detector_importable_from_app(self):
        """Verify the import path used in app.py works."""
        from crack.detector import YOLOCrackSegDetector
        self.assertTrue(callable(YOLOCrackSegDetector))

    def test_crack_vis_color_matches_app(self):
        """The app hardcodes (0, 80, 255) for crack — must match config."""
        from crack.config import CRACK_VIS_COLOR_BGR
        self.assertEqual(CRACK_VIS_COLOR_BGR, (0, 80, 255))


class TestFileNotFound(unittest.TestCase):
    """Verify proper error when model file doesn't exist."""

    def test_raises_on_missing_model(self):
        from crack.detector import YOLOCrackSegDetector
        with self.assertRaises(FileNotFoundError):
            YOLOCrackSegDetector("/nonexistent/model.onnx")


if __name__ == "__main__":
    unittest.main(verbosity=2)
