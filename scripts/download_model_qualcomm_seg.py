"""
Download YOLOv8 segmentation from Qualcomm AI Hub and save as models/yolov8_seg.onnx.

Uses Qualcomm AI Hub Models (qai_hub_models) to export the v8 segmentation model
to ONNX for use with the Drone/PHANTOM backend (NPU or CPU).

Install first (WSL/Linux or Windows, in your .venv):
    pip install "qai_hub_models[yolov8_seg]"

Optional: sign in at https://app.aihub.qualcomm.com/ and set API token for cloud compile:
    python -m qai_hub configure --api_token YOUR_TOKEN

If Qualcomm AI Hub is not available, falls back to Ultralytics YOLOv8n-seg export.

Paste-able export command (run from repo root with venv activated):
    python -m qai_hub_models.models.yolov8_seg.export --target-runtime onnx --device "Samsung Galaxy S25 (Family)" --skip-profiling --skip-inferencing --skip-summary --output-dir models/qualcomm_export_seg
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_ONNX = MODELS_DIR / "yolov8_seg.onnx"
QUALCOMM_OUT_DIR = MODELS_DIR / "qualcomm_export_seg"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_qualcomm_export() -> bool:
    """Run Qualcomm AI Hub YOLOv8 segmentation export to ONNX. Returns True if ONNX was produced."""
    try:
        from qai_hub_models.models.yolov8_seg import Model as YOLOv8SegModel
        from qai_hub_models import TargetRuntime
    except ImportError:
        print("Installing qai_hub_models with YOLOv8-seg support...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "qai-hub-models", "qai-hub-models[yolov8_seg]", "-q"
        ], cwd=str(PROJECT_ROOT))
        from qai_hub_models.models.yolov8_seg import Model as YOLOv8SegModel
        from qai_hub_models import TargetRuntime

    QUALCOMM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device: default from Qualcomm export; override with env for different targets
    device = os.environ.get("QAI_HUB_DEVICE", "Samsung Galaxy S25 (Family)")

    print("Exporting YOLOv8 segmentation from Qualcomm AI Hub...")
    print("(First run may download assets from Hugging Face / Qualcomm)")

    cmd = [
        sys.executable, "-m", "qai_hub_models.models.yolov8_seg.export",
        "--target-runtime", "onnx",
        "--device", device,
        "--skip-profiling",
        "--skip-inferencing",
        "--skip-summary",
        "--output-dir", str(QUALCOMM_OUT_DIR),
    ]
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, timeout=600)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Qualcomm export failed: {e}")
        return False

    # Find the exported ONNX file
    onnx_files = list(QUALCOMM_OUT_DIR.rglob("*.onnx"))
    if not onnx_files:
        export_assets = PROJECT_ROOT / "export_assets"
        if export_assets.exists():
            onnx_files = list(export_assets.rglob("*.onnx"))
    if not onnx_files:
        onnx_files = list(PROJECT_ROOT.rglob("*.onnx"))
        onnx_files = [p for p in onnx_files if "yolov8" in p.name.lower() and "seg" in p.name.lower()]
    if onnx_files:
        src = onnx_files[0]
        shutil.copy2(src, OUTPUT_ONNX)
        print(f"Copied {src.name} -> {OUTPUT_ONNX}")
        return True
    return False


def run_ultralytics_fallback() -> bool:
    """Fallback: Ultralytics YOLOv8n-seg export (CPU-friendly ONNX)."""
    print("Falling back to Ultralytics YOLOv8n-seg export...")
    try:
        from ultralytics import YOLO
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
        from ultralytics import YOLO

    model = YOLO("yolov8n-seg.pt")
    exported = model.export(format="onnx", imgsz=640, dynamic=False, half=False)
    if isinstance(exported, (list, tuple)):
        exported = Path(exported[0]) if exported else None
    else:
        exported = Path(exported) if exported else None
    if exported and exported.is_file():
        if exported.resolve() != OUTPUT_ONNX.resolve():
            shutil.copy2(exported, OUTPUT_ONNX)
        return OUTPUT_ONNX.is_file()
    # Often writes to cwd
    for name in ("yolov8n-seg.onnx",):
        p = PROJECT_ROOT / name
        if p.is_file():
            shutil.copy2(p, OUTPUT_ONNX)
            return True
    return False


def main():
    print("YOLOv8 segmentation — Qualcomm AI Hub export")
    print(f"Output: {OUTPUT_ONNX}\n")

    if run_qualcomm_export():
        print(f"\nDone. Model saved to {OUTPUT_ONNX}")
        return 0

    if run_ultralytics_fallback():
        print(f"\nDone (Ultralytics fallback). Model saved to {OUTPUT_ONNX}")
        return 0

    print("\nAll methods failed. Install: pip install 'qai-hub-models[yolov8_seg]' or pip install ultralytics")
    return 1


if __name__ == "__main__":
    sys.exit(main())
