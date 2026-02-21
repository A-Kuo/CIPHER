"""
Download OpenSistemas/YOLOv8-crack-seg and export to ONNX.

Mirrors Drone/local_backend workflow: run from repo root or from this folder.
Output: models/yolov8_crack_seg.onnx (relative to repo root).

Usage:
    cd CIPHER
    pip install -r crack_detection/requirements.txt
    python -m crack_detection.download_model [--variant yolov8n]
"""

import argparse
import shutil
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PACKAGE_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_ONNX = MODELS_DIR / "yolov8_crack_seg.onnx"
HF_REPO = "OpenSistemas/YOLOv8-crack-seg"
VARIANT_FILES = {
    "yolov8n": "yolov8n/weights/best.pt",
    "yolov8s": "yolov8s/weights/best.pt",
    "yolov8m": "yolov8m/weights/best.pt",
    "yolov8l": "yolov8l/weights/best.pt",
    "yolov8x": "yolov8x/weights/best.pt",
}


def download_and_export(variant: str = "yolov8n") -> bool:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    hf_filename = VARIANT_FILES.get(variant) or VARIANT_FILES["yolov8n"]
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics and huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "huggingface_hub", "-q"])
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

    print(f"Downloading {HF_REPO} ({variant})...")
    pt_path = hf_hub_download(repo_id=HF_REPO, filename=hf_filename, local_dir=None, local_dir_use_symlinks=False)
    pt_path = Path(pt_path)

    print("Exporting to ONNX (1-class crack-seg, 640x640)...")
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", imgsz=640, dynamic=False, half=False)
    if isinstance(exported, (list, tuple)):
        exported = Path(exported[0]) if exported else None
    else:
        exported = Path(exported) if exported else None

    if exported and exported.is_file():
        if exported.resolve() != OUTPUT_ONNX.resolve():
            shutil.copy2(exported, OUTPUT_ONNX)
        if not OUTPUT_ONNX.is_file():
            for name in ("yolov8n-seg.onnx", "best.onnx", f"{variant}-seg.onnx"):
                for base in (REPO_ROOT, Path.cwd()):
                    p = base / name
                    if p.is_file():
                        shutil.copy2(p, OUTPUT_ONNX)
                        return OUTPUT_ONNX.is_file()
        return OUTPUT_ONNX.is_file()
    return False


def main():
    parser = argparse.ArgumentParser(description="Download YOLOv8-crack-seg and export to ONNX")
    parser.add_argument("--variant", default="yolov8n", choices=list(VARIANT_FILES), help="Model size (yolov8n fastest)")
    args = parser.parse_args()

    print("YOLO Crack Segmentation — OpenSistemas/YOLOv8-crack-seg")
    print(f"Output: {OUTPUT_ONNX}\n")

    if download_and_export(variant=args.variant):
        print(f"\nDone. Model saved to {OUTPUT_ONNX}")
        print("Backend/Drone merge crack detections when YOLO_CRACK_ENABLED and file exists.")
        return 0

    print("\nFailed. Ensure: pip install ultralytics huggingface_hub")
    return 1


if __name__ == "__main__":
    sys.exit(main())
