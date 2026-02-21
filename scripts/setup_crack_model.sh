#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Setup crack detection model (OpenSistemas/YOLOv8-crack-seg) — PLACEHOLDER
# Edit the variables below for your environment; run from repo root.
# -----------------------------------------------------------------------------
set -e

# ----- EDIT ME -----
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
CRACK_VARIANT="${CRACK_VARIANT:-yolov8n}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
# -------------------

cd "$REPO_ROOT"

if [ -d "$VENV_PATH" ]; then
  echo "Activating venv: $VENV_PATH"
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate"
fi

echo "Installing deps (ultralytics, huggingface_hub)..."
pip install -q ultralytics huggingface_hub

echo "Downloading and exporting crack-seg model (variant=$CRACK_VARIANT) to $MODELS_DIR..."
mkdir -p "$MODELS_DIR"
python scripts/download_model_crack_seg.py --variant "$CRACK_VARIANT"

echo "Done. Crack ONNX: $MODELS_DIR/yolov8_crack_seg.onnx"
echo "Enable in app via config or: export YOLO_CRACK_ENABLED=1"
