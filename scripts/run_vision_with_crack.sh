#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Run vision backend with crack detection enabled — PLACEHOLDER
# Edit the variables below; run from repo root. Uses env so no code change needed.
# -----------------------------------------------------------------------------
set -e

# ----- EDIT ME -----
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
PORT="${PORT:-8000}"
# Optional: uncomment to override config via env
# export YOLO_CRACK_ENABLED=1
# export YOLO_CRACK_CONFIDENCE_THRESHOLD=0.35
# export YOLO_CRACK_SEG_ONNX_PATH=models/yolov8_crack_seg.onnx
# -------------------

cd "$REPO_ROOT"

if [ -d "$VENV_PATH" ]; then
  echo "Activating venv: $VENV_PATH"
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate"
fi

export YOLO_CRACK_ENABLED="${YOLO_CRACK_ENABLED:-1}"
echo "Starting backend (crack enabled=$YOLO_CRACK_ENABLED) on port $PORT..."
echo "  Tactical / Live: http://localhost:$PORT"
PHANTOM_HTTP_ONLY=1 python backend/main.py
# Or Drone local_backend: cd Drone/local_backend && uvicorn app:app --host 0.0.0.0 --port "$PORT"
