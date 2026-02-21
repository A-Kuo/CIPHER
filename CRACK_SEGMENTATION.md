# YOLO Crack Segmentation — Drone Environmental Detection

This doc describes the **YOLO Crack Segmentation** feature for drone environmental detection (road/wall cracks) and how a future agent can extend it.

---

## What Was Implemented

- **Model**: [OpenSistemas/YOLOv8-crack-seg](https://huggingface.co/OpenSistemas/YOLOv8-crack-seg) (Hugging Face). YOLOv8 trained for crack segmentation; single class `"crack"`.
- **Backend**: New detector `YOLOCrackSegDetector` in `backend/perception.py` for 1-class seg ONNX (output shape `(1, 37, 8400)` = 4 bbox + 1 class + 32 mask coeffs). Same API as `YOLOSegDetector`: `detect(frame)` → `[{"class": "crack", "confidence", "bbox", "center"}, ...]`.
- **Config**: `backend/config.py`:
  - `YOLO_CRACK_SEG_ONNX_PATH = "models/yolov8_crack_seg.onnx"`
  - `YOLO_CRACK_ENABLED = True`
  - `YOLO_CRACK_CONFIDENCE_THRESHOLD = 0.35`
- **Integration**: Crack detections are **merged** with main YOLO detections in:
  - `backend/main.py` (background loop: main YOLO + crack detector; results in `state["raw_detections"]` and drawn on `/live`).
  - `Drone/local_backend/app.py` (phantom_background_loop and _simple_yolo_loop; same merge).
- **Download script**: `scripts/download_model_crack_seg.py` downloads the chosen variant from Hugging Face and exports to ONNX.
  ```bash
  pip install ultralytics huggingface_hub
  python scripts/download_model_crack_seg.py [--variant yolov8n|yolov8s|yolov8m|yolov8l|yolov8x]
  ```
  Output: `models/yolov8_crack_seg.onnx`. Default variant is `yolov8n` (fastest for edge).

---

## How to Enable / Disable (Config and Env — No Code Edit)

- **Enable**: Run the download script so `models/yolov8_crack_seg.onnx` exists. With `YOLO_CRACK_ENABLED = True` (default), backend and Drone app load the crack detector and merge detections.
- **Disable**: Set `YOLO_CRACK_ENABLED=0` in the environment, or set `YOLO_CRACK_ENABLED = False` in `backend/config.py`, or remove/rename the ONNX file.
- **Override path/confidence**: Use env `YOLO_CRACK_SEG_ONNX_PATH`, `YOLO_CRACK_CONFIDENCE_THRESHOLD` (see [VISION_AND_CRACK_INTEGRATION.md](VISION_AND_CRACK_INTEGRATION.md)).

---

## Frontend / API (no changes required)

- Existing APIs already expose detections with `class` and `confidence`: `/api/detections`, `/live_detections`, processed frames with boxes. Crack appears as `class: "crack"`.
- **Optional for a future agent**: In the Drone frontend, give crack a distinct color or label (e.g. “Environmental: crack”) in the tactical view or live overlay.

---

## Possible Next Steps (handoff for future agent)

1. **Mask overlay**: `YOLOCrackSegDetector` does not decode segmentation masks yet (only bbox). The ONNX has a second output (mask proto); decoding it and drawing masks on the frame would match the crack regions more precisely. See `YOLOSegDetector` and Ultralytics seg postprocess for reference.
2. **Separate “environmental” channel**: Expose crack detections on a dedicated API (e.g. `/api/detections/environmental`) or flag in the payload (`source: "crack_seg"`) so the UI can filter or style them.
3. **Other environmental models**: Same pattern (new ONNX path in config, detector class if different output shape, merge or separate API) can be used for other Hugging Face or custom models (e.g. potholes, damage).
4. **Variant selection**: Allow config or env (e.g. `YOLO_CRACK_VARIANT=yolov8s`) so the download script or a small loader chooses n/s/m/l/x without code change.

---

## Combining with Vision and UI (Less Code–Reliant)

See **[VISION_AND_CRACK_INTEGRATION.md](VISION_AND_CRACK_INTEGRATION.md)** for:
- Config- and env-driven pipeline (single merged detection list → same APIs → UI).
- How to toggle crack or add other detectors without editing detection loops.
- Optional UI styling via a `source`/`category` field instead of class names.

**Placeholder shell scripts** (edit vars at top for your environment):
- `scripts/setup_crack_model.sh` — install deps and download/export crack ONNX.
- `scripts/run_vision_with_crack.sh` — run backend with crack enabled via env.
- `scripts/run_vision_with_crack.bat` — same for Windows.

---

## Model Card Reference

- **Hugging Face**: https://huggingface.co/OpenSistemas/YOLOv8-crack-seg  
- **Usage (original)**: `yolo segment predict model=YOLOv8-crack-seg/yolov8n/weights/best.pt` (Ultralytics). We use the same weights, exported to ONNX for the backend/Drone stack.
