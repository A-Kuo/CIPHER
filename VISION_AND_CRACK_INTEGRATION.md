# Combining Crack Detection with Vision and UI (Config-First)

This doc describes **how crack detection is combined with other vision detection and the UI** in a **less code-reliant** way: driven by config and env, so toggling or adding models does not require editing detection loops.

---

## Pipeline Overview

```
Config (config.py + env) → Which detectors load (main YOLO, optional crack)
        → Each frame → run each enabled detector → merge detections (same list)
        → state["raw_detections"] / state["detections"] (mapped)
        → Same APIs: /api/detections, /live_detections, processed_frames (boxes drawn)
        → UI: tactical map, live view, advisory (no special-case code for "crack")
```

- **Single pipeline**: One background loop grabs frames, runs whatever detectors are enabled (main YOLO + optional crack), **merges** results into one list. No separate “crack pipeline” in the UI.
- **Same API shape**: Every detection has `class`, `confidence`, `bbox`, `center`. Crack appears as `class: "crack"`. The UI already renders any class; no frontend change required.
- **Toggle without code**: Use **env** (and optionally config) to enable/disable crack or point to a different model.

---

## Config and Env (No Code Edit to Toggle)

| Purpose | Config (`backend/config.py`) | Env override |
|--------|-----------------------------|--------------|
| Enable/disable crack | `YOLO_CRACK_ENABLED = True` | `YOLO_CRACK_ENABLED=0` to disable, `1` or unset to enable |
| Crack model path | `YOLO_CRACK_SEG_ONNX_PATH = "models/yolov8_crack_seg.onnx"` | `YOLO_CRACK_SEG_ONNX_PATH=/path/to/model.onnx` |
| Crack confidence | `YOLO_CRACK_CONFIDENCE_THRESHOLD = 0.35` | `YOLO_CRACK_CONFIDENCE_THRESHOLD=0.4` |

**Backend and Drone app** both read these (Drone uses `backend.config` when the Drone2 stack is loaded). Change env or config; no need to touch `main.py` or `app.py` for simple on/off or path changes.

---

## Where Combination Happens (For Reference)

- **Backend**: `backend/main.py` — background loop runs `yolo_detector.detect(frame)`, then if `yolo_crack_detector` is not None, `yolo_crack_detector.detect(frame)`; results are concatenated. Same `state["raw_detections"]` and `state["detections"]` feed `/api/detections`, `/live_detections`, and the processed frames (boxes drawn).
- **Drone local_backend**: `Drone/local_backend/app.py` — same idea in `phantom_background_loop` and `_simple_yolo_loop`: main detections + crack detections merged, then mapped and written to `phantom_state`; UI reads from the same endpoints/state.

To **add another detector** (e.g. potholes) in a config-first way: add a config flag and path, load the detector at startup if enabled, and in the same loop append its `detect(frame)` output to the list. No UI change if the new model emits the same `{class, confidence, bbox, center}` shape.

---

## UI Side (No Crack-Specific Code Required)

- **Tactical map**: Uses `state["detections"]` (or Drone equivalent); each item has `class`, `confidence`, `bbox`/map coords. Crack shows up like any other class.
- **Live stream**: Processed frames already draw a box and label for every detection; crack gets `"crack"` as label.
- **Optional (future)**: To style “environmental” (e.g. crack) differently (e.g. color, icon), the backend can add a field like `"source": "environmental"` or `"category": "crack"` to each detection. The UI can then branch on that field instead of on the string `"crack"`, keeping class names out of the frontend.

---

## Shell Scripts (Placeholders)

- **`scripts/setup_crack_model.sh`** — Install deps and download/export crack ONNX. Edit the variables at the top (venv path, variant, output dir) to match your setup.
- **`scripts/run_vision_with_crack.sh`** — Run the main backend (or Drone app) with crack enabled via env. Edit port, venv, and optional env vars at the top.

Run from repo root. Scripts are intended to be copied or edited for your environment.
