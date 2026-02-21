# Qualcomm AI Hub — Export commands (YOLOv8 det & seg)

Run all commands from **repo root** with your **Python 3.12.0** venv activated.  
Use WSL or Linux when possible; on ARM you can run the export (cloud compile may run on Qualcomm devices).  
No `sudo` required.

---

## One-time setup

```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install "qai_hub_models[yolov8]"
pip install "qai_hub_models[yolov8_seg]"
# Optional: API token for cloud compile/profiling
python -m qai_hub configure --api_token YOUR_TOKEN
```

---

## YOLOv8 Detection (yolov8_det.onnx)

**Script (recommended):**
```bash
python scripts/download_model_qualcomm.py
```
Output: `models/yolov8_det.onnx`

**Raw export (paste into terminal):**
```bash
python -m qai_hub_models.models.yolov8_det.export --target-runtime onnx --device "Samsung Galaxy S25 (Family)" --skip-profiling --skip-inferencing --skip-summary --output-dir models/qualcomm_export
```
Then copy the produced `.onnx` from `models/qualcomm_export/` (or `export_assets/`) to `models/yolov8_det.onnx`.

**Other runtimes (QNN / TFLite):**
```bash
# QNN context binary (on-device)
python -m qai_hub_models.models.yolov8_det.export --target-runtime qnn_context_binary --device "Samsung Galaxy S25 (Family)" --skip-profiling --skip-inferencing --skip-summary --output-dir models/qualcomm_export

# TFLite
python -m qai_hub_models.models.yolov8_det.export --target-runtime tflite --device "Samsung Galaxy S25 (Family)" --skip-profiling --skip-inferencing --skip-summary --output-dir models/qualcomm_export
```

---

## YOLOv8 Segmentation (yolov8_seg.onnx)

**Script (recommended):**
```bash
python scripts/download_model_qualcomm_seg.py
```
Output: `models/yolov8_seg.onnx`

**Raw export (paste into terminal):**
```bash
python -m qai_hub_models.models.yolov8_seg.export --target-runtime onnx --device "Samsung Galaxy S25 (Family)" --skip-profiling --skip-inferencing --skip-summary --output-dir models/qualcomm_export_seg
```
Then copy the produced `.onnx` to `models/yolov8_seg.onnx`.

**Other runtimes:**
```bash
# QNN
python -m qai_hub_models.models.yolov8_seg.export --target-runtime qnn_context_binary --device "Samsung Galaxy S25 (Family)" --skip-profiling --skip-inferencing --skip-summary --output-dir models/qualcomm_export_seg

# TFLite
python -m qai_hub_models.models.yolov8_seg.export --target-runtime tflite --device "Samsung Galaxy S25 (Family)" --skip-profiling --skip-inferencing --skip-summary --output-dir models/qualcomm_export_seg
```

**Different device (e.g. Snapdragon X Elite CRD):**
```bash
export QAI_HUB_DEVICE="Snapdragon X Elite CRD"
python -m qai_hub_models.models.yolov8_seg.export --target-runtime onnx --device "$QAI_HUB_DEVICE" --skip-profiling --skip-inferencing --skip-summary --output-dir models/qualcomm_export_seg
```
Windows PowerShell:
```powershell
$env:QAI_HUB_DEVICE = "Snapdragon X Elite CRD"
python -m qai_hub_models.models.yolov8_seg.export --target-runtime onnx --device $env:QAI_HUB_DEVICE --skip-profiling --skip-inferencing --skip-summary --output-dir models/qualcomm_export_seg
```

---

## Behaviour in the app

- If `models/yolov8_seg.onnx` exists, the backend and Drone app use **YOLOSegDetector** (same bbox/class API as detection; mask decoding can be added later).
- If only `models/yolov8_det.onnx` exists, they use **YOLODetector** as before.
- Paths are under `models/` and relative to repo root; no Windows-specific paths required for the model files.

---

## ARM / no Qualcomm device

On ARM (e.g. WSL on ARM, or M1/M2 Mac), `qai_hub_models` may still run: export can use “export without hub access” and produce ONNX locally, or submit cloud jobs. If the Qualcomm export fails, the scripts fall back to **Ultralytics** (det: `yolov8n.pt` → ONNX, seg: `yolov8n-seg.pt` → ONNX). Install: `pip install ultralytics`.
