"""Analyze uploaded video with YOLO (+ optional depth), store per-frame detections for playback. Generate PDF report."""

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Job state: status, progress, result
_video_jobs: Dict[str, Dict[str, Any]] = {}
_VIDEO_ANALYZE_DIR: Optional[Path] = None


def get_analyze_dir() -> Path:
    global _VIDEO_ANALYZE_DIR
    if _VIDEO_ANALYZE_DIR is None:
        from pathlib import Path
        _here = Path(__file__).resolve().parent
        _root = _here.parent.parent
        _VIDEO_ANALYZE_DIR = _root / "exports" / "video_analysis"
    return _VIDEO_ANALYZE_DIR


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _video_jobs.get(job_id)


def create_job() -> str:
    job_id = str(uuid.uuid4())[:8]
    _video_jobs[job_id] = {
        "status": "idle",
        "current": 0,
        "total": 0,
        "message": "",
        "video_url": None,
        "detections_by_frame": [],
        "fps": 30.0,
        "total_frames": 0,
        "summary": {},
        "error": None,
    }
    return job_id


def run_analyze(
    job_id: str,
    video_path: str,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    use_depth: bool = False,
) -> None:
    """Process video: YOLO (and optional depth) per frame; save detections and copy video for playback."""
    import cv2
    import numpy as np

    job = _video_jobs.get(job_id)
    if not job or job["status"] == "running":
        return

    job["status"] = "running"
    job["current"] = 0
    job["total"] = 0
    job["message"] = "Opening video..."
    job["error"] = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            job["status"] = "error"
            job["error"] = "Could not open video"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        job["total_frames"] = total_frames
        job["fps"] = fps
        job["total"] = total_frames
        job["message"] = f"Processing 0 / {total_frames}"

        # Process every Nth frame: ~2 FPS analysis for speed (was ~5 FPS)
        analysis_fps = 2.0
        step = max(1, int(round(fps / analysis_fps)))
        detections_by_frame: List[List[Dict]] = []
        for fi in range(total_frames):
            detections_by_frame.append([])  # default empty

        # YOLO: create detector ONCE and reuse (was creating per frame — very slow)
        _here = Path(__file__).resolve().parent
        root = _here.parent.parent  # repo root (Drone2)
        yolo_detector = None
        try:
            from backend.perception import YOLODetector
            from backend import config as cfg
            # Use same NPU/GPU config as live feed so video analysis runs on NPU when available
            qnn_path = getattr(cfg, "QNN_DLL_PATH", None)
            use_gpu = getattr(cfg, "USE_GPU", True)
            split_npu_gpu = getattr(cfg, "SPLIT_NPU_GPU", False)
            prefer_npu = getattr(cfg, "PREFER_NPU_OVER_GPU", True)
            # Prefer 320 model for video (faster on NPU); fallback to 640
            fast_path = root / "models" / "yolov8_det_320.onnx"
            main_path = root / "models" / "yolov8_det.onnx"
            if fast_path.exists():
                yolo_detector = YOLODetector(
                    str(fast_path),
                    qnn_dll_path=qnn_path,
                    confidence_threshold=getattr(cfg, "YOLO_CONFIDENCE_THRESHOLD", 0.45),
                    input_size=320,
                    use_gpu=use_gpu,
                    split_npu_gpu=split_npu_gpu,
                    prefer_npu_over_gpu=prefer_npu,
                )
            elif main_path.exists():
                yolo_detector = YOLODetector(
                    str(main_path),
                    qnn_dll_path=qnn_path,
                    confidence_threshold=getattr(cfg, "YOLO_CONFIDENCE_THRESHOLD", 0.45),
                    input_size=640,
                    use_gpu=use_gpu,
                    split_npu_gpu=split_npu_gpu,
                    prefer_npu_over_gpu=prefer_npu,
                )
        except Exception:
            pass

        def run_yolo(frame_bgr: np.ndarray) -> List[Dict]:
            if yolo_detector is not None:
                dets = yolo_detector.detect(frame_bgr)
                return [{"class": d.get("class", "?"), "confidence": round(float(d.get("confidence", 0)), 2), "bbox": list(d.get("bbox", [0, 0, 0, 0])), "distance_meters": d.get("distance_meters")} for d in dets]
            try:
                from PIL import Image
                from Drone import models
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                dets = models.detect_objects(img)
                return [{"class": d.get("class", "?"), "confidence": round(float(d.get("confidence", 0)), 2), "bbox": list(d.get("bbox", [0, 0, 0, 0])), "distance_meters": None} for d in dets]
            except Exception:
                return []

        depth_estimator = None
        if use_depth:
            try:
                from Drone.local_backend.depth_estimator import DepthEstimator
                depth_estimator = DepthEstimator()
                if not depth_estimator.loaded:
                    depth_estimator = None
            except Exception:
                pass

        # Run depth only every N-th processed frame to save time
        depth_every_n = 3 if use_depth else 0
        processed_count = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                dets = run_yolo(frame)
                run_depth = (
                    depth_estimator
                    and depth_estimator.loaded
                    and frame is not None
                    and (depth_every_n <= 0 or (processed_count % depth_every_n) == 0)
                )
                if run_depth:
                    try:
                        h, w = frame.shape[:2]
                        depth_map = depth_estimator.infer(frame)
                        if depth_map is not None:
                            for d in dets:
                                d["distance_meters"] = depth_estimator.depth_at_bbox(
                                    depth_map, d.get("bbox", [0, 0, 1, 1]), w, h
                                )
                    except Exception:
                        pass
                processed_count += 1
                detections_by_frame[frame_idx] = dets
                job["current"] = frame_idx + 1
                job["message"] = f"Processing frame {frame_idx + 1} / {total_frames}"
                if on_progress:
                    on_progress(frame_idx + 1, total_frames, job["message"])
            frame_idx += 1

        cap.release()

        # Fill gaps: copy detections from nearest processed frame so playback has data every frame
        last_dets = []
        for i in range(total_frames):
            if detections_by_frame[i]:
                last_dets = detections_by_frame[i]
            else:
                detections_by_frame[i] = list(last_dets)

        job["detections_by_frame"] = detections_by_frame

        # Summary: unique classes and counts (max per frame)
        class_counts: Dict[str, int] = {}
        for dets in detections_by_frame:
            seen = set()
            for d in dets:
                c = d.get("class", "?")
                if c not in seen:
                    seen.add(c)
                    class_counts[c] = class_counts.get(c, 0) + 1
        job["summary"] = {"objects_found": class_counts, "total_frames": total_frames, "fps": fps}

        # Copy video to exports for playback
        out_dir = get_analyze_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / f"{job_id}.mp4"
        try:
            import shutil
            shutil.copy2(video_path, str(out_video))
            job["video_url"] = f"/exports/video_analysis/{job_id}.mp4"
        except Exception as e:
            job["video_url"] = None
            job["error"] = f"Video copy failed: {e}"

        # Save detections JSON for frontend
        det_json = out_dir / f"{job_id}_detections.json"
        with open(det_json, "w") as f:
            json.dump({"fps": fps, "total_frames": total_frames, "detections_by_frame": detections_by_frame}, f)

        job["status"] = "complete"
        job["message"] = f"Complete — {total_frames} frames analyzed"
    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"] = str(e)
        job["message"] = traceback.format_exc()
    finally:
        try:
            if os.path.isfile(video_path):
                os.unlink(video_path)
        except Exception:
            pass


def run_analyze_async(job_id: str, video_path: str, use_depth: bool = False) -> threading.Thread:
    def task():
        run_analyze(job_id, video_path, use_depth=use_depth)

    t = threading.Thread(target=task, daemon=True)
    t.start()
    return t


def generate_report_pdf(job_id: str, out_path: Path) -> bool:
    """Generate a PDF report: list of objects found and a simple plan. Returns True on success."""
    job = _video_jobs.get(job_id)
    if not job or job["status"] != "complete":
        return False
    try:
        from fpdf import FPDF
    except ImportError:
        return False
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "CIPHER — Video Analysis Report", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Job: {job_id}", ln=True)
        pdf.cell(0, 8, f"Frames: {job.get('total_frames', 0)} @ {job.get('fps', 30):.1f} fps", ln=True)
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Objects detected", ln=True)
        pdf.set_font("Helvetica", "", 11)
        summary = job.get("summary", {})
        objs = summary.get("objects_found", {})
        for cls, count in sorted(objs.items(), key=lambda x: -x[1]):
            pdf.cell(0, 6, f"  • {cls}: {count} (max in a single frame)", ln=True)
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Plan / summary", ln=True)
        pdf.set_font("Helvetica", "", 11)
        plan = (
            "Review the video playback with overlaid detections for full context. "
            "Prioritize high-confidence detections. Use the object list above for inventory or reporting."
        )
        pdf.multi_cell(0, 6, plan)
        pdf.output(str(out_path))
        return True
    except Exception:
        return False
