"""
Build ONNX Runtime execution provider lists for NPU + GPU acceleration.

- NPU (QNN) first when available — HTP backend so Task Manager shows NPU Compute (not just Shared Memory).
- GPU (CUDA / DirectML) next when not preferring NPU.
- CPU as fallback.

Use PREFER_NPU_OVER_GPU so YOLO/Depth run on NPU. Requires: pip install onnxruntime-qnn
"""

from pathlib import Path
from typing import Any, Dict, List

# QNN provider options to force HTP (NPU) compute so Task Manager shows "NPU Compute" not just "Shared Memory"
# backend_type 'htp' = offload to NPU; htp_performance_mode = drive NPU; enable_htp_shared_memory_allocator '0' = use NPU compute path
QNN_HTP_OPTIONS: Dict[str, str] = {
    "backend_type": "htp",  # HTP = Hexagon Tensor Processor (NPU)
    "htp_performance_mode": "high_performance",  # Use NPU actively so NPU Compute shows in Task Manager
    "qnn_context_priority": "high",  # Prefer NPU context
    "enable_htp_shared_memory_allocator": "0",  # 0 = default, use NPU compute; 1 = shared memory allocator
}


def get_available_providers():
    """Return set of available ONNX Runtime execution provider names."""
    try:
        import onnxruntime as ort
        return set(ort.get_available_providers())
    except Exception:
        return set()


def resolve_qnn_backend_path(config_path: str | None) -> str | None:
    """
    Resolve path to QnnHtp.dll for NPU. Tries config path, then DLL bundled with
    onnxruntime-qnn (capi/ or lib/), then Qualcomm AIStack. Returns None if not found.
    """
    if config_path and Path(config_path).exists():
        return config_path
    try:
        import onnxruntime as ort
        ort_root = Path(ort.__file__).resolve().parent
        # onnxruntime-qnn 1.18+: often in capi/ or lib/
        for subdir in ("capi", "lib", "."):
            for name in ("QnnHtp.dll", "QnnHtp.so", "libQnnHtp.so"):
                p = (ort_root / subdir / name) if subdir != "." else (ort_root / name)
                if p.exists():
                    return str(p)
        # Windows Snapdragon: Qualcomm AIStack (optional)
        import os
        qairt = os.environ.get("QAIRT_PATH", r"C:\Qualcomm\AIStack\QAIRT")
        for libdir in ("lib", "lib/arm64x-windows-msvc", "lib/x86_64-windows-msvc"):
            p = Path(qairt) / libdir / "QnnHtp.dll"
            if p.exists():
                return str(p)
    except Exception:
        pass
    return None


def build_providers(
    available: set,
    qnn_dll_path: str | None,
    use_gpu: bool,
    prefer_npu: bool = True,
) -> List[Any]:
    """
    Build provider list: NPU (QNN) and/or GPU (CUDA, DirectML), then CPU.
    When prefer_npu and QNN path exists: [QNN, CPU] so NPU is used (no GPU).
    Otherwise: [QNN?, CUDA?, DML?, CPU].
    """
    resolved_qnn = resolve_qnn_backend_path(qnn_dll_path)
    qnn = []
    if "QNNExecutionProvider" in available:
        qnn.append(("QNNExecutionProvider", _qnn_provider_options(resolved_qnn)))
    gpu = []
    if use_gpu and not prefer_npu:
        if "CUDAExecutionProvider" in available:
            gpu.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:
            gpu.append("DmlExecutionProvider")
    # Prefer NPU: use only [QNN, CPU] so inference runs on NPU (Task Manager shows NPU)
    if prefer_npu and qnn:
        return qnn + ["CPUExecutionProvider"]
    if prefer_npu:
        return gpu + ["CPUExecutionProvider"]
    return qnn + gpu + ["CPUExecutionProvider"]


def _qnn_provider_options(backend_path: str | None) -> Dict[str, str]:
    """Build QNN EP options so HTP (NPU) compute is used and Task Manager shows NPU Compute, not just Shared Memory."""
    opts = dict(QNN_HTP_OPTIONS)
    if backend_path:
        opts["backend_path"] = backend_path
        opts.pop("backend_type", None)  # backend_path and backend_type are mutually exclusive
    return opts


def yolo_providers(
    available: set,
    qnn_dll_path: str | None,
    use_gpu: bool,
    split_npu_gpu: bool,
    prefer_npu_over_gpu: bool = True,
) -> List[Any]:
    """
    Providers for YOLO. When prefer_npu_over_gpu: [QNN, CPU] so NPU is used.
    When split_npu_gpu and not prefer_npu: YOLO→NPU, Depth→GPU.
    """
    if prefer_npu_over_gpu and "QNNExecutionProvider" in available:
        # Force HTP (NPU) so Task Manager shows NPU Compute
        resolved = resolve_qnn_backend_path(qnn_dll_path)
        opts = _qnn_provider_options(resolved)
        try:
            return [("QNNExecutionProvider", opts), "CPUExecutionProvider"]
        except Exception:
            if resolved:
                return [("QNNExecutionProvider", {"backend_path": resolved}), "CPUExecutionProvider"]
            try:
                return [("QNNExecutionProvider", {"backend_type": "htp"}), "CPUExecutionProvider"]
            except Exception:
                pass
    if prefer_npu_over_gpu:
        return ["CPUExecutionProvider"]
    if split_npu_gpu and use_gpu:
        resolved = resolve_qnn_backend_path(qnn_dll_path)
        prov = []
        if "QNNExecutionProvider" in available:
            prov.append(("QNNExecutionProvider", _qnn_provider_options(resolved)))
        prov.append("CPUExecutionProvider")
        return prov
    return build_providers(available, qnn_dll_path, use_gpu, prefer_npu=True)


def depth_providers(
    available: set,
    qnn_dll_path: str | None,
    use_gpu: bool,
    split_npu_gpu: bool,
    prefer_npu_over_gpu: bool = True,
) -> List[Any]:
    """
    Providers for Depth. When prefer_npu_over_gpu: [QNN, CPU] so Depth runs on NPU (HTP) too.
    """
    if prefer_npu_over_gpu and "QNNExecutionProvider" in available:
        resolved = resolve_qnn_backend_path(qnn_dll_path)
        opts = _qnn_provider_options(resolved)
        try:
            return [("QNNExecutionProvider", opts), "CPUExecutionProvider"]
        except Exception:
            if resolved:
                return [("QNNExecutionProvider", {"backend_path": resolved}), "CPUExecutionProvider"]
            try:
                return [("QNNExecutionProvider", {"backend_type": "htp"}), "CPUExecutionProvider"]
            except Exception:
                pass
    if prefer_npu_over_gpu:
        return ["CPUExecutionProvider"]
    if split_npu_gpu and use_gpu:
        prov = []
        if "CUDAExecutionProvider" in available:
            prov.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:
            prov.append("DmlExecutionProvider")
        resolved = resolve_qnn_backend_path(qnn_dll_path)
        if "QNNExecutionProvider" in available:
            prov.append(("QNNExecutionProvider", _qnn_provider_options(resolved)))
        prov.append("CPUExecutionProvider")
        return prov
    return build_providers(available, qnn_dll_path, use_gpu, prefer_npu=True)
