import { useEffect, useState, useCallback, useRef } from "react";
import { API_BASE_URL } from "../config";

const MAIN_BACKEND = API_BASE_URL || (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

const SEMANTIC_CLASS_COLORS: Record<string, string> = {
  person: "#00ff66",
  bicycle: "#4488ff",
  car: "#4488ff",
  motorcycle: "#4488ff",
  bus: "#4488ff",
  truck: "#4488ff",
  default: "#888888",
};
function getSemanticClassColor(className: string): string {
  return SEMANTIC_CLASS_COLORS[className] ?? SEMANTIC_CLASS_COLORS.default;
}

interface Detection {
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
  distance_meters?: number | null;
}

interface VideoAnalysisJob {
  job_id: string;
  status: "idle" | "running" | "complete" | "error";
  current: number;
  total: number;
  message: string;
  error?: string;
  video_url?: string;
  fps?: number;
  total_frames?: number;
  summary?: { objects_found?: Record<string, number> };
  detections_by_frame?: Detection[][];
}

export function AutomaticPage() {
  const [videoJobId, setVideoJobId] = useState<string | null>(null);
  const [videoJob, setVideoJob] = useState<VideoAnalysisJob | null>(null);
  const [videoUploading, setVideoUploading] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const videoPlaybackRef = useRef<HTMLVideoElement>(null);
  const videoOverlayRef = useRef<HTMLCanvasElement>(null);
  const videoPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!videoJobId || !MAIN_BACKEND) return;
    const poll = async () => {
      try {
        const r = await fetch(`${MAIN_BACKEND}/api/video/analysis/${videoJobId}`);
        const data: VideoAnalysisJob = await r.json();
        setVideoJob(data);
        if (data.status === "complete" || data.status === "error") {
          if (videoPollRef.current) {
            clearInterval(videoPollRef.current);
            videoPollRef.current = null;
          }
        }
      } catch {}
    };
    poll();
    videoPollRef.current = setInterval(poll, 800);
    return () => {
      if (videoPollRef.current) {
        clearInterval(videoPollRef.current);
        videoPollRef.current = null;
      }
    };
  }, [videoJobId, MAIN_BACKEND]);

  const startVideoAnalysis = useCallback(async (file: File) => {
    if (!MAIN_BACKEND) return;
    setVideoUploading(true);
    setVideoJob(null);
    setVideoJobId(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await fetch(`${MAIN_BACKEND}/api/video/analyze?use_depth=true`, {
        method: "POST",
        body: form,
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        setVideoJob({
          job_id: "",
          status: "error",
          current: 0,
          total: 0,
          message: "",
          error: (err.detail as string) || r.statusText,
        });
        return;
      }
      const { job_id } = await r.json();
      setVideoJobId(job_id);
    } finally {
      setVideoUploading(false);
    }
  }, [MAIN_BACKEND]);

  const totalFrames = videoJob?.status === "complete" && videoJob?.total_frames != null
    ? Math.max(0, videoJob.total_frames - 1)
    : 0;
  const frameIndex = Math.max(0, Math.min(currentFrame, totalFrames));

  const drawVideoOverlay = useCallback(() => {
    const video = videoPlaybackRef.current;
    const canvas = videoOverlayRef.current;
    const job = videoJob;
    if (!video || !canvas || !job || job.status !== "complete" || !job.detections_by_frame) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const idx = Math.min(frameIndex, job.detections_by_frame.length - 1);
    if (idx < 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const dets = job.detections_by_frame[idx] ?? [];
    const vw = video.videoWidth || video.clientWidth;
    const vh = video.videoHeight || video.clientHeight;
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    const scaleX = canvas.width / (vw || 1);
    const scaleY = canvas.height / (vh || 1);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    dets.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const bx = x1 * scaleX;
      const by = y1 * scaleY;
      const bw = (x2 - x1) * scaleX;
      const bh = (y2 - y1) * scaleY;
      const distStr = det.distance_meters != null ? ` · ${Math.round(det.distance_meters * 100)}cm` : "";
      const label = `${det.class} ${(det.confidence * 100).toFixed(0)}%${distStr}`;
      ctx.strokeStyle = getSemanticClassColor(det.class);
      ctx.lineWidth = 2;
      ctx.strokeRect(bx, by, bw, bh);
      ctx.font = "bold 12px Inter, sans-serif";
      const textW = ctx.measureText(label).width + 8;
      ctx.fillStyle = getSemanticClassColor(det.class);
      ctx.fillRect(bx, by - 18, textW, 18);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, bx + 4, by - 4);
    });
  }, [videoJob, frameIndex]);

  // Sync video time to current frame
  useEffect(() => {
    const video = videoPlaybackRef.current;
    if (!video || videoJob?.status !== "complete" || !videoJob?.fps) return;
    const t = frameIndex / videoJob.fps;
    if (Math.abs(video.currentTime - t) > 0.05) {
      video.currentTime = t;
    }
  }, [frameIndex, videoJob?.status, videoJob?.fps]);

  // Redraw overlay when frame or video size changes
  useEffect(() => {
    const video = videoPlaybackRef.current;
    if (!video || videoJob?.status !== "complete") return;
    drawVideoOverlay();
    const onResize = () => drawVideoOverlay();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [videoJob?.status, frameIndex, drawVideoOverlay]);

  // Reset to frame 0 when new video is ready
  useEffect(() => {
    if (videoJob?.status === "complete" && videoJob?.total_frames != null) {
      setCurrentFrame(0);
    }
  }, [videoJob?.job_id, videoJob?.status]);

  // Arrow keys: frame by frame (only when video is ready and container is focused or document)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (videoJob?.status !== "complete" || videoJob?.total_frames == null) return;
      const total = Math.max(0, videoJob.total_frames - 1);
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setCurrentFrame((f) => Math.max(0, f - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setCurrentFrame((f) => Math.min(total, f + 1));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [videoJob?.status, videoJob?.total_frames]);

  const objectsList = videoJob?.status === "complete" && videoJob?.summary?.objects_found
    ? Object.entries(videoJob.summary.objects_found).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <section className="manual-page" style={{ padding: "1rem" }}>
      <h2 style={{ margin: "0 0 1rem", fontSize: "1.25rem", color: "rgba(255,255,255,0.95)" }}>
        Automatic — Upload video
      </h2>
      <p style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "rgba(255,255,255,0.8)" }}>
        Upload a video to run YOLO object detection and depth estimation. Play back with overlaid detections and download a PDF report with a list of objects found.
      </p>

      <div style={{ display: "flex", flex: 1, minHeight: 0, gap: "1rem", flexWrap: "wrap" }}>
        {/* Left: upload + playback */}
        <div style={{ flex: "1 1 400px", minWidth: 0, display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          <div style={{
            background: "rgba(0,0,0,0.35)",
            border: "2px dashed rgba(0,255,102,0.5)",
            borderRadius: 8,
            padding: "1rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.5rem",
          }}>
            <span className="ai-panel-label">UPLOAD YOUR VIDEO</span>
            <label style={{ fontSize: "0.9rem", color: "rgba(255,255,255,0.95)", cursor: videoUploading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <input
                type="file"
                accept=".mp4,.avi,.mov,.mkv,.webm"
                disabled={videoUploading}
                style={{ position: "absolute", width: 0, height: 0, opacity: 0 }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) startVideoAnalysis(f);
                  e.target.value = "";
                }}
              />
              <span style={{ padding: "0.4rem 0.75rem", background: "#00ff66", color: "#0a0f19", borderRadius: 6, fontWeight: 600 }}>Choose file</span>
              <span>MP4, AVI, MOV, MKV, WEBM — YOLO + depth</span>
            </label>
            {videoUploading && <span style={{ fontSize: "0.8rem" }}>Uploading…</span>}
            {videoJob?.status === "running" && (
              <span style={{ fontSize: "0.8rem" }}>
                Analyzing… {videoJob.current} / {videoJob.total} (YOLO + depth)
              </span>
            )}
            {videoJob?.status === "error" && (
              <span style={{ fontSize: "0.8rem", color: "#ff6b6b" }}>{videoJob.error ?? "Analysis failed"}</span>
            )}
            {videoJob?.status === "complete" && videoJob.video_url && (
              <div ref={containerRef} style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                <div style={{ position: "relative", width: "100%", background: "#000", borderRadius: 4, overflow: "hidden" }}>
                  <video
                    ref={videoPlaybackRef}
                    src={`${MAIN_BACKEND}${videoJob.video_url}`}
                    style={{ width: "100%", display: "block" }}
                    crossOrigin="anonymous"
                    playsInline
                  />
                  <canvas
                    ref={videoOverlayRef}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: "100%",
                      pointerEvents: "none",
                    }}
                  />
                </div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem", fontSize: "0.85rem", color: "rgba(255,255,255,0.85)" }}>
                  <span>
                    Frame <strong>{frameIndex + 1}</strong> / {videoJob.total_frames ?? 0}
                    {videoJob.detections_by_frame?.[frameIndex]?.length != null && (
                      <span style={{ marginLeft: "0.5rem", color: "rgba(255,255,255,0.7)" }}>
                        — {videoJob.detections_by_frame[frameIndex].length} object(s) in this frame
                      </span>
                    )}
                  </span>
                  <span style={{ color: "rgba(0,255,102,0.9)" }}>← → arrow keys: previous / next frame</span>
                </div>
                <a
                  href={`${MAIN_BACKEND}/api/video/analysis/${videoJob.job_id}/report.pdf`}
                  download={`cipher_video_report_${videoJob.job_id}.pdf`}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    fontSize: "0.9rem",
                    color: "#00ff66",
                    textDecoration: "none",
                    fontWeight: 600,
                  }}
                >
                  Download PDF report
                </a>
              </div>
            )}
          </div>
        </div>

        {/* Right: list of objects detected */}
        <div style={{ flex: "0 0 280px", minWidth: 0, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <span className="ai-panel-label">OBJECTS DETECTED</span>
          <div style={{
            background: "rgba(0,0,0,0.3)",
            border: "1px solid rgba(255,255,255,0.15)",
            borderRadius: 8,
            padding: "0.75rem",
            maxHeight: 320,
            overflowY: "auto",
          }}>
            {objectsList.length === 0 && !videoJob?.summary && (
              <p style={{ margin: 0, fontSize: "0.85rem", color: "rgba(255,255,255,0.6)" }}>
                Upload and analyze a video to see the list of objects found (with max count per frame).
              </p>
            )}
            {videoJob?.status === "running" && (
              <p style={{ margin: 0, fontSize: "0.85rem", color: "rgba(255,255,255,0.8)" }}>
                Building list…
              </p>
            )}
            {videoJob?.status === "error" && (
              <p style={{ margin: 0, fontSize: "0.85rem", color: "#ff6b6b" }}>
                {videoJob.error ?? "Analysis failed"}
              </p>
            )}
            {objectsList.length > 0 && (
              <ul style={{ margin: 0, paddingLeft: "1.25rem", fontSize: "0.9rem", color: "rgba(255,255,255,0.9)" }}>
                {objectsList.map(([cls, count]) => (
                  <li key={cls} style={{ marginBottom: "0.35rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: 2,
                        background: getSemanticClassColor(cls),
                        flexShrink: 0,
                      }}
                    />
                    <strong>{cls}</strong>
                    <span style={{ color: "rgba(255,255,255,0.7)" }}>
                      (max {count} in a frame)
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
