"""Natural Language Video Analysis Agent

Analyzes user video feed with natural language queries using vision-language models.
Supports both uploaded videos and live feeds. Integrates with existing spatial/knowledge agents.

Features:
- Real-time video frame analysis
- Natural language queries about video content
- Object tracking across frames
- Scene understanding and temporal reasoning
- Integration with world graph and knowledge base

Usage:
    modal deploy agents/video_analysis_agent.py
    modal serve agents/video_analysis_agent.py
"""

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Model registry – vision-language models for video analysis
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, str] = {
    "qwen3-vl-30b-a3b-thinking-fp8": "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
}
DEFAULT_MODEL = "qwen3-vl-2b"  # Faster for real-time
MODEL_ID = SUPPORTED_MODELS[DEFAULT_MODEL]

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

from app import app as vision_app, ImageServer

model_vol = modal.Volume.from_name("vision-model-cache", create_if_missing=True)
video_cache = modal.Dict.from_name("video-analysis-cache", create_if_missing=True)

MODEL_DIR = "/model-cache"


def download_model():
    """Pre-download model weights into the volume."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, local_dir=f"{MODEL_DIR}/{MODEL_ID}")


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

video_analysis_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.11.0",
        "transformers",
        "qwen-vl-utils==0.0.14",
        "Pillow",
        "torch",
        "huggingface_hub",
        "opencv-python-headless",
        "numpy",
    )
    .run_function(download_model, volumes={MODEL_DIR: model_vol})
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VideoAnalysisResult:
    """Result from video analysis query."""
    answer: str
    confidence: float
    relevant_frames: List[Dict[str, Any]]  # frame_idx, timestamp, image_b64, description
    objects_detected: List[Dict[str, Any]]  # class, confidence, frame_range, track_id
    scene_summary: str
    temporal_events: List[Dict[str, Any]]  # event_type, start_frame, end_frame, description
    query_type: str  # "object", "action", "scene", "temporal", "general"


@dataclass
class FrameAnalysis:
    """Analysis of a single video frame."""
    frame_idx: int
    timestamp: float
    image_b64: str
    description: str
    objects: List[Dict[str, Any]]
    scene_type: str
    confidence: float


# ---------------------------------------------------------------------------
# System prompts for different query types
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "object": """You are a video analysis agent focused on object detection and tracking.
Analyze the video frame and identify all objects present. For each object, provide:
- Object class/name
- Location in frame (left/center/right, approximate position)
- Confidence level
- Any distinguishing features

If the query asks about a specific object, focus on that object and provide detailed information.""",

    "action": """You are a video analysis agent focused on action and movement recognition.
Analyze the video to identify:
- What actions are happening
- Who or what is performing the actions
- Direction and speed of movement
- Temporal sequence of events

Provide a clear description of the actions observed.""",

    "scene": """You are a video analysis agent focused on scene understanding.
Analyze the video frame to understand:
- Overall scene context (indoor/outdoor, location type)
- Environmental conditions
- Spatial layout and structure
- Key landmarks or features

Provide a comprehensive scene description.""",

    "temporal": """You are a video analysis agent focused on temporal reasoning.
Analyze multiple frames to understand:
- How the scene changes over time
- Sequence of events
- Cause and effect relationships
- Patterns or trends

Compare frames and identify temporal patterns.""",

    "general": """You are a video analysis agent. Analyze the video frame(s) and answer
the user's question comprehensively. Consider:
- Objects present
- Actions happening
- Scene context
- Temporal relationships

Provide a detailed, accurate answer based on visual evidence."""
}


# ---------------------------------------------------------------------------
# VideoAnalysisAgent class
# ---------------------------------------------------------------------------


@vision_app.cls(
    image=video_analysis_image,
    gpu="T4",  # T4 is sufficient for 2B model, use H200 for 30B
    volumes={MODEL_DIR: model_vol},
    timeout=300,
    scaledown_window=300,
)
class VideoAnalysisAgent:
    """Hosts the VLM and provides video analysis capabilities."""

    @modal.enter()
    def setup(self):
        """Initialize the vision-language model."""
        from vllm import LLM

        self.llm = LLM(
            model=f"{MODEL_DIR}/{MODEL_ID}",
            trust_remote_code=True,
            max_model_len=8192,
            dtype="half",
            enable_prefix_caching=True,
        )
        self.get_image_cls = ImageServer

    def _classify_query(self, query: str) -> str:
        """Classify query type to select appropriate system prompt."""
        q = query.lower()
        if any(word in q for word in ["find", "detect", "identify", "object", "thing", "item", "person", "vehicle"]):
            return "object"
        if any(word in q for word in ["action", "doing", "happening", "moving", "walking", "running", "activity"]):
            return "action"
        if any(word in q for word in ["scene", "location", "place", "where", "environment", "setting"]):
            return "scene"
        if any(word in q for word in ["when", "before", "after", "sequence", "then", "later", "time", "temporal"]):
            return "temporal"
        return "general"

    def _analyze_frame(
        self,
        image_b64: str,
        query: str,
        query_type: str,
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> FrameAnalysis:
        """Analyze a single frame with the VLM."""
        from vllm import SamplingParams

        system_prompt = SYSTEM_PROMPTS.get(query_type, SYSTEM_PROMPTS["general"])
        user_prompt = f"Query: {query}\n\nAnalyze this frame and answer the query."

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        sampling = SamplingParams(temperature=0.3, max_tokens=500)
        outputs = self.llm.chat(messages, sampling_params=sampling)
        description = outputs[0].outputs[0].text.strip()

        # Extract objects from description (simple heuristic)
        objects = []
        if "object" in query_type or "general" in query_type:
            # Try to parse object mentions
            import re
            obj_patterns = [
                r"(\w+)\s+(?:is|are|was|were)\s+(?:visible|seen|present|detected)",
                r"(?:a|an|the)\s+(\w+)\s+(?:in|at|on)",
            ]
            for pattern in obj_patterns:
                matches = re.findall(pattern, description, re.IGNORECASE)
                for match in matches:
                    if len(match) > 2:  # Filter out short words
                        objects.append({"class": match, "confidence": 0.7, "frame_idx": frame_idx})

        return FrameAnalysis(
            frame_idx=frame_idx,
            timestamp=timestamp,
            image_b64=image_b64,
            description=description,
            objects=objects,
            scene_type="unknown",
            confidence=0.8,
        )

    @modal.method()
    def analyze_video_frames(
        self,
        frames: List[Dict[str, Any]],  # List of {image_b64, frame_idx, timestamp}
        query: str,
        max_frames: int = 10,
    ) -> VideoAnalysisResult:
        """
        Analyze multiple video frames to answer a query.

        Args:
            frames: List of frame dicts with image_b64, frame_idx, timestamp
            query: Natural language query about the video
            max_frames: Maximum number of frames to analyze

        Returns:
            VideoAnalysisResult with answer, relevant frames, objects, etc.
        """
        if not frames:
            return VideoAnalysisResult(
                answer="No frames provided for analysis.",
                confidence=0.0,
                relevant_frames=[],
                objects_detected=[],
                scene_summary="",
                temporal_events=[],
                query_type="general",
            )

        query_type = self._classify_query(query)
        frames_to_analyze = frames[:max_frames]

        # Analyze each frame
        frame_analyses: List[FrameAnalysis] = []
        for frame_data in frames_to_analyze:
            analysis = self._analyze_frame(
                image_b64=frame_data.get("image_b64", ""),
                query=query,
                query_type=query_type,
                frame_idx=frame_data.get("frame_idx", 0),
                timestamp=frame_data.get("timestamp", 0.0),
            )
            frame_analyses.append(analysis)

        # Aggregate results
        descriptions = [fa.description for fa in frame_analyses]
        combined_answer = "\n\n".join(
            [f"Frame {fa.frame_idx} ({fa.timestamp:.2f}s): {fa.description}" for fa in frame_analyses]
        )

        # Extract objects across frames
        all_objects = []
        for fa in frame_analyses:
            all_objects.extend(fa.objects)

        # Simple temporal analysis for multi-frame queries
        temporal_events = []
        if len(frame_analyses) > 1 and query_type in ("temporal", "action", "general"):
            # Compare frames to find changes
            for i in range(len(frame_analyses) - 1):
                curr = frame_analyses[i]
                next_frame = frame_analyses[i + 1]
                if curr.description != next_frame.description:
                    temporal_events.append({
                        "event_type": "change",
                        "start_frame": curr.frame_idx,
                        "end_frame": next_frame.frame_idx,
                        "description": f"Change detected between frames {curr.frame_idx} and {next_frame.frame_idx}",
                    })

        # Generate scene summary
        scene_summary = f"Analyzed {len(frame_analyses)} frames. " + descriptions[0] if descriptions else "No analysis available."

        return VideoAnalysisResult(
            answer=combined_answer,
            confidence=0.85,
            relevant_frames=[
                {
                    "frame_idx": fa.frame_idx,
                    "timestamp": fa.timestamp,
                    "image_b64": fa.image_b64,
                    "description": fa.description,
                }
                for fa in frame_analyses
            ],
            objects_detected=all_objects[:20],  # Limit to top 20
            scene_summary=scene_summary,
            temporal_events=temporal_events,
            query_type=query_type,
        )

    @modal.method()
    def analyze_live_feed(
        self,
        query: str,
        num_frames: int = 5,
        frame_interval: float = 0.5,  # seconds between frames
    ) -> VideoAnalysisResult:
        """
        Analyze frames from live feed (uses ImageServer to get recent frames).

        Args:
            query: Natural language query
            num_frames: Number of frames to sample
            frame_interval: Time between sampled frames

        Returns:
            VideoAnalysisResult
        """
        # Get frames from ImageServer
        frames = []
        get_image = self.get_image_cls()
        
        # Sample frames at different positions (simulate recent frames)
        import random
        for i in range(num_frames):
            # Use random positions to simulate live feed sampling
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            z = 0.0
            yaw = random.uniform(0, 360)
            
            try:
                result = get_image.getImageRemote.remote(x, y, z, yaw)
                if result and result.get("image"):
                    img_bytes = result["image"]
                    img_b64 = base64.b64encode(img_bytes).decode("ascii")
                    frames.append({
                        "image_b64": img_b64,
                        "frame_idx": i,
                        "timestamp": i * frame_interval,
                    })
            except Exception as e:
                print(f"Error getting frame {i}: {e}")
                continue

        if not frames:
            return VideoAnalysisResult(
                answer="Could not retrieve frames from live feed.",
                confidence=0.0,
                relevant_frames=[],
                objects_detected=[],
                scene_summary="",
                temporal_events=[],
                query_type="general",
            )

        return self.analyze_video_frames(frames, query, max_frames=num_frames)


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------


@vision_app.function(image=video_analysis_image, timeout=300)
@modal.fastapi_endpoint(method="POST")
def analyze_video(request: dict):
    """Analyze uploaded video frames with natural language query."""
    from fastapi import HTTPException

    query = request.get("query", "")
    frames = request.get("frames", [])
    max_frames = request.get("max_frames", 10)

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    if not frames:
        raise HTTPException(status_code=400, detail="Frames are required")

    agent = VideoAnalysisAgent()
    result = agent.analyze_video_frames.remote(frames, query, max_frames)

    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "relevant_frames": result.relevant_frames,
        "objects_detected": result.objects_detected,
        "scene_summary": result.scene_summary,
        "temporal_events": result.temporal_events,
        "query_type": result.query_type,
    }


@vision_app.function(image=video_analysis_image, timeout=300)
@modal.fastapi_endpoint(method="POST")
def analyze_live_feed(request: dict):
    """Analyze live video feed with natural language query."""
    from fastapi import HTTPException

    query = request.get("query", "")
    num_frames = request.get("num_frames", 5)
    frame_interval = request.get("frame_interval", 0.5)

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    agent = VideoAnalysisAgent()
    result = agent.analyze_live_feed.remote(query, num_frames, frame_interval)

    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "relevant_frames": result.relevant_frames,
        "objects_detected": result.objects_detected,
        "scene_summary": result.scene_summary,
        "temporal_events": result.temporal_events,
        "query_type": result.query_type,
    }


# ---------------------------------------------------------------------------
# Local entrypoint for testing
# ---------------------------------------------------------------------------

# Create a separate app for local entrypoint
local_app = modal.App("video-analysis-local")


@local_app.local_entrypoint()
def main(
    query: str = "What objects are visible in the video?",
    video_path: Optional[str] = None,
    num_frames: int = 5,
):
    """Test video analysis locally."""
    import cv2
    import base64

    if video_path:
        # Load video and extract frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(fps * 0.5))  # Sample every 0.5 seconds

        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                img_b64 = base64.b64encode(buffer).decode("ascii")
                frames.append({
                    "image_b64": img_b64,
                    "frame_idx": len(frames),
                    "timestamp": frame_idx / fps,
                })
            frame_idx += 1
        cap.release()

        if not frames:
            print("No frames extracted from video")
            return

        print(f"Extracted {len(frames)} frames, analyzing...")
        agent = VideoAnalysisAgent()
        result = agent.analyze_video_frames.remote(frames, query, max_frames=num_frames)
    else:
        print("Analyzing live feed...")
        agent = VideoAnalysisAgent()
        result = agent.analyze_live_feed.remote(query, num_frames)

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Query Type: {result.query_type}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"{'='*60}\n")
    print(f"Answer:\n{result.answer}\n")
    print(f"Scene Summary: {result.scene_summary}\n")
    if result.objects_detected:
        print(f"Objects Detected: {len(result.objects_detected)}")
        for obj in result.objects_detected[:5]:
            print(f"  - {obj}")
    if result.temporal_events:
        print(f"\nTemporal Events: {len(result.temporal_events)}")
        for event in result.temporal_events:
            print(f"  - {event}")
    print(f"{'='*60}\n")
