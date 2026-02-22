"""Local Video Analysis Agent

Natural language video analysis that works locally without Modal.
Integrates with existing YOLO, CLIP, and knowledge agents.

Usage:
    from video_analysis_agent import analyze_video_query
    
    result = analyze_video_query(
        query="What objects are in the video?",
        frames=[{"image_b64": "...", "frame_idx": 0, "timestamp": 0.0}],
        world_graph=world_graph
    )
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import base64
import io
from PIL import Image


@dataclass
class VideoAnalysisResult:
    """Result from video analysis query."""
    answer: str
    confidence: float
    relevant_frames: List[Dict[str, Any]]
    objects_detected: List[Dict[str, Any]]
    scene_summary: str
    temporal_events: List[Dict[str, Any]]
    query_type: str
    spatial_match: bool = False
    knowledge_match: bool = False


def classify_query(query: str) -> str:
    """Classify query type."""
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


def analyze_video_query(
    query: str,
    frames: List[Dict[str, Any]],
    world_graph: Any = None,
    models: Any = None,
    max_frames: int = 10,
) -> VideoAnalysisResult:
    """
    Analyze video frames with natural language query.
    
    Args:
        query: Natural language query about the video
        frames: List of frame dicts with image_b64, frame_idx, timestamp
        world_graph: Optional world graph for spatial/knowledge integration
        models: Optional models object with detect_objects method
        max_frames: Maximum frames to analyze
    
    Returns:
        VideoAnalysisResult with comprehensive analysis
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
    
    query_type = classify_query(query)
    frames_to_analyze = frames[:max_frames]
    
    # Try spatial agent first if query matches
    spatial_answer = ""
    spatial_match = False
    if world_graph:
        q_lower = query.lower()
        spatial_triggers = ("where", "find", "locate", "show me", "which node", "which frame")
        if any(t in q_lower for t in spatial_triggers):
            try:
                from spatial_agent import run_search
                spatial_res = run_search(query, world_graph)
                if spatial_res.found:
                    spatial_answer = f"Spatial match found: {spatial_res.description}"
                    spatial_match = True
            except Exception:
                pass
    
    # Try knowledge agent for procedural queries
    knowledge_answer = ""
    knowledge_match = False
    q_lower = query.lower()
    knowledge_triggers = ("what should", "how do", "procedure", "protocol", "recommend", "advice")
    if any(t in q_lower for t in knowledge_triggers) and world_graph:
        try:
            from knowledge_agent import answer_question
            kr = answer_question(query, world_graph)
            if kr.answer_text:
                knowledge_answer = kr.answer_text
                knowledge_match = True
        except Exception:
            pass
    
    # Analyze frames with YOLO + CLIP
    analyzed_frames = []
    all_objects = []
    frame_descriptions = []
    
    for frame_data in frames_to_analyze:
        image_b64 = frame_data.get("image_b64", "")
        if not image_b64:
            continue
        
        try:
            # Decode image
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # YOLO detection
            detections = []
            if models and hasattr(models, "detect_objects"):
                try:
                    detections = models.detect_objects(image)
                except Exception:
                    pass
            
            # Build description
            det_names = [d["class"] for d in detections if d.get("confidence", 0) > 0.4]
            description = f"Frame {frame_data.get('frame_idx', 0)} (t={frame_data.get('timestamp', 0.0):.2f}s): "
            
            if det_names:
                description += f"Detected {', '.join(det_names)}. "
            else:
                description += "No objects detected. "
            
            # CLIP semantic matching for object queries
            if query_type == "object" and world_graph:
                try:
                    from clip_navigator import embed_text, embed_frame, _cosine
                    query_emb = embed_text(query)
                    frame_emb = embed_frame(image)
                    if query_emb is not None and frame_emb is not None:
                        similarity = _cosine(query_emb, frame_emb)
                        description += f"Query relevance: {similarity:.2f}. "
                except Exception:
                    pass
            
            analyzed_frames.append({
                "frame_idx": frame_data.get("frame_idx", 0),
                "timestamp": frame_data.get("timestamp", 0.0),
                "image_b64": image_b64,
                "description": description,
            })
            
            frame_descriptions.append(description)
            
            # Collect objects
            for det in detections:
                if det.get("confidence", 0) > 0.4:
                    all_objects.append({
                        "class": det["class"],
                        "confidence": det["confidence"],
                        "frame_idx": frame_data.get("frame_idx", 0),
                        "timestamp": frame_data.get("timestamp", 0.0),
                    })
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error analyzing frame: {e}")
            continue
    
    # Build combined answer
    answer_parts = []
    if spatial_answer:
        answer_parts.append(spatial_answer)
    if knowledge_answer:
        answer_parts.append(f"Knowledge base: {knowledge_answer}")
    
    if frame_descriptions:
        answer_parts.append("\n".join(frame_descriptions))
    else:
        answer_parts.append("Frame analysis complete.")
    
    final_answer = "\n\n".join(answer_parts) if answer_parts else "Analysis complete."
    
    # Temporal events (simple change detection)
    temporal_events = []
    if len(analyzed_frames) > 1 and query_type in ("temporal", "action", "general"):
        prev_detections = set()
        for frame in analyzed_frames:
            frame_objs = set()
            for obj in all_objects:
                if obj["frame_idx"] == frame["frame_idx"]:
                    frame_objs.add(obj["class"])
            
            if prev_detections and frame_objs != prev_detections:
                added = frame_objs - prev_detections
                removed = prev_detections - frame_objs
                if added or removed:
                    event_desc = []
                    if added:
                        event_desc.append(f"Added: {', '.join(added)}")
                    if removed:
                        event_desc.append(f"Removed: {', '.join(removed)}")
                    temporal_events.append({
                        "event_type": "change",
                        "start_frame": frame["frame_idx"] - 1,
                        "end_frame": frame["frame_idx"],
                        "description": "; ".join(event_desc),
                    })
            prev_detections = frame_objs
    
    # Scene summary
    scene_summary = f"Analyzed {len(analyzed_frames)} frames. "
    if all_objects:
        unique_objects = list(set(obj["class"] for obj in all_objects))
        scene_summary += f"Found {len(unique_objects)} unique object types: {', '.join(unique_objects[:5])}"
        if len(unique_objects) > 5:
            scene_summary += f" and {len(unique_objects) - 5} more."
    else:
        scene_summary += "No objects detected."
    
    return VideoAnalysisResult(
        answer=final_answer,
        confidence=0.8 if analyzed_frames else 0.0,
        relevant_frames=analyzed_frames,
        objects_detected=all_objects[:20],  # Limit to top 20
        scene_summary=scene_summary,
        temporal_events=temporal_events,
        query_type=query_type,
        spatial_match=spatial_match,
        knowledge_match=knowledge_match,
    )
