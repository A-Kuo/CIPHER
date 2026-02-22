"""Agentic video analysis with natural-language queries.

Given a set of pre-extracted video frames and a user query, this module:
  1. Parses the query into structured analysis tokens (via query_parser).
  2. Employs an adaptive frame-sampling strategy (coarse scan → focused
     re-examination) so the VLM inspects a manageable subset of frames.
  3. For each sampled frame, builds a token-specific VLM prompt and
     collects the model's structured JSON response.
  4. Aggregates per-frame findings into a final AnalysisResult with
     timestamps, confidence, evidence chains, and a natural-language
     summary.

Designed to run on Modal (GPU container with vLLM), but the analysis
logic is pure Python and can be unit-tested locally with a mock LLM.

Usage (Modal):
    modal run agents/video_analyzer.py --query "find the fire extinguisher"

Usage (imported):
    from video_analyzer import VideoAnalyzer
    analyzer = VideoAnalyzer()
    analyzer.setup()
    result = analyzer.analyze.remote(query="where is the exit?")
"""

from __future__ import annotations

import base64
import json
import math
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional, Dict, Any

import modal

from query_parser import (
    ParsedQuery,
    QueryToken,
    parse_query,
    build_frame_analysis_prompt,
)

# ---------------------------------------------------------------------------
# Model configuration (mirrors agents.py)
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, str] = {
    "qwen3-vl-30b-a3b-thinking-fp8": "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
}
DEFAULT_MODEL = "qwen3-vl-30b-a3b-thinking-fp8"
MODEL_ID = SUPPORTED_MODELS[DEFAULT_MODEL]
MODEL_DIR = "/model-cache"

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------

COARSE_SAMPLE_COUNT = 12     # frames in the initial sweep
FOCUSED_SAMPLE_COUNT = 8     # extra frames around promising regions
MIN_RELEVANCE_THRESHOLD = 0.4
HIGH_CONFIDENCE_THRESHOLD = 0.75
MAX_ANALYSIS_FRAMES = 24     # hard cap on total VLM calls per query

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FrameFinding:
    """Single-frame analysis result."""

    frame_index: int
    timestamp_s: float
    found: bool
    description: str
    confidence: str  # low | medium | high
    evidence: List[str]
    relevance_score: float
    image_b64: str = ""

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 3),
            "found": self.found,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "relevance_score": round(self.relevance_score, 3),
        }


@dataclass
class AnalysisResult:
    """Aggregated result across all analyzed frames."""

    query: str
    parsed: dict
    total_frames: int
    frames_analyzed: int
    best_finding: Optional[FrameFinding]
    findings: List[FrameFinding] = field(default_factory=list)
    summary: str = ""
    overall_confidence: float = 0.0
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "parsed": self.parsed,
            "total_frames": self.total_frames,
            "frames_analyzed": self.frames_analyzed,
            "best_finding": self.best_finding.to_dict() if self.best_finding else None,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "overall_confidence": round(self.overall_confidence, 3),
            "elapsed_s": round(self.elapsed_s, 2),
        }


# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

from app import app as vision_app, ImageServer

model_vol = modal.Volume.from_name("vision-model-cache", create_if_missing=True)


def _download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, local_dir=f"{MODEL_DIR}/{MODEL_ID}")


analyzer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.11.0",
        "transformers",
        "qwen-vl-utils==0.0.14",
        "Pillow",
        "torch",
        "huggingface_hub",
    )
    .run_function(_download_model, volumes={MODEL_DIR: model_vol})
)


# ---------------------------------------------------------------------------
# Frame sampling strategies
# ---------------------------------------------------------------------------

def _coarse_sample_indices(total: int, count: int) -> List[int]:
    """Uniformly sample *count* frame indices from [0, total)."""
    if total <= count:
        return list(range(total))
    step = total / count
    return [min(int(i * step), total - 1) for i in range(count)]


def _focused_sample_indices(
    total: int,
    hot_indices: List[int],
    count: int,
    already_seen: set,
) -> List[int]:
    """Sample frames *around* hot (high-relevance) indices.

    For each hot index, include the two neighbours not yet analyzed.
    Fill remaining budget with midpoints between hot indices.
    """
    candidates: set = set()
    for idx in hot_indices:
        for offset in (-2, -1, 1, 2):
            nb = idx + offset
            if 0 <= nb < total and nb not in already_seen:
                candidates.add(nb)

    sorted_hot = sorted(hot_indices)
    for i in range(len(sorted_hot) - 1):
        mid = (sorted_hot[i] + sorted_hot[i + 1]) // 2
        if mid not in already_seen:
            candidates.add(mid)

    candidates -= already_seen
    result = sorted(candidates)[:count]
    return result


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _parse_vlm_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from VLM output (handles markdown fences)."""
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# VideoAnalyzer Modal class
# ---------------------------------------------------------------------------


@vision_app.cls(
    image=analyzer_image,
    gpu="H200",
    volumes={MODEL_DIR: model_vol},
    timeout=900,
    scaledown_window=300,
)
class VideoAnalyzer:
    """Agentic video analysis: NL query → structured frame-by-frame findings."""

    @modal.enter()
    def setup(self):
        from vllm import LLM

        self.llm = LLM(
            model=f"{MODEL_DIR}/{MODEL_ID}",
            trust_remote_code=True,
            max_model_len=16384,
            dtype="half",
            enable_prefix_caching=True,
        )
        self.image_server_cls = ImageServer

    # ------------------------------------------------------------------ #
    # Core analysis
    # ------------------------------------------------------------------ #

    @modal.method()
    def analyze(
        self,
        query: str,
        max_frames: int = MAX_ANALYSIS_FRAMES,
    ) -> dict:
        """Run full agentic analysis on the pre-loaded video frames.

        1. Parse query into tokens.
        2. Coarse scan: uniformly sample frames, run VLM per token.
        3. Focused scan: re-sample around high-relevance frames.
        4. Aggregate and return AnalysisResult.
        """
        from vllm import SamplingParams

        t0 = time.monotonic()
        parsed = parse_query(query)
        primary_token = parsed.tokens[0] if parsed.tokens else None
        if primary_token is None:
            return AnalysisResult(
                query=query,
                parsed=parsed.to_dict(),
                total_frames=0,
                frames_analyzed=0,
                best_finding=None,
                summary="Could not parse the query.",
            ).to_dict()

        sampling = SamplingParams(temperature=0.3, max_tokens=400)

        img_server = self.image_server_cls()
        db_size = len(img_server.db) if hasattr(img_server, "db") else 0
        if db_size == 0:
            return AnalysisResult(
                query=query,
                parsed=parsed.to_dict(),
                total_frames=0,
                frames_analyzed=0,
                best_finding=None,
                summary="No video frames loaded.",
            ).to_dict()

        frame_interval = 0.1  # seconds per frame

        # --- Phase 1: Coarse scan ---
        coarse_budget = min(COARSE_SAMPLE_COUNT, max_frames, db_size)
        coarse_indices = _coarse_sample_indices(db_size, coarse_budget)

        all_findings: List[FrameFinding] = []
        analyzed: set = set()
        prior_descriptions: List[str] = []

        print(f"[VideoAnalyzer] Query: {query!r}")
        print(f"[VideoAnalyzer] Parsed: {parsed.summary}")
        print(f"[VideoAnalyzer] Phase 1: coarse scan ({len(coarse_indices)} frames of {db_size})")

        for idx in coarse_indices:
            finding = self._analyze_frame(
                img_server, idx, db_size, frame_interval,
                primary_token, prior_descriptions, sampling,
            )
            all_findings.append(finding)
            analyzed.add(idx)
            if finding.found and finding.relevance_score >= HIGH_CONFIDENCE_THRESHOLD:
                prior_descriptions.append(
                    f"Frame {idx}: FOUND — {finding.description}"
                )
            elif finding.relevance_score > MIN_RELEVANCE_THRESHOLD:
                prior_descriptions.append(
                    f"Frame {idx}: partial — {finding.description[:100]}"
                )

        # --- Phase 2: Focused scan around promising frames ---
        hot = [
            f.frame_index
            for f in all_findings
            if f.relevance_score >= MIN_RELEVANCE_THRESHOLD
        ]
        focused_budget = min(
            FOCUSED_SAMPLE_COUNT,
            max_frames - len(analyzed),
            db_size - len(analyzed),
        )

        if hot and focused_budget > 0:
            focused_indices = _focused_sample_indices(
                db_size, hot, focused_budget, analyzed,
            )
            print(f"[VideoAnalyzer] Phase 2: focused scan ({len(focused_indices)} frames around {len(hot)} hot regions)")

            for idx in focused_indices:
                finding = self._analyze_frame(
                    img_server, idx, db_size, frame_interval,
                    primary_token, prior_descriptions, sampling,
                )
                all_findings.append(finding)
                analyzed.add(idx)
                if finding.found:
                    prior_descriptions.append(
                        f"Frame {idx}: FOUND — {finding.description}"
                    )

        # --- Secondary tokens (bonus pass for multi-intent queries) ---
        if len(parsed.tokens) > 1 and len(analyzed) < max_frames:
            secondary_token = parsed.tokens[1]
            sec_budget = min(6, max_frames - len(analyzed))
            sec_hot = [f.frame_index for f in all_findings if f.found]
            if sec_hot:
                sec_indices = sec_hot[:sec_budget]
            else:
                sec_indices = _coarse_sample_indices(db_size, sec_budget)

            print(f"[VideoAnalyzer] Phase 3: secondary intent ({secondary_token.intent}, {len(sec_indices)} frames)")
            for idx in sec_indices:
                finding = self._analyze_frame(
                    img_server, idx, db_size, frame_interval,
                    secondary_token, prior_descriptions, sampling,
                )
                all_findings.append(finding)
                analyzed.add(idx)

        # --- Aggregate ---
        all_findings.sort(key=lambda f: f.relevance_score, reverse=True)
        best = all_findings[0] if all_findings else None
        positive = [f for f in all_findings if f.found]

        summary = self._build_summary(query, parsed, all_findings, positive, db_size)
        overall_conf = (
            max(f.relevance_score for f in all_findings) if all_findings else 0.0
        )

        elapsed = time.monotonic() - t0
        print(f"[VideoAnalyzer] Done in {elapsed:.1f}s — {len(all_findings)} frames analyzed, {len(positive)} positive")

        result = AnalysisResult(
            query=query,
            parsed=parsed.to_dict(),
            total_frames=db_size,
            frames_analyzed=len(analyzed),
            best_finding=best,
            findings=positive if positive else all_findings[:5],
            summary=summary,
            overall_confidence=overall_conf,
            elapsed_s=elapsed,
        )
        return result.to_dict()

    # ------------------------------------------------------------------ #
    # Streaming analysis — yields per-frame events
    # ------------------------------------------------------------------ #

    @modal.method()
    def analyze_streaming(
        self,
        query: str,
        max_frames: int = MAX_ANALYSIS_FRAMES,
    ):
        """Generator that yields analysis events frame by frame.

        Event types:
            analysis_start   — initial metadata
            frame_analyzed   — per-frame finding
            analysis_complete — final summary
        """
        from vllm import SamplingParams

        t0 = time.monotonic()
        parsed = parse_query(query)
        primary_token = parsed.tokens[0] if parsed.tokens else None

        yield {
            "type": "analysis_start",
            "query": query,
            "parsed": parsed.to_dict(),
        }

        if primary_token is None:
            yield {
                "type": "analysis_complete",
                "summary": "Could not parse the query.",
                "findings": [],
                "overall_confidence": 0.0,
            }
            return

        sampling = SamplingParams(temperature=0.3, max_tokens=400)
        img_server = self.image_server_cls()
        db_size = len(img_server.db) if hasattr(img_server, "db") else 0

        if db_size == 0:
            yield {
                "type": "analysis_complete",
                "summary": "No video frames loaded.",
                "findings": [],
                "overall_confidence": 0.0,
            }
            return

        frame_interval = 0.1
        coarse_indices = _coarse_sample_indices(
            db_size, min(COARSE_SAMPLE_COUNT, max_frames, db_size),
        )
        all_findings: List[FrameFinding] = []
        analyzed: set = set()
        prior_descriptions: List[str] = []

        for idx in coarse_indices:
            finding = self._analyze_frame(
                img_server, idx, db_size, frame_interval,
                primary_token, prior_descriptions, sampling,
            )
            all_findings.append(finding)
            analyzed.add(idx)

            yield {
                "type": "frame_analyzed",
                "finding": finding.to_dict(),
                "frames_analyzed": len(analyzed),
                "total_frames": db_size,
            }

            if finding.found and finding.relevance_score >= HIGH_CONFIDENCE_THRESHOLD:
                prior_descriptions.append(
                    f"Frame {idx}: FOUND — {finding.description}"
                )

        hot = [
            f.frame_index for f in all_findings
            if f.relevance_score >= MIN_RELEVANCE_THRESHOLD
        ]
        focused_budget = min(
            FOCUSED_SAMPLE_COUNT, max_frames - len(analyzed), db_size - len(analyzed),
        )
        if hot and focused_budget > 0:
            focused_indices = _focused_sample_indices(
                db_size, hot, focused_budget, analyzed,
            )
            for idx in focused_indices:
                finding = self._analyze_frame(
                    img_server, idx, db_size, frame_interval,
                    primary_token, prior_descriptions, sampling,
                )
                all_findings.append(finding)
                analyzed.add(idx)
                yield {
                    "type": "frame_analyzed",
                    "finding": finding.to_dict(),
                    "frames_analyzed": len(analyzed),
                    "total_frames": db_size,
                }

        all_findings.sort(key=lambda f: f.relevance_score, reverse=True)
        positive = [f for f in all_findings if f.found]
        summary = self._build_summary(query, parsed, all_findings, positive, db_size)
        elapsed = time.monotonic() - t0

        yield {
            "type": "analysis_complete",
            "summary": summary,
            "findings": [f.to_dict() for f in (positive or all_findings[:5])],
            "overall_confidence": max(
                (f.relevance_score for f in all_findings), default=0.0,
            ),
            "elapsed_s": round(elapsed, 2),
            "frames_analyzed": len(analyzed),
        }

    # ------------------------------------------------------------------ #
    # Internal: analyze a single frame
    # ------------------------------------------------------------------ #

    def _analyze_frame(
        self,
        img_server,
        frame_idx: int,
        total_frames: int,
        frame_interval: float,
        token: QueryToken,
        prior_findings: List[str],
        sampling,
    ) -> FrameFinding:
        """Run VLM on one frame and return a FrameFinding."""
        from PIL import Image

        entry = img_server.db[frame_idx]
        img_bytes: bytes = entry["jpeg"]
        timestamp_s = frame_idx * frame_interval

        img_b64 = base64.b64encode(img_bytes).decode("ascii")

        prompt_text = build_frame_analysis_prompt(
            token, frame_idx, total_frames, prior_findings,
        )

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a video analysis agent.  You examine individual "
                            "frames from a video to answer the user's query.  Always "
                            "respond with ONLY a JSON object — no markdown, no extra text."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        outputs = self.llm.chat(messages, sampling_params=sampling)
        raw = outputs[0].outputs[0].text.strip()

        parsed = _parse_vlm_json(raw)
        if parsed is None:
            return FrameFinding(
                frame_index=frame_idx,
                timestamp_s=timestamp_s,
                found=False,
                description=raw[:200],
                confidence="low",
                evidence=[],
                relevance_score=0.1,
            )

        return FrameFinding(
            frame_index=frame_idx,
            timestamp_s=timestamp_s,
            found=bool(parsed.get("found", False)),
            description=str(parsed.get("description", "")),
            confidence=str(parsed.get("confidence", "low")).lower(),
            evidence=parsed.get("evidence", []) if isinstance(parsed.get("evidence"), list) else [],
            relevance_score=float(parsed.get("relevance_score", 0.0)),
        )

    # ------------------------------------------------------------------ #
    # Summary builder
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_summary(
        query: str,
        parsed: ParsedQuery,
        all_findings: List[FrameFinding],
        positive: List[FrameFinding],
        total_frames: int,
    ) -> str:
        """Build a human-readable summary of the analysis."""
        if not all_findings:
            return f"No frames were analyzed for query: {query}"

        n_analyzed = len(all_findings)
        n_positive = len(positive)

        if n_positive == 0:
            best = all_findings[0]
            return (
                f"Analyzed {n_analyzed} of {total_frames} frames.  "
                f"No definitive match found for \"{query}\".  "
                f"Closest observation (frame {best.frame_index}, "
                f"t={best.timestamp_s:.1f}s): {best.description}"
            )

        best = positive[0]
        if n_positive == 1:
            return (
                f"Found match in frame {best.frame_index} "
                f"(t={best.timestamp_s:.1f}s, confidence={best.confidence}).  "
                f"{best.description}"
            )

        time_range = (
            f"{positive[-1].timestamp_s:.1f}s–{positive[0].timestamp_s:.1f}s"
            if positive[-1].timestamp_s != positive[0].timestamp_s
            else f"{best.timestamp_s:.1f}s"
        )
        return (
            f"Found {n_positive} matching frames across {time_range} "
            f"(analyzed {n_analyzed} of {total_frames}).  "
            f"Best match at frame {best.frame_index} "
            f"(t={best.timestamp_s:.1f}s, confidence={best.confidence}): "
            f"{best.description}"
        )


# ---------------------------------------------------------------------------
# SSE streaming endpoint for video analysis
# ---------------------------------------------------------------------------


@vision_app.function(image=analyzer_image, timeout=900)
@modal.fastapi_endpoint(method="POST")
def analyze_video(request: dict):
    """SSE endpoint: analyze video frames with a natural-language query.

    Request body:
        {"query": "where is the fire extinguisher?", "max_frames": 20}

    Streams SSE events:
        analysis_start, frame_analyzed (×N), analysis_complete
    """
    from starlette.responses import StreamingResponse

    query = request.get("query", "")
    max_frames = request.get("max_frames", MAX_ANALYSIS_FRAMES)

    def event_gen():
        analyzer = VideoAnalyzer()
        for event in analyzer.analyze_streaming.remote_gen(
            query=query, max_frames=max_frames,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


@vision_app.function(image=analyzer_image)
@modal.fastapi_endpoint(method="OPTIONS")
def analyze_video_options():
    """CORS preflight for the video analysis endpoint."""
    from starlette.responses import Response
    return Response(
        content="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


# ---------------------------------------------------------------------------
# Synchronous endpoint (non-streaming)
# ---------------------------------------------------------------------------


@vision_app.function(image=analyzer_image, timeout=900)
@modal.fastapi_endpoint(method="POST")
def analyze_video_sync(request: dict):
    """Non-streaming analysis endpoint.  Returns the full result as JSON.

    Request body:
        {"query": "where is the fire extinguisher?", "max_frames": 20}
    """
    from starlette.responses import JSONResponse

    query = request.get("query", "")
    max_frames = request.get("max_frames", MAX_ANALYSIS_FRAMES)

    analyzer = VideoAnalyzer()
    result = analyzer.analyze.remote(query=query, max_frames=max_frames)

    return JSONResponse(
        content=result,
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@vision_app.local_entrypoint()
def main(
    query: str = "describe what you see in the video",
    max_frames: int = MAX_ANALYSIS_FRAMES,
):
    """Run video analysis from the command line.

    Usage:
        modal run agents/video_analyzer.py --query "find the fire extinguisher"
    """
    print(f"\n{'#' * 60}")
    print(f"# Video Analysis")
    print(f"# Query: {query!r}")
    print(f"# Max frames: {max_frames}")
    print(f"{'#' * 60}\n")

    analyzer = VideoAnalyzer()
    result = analyzer.analyze.remote(query=query, max_frames=max_frames)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {result['summary']}")
    print(f"Confidence: {result['overall_confidence']:.2f}")
    print(f"Frames analyzed: {result['frames_analyzed']} / {result['total_frames']}")
    print(f"Elapsed: {result['elapsed_s']:.1f}s")
    if result["best_finding"]:
        bf = result["best_finding"]
        print(f"\nBest finding:")
        print(f"  Frame {bf['frame_index']} (t={bf['timestamp_s']:.1f}s)")
        print(f"  {bf['description']}")
        print(f"  Confidence: {bf['confidence']}")
        print(f"  Evidence: {bf['evidence']}")
    print(f"{'=' * 60}\n")
