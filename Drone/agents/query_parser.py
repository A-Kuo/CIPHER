"""Natural-language query parser for agentic video analysis.

Decomposes a free-form user query into structured *analysis tokens* that
the VLM-based video analyzer can act on independently.  Each token carries
an intent type, focus keywords, and a pre-built VLM prompt fragment so
downstream modules never need to re-interpret raw text.

No external dependencies — runs anywhere (Modal, local backend, CLI).

Supported intent types
----------------------
- OBJECT_SEARCH   : find / locate a specific object or entity
- SCENE_DESCRIBE  : describe what is visible in the scene
- ACTION_DETECT   : detect an activity or event happening
- SPATIAL_REASON  : answer questions about spatial relationships
- TEMPORAL_TRACK  : track something across time / frames
- COUNT           : count instances of something
- SAFETY_ASSESS   : assess safety / hazard / structural risk
- OPEN_QUERY      : catch-all for queries that don't match above
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryToken:
    """A single analysis intent extracted from the user query."""

    intent: str
    keywords: List[str]
    original_fragment: str
    vlm_prompt: str
    priority: int = 1  # 1 = highest, 3 = lowest
    spatial_hint: Optional[str] = None  # e.g. "left", "near the door"
    temporal_hint: Optional[str] = None  # e.g. "beginning", "after 10s"

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "keywords": self.keywords,
            "original_fragment": self.original_fragment,
            "vlm_prompt": self.vlm_prompt,
            "priority": self.priority,
            "spatial_hint": self.spatial_hint,
            "temporal_hint": self.temporal_hint,
        }


@dataclass
class ParsedQuery:
    """Result of parsing a user query."""

    raw_query: str
    tokens: List[QueryToken] = field(default_factory=list)
    primary_intent: str = "OPEN_QUERY"
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "raw_query": self.raw_query,
            "tokens": [t.to_dict() for t in self.tokens],
            "primary_intent": self.primary_intent,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Pattern table — order matters (first match wins for primary intent)
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: List[tuple] = [
    # (intent, compiled_regex, priority, prompt_template)
    (
        "SAFETY_ASSESS",
        re.compile(
            r"(?:is\s+it\s+safe|hazard|danger|risk|structural\s+(?:damage|risk|integrity)|"
            r"fire|smoke|gas\s+leak|collapse|crack|unsafe|emergency)",
            re.I,
        ),
        1,
        (
            "Analyze this frame for safety hazards.  Identify any structural damage, "
            "fire, smoke, gas indicators, collapse risk, cracks, or unsafe conditions.  "
            "Focus on: {keywords}.  "
            "Report what you see, severity (low/medium/high), and recommended action."
        ),
    ),
    (
        "OBJECT_SEARCH",
        re.compile(
            r"(?:where\s+is|find|locate|show\s+me|spot|identify|look\s+for|"
            r"nearest|search\s+for|is\s+there\s+(?:a|an|any))",
            re.I,
        ),
        1,
        (
            "Search this frame for the following: {keywords}.  "
            "If found, describe its exact location in the frame (left/center/right, "
            "near/far, relative to landmarks), its visual attributes (color, shape, size), "
            "and your confidence (low/medium/high).  If not found, say so clearly."
        ),
    ),
    (
        "COUNT",
        re.compile(
            r"(?:how\s+many|count|number\s+of|total\s+(?:of|number))",
            re.I,
        ),
        1,
        (
            "Count the number of {keywords} visible in this frame.  "
            "For each instance, note its approximate location.  "
            "Give a total count and confidence level."
        ),
    ),
    (
        "ACTION_DETECT",
        re.compile(
            r"(?:what\s+(?:is|are)\s+(?:\w+\s+)?(?:doing|happening)|"
            r"detect\s+(?:activity|motion|movement)|"
            r"is\s+(?:anyone|someone|somebody)\s+(?:\w+ing))",
            re.I,
        ),
        2,
        (
            "Analyze this frame for activities or events.  "
            "Focus on: {keywords}.  "
            "Describe any actions, movements, or events you observe, "
            "who/what is involved, and the context."
        ),
    ),
    (
        "SPATIAL_REASON",
        re.compile(
            r"(?:how\s+far|distance|between|next\s+to|near|behind|"
            r"in\s+front\s+of|above|below|left\s+of|right\s+of|"
            r"which\s+way|direction\s+to|path\s+to|route\s+to)",
            re.I,
        ),
        2,
        (
            "Analyze the spatial layout in this frame.  "
            "Focus on: {keywords}.  "
            "Describe relative positions, distances, and spatial relationships "
            "between objects or landmarks you can see."
        ),
    ),
    (
        "TEMPORAL_TRACK",
        re.compile(
            r"(?:track|follow|over\s+time|across\s+frames|"
            r"when\s+(?:does|did)|at\s+what\s+(?:time|point)|"
            r"first\s+(?:appear|time|seen)|last\s+(?:seen|appear))",
            re.I,
        ),
        2,
        (
            "Examine this frame as part of a temporal sequence.  "
            "Focus on: {keywords}.  "
            "Note the presence, position, and state of the target.  "
            "This will be correlated across multiple frames."
        ),
    ),
    (
        "SCENE_DESCRIBE",
        re.compile(
            r"(?:describe|what\s+(?:is|do\s+you\s+see)|"
            r"tell\s+me\s+about|overview|summarize|summary\s+of|"
            r"what(?:'s|\s+is)\s+(?:in|visible|here|there))",
            re.I,
        ),
        2,
        (
            "Describe what you see in this frame in detail.  "
            "Include: the environment type (indoor/outdoor, room type), "
            "key objects and their positions, lighting conditions, "
            "any text or signage, and overall scene context.  "
            "Focus especially on: {keywords}."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "there", "their",
    "they", "them", "it", "its", "that", "this", "these", "those", "what",
    "which", "who", "whom", "where", "when", "how", "why", "if", "or",
    "and", "but", "not", "no", "nor", "so", "than", "too", "very", "just",
    "about", "above", "after", "again", "all", "also", "any", "as", "at",
    "back", "because", "before", "between", "both", "by", "come", "each",
    "for", "from", "get", "give", "go", "good", "he", "her", "here",
    "him", "his", "i", "in", "into", "know", "like", "make", "me", "most",
    "my", "new", "of", "on", "one", "only", "other", "our", "out", "over",
    "say", "she", "some", "take", "tell", "to", "up", "us", "use", "want",
    "we", "well", "with", "you", "your",
    "find", "show", "locate", "search", "look", "see", "detect", "identify",
    "describe", "count", "track", "follow", "many", "much", "nearest",
})

_SPATIAL_WORDS = re.compile(
    r"\b(left|right|center|top|bottom|near|far|front|behind|above|below|"
    r"next\s+to|close\s+to|beside|corner|edge|middle|ceiling|floor|wall|door)\b",
    re.I,
)

_TEMPORAL_WORDS = re.compile(
    r"\b(beginning|start|end|first|last|before|after|during|"
    r"(\d+)\s*(?:s|sec|seconds?|min|minutes?)|"
    r"early|late|middle|throughout)\b",
    re.I,
)


def _extract_keywords(text: str) -> List[str]:
    """Pull meaningful content words from the query."""
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _extract_spatial_hint(text: str) -> Optional[str]:
    m = _SPATIAL_WORDS.search(text)
    return m.group(0).strip() if m else None


def _extract_temporal_hint(text: str) -> Optional[str]:
    m = _TEMPORAL_WORDS.search(text)
    return m.group(0).strip() if m else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_query(query: str) -> ParsedQuery:
    """Parse a natural-language query into structured analysis tokens.

    Returns a ``ParsedQuery`` with one or more ``QueryToken`` items.
    The first token is always the highest-priority match; an ``OPEN_QUERY``
    fallback is appended if no specific pattern matched.
    """
    query = (query or "").strip()
    if not query:
        return ParsedQuery(
            raw_query=query,
            tokens=[],
            primary_intent="OPEN_QUERY",
            summary="Empty query",
        )

    keywords = _extract_keywords(query)
    keyword_str = ", ".join(keywords) if keywords else query
    spatial = _extract_spatial_hint(query)
    temporal = _extract_temporal_hint(query)

    tokens: List[QueryToken] = []
    seen_intents: set = set()

    for intent, pattern, priority, prompt_tpl in _INTENT_PATTERNS:
        if pattern.search(query):
            if intent in seen_intents:
                continue
            seen_intents.add(intent)
            tokens.append(
                QueryToken(
                    intent=intent,
                    keywords=keywords,
                    original_fragment=query,
                    vlm_prompt=prompt_tpl.format(keywords=keyword_str),
                    priority=priority,
                    spatial_hint=spatial,
                    temporal_hint=temporal,
                )
            )

    if not tokens:
        tokens.append(
            QueryToken(
                intent="OPEN_QUERY",
                keywords=keywords,
                original_fragment=query,
                vlm_prompt=(
                    f"The user asked: \"{query}\"\n"
                    "Analyze this frame and answer the question.  "
                    "Be specific about what you see and provide evidence for your answer."
                ),
                priority=3,
                spatial_hint=spatial,
                temporal_hint=temporal,
            )
        )

    tokens.sort(key=lambda t: t.priority)

    primary = tokens[0].intent if tokens else "OPEN_QUERY"
    summary = (
        f"{primary}: {keyword_str}"
        if keywords
        else f"{primary}: {query[:80]}"
    )

    return ParsedQuery(
        raw_query=query,
        tokens=tokens,
        primary_intent=primary,
        summary=summary,
    )


def build_frame_analysis_prompt(
    token: QueryToken,
    frame_index: int,
    total_frames: int,
    prior_findings: Optional[List[str]] = None,
) -> str:
    """Build a complete VLM prompt for analyzing a single video frame.

    Incorporates the token's VLM prompt, frame context, and any accumulated
    findings from prior frames so the model can reason across the sequence.
    """
    parts = [token.vlm_prompt]

    parts.append(
        f"\nFrame {frame_index + 1} of {total_frames}."
    )

    if token.temporal_hint:
        parts.append(f"Temporal context: {token.temporal_hint}.")

    if prior_findings:
        parts.append(
            "\nPrior observations from earlier frames:\n"
            + "\n".join(f"- {f}" for f in prior_findings[-5:])
        )

    parts.append(
        "\nRespond with ONLY a JSON object:\n"
        '{"found": true/false, '
        '"description": "<what you see>", '
        '"confidence": "<low|medium|high>", '
        '"evidence": ["<detail 1>", "<detail 2>"], '
        '"relevance_score": <0.0-1.0>}'
    )

    return "\n".join(parts)
