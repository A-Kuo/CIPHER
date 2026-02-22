"""Agent navigation logic using Llama Vision + YOLO.

This module handles the exploration logic for autonomous agents.
Uses discrete actions (forward/backward/left/right/turnLeft/turnRight)
instead of absolute coordinates for reliable navigation.
"""

import base64
import json
import math
import re
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

STEP_SIZE = 0.1  # meters per movement step

DIRECTION_OFFSETS = {
    0:   {"forward": (1, 0),  "backward": (-1, 0),  "left": (0, -1), "right": (0, 1)},
    90:  {"forward": (0, 1),  "backward": (0, -1),  "left": (1, 0),  "right": (-1, 0)},
    180: {"forward": (-1, 0), "backward": (1, 0),   "left": (0, 1),  "right": (0, -1)},
    270: {"forward": (0, -1), "backward": (0, 1),   "left": (-1, 0), "right": (1, 0)},
}

BOUNDS = {
    "x": (-200, 200),
    "y": (-200, 200),
    "z": (-100, 100),
}

SYSTEM_PROMPT = """\
You are a building-exploration agent. You navigate a discrete grid by choosing movement actions.

## Coordinate system
- The world uses (x, y, z) coordinates with step size {step_size} meters.
- **yaw** is your facing direction in degrees: 0, 90, 180, or 270 only.

## Available actions
Each turn you will see which directions are **allowed** (true/false).
You MUST only choose an allowed action.

Actions:
- **forward**: move {step_size}m in your facing direction
- **backward**: move {step_size}m opposite to your facing direction
- **left**: strafe {step_size}m to your left
- **right**: strafe {step_size}m to your right
- **turnLeft**: rotate yaw by -90 degrees (position stays the same)
- **turnRight**: rotate yaw by +90 degrees (position stays the same)

## Your task
The user asked: "{{query}}"
Explore the building to find what they asked for.

## How to respond
Output **only** a JSON object (no markdown, no extra text):

If you have NOT found the target:
{{"action": "<forward|backward|left|right|turnLeft|turnRight>", "reasoning": "<1-2 sentences>"}}

If you CAN SEE the target in the current image:
{{"action": "found", "description": "<what and where you see it>", "confidence": "<low|medium|high>", "evidence": ["<visual cue 1>", "<visual cue 2>"]}}

## Rules
- Use "found" ONLY when confidence is HIGH with direct visual evidence.
- Do NOT revisit positions you have already been to.
- Prefer exploring new directions over going back and forth.
""".format(step_size=STEP_SIZE)


def _snap_yaw(yaw: float) -> float:
    """Snap yaw to nearest cardinal direction (0, 90, 180, 270)."""
    return round(yaw / 90) % 4 * 90


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _apply_action(
    action: str,
    x: float, y: float, z: float, yaw: float,
    allowed: Dict,
) -> Tuple[float, float, float, float]:
    """Compute new (x, y, z, yaw) from a discrete action.

    Returns the original position unchanged if the action is not allowed.
    """
    yaw_key = int(round(yaw)) % 360

    if action == "turnLeft":
        if allowed.get("turnLeft", False):
            return x, y, z, (yaw_key - 90) % 360
        return x, y, z, yaw_key

    if action == "turnRight":
        if allowed.get("turnRight", False):
            return x, y, z, (yaw_key + 90) % 360
        return x, y, z, yaw_key

    if action in ("forward", "backward", "left", "right"):
        if not allowed.get(action, False):
            return x, y, z, yaw_key
        offsets = DIRECTION_OFFSETS.get(yaw_key, DIRECTION_OFFSETS[0])
        dx, dy = offsets.get(action, (0, 0))
        new_x = _clamp(x + dx * STEP_SIZE, *BOUNDS["x"])
        new_y = _clamp(y + dy * STEP_SIZE, *BOUNDS["y"])
        return new_x, new_y, z, yaw_key

    return x, y, z, yaw_key


def _annotate_allowed_revisits(
    x: float, y: float, z: float, yaw: float,
    allowed: Dict,
    visited: Set[Tuple[float, float, float, float]],
) -> Dict:
    """Return allowed dict with '(revisit)' warnings."""
    yaw_key = int(round(yaw)) % 360
    offsets = DIRECTION_OFFSETS.get(yaw_key, DIRECTION_OFFSETS[0])
    annotated = {}

    for direction, is_allowed in allowed.items():
        if not is_allowed:
            annotated[direction] = False
            continue

        if direction in ("turnLeft", "turnRight"):
            turn_yaw = (yaw_key - 90) % 360 if direction == "turnLeft" else (yaw_key + 90) % 360
            dest_key = (round(x, 1), round(y, 1), round(z, 1), turn_yaw)
            annotated[direction] = "true (revisit)" if dest_key in visited else True
        elif direction in offsets:
            dx, dy = offsets[direction]
            dest_key = (round(x + dx * STEP_SIZE, 1), round(y + dy * STEP_SIZE, 1), round(z, 1), yaw_key)
            annotated[direction] = "true (revisit)" if dest_key in visited else True
        else:
            annotated[direction] = is_allowed

    return annotated


def _parse_action(response: str) -> Optional[Dict]:
    """Parse JSON action from LLM response. Handles <think> tags and markdown fences."""
    # Strip Qwen3 / thinking tags
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


class AgentRunner:
    """Runs exploration agent with Llama Vision + YOLO."""

    def __init__(self, model_manager, image_db):
        self.models = model_manager
        self.image_db = image_db

    def run_agent(
        self,
        query: str,
        start_x: float,
        start_y: float,
        start_z: float,
        start_yaw: float,
        agent_id: int,
        max_steps: int = 20,
    ):
        """Run one exploration agent. Yields step events."""
        x, y, z, yaw = start_x, start_y, start_z, _snap_yaw(start_yaw)
        trajectory = []
        visited: Set[Tuple[float, float, float, float]] = set()

        for step in range(max_steps):
            # Get current frame
            idx = self.image_db.find_best(x, y, z, yaw)
            frame_data = self.image_db.db[idx]

            # Load image
            image = Image.open(frame_data["path"])

            # Get allowed directions
            allowed = self.image_db.check_allowed(
                frame_data["x"], frame_data["y"], frame_data["z"], yaw
            )

            # Track visited
            pos_key = (round(frame_data["x"], 1), round(frame_data["y"], 1), round(frame_data["z"], 1), _snap_yaw(frame_data["yaw"]))
            visited.add(pos_key)

            # Run YOLO detection
            try:
                detections = self.models.detect_objects(image)
            except Exception:
                detections = []
            detected_objects = [d["class"] for d in detections if d.get("confidence", 0) > 0.5]

            # Annotate allowed with revisit warnings
            allowed_annotated = _annotate_allowed_revisits(x, y, z, yaw, allowed, visited)

            # Llama Vision inference (with YOLO-only fallback)
            action = None
            try:
                prompt = self._build_prompt(
                    query, step, max_steps, x, y, z, yaw,
                    allowed_annotated, detected_objects, len(visited)
                )
                response = self.models.infer_llama(image, prompt)
                action = _parse_action(response)
            except Exception:
                # Fallback: YOLO-only "found" or turn to explore
                q_lower = query.lower()
                if any(
                    cls.lower() in q_lower or q_lower in cls.lower()
                    for cls in detected_objects
                ):
                    action = {
                        "action": "found",
                        "description": f"Found {', '.join(detected_objects)}",
                        "confidence": "high",
                        "evidence": [f"YOLO detected: {cls}" for cls in detected_objects[:2]] or ["YOLO detection"],
                    }
                else:
                    # Pick first allowed non-revisit direction
                    action = self._fallback_action(allowed, visited, x, y, z, yaw)

            # Downscale image for streaming
            img_small = image.resize((256, 256), Image.LANCZOS)
            buf = BytesIO()
            img_small.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            # Determine action result
            act_name = action.get("action", "forward") if action else "turnRight"
            reasoning = action.get("reasoning", "") if action else "Exploring (vision model unavailable)."

            step_event = {
                "type": "agent_step",
                "agent_id": agent_id,
                "step": step,
                "total_steps": max_steps,
                "pose": {"x": x, "y": y, "z": z, "yaw": yaw},
                "image_b64": img_b64,
                "reasoning": reasoning,
                "action": "found" if act_name == "found" else "move",
            }

            yield step_event

            # Check if found
            if action and action.get("action") == "found":
                # Validate found action
                desc = action.get("description", "Target found")
                confidence = str(action.get("confidence", "")).lower()
                evidence = action.get("evidence", [])
                if confidence != "high" or not isinstance(evidence, list) or len(evidence) < 2:
                    # Reject low-confidence found; turn instead
                    yaw = (yaw + 90) % 360
                    trajectory.append({"x": x, "y": y, "z": z, "yaw": yaw, "step": step})
                    continue

                # Full resolution image for final result
                buf_full = BytesIO()
                image.save(buf_full, format="JPEG", quality=95)
                img_full_b64 = base64.b64encode(buf_full.getvalue()).decode("ascii")

                found_event = {
                    "type": "agent_found",
                    "agent_id": agent_id,
                    "description": desc,
                    "final_image_b64": img_full_b64,
                    "steps": step + 1,
                    "trajectory": trajectory,
                }
                yield found_event
                return

            # Apply discrete action
            new_x, new_y, new_z, new_yaw = _apply_action(
                act_name, x, y, z, yaw, allowed
            )

            if new_x == x and new_y == y and new_z == z and new_yaw == yaw:
                # Action not allowed; fallback turn
                yaw = (yaw + 90) % 360
            else:
                x, y, z, yaw = new_x, new_y, new_z, new_yaw

            trajectory.append({"x": x, "y": y, "z": z, "yaw": yaw, "step": step})

        # Max steps reached
        done_event = {
            "type": "agent_done",
            "agent_id": agent_id,
            "found": False,
            "steps": max_steps,
            "trajectory": trajectory,
        }
        yield done_event

    def _build_prompt(
        self,
        query: str,
        step: int,
        max_steps: int,
        x: float,
        y: float,
        z: float,
        yaw: float,
        allowed: Dict,
        detected_objects: List[str],
        visited_count: int,
    ) -> str:
        """Build prompt for Llama Vision."""
        system = SYSTEM_PROMPT.format(query=query)

        context = f"""
Current position: ({x:.2f}, {y:.2f}, {z:.2f}), yaw={yaw:.1f} deg
Step {step + 1}/{max_steps}

Allowed actions: {json.dumps(allowed)}

Objects detected by YOLO: {', '.join(detected_objects) if detected_objects else 'none'}

Visited {visited_count} positions so far.

Look at the image carefully. Choose one allowed action. Respond with JSON only.
"""

        return system + context

    def _fallback_action(
        self,
        allowed: Dict,
        visited: Set,
        x: float, y: float, z: float, yaw: float,
    ) -> Dict:
        """Pick the best fallback action when vision model is unavailable."""
        yaw_key = int(round(yaw)) % 360

        # Prefer forward if allowed and not a revisit
        for act in ("forward", "turnRight", "turnLeft", "right", "left", "backward"):
            if not allowed.get(act, False):
                continue
            nx, ny, nz, nyaw = _apply_action(act, x, y, z, yaw_key, allowed)
            dest_key = (round(nx, 1), round(ny, 1), round(nz, 1), nyaw)
            if dest_key not in visited:
                return {"action": act, "reasoning": "Exploring (vision model unavailable)."}

        # Everything is a revisit; just turn
        return {"action": "turnRight", "reasoning": "All directions visited, turning."}

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(value, max_val))
