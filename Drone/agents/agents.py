"""Drone vision agents: parallel building exploration with vision LLM.

Usage:
    modal deploy agents/agents.py

Requires app.py to be deployed first (provides the ImageServer class).
"""

import base64
import json
import math
import re
import time
import uuid

import modal
from starlette.responses import StreamingResponse, Response

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, str] = {
    "qwen3-vl-30b-a3b-thinking-fp8": "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
}
DEFAULT_MODEL = "qwen3-vl-30b-a3b-thinking-fp8"
MODEL_ID = SUPPORTED_MODELS[DEFAULT_MODEL]

MAX_STEPS = 20  # synced with frontend MAX_AGENT_STEPS
STEP_SIZE = 0.1  # meters per movement step

# ---------------------------------------------------------------------------
# Direction offsets: yaw -> {direction: (dx, dy)}
# yaw=0 -> facing +x, yaw=90 -> facing +y, etc.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

from app import app as vision_app, ImageServer

model_vol = modal.Volume.from_name("vision-model-cache", create_if_missing=True)
cancel_dict = modal.Dict.from_name("vision-cancel", create_if_missing=True)

MODEL_DIR = "/model-cache"


def download_model():
    """Pre-download model weights into the volume (runs at image build)."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, local_dir=f"{MODEL_DIR}/{MODEL_ID}")


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

agent_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.11.0",
        "transformers",
        "qwen-vl-utils==0.0.14",
        "Pillow",
        "torch",
        "huggingface_hub",
    )
    .run_function(download_model, volumes={MODEL_DIR: model_vol})
)

# ---------------------------------------------------------------------------
# System prompt — uses discrete actions instead of absolute coordinates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a building-exploration agent. You navigate a discrete grid by
choosing movement actions.

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
- Before "found", verify: (1) identity matches query, (2) location in image
  is clear, (3) at least two visual attributes confirm it.
- Do NOT use "found" if the object is occluded, blurry, too far, or ambiguous.
- Do NOT revisit positions you have already been to.
- Prefer exploring new directions over going back and forth.
""".format(step_size=STEP_SIZE)


# ---------------------------------------------------------------------------
# VisionAgent class
# ---------------------------------------------------------------------------


@vision_app.cls(
    image=agent_image,
    gpu="H200",
    volumes={MODEL_DIR: model_vol},
    timeout=600,
    scaledown_window=300,
)
class VisionAgent:
    """Hosts the VLM and exposes send_agent for multi-turn exploration."""

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
        self.get_image_cls = ImageServer

    # ------------------------------------------------------------------ #

    @modal.method()
    def send_agent(
        self,
        query: str,
        start_x: float,
        start_y: float,
        start_z: float,
        start_yaw: float,
        agent_id: int,
        session_key: str,
    ) -> dict:
        """Run one exploration agent.

        Returns dict with keys:
            found, agent_id, description, final_image_b64,
            steps, trajectory, directions, filename
        """
        from vllm import SamplingParams

        x, y, z, yaw = start_x, start_y, start_z, _snap_yaw(start_yaw)
        trajectory: list[dict] = []
        history_lines: list[str] = []
        visited: set[tuple[float, float, float, float]] = set()
        last_image_b64 = ""
        last_filename = ""
        sampling = SamplingParams(temperature=0.2, max_tokens=300)

        print(f"\n{'='*60}")
        print(f"[Agent {agent_id}] START  query={query!r}")
        print(f"[Agent {agent_id}]        pos=({x:.2f}, {y:.2f}, {z:.2f})  yaw={yaw:.1f}")
        print(f"{'='*60}")

        base_sys_text = SYSTEM_PROMPT.format(query=query)

        for step in range(MAX_STEPS):
            # -- cancel check -------------------------------------------
            try:
                if cancel_dict[session_key]:
                    print(f"[Agent {agent_id}] CANCELLED at step {step}")
                    return self._result(False, agent_id, "Cancelled",
                                        last_image_b64, step, trajectory)
            except KeyError:
                pass

            # -- call getImageRemote ------------------------------------
            print(f"\n[Agent {agent_id}] Step {step}")
            print(f"[Agent {agent_id}]   getImageRemote(x={x:.4f}, y={y:.4f}, z={z:.4f}, yaw={yaw:.2f})")

            get_image = self.get_image_cls()
            result = get_image.getImageRemote.remote(x, y, z, yaw)

            img_bytes: bytes = result["image"]
            actual_x, actual_y, actual_z = result["x"], result["y"], result["z"]
            actual_yaw = result["yaw"]
            allowed = result["allowed"]
            filename = result.get("filename", "?")
            last_filename = filename
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            last_image_b64 = img_b64

            # Track visited positions
            pos_key = (round(actual_x, 1), round(actual_y, 1), round(actual_z, 1), _snap_yaw(actual_yaw))
            visited.add(pos_key)

            print(f"[Agent {agent_id}]   -> file={filename}  actual=({actual_x:.2f},{actual_y:.2f},{actual_z:.2f}) yaw={actual_yaw:.1f}  allowed={allowed}")

            trajectory.append({
                "x": x, "y": y, "z": z, "yaw": yaw,
                "step": step, "filename": filename,
            })

            # -- build system prompt with trajectory summary ------------
            sys_text = base_sys_text
            if history_lines:
                sys_text += (
                    "\n\n## Trajectory so far\n"
                    + "\n".join(history_lines)
                )

            # -- build allowed string with revisit warnings -------------
            allowed_with_revisit = _annotate_allowed_revisits(
                x, y, z, yaw, allowed, visited
            )

            # -- fresh prompt: system + current image only --------------
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_text}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Position: ({actual_x:.2f}, {actual_y:.2f}, {actual_z:.2f}), yaw={yaw:.1f}.\n"
                                f"Step {step + 1}/{MAX_STEPS}.\n"
                                f"Allowed: {json.dumps(allowed_with_revisit)}"
                            ),
                        },
                    ],
                },
            ]

            # -- VLM inference ------------------------------------------
            outputs = self.llm.chat(messages, sampling_params=sampling)
            raw_text = outputs[0].outputs[0].text.strip()
            print(f"[Agent {agent_id}]   LLM raw: {raw_text[:200]}")

            # -- parse JSON action --------------------------------------
            action = _parse_action(raw_text)
            if action is None:
                print(f"[Agent {agent_id}]   (parse failed - turning right 90)")
                history_lines.append(
                    f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                    f"- could not decide, turned right"
                )
                yaw = (yaw + 90) % 360
                continue

            print(f"[Agent {agent_id}]   Parsed action: {action}")

            if action.get("action") == "found":
                ok, validation_msg = _validate_found_action(action)
                if not ok:
                    print(f"[Agent {agent_id}]   (reject found: {validation_msg})")
                    history_lines.append(
                        f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                        f"- rejected found ({validation_msg}), turned right"
                    )
                    yaw = (yaw + 90) % 360
                    continue

                desc = str(action.get("description", ""))
                print(f"\n[Agent {agent_id}] *** FOUND at step {step}: {desc} ***")
                directions = _build_directions(trajectory)
                return self._result(True, agent_id, desc,
                                    last_image_b64, step, trajectory,
                                    directions=directions, filename=filename)

            # -- apply discrete action ----------------------------------
            act_name = action.get("action", "")
            reasoning = action.get("reasoning", "")

            new_x, new_y, new_z, new_yaw = _apply_action(
                act_name, x, y, z, yaw, allowed
            )

            if new_x == x and new_y == y and new_z == z and new_yaw == yaw:
                # Action was not allowed or unrecognized; fallback turn
                print(f"[Agent {agent_id}]   (action '{act_name}' not allowed - turning right)")
                history_lines.append(
                    f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                    f"- '{act_name}' blocked, turned right"
                )
                yaw = (yaw + 90) % 360
            else:
                history_lines.append(
                    f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} -> {act_name} - {reasoning}"
                )
                x, y, z, yaw = new_x, new_y, new_z, new_yaw

        print(f"[Agent {agent_id}] Max steps reached")
        directions = _build_directions(trajectory)
        return self._result(False, agent_id,
                            "Max steps reached without finding target",
                            last_image_b64, MAX_STEPS, trajectory,
                            directions=directions, filename=last_filename)

    # ------------------------------------------------------------------ #
    # Streaming version
    # ------------------------------------------------------------------ #

    @modal.method()
    def send_agent_streaming(
        self,
        query: str,
        start_x: float,
        start_y: float,
        start_z: float,
        start_yaw: float,
        agent_id: int,
        session_key: str,
    ):
        """Generator version of send_agent - yields step-by-step events."""
        from vllm import SamplingParams
        from io import BytesIO
        from PIL import Image

        x, y, z, yaw = start_x, start_y, start_z, _snap_yaw(start_yaw)
        trajectory = []
        history_lines = []
        visited: set[tuple[float, float, float, float]] = set()
        last_image_b64 = ""
        sampling = SamplingParams(temperature=0.2, max_tokens=300)

        base_sys_text = SYSTEM_PROMPT.format(query=query)

        for step in range(MAX_STEPS):
            # Cancel check
            try:
                if cancel_dict[session_key]:
                    return
            except KeyError:
                pass

            # Get image
            get_image = self.get_image_cls()
            result = get_image.getImageRemote.remote(x, y, z, yaw)
            img_bytes = result["image"]
            actual_x, actual_y, actual_z = result["x"], result["y"], result["z"]
            allowed = result["allowed"]
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            last_image_b64 = img_b64

            # Track visited
            pos_key = (round(actual_x, 1), round(actual_y, 1), round(actual_z, 1), _snap_yaw(result["yaw"]))
            visited.add(pos_key)

            # Downscale for streaming (256x256)
            img = Image.open(BytesIO(img_bytes))
            img_small = img.resize((256, 256), Image.LANCZOS)
            buf = BytesIO()
            img_small.save(buf, format="JPEG", quality=80)
            small_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            trajectory.append({"x": x, "y": y, "z": z, "yaw": yaw, "step": step})

            # Build system prompt with trajectory summary
            sys_text = base_sys_text
            if history_lines:
                sys_text += "\n\n## Trajectory so far\n" + "\n".join(history_lines)

            allowed_with_revisit = _annotate_allowed_revisits(
                x, y, z, yaw, allowed, visited
            )

            # Fresh prompt: system + current image only
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_text}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {
                            "type": "text",
                            "text": (
                                f"Position: ({actual_x:.2f}, {actual_y:.2f}, {actual_z:.2f}), yaw={yaw:.1f}.\n"
                                f"Step {step + 1}/{MAX_STEPS}.\n"
                                f"Allowed: {json.dumps(allowed_with_revisit)}"
                            ),
                        },
                    ],
                },
            ]

            # VLM inference
            outputs = self.llm.chat(messages, sampling_params=sampling)
            raw_text = outputs[0].outputs[0].text.strip()

            # Parse action
            action = _parse_action(raw_text)
            reasoning = ""
            action_type = "move"

            if action is None:
                reasoning = "(parse failed - turning right)"
                history_lines.append(
                    f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                    f"- could not decide, turned right"
                )
                yaw = (yaw + 90) % 360
            elif action.get("action") == "found":
                ok, validation_msg = _validate_found_action(action)
                if not ok:
                    reasoning = f"(rejected found: {validation_msg})"
                    history_lines.append(
                        f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                        f"- rejected found ({validation_msg}), turned right"
                    )
                    yaw = (yaw + 90) % 360
                else:
                    action_type = "found"
                    reasoning = action.get("description", "")
            else:
                act_name = action.get("action", "")
                reasoning = action.get("reasoning", "")
                new_x, new_y, new_z, new_yaw = _apply_action(
                    act_name, x, y, z, yaw, allowed
                )
                if new_x == x and new_y == y and new_z == z and new_yaw == yaw:
                    reasoning = f"('{act_name}' blocked - turning right)"
                    history_lines.append(
                        f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                        f"- '{act_name}' blocked, turned right"
                    )
                    yaw = (yaw + 90) % 360
                else:
                    history_lines.append(
                        f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} -> {act_name} - {reasoning}"
                    )
                    x, y, z, yaw = new_x, new_y, new_z, new_yaw

            # Yield the step event
            yield {
                "type": "agent_step",
                "agent_id": agent_id,
                "step": step,
                "total_steps": MAX_STEPS,
                "pose": {"x": x, "y": y, "z": z, "yaw": yaw},
                "image_b64": small_b64,
                "reasoning": reasoning,
                "action": action_type,
            }

            if action_type == "found":
                yield {
                    "type": "agent_found",
                    "agent_id": agent_id,
                    "description": reasoning,
                    "final_image_b64": last_image_b64,
                    "filename": result.get("filename", ""),
                    "steps": step + 1,
                    "trajectory": trajectory,
                }
                return

        # Max steps reached
        yield {
            "type": "agent_done",
            "agent_id": agent_id,
            "found": False,
            "steps": MAX_STEPS,
            "trajectory": trajectory,
        }

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _result(found, agent_id, description, image_b64, steps, trajectory,
                directions=None, filename=None):
        return {
            "found": found,
            "agent_id": agent_id,
            "description": description,
            "final_image_b64": image_b64,
            "filename": filename or "",
            "steps": steps,
            "trajectory": trajectory,
            "directions": directions or [],
        }


# ---------------------------------------------------------------------------
# Pure-function helpers (no self — testable in isolation)
# ---------------------------------------------------------------------------

def _snap_yaw(yaw: float) -> float:
    """Snap yaw to nearest cardinal direction (0, 90, 180, 270)."""
    return round(yaw / 90) % 4 * 90


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _apply_action(
    action: str,
    x: float, y: float, z: float, yaw: float,
    allowed: dict,
) -> tuple[float, float, float, float]:
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

    # Unknown action
    return x, y, z, yaw_key


def _annotate_allowed_revisits(
    x: float, y: float, z: float, yaw: float,
    allowed: dict,
    visited: set[tuple[float, float, float, float]],
) -> dict:
    """Return allowed dict with '(revisit)' warnings for directions leading
    to already-visited positions."""
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
            if dest_key in visited:
                annotated[direction] = "true (revisit)"
            else:
                annotated[direction] = True
        elif direction in offsets:
            dx, dy = offsets[direction]
            dest_key = (round(x + dx * STEP_SIZE, 1), round(y + dy * STEP_SIZE, 1), round(z, 1), yaw_key)
            if dest_key in visited:
                annotated[direction] = "true (revisit)"
            else:
                annotated[direction] = True
        else:
            annotated[direction] = is_allowed

    return annotated


def _parse_action(text: str) -> dict | None:
    """Best-effort JSON extraction from LLM output.

    Handles:
    - Qwen3 <think>...</think> blocks
    - Markdown ```json fences
    - Raw JSON
    """
    # Strip Qwen3 thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown fences if present
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _validate_found_action(action: dict) -> tuple[bool, str]:
    """Require high-confidence structured evidence before accepting found."""
    desc = str(action.get("description", "")).strip()
    if not desc:
        return False, "missing description"

    confidence = str(action.get("confidence", "")).strip().lower()
    if confidence != "high":
        return False, f'confidence must be "high" (got {confidence or "missing"})'

    evidence = action.get("evidence")
    if not isinstance(evidence, list):
        return False, "missing evidence list"

    evidence_items = [str(item).strip() for item in evidence if str(item).strip()]
    if len(evidence_items) < 2:
        return False, "need at least 2 evidence items"

    return True, "ok"


def _build_directions(trajectory: list[dict]) -> list[str]:
    """Build human-readable step-by-step directions from the trajectory."""
    if len(trajectory) < 2:
        return ["You are already at the destination."]

    YAW_NAMES = {0: "north (+x)", 90: "east (+y)", 180: "south (-x)", 270: "west (-y)"}

    directions = []
    prev = trajectory[0]
    facing = int(round(prev["yaw"])) % 360
    facing_name = YAW_NAMES.get(facing, f"{facing} deg")
    directions.append(f"Start at ({prev['x']:.1f}, {prev['y']:.1f}, {prev['z']:.1f}) facing {facing_name}.")

    i = 1
    while i < len(trajectory):
        curr = trajectory[i]
        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        dyaw = (curr["yaw"] - prev["yaw"]) % 360

        parts = []
        if dyaw == 90:
            parts.append("turn right")
        elif dyaw == 270:
            parts.append("turn left")
        elif dyaw == 180:
            parts.append("turn around")

        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0.05:
            # Count consecutive forward moves
            count = 1
            while i + count < len(trajectory):
                nxt = trajectory[i + count]
                ndx = nxt["x"] - trajectory[i + count - 1]["x"]
                ndy = nxt["y"] - trajectory[i + count - 1]["y"]
                ndyaw = (nxt["yaw"] - trajectory[i + count - 1]["yaw"]) % 360
                if ndyaw == 0 and math.sqrt(ndx**2 + ndy**2) > 0.05:
                    count += 1
                else:
                    break
            if count > 1:
                parts.append(f"walk forward {count} steps")
                i += count - 1
            else:
                parts.append("walk forward")

        if parts:
            directions.append(f"{len(directions)}. {' then '.join(parts).capitalize()}.")
        prev = trajectory[i]
        i += 1

    return directions


# ---------------------------------------------------------------------------
# spawn_agent
# ---------------------------------------------------------------------------


@vision_app.function(image=agent_image, gpu="H200", volumes={MODEL_DIR: model_vol}, timeout=600)
def spawn_agent(
    query: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
    agent_id: int,
    session_key: str,
) -> dict:
    """Run a single exploration agent as a standalone Modal function."""
    runner = VisionAgent()
    return runner.send_agent.remote(
        query=query,
        start_x=x, start_y=y, start_z=z,
        start_yaw=yaw,
        agent_id=agent_id,
        session_key=session_key,
    )


# ---------------------------------------------------------------------------
# SSE streaming endpoint
# ---------------------------------------------------------------------------


@vision_app.function(image=agent_image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def stream_agents(request: dict):
    """SSE endpoint for streaming agent exploration to the frontend."""
    query = request["query"]
    start_x = request.get("start_x", 0.0)
    start_y = request.get("start_y", 0.0)
    start_z = request.get("start_z", 0.0)
    start_yaw = request.get("start_yaw", 0.0)
    num_agents = request.get("num_agents", 2)

    def event_generator():
        session_key = str(uuid.uuid4())
        cancel_dict[session_key] = False

        # Compute diverse starting yaws for agents
        agent_configs = []
        for i in range(num_agents):
            if num_agents <= 2:
                agent_yaw = (start_yaw + i * 180) % 360
            else:
                agent_yaw = (start_yaw + i * (360 // num_agents)) % 360
            agent_configs.append((i, _snap_yaw(agent_yaw)))
            event = {
                "type": "agent_started",
                "agent_id": i,
                "start_pose": {"x": start_x, "y": start_y, "z": start_z, "yaw": agent_yaw},
            }
            yield f"data: {json.dumps(event)}\n\n"

        # Spawn agents and poll for completion
        runner = VisionAgent()
        handles = []
        for agent_id, agent_yaw in agent_configs:
            h = runner.send_agent.spawn(
                query=query,
                start_x=start_x, start_y=start_y, start_z=start_z,
                start_yaw=agent_yaw,
                agent_id=agent_id,
                session_key=session_key,
            )
            handles.append(h)

        completed = [False] * num_agents
        results = [None] * num_agents
        winner = None

        while not all(completed):
            time.sleep(2)
            for i, h in enumerate(handles):
                if completed[i]:
                    continue
                try:
                    r = h.get(timeout=0)
                except TimeoutError:
                    continue
                except Exception:
                    completed[i] = True
                    error_event = {
                        "type": "error",
                        "agent_id": i,
                        "message": "Agent encountered an error",
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"
                    continue

                completed[i] = True
                results[i] = r

                # Emit trajectory steps retroactively
                for step_data in r.get("trajectory", []):
                    step_event = {
                        "type": "agent_step",
                        "agent_id": i,
                        "step": step_data["step"],
                        "total_steps": MAX_STEPS,
                        "pose": {
                            "x": step_data["x"], "y": step_data["y"],
                            "z": step_data["z"], "yaw": step_data["yaw"],
                        },
                        "image_b64": "",
                        "reasoning": "",
                        "action": "move",
                    }
                    yield f"data: {json.dumps(step_event)}\n\n"

                if r["found"]:
                    found_event = {
                        "type": "agent_found",
                        "agent_id": i,
                        "description": r["description"],
                        "final_image_b64": r.get("final_image_b64", ""),
                        "filename": r.get("filename", ""),
                        "steps": r["steps"],
                        "trajectory": r["trajectory"],
                        "directions": r.get("directions", []),
                    }
                    yield f"data: {json.dumps(found_event)}\n\n"
                    if winner is None:
                        winner = i
                        cancel_dict[session_key] = True
                else:
                    done_event = {
                        "type": "agent_done",
                        "agent_id": i,
                        "found": False,
                        "steps": r["steps"],
                        "trajectory": r["trajectory"],
                    }
                    yield f"data: {json.dumps(done_event)}\n\n"

        # Session complete
        complete_event = {
            "type": "session_complete",
            "winner_agent_id": winner,
            "description": results[winner]["description"] if winner is not None else "No target found",
            "filename": results[winner].get("filename", "") if winner is not None else "",
            "directions": results[winner].get("directions", []) if winner is not None else [],
        }
        yield f"data: {json.dumps(complete_event)}\n\n"

        # Cleanup
        try:
            del cancel_dict[session_key]
        except KeyError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


@vision_app.function(image=agent_image)
@modal.fastapi_endpoint(method="OPTIONS")
def stream_agents_options():
    """Handle CORS preflight requests for the streaming endpoint."""
    return Response(
        content="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@vision_app.local_entrypoint()
def main(
    query: str = "find the nearest bathroom",
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
    n: int = 1,
):
    """Launch n parallel agents.  First one to find the target wins.

    Usage:
        modal run agents/agents.py --query "find the nearest bathroom" --n 3
    """
    session_key = str(uuid.uuid4())
    cancel_dict[session_key] = False

    print(f"\n{'#'*60}")
    print(f"# Launching {n} agent(s)  query={query!r}")
    print(f"# start=({x:.2f}, {y:.2f}, {z:.2f})  yaw={yaw:.1f}")
    print(f"# session={session_key}")
    print(f"{'#'*60}\n")

    # Spawn N agents with diverse starting yaws
    handles = []
    for i in range(n):
        if n <= 2:
            agent_yaw = (yaw + i * 180) % 360
        else:
            agent_yaw = (yaw + i * (360 // n)) % 360
        agent_yaw = _snap_yaw(agent_yaw)
        print(f"Spawning agent {i}  yaw={agent_yaw:.1f}")
        h = spawn_agent.spawn(
            query=query,
            x=x, y=y, z=z,
            yaw=agent_yaw,
            agent_id=i,
            session_key=session_key,
        )
        handles.append(h)

    # Poll until one agent finds the target or all finish
    completed = [False] * n
    results: list[dict | None] = [None] * n
    winner = None

    while not all(completed):
        time.sleep(2)
        for i, h in enumerate(handles):
            if completed[i]:
                continue
            try:
                r = h.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:
                print(f"Agent {i} errored: {exc}")
                completed[i] = True
                continue

            completed[i] = True
            results[i] = r
            print(f"\nAgent {i} finished: found={r['found']}  steps={r['steps']}")
            if r.get("description"):
                print(f"  description: {r['description']}")

            if r["found"] and winner is None:
                winner = r
                cancel_dict[session_key] = True
                print(f"\n>>> Agent {i} found the target - cancelling others <<<\n")

    # Cleanup
    try:
        del cancel_dict[session_key]
    except KeyError:
        pass

    # Print final result
    final = winner
    if final is None:
        final = max(
            (r for r in results if r is not None),
            key=lambda r: r["steps"],
            default=None,
        )

    if final:
        print(f"\n{'='*60}")
        if final["found"]:
            print(f"RESULT: Agent {final['agent_id']} found the target in {final['steps']} steps")
        else:
            print(f"RESULT: No agent found the target. Best effort from agent {final['agent_id']}.")
        print(f"  description: {final['description']}")
        if final.get("filename"):
            print(f"  filename: {final['filename']}")
        print(f"  trajectory points: {len(final['trajectory'])}")
        if final.get("directions"):
            print(f"\n  Directions:")
            for d in final["directions"]:
                print(f"    {d}")
        print(f"{'='*60}\n")
    else:
        print("\nAll agents failed.")
