from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


SYSTEM_PROMPT = """Kamu adalah task planner untuk robot manipulasi meja Franka Panda.

Kamu menerima CONTEXT.MD yang menggambarkan kondisi scene.

Tugasmu: hasilkan SATU file TaskPlan JSON yang valid.

ATURAN KERAS:
1. Gunakan HANYA object_id yang ada di context.
2. Gunakan HANYA predicate dari allowed_predicates.
3. Jangan tentukan joint angles, trajectory, atau pose IK.
4. Jangan geser atau sentuh obstacle yang fragile.
5. Jika task tidak mungkin (object tidak reachable, dsb): output {"status": "UNSAT", "reason": "..."}.
6. Jika context ambigu: output {"status": "NEEDS_CLARIFICATION", "missing": [...]}.
7. Output harus JSON valid. Tidak ada Markdown. Tidak ada komentar.
"""


class PlanGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    timeout_seconds: float = 90.0

    @classmethod
    def from_env(cls) -> "LLMSettings":
        if load_dotenv is not None:
            load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        if provider not in {"openai", "anthropic", "local"}:
            raise PlanGenerationError(
                "LLM_PROVIDER must be one of: openai, anthropic, local"
            )
        model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if provider == "openai" and not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if provider != "local" and not api_key:
            raise PlanGenerationError(
                "LLM_API_KEY is required for non-local plan generation"
            )
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL") or None,
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "90")),
        )


def request_task_plan(context_text: str, settings: LLMSettings) -> dict[str, Any]:
    if settings.provider == "anthropic":
        url = settings.base_url or "https://api.anthropic.com/v1/messages"
        payload = {
            "model": settings.model,
            "max_tokens": 4096,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": context_text}],
        }
        headers = {
            "content-type": "application/json",
            "x-api-key": settings.api_key,
            "anthropic-version": "2023-06-01",
        }
    else:
        default_url = (
            "http://localhost:11434/v1/chat/completions"
            if settings.provider == "local"
            else "https://api.openai.com/v1/chat/completions"
        )
        url = settings.base_url or default_url
        payload = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context_text},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        headers = {"content-type": "application/json"}
        if settings.api_key:
            headers["authorization"] = f"Bearer {settings.api_key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=settings.timeout_seconds
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise PlanGenerationError(
            f"LLM request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise PlanGenerationError(f"LLM request failed: {exc}") from exc

    try:
        if settings.provider == "anthropic":
            content = body["content"][0]["text"]
        else:
            content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PlanGenerationError("LLM response does not contain JSON text") from exc
    return parse_llm_json(content)


def parse_llm_json(content: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanGenerationError(
            f"LLM output is not valid JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise PlanGenerationError("LLM output must be a JSON object")
    return payload


def build_task_prompt(context_text: str, world: Any, slots: dict) -> str:
    """Build a grounded prompt for the align_color_grouped_wall task.

    Contains the full TaskPlan JSON schema, exact object/slot IDs, colors,
    poses, constraints, and instructions for the LLM. No joint angles,
    IK poses, trajectories, invented slots, or invented IDs.
    """
    from task_planning.types import SCHEMA_VERSION

    gt = world.grouped_tidy
    objects_data = {
        obj.id: {
            "pose": list(obj.pose),
            "class": obj.cls,
            "color": getattr(obj, "color", None) or "unknown",
        }
        for obj in world.objects
    }
    slots_data = {slot_id: list(slots[slot_id]) for slot_id in sorted(slots)}
    groups_data = {
        group.id: {"color": group.color, "objects": list(group.objects)}
        for group in (gt.groups if gt and gt.enabled else ())
    }
    obstacles_data = {
        obstacle.id: {
            "pose": list(obstacle.pose),
            "size": list(obstacle.size) if obstacle.size is not None else None,
            "fragile": obstacle.fragile,
        }
        for obstacle in world.obstacles
    }

    example_objects: list[str] = []
    example_slots: list[str] = []
    if gt and gt.enabled:
        for group in gt.groups[:2]:
            object_id = group.objects[0]
            matching_slots = sorted(
                slot_id
                for slot_id in slots
                if slot_id.startswith(f"{gt.slot_prefix}_{group.id}_")
            )
            if matching_slots:
                example_objects.append(object_id)
                example_slots.append(matching_slots[-1])
    if not example_objects:
        example_objects = list(world.target_objects[:2])
        example_slots = sorted(slots)[: len(example_objects)]

    example_plan = {
        "schema_version": SCHEMA_VERSION,
        "task": "align",
        "scene_id": world.scene_id,
        "target_objects": example_objects,
        "goal_predicates": [
            {"name": "at", "args": [obj_id, slot_id]}
            for obj_id, slot_id in zip(example_objects, example_slots)
        ],
        "slot_config": {
            "type": "line",
            "axis": gt.axis if gt else "x",
            "spacing_m": gt.spacing if gt else 0.085,
            "center_x": world.goal_center[0],
            "row_y": world.goal_center[1],
            "base_z": world.table_z_top + 0.033,
        },
        "steps": [
            step
            for index, (object_id, slot_id) in enumerate(
                zip(example_objects, example_slots)
            )
            for step in (
                {"step_id": index * 2, "action": "pick", "object": object_id},
                {
                    "step_id": index * 2 + 1,
                    "action": "place",
                    "object": object_id,
                    "slot": slot_id,
                },
            )
        ],
        "constraints": {
            "preserve_obstacles": True,
            "flexible_order": True,
            "grouped_tidy": True if gt else False,
        },
    }

    n_steps = len(world.target_objects) * 2
    slot_list = ", ".join(sorted(slots))
    wall = next(
        (obstacle for obstacle in world.obstacles if obstacle.kind == "wall"),
        None,
    )
    if wall is not None and wall.size is not None:
        wall_aabb = (
            f"x=[{wall.pose[0] - wall.size[0] / 2.0:.2f}, "
            f"{wall.pose[0] + wall.size[0] / 2.0:.2f}], "
            f"y=[{wall.pose[1] - wall.size[1] / 2.0:.2f}, "
            f"{wall.pose[1] + wall.size[1] / 2.0:.2f}], "
            f"z=[{wall.pose[2] - wall.size[2] / 2.0:.2f}, "
            f"{wall.pose[2] + wall.size[2] / 2.0:.2f}]"
        )
        wall_right_edge = wall.pose[0] + wall.size[0] / 2.0
        slot_right_min = wall_right_edge + 0.05
    else:
        wall_aabb = "none"
        wall_right_edge = 0.0
        slot_right_min = 0.0

    prompt = f"""You are a task planner for a Franka Panda table-top manipulator.

CONTEXT:
{context_text}

SCENE DATA:
Objects:
{json.dumps(objects_data, indent=2)}

Slots:
{json.dumps(slots_data, indent=2)}

Obstacles:
{json.dumps(obstacles_data, indent=2)}

Tidy Groups:
{json.dumps(groups_data, indent=2)}

SCHEMA VERSION: {SCHEMA_VERSION}
REQUIRED STEPS: {n_steps} (alternating pick/place for {len(world.target_objects)} cubes)

WALL GEOMETRY (hard constraint):
  Wall AABB: {wall_aabb}
  The arm CANNOT pass through this wall.
  All target slots are at x > {slot_right_min:.2f} on the right flank.
  Approach all cubes through the right flank (x > {wall_right_edge:.2f} side) only.

COLOR ASSIGNMENT (fixed, non-negotiable):
  blue cubes -> tidy_slot_blue_lane_* slots
  red cubes  -> tidy_slot_red_lane_* slots

PICK ORDERING:
  Process cubes with highest Y first within each color group.
  Interleave: pick one blue, pick one red, alternating.

JSON CONTRACT:
  Exactly {n_steps} steps ({len(world.target_objects)} pick + {len(world.target_objects)} place).
  Every cube appears exactly once as a pick and once as a place.
  Every slot appears at most once.
  Do NOT invent new slot names. Use only: {slot_list}
  Do NOT output joint angles, coordinates, or trajectories.

TASKPLAN JSON SCHEMA:
{{
  "schema_version": "{SCHEMA_VERSION}",
  "task": "align",
  "scene_id": "<scene_id>",
  "target_objects": ["<object_id>", ...],
  "goal_predicates": [{{"name": "at", "args": ["<object_id>", "<slot_id>"]}}, ...],
  "slot_config": {{"type": "line", "axis": "<x|y>", "spacing_m": <float>, "center_x": <float>, "row_y": <float>, "base_z": <float>}},
  "steps": [{{"step_id": <int>, "action": "pick"|"place", "object": "<object_id>", "slot": "<slot_id>"}}],
  "constraints": {{"preserve_obstacles": true, "flexible_order": true, "grouped_tidy": true}}
}}

RULES:
1. Use ONLY object IDs and slot IDs listed above.
2. Each step must be exactly "pick" or "place".
3. Alternate pick/place starting with pick. Exactly {n_steps} steps.
4. Each object picked and placed exactly once.
5. Each slot used at most once.
6. Assign objects to slots matching their color group (blue to blue_lane, red to red_lane).
7. Fill deep lane slots before corridor-near slots (higher y-index first for y-axis).
8. Minimize wall crossings. Prefer routing around the right side of the wall.
9. Do NOT invent joint angles, IK poses, trajectories, slot IDs, or object IDs.
10. Output valid JSON only. No Markdown, no comments.

EXAMPLE (complete valid structure for 2 cubes):
{json.dumps(example_plan, indent=2)}

Now produce the complete TaskPlan JSON for all {len(world.target_objects)} cubes:
"""
    return prompt
