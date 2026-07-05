from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

from task_planning.types import ALLOWED_ACTIONS

from .state import (
    ChallengeConfig,
    GroupedTidyConfig,
    ObstacleState,
    ObjectState,
    TidyGroup,
    WorldState,
)


VALID_VARIANTS = {
    "group_no_obs",
    "ungroup_no_obs",
    "group_obs",
    "ungroup_obs",
    "group_long_obs",
    "ungroup_long_obs",
    "align_grouped_tidy_wall_world",
}
_SECTION_RE = re.compile(r"^##\s+([a-zA-Z0-9_-]+)\s*$")
_FIELD_RE = re.compile(r"^\s*-\s+([a-zA-Z0-9_-]+)\s*:\s*(.*?)\s*$")
_CONTINUATION_RE = re.compile(r"^\s+([a-zA-Z0-9_-]+)\s*:\s*(.*?)\s*$")


class ContextValidationError(ValueError):
    """The runtime context is missing or contradicts required scene facts."""


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            # CONTEXT.MD examples intentionally use compact YAML-style lists
            # such as [pick, place] without Python/JSON quotes.
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(item) for item in inner.split(",")]
    if value.startswith("{"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ContextValidationError(f"invalid context literal: {value}") from exc
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def _parse_markdown(text: str) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    section: str | None = None
    current_record: dict[str, Any] | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("# ") or stripped.startswith("~~~"):
            continue
        section_match = _SECTION_RE.match(stripped)
        if section_match:
            section = section_match.group(1).lower()
            sections[section] = (
                [] if section in {"objects", "obstacles", "tidy_groups"} else {}
            )
            current_record = None
            continue
        if section is None:
            continue

        field_match = _FIELD_RE.match(raw_line)
        continuation_match = _CONTINUATION_RE.match(raw_line)
        match = field_match or continuation_match
        if match is None:
            continue

        key, raw_value = match.groups()
        value = _parse_scalar(raw_value)
        if section in {"objects", "obstacles", "tidy_groups"}:
            if field_match and key == "id":
                current_record = {"id": value}
                sections[section].append(current_record)
            elif current_record is None:
                raise ContextValidationError(
                    f"line {line_number}: {section} record must start with '- id:'"
                )
            else:
                current_record[key] = value
        else:
            sections[section][key] = value
    return sections


def _required_map(sections: dict[str, Any], name: str, fields: set[str]) -> dict:
    value = sections.get(name)
    if not isinstance(value, dict):
        raise ContextValidationError(f"context is missing required section '## {name}'")
    missing = sorted(fields - set(value))
    if missing:
        raise ContextValidationError(
            f"section '## {name}' is missing fields: {', '.join(missing)}"
        )
    return value


def _tuple_numbers(value: Any, length: int, path: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ContextValidationError(f"{path} must contain {length} numbers")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContextValidationError(f"{path} must contain only numbers") from exc


def _object_rgba(raw: dict, path: str) -> tuple[float, float, float, float] | None:
    if "rgba" in raw:
        return _tuple_numbers(raw["rgba"], 4, f"{path}.rgba")  # type: ignore[return-value]
    color = str(raw.get("color", "")).strip().lower()
    if not color:
        return None
    palette = {
        "green": (0.0, 0.85, 0.10, 1.0),
        "red": (1.0, 0.05, 0.02, 1.0),
        "yellow": (1.0, 0.90, 0.0, 1.0),
        "blue": (0.0, 0.20, 1.0, 1.0),
        "white": (1.0, 1.0, 1.0, 1.0),
    }
    if color not in palette:
        raise ContextValidationError(
            f"{path}.color must be one of {sorted(palette)}"
        )
    return palette[color]


def _parse_grouped_tidy(
    sections: dict[str, Any],
    targets: tuple[str, ...],
    object_ids: set[str],
    objects_by_id: dict[str, Any] | None = None,
) -> GroupedTidyConfig | None:
    raw = sections.get("grouped_tidy")
    if not raw or not isinstance(raw, dict):
        return None
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return None
    require_ordered = bool(raw.get("require_ordered", False))
    slot_prefix = str(raw.get("slot_prefix", "tidy_slot"))
    axis = str(raw.get("axis", "x"))
    if axis not in {"x", "y"}:
        raise ContextValidationError(
            f"grouped_tidy.axis must be 'x' or 'y', got {axis!r}"
        )
    spacing = float(raw.get("spacing", 0.085))
    if spacing <= 0:
        raise ContextValidationError("grouped_tidy.spacing must be positive")
    row_spacing = float(raw.get("row_spacing", 0.105))
    if row_spacing <= 0:
        raise ContextValidationError("grouped_tidy.row_spacing must be positive")

    group_rows = sections.get("tidy_groups", [])
    if not group_rows:
        raise ContextValidationError(
            "grouped_tidy enabled but no ## tidy_groups section found"
        )
    groups: list[TidyGroup] = []
    assigned_objects: set[str] = set()
    for idx, raw_group in enumerate(group_rows):
        required = {"id", "color", "objects", "center"}
        missing = sorted(required - set(raw_group))
        if missing:
            raise ContextValidationError(
                f"tidy_groups[{idx}] is missing fields: {', '.join(missing)}"
            )
        group_id = str(raw_group["id"])
        color = str(raw_group["color"])
        objects_raw = raw_group["objects"]
        if not isinstance(objects_raw, list) or not objects_raw:
            raise ContextValidationError(
                f"tidy_groups[{idx}].objects must be a non-empty list"
            )
        objects = tuple(str(o) for o in objects_raw)
        center = _tuple_numbers(raw_group["center"], 3, f"tidy_groups[{idx}].center")

        unknown = sorted(set(objects) - object_ids)
        if unknown:
            raise ContextValidationError(
                f"tidy_groups[{idx}] references unknown objects: "
                + ", ".join(unknown)
            )
        if objects_by_id is not None:
            wrong_color = sorted(
                oid for oid in objects
                if objects_by_id.get(oid) is not None
                and objects_by_id[oid].get("color") is not None
                and str(objects_by_id[oid]["color"]).strip().lower() != color.strip().lower()
            )
            if wrong_color:
                raise ContextValidationError(
                    f"tidy_groups[{idx}] color={color!r} but objects have wrong color: "
                    + ", ".join(wrong_color)
                )
        unseen = sorted(set(objects) - set(targets))
        if unseen:
            raise ContextValidationError(
                f"tidy_groups[{idx}] has objects not in target_objects: "
                + ", ".join(unseen)
            )
        overlap = sorted(set(objects) & assigned_objects)
        if overlap:
            raise ContextValidationError(
                f"tidy_groups[{idx}] duplicates objects already assigned: "
                + ", ".join(overlap)
            )
        assigned_objects.update(objects)
        groups.append(
            TidyGroup(id=group_id, color=color, objects=objects, center=center)
        )

    unassigned = sorted(set(targets) - assigned_objects)
    if unassigned:
        raise ContextValidationError(
            "target_objects not assigned to any tidy group: " + ", ".join(unassigned)
        )
    return GroupedTidyConfig(
        enabled=enabled,
        require_ordered=require_ordered,
        slot_prefix=slot_prefix,
        axis=axis,
        spacing=spacing,
        row_spacing=row_spacing,
        groups=tuple(groups),
    )


def _parse_challenge(
    sections: dict[str, Any],
    obstacle_ids: set[str],
) -> ChallengeConfig | None:
    raw = sections.get("challenge")
    if not raw or not isinstance(raw, dict):
        return None
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return None
    challenge_type = str(raw.get("type", "")).strip()
    if not challenge_type:
        raise ContextValidationError("challenge enabled but missing type")
    obs_ids_raw = raw.get("obstacle_ids", [])
    if not isinstance(obs_ids_raw, list) or not obs_ids_raw:
        raise ContextValidationError(
            "challenge.enabled requires non-empty obstacle_ids"
        )
    obs_ids = tuple(str(o) for o in obs_ids_raw)
    unknown = sorted(set(obs_ids) - obstacle_ids)
    if unknown:
        raise ContextValidationError(
            "challenge.obstacle_ids references unknown obstacles: "
            + ", ".join(unknown)
        )
    return ChallengeConfig(
        type=challenge_type,
        enabled=enabled,
        obstacle_ids=obs_ids,
        require_obstacle_aware_slots=bool(
            raw.get("require_obstacle_aware_slots", False)
        ),
        require_motion_probe=bool(raw.get("require_motion_probe", False)),
        compare_planners=tuple(
            str(p) for p in raw.get("compare_planners", [])
        ),
        min_gap_width=float(raw.get("min_gap_width", 0.0)),
        inflated_clearance_required=bool(
            raw.get("inflated_clearance_required", False)
        ),
        wall_blocks_direct_path=bool(raw.get("wall_blocks_direct_path", False)),
        side_corridors_required=bool(raw.get("side_corridors_required", False)),
    )


def build_world_state(path: str | Path) -> WorldState:
    context_path = Path(path)
    try:
        text = context_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ContextValidationError(
            f"context file does not exist: {context_path}"
        ) from exc
    sections = _parse_markdown(text)

    scene = _required_map(sections, "scene", {"scene_id", "variant"})
    table = _required_map(
        sections,
        "table",
        {"x_range", "y_range", "z_top", "goal_center"},
    )
    robot = _required_map(
        sections,
        "robot",
        {
            "id",
            "reach_min_xy",
            "reach_max_xy",
            "base_xy",
            "capabilities",
        },
    )
    task = _required_map(
        sections,
        "task",
        {"name", "target_objects", "description"},
    )
    constraints = _required_map(
        sections,
        "constraints",
        {
            "preserve_obstacles",
            "max_retries_per_object",
            "allowed_predicates",
        },
    )

    variant = str(scene["variant"])
    scene_id = str(scene["scene_id"]).strip()
    if not scene_id:
        raise ContextValidationError("scene.scene_id must not be empty")
    if variant not in VALID_VARIANTS:
        raise ContextValidationError(
            f"scene.variant {variant!r} is invalid; expected one of {sorted(VALID_VARIANTS)}"
        )
    task_name = str(task["name"])
    if re.fullmatch(r"[a-z][a-z0-9_-]*", task_name) is None:
        raise ContextValidationError(
            f"task.name {task_name!r} must be a lowercase plugin identifier"
        )
    task_description = str(task["description"]).strip()
    if not task_description:
        raise ContextValidationError("task.description must not be empty")

    base_xy = _tuple_numbers(robot["base_xy"], 2, "robot.base_xy")
    base_z = float(robot.get("base_z", 0.80))
    reach_min = float(robot["reach_min_xy"])
    reach_max = float(robot["reach_max_xy"])
    if not 0 <= reach_min < reach_max:
        raise ContextValidationError(
            "robot reach bounds must satisfy 0 <= reach_min_xy < reach_max_xy"
        )

    obstacle_rows = sections.get("obstacles", [])
    if not isinstance(obstacle_rows, list):
        raise ContextValidationError("section '## obstacles' must be a record list")
    obstacles: list[ObstacleState] = []
    obstacle_ids: set[str] = set()
    for index, raw in enumerate(obstacle_rows):
        required = {"id", "pose", "fragile", "radius", "height"}
        missing = sorted(required - set(raw))
        if missing:
            raise ContextValidationError(
                f"obstacles[{index}] is missing fields: {', '.join(missing)}"
            )
        obstacle_id = str(raw["id"])
        if not obstacle_id.strip():
            raise ContextValidationError(f"obstacles[{index}].id must not be empty")
        if obstacle_id in obstacle_ids:
            raise ContextValidationError(f"duplicate obstacle id: {obstacle_id}")
        obstacle_ids.add(obstacle_id)
        height = str(raw["height"])
        if height not in {"short", "long"}:
            raise ContextValidationError(
                f"obstacles[{index}].height must be 'short' or 'long'"
            )
        if not isinstance(raw["fragile"], bool):
            raise ContextValidationError(
                f"obstacles[{index}].fragile must be true or false"
            )
        obstacles.append(
            ObstacleState(
                id=obstacle_id,
                pose=_tuple_numbers(raw["pose"], 3, f"obstacles[{index}].pose"),
                fragile=bool(raw["fragile"]),
                radius=float(raw["radius"]),
                height=height,  # type: ignore[arg-type]
                size=(
                    _tuple_numbers(raw["size"], 3, f"obstacles[{index}].size")
                    if "size" in raw
                    else None
                ),
            )
        )

    object_rows = sections.get("objects")
    if not isinstance(object_rows, list) or not object_rows:
        raise ContextValidationError(
            "context requires at least one record under '## objects'"
        )
    objects: list[ObjectState] = []
    object_ids: set[str] = set()
    for index, raw in enumerate(object_rows):
        required = {"id", "class", "pose", "reachable", "near_obstacle"}
        missing = sorted(required - set(raw))
        if missing:
            raise ContextValidationError(
                f"objects[{index}] is missing fields: {', '.join(missing)}"
            )
        object_id = str(raw["id"])
        if not object_id.strip():
            raise ContextValidationError(f"objects[{index}].id must not be empty")
        if object_id in object_ids:
            raise ContextValidationError(f"duplicate object id: {object_id}")
        object_ids.add(object_id)
        cls = str(raw["class"])
        if cls not in {"cube", "cylinder"}:
            raise ContextValidationError(
                f"objects[{index}].class must be 'cube' or 'cylinder'"
            )
        if not isinstance(raw["reachable"], bool):
            raise ContextValidationError(
                f"objects[{index}].reachable must be true or false"
            )
        if not isinstance(raw["near_obstacle"], bool):
            raise ContextValidationError(
                f"objects[{index}].near_obstacle must be true or false"
            )
        pose = _tuple_numbers(raw["pose"], 3, f"objects[{index}].pose")
        distance = math.dist(pose[:2], base_xy)
        reachable = reach_min <= distance <= reach_max
        near_obstacle = any(
            math.dist(pose[:2], obstacle.pose[:2]) <= 0.18
            for obstacle in obstacles
        )
        objects.append(
            ObjectState(
                id=object_id,
                cls=cls,  # type: ignore[arg-type]
                pose=pose,
                reachable=reachable,
                near_obstacle=near_obstacle,
                rgba=_object_rgba(raw, f"objects[{index}]"),
                color=str(raw.get("color", "")).strip().lower() or None,
            )
        )

    targets_raw = task["target_objects"]
    if not isinstance(targets_raw, list) or not targets_raw:
        raise ContextValidationError("task.target_objects must be a non-empty list")
    targets = tuple(str(value) for value in targets_raw)
    if len(set(targets)) != len(targets):
        raise ContextValidationError("task.target_objects contains duplicates")
    unknown_targets = sorted(set(targets) - object_ids)
    if unknown_targets:
        raise ContextValidationError(
            "task.target_objects references unknown objects: "
            + ", ".join(unknown_targets)
        )

    capabilities_raw = robot["capabilities"]
    if not isinstance(capabilities_raw, list) or not capabilities_raw:
        raise ContextValidationError("robot.capabilities must be a non-empty list")
    robot_id = str(robot["id"]).strip()
    if not robot_id:
        raise ContextValidationError("robot.id must not be empty")
    unsupported_capabilities = sorted(set(map(str, capabilities_raw)) - set(ALLOWED_ACTIONS))
    if unsupported_capabilities:
        raise ContextValidationError(
            "robot.capabilities contains unsupported actions: "
            + ", ".join(unsupported_capabilities)
        )

    allowed_raw = constraints["allowed_predicates"]
    if not isinstance(allowed_raw, list) or not allowed_raw:
        raise ContextValidationError(
            "constraints.allowed_predicates must be a non-empty list"
        )
    allowed = tuple(str(value) for value in allowed_raw)
    invalid_predicates = sorted(
        name
        for name in allowed
        if re.fullmatch(r"[a-z][a-z0-9-]*", name) is None
    )
    if invalid_predicates:
        raise ContextValidationError(
            "context contains invalid predicate identifiers: "
            + ", ".join(invalid_predicates)
        )

    if variant.endswith("_no_obs") and obstacles:
        raise ContextValidationError(
            f"variant {variant!r} must not define obstacles"
        )
    if not variant.endswith("_no_obs") and not obstacles:
        raise ContextValidationError(
            f"variant {variant!r} requires obstacle records"
        )
    if not isinstance(constraints["preserve_obstacles"], bool):
        raise ContextValidationError(
            "constraints.preserve_obstacles must be true or false"
        )
    max_retries = int(constraints["max_retries_per_object"])
    if max_retries < 0:
        raise ContextValidationError(
            "constraints.max_retries_per_object must be non-negative"
        )
    if bool(constraints["preserve_obstacles"]):
        non_fragile = [obstacle.id for obstacle in obstacles if not obstacle.fragile]
        if non_fragile:
            raise ContextValidationError(
                "preserve_obstacles=true requires every obstacle to be fragile: "
                + ", ".join(non_fragile)
            )

    raw_by_id = {str(raw["id"]): raw for raw in object_rows}
    grouped_tidy = _parse_grouped_tidy(sections, targets, object_ids, raw_by_id)
    challenge = _parse_challenge(sections, obstacle_ids)

    return WorldState(
        scene_id=scene_id,
        variant=variant,
        objects=tuple(objects),
        obstacles=tuple(obstacles),
        table_x_range=_tuple_numbers(table["x_range"], 2, "table.x_range"),
        table_y_range=_tuple_numbers(table["y_range"], 2, "table.y_range"),
        table_z_top=float(table["z_top"]),
        goal_center=_tuple_numbers(table["goal_center"], 3, "table.goal_center"),
        goal_area_size_xy=_tuple_numbers(
            table.get("goal_area_size_xy", [0.52, 0.40]),
            2,
            "table.goal_area_size_xy",
        ),
        robot_id=robot_id,
        robot_base_xy=base_xy,
        robot_reach_min=reach_min,
        robot_reach_max=reach_max,
        robot_capabilities=tuple(str(value) for value in capabilities_raw),
        task_name=task_name,
        target_objects=targets,
        task_description=task_description,
        preserve_obstacles=bool(constraints["preserve_obstacles"]),
        max_retries_per_object=max_retries,
        allowed_predicates=allowed,
        grouped_tidy=grouped_tidy,
        challenge=challenge,
        robot_base_z=base_z,
    )
