from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .types import SlotConfig, Step, TaskPlan


class PlanLoadError(ValueError):
    """Raised when JSON cannot be converted to the typed plan contract."""


def _ensure_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanLoadError(f"{path} must be a JSON object")
    return value


def _ensure_sequence(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise PlanLoadError(f"{path} must be a JSON array")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise PlanLoadError(
            f"{path} contains unsupported fields: {', '.join(unknown)}"
        )


def _tuple2(value: Any, path: str) -> tuple[float, float]:
    values = _ensure_sequence(value, path)
    if len(values) != 2:
        raise PlanLoadError(f"{path} must contain exactly two numbers")
    try:
        return float(values[0]), float(values[1])
    except (TypeError, ValueError) as exc:
        raise PlanLoadError(f"{path} must contain numbers") from exc


def parse_plan(payload: Mapping[str, Any]) -> TaskPlan:
    data = _ensure_mapping(payload, "plan")
    required = {
        "schema_version",
        "task",
        "scene_id",
        "target_objects",
        "goal_predicates",
        "slot_config",
        "steps",
    }
    missing = sorted(required - set(data))
    if missing:
        raise PlanLoadError(f"plan is missing required fields: {', '.join(missing)}")
    _reject_unknown(data, required | {"constraints"}, "plan")

    slot_raw = _ensure_mapping(data["slot_config"], "slot_config")
    _reject_unknown(
        slot_raw,
        {
            "type",
            "axis",
            "spacing_m",
            "row_y",
            "row_spacing_m",
            "row_count",
            "base_row_length",
            "center_x",
            "base_y",
            "base_z",
            "base_xy",
            "layer_height_m",
        },
        "slot_config",
    )
    slot_type = str(slot_raw.get("type", "")).strip()
    if not slot_type:
        raise PlanLoadError("slot_config.type is required")
    slot = SlotConfig(
        type=slot_type,  # type: ignore[arg-type]
        axis=str(slot_raw.get("axis", "x")),
        spacing_m=float(slot_raw.get("spacing_m", 0.125)),
        row_y=float(slot_raw.get("row_y", -0.06)),
        row_spacing_m=float(slot_raw.get("row_spacing_m", 0.08)),
        row_count=int(slot_raw.get("row_count", 0)),
        base_row_length=int(slot_raw.get("base_row_length", 0)),
        center_x=float(slot_raw.get("center_x", 0.22)),
        base_y=float(slot_raw.get("base_y", -0.06)),
        base_z=float(slot_raw.get("base_z", 0.83)),
        base_xy=_tuple2(slot_raw.get("base_xy", [0.22, -0.06]), "slot_config.base_xy"),
        layer_height_m=float(slot_raw.get("layer_height_m", 0.06)),
    )

    steps: list[Step] = []
    for index, raw in enumerate(_ensure_sequence(data["steps"], "steps")):
        item = _ensure_mapping(raw, f"steps[{index}]")
        _reject_unknown(
            item,
            {
                "step_id",
                "action",
                "object",
                "slot",
                "on_top_of",
                "preconditions",
                "effects",
            },
            f"steps[{index}]",
        )
        for field_name in ("step_id", "action", "object"):
            if field_name not in item:
                raise PlanLoadError(f"steps[{index}].{field_name} is required")
        steps.append(
            Step(
                step_id=int(item["step_id"]),
                action=str(item["action"]),  # type: ignore[arg-type]
                object=str(item["object"]),
                slot=str(item["slot"]) if item.get("slot") is not None else None,
                on_top_of=(
                    str(item["on_top_of"])
                    if item.get("on_top_of") is not None
                    else None
                ),
                preconditions=tuple(
                    str(value)
                    for value in _ensure_sequence(
                        item.get("preconditions", []),
                        f"steps[{index}].preconditions",
                    )
                ),
                effects=tuple(
                    str(value)
                    for value in _ensure_sequence(
                        item.get("effects", []),
                        f"steps[{index}].effects",
                    )
                ),
            )
        )

    predicates_list: list[dict] = []
    for index, value in enumerate(
        _ensure_sequence(data["goal_predicates"], "goal_predicates")
    ):
        predicate = _ensure_mapping(value, f"goal_predicates[{index}]")
        _reject_unknown(predicate, {"name", "args"}, f"goal_predicates[{index}]")
        predicates_list.append(dict(predicate))
    predicates = tuple(predicates_list)
    targets = tuple(
        str(value)
        for value in _ensure_sequence(data["target_objects"], "target_objects")
    )
    constraints = data.get("constraints", {})
    if not isinstance(constraints, dict):
        raise PlanLoadError("constraints must be a JSON object")

    return TaskPlan(
        schema_version=str(data["schema_version"]),
        task=str(data["task"]),
        scene_id=str(data["scene_id"]),
        target_objects=targets,
        goal_predicates=predicates,
        slot_config=slot,
        steps=tuple(steps),
        constraints=dict(constraints),
    )


def load_plan(path: str | Path) -> TaskPlan:
    plan_path = Path(path)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlanLoadError(f"plan file does not exist: {plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise PlanLoadError(
            f"invalid JSON in {plan_path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    return parse_plan(payload)
