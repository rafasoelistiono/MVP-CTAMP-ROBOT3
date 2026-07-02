from __future__ import annotations

import re
from collections.abc import Iterable

from .types import (
    ALLOWED_ACTIONS,
    ALLOWED_PREDICATES,
    SCHEMA_VERSION,
    TaskPlan,
)


_PREDICATE_RE = re.compile(r"^\s*([a-z][a-z0-9-]*)(?:\(([^()]*)\))?\s*$")


class PlanValidationError(ValueError):
    """A plan failed a deterministic pre-execution validation gate."""


def predicate_name(expression: str) -> str:
    match = _PREDICATE_RE.match(expression)
    if match is None:
        raise PlanValidationError(f"malformed predicate expression: {expression!r}")
    return match.group(1)


def validate(
    plan: TaskPlan,
    world_object_ids: set[str],
    allowed_predicates: Iterable[str] | None = None,
) -> None:
    _gate_1_schema(plan)
    _gate_2_object_whitelist(plan, world_object_ids)
    _gate_3_predicate_whitelist(plan, allowed_predicates)
    _gate_4_action_sequence(plan)


def _gate_1_schema(plan: TaskPlan) -> None:
    if plan.schema_version != SCHEMA_VERSION:
        raise PlanValidationError(
            f"gate 1: unsupported schema_version {plan.schema_version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if re.fullmatch(r"[a-z][a-z0-9_-]*", plan.task) is None:
        raise PlanValidationError(
            f"gate 1: task name must be a lowercase plugin identifier, got {plan.task!r}"
        )
    if plan.slot_config.type not in {"line", "tower", "pyramid"}:
        raise PlanValidationError(
            f"gate 1: slot_config.type must be 'line', 'tower', or 'pyramid', "
            f"got {plan.slot_config.type!r}"
        )
    if plan.slot_config.axis != "x":
        raise PlanValidationError("gate 1: only slot axis 'x' is currently supported")
    if plan.slot_config.spacing_m <= 0:
        raise PlanValidationError("gate 1: slot spacing must be positive")
    if plan.slot_config.layer_height_m <= 0:
        raise PlanValidationError("gate 1: tower layer height must be positive")
    if plan.slot_config.type == "pyramid":
        _validate_pyramid_slot_schema(plan)
    if not plan.scene_id.strip():
        raise PlanValidationError("gate 1: scene_id must not be empty")
    if not plan.steps:
        raise PlanValidationError("gate 1: steps must not be empty")
    if not plan.target_objects:
        raise PlanValidationError("gate 1: target_objects must not be empty")
    if len(set(plan.target_objects)) != len(plan.target_objects):
        raise PlanValidationError("gate 1: target_objects contains duplicates")
    expected_ids = list(range(len(plan.steps)))
    actual_ids = [step.step_id for step in plan.steps]
    if actual_ids != expected_ids:
        raise PlanValidationError(
            f"gate 1: step_id values must be contiguous from 0; got {actual_ids}"
        )
    for step in plan.steps:
        if step.action not in ALLOWED_ACTIONS:
            raise PlanValidationError(
                f"gate 1: unsupported action {step.action!r} at step {step.step_id}"
            )
        if step.action == "place" and not step.slot:
            raise PlanValidationError(
                f"gate 1: place step {step.step_id} requires slot"
            )
        if step.action == "stack_place" and not step.on_top_of:
            raise PlanValidationError(
                f"gate 1: stack_place step {step.step_id} requires on_top_of"
            )


def _validate_pyramid_slot_schema(plan: TaskPlan) -> None:
    config = plan.slot_config
    if config.row_count <= 0:
        raise PlanValidationError("gate 1: pyramid row_count must be positive")
    if config.base_row_length <= 0:
        raise PlanValidationError("gate 1: pyramid base_row_length must be positive")
    if config.row_count != config.base_row_length:
        raise PlanValidationError(
            "gate 1: pyramid row_count must equal base_row_length for a 4-3-2-1 style task"
        )
    expected_targets = config.row_count * (config.row_count + 1) // 2
    if expected_targets != len(plan.target_objects):
        raise PlanValidationError(
            "gate 1: pyramid target count must equal "
            f"row_count*(row_count+1)/2; expected {expected_targets}, "
            f"got {len(plan.target_objects)}"
        )
    if config.row_spacing_m <= 0:
        raise PlanValidationError("gate 1: pyramid row_spacing_m must be positive")


def _gate_2_object_whitelist(
    plan: TaskPlan,
    world_object_ids: set[str],
) -> None:
    referenced = set(plan.target_objects)
    for step in plan.steps:
        referenced.add(step.object)
        if step.on_top_of:
            referenced.add(step.on_top_of)
        for expression in step.preconditions + step.effects:
            match = _PREDICATE_RE.match(expression)
            if match is not None:
                referenced.update(_predicate_object_args(match.group(1), match.group(2)))
    for predicate in plan.goal_predicates:
        name = predicate.get("name")
        args = predicate.get("args", [])
        if isinstance(name, str) and isinstance(args, list):
            referenced.update(_predicate_object_args(name, args))
    unknown = sorted(referenced - world_object_ids)
    if unknown:
        raise PlanValidationError(
            "gate 2: plan references object IDs not present in context: "
            + ", ".join(unknown)
        )
    step_objects = {step.object for step in plan.steps}
    missing_steps = sorted(set(plan.target_objects) - step_objects)
    if missing_steps:
        raise PlanValidationError(
            "gate 2: target objects have no action steps: " + ", ".join(missing_steps)
        )


def _predicate_object_args(name: str, raw_args) -> set[str]:
    if isinstance(raw_args, str):
        args = [part.strip() for part in raw_args.split(",") if part.strip()]
    elif isinstance(raw_args, list):
        args = [str(part) for part in raw_args]
    else:
        return set()
    if name == "at":
        return set(args[:1])
    if name == "on":
        return set(args[:2])
    if name in {"clear", "holding"}:
        return set(args[:1])
    return set()


def _gate_3_predicate_whitelist(
    plan: TaskPlan,
    allowed_predicates: Iterable[str] | None = None,
) -> None:
    effective_allowed = set(allowed_predicates or ALLOWED_PREDICATES)
    expressions: list[str] = []
    for index, predicate in enumerate(plan.goal_predicates):
        name = predicate.get("name")
        if not isinstance(name, str):
            raise PlanValidationError(
                f"gate 3: goal_predicates[{index}].name must be a string"
            )
        expressions.append(name)
        args = predicate.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise PlanValidationError(
                f"gate 3: goal_predicates[{index}].args must be an array of strings"
            )
    for step in plan.steps:
        expressions.extend(step.preconditions)
        expressions.extend(step.effects)
    for expression in expressions:
        name = predicate_name(expression)
        if name not in effective_allowed:
            raise PlanValidationError(
                f"gate 3: predicate {name!r} is not allowed; "
                f"allowed: {sorted(effective_allowed)}"
            )


def _gate_4_action_sequence(plan: TaskPlan) -> None:
    held: str | None = None
    placed: set[str] = set()
    picked_once: set[str] = set()

    for step in plan.steps:
        if step.action == "pick":
            if held is not None:
                raise PlanValidationError(
                    f"gate 4: step {step.step_id} picks {step.object!r} "
                    f"while {held!r} is still held"
                )
            if step.object in picked_once and step.object not in placed:
                raise PlanValidationError(
                    f"gate 4: object {step.object!r} is picked again before placement"
                )
            held = step.object
            picked_once.add(step.object)
            placed.discard(step.object)
            continue

        if held != step.object:
            raise PlanValidationError(
                f"gate 4: step {step.step_id} {step.action}({step.object}) "
                f"requires holding that object; currently held: {held!r}"
            )

        if step.action == "stack_place":
            support = step.on_top_of
            if support == step.object:
                raise PlanValidationError(
                    f"gate 4: step {step.step_id} cannot stack an object on itself"
                )
            if support not in placed:
                raise PlanValidationError(
                    f"gate 4: support object {support!r} has not been placed "
                    f"before step {step.step_id}"
                )

        placed.add(step.object)
        held = None

    if held is not None:
        raise PlanValidationError(
            f"gate 4: plan ends while object {held!r} is still held"
        )
    not_placed = sorted(set(plan.target_objects) - placed)
    if not_placed:
        raise PlanValidationError(
            "gate 4: target objects are not placed by the plan: "
            + ", ".join(not_placed)
        )


def validate_grouped_align_order(
    plan: TaskPlan,
    slot_prefix: str,
    groups: tuple,
) -> None:
    """Validate object-to-slot assignment for grouped tidy align plans.

    groups is a tuple of TidyGroup-like objects with .id, .objects attributes.
    """
    expected_slot: dict[str, str] = {}
    for group in groups:
        for i, obj_id in enumerate(group.objects):
            expected_slot[obj_id] = f"{slot_prefix}_{group.id}_{i}"

    for step in plan.steps:
        if step.action != "place" or not step.slot:
            continue
        obj_id = step.object
        if obj_id not in expected_slot:
            raise PlanValidationError(
                f"grouped align: object {obj_id!r} is not assigned to any tidy group"
            )
        if step.slot != expected_slot[obj_id]:
            raise PlanValidationError(
                f"grouped align: object {obj_id!r} must go to slot "
                f"{expected_slot[obj_id]!r}, got {step.slot!r}"
            )
