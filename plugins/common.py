from __future__ import annotations

from collections.abc import Iterable

from task_planning.types import SlotConfig, Step, TaskPlan
from task_planning.validator import PlanValidationError
from world.state import WorldState


def validate_common_cube_plan(
    plan: TaskPlan,
    world: WorldState,
    *,
    task: str,
    slot_type: str,
    supported_actions: set[str],
) -> None:
    if plan.task != task:
        raise PlanValidationError(f"{task} plugin cannot execute task {plan.task!r}")
    if plan.slot_config.type != slot_type:
        raise PlanValidationError(f"{task} task requires {slot_type} slot_config")
    if plan.target_objects != world.target_objects:
        raise PlanValidationError(
            f"{task} plan target_objects must exactly match context task targets"
        )
    if plan.constraints.get("preserve_obstacles", True) is not True:
        raise PlanValidationError(f"{task} plan cannot disable obstacle preservation")

    actions = {step.action for step in plan.steps}
    unsupported = sorted(actions - supported_actions)
    if unsupported:
        raise PlanValidationError(f"{task} task does not support actions: {unsupported}")

    missing_capabilities = sorted(actions - set(world.robot_capabilities))
    if missing_capabilities:
        raise PlanValidationError(
            f"robot lacks capabilities required by {task} plan: {missing_capabilities}"
        )

    non_cubes = [
        object_id
        for object_id in plan.target_objects
        if world.object_by_id(object_id).cls != "cube"
    ]
    if non_cubes:
        raise PlanValidationError(
            f"{task} task accepts cube objects only: " + ", ".join(non_cubes)
        )

    unreachable = [
        object_id
        for object_id in plan.target_objects
        if not world.object_by_id(object_id).reachable
    ]
    if unreachable:
        raise PlanValidationError(
            f"{task} target objects are unreachable: " + ", ".join(unreachable)
        )


def pick_place_pairs(plan: TaskPlan, task: str) -> Iterable[tuple[int, str, Step, Step]]:
    expected = plan.target_objects
    if len(plan.steps) != len(expected) * 2:
        raise PlanValidationError(
            f"{task} plan must contain exactly one pick/place pair per cube"
        )
    for index, object_id in enumerate(expected):
        yield index, object_id, plan.steps[index * 2], plan.steps[index * 2 + 1]


def pyramid_slot_order(config: SlotConfig) -> tuple[str, ...]:
    return tuple(
        f"row{row}_col{column}"
        for row in range(config.row_count)
        for column in range(config.base_row_length - row)
    )
