# Codebase Context

## Scope

This repository is an align-only CTAMP robot pipeline for the `align_grouped_tidy_wall_world` scene. The robot moves red/blue cubes around a frontal wall into tidy grouped lanes using `pick` and `place` actions only.

Removed task families: stack cubes, pyramid cubes.

## Main Flow

1. Context markdown is parsed by `world.builder.build_world_state()`.
2. TaskPlan JSON is parsed by `task_planning.loader.load_plan()`.
3. `task_planning.validator.validate()` checks schema, object references, allowed predicates, and `pick/place` action order.
4. `plugins.registry.DEFAULT_REGISTRY` resolves the only task plugin: `align`.
5. `plugins.align_task.AlignTaskPlugin` validates target cubes, grouped tidy membership, slot prefixes, and object-slot assignment.
6. `world.slot_allocator.allocate_grouped_align_slots()` creates obstacle-aware tidy slots for each color lane.
7. `cli.run_simulation` selects runtime config, prepares the MuJoCo scene, and starts `execution.runner.TaskRunner`.
8. If the context challenge requires motion probing, robust align generates and ranks candidate plans before execution.
9. `execution.primitives.MuJoCoExecutorPrimitives` maps plan steps to backend `pick()` and `place()` calls.
10. `execution.verifier.ObservedPredicateVerifier` checks observed effects and final goal state.
11. Telemetry writes event logs, summary CSV, and run manifest under `logs/`.

## Supported Plan Contract

- `task`: `align`
- `slot_config.type`: `line`
- `actions`: `pick`, `place`
- `predicates`: `at`, `clear`, `handempty`, `holding`, `stable`
- Target objects must match context `target_objects` exactly.
- Grouped tidy plans must use slots prefixed by `tidy_slot` and matching each object's group.

## Key Files

- `contexts/examples/align_grouped_tidy_wall_world.md`: source scene/context.
- `task_plans/examples/align_grouped_tidy_wall_world.json`: reference plan.
- `world/builder.py`: context parser and `WorldState` builder.
- `world/slot_allocator.py`: line slot allocation and grouped tidy obstacle-aware slots.
- `task_planning/types.py`: TaskPlan dataclasses and allowed actions/predicates.
- `task_planning/validator.py`: deterministic validation gates.
- `task_planning/generator.py`: LLM prompt builder and request logic.
- `task_planning/candidate_generator.py`: robust align candidate plans.
- `task_planning/cost_model.py`: candidate ranking cost model.
- `execution/motion_probe.py`: geometric/backend IK/OMPL feasibility probes.
- `execution/runner.py`: execution loop, robust align selection, align recovery.
- `execution/primitives.py`: generic step to MuJoCo backend adapter.
- `execution/verifier.py`: observed predicate checks and final grouped tidy checks.
- `plugins/align_task.py`: align-specific validation, runtime tuning, progress, goal verification.
- `configuration/`: typed runtime config and built-in profiles.
- `cli/generate_plan.py`: generate or validate a TaskPlan.
- `cli/run_simulation.py`: run plan in MuJoCo.
- `telemetry/`: summary, naming, and run manifest helpers.
- `tests/`: unit and integration coverage for align flow.

## Commands

Validate/generate from reference response:

```bash
python -m cli.generate_plan \
  --context contexts/examples/align_grouped_tidy_wall_world.md \
  --task align \
  --response-file task_plans/examples/align_grouped_tidy_wall_world.json \
  --output task_plans/generated
```

Run simulation:

```bash
python -m cli.run_simulation \
  --plan task_plans/examples/align_grouped_tidy_wall_world.json \
  --context contexts/examples/align_grouped_tidy_wall_world.md \
  --scene align_grouped_tidy_wall_world \
  --runtime-profile obstacle \
  --robust-align
```

Run tests:

```bash
pytest
```

## Runtime Notes

- `obstacle` profile is the normal profile for the wall world.
- `--robust-align` or context `challenge.require_motion_probe` enables candidate probing.
- `--use-adaptive-cache` enables cache-backed candidate ranking.
- Generated files live in ignored paths: `logs/`, `models/generated/`, `task_plans/generated/`, `.pytest_cache/`, `__pycache__/`.

## Current Review Notes

- Generic `line` plans can declare `axis="y"`, but `allocate_slots()` supports only `x`; grouped tidy uses its own allocator and is unaffected.
- Context `allowed_predicates` should stay within verifier-supported names.
- `AlignTaskPlugin.verify_goal()` primarily enforces `at`, stability, hand-empty, and grouped tidy geometry.
