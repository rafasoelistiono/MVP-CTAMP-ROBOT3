# File Reference

| Path | Purpose |
| --- | --- |
| `contexts/examples/align_grouped_tidy_wall_world.md` | Source scene/context. |
| `task_plans/examples/align_grouped_tidy_wall_world.json` | Reference align TaskPlan. |
| `plugins/align_task.py` | Align plugin. |
| `plugins/registry.py` | Align-only plugin registry. |
| `task_planning/types.py` | Plan dataclasses and allowed actions/predicates. |
| `task_planning/validator.py` | Deterministic plan gates. |
| `world/builder.py` | Context parser. |
| `world/slot_allocator.py` | Line/grouped tidy slots. |
| `execution/runner.py` | `pick/place` execution loop and align recovery. |
| `execution/primitives.py` | MuJoCo primitive adapter. |
| `configuration/types.py` | Runtime config schema. |
| `configuration/defaults.py` | Built-in runtime profiles. |
| `cli/generate_plan.py` | LLM/response-file plan generation entrypoint. |
| `cli/run_simulation.py` | Simulation entrypoint. |
| `tests/` | Unit/integration coverage for align flow. |
