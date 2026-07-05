# Architecture

Pipeline is align-only:

1. `world.builder.build_world_state()` parses context markdown.
2. `task_planning.loader.load_plan()` parses TaskPlan JSON.
3. `task_planning.validator.validate()` enforces schema, object references, predicates, and `pick/place` sequence.
4. `plugins.registry.DEFAULT_REGISTRY` resolves the `align` plugin.
5. `world.slot_allocator.allocate_grouped_align_slots()` creates obstacle-aware tidy slots.
6. `execution.runner.TaskRunner` executes `pick/place`, verifies observed predicates, and retries align recovery when needed.
7. `telemetry` writes manifest and summary files.

Runtime profiles live in `configuration/`. `obstacle` is the default profile for the wall scene.
