# CTAMP Robot

Target repo now uses the migrated `ctamp` pipeline from `ctamp_learned_heuristic`.

Main flow:

1. Read scene YAML or `CONTEXT.MD` adapter.
2. Build grouped tidy slots and MuJoCo world.
3. Run source motion probe, Panda IK, TMM A* search, cost, planning, and learning modules.
4. Write `final_plan.json`, `metrics.json`, `challenge_ablation.json`, `scene_summary.json`, and `OBSERVATION.md`.

Commands:

```bash
python -m cli.run_simulation --context contexts/examples/align_grouped_tidy_wall_world.md --output runs/example
python -m cli.run_simulation --config configs/scenes/align_grouped_tidy_wall_world.yaml --output runs/example_yaml
python -m cli.generate_plan --context contexts/examples/align_grouped_tidy_wall_world.md --output task_plans/generated
```

Old TaskPlan/OMPL/adaptive-cache runner was removed. Use `ctamp.*` modules for learning, planning, cost, search, and TMM.
