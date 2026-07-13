# Usage

Supported task: `align` for `align_grouped_tidy_wall_world`.

Generate or validate a plan:

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

Run performance v2 without changing v1:

```bash
python -m cli.run_simulation_v2 \
  --config configs/scenes/align_grouped_tidy_wall_world.yaml \
  --output runs/example_v2
```

Run cube stacking v2:

```bash
python -m cli.run_stacking_v2 \
  --config configs/scenes/stacking_wall_world_v2.yaml \
  --output runs/stacking_v2
```

The continuous stack order is largest to smallest: `c6`, `c5`, `c4`, `c3`, `c2`, `c1`.

Generate only preview configs and planned safe/final positions:

```bash
python -m cli.run_stacking_v2 --dry-run --output runs/stacking_v2_plan
```

Use `--use-adaptive-cache` when you want cache-backed candidate ranking. Generated logs go to `logs/` by default.
