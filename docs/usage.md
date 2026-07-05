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

Use `--use-adaptive-cache` when you want cache-backed candidate ranking. Generated logs go to `logs/` by default.
