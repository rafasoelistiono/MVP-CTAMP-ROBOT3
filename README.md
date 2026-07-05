# CTAMP Robot - Align Grouped Tidy Wall World

Repo ini menjalankan pipeline CTAMP untuk satu task: `align` pada scene `align_grouped_tidy_wall_world`.

Alur utama:

1. Load context: `contexts/examples/align_grouped_tidy_wall_world.md`
2. Load atau generate TaskPlan JSON: `task_plans/examples/align_grouped_tidy_wall_world.json`
3. Validasi schema, object ID, predicate, dan urutan `pick/place`
4. Alokasi slot grouped tidy dengan obstacle-aware offset
5. Jalankan MuJoCo/OMPL/IK atau deterministic test runner
6. Tulis telemetry summary dan manifest ke `logs/`

## Commands

Generate/validasi plan dari response file:

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

## Important Paths

- `plugins/align_task.py`: align task validation, runtime tuning, progress, goal verification.
- `world/builder.py`: context parser and `WorldState` builder.
- `world/slot_allocator.py`: line and grouped tidy slot allocation.
- `task_planning/`: plan schema, validation, candidate generation, scoring, LLM prompt.
- `execution/runner.py`: generic `pick/place` loop plus align recovery.
- `execution/primitives.py`: adapter from task steps to MuJoCo primitives.
- `configuration/`: typed runtime profiles.
- `telemetry/`: manifest, naming, summary CSV.

Generated artifacts are ignored: `.pytest_cache/`, `__pycache__/`, `.ruff_cache/`, `logs/`, `models/generated/`, `task_plans/generated/`.
