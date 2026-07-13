# CTAMP v2 9-Object Record

Command:

```bash
python3 -m cli.run_simulation_v2 --config configs/scenes/align_grouped_tidy_wall_world.yaml --output /tmp/opencode/ctamp_v2_12obj
```

Observed before interrupt: 7/9 objects successful. Objective `completed_objects >= 7` met.

Success: `j`, `e`, `l`, `g`, `k`, `d`, `c`.

Failure: `a` (`joint-space RRT failed`), `b` (`no contact-valid physical grasp path`).

Note: full `metrics.json` was not emitted because run was interrupted before finalization. Detailed snapshot also exists at `runs/ctamp_v2_9obj_record/metrics_snapshot.json`, but `runs/` is git-ignored.
