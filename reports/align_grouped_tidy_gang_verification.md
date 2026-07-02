# align_grouped_tidy_gang Verification Report

## Summary

* Overall status: PARTIAL
* Commit hash: `b3f41d8d47e8f9475091d95e35ee721c899cc106`
* Date/time: `2026-07-02T22:12:17+07:00`
* Runtime available: yes
* MuJoCo available: yes (`3.9.0`)
* OMPL available: yes
* Final recommendation: ready for runtime tuning, not ready to merge as a fully passing end-to-end challenge

Static, offline, deterministic, and regression checks pass. Native MuJoCo loads the
12 movable objects and two fixed tall obstacles, and both RRTConnect and BITstar
produce motion plans. The full task does not complete: RRTConnect encounters IK
failures, while BITstar additionally reaches a real narrow-gang collision that the
existing collision policy correctly blocks.

## Changed files reviewed

* `contexts/examples/align_grouped_tidy_gang.md`
* `plans/examples/align_grouped_tidy_gang_valid.json`
* `world/builder.py`, `world/state.py`, `world/slot_allocator.py`
* `plugins/align_task.py`
* `task_planning/validator.py`, `task_planning/candidate_generator.py`
* `execution/verifier.py`, `execution/runner.py`
* `scene/builder.py`
* `telemetry/summary.py`, `telemetry/run_manifest.py`
* `cli/run_simulation.py`
* `backends/mujoco/runtime.py`, `backends/mujoco/collision.py`
* Four grouped challenge test files under `tests/`

Guardrail review:

* `plugins/protocol.py` is unchanged.
* No `tidy_up` task type exists; the task remains `align` with variant
  `align_grouped_tidy_gang`.
* No parallel collision checker or telemetry subsystem was added.
* Pick/place primitives are unchanged.
* The localized runtime change was necessary because object discovery only accepted
  `cube*`/`circle*` names and rejected the challenge's `a-l` body names.

## Test results

| Phase | Status | Notes |
| --- | --- | --- |
| Static compile/lint | PASS/SKIP | `compileall` passes; Ruff and mypy are not installed, so both were skipped. |
| Slot allocator | PASS | 9/9; 12 slots, four groups, bounds/reach/obstacle checks, physical minimum separation 0.080 m. |
| Validator | PASS | 9/9 plus direct unknown-slot rejection; valid plan accepted and invalid assignments/sequences rejected. |
| Scene builder | PASS | 9/9; generated XML has 12 movable bodies, two collision geoms, and zero joints on both tall obstacles. |
| Regression | PASS | Baseline align 61, stack 17, pyramid 5; full suite 170/170. |
| Offline plan validation | PASS | Loader and generic/plugin validators accept the 24-step deterministic plan. |
| Smoke world-slot | PASS | 12 objects, four groups, two obstacles, 12 valid slots. |
| Full simulation | FAIL | Native execution starts; first relevant failure is IK at the left grouped slot, followed by repeated pick/grasp IK failure. |
| Robust-align | PARTIAL | Three unique candidates, fixed assignments, candidate confirmed; native execution fails on `pick(f)` and was bounded at 90 s. |
| Planner comparison | PARTIAL | Both planners execute; neither completes. BITstar records blocked narrow-gang contacts. |
| Telemetry | PARTIAL | Required summary fields are populated in a completed failure run; tuning metrics remain empty where no producer/result exists. |

The bare `pytest` executable in this environment failed collection because it did
not put the repository root on `sys.path`. All authoritative results use
`python -m pytest`, which is the repository interpreter invocation.

## Smoke world-slot result

| Check | Expected | Actual | Status |
| --- | ---: | ---: | --- |
| objects | 12 | 12 | PASS |
| groups | 4 | 4 | PASS |
| obstacles | 2 | 2 | PASS |
| slots | 12 | 12 | PASS |
| invalid_slots | 0 | 0 | PASS |

No initial object-object or object-obstacle overlap was found. The final slot layout
also has no physical overlap; its minimum center distance is 0.080 m for 0.066 m cubes.

## Offline deterministic plan

File: `plans/examples/align_grouped_tidy_gang_valid.json`

* Loader: PASS
* Generic validator: PASS
* Align plugin validator: PASS
* Plan length: 24 steps
* Assignment order: `bdf`, `hjl`, `ace`, `gik`

## Full simulation

Command:

```bash
python -m cli.run_simulation \
  --context contexts/examples/align_grouped_tidy_gang.md \
  --plan plans/examples/align_grouped_tidy_gang_valid.json \
  --scene align_grouped_tidy_gang \
  --no-viewer \
  --log-dir logs/verification_align_grouped_tidy_gang/rrtconnect_fixed
```

Status: FAIL (`IK_FAIL`; run stopped after repeated deterministic failure rather
than weakening IK limits).

First failing target edge:

```text
place(-0.225, 0.260, 0.833) preplace
best_pos=0.0205, pos_limit=0.0200
failure_reason=ik_error_above_limit
```

Subsequent blocking failure:

```text
pick(f) grasp
failure_reason=ik_error_above_limit / expected_effect_not_observed
```

No verifier, safety, IK, or collision tolerance was relaxed.

## Robust-align check

Deterministic candidate check:

| Candidate strategy | Assignment fixed | Geometric confirmation |
| --- | --- | --- |
| `grouped_nearest_first` | yes | feasible |
| `grouped_nearest_to_slot` | yes | feasible |
| `grouped_random` | yes | feasible |

`grouped_obstacle_aware` was generated but deduplicated because its ordering matched
another candidate. Every retained candidate preserves all 12 required object-to-slot
assignments; only execution order changes.

Native command added `--robust-align` to the full simulation command and used a
90-second bound. It logged three candidates and confirmed `candidate_0`
(`grouped_nearest_first`), then failed `pick(f)` with
`expected_effect_not_observed` after IK grasp failures. Native robust status is
therefore PARTIAL/IK_FAIL, not PASS.

## Planner comparison

RRTConnect used the default obstacle profile. BITstar used the supported
`--runtime-config` path because the CLI has no direct `--planner` flag; both
`motion.planner` and `motion.fragile_planner` were set to BITstar.

| Metric | RRTConnect | BITstar |
| --- | ---: | ---: |
| success | no (stopped after repeated failure) | no (90 s bound) |
| collision-related event rows | 1 | 13 |
| plan_steps | 24 | 24 |
| executed STEP rows | 6 | 4 |
| retry_count observed | 1 | 0 |
| replan_count | unavailable | unavailable |
| observed runtime | 113.3 s | 86.0 s |
| alignment_error_max | unavailable | unavailable |
| spacing_error_max | unavailable | unavailable |
| failed_candidate_count | n/a (standard run) | n/a (standard run) |
| planner_name | `RRTConnect` | `BITstar` |

BITstar successfully solves OMPL segments but then encounters contacts such as:

```text
robot-env contact: tall_obs_right <-> left_finger
failure_reason=collision_at_waypoint_0
```

The contact is blocked by the existing collision policy, as required.

## Telemetry availability

The completed failure run under
`logs/verification_align_grouped_tidy_gang/rrtconnect/` produced summary CSV,
motion events, runner events, and a run manifest. The longer fixed/robust/planner
runs produced manifests and event CSVs but no final summary because they were
intentionally interrupted or bounded after repeated runtime failures.

| Field | Present | Populated | Notes |
| --- | --- | --- | --- |
| task_variant | yes | yes | `align_grouped_tidy_gang` |
| challenge_type | yes | yes | `dual_tall_obstacle_gang` |
| num_objects | yes | yes | 12 |
| num_groups | yes | yes | 4 |
| num_obstacles | yes | yes | 2 |
| planner_name | yes | yes | RRTConnect in completed summary; BITstar appears in motion CSV. |
| success | yes | yes | False for completed failure run. |
| collision_count | yes | yes | Event-derived; zero in completed pre-execution failure summary. |
| plan_steps | yes | yes | 24 |
| executed_steps | yes | yes | 1 in completed early failure summary. |
| retry_count | yes | yes | 2 in completed early failure summary. |
| replan_count | yes | no | No current producer. |
| execution_time | yes | yes | Seconds; completed early failure was 0.019 s. |
| alignment_error_mean/max | yes | no | No completed robust result. |
| spacing_error_mean/max | yes | no | No completed robust result. |
| selected_candidate_strategy | yes | conditional | Populated only when robust run reaches summary writing. |
| failed_candidate_count | yes | conditional | Zero/non-applicable in standard run. |
| motion_probe_failure_count | yes | conditional | Derived from robust IK + OMPL probe failures. |

The manifest challenge block is populated with variant, challenge type, and all
three counts. Minimal required fields (`success`, `collision_count`, `plan_steps`,
`execution_time`) are present and populated.

## Failure classification

### SLOT_ALLOCATION_FAIL — fixed

* Command: initial full CLI command.
* Cause: `cli/run_simulation.py` discarded grouped slot allocation and regenerated
  generic `align_slot_*` positions.
* Patch: route enabled grouped variants through `allocate_grouped_align_slots`.

### SCENE_BUILD_FAIL — fixed

* Evidence: generated obstacle bodies initially had free joints and moved before task execution.
* Cause: fixed obstacle logic only recognized variant names ending in `_long_obs`.
* Patch: make both challenge obstacles jointless/static and assert this in the scene test.

### EXECUTION_PRIMITIVE_FAIL — fixed

* Evidence: `pick(b) -> unknown_object` despite body `b` existing in XML.
* Cause: runtime movable-body discovery only accepted `cube*`/`circle*` names.
* Patch: discover movable bodies from MuJoCo free joints and classify `_obs` names as obstacles.

### IK_FAIL — remaining

* Command: fixed RRTConnect full simulation and robust-align run.
* First failing functions: runtime preplace IK and later `pick(f)` grasp IK.
* Cause: edge/initial poses are marginal or infeasible under current strict IK settings.
* Patch applied: none; this requires runtime/context tuning and must not be solved by relaxing safety gates blindly.

### COLLISION_FAIL — remaining for BITstar

* Command: BITstar runtime-config comparison.
* First relevant collision: `tall_obs_right <-> left_finger` at trajectory waypoint 0.
* Patch applied: none; the collision checker correctly rejects the path.

## Acceptance checklist

* [x] Context can be parsed
* [x] WorldState recognizes align_grouped_tidy_gang
* [x] 12 objects are available
* [x] 2 tall obstacles are available
* [x] 12 grouped slots are generated
* [x] Validator rejects wrong group assignment
* [x] Offline valid plan passes
* [x] Baseline align regression passes
* [x] Stack regression passes
* [x] Pyramid regression passes
* [x] Scene builder passes
* [ ] Simulation succeeds or is explicitly skipped due runtime unavailable
* [ ] MuJoCo final verifier confirms grouped tidy state (deterministic integration verifier passes)
* [x] Telemetry contains required metrics

## Add When decisions

### 1. More complex overlap resolution

* Needed now: no
* Evidence: deterministic allocator finds all 12 slots with 0.080 m minimum center distance and zero invalid slots.
* Recommended action: keep the current bounded offset search; revisit only if new group sizes or obstacle layouts cannot allocate safely.

### 2. Obstacle-aware motion probe metrics

* Needed now: yes, for the next runtime-tuning iteration
* Evidence: geometric probes accept candidates, while native BITstar reaches a finger-obstacle collision.
* Recommended action: expose existing native probe clearance/collision evidence in summary telemetry; do not add a parallel probe subsystem.

### 3. Planner comparison logging

* Needed now: no additional subsystem
* Evidence: `planner_name`, motion events, duration, and collision events already distinguish both runs.
* Recommended action: obtain completed runs first; add only missing aggregate fields to the existing summary writer if later analysis requires them.

### 4. Motion probe failure count telemetry

* Needed now: yes
* Evidence: robust failure diagnosis needs a compact aggregate.
* Recommended action: applied by mapping the existing robust IK and OMPL failure counters into `motion_probe_failure_count`.

## Final recommendation

The feature is ready for runtime tuning, not ready to merge as an end-to-end passing
challenge. Parsing, allocation, validation, scene construction, offline planning,
deterministic execution, regression safety, and telemetry schema are in good shape.
The remaining work is to make the challenge's initial/goal geometry executable under
the existing IK and collision constraints, then rerun standard, robust-align, and both
planner configurations to completion.
