# Referensi File

Dokumen ini menjelaskan file source yang dipertahankan setelah cleanup.

## Root dan data

| File/directory | Fungsi |
|---|---|
| `pyproject.toml` | Metadata package, dependencies, pytest config, dan CLI entrypoints. |
| `.env.example` | Template credential/endpoint LLM; tidak berisi robot tuning. |
| `contexts/examples/ungroup_obs_align_cubes.md` | Context ungrouped obstacle untuk align empat cube. |
| `contexts/examples/ungroup_obs_stack_cubes.md` | Context ungrouped obstacle untuk stack empat cube. |
| `configuration/profiles/models/panda.toml` | Model XML dan reference joint/pose Franka Panda. |
| `configuration/profiles/runtime/conservative.toml` | Fine-tuning normal tanpa obstacle. |
| `configuration/profiles/runtime/obstacle.toml` | Fine-tuning scene obstacle. |
| `configuration/profiles/runtime/verification_strict.toml` | Contoh verifier tolerance yang lebih ketat. |
| `task_plans/examples/ungroup_obs_align_cubes.json` | TaskPlan align obstacle siap dijalankan. |
| `task_plans/examples/ungroup_obs_stack_cubes.json` | TaskPlan stack obstacle siap dijalankan. |
| `models/panda.xml` | Base MuJoCo model; generated scene dibuat dari file ini. |
| `assets/` | Mesh visual dan collision yang direferensikan `panda.xml`. |

## CLI

| File | Fungsi utama |
|---|---|
| `cli/generate_plan.py` | Parse CLI, build context, panggil/offline-load LLM response, validasi, simpan TaskPlan. |
| `cli/run_simulation.py` | Semua gates, resolve plugin/config/slot/scene, import backend, jalankan TaskRunner, tulis artefak. |

## Plan

| File | Fungsi utama |
|---|---|
| `task_planning/types.py` | Immutable `TaskPlan`, `Step`, dan `SlotConfig`; daftar action/predicate core. |
| `task_planning/loader.py` | Strict JSON-to-dataclass conversion dan penolakan field asing. |
| `task_planning/validator.py` | Gate schema, object whitelist, predicate whitelist, dan logical action sequence. |
| `task_planning/generator.py` | System prompt, provider adapter, HTTP request, dan parsing JSON response LLM. |

## World dan scene

| File | Fungsi utama |
|---|---|
| `world/state.py` | Immutable object, obstacle, dan world contracts. |
| `world/builder.py` | Strict context Markdown parser; hitung reachability/proximity. |
| `world/slot_allocator.py` | Pure line/tower target allocation dan target safety check. |
| `scene/builder.py` | Scene aliases/variants, MuJoCo XML generation, object/obstacle geometry. |

## Configuration

| File | Fungsi utama |
|---|---|
| `configuration/types.py` | Typed immutable schema untuk semua runtime parameter dan invariants. |
| `configuration/defaults.py` | Built-in conservative/obstacle defaults dan profile registry. |
| `configuration/loader.py` | Strict TOML overlay, path resolution, unknown field/type rejection. |
| `configuration/runtime.py` | Aktivasi dan akses satu resolved config sebelum backend import. |
| `configuration/profiles/models/*.toml` | Override model robot dan reference pose. |
| `configuration/profiles/runtime/*.toml` | Runtime tuning yang dapat diedit tanpa mengubah kode. |

## Plugin

| File | Fungsi utama |
|---|---|
| `plugins/protocol.py` | `TaskPlugin` structural contract. |
| `plugins/registry.py` | Deterministic trusted package discovery dan API-version validation. |
| `plugins/align_task.py` | Align-specific validation, config hook, dan final row verification. |
| `plugins/stack_task.py` | Stack dependencies, HOME ready-pose policy, dan tower verification. |

## Execution

| File | Fungsi utama |
|---|---|
| `execution/runner.py` | Generic step loop, verification, staging, bounded stack suffix rebuild, logging, final result. |
| `execution/primitives.py` | Backend protocol/result serta MuJoCo backend adapter. |
| `execution/verifier.py` | `at`, `on`, `clear`, `holding`, `handempty`, dan row checks dari live pose. |
| `execution/recovery.py` | Bounded retry/replan/abort policy; obstacle failure selalu fatal. |

## Backend adaptive dan telemetry

| File | Fungsi utama |
|---|---|
| `backends/adaptive/hint_cache.py` | Historical hint untuk backend, IK tolerance, dan grasp profile. |
| `backends/adaptive/event_log.py` | Structured append-only TaskRunner CSV. |
| `telemetry/run_manifest.py` | Full resolved config, file hashes, platform, dan provenance JSON. |
| `telemetry/summary.py` | Compact result/failure summary CSV. |

## MuJoCo backend

| File | Fungsi utama |
|---|---|
| `backends/mujoco/runtime.py` | Model/viewer lifecycle, IK candidates, pick/place macro, trajectory execution. |
| `backends/mujoco/ompl_backend.py` | 7D joint state space, validity callback, planner portfolio, path densification. |
| `backends/mujoco/collision.py` | Robot/environment contact classification dan allowed-contact tolerances. |
| `backends/mujoco/ik_diagnostics.py` | IK failure taxonomy, candidate scoring, and joint-limit checks. |
| `backends/mujoco/adaptive_hints.py` | Backend-internal historical hints untuk compatibility executor. |
| `backends/mujoco/trace.py` | Detailed native backend event CSV dan console trace. |

## Tests

| File | Coverage |
|---|---|
| `tests/test_plan_validator.py` | Empat plan gates dan strict schema. |
| `tests/test_world_builder.py` | Required context, obstacle/no-obstacle, computed reachability. |
| `tests/test_slot_allocator.py` | Exact align/stack targets. |
| `tests/test_scene_manager.py` | Scene aliases, XML variants, long obstacle, summary provenance. |
| `tests/test_runtime_config.py` | Profiles, TOML strictness, no tuning from env, registry. |
| `tests/test_registry.py` | Plugin discovery/API and stack-specific config hook. |
| `tests/test_verifier.py` | Observed predicate tolerances. |
| `tests/test_hint_cache.py` | Cold start, backend fallback, verifier isolation. |
| `tests/test_recovery.py` | Retry bounds dan fatal obstacle policy. |
| `tests/test_run_manifest.py` | Hashes dan full resolved config provenance. |
| `tests/test_ik_diagnostics.py` | Low-level error classification/ranking. |
| `tests/integration/test_task_runner.py` | Generic align dan stack flow dengan fake backend. |
| `tests/integration/test_simulation_examples.py` | Dua obstacle examples dan injected stack-fall suffix rebuild. |
