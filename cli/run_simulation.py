from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

from adaptive.event_log import EventLog
from adaptive.hint_cache import HintCache
from configuration import (
    DEFAULT_PROFILE_REGISTRY,
    activate_runtime_config,
    load_runtime_config,
)
from execution.primitives import MuJoCoExecutorPrimitives
from execution.runner import TaskRunner
from task_planning.loader import load_plan
from task_planning.validator import validate
from plugins.registry import DEFAULT_REGISTRY, PluginRegistry
from scene import prepare_scene_variant
from telemetry import write_run_manifest, write_summary_csv
from telemetry.naming import (
    infer_experiment_label,
    normalize_experiment_label,
    with_experiment_label,
)
from world.builder import build_world_state
from world.slot_allocator import allocate_slots, validate_slots


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and execute a pre-generated TaskPlan in MuJoCo."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--log-dir", default=ROOT_DIR / "logs", type=Path)
    parser.add_argument(
        "--runtime-profile",
        default="auto",
        choices=("auto",) + DEFAULT_PROFILE_REGISTRY.names(),
        help="Typed code profile. 'auto' selects obstacle/conservative from scene.",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="Optional strict TOML overlay for model/tuning parameters.",
    )
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override profile viewer setting.",
    )
    parser.add_argument(
        "--plugin-package",
        default="plugins",
        help="Trusted package containing deterministic *_task.py plugins.",
    )
    parser.add_argument(
        "--plan-source",
        choices=("unspecified", "original_no_llm", "response_file", "llm_generated"),
        default="unspecified",
        help="Plan provenance recorded in benchmark CSV/manifest.",
    )
    parser.add_argument(
        "--benchmark-role",
        choices=("reference", "candidate"),
        default="candidate",
        help="Use 'reference' only for the accepted original baseline.",
    )
    parser.add_argument(
        "--benchmark-label",
        default="",
        help="Stable experiment label used to group reference and candidate runs.",
    )
    parser.add_argument(
        "--experiment-label",
        default="",
        help="Filename label; inferred from a labeled plan filename when omitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()

    # No MuJoCo/executor import is allowed above this line or before all gates pass.
    world = build_world_state(args.context)
    plan = load_plan(args.plan)
    experiment_label = normalize_experiment_label(args.experiment_label)
    if not experiment_label:
        experiment_label = infer_experiment_label(
            args.plan,
            scene_id=plan.scene_id,
            task=plan.task,
        )
    if args.scene != world.variant:
        raise ValueError(
            f"--scene {args.scene!r} does not match context variant {world.variant!r}"
        )
    if plan.scene_id != world.scene_id:
        raise ValueError(
            f"plan scene_id {plan.scene_id!r} does not match context {world.scene_id!r}"
        )
    if plan.task != world.task_name:
        raise ValueError(
            f"plan task {plan.task!r} does not match context {world.task_name!r}"
        )
    validate(plan, world.all_object_ids(), world.allowed_predicates)
    registry = (
        DEFAULT_REGISTRY
        if args.plugin_package == "plugins"
        else PluginRegistry.discover(args.plugin_package)
    )
    plugin = registry.get(plan.task)
    plugin.validate_plan(plan, world)
    slot_config = plugin.make_slot_config(plan, world)
    slots = allocate_slots(slot_config, len(plan.target_objects))

    profile_name = args.runtime_profile
    if profile_name == "auto":
        profile_name = "conservative" if world.variant.endswith("_no_obs") else "obstacle"
    runtime_config = load_runtime_config(
        profile_name,
        config_file=args.runtime_config,
        enable_viewer=args.viewer,
    )
    runtime_config = plugin.configure_runtime(plan, world, runtime_config).validate()
    validate_slots(
        slots,
        world,
        obstacle_buffer_m=runtime_config.safety.target_obstacle_buffer_m,
    )
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_tag = with_experiment_label(run_id, experiment_label)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    scene_file = prepare_scene_variant(
        world.variant,
        base_model_file=runtime_config.model.xml_path,
        object_states=world.objects,
        obstacle_states=world.obstacles,
    )
    runtime_config = replace(
        runtime_config,
        model=replace(runtime_config.model, xml_path=scene_file),
    ).validate()
    backend_event_path = args.log_dir / f"motion_{plan.task}_{world.variant}_{run_tag}_events.csv"
    runner_event_path = args.log_dir / f"events_{plan.task}_{world.variant}_{run_tag}.csv"
    runtime_config = replace(
        runtime_config,
        telemetry=replace(
            runtime_config.telemetry,
            event_log_csv=str(backend_event_path),
            scenario_type="static",
            obstacle_mode=(
                "no_obs" if world.variant.endswith("_no_obs") else "obs"
            ),
        ),
    ).validate()
    manifest_path = write_run_manifest(
        args.log_dir / f"run_{plan.task}_{world.variant}_{run_tag}_manifest.json",
        run_id=run_tag,
        config=runtime_config,
        plan_file=args.plan,
        context_file=args.context,
        scene_id=world.scene_id,
        scene_variant=world.variant,
        task=plan.task,
        plugin_package=args.plugin_package,
        plan_source=args.plan_source,
        benchmark_role=args.benchmark_role,
        benchmark_label=args.benchmark_label or experiment_label,
    )
    activate_runtime_config(runtime_config)

    # The native backend owns process-level simulator state, so import it only
    # after every deterministic validation gate has passed.
    from backends.mujoco import runtime

    started = time.perf_counter()
    result = None
    runtime_error: str | None = None
    try:
        runtime.init_hint_cache(log_dir=str(args.log_dir), scene_filter=world.variant)
        primitives = MuJoCoExecutorPrimitives(runtime)
        runner = TaskRunner(
            plan=plan,
            world=world,
            slots=slots,
            hint_cache=HintCache(
                args.log_dir,
                min_samples=runtime_config.adaptive.min_samples,
                fallback_threshold=runtime_config.adaptive.pinocchio_skip_rate,
            ),
            plugin_registry=registry,
            event_log=EventLog(runner_event_path, run_tag),
            primitives=primitives,
            runtime_config=runtime_config,
        )
        result = runner.run()
    except Exception as exc:
        runtime_error = f"{exc.__class__.__name__}:{exc}"
    finally:
        runtime.shutdown_runtime()

    duration_ms = int((time.perf_counter() - started) * 1000)
    failures = (
        [{"failure_reason": reason} for reason in result.failure_reasons]
        if result is not None
        else [{"failure_reason": runtime_error or "runtime_failed"}]
    )
    success = bool(result and result.success and not runtime_error)
    summary_path = write_summary_csv(
        task_name=f"task_plan_{plan.task}",
        scene_key=world.variant,
        summary={
            "success": success,
            "objects_moved": result.moved_count if result else 0,
            "objects_total": len(plan.target_objects),
            "failed": failures,
            "duration_ms": duration_ms,
            "llm_used": args.plan_source in {"response_file", "llm_generated"},
            "plan_source": args.plan_source,
            "benchmark_role": args.benchmark_role,
            "benchmark_label": args.benchmark_label or experiment_label,
            "experiment_label": experiment_label,
            "run_id": run_id,
            "plan_file": str(args.plan),
            "runtime_profile": runtime_config.name,
            "runtime_config_file": str(args.runtime_config or ""),
            "run_manifest": str(manifest_path),
        },
        log_dir=args.log_dir,
    )
    print(
        f"success={success} moved={result.moved_count if result else 0}/"
        f"{len(plan.target_objects)} summary={summary_path}"
    )
    if runtime_error:
        print(f"runtime_error={runtime_error}", file=sys.stderr)
    return 0 if success else 1


def cli() -> None:
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR before simulation start: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    cli()
