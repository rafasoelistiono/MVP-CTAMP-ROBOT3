from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

from backends.adaptive.event_log import EventLog
from backends.adaptive.hint_cache import HintCache
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
from world.slot_allocator import (
    allocate_grouped_align_slots,
    allocate_slots,
    validate_slots,
)


def _collision_count(event_path: Path) -> int:
    if not event_path.exists():
        return 0
    with event_path.open(newline="", encoding="utf-8") as stream:
        return sum(
            1
            for row in csv.DictReader(stream)
            if row.get("collision_pair")
            or "collision" in str(row.get("failure_reason", "")).lower()
        )


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
    parser.add_argument(
        "--robust-align",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable robust candidate planning for align tasks.",
    )
    parser.add_argument(
        "--use-adaptive-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable adaptive cache-backed heuristic for align candidate ranking.",
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
    if world.grouped_tidy and world.grouped_tidy.enabled:
        slots = allocate_grouped_align_slots(world, world.grouped_tidy)
    else:
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
    if args.use_adaptive_cache:
        from configuration.types import AlignCacheConfig
        runtime_config = replace(
            runtime_config,
            align_cache=AlignCacheConfig(use_adaptive_cache=True),
        ).validate()
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
        goal_center=world.goal_center,
        goal_area_size_xy=world.goal_area_size_xy,
        table_size_xy=(
            world.table_x_range[1] - world.table_x_range[0],
            world.table_y_range[1] - world.table_y_range[0],
        ),
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
        task_variant=world.variant,
        challenge_type=world.challenge.type if world.challenge else "",
        num_objects=len(world.objects),
        num_groups=len(world.grouped_tidy.groups) if world.grouped_tidy else 0,
        num_obstacles=len(world.obstacles),
    )
    activate_runtime_config(runtime_config)

    # The native backend owns process-level simulator state, so import it only
    # after every deterministic validation gate has passed.
    from backends.mujoco import runtime

    started = time.perf_counter()
    result = None
    runner = None
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
            robust_align=args.robust_align,
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
    robust_telemetry = {}
    if runner is not None and hasattr(runner, "robust_align_telemetry"):
        rat = runner.robust_align_telemetry
        robust_telemetry = {
            "robust_align_candidate_count": rat.candidate_count,
            "robust_align_ranked_costs": list(rat.ranked_costs),
            "robust_align_selected_plan_id": rat.selected_candidate_id or "",
            "selected_candidate_strategy": rat.selected_candidate_strategy or "",
            "robust_align_failed_before_success": rat.failed_before_success,
            "failed_candidate_count": rat.failed_before_success,
            "robust_align_probe_planning_time": rat.probe_planning_time,
            "robust_align_ik_failure_count": rat.ik_failure_count,
            "robust_align_ompl_failure_count": rat.ompl_failure_count,
            "motion_probe_failure_count": (
                rat.ik_failure_count + rat.ompl_failure_count
            ),
            "robust_align_alignment_error": rat.alignment_error,
            "robust_align_spacing_error": rat.spacing_error,
        }
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
            "task_variant": world.variant,
            "challenge_type": world.challenge.type if world.challenge else "",
            "num_objects": len(world.objects),
            "num_groups": len(world.grouped_tidy.groups) if world.grouped_tidy else 0,
            "num_obstacles": len(world.obstacles),
            "planner_name": runtime_config.motion.planner,
            "collision_count": _collision_count(backend_event_path),
            "plan_steps": len(runner.plan.steps) if runner else len(plan.steps),
            "executed_steps": len(result.step_results) if result else 0,
            "retry_count": (
                sum(max(0, step.attempts - 1) for step in result.step_results)
                if result
                else 0
            ),
            "execution_time": round(duration_ms / 1000.0, 3),
            "alignment_error_max": robust_telemetry.get(
                "robust_align_alignment_error", ""
            ),
            "spacing_error_max": robust_telemetry.get(
                "robust_align_spacing_error", ""
            ),
            **robust_telemetry,
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
