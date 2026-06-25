from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    xml_path: Path
    active_arm: str = "left"
    base_xy: tuple[float, float] = (-0.4, 0.0)
    home_q: tuple[float, ...] = (0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854)
    grasp_ready_q: tuple[float, ...] = (0.0, 0.529, 0.0, -1.98, 0.0, 2.495, -0.75)
    elbow_up_q: tuple[float, ...] = (0.0, 0.20, 0.0, -2.40, 0.0, 2.60, -0.75)
    desired_tool_z: tuple[float, float, float] = (0.0, 0.0, -1.0)


@dataclass(frozen=True)
class IKConfig:
    backend: str = "auto"
    require_pinocchio: bool = False
    use_fallback: bool = False
    plan_position_error_m: float = 0.020
    pregrasp_position_error_m: float = 0.030
    plan_orientation_error_rad: float = 0.35
    pregrasp_orientation_error_rad: float = 0.50
    max_valid_candidates: int = 6
    max_attempts_per_segment: int = 80


@dataclass(frozen=True)
class MotionConfig:
    ompl_enabled: bool = True
    ompl_required: bool = False
    planner: str = "RRTConnect"
    fragile_planner: str = "BITstar"
    time_limit_s: float = 6.0
    state_validity_resolution: float = 0.004
    sampler_range: float = 0.08
    waypoint_step: float = 0.010
    goal_tolerance: float = 0.001
    settle_steps_per_waypoint: int = 14
    final_settle_steps: int = 40


@dataclass(frozen=True)
class GraspConfig:
    grasp_offset_m: float = 0.10
    approach_clearance_m: float = 0.30
    far_pick_xy_m: float = 0.74
    open_grip_m: float = 0.05
    place_release_lift_m: float = 0.005
    stack_release_lift_m: float = 0.008
    release_guide_clearance_m: float = 0.008
    place_settle_before_open_steps: int = 120
    release_guide_steps: int = 240
    release_guide_settle_steps: int = 160
    release_open_steps: int = 320
    release_post_open_settle_steps: int = 160
    # A 66 mm cube contacts each finger near q=0.033 m.  Commanding the old
    # 15-21 mm targets over-compressed the cube, while 28-30 mm left too
    # little normal force during the tall-stack descent. These moderate
    # targets hold reliably and pair with a pre-open clearance that releases
    # pad friction before the gripper moves to its fully open state.
    pick_grip_sequence: tuple[float, ...] = (0.026, 0.025, 0.024)
    pick_offset_sequence_m: tuple[float, ...] = (0.10, 0.10, 0.10)
    pick_clearance_bonus_sequence_m: tuple[float, ...] = (0.0, 0.035, 0.06)
    cylinder_grip_sequence: tuple[float, ...] = (0.014, 0.012, 0.010)
    cylinder_offset_sequence_m: tuple[float, ...] = (0.095, 0.095, 0.085)
    cylinder_retry_min_offset_m: float = 0.095
    cylinder_tipped_center_z_m: float = 0.832
    cylinder_tipped_offset_m: float = 0.075
    cylinder_tipped_grip: float = 0.010
    obstacle_cube_grip: float = 0.026
    obstacle_cylinder_grip: float = 0.012


@dataclass(frozen=True)
class SafetyConfig:
    target_obstacle_buffer_m: float = 0.13
    min_pick_obstacle_clearance_m: float = 0.18
    cautious_obstacle_clearance_m: float = 0.24
    min_pick_object_z_m: float = 0.70
    max_pick_object_xy_m: float = 0.92
    obstacle_contact_tolerance_m: float = 0.003
    finger_movable_contact_tolerance_m: float = 0.018
    table_finger_contact_tolerance_m: float = 0.005
    allow_movable_object_contact: bool = False


@dataclass(frozen=True)
class AdaptiveConfig:
    min_samples: int = 5
    pinocchio_skip_rate: float = 0.70
    near_miss_rate: float = 0.40
    near_miss_factor: float = 1.20
    tolerance_headroom: float = 1.05
    max_tolerance_factor: float = 1.60


@dataclass(frozen=True)
class VerificationConfig:
    at_x_m: float = 0.055
    at_y_m: float = 0.035
    at_z_m: float = 0.050
    on_xy_m: float = 0.045
    on_z_m: float = 0.035
    pick_z_m: float = 0.90
    row_y_spread_m: float = 0.045
    stack_max_tilt_rad: float = 0.35
    stack_max_linear_velocity_mps: float = 0.02
    stack_max_angular_velocity_radps: float = 0.20


@dataclass(frozen=True)
class RecoveryConfig:
    max_stack_rebuilds: int = 3
    staging_clearance_m: float = 0.11
    staging_grid_step_m: float = 0.10
    verification_settle_steps: int = 60


@dataclass(frozen=True)
class TelemetryConfig:
    event_log_csv: str = ""
    console: bool = True
    flush_every: int = 1
    scenario_type: str = "static"
    obstacle_mode: str = "unknown"


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    model: ModelConfig
    ik: IKConfig = IKConfig()
    motion: MotionConfig = MotionConfig()
    grasp: GraspConfig = GraspConfig()
    safety: SafetyConfig = SafetyConfig()
    adaptive: AdaptiveConfig = AdaptiveConfig()
    verification: VerificationConfig = VerificationConfig()
    recovery: RecoveryConfig = RecoveryConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    enable_viewer: bool = True

    def validate(self) -> "RuntimeConfig":
        errors: list[str] = []
        if not self.name.strip():
            errors.append("name must not be empty")
        if self.model.active_arm not in {"left", "right"}:
            errors.append("model.active_arm must be left or right")
        for field_name, values, expected in (
            ("model.home_q", self.model.home_q, 7),
            ("model.grasp_ready_q", self.model.grasp_ready_q, 7),
            ("model.elbow_up_q", self.model.elbow_up_q, 7),
        ):
            if len(values) != expected:
                errors.append(f"{field_name} must contain {expected} values")
        if self.ik.backend not in {"auto", "pinocchio", "mujoco_dls"}:
            errors.append("ik.backend must be auto, pinocchio, or mujoco_dls")
        if not 0 < self.ik.plan_position_error_m <= 0.030:
            errors.append("ik.plan_position_error_m must be in (0, 0.030]")
        if self.ik.pregrasp_position_error_m < self.ik.plan_position_error_m:
            errors.append("pregrasp IK tolerance cannot be stricter than plan tolerance")
        if self.ik.max_valid_candidates <= 0 or self.ik.max_attempts_per_segment <= 0:
            errors.append("IK candidate and attempt limits must be positive")
        if self.motion.time_limit_s <= 0 or self.motion.waypoint_step <= 0:
            errors.append("motion time limit and waypoint step must be positive")
        if not (
            0
            < self.safety.min_pick_obstacle_clearance_m
            <= self.safety.cautious_obstacle_clearance_m
        ):
            errors.append("obstacle clearances must satisfy 0 < min <= cautious")
        if self.safety.target_obstacle_buffer_m < 0:
            errors.append("safety.target_obstacle_buffer_m must be non-negative")
        if not 0 < self.grasp.open_grip_m <= 0.05:
            errors.append("grasp.open_grip_m must be in (0, 0.05]")
        if (
            self.grasp.place_release_lift_m < 0
            or self.grasp.stack_release_lift_m < 0
            or self.grasp.release_guide_clearance_m <= 0
        ):
            errors.append("grasp release tuning must be positive or zero-lift")
        release_step_values = (
            self.grasp.place_settle_before_open_steps,
            self.grasp.release_guide_steps,
            self.grasp.release_guide_settle_steps,
            self.grasp.release_open_steps,
            self.grasp.release_post_open_settle_steps,
        )
        if any(value < 0 for value in release_step_values):
            errors.append("grasp release step counts must be non-negative")
        if len(self.grasp.pick_grip_sequence) != len(self.grasp.pick_offset_sequence_m):
            errors.append("cube grip and offset sequences must have equal length")
        if len(self.grasp.cylinder_grip_sequence) != len(
            self.grasp.cylinder_offset_sequence_m
        ):
            errors.append("cylinder grip and offset sequences must have equal length")
        verifier_values = (
            self.verification.at_x_m,
            self.verification.at_y_m,
            self.verification.at_z_m,
            self.verification.on_xy_m,
            self.verification.on_z_m,
            self.verification.pick_z_m,
            self.verification.row_y_spread_m,
            self.verification.stack_max_tilt_rad,
            self.verification.stack_max_linear_velocity_mps,
            self.verification.stack_max_angular_velocity_radps,
        )
        if any(value <= 0 for value in verifier_values):
            errors.append("verification values must be positive")
        if self.recovery.max_stack_rebuilds < 0:
            errors.append("recovery.max_stack_rebuilds must be non-negative")
        if self.recovery.verification_settle_steps < 0:
            errors.append("recovery.verification_settle_steps must be non-negative")
        if (
            self.recovery.staging_clearance_m <= 0
            or self.recovery.staging_grid_step_m <= 0
        ):
            errors.append("recovery staging values must be positive")
        if self.telemetry.flush_every <= 0:
            errors.append("telemetry.flush_every must be positive")
        if errors:
            raise ValueError("invalid RuntimeConfig: " + "; ".join(errors))
        return self
