from __future__ import annotations

import time
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from configuration import get_active_runtime_config

from .collision import (
    CollisionPolicy,
    resolve_obstacle_body_ids,
    tall_box_obstacle_bounds_xy,
)
from .trace import log_event
from .ik_diagnostics import (
    IKAttemptResult,
    IK_GOAL_STATE_INVALID,
    IK_SUCCESS,
    IK_UNREACHABLE,
    OMPL_TIMEOUT,
    classify_ik_attempt,
    joint_limits_valid,
    rank_ik_attempts,
)

try:
    from .ompl_backend import make_default_panda_planner
except ImportError:
    make_default_panda_planner = None

try:
    import pinocchio as pin
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ImportError as _pinocchio_import_error:
    pin = None
    load_robot_description = None
    _PINOCCHIO_IMPORT_ERROR = _pinocchio_import_error
else:
    _PINOCCHIO_IMPORT_ERROR = None


CONFIG = get_active_runtime_config()


# =============================
# LOAD SIMULATION (ONCE)
# =============================

model = mujoco.MjModel.from_xml_path(str(CONFIG.model.xml_path))
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
log_event(
    "SIM_LOAD",
    "OK",
    runtime_profile=CONFIG.name,
    model_name=CONFIG.model.name,
    model_file=str(CONFIG.model.xml_path),
    bodies=model.nbody,
    geoms=model.ngeom,
)

class _NullViewer:
    def sync(self) -> None:
        pass

    def close(self) -> None:
        pass


if CONFIG.enable_viewer:
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -30
    viewer.cam.lookat[:] = [0, 0, 0.7]
else:
    viewer = _NullViewer()
log_event("VIEWER_INIT", "OK", enabled=CONFIG.enable_viewer)

# Planning-side copy. Never use the live MuJoCo state directly inside IK.
_plan_data = mujoco.MjData(model)

# =============================
# ARM SETUP
# =============================

HOME = np.asarray(CONFIG.model.home_q, dtype=float)
GRASP_READY = np.asarray(CONFIG.model.grasp_ready_q, dtype=float)
_DESIRED_Z = np.asarray(CONFIG.model.desired_tool_z, dtype=float)
_DESIRED_X = (
    None
    if CONFIG.model.desired_tool_x is None
    else np.asarray(CONFIG.model.desired_tool_x, dtype=float)
)
GRASP_OFFSET = CONFIG.grasp.grasp_offset_m
APPROACH_CLEARANCE = CONFIG.grasp.approach_clearance_m
OPEN_GRIP = CONFIG.grasp.open_grip_m
MIN_PICK_OBSTACLE_CLEARANCE = CONFIG.safety.min_pick_obstacle_clearance_m
CAUTIOUS_OBSTACLE_CLEARANCE = CONFIG.safety.cautious_obstacle_clearance_m

# Elbow-up null-space reference: joint2=0.20 keeps link2 well above the table.
# Used as a secondary IK seed when the primary converges to an elbow-down
# configuration (joint2 > ~0.55) that would place link2 below z=0.80.
_ELBOW_UP_REF = np.asarray(CONFIG.model.elbow_up_q, dtype=float)
_WALL_RIGHT_REF = np.asarray(CONFIG.model.wall_right_q, dtype=float)
_GATEWAY_HOME_REFERENCE = np.asarray(
    (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785),
    dtype=float,
)
_MAX_GATEWAY_HOME_L1 = 3.5
_OBSTACLE_KINDS = dict(CONFIG.model.obstacle_kinds)

# When fragile objects are present, never use IK fallback for physical motion.
USE_IK_FALLBACK = CONFIG.ik.use_fallback

# Keep OMPL paths dense and the commanded motion slow.
DEFAULT_PLANNER_NAME = CONFIG.motion.planner
DEFAULT_TIME_LIMIT = CONFIG.motion.time_limit_s
DEFAULT_SETTLE_STEPS_PER_WP = CONFIG.motion.settle_steps_per_waypoint
DEFAULT_FINAL_SETTLE_STEPS = CONFIG.motion.final_settle_steps
PICK_GRIP_SEQUENCE = CONFIG.grasp.pick_grip_sequence
PICK_GRASP_OFFSET_SEQUENCE = CONFIG.grasp.pick_offset_sequence_m
PICK_CLEARANCE_BONUS_SEQUENCE = CONFIG.grasp.pick_clearance_bonus_sequence_m
COMPACT_CYLINDER_PICK_GRIP_SEQUENCE = CONFIG.grasp.cylinder_grip_sequence
COMPACT_CYLINDER_PICK_GRASP_OFFSET_SEQUENCE = CONFIG.grasp.cylinder_offset_sequence_m
CYLINDER_RETRY_MIN_GRASP_OFFSET = CONFIG.grasp.cylinder_retry_min_offset_m
CYLINDER_TIPPED_CENTER_Z = CONFIG.grasp.cylinder_tipped_center_z_m
CYLINDER_TIPPED_GRASP_OFFSET = CONFIG.grasp.cylinder_tipped_offset_m
CYLINDER_TIPPED_GRIP = CONFIG.grasp.cylinder_tipped_grip
OBSTACLE_CAUTIOUS_CUBE_GRIP = CONFIG.grasp.obstacle_cube_grip
OBSTACLE_CAUTIOUS_CYLINDER_GRIP = CONFIG.grasp.obstacle_cylinder_grip
HELD_Z_THRESHOLD = 0.90
IK_PLAN_POS_ERR_LIMIT = CONFIG.ik.plan_position_error_m
IK_PREGRASP_POS_ERR_LIMIT = CONFIG.ik.pregrasp_position_error_m
IK_PLAN_ORI_ERR_LIMIT = CONFIG.ik.plan_orientation_error_rad
IK_PREGRASP_ORI_ERR_LIMIT = CONFIG.ik.pregrasp_orientation_error_rad
FAR_PICK_XY_DISTANCE = CONFIG.grasp.far_pick_xy_m
MAX_VALID_IK_CANDIDATES = CONFIG.ik.max_valid_candidates
MAX_IK_ATTEMPTS_PER_SEGMENT = CONFIG.ik.max_attempts_per_segment
MIN_PICK_OBJECT_Z = CONFIG.safety.min_pick_object_z_m
MAX_PICK_OBJECT_XY_DISTANCE = CONFIG.safety.max_pick_object_xy_m

_IK_BACKEND_NAME = "uninitialized"
_PINOCCHIO_ROBOT = None
_PINOCCHIO_MODEL = None
_PINOCCHIO_DATA = None
_PINOCCHIO_FRAME_ID = None
_LAST_SUCCESSFUL_SEED: Optional[np.ndarray] = None

_hint_cache = None          # set by init_hint_cache()
_hint_context: dict = {}    # set by pick()/place() before each IK call sequence


class MotionProbeTimeout(RuntimeError):
    def __init__(self, phase: str, attempted_seeds: int = 0):
        super().__init__(f"motion_probe_timeout:{phase}")
        self.phase = phase
        self.attempted_seeds = attempted_seeds


def _check_deadline(deadline: Optional[float], phase: str, attempted_seeds: int = 0) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise MotionProbeTimeout(phase, attempted_seeds)


def _remaining_probe_segment_time(deadline: float) -> float:
    return max(
        0.001,
        min(
            deadline - time.monotonic(),
            float(CONFIG.motion.probe_segment_time_limit_s),
        ),
    )


def _runtime_config_fingerprint() -> str:
    model_path = Path(CONFIG.model.xml_path).resolve()
    try:
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    except OSError:
        model_sha256 = "unreadable"
    payload = {
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "runtime_profile": CONFIG.name,
        "active_arm": ACTIVE_ARM,
        "base_xy": [float(v) for v in CONFIG.model.base_xy],
        "base_z": float(CONFIG.model.base_z),
        "joint_limits": np.asarray(arm_ranges, dtype=float).round(8).tolist(),
        "ik_backend": CONFIG.ik.backend,
        "require_pinocchio": bool(CONFIG.ik.require_pinocchio),
        "plan_position_error_m": float(CONFIG.ik.plan_position_error_m),
        "pregrasp_position_error_m": float(CONFIG.ik.pregrasp_position_error_m),
        "plan_orientation_error_rad": float(CONFIG.ik.plan_orientation_error_rad),
        "pregrasp_orientation_error_rad": float(CONFIG.ik.pregrasp_orientation_error_rad),
        "desired_tool_z": [float(v) for v in CONFIG.model.desired_tool_z],
        "desired_tool_x": CONFIG.model.desired_tool_x,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

_BASE_ROBOT_BODY_NAMES = (
    "link0",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "hand",
    "left_finger",
    "right_finger",
)

ACTIVE_ARM = CONFIG.model.active_arm
_planner_by_arm = {}
_live_collision_policy = None
ee_id = None
BASE_XY = np.asarray(CONFIG.model.base_xy, dtype=float)
arm_joint_names = []
arm_qpos_adr = np.array([], dtype=int)
arm_dof_adr = np.array([], dtype=int)
arm_ranges = np.zeros((0, 2), dtype=float)
arm_ctrl_adr = np.array([], dtype=int)
finger_ctrl_adr = 7
finger_joint_name = "finger_joint1"
active_robot_body_names = _BASE_ROBOT_BODY_NAMES


def _arm_prefix(arm: str) -> str:
    normalized = (arm or "left").strip().lower()
    if normalized in {"left", "l", ""}:
        return ""
    if normalized in {"right", "r"}:
        return "right_"
    raise ValueError(f"unknown arm '{arm}', expected left/right")


def _body_exists(name: str) -> bool:
    try:
        model.body(name)
        return True
    except KeyError:
        return False


def _joint_ctrl_index(joint_name: str) -> int:
    joint_id = model.joint(joint_name).id
    for actuator_id in range(model.nu):
        if int(model.actuator_trnid[actuator_id][0]) == int(joint_id):
            return actuator_id
    raise RuntimeError(f"no actuator controls joint '{joint_name}'")


def _robot_body_names_for_prefix(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}{name}" for name in _BASE_ROBOT_BODY_NAMES)


def available_arms() -> list[str]:
    arms = ["left"]
    if _body_exists("right_link0"):
        arms.append("right")
    return arms


def set_active_arm(arm: str) -> None:
    global ACTIVE_ARM, BASE_XY, ee_id, arm_joint_names, arm_qpos_adr, arm_dof_adr
    global arm_ranges, arm_ctrl_adr, finger_ctrl_adr, finger_joint_name
    global active_robot_body_names, _live_collision_policy

    prefix = _arm_prefix(arm)
    if prefix and not _body_exists(f"{prefix}link0"):
        raise RuntimeError(f"model has no '{arm}' arm")

    ACTIVE_ARM = "right" if prefix else "left"
    active_robot_body_names = _robot_body_names_for_prefix(prefix)
    arm_joint_names = [f"{prefix}joint{i}" for i in range(1, 8)]
    finger_joint_name = f"{prefix}finger_joint1"

    ee_id = model.body(f"{prefix}hand").id
    BASE_XY = np.asarray(model.body(f"{prefix}link0").pos[:2], dtype=float)
    arm_qpos_adr = np.array([model.joint(n).qposadr[0] for n in arm_joint_names], dtype=int)
    arm_dof_adr = np.array([model.joint(n).dofadr[0] for n in arm_joint_names], dtype=int)
    arm_ranges = np.array([model.joint(n).range for n in arm_joint_names], dtype=float)
    arm_ctrl_adr = np.array([_joint_ctrl_index(n) for n in arm_joint_names], dtype=int)
    finger_ctrl_adr = _joint_ctrl_index(finger_joint_name)
    _live_collision_policy = CollisionPolicy(
        model,
        robot_body_names=active_robot_body_names,
        obstacle_body_names=CONFIG.model.obstacle_body_names,
        obstacle_kinds=CONFIG.model.obstacle_kinds,
    )
    log_event("ARM_SELECT", "OK", arm=ACTIVE_ARM, base_xy=[round(float(v), 4) for v in BASE_XY])


def _initialize_available_arms() -> None:
    arms = available_arms()
    requested = ACTIVE_ARM if ACTIVE_ARM in arms else "left"
    for arm in arms:
        set_active_arm(arm)
        data.qpos[arm_qpos_adr] = HOME
        _set_arm_ctrl(HOME, OPEN_GRIP)
    set_active_arm(requested)

# =============================
# FIND CUBES
# =============================

name_to_cube = {}
_configured_obstacle_names = set(CONFIG.model.obstacle_body_names)
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    joint_count = int(model.body_jntnum[i])
    first_joint = int(model.body_jntadr[i]) if joint_count else -1
    is_free_body = (
        first_joint >= 0
        and int(model.jnt_type[first_joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    is_obstacle = bool(name and name in _configured_obstacle_names)
    if name and is_free_body and not is_obstacle:
        name_to_cube[name] = model.body(name).id

# =============================
# OPTIONAL OBSTACLE MONITORING
# =============================

_obstacle_ids = resolve_obstacle_body_ids(
    model,
    CONFIG.model.obstacle_body_names,
)
_wall_ids = {
    name: body_id
    for name, body_id in _obstacle_ids.items()
    if _OBSTACLE_KINDS.get(name) == "wall"
}

_obstacle_init_z: dict[str, float] = {}
_obstacle_init_xy: dict[str, np.ndarray] = {}


def _init_obstacle_monitoring() -> None:
    mujoco.mj_forward(model, data)
    for name, bid in _obstacle_ids.items():
        pos = data.xpos[bid].copy()
        _obstacle_init_z[name] = float(pos[2])
        _obstacle_init_xy[name] = pos[:2].copy()
        print(f"[init] obstacle '{name}' z0={pos[2]:.4f} xy0=({pos[0]:.3f},{pos[1]:.3f})")
        log_event(
            "OBSTACLE_MONITOR",
            "INIT",
            object_id=name,
            target_xyz=[round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
        )


def _check_obstacles_fallen(context: str) -> None:
    if not _obstacle_ids:
        return
    mujoco.mj_forward(model, data)
    for name, bid in _obstacle_ids.items():
        init_z = _obstacle_init_z.get(name)
        init_xy = _obstacle_init_xy.get(name)
        if init_z is None or init_xy is None:
            continue
        pos = data.xpos[bid]
        z = float(pos[2])
        xy_dist = float(np.linalg.norm(pos[:2] - init_xy))
        z_drop = init_z - z
        if z_drop > 0.06 or xy_dist > 0.08:
            cur_pos = [round(float(pos[0]), 3), round(float(pos[1]), 3), round(z, 3)]
            ee_pos = [round(float(data.xpos[ee_id][0]), 3), round(float(data.xpos[ee_id][1]), 3), round(float(data.xpos[ee_id][2]), 3)]
            print(
                f"[OBSTACLE] {name} DISPLACED during '{context}' | "
                f"pos={cur_pos}  z_drop={z_drop:.3f} m  xy_shift={xy_dist:.3f} m "
                f"| arm_ee={ee_pos}"
            )
            log_event(
                "OBSTACLE_MONITOR",
                "FATAL",
                object_id=name,
                phase=context,
                target_xyz=cur_pos,
                failure_reason="obstacle_displaced",
                z_drop=round(z_drop, 4),
                xy_shift=round(xy_dist, 4),
                ee_pos=ee_pos,
            )
            raise RuntimeError(f"fatal obstacle displacement: {name} during {context}")


def _min_obstacle_xy_distance(pos: np.ndarray) -> float:
    if not _obstacle_ids:
        return float("inf")
    mujoco.mj_forward(model, data)
    xy = np.asarray(pos[:2], dtype=float)
    distances = []
    for bid in _obstacle_ids.values():
        distances.append(float(np.linalg.norm(xy - data.xpos[bid][:2])))
    return min(distances) if distances else float("inf")


def _grouped_wall_bounds() -> tuple[float, float, float, float] | None:
    return tall_box_obstacle_bounds_xy(
        model,
        data,
        _wall_ids.values(),
    )


def _grouped_wall_bypass_waypoints(
    target_xyz: np.ndarray,
) -> tuple[str, tuple[np.ndarray, ...]] | None:
    bounds = _grouped_wall_bounds()
    if bounds is None:
        return None
    min_x, max_x, min_y, max_y = bounds
    current = np.asarray(_ee_xyz(), dtype=float)
    side_tolerance = 0.02
    current_behind = float(current[1]) < min_y - side_tolerance
    target_behind = float(target_xyz[1]) < min_y - side_tolerance
    current_front = float(current[1]) > max_y + side_tolerance
    target_front = float(target_xyz[1]) > max_y + side_tolerance
    current_right = float(current[0]) > max_x
    target_right = float(target_xyz[0]) > max_x
    current_across = not current_behind and (current_front or current_right)
    target_across = not target_behind and (target_front or target_right)
    if not (
        (current_behind and target_across)
        or (current_across and target_behind)
    ):
        return None

    side = "right"
    corridor_x = max_x + 0.24
    behind_y = min_y - 0.12
    front_y = max_y + 0.12
    bypass_z = 1.15
    behind_gateway = np.array([corridor_x, behind_y, bypass_z])
    front_gateway = np.array([corridor_x, front_y, bypass_z])
    target_hover = np.array(
        [float(target_xyz[0]), float(target_xyz[1]), bypass_z]
    )
    if current_behind:
        return side, (behind_gateway, front_gateway, target_hover)
    return side, (front_gateway, behind_gateway, target_hover)


def _validated_wall_right_q(
    planner,
    ignored_body_names: Optional[Sequence[str]] = None,
) -> tuple[bool, str, np.ndarray]:
    q = clip_arm(_WALL_RIGHT_REF.copy())
    if not joint_limits_valid(q, arm_ranges[:, 0], arm_ranges[:, 1]):
        return False, "joint_limits", q
    home_l1 = float(np.sum(np.abs(q - _GATEWAY_HOME_REFERENCE)))
    if home_l1 > _MAX_GATEWAY_HOME_L1:
        return False, f"home_l1_distance:{home_l1:.6f}>{_MAX_GATEWAY_HOME_L1:.6f}", q
    if not planner.is_state_valid_q(q, ignored_body_names=ignored_body_names):
        return False, getattr(planner, "_last_invalid_reason", None) or "collision", q
    ee_xyz = np.asarray(_mujoco_fk_xyz(q), dtype=float)
    bounds = _grouped_wall_bounds()
    if bounds is None:
        return True, "no_wall", q
    min_x, max_x, min_y, max_y = bounds
    dx = max(min_x - ee_xyz[0], 0.0, ee_xyz[0] - max_x)
    dy = max(min_y - ee_xyz[1], 0.0, ee_xyz[1] - max_y)
    clearance = float(np.hypot(dx, dy))
    if clearance < 0.08:
        return False, f"ee_corridor_clearance:{clearance:.4f}", q
    return True, (
        f"ee_corridor_clearance:{clearance:.4f};"
        f"home_l1_distance:{home_l1:.6f}"
    ), q


def _object_lifted(obj: str) -> tuple[bool, float]:
    body_id = name_to_cube.get(obj)
    if body_id is None:
        return False, 0.0
    mujoco.mj_forward(model, data)
    z = float(data.xpos[body_id][2])
    return z > HELD_Z_THRESHOLD, z


def _finger_contacts_object(obj: str) -> set[str]:
    body_id = name_to_cube.get(obj)
    if body_id is None:
        return set()
    mujoco.mj_forward(model, data)
    contacts: set[str] = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body1_id = int(model.geom_bodyid[int(contact.geom1)])
        body2_id = int(model.geom_bodyid[int(contact.geom2)])
        if body1_id != body_id and body2_id != body_id:
            continue
        other_id = body2_id if body1_id == body_id else body1_id
        other_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            other_id,
        )
        if other_name and other_name.endswith(("left_finger", "right_finger")):
            contacts.add(other_name)
    return contacts


# =============================
# OMPL PLANNER
# =============================

_ompl_planner = None
_OMPL_AVAILABLE = CONFIG.motion.ompl_enabled and make_default_panda_planner is not None
log_event(
    "OMPL_INIT",
    "OK" if _OMPL_AVAILABLE else "FAILED",
    enabled=CONFIG.motion.ompl_enabled,
    failure_reason=None if _OMPL_AVAILABLE else "ompl_unavailable",
)
if CONFIG.motion.ompl_required and not _OMPL_AVAILABLE:
    raise RuntimeError(
        "OMPL_REQUIRED=true but OMPL planner bindings are unavailable. "
        "Install OMPL Python bindings in this environment or set OMPL_REQUIRED=false for diagnostics."
    )


def _initialize_ik_backend() -> None:
    global _IK_BACKEND_NAME, _PINOCCHIO_ROBOT, _PINOCCHIO_MODEL, _PINOCCHIO_DATA, _PINOCCHIO_FRAME_ID

    requested = (CONFIG.ik.backend or "auto").strip().lower()
    if requested not in {"auto", "pinocchio", "mujoco_dls"}:
        log_event(
            "IK_INIT",
            "PINOCCHIO_FAILED",
            failure_reason=f"unknown IK backend={CONFIG.ik.backend}",
        )
        requested = "auto"

    if requested in {"auto", "pinocchio"}:
        if pin is None or load_robot_description is None:
            log_event("IK_INIT", "PINOCCHIO_FAILED", failure_reason=str(_PINOCCHIO_IMPORT_ERROR))
            if CONFIG.ik.require_pinocchio or requested == "pinocchio":
                raise RuntimeError(
                    "Pinocchio IK requested but dependencies are unavailable. "
                    "Install pin and robot_descriptions, or set IK_BACKEND=mujoco_dls for development fallback."
                )
        else:
            try:
                _PINOCCHIO_ROBOT = load_robot_description("panda_description")
                _PINOCCHIO_MODEL = getattr(_PINOCCHIO_ROBOT, "model", _PINOCCHIO_ROBOT)
                _PINOCCHIO_DATA = _PINOCCHIO_MODEL.createData()
                for frame_name in ("panda_hand", "panda_link8", "panda_hand_tcp"):
                    frame_id = _PINOCCHIO_MODEL.getFrameId(frame_name)
                    if frame_id < len(_PINOCCHIO_MODEL.frames):
                        _PINOCCHIO_FRAME_ID = frame_id
                        break
                if _PINOCCHIO_FRAME_ID is None:
                    raise RuntimeError("Pinocchio Panda model has no panda_hand-compatible frame")
                _IK_BACKEND_NAME = "pinocchio"
                log_event("IK_INIT", "PINOCCHIO_OK", frame=_PINOCCHIO_MODEL.frames[_PINOCCHIO_FRAME_ID].name)
                return
            except Exception as exc:
                log_event("IK_INIT", "PINOCCHIO_FAILED", failure_reason=str(exc))
                if CONFIG.ik.require_pinocchio or requested == "pinocchio":
                    raise

    _IK_BACKEND_NAME = "mujoco_dls"
    log_event("IK_INIT", "MUJOCO_DLS_FALLBACK", failure_reason="pinocchio_unavailable_or_not_requested")


def _get_ompl_planner():
    global _ompl_planner
    if not _OMPL_AVAILABLE:
        return None
    planner_key = ACTIVE_ARM
    planner = _planner_by_arm.get(planner_key)
    if planner is None:
        planner = make_default_panda_planner(
            model=model,
            data=data,
            planner_name=DEFAULT_PLANNER_NAME,
            fragile_planner_name=CONFIG.motion.fragile_planner,
            time_limit=DEFAULT_TIME_LIMIT,
            state_validity_resolution=CONFIG.motion.state_validity_resolution,
            sampler_range=CONFIG.motion.sampler_range,
            waypoint_step=CONFIG.motion.waypoint_step,
            goal_tolerance=CONFIG.motion.goal_tolerance,
            valid_state_sampler=CONFIG.motion.valid_state_sampler,
            optimization_planner=CONFIG.motion.optimization_planner,
            minimum_obstacle_clearance=CONFIG.safety.planning_obstacle_clearance_m,
            obstacle_body_names=CONFIG.model.obstacle_body_names,
            obstacle_kinds=CONFIG.model.obstacle_kinds,
            robot_body_names=active_robot_body_names,
            arm_joint_names=arm_joint_names,
        )
        if _held_object_name is not None:
            planner.attach_body(
                _held_object_name,
                f"{_arm_prefix(ACTIVE_ARM)}hand",
            )
        _planner_by_arm[planner_key] = planner
    else:
        planner.sync_live_data(data)
    _ompl_planner = planner
    return planner


# =============================
# HELPERS
# =============================

def clip_arm(q: np.ndarray) -> np.ndarray:
    return np.clip(q, arm_ranges[:, 0], arm_ranges[:, 1])


def current_q() -> np.ndarray:
    return data.qpos[arm_qpos_adr].copy()


def cube_null_ref(pos: np.ndarray) -> np.ndarray:
    """GRASP_READY with joint1 aimed at the cube bearing from the arm base."""
    dx = pos[0] - BASE_XY[0]
    dy = pos[1] - BASE_XY[1]
    j1 = float(np.clip(np.arctan2(dy, dx), arm_ranges[0, 0], arm_ranges[0, 1]))
    ref = GRASP_READY.copy()
    ref[0] = j1
    return ref


def _sync_plan_data_from_live() -> None:
    _plan_data.qpos[:] = data.qpos[:]
    _plan_data.qvel[:] = data.qvel[:]
    mujoco.mj_forward(model, _plan_data)


def _round_vec(values, ndigits: int = 4) -> list[float]:
    return [round(float(v), ndigits) for v in np.asarray(values, dtype=float).reshape(-1)]


def _ee_xyz() -> list[float]:
    mujoco.mj_forward(model, data)
    return _round_vec(data.xpos[ee_id], 4)


def _object_xyz(obj: Optional[str]) -> Optional[list[float]]:
    if obj is None or obj not in name_to_cube:
        return None
    mujoco.mj_forward(model, data)
    return _round_vec(data.xpos[name_to_cube[obj]], 4)


def _arm_q() -> list[float]:
    return _round_vec(current_q(), 4)


def _finger_snapshot() -> float:
    try:
        return round(_finger_pos(), 4)
    except Exception:
        return 0.0


def _distance_xy_to_base(pos: Sequence[float]) -> float:
    return round(float(np.linalg.norm(np.asarray(pos[:2], dtype=float) - BASE_XY)), 4)


def _object_pose_failure_reason(pos: Sequence[float]) -> Optional[str]:
    pos_arr = np.asarray(pos, dtype=float).reshape(3)
    if float(pos_arr[2]) < MIN_PICK_OBJECT_Z:
        return "object_displaced_below_table"
    if float(np.linalg.norm(pos_arr[:2] - BASE_XY)) > MAX_PICK_OBJECT_XY_DISTANCE:
        return "object_outside_robot_reach_after_displacement"
    return None


def _rotation_error_vector(
    current_rotation: np.ndarray,
    desired_rotation: np.ndarray,
) -> np.ndarray:
    current_quaternion = np.zeros(4, dtype=float)
    desired_quaternion = np.zeros(4, dtype=float)
    current_inverse = np.zeros(4, dtype=float)
    error_quaternion = np.zeros(4, dtype=float)
    error_vector = np.zeros(3, dtype=float)
    mujoco.mju_mat2Quat(current_quaternion, np.asarray(current_rotation).reshape(-1))
    mujoco.mju_mat2Quat(desired_quaternion, np.asarray(desired_rotation).reshape(-1))
    mujoco.mju_negQuat(current_inverse, current_quaternion)
    mujoco.mju_mulQuat(error_quaternion, desired_quaternion, current_inverse)
    if error_quaternion[0] < 0.0:
        error_quaternion *= -1.0
    mujoco.mju_quat2Vel(error_vector, error_quaternion, 1.0)
    return error_vector


def _desired_orientation_error_from_matrix(
    rot_matrix: np.ndarray,
    target_rotation: Optional[np.ndarray] = None,
) -> float:
    current = np.asarray(rot_matrix, dtype=float).reshape(3, 3)
    desired = (
        _tool_target_rotation(current)
        if target_rotation is None
        else np.asarray(target_rotation, dtype=float).reshape(3, 3)
    )
    return float(np.linalg.norm(_rotation_error_vector(current, desired)))


def _mujoco_fk_error(
    q: Sequence[float],
    target_xyz: Sequence[float],
    target_rotation: Optional[np.ndarray] = None,
) -> tuple[float, float, list[float]]:
    q_arr = clip_arm(np.asarray(q, dtype=float).reshape(-1))
    _plan_data.qpos[:] = data.qpos[:]
    _plan_data.qvel[:] = 0.0
    _plan_data.qpos[arm_qpos_adr] = q_arr
    mujoco.mj_forward(model, _plan_data)
    ee_pos = _plan_data.xpos[ee_id].copy()
    pos_err = float(np.linalg.norm(np.asarray(target_xyz, dtype=float).reshape(3) - ee_pos))
    ori_err = _desired_orientation_error_from_matrix(
        _plan_data.xmat[ee_id].reshape(3, 3),
        target_rotation,
    )
    return pos_err, ori_err, _round_vec(ee_pos, 4)


def _mujoco_fk_xyz(q: Sequence[float]) -> np.ndarray:
    q_arr = clip_arm(np.asarray(q, dtype=float).reshape(-1))
    _plan_data.qpos[:] = data.qpos[:]
    _plan_data.qvel[:] = 0.0
    _plan_data.qpos[arm_qpos_adr] = q_arr
    mujoco.mj_forward(model, _plan_data)
    return _plan_data.xpos[ee_id].copy()


def _log_arm_state(stage: str, status: str, **fields) -> None:
    obj = fields.get("object_id") or fields.get("held_object") or _held_object_name
    fields.setdefault("arm", ACTIVE_ARM)
    fields.setdefault("scenario_type", CONFIG.telemetry.scenario_type)
    fields.setdefault("obstacle_mode", CONFIG.telemetry.obstacle_mode)
    fields.setdefault("ee_xyz", _ee_xyz())
    fields.setdefault("q", _arm_q())
    fields.setdefault("finger_pos", _finger_snapshot())
    fields.setdefault("held_object", _held_object_name)
    if obj is not None and "object_xyz" not in fields:
        fields["object_xyz"] = _object_xyz(str(obj))
    log_event(stage, status, **fields)


def _set_arm_ctrl(q: np.ndarray, grip: float) -> None:
    data.ctrl[arm_ctrl_adr] = clip_arm(np.asarray(q, dtype=float))
    data.ctrl[finger_ctrl_adr] = float(grip)


def _step_sim(steps: int, q: Optional[np.ndarray] = None, grip: Optional[float] = None) -> None:
    for _ in range(steps):
        if q is not None:
            data.ctrl[arm_ctrl_adr] = clip_arm(np.asarray(q, dtype=float))
        if grip is not None:
            data.ctrl[finger_ctrl_adr] = float(grip)
        mujoco.mj_step(model, data)
        viewer.sync()


def _is_table_finger_pair(report) -> bool:
    bodies = {report.body1, report.body2}
    return "table" in bodies and any(str(body).endswith(("left_finger", "right_finger")) for body in bodies)


def _check_live_collision(
    context: str,
    ignored_body_names: Optional[Sequence[str]] = None,
    allow_start_table_finger: bool = False,
) -> bool:
    _live_collision_policy.set_ignored_bodies(ignored_body_names)
    mujoco.mj_forward(model, data)
    report = _live_collision_policy.check_contacts(data)
    if report.valid:
        return True
    if allow_start_table_finger and _is_table_finger_pair(report):
        _log_arm_state(
            "COLLISION_CHECK",
            "IGNORED_START",
            phase=context,
            failure_reason=report.reason,
            collision_pair=[report.body1, report.body2],
            contact_count=int(data.ncon),
            penetration=round(float(getattr(report, "penetration", 0.0)), 5),
            ignored_body_names=list(ignored_body_names or []),
        )
        return True
    print(f"[collision] blocked during {context}: {report.reason}")
    _log_arm_state(
        "COLLISION_CHECK",
        "BLOCKED",
        phase=context,
        failure_reason=report.reason,
        collision_pair=[report.body1, report.body2],
        contact_count=int(data.ncon),
        penetration=round(float(getattr(report, "penetration", 0.0)), 5),
        ignored_body_names=list(ignored_body_names or []),
    )
    return False


def _body_name_for_geom(geom_id: int) -> Optional[str]:
    body_id = int(model.geom_bodyid[int(geom_id)])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def _merge_ignored_body_names(
    *groups: Optional[Sequence[str]],
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for body_name in group or ():
            if body_name not in seen:
                seen.add(body_name)
                merged.append(body_name)
    return merged


def _current_robot_movable_contacts(
    ignored_body_names: Optional[Sequence[str]] = None,
) -> list[str]:
    """Movable bodies already touching the robot in the live simulator state."""
    ignored = set(ignored_body_names or [])
    contacts: list[str] = []
    seen: set[str] = set()
    mujoco.mj_forward(model, data)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body1 = _body_name_for_geom(int(contact.geom1))
        body2 = _body_name_for_geom(int(contact.geom2))
        if body1 in active_robot_body_names and body2 in name_to_cube:
            movable_body = body2
        elif body2 in active_robot_body_names and body1 in name_to_cube:
            movable_body = body1
        else:
            continue
        if movable_body in ignored or movable_body in seen:
            continue
        seen.add(movable_body)
        contacts.append(movable_body)
    return contacts


def _finger_pos() -> float:
    finger_qposadr = model.joint(finger_joint_name).qposadr[0]
    return float(data.qpos[finger_qposadr])


def go_to_cube_ready(cube_pos, steps=400):
    """Deprecated for fragile scenes. Kept only for backward compatibility."""
    target = cube_null_ref(np.asarray(cube_pos, dtype=float))
    for _ in range(steps):
        data.ctrl[arm_ctrl_adr] += 0.01 * (target - data.ctrl[arm_ctrl_adr])
        data.ctrl[finger_ctrl_adr] = OPEN_GRIP
        mujoco.mj_step(model, data)
        viewer.sync()


# =============================
# IK SOLVER (planning only; does NOT move the live robot)
# =============================

def _ik_solve_to(
    target_xyz: Sequence[float],
    null_ref: Optional[np.ndarray] = None,
    q_seed: Optional[Sequence[float]] = None,
    steps: int = 600,
    pos_tol: float = 0.008,
    ori_tol: float = 0.20,
    target_rotation: Optional[np.ndarray] = None,
    deadline: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    if null_ref is None:
        null_ref = GRASP_READY

    q_target = np.array(q_seed if q_seed is not None else current_q(), dtype=float).reshape(-1)
    target_xyz = np.array(target_xyz, dtype=float).reshape(3)

    info = {
        "converged": False,
        "pos_err_norm": None,
        "ori_err_norm": None,
        "iters": 0,
    }

    _plan_data.qpos[:] = data.qpos[:]
    _plan_data.qvel[:] = 0.0
    _plan_data.qpos[arm_qpos_adr] = q_target
    mujoco.mj_forward(model, _plan_data)
    desired_rotation = (
        _tool_target_rotation(_plan_data.xmat[ee_id].reshape(3, 3))
        if target_rotation is None
        else np.asarray(target_rotation, dtype=float).reshape(3, 3)
    )

    for it in range(steps):
        _check_deadline(deadline, "mujoco_dls_iteration")
        _plan_data.qpos[:] = data.qpos[:]
        _plan_data.qvel[:] = 0.0
        _plan_data.qpos[arm_qpos_adr] = q_target
        mujoco.mj_forward(model, _plan_data)

        pos_error = target_xyz - _plan_data.xpos[ee_id]
        R = _plan_data.xmat[ee_id].reshape(3, 3)
        ori_error = _rotation_error_vector(R, desired_rotation)

        pos_norm = float(np.linalg.norm(pos_error))
        ori_norm = float(np.linalg.norm(ori_error))

        info["pos_err_norm"] = pos_norm
        info["ori_err_norm"] = ori_norm
        info["iters"] = it + 1

        if pos_norm < pos_tol and ori_norm < ori_tol:
            info["converged"] = True
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, _plan_data, jacp, jacr, ee_id)
        J_pos = jacp[:, arm_dof_adr]
        J_rot = jacr[:, arm_dof_adr]

        w = 0.5
        J6 = np.vstack([J_pos, w * J_rot])
        err6 = np.concatenate([pos_error, w * ori_error])

        lam = 0.05
        JJT = J6 @ J6.T + lam * np.eye(6)
        J_pinv = J6.T @ np.linalg.solve(JJT, np.eye(6))

        dq_task = J_pinv @ err6
        null_proj = np.eye(7) - J_pinv @ J6
        dq_null = null_proj @ (np.asarray(null_ref, dtype=float) - q_target) * 0.05

        dq = np.clip(dq_task + dq_null, -0.08, 0.08)
        q_target = clip_arm(q_target + 0.015 * dq)

    info["backend"] = "mujoco_dls"
    return clip_arm(q_target), info


def _pinocchio_ik_solve_to(
    target_xyz: Sequence[float],
    q_seed: Optional[Sequence[float]] = None,
    steps: int = 120,
    pos_tol: float = 0.006,
    ori_tol: float = 0.25,
    target_rotation: Optional[np.ndarray] = None,
    deadline: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    if _PINOCCHIO_MODEL is None or _PINOCCHIO_DATA is None or _PINOCCHIO_FRAME_ID is None:
        raise RuntimeError("Pinocchio IK backend is not initialized")

    nq = int(_PINOCCHIO_MODEL.nq)
    nv = int(_PINOCCHIO_MODEL.nv)
    q = np.zeros(nq, dtype=float)
    seed = np.asarray(q_seed if q_seed is not None else current_q(), dtype=float).reshape(-1)
    q[: min(seed.shape[0], nq)] = seed[: min(seed.shape[0], nq)]
    if nq > 7:
        q[7:] = OPEN_GRIP

    target_world_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
    target_xyz = target_world_xyz - _pinocchio_base_translation()
    info = {
        "converged": False,
        "pos_err_norm": float("inf"),
        "ori_err_norm": float("inf"),
        "iters": 0,
        "backend": "pinocchio",
    }

    damping = 1e-4
    pin.forwardKinematics(_PINOCCHIO_MODEL, _PINOCCHIO_DATA, q)
    pin.updateFramePlacements(_PINOCCHIO_MODEL, _PINOCCHIO_DATA)
    initial_rotation = np.asarray(
        _PINOCCHIO_DATA.oMf[_PINOCCHIO_FRAME_ID].rotation
    ).reshape(3, 3)
    desired_rotation = (
        _tool_target_rotation(initial_rotation)
        if target_rotation is None
        else np.asarray(target_rotation, dtype=float).reshape(3, 3)
    )
    orientation_weight = 0.45
    error_history: list[tuple[float, float]] = []
    two_cycle_hits = 0
    for it in range(steps):
        _check_deadline(deadline, "pinocchio_iteration")
        pin.forwardKinematics(_PINOCCHIO_MODEL, _PINOCCHIO_DATA, q)
        pin.updateFramePlacements(_PINOCCHIO_MODEL, _PINOCCHIO_DATA)
        frame = _PINOCCHIO_DATA.oMf[_PINOCCHIO_FRAME_ID]
        pos_error = target_xyz - np.asarray(frame.translation).reshape(3)
        pos_norm = float(np.linalg.norm(pos_error))
        frame_rotation = np.asarray(frame.rotation).reshape(3, 3)
        orientation_error = np.asarray(
            pin.log3(desired_rotation @ frame_rotation.T)
        ).reshape(3)
        ori_norm = float(np.linalg.norm(orientation_error))
        log_event(
            "IK_ITERATION",
            "OK" if pos_norm <= pos_tol and ori_norm <= ori_tol else "RUNNING",
            backend="pinocchio",
            iteration=it + 1,
            pos_err=round(pos_norm, 6),
            ori_err=round(ori_norm, 6),
            pos_limit=pos_tol,
            ori_limit=ori_tol,
        )
        info.update({
            "converged": pos_norm <= pos_tol and ori_norm <= ori_tol,
            "pos_err_norm": pos_norm,
            "ori_err_norm": ori_norm,
            "iters": it + 1,
        })
        if info["converged"]:
            break

        error_state = (pos_norm, ori_norm)
        if len(error_history) >= 2 and np.allclose(
            error_state,
            error_history[-2],
            rtol=0.0,
            atol=1e-5,
        ):
            two_cycle_hits += 1
        else:
            two_cycle_hits = 0
        error_history.append(error_state)
        if two_cycle_hits >= 6:
            info["termination_reason"] = "two_cycle_stagnation"
            log_event(
                "IK_ITERATION",
                "STAGNATION_EXIT",
                backend="pinocchio",
                iteration=it + 1,
                pos_err=round(pos_norm, 6),
                ori_err=round(ori_norm, 6),
                cycle_repetitions=4,
            )
            break

        jac = pin.computeFrameJacobian(
            _PINOCCHIO_MODEL,
            _PINOCCHIO_DATA,
            q,
            _PINOCCHIO_FRAME_ID,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        jpos = jac[:3, :nv]
        jrot = jac[3:6, :nv]
        jacobian_6d = np.vstack((jpos, orientation_weight * jrot))
        error_6d = np.concatenate(
            (pos_error, orientation_weight * orientation_error)
        )
        lhs = jacobian_6d @ jacobian_6d.T + damping * np.eye(6)
        dq = jacobian_6d.T @ np.linalg.solve(lhs, error_6d)
        dq = np.clip(dq, -0.12, 0.12)
        q = pin.integrate(_PINOCCHIO_MODEL, q, dq)

    return clip_arm(q[:7]), info


def _tool_target_rotation(
    reference_rotation: np.ndarray,
    desired_z: Optional[Sequence[float]] = None,
    desired_x: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Align tool Z and optionally lock yaw through a configured tool X axis."""
    desired_z_vec = np.asarray(
        _DESIRED_Z if desired_z is None else desired_z,
        dtype=float,
    ).reshape(3)
    desired_z_vec /= max(float(np.linalg.norm(desired_z_vec)), 1e-9)
    reference_x = (
        np.asarray(reference_rotation, dtype=float).reshape(3, 3)[:, 0]
        if desired_x is None and _DESIRED_X is None
        else np.asarray(_DESIRED_X if desired_x is None else desired_x, dtype=float).reshape(3)
    )
    desired_x_vec = reference_x - desired_z_vec * float(np.dot(reference_x, desired_z_vec))
    if float(np.linalg.norm(desired_x_vec)) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(fallback, desired_z_vec))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        desired_x_vec = fallback - desired_z_vec * float(np.dot(fallback, desired_z_vec))
    desired_x_vec /= float(np.linalg.norm(desired_x_vec))
    desired_y = np.cross(desired_z_vec, desired_x_vec)
    desired_y /= float(np.linalg.norm(desired_y))
    desired_x_vec = np.cross(desired_y, desired_z_vec)
    return np.column_stack((desired_x_vec, desired_y, desired_z_vec))


def _grasp_pose_candidates(
    cube_pos: np.ndarray,
    grasp_offset: float,
    pregrasp_clearance: float,
    lift_clearance: float,
    *,
    lateral_enabled: bool = True,
) -> list[dict]:
    reference_rotation = data.xmat[ee_id].reshape(3, 3).copy()
    specifications = [
        ("top_down", np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])),
    ]
    if lateral_enabled:
        specifications.extend(
            [
                ("side_pos_x", np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
                ("side_neg_x", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
                ("side_pos_y", np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            ]
        )

    bounds = _grouped_wall_bounds()
    near_wall_boundary = False
    if bounds is not None:
        min_x, max_x, _, _ = bounds
        near_wall_boundary = min(
            abs(float(cube_pos[0]) - min_x),
            abs(float(cube_pos[0]) - max_x),
        ) <= 0.06
    if near_wall_boundary and lateral_enabled:
        specifications = specifications[1:] + specifications[:1]

    candidates: list[dict] = []
    for name, approach, tool_x in specifications:
        rotation = _tool_target_rotation(
            reference_rotation,
            desired_z=approach,
            desired_x=tool_x,
        )
        candidates.append(
            {
                "name": name,
                "approach": approach,
                "rotation": rotation,
                "pregrasp_xyz": cube_pos - approach * pregrasp_clearance,
                "grasp_xyz": cube_pos - approach * grasp_offset,
                # Every successful grasp clears the table vertically before
                # transit, including lateral approaches.
                "lift_xyz": cube_pos + np.array([0.0, 0.0, lift_clearance]),
            }
        )
    return candidates


def _select_grasp_pose_candidate(
    obj: str,
    candidates: Sequence[dict],
    cube_ref: np.ndarray,
) -> Optional[dict]:
    for candidate_index, candidate in enumerate(candidates):
        name = str(candidate["name"])
        rotation = np.asarray(candidate["rotation"], dtype=float)
        pregrasp_attempts = _ranked_ik_goals(
            candidate["pregrasp_xyz"],
            cube_ref,
            f"pick({obj}) pregrasp candidate {name}",
            target_rotation=rotation,
            max_valid_candidates=1,
        )
        pregrasp_valid = next(
            (attempt for attempt in pregrasp_attempts if attempt.failure_reason == IK_SUCCESS),
            None,
        )
        if pregrasp_valid is None:
            continue
        grasp_attempts = _ranked_ik_goals(
            candidate["grasp_xyz"],
            cube_ref,
            f"pick({obj}) grasp candidate {name}",
            ignored_body_names=[obj],
            target_rotation=rotation,
            max_valid_candidates=1,
            start_q=pregrasp_valid.q,
        )
        if any(attempt.failure_reason == IK_SUCCESS for attempt in grasp_attempts):
            _log_arm_state(
                "GRASP_CANDIDATE",
                "SELECT",
                object_id=obj,
                grasp_candidate=name,
                approach_vector=_round_vec(candidate["approach"], 3),
            )
            return candidate
    _log_arm_state(
        "GRASP_CANDIDATE",
        "FAILED",
        object_id=obj,
        failure_reason="no_ik_feasible_collision_free_grasp_candidate",
    )
    return None


def grasp_pose_candidates(
    cube_pos: Sequence[float],
    *,
    lateral_enabled: bool = True,
    pregrasp_clearance: Optional[float] = None,
) -> list[dict]:
    """Public shared grasp geometry used by execution and dry-run probing."""
    mujoco.mj_forward(model, data)
    return _grasp_pose_candidates(
        np.asarray(cube_pos, dtype=float),
        GRASP_OFFSET,
        float(APPROACH_CLEARANCE if pregrasp_clearance is None else pregrasp_clearance),
        APPROACH_CLEARANCE,
        lateral_enabled=lateral_enabled,
    )


def _rotation_quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    quaternion = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=float).reshape(-1))
    return _round_vec(quaternion, 6)


def probe_grasp_candidates(
    object_id: str,
    cube_pos: Sequence[float],
    *,
    ignored_body_names: Optional[Sequence[str]] = None,
    time_limit: float = 2.0,
    lateral_enabled: bool = True,
) -> dict:
    """Probe the exact ordered grasp orientations used by pick()."""
    started = time.monotonic()
    deadline = started + float(time_limit)
    diagnostics: list[dict] = []
    candidates = grasp_pose_candidates(
        cube_pos,
        lateral_enabled=lateral_enabled,
    )
    for candidate_index, candidate in enumerate(candidates):
        name = str(candidate["name"])
        if time.monotonic() >= deadline:
            log_event(
                "GRASP_PROBE",
                "TIMEOUT",
                object_id=object_id,
                phase=f"grasp_candidate:{name}",
                failure_reason="motion_probe_timeout",
                grasp_diagnostics=diagnostics,
            )
            return {
                "success": False,
                "ik_success": False,
                "ompl_success": False,
                "failure_reason": "motion_probe_timeout",
                "timeout_phase": f"grasp_candidate:{name}",
                "attempted_orientations": len(diagnostics),
                "attempted_seeds": sum(int(item.get("attempted_seeds", 0)) for item in diagnostics),
                "planning_time": time.monotonic() - started,
                "grasp_diagnostics": diagnostics,
            }
        candidate_started = time.monotonic()
        candidate_deadline = deadline
        report = probe_motion_to(
            candidate["pregrasp_xyz"],
            label=f"pick({object_id}) pregrasp probe {name}",
            ignored_body_names=ignored_body_names,
            time_limit=max(0.001, candidate_deadline - time.monotonic()),
            target_rotation=np.asarray(candidate["rotation"], dtype=float),
            deadline=candidate_deadline,
        )
        diagnostic = {
            "candidate": name,
            "approach_vector": _round_vec(candidate["approach"], 6),
            "pregrasp_position": _round_vec(candidate["pregrasp_xyz"], 6),
            "target_quaternion_wxyz": _rotation_quaternion_wxyz(candidate["rotation"]),
            "ik_success": bool(report.get("ik_success", False)),
            "position_error_m": report.get("position_error_m"),
            "orientation_error_rad": report.get("orientation_error_rad"),
            "joint_limit_valid": report.get("joint_limit_valid"),
            "collision_valid": report.get("collision_valid"),
            "collision_reason": report.get("collision_reason"),
            "ompl_success": bool(report.get("ompl_success", False)),
            "planning_time_s": round(time.monotonic() - candidate_started, 6),
            "attempted_seeds": int(report.get("attempted_seeds", 0)),
            "failure_reason": report.get("failure_reason"),
        }
        diagnostics.append(diagnostic)
        log_event(
            "GRASP_PROBE_CANDIDATE",
            "OK" if report.get("success") else "REJECT",
            object_id=object_id,
            phase=name,
            failure_reason=report.get("failure_reason"),
            grasp_diagnostic=diagnostic,
        )
        if report.get("success"):
            result = dict(report)
            result.update(
                selected_grasp_candidate=name,
                attempted_orientations=len(diagnostics),
                grasp_diagnostics=diagnostics,
                planning_time=time.monotonic() - started,
            )
            log_event(
                "GRASP_PROBE",
                "OK",
                object_id=object_id,
                phase=name,
                grasp_diagnostics=diagnostics,
            )
            return result
        if report.get("failure_reason") == "motion_probe_timeout":
            if time.monotonic() >= deadline:
                log_event(
                    "GRASP_PROBE",
                    "TIMEOUT",
                    object_id=object_id,
                    phase=report.get("timeout_phase", f"grasp_candidate:{name}"),
                    failure_reason="motion_probe_timeout",
                    grasp_diagnostics=diagnostics,
                )
                return {
                    "success": False,
                    "ik_success": False,
                    "ompl_success": False,
                    "failure_reason": "motion_probe_timeout",
                    "timeout_phase": report.get("timeout_phase", f"grasp_candidate:{name}"),
                    "attempted_orientations": len(diagnostics),
                    "attempted_seeds": sum(int(item.get("attempted_seeds", 0)) for item in diagnostics),
                    "planning_time": time.monotonic() - started,
                    "grasp_diagnostics": diagnostics,
                }
            continue

    log_event(
        "GRASP_PROBE",
        "FAILED",
        object_id=object_id,
        failure_reason="no_grasp_found",
        grasp_diagnostics=diagnostics,
    )
    return {
        "success": False,
        "ik_success": any(item["ik_success"] for item in diagnostics),
        "ompl_success": False,
        "failure_reason": "no_grasp_found",
        "attempted_orientations": len(diagnostics),
        "attempted_seeds": sum(int(item.get("attempted_seeds", 0)) for item in diagnostics),
        "planning_time": time.monotonic() - started,
        "grasp_diagnostics": diagnostics,
    }


def _pinocchio_base_translation() -> np.ndarray:
    prefix = _arm_prefix(ACTIVE_ARM)
    return np.asarray(model.body(f"{prefix}link0").pos[:3], dtype=float)


def init_hint_cache(log_dir: str = "logs", scene_filter: Optional[str] = None) -> None:
    """Load HintCache from past event logs. Call once after importing executor."""
    global _hint_cache
    try:
        from .adaptive_hints import HintCache
        fingerprint = _runtime_config_fingerprint()
        log_event(
            "RUNTIME_FINGERPRINT",
            "OK",
            backend=_IK_BACKEND_NAME,
            config_fingerprint=fingerprint,
            adaptive_history_consulted=not (
                CONFIG.ik.require_pinocchio or CONFIG.ik.backend == "pinocchio"
            ),
        )
        _hint_cache = HintCache(
            log_dir=log_dir,
            scene_filter=scene_filter,
            min_samples=CONFIG.adaptive.min_samples,
            pinocchio_skip_rate=CONFIG.adaptive.pinocchio_skip_rate,
            near_miss_rate=CONFIG.adaptive.near_miss_rate,
            near_miss_factor=CONFIG.adaptive.near_miss_factor,
            tolerance_headroom=CONFIG.adaptive.tolerance_headroom,
            max_tolerance_factor=CONFIG.adaptive.max_tolerance_factor,
            config_fingerprint=fingerprint,
        )
        summary = _hint_cache.summary()
        log_event(
            "HINT_CACHE",
            "LOADED",
            backend=_IK_BACKEND_NAME,
            history_policy=(
                "backend_pinned_pinocchio"
                if CONFIG.ik.require_pinocchio or CONFIG.ik.backend == "pinocchio"
                else "matching_fingerprint_only"
            ),
            **summary,
        )
    except Exception as exc:
        log_event("HINT_CACHE", "FAILED", failure_reason=str(exc))


def _solve_ik_to(
    target_xyz,
    null_ref=None,
    q_seed=None,
    steps=800,
    pos_limit: Optional[float] = None,
    ori_limit: Optional[float] = None,
    target_rotation: Optional[np.ndarray] = None,
    deadline: Optional[float] = None,
):
    max_pos = float(pos_limit if pos_limit is not None else IK_PLAN_POS_ERR_LIMIT)
    max_ori = float(ori_limit if ori_limit is not None else IK_PLAN_ORI_ERR_LIMIT)
    pinocchio_pinned = bool(
        CONFIG.ik.require_pinocchio or CONFIG.ik.backend == "pinocchio"
    )
    if _IK_BACKEND_NAME == "pinocchio":
        if not pinocchio_pinned and _hint_cache is not None and _hint_cache.preferred_backend(
            reach_dist=_hint_context.get("reach_dist", 0.0),
            obstacle_dist=_hint_context.get("obstacle_dist", float("inf")),
        ) == "mujoco_dls":
            log_event(
                "IK_BACKEND_POLICY",
                "ADAPTIVE_OVERRIDE",
                backend="mujoco_dls",
                config_fingerprint=_runtime_config_fingerprint(),
                reason="matching_history_requested_fallback",
            )
            return _ik_solve_to(
                target_xyz,
                null_ref=null_ref,
                q_seed=q_seed,
                steps=steps,
                pos_tol=max_pos,
                ori_tol=max_ori,
                target_rotation=target_rotation,
                deadline=deadline,
            )
        try:
            _check_deadline(deadline, "pinocchio_seed_start")
            seed = q_seed if q_seed is not None else null_ref
            pin_q, pin_info = _pinocchio_ik_solve_to(
                target_xyz,
                q_seed=seed,
                steps=180,
                pos_tol=max_pos,
                ori_tol=max_ori,
                target_rotation=target_rotation,
                deadline=deadline,
            )
            fk_pos_err, fk_ori_err, _ = _mujoco_fk_error(
                pin_q,
                target_xyz,
                target_rotation,
            )
            if fk_pos_err <= max_pos and fk_ori_err <= max_ori:
                return pin_q, pin_info

            if pinocchio_pinned:
                log_event(
                    "IK_SOLVE",
                    "PINOCCHIO_REQUIRED_REJECT",
                    backend="pinocchio",
                    failure_reason="pinocchio_fk_validation_failed",
                    pos_err=fk_pos_err,
                    ori_err=fk_ori_err,
                    config_fingerprint=_runtime_config_fingerprint(),
                )
                return pin_q, pin_info

            dls_q, dls_info = _ik_solve_to(
                target_xyz,
                null_ref=null_ref,
                q_seed=q_seed,
                steps=steps,
                pos_tol=max_pos,
                ori_tol=max_ori,
                target_rotation=target_rotation,
                deadline=deadline,
            )
            dls_pos_err, dls_ori_err, _ = _mujoco_fk_error(
                dls_q,
                target_xyz,
                target_rotation,
            )
            use_dls = (
                (dls_pos_err <= max_pos and dls_ori_err <= max_ori)
                or (dls_pos_err + 0.15 * dls_ori_err < fk_pos_err + 0.15 * fk_ori_err)
            )
            log_event(
                "IK_SOLVE",
                "BACKEND_FALLBACK",
                backend="pinocchio",
                failure_reason="pinocchio_fk_validation_failed",
                pos_err=round(float(fk_pos_err), 5),
                ori_err=round(float(fk_ori_err), 5),
                pos_limit=max_pos,
                ori_limit=max_ori,
                fallback_backend="mujoco_dls",
                dls_pos_err=round(float(dls_pos_err), 5),
                dls_ori_err=round(float(dls_ori_err), 5),
                use_dls=use_dls,
            )
            if use_dls:
                return dls_q, dls_info
            return pin_q, pin_info
        except MotionProbeTimeout:
            raise
        except Exception as exc:
            log_event("IK_SOLVE", "BACKEND_FALLBACK", backend="pinocchio", failure_reason=str(exc))
            if pinocchio_pinned:
                raise RuntimeError(f"required_pinocchio_solve_failed:{exc}") from exc
    return _ik_solve_to(
        target_xyz,
        null_ref=null_ref,
        q_seed=q_seed,
        steps=steps,
        pos_tol=max_pos,
        ori_tol=max_ori,
        target_rotation=target_rotation,
        deadline=deadline,
    )


def _solve_safe_goal_candidates(target_xyz, base_null_ref, label=""):
    planner = _get_ompl_planner()
    candidate_refs = [
        base_null_ref.copy(),
        _ELBOW_UP_REF.copy(),
    ]

    # Small null-space variations help escape one bad IK basin.
    for delta in [0.12, -0.12]:
        ref = base_null_ref.copy()
        ref[1] = np.clip(ref[1] + delta, arm_ranges[1, 0], arm_ranges[1, 1])
        candidate_refs.append(ref)

    for null_ref in candidate_refs:
        goal_q, info = _ik_solve_to(target_xyz, null_ref=null_ref, steps=800)
        if not info["converged"]:
            continue
        if planner is None or planner.is_state_valid_q(goal_q):
            return goal_q, info

    return None, None


def _target_xyz_candidates(target_xyz: Sequence[float], label: str) -> list[np.ndarray]:
    base = np.asarray(target_xyz, dtype=float).reshape(3)
    candidates = [base]
    if label.startswith("pick("):
        xy = base[:2]
        base_distance = float(np.linalg.norm(xy - BASE_XY))
        radial = BASE_XY - xy
        radial_norm = float(np.linalg.norm(radial))
        radial_dir = radial / radial_norm if radial_norm > 1e-6 else np.array([-1.0, 0.0])
        tangent_dir = np.array([-radial_dir[1], radial_dir[0]])

        xy_offsets = [
            np.array([0.025, 0.0, 0.0]),
            np.array([-0.025, 0.0, 0.0]),
            np.array([0.0, 0.025, 0.0]),
            np.array([0.0, -0.025, 0.0]),
        ]
        candidates.extend(base + offset for offset in xy_offsets)
        if _obstacle_ids:
            nearest_xy = None
            nearest_distance = float("inf")
            for obstacle_id in _obstacle_ids.values():
                obstacle_xy = data.xpos[obstacle_id][:2].copy()
                distance = float(np.linalg.norm(xy - obstacle_xy))
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_xy = obstacle_xy
            if nearest_xy is not None and nearest_distance < CAUTIOUS_OBSTACLE_CLEARANCE:
                away = xy - nearest_xy
                away_norm = float(np.linalg.norm(away))
                away_dir = away / away_norm if away_norm > 1e-6 else radial_dir
                tangent_dir = np.array([-away_dir[1], away_dir[0]])
                if "grasp" in label and "pregrasp" not in label:
                    if not label.startswith("pick(circle"):
                        for offset in (
                            np.array([0.015, -0.04, 0.0]),
                            np.array([0.0, -0.045, 0.0]),
                            np.array([-0.045, -0.012, 0.0]),
                            np.array([-0.045, -0.03, 0.0]),
                        ):
                            candidates.append(base + offset)
                    for away_offset in (0.018, 0.028, 0.038):
                        candidates.append(base + np.r_[away_dir * away_offset, -0.012])
                    for side_offset in (0.014, -0.014):
                        candidates.append(base + np.r_[away_dir * 0.03 + tangent_dir * side_offset, -0.01])
                else:
                    for away_offset in (0.04, 0.065, 0.09):
                        candidates.append(base + np.r_[away_dir * away_offset, 0.0])

        # For far objects, approach from the robot-facing side and avoid an
        # unnecessarily high vertical pregrasp that pushes IK outside workspace.
        if base_distance > FAR_PICK_XY_DISTANCE:
            for radial_offset in (0.045, 0.075):
                candidates.append(base + np.r_[radial_dir * radial_offset, 0.0])
            if "pregrasp" in label:
                for z_drop in (0.045, 0.075):
                    candidates.append(base + np.array([0.0, 0.0, -z_drop]))
                    candidates.append(base + np.r_[radial_dir * 0.055, -z_drop])

        # Cylinders are more forgiving with a radial side-biased contact than a
        # pure top-center target, especially near workspace limits.
        if label.startswith("pick(circle"):
            for radial_offset in (0.035, 0.06, 0.085):
                candidates.append(base + np.r_[radial_dir * radial_offset, 0.0])
            for side_offset in (0.035, -0.035):
                candidates.append(base + np.r_[radial_dir * 0.055 + tangent_dir * side_offset, 0.0])

        if "grasp" in label and "pregrasp" not in label:
            candidates.extend(base + offset + np.array([0.0, 0.0, -0.01]) for offset in xy_offsets[:2])
    elif label.startswith("place("):
        xy_offsets = [
            np.array([0.018, 0.0, 0.0]),
            np.array([-0.018, 0.0, 0.0]),
            np.array([0.0, 0.018, 0.0]),
            np.array([0.0, -0.018, 0.0]),
        ]
        candidates.extend(base + offset for offset in xy_offsets)
        if "release" in label:
            candidates.extend(base + np.array([0.0, 0.0, z]) for z in (0.015, -0.015))
    return candidates


def _ik_pos_limit_for_label(label: str) -> float:
    if label.startswith("pick(") and "pregrasp" in label:
        return IK_PREGRASP_POS_ERR_LIMIT
    return IK_PLAN_POS_ERR_LIMIT


def _ik_ori_limit_for_label(label: str) -> float:
    if label.startswith("pick(") and "pregrasp" in label:
        return IK_PREGRASP_ORI_ERR_LIMIT
    return IK_PLAN_ORI_ERR_LIMIT


def _null_ref_candidates(
    null_ref: Optional[np.ndarray],
    seed_key: str = "",
) -> list[np.ndarray]:
    base_ref = np.asarray(null_ref if null_ref is not None else GRASP_READY, dtype=float).copy()
    refs = [current_q(), base_ref, GRASP_READY.copy(), _ELBOW_UP_REF.copy()]
    if _grouped_wall_bounds() is not None and joint_limits_valid(
        _WALL_RIGHT_REF, arm_ranges[:, 0], arm_ranges[:, 1]
    ):
        refs.append(_WALL_RIGHT_REF.copy())
    if _LAST_SUCCESSFUL_SEED is not None:
        refs.append(_LAST_SUCCESSFUL_SEED.copy())

    # Pinocchio does not expose a stable Panda arm-angle parameterization here.
    # Use a deterministic, low-discrepancy Halton design over all seven real
    # hardware joint ranges instead of coupling only joints 3/5/7.
    def radical_inverse(index: int, base: int) -> float:
        result = 0.0
        factor = 1.0 / base
        while index:
            result += factor * (index % base)
            index //= base
            factor /= base
        return result

    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    offset = 1 + int.from_bytes(digest[:4], "little") % 4096
    primes = (2, 3, 5, 7, 11, 13, 17)
    lower = arm_ranges[:, 0]
    span = arm_ranges[:, 1] - lower
    for sample_index in range(24):
        unit = np.array(
            [radical_inverse(offset + sample_index, prime) for prime in primes],
            dtype=float,
        )
        refs.append(lower + unit * span)
    unique = []
    seen = set()
    for ref in refs:
        clipped = clip_arm(ref)
        key = tuple(np.round(clipped, 4))
        if key not in seen:
            seen.add(key)
            unique.append(clipped)
    return unique[:32]


def _ranked_ik_goals(
    target_xyz: Sequence[float],
    null_ref: Optional[np.ndarray],
    label: str,
    ignored_body_names: Optional[Sequence[str]] = None,
    max_valid_candidates: Optional[int] = None,
    start_q: Optional[Sequence[float]] = None,
    target_rotation: Optional[np.ndarray] = None,
    deadline: Optional[float] = None,
) -> list[IKAttemptResult]:
    planner = _get_ompl_planner()
    pos_limit = _ik_pos_limit_for_label(label)
    ori_limit = _ik_ori_limit_for_label(label)
    if _hint_cache is not None:
        _hint_tol = _hint_cache.pos_err_tolerance(
            reach_dist=_hint_context.get("reach_dist", 0.0),
            obstacle_dist=_hint_context.get("obstacle_dist", float("inf")),
        )
        if _hint_tol is not None and _hint_tol > pos_limit:
            pos_limit = _hint_tol
    attempts: list[IKAttemptResult] = []
    valid_limit = int(max_valid_candidates or MAX_VALID_IK_CANDIDATES)
    reference_q = (
        current_q()
        if start_q is None
        else clip_arm(np.asarray(start_q, dtype=float).reshape(-1))
    )

    for candidate_idx, xyz in enumerate(_target_xyz_candidates(target_xyz, label)):
        _check_deadline(deadline, f"{label}:target_variation", len(attempts))
        seed_key = f"{label}:{','.join(f'{float(v):.6f}' for v in xyz)}"
        for seed_idx, ref in enumerate(_null_ref_candidates(null_ref, seed_key)):
            _check_deadline(deadline, f"{label}:ik_seed", len(attempts))
            q_seed = reference_q if seed_idx == 0 else ref
            log_event(
                "IK_SEED",
                "TRY",
                phase=label,
                candidate_id=candidate_idx,
                seed_id=seed_idx,
                q_target=_round_vec(q_seed, 6),
            )
            q, info = _solve_ik_to(
                xyz,
                null_ref=ref,
                q_seed=q_seed,
                pos_limit=pos_limit,
                ori_limit=ori_limit,
                target_rotation=target_rotation,
                deadline=deadline,
            )
            fk_pos_err, fk_ori_err, fk_ee_xyz = _mujoco_fk_error(
                q,
                xyz,
                target_rotation,
            )
            pos_err = fk_pos_err
            ori_err = fk_ori_err
            joint_ok = joint_limits_valid(q, arm_ranges[:, 0], arm_ranges[:, 1])
            state_valid = None
            state_reason = None
            if planner is not None and joint_ok:
                state_valid = planner.is_state_valid_q(q, ignored_body_names=ignored_body_names)
                state_reason = getattr(planner, "_last_invalid_reason", None)
            reason = classify_ik_attempt(
                pos_err=pos_err,
                ori_err=ori_err,
                pos_limit=pos_limit,
                ori_limit=ori_limit,
                joint_limit_valid=joint_ok,
                state_valid=state_valid,
                state_invalid_reason=state_reason,
                converged=bool(info.get("converged", False)) or np.isfinite(pos_err),
            )
            score = (
                pos_err
                + 0.15 * ori_err
                + 0.02 * float(np.linalg.norm(np.asarray(q, dtype=float) - reference_q))
            )
            if label.startswith("pick(") and "grasp" in label and "pregrasp" not in label:
                score += 1.5 * float(np.linalg.norm(np.asarray(xyz, dtype=float) - np.asarray(target_xyz, dtype=float)))
            attempt = IKAttemptResult(
                q=q,
                target_xyz=np.asarray(xyz, dtype=float),
                backend=str(info.get("backend", _IK_BACKEND_NAME)),
                candidate_id=candidate_idx,
                seed_id=seed_idx,
                pos_err=pos_err,
                ori_err=ori_err,
                iterations=int(info.get("iters", 0)),
                converged=bool(info.get("converged", False)),
                joint_limit_valid=joint_ok,
                state_valid=state_valid,
                state_invalid_reason=state_reason,
                failure_reason=reason,
                score=score,
            )
            attempts.append(attempt)
            log_event(
                "IK_CANDIDATE",
                "OK" if reason == IK_SUCCESS else "REJECT",
                arm=ACTIVE_ARM,
                phase=label,
                target_xyz=[round(float(v), 4) for v in xyz],
                actual_xyz=fk_ee_xyz,
                ee_xyz=_ee_xyz(),
                q=_round_vec(q, 4),
                q_error_norm=round(float(np.linalg.norm(np.asarray(q, dtype=float) - reference_q)), 5),
                pos_err=round(pos_err, 5),
                ori_err=round(ori_err, 5),
                iterations=info.get("iters", 0),
                candidate_idx=candidate_idx,
                candidate_id=candidate_idx,
                seed_id=seed_idx,
                backend=info.get("backend", _IK_BACKEND_NAME),
                pos_limit=pos_limit,
                ori_limit=ori_limit,
                converged=info.get("converged", False),
                joint_limit_valid=joint_ok,
                state_valid=state_valid,
                state_invalid_reason=state_reason,
                failure_reason=(
                    None if reason == IK_SUCCESS else "ik_candidate_rejected"
                ),
                position_within_limit=pos_err <= pos_limit,
                orientation_within_limit=ori_err <= ori_limit,
                score=round(score, 5),
            )
            valid_count = sum(1 for item in attempts if item.failure_reason == IK_SUCCESS)
            if valid_count >= valid_limit:
                return rank_ik_attempts(attempts)
            if len(attempts) >= MAX_IK_ATTEMPTS_PER_SEGMENT:
                return rank_ik_attempts(attempts)

    return rank_ik_attempts(attempts)


def _select_ik_goal(target_xyz: Sequence[float], null_ref: Optional[np.ndarray], label: str):
    ranked = _ranked_ik_goals(target_xyz, null_ref, label)
    valid = [item for item in ranked if item.failure_reason == IK_SUCCESS]
    if not valid:
        return None, None, None, IK_UNREACHABLE
    best = valid[0]
    info = {
        "backend": best.backend,
        "pos_err_norm": best.pos_err,
        "ori_err_norm": best.ori_err,
        "iters": best.iterations,
        "converged": best.converged,
        "candidate_id": best.candidate_id,
        "seed_id": best.seed_id,
    }
    return best.q, info, best.target_xyz, IK_SUCCESS

# =============================
# OMPL EXECUTION
# =============================

def _execute_joint_trajectory(
    traj: np.ndarray,
    grip: float,
    ignored_body_names: Optional[Sequence[str]] = None,
    settle_steps_per_wp: int = DEFAULT_SETTLE_STEPS_PER_WP,
    final_settle_steps: int = DEFAULT_FINAL_SETTLE_STEPS,
) -> bool:
    if traj is None or len(traj) == 0:
        _log_arm_state("TRAJECTORY_EXEC", "SKIP", waypoints=0, grip=grip)
        return True

    _log_arm_state(
        "TRAJECTORY_EXEC",
        "START",
        waypoints=len(traj),
        grip=grip,
        q_target=_round_vec(traj[-1], 4),
        q_error_norm=round(float(np.linalg.norm(clip_arm(np.asarray(traj[-1], dtype=float)) - current_q())), 5),
        ignored_body_names=list(ignored_body_names or []),
        settle_steps_per_wp=settle_steps_per_wp,
        final_settle_steps=final_settle_steps,
    )
    started = time.perf_counter()
    for waypoint_index, q in enumerate(traj):
        q = clip_arm(np.asarray(q, dtype=float))
        if waypoint_index in {0, len(traj) - 1} or waypoint_index % max(1, len(traj) // 4) == 0:
            _log_arm_state(
                "TRAJECTORY_WAYPOINT",
                "EXEC",
                waypoints=f"{waypoint_index + 1}/{len(traj)}",
                grip=grip,
                q=[round(float(v), 4) for v in q],
                q_target=[round(float(v), 4) for v in traj[-1]],
                q_error_norm=round(float(np.linalg.norm(q - current_q())), 5),
            )
        for _ in range(settle_steps_per_wp):
            _set_arm_ctrl(q, grip)
            mujoco.mj_step(model, data)
            viewer.sync()
            if not _check_live_collision(
                context=f"trajectory waypoint {waypoint_index}",
                ignored_body_names=ignored_body_names,
                allow_start_table_finger=waypoint_index == 0,
            ):
                _log_arm_state(
                    "TRAJECTORY_EXEC",
                    "FAILED",
                    waypoints=len(traj),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    failure_reason=f"collision_at_waypoint_{waypoint_index}",
                    q_target=_round_vec(q, 4),
                    q_error_norm=round(float(np.linalg.norm(q - current_q())), 5),
                )
                return False

    if final_settle_steps > 0:
        final_q = clip_arm(np.asarray(traj[-1], dtype=float))
        for _ in range(final_settle_steps):
            _set_arm_ctrl(final_q, grip)
            mujoco.mj_step(model, data)
            viewer.sync()
            if not _check_live_collision(
                context="trajectory final settle",
                ignored_body_names=ignored_body_names,
            ):
                _log_arm_state(
                    "TRAJECTORY_EXEC",
                    "FAILED",
                    waypoints=len(traj),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    failure_reason="collision_during_final_settle",
                    q_target=_round_vec(final_q, 4),
                    q_error_norm=round(float(np.linalg.norm(final_q - current_q())), 5),
                )
                return False

    _log_arm_state(
        "TRAJECTORY_EXEC",
        "OK",
        waypoints=len(traj),
        duration_ms=int((time.perf_counter() - started) * 1000),
        q_target=_round_vec(traj[-1], 4),
        q_error_norm=round(float(np.linalg.norm(clip_arm(np.asarray(traj[-1], dtype=float)) - current_q())), 5),
    )
    return True


def _move_with_ompl(
    goal_q: Sequence[float],
    grip: float,
    ignored_body_names: Optional[Sequence[str]] = None,
    planner_name: str = DEFAULT_PLANNER_NAME,
    time_limit: float = DEFAULT_TIME_LIMIT,
    label: str = "",
    settle_steps_per_wp: int = DEFAULT_SETTLE_STEPS_PER_WP,
    final_settle_steps: int = DEFAULT_FINAL_SETTLE_STEPS,
) -> bool:
    goal_q = np.asarray(goal_q, dtype=float).reshape(7)
    start_q = current_q()

    if not _OMPL_AVAILABLE:
        print("[exec][OMPL] unavailable")
        _log_arm_state("OMPL_PLAN", "UNAVAILABLE", phase=label, failure_reason="ompl_unavailable", q_target=_round_vec(goal_q, 4))
        return False

    planner = _get_ompl_planner()
    if planner is None:
        print("[exec][OMPL] planner init failed")
        _log_arm_state("OMPL_PLAN", "FAILED", phase=label, failure_reason="planner_init_failed", q_target=_round_vec(goal_q, 4))
        return False

    planner_attempts = []
    for candidate in (
        planner_name,
        CONFIG.motion.fragile_planner,
        "BITstar",
        "RRTConnect",
    ):
        if candidate and candidate not in planner_attempts:
            planner_attempts.append(candidate)

    start_state_valid = planner.is_state_valid_q(start_q)
    start_state_reason = getattr(planner, "_last_invalid_reason", None)
    last_info = None
    try:
        _log_arm_state(
            "OMPL_PLAN",
            "START",
            phase=label,
            planner=planner_attempts[0],
            grip=grip,
            ignored_body_names=list(ignored_body_names or []),
            time_limit=time_limit,
            start_q=[round(float(v), 4) for v in start_q],
            goal_q=[round(float(v), 4) for v in goal_q],
            q_target=_round_vec(goal_q, 4),
            q_error_norm=round(float(np.linalg.norm(goal_q - start_q)), 5),
            planner_attempts=planner_attempts,
            start_state_valid=start_state_valid,
            start_state_invalid_reason=start_state_reason,
        )
        started = time.perf_counter()
        for attempt_index, attempt_planner in enumerate(planner_attempts, start=1):
            traj, info = planner.plan(
                start_q=start_q,
                goal_q=goal_q,
                time_limit=time_limit,
                planner_name=attempt_planner,
                ignored_body_names=ignored_body_names,
                fragile_mode=attempt_planner == CONFIG.motion.fragile_planner,
            )
            last_info = info
            if traj is None:
                attempt_duration_ms = int((time.perf_counter() - started) * 1000)
                goal_attempts = info.get("goal_attempts") or []
                statuses = [item.get("status") for item in goal_attempts if isinstance(item, dict)]
                if statuses and all(status == "invalid_goal" for status in statuses):
                    failure_reason = IK_GOAL_STATE_INVALID
                elif attempt_duration_ms >= int(time_limit * 950):
                    failure_reason = OMPL_TIMEOUT
                else:
                    failure_reason = "ompl_no_path_found"
                _log_arm_state(
                    "OMPL_PLAN_ATTEMPT",
                    "FAILED",
                    phase=label,
                    planner=info.get("planner_name", attempt_planner),
                    attempt=attempt_index,
                    failure_reason=failure_reason,
                    ompl_result=failure_reason,
                    goal_attempts=goal_attempts,
                    q_target=_round_vec(goal_q, 4),
                    q_error_norm=round(float(np.linalg.norm(goal_q - current_q())), 5),
                )
                continue

            selected_goal_q = np.asarray(
                info.get("selected_goal_q", goal_q), dtype=float
            )
            endpoint_error = float(
                np.linalg.norm(np.asarray(traj[-1], dtype=float) - selected_goal_q)
            )
            if endpoint_error > 0.02:
                _log_arm_state(
                    "OMPL_PLAN_ATTEMPT",
                    "FAILED",
                    phase=label,
                    planner=info.get("planner_name", attempt_planner),
                    attempt=attempt_index,
                    failure_reason="ompl_approximate_endpoint",
                    q_target=_round_vec(selected_goal_q, 4),
                    endpoint_q=_round_vec(traj[-1], 4),
                    endpoint_error=round(endpoint_error, 5),
                )
                continue

            duration_ms = int((time.perf_counter() - started) * 1000)
            print(
                f"[exec][OMPL] solved={info.get('solved')} "
                f"planner={info.get('planner_name')} waypoints={info.get('num_waypoints')}"
            )
            _log_arm_state(
                "OMPL_PLAN",
                "OK",
                phase=label,
                planner=info.get("planner_name"),
                waypoints=info.get("num_waypoints"),
                duration_ms=duration_ms,
                path_length=round(float(info.get("path_length_joint_space", 0.0)), 4),
                selected_goal_q=info.get("selected_goal_q"),
                q_target=_round_vec(info.get("selected_goal_q", goal_q), 4),
                q_error_norm=round(float(np.linalg.norm(np.asarray(info.get("selected_goal_q", goal_q), dtype=float) - current_q())), 5),
                ompl_result="success",
            )
            return _execute_joint_trajectory(
                traj,
                grip=grip,
                ignored_body_names=ignored_body_names,
                settle_steps_per_wp=settle_steps_per_wp,
                final_settle_steps=final_settle_steps,
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        goal_attempts = (last_info or {}).get("goal_attempts") or []
        statuses = [item.get("status") for item in goal_attempts if isinstance(item, dict)]
        if statuses and all(status == "invalid_goal" for status in statuses):
            failure_reason = IK_GOAL_STATE_INVALID
        elif duration_ms >= int(time_limit * 950):
            failure_reason = OMPL_TIMEOUT
        else:
            failure_reason = "ompl_no_path_found"
        print(f"[exec][OMPL] planning failed: {last_info}")
        _log_arm_state(
            "OMPL_PLAN",
            "FAILED",
            phase=label,
            planner=(last_info or {}).get("planner_name", planner_name),
            duration_ms=duration_ms,
            failure_reason=failure_reason,
            ompl_result=failure_reason,
            goal_attempts=goal_attempts,
            q_target=_round_vec(goal_q, 4),
            q_error_norm=round(float(np.linalg.norm(goal_q - current_q())), 5),
        )
        return False
    except Exception as e:
        print(f"[exec][OMPL] error: {e}")
        _log_arm_state("OMPL_PLAN", "ERROR", phase=label, failure_reason=str(e), q_target=_round_vec(goal_q, 4))
        return False


def _move_pose_safe(
    target_xyz: Sequence[float],
    grip: float,
    null_ref: Optional[np.ndarray] = None,
    ignored_body_names: Optional[Sequence[str]] = None,
    label: str = "",
    cautious_motion: bool = False,
    allow_wall_bypass: bool = True,
    target_rotation: Optional[np.ndarray] = None,
) -> bool:
    global _LAST_SUCCESSFUL_SEED
    if allow_wall_bypass:
        bypass = _grouped_wall_bypass_waypoints(
            np.asarray(target_xyz, dtype=float)
        )
        if bypass is not None:
            side, cartesian_gateways = bypass
            planner = _get_ompl_planner()
            fixed_valid = False
            fixed_q = _WALL_RIGHT_REF
            if planner is not None:
                fixed_valid, fixed_reason, fixed_q = _validated_wall_right_q(
                    planner, ignored_body_names
                )
                log_event(
                    "WALL_GATEWAY_VALIDATE",
                    "OK" if fixed_valid else "REJECT",
                    phase=label,
                    failure_reason=None if fixed_valid else fixed_reason,
                    q_target=_round_vec(fixed_q, 4),
                )
            if fixed_valid:
                if not _move_with_ompl(
                    goal_q=fixed_q,
                    grip=grip,
                    ignored_body_names=ignored_body_names,
                    planner_name=DEFAULT_PLANNER_NAME,
                    time_limit=DEFAULT_TIME_LIMIT,
                    label=f"wall gateway_q {side} for {label}",
                    settle_steps_per_wp=DEFAULT_SETTLE_STEPS_PER_WP,
                    final_settle_steps=DEFAULT_FINAL_SETTLE_STEPS,
                ):
                    fixed_valid = False
            if not fixed_valid:
                for index, waypoint in enumerate(cartesian_gateways, start=1):
                    if not _move_pose_safe(
                        waypoint,
                        grip=grip,
                        null_ref=cube_null_ref(waypoint),
                        ignored_body_names=ignored_body_names,
                        label=f"wall gateway {side} {index} for {label}",
                        cautious_motion=cautious_motion,
                        allow_wall_bypass=False,
                    ):
                        return False
    _log_arm_state(
        "MOVE_POSE",
        "START",
        phase=label,
        target_xyz=[round(float(v), 4) for v in target_xyz],
        grip=grip,
        ignored_body_names=list(ignored_body_names or []),
    )
    ranked_attempts = _ranked_ik_goals(
        target_xyz,
        null_ref,
        label,
        ignored_body_names=ignored_body_names,
        target_rotation=target_rotation,
    )
    valid_attempts = [item for item in ranked_attempts if item.failure_reason == IK_SUCCESS]
    if not valid_attempts:
        best_attempt = ranked_attempts[0] if ranked_attempts else None
        pos_err = float(best_attempt.pos_err if best_attempt is not None else float("inf"))
        ori_err = float(best_attempt.ori_err if best_attempt is not None else float("inf"))
        failure_reason = best_attempt.failure_reason if best_attempt is not None else IK_UNREACHABLE
        pos_limit = _ik_pos_limit_for_label(label)
        ori_limit = _ik_ori_limit_for_label(label)
        print(
            f"[exec][IK] reject {label or 'pose'}: "
            f"best_pos={pos_err:.4f} best_ori={ori_err:.4f} "
            f"pos_limit={pos_limit:.3f} ori_limit={ori_limit:.3f} reason={failure_reason}"
        )
        _log_arm_state(
            "IK_SOLVE",
            "FAILED",
            phase=label,
            target_xyz=[round(float(v), 4) for v in target_xyz],
            failure_reason=failure_reason,
            pos_err=round(pos_err, 5) if np.isfinite(pos_err) else None,
            ori_err=round(ori_err, 5) if np.isfinite(ori_err) else None,
            pos_limit=pos_limit,
            ori_limit=ori_limit,
            backend=best_attempt.backend if best_attempt is not None else _IK_BACKEND_NAME,
            candidate_id=best_attempt.candidate_id if best_attempt is not None else None,
            seed_id=best_attempt.seed_id if best_attempt is not None else None,
        )
        return False

    for goal_attempt, attempt in enumerate(valid_attempts, start=1):
        selected_goal_q = attempt.q
        selected_xyz = attempt.target_xyz
        _log_arm_state(
            "IK_SOLVE",
            "OK",
            phase=label,
            target_xyz=[round(float(v), 4) for v in selected_xyz],
            q_target=_round_vec(selected_goal_q, 4),
            pos_err=round(float(attempt.pos_err), 5),
            ori_err=round(float(attempt.ori_err), 5),
            iterations=attempt.iterations,
            backend=attempt.backend,
            candidate_id=attempt.candidate_id,
            seed_id=attempt.seed_id,
            goal_attempt=goal_attempt,
        )
        ok = _move_with_ompl(
            goal_q=selected_goal_q,
            grip=grip,
            ignored_body_names=ignored_body_names,
            planner_name=DEFAULT_PLANNER_NAME,
            time_limit=DEFAULT_TIME_LIMIT * (1.7 if cautious_motion else 1.0),
            label=label,
            settle_steps_per_wp=DEFAULT_SETTLE_STEPS_PER_WP * (2 if cautious_motion else 1),
            final_settle_steps=DEFAULT_FINAL_SETTLE_STEPS * (2 if cautious_motion else 1),
        )
        if ok:
            _LAST_SUCCESSFUL_SEED = np.asarray(selected_goal_q, dtype=float).copy()
            _log_arm_state(
                "MOVE_POSE",
                "OK",
                phase=label,
                target_xyz=[round(float(v), 4) for v in selected_xyz],
                backend=attempt.backend,
                candidate_id=attempt.candidate_id,
                seed_id=attempt.seed_id,
                execution_result="success",
            )
            return True
        _log_arm_state(
            "MOVE_POSE",
            "RETRY_NEXT_IK_GOAL",
            phase=label,
            target_xyz=[round(float(v), 4) for v in selected_xyz],
            backend=attempt.backend,
            candidate_id=attempt.candidate_id,
            seed_id=attempt.seed_id,
            failure_reason="ompl_or_execution_failed",
        )

    if USE_IK_FALLBACK:
        print(f"[exec] OMPL failed for {label or 'pose'}; using IK fallback")
        fallback_xyz = valid_attempts[0].target_xyz
        _log_arm_state("MOVE_POSE", "IK_FALLBACK", phase=label, target_xyz=[round(float(v), 4) for v in fallback_xyz])
        move_ee_to(fallback_xyz, grip=grip, steps=300, null_ref=null_ref)
        return True
    print(f"[exec] OMPL failed for {label or 'pose'}; no fallback in fragile-scene mode")
    _log_arm_state(
        "MOVE_POSE",
        "FAILED",
        phase=label,
        target_xyz=[round(float(v), 4) for v in valid_attempts[0].target_xyz],
        failure_reason="ompl_failed_no_fallback",
        execution_result="execution_failed",
    )
    return False


def probe_motion_to(
    target_xyz: Sequence[float],
    *,
    label: str = "motion probe",
    ignored_body_names: Optional[Sequence[str]] = None,
    time_limit: Optional[float] = None,
    target_rotation: Optional[np.ndarray] = None,
    deadline: Optional[float] = None,
) -> dict:
    """Run IK and a real OMPL solve without mutating the live simulation."""
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    ctrl_before = data.ctrl.copy()
    started = time.perf_counter()
    budget = float(time_limit if time_limit is not None else min(DEFAULT_TIME_LIMIT, 6.0))
    deadline = deadline if deadline is not None else time.monotonic() + budget
    result = {
        "success": False,
        "ik_success": False,
        "ompl_success": False,
        "planning_time": 0.0,
        "path_length": 0.0,
        "failure_reason": "ik_unreachable",
        "attempted_seeds": 0,
    }
    try:
        _check_deadline(deadline, "probe_start")
        planner = _get_ompl_planner()
        if planner is None:
            result["failure_reason"] = "ompl_unavailable"
            return result
        target = np.asarray(target_xyz, dtype=float)
        bypass = _grouped_wall_bypass_waypoints(target)
        route_targets: list[tuple[str, np.ndarray]] = []
        gateway_required = bypass is not None
        if bypass is not None:
            side, cartesian_gateways = bypass
            result["route"] = f"wall_{side}"
        else:
            result["route"] = "direct"
        route_targets.append((label, target))
        virtual_q = current_q().copy()
        path_length = 0.0
        waypoint_count = 0
        segment_count = 0
        if gateway_required:
            fixed_valid, fixed_reason, fixed_q = _validated_wall_right_q(
                planner, ignored_body_names
            )
            result["fixed_gateway_validation"] = fixed_reason
            log_event(
                "WALL_GATEWAY_VALIDATE",
                "OK" if fixed_valid else "REJECT",
                phase=label,
                q_target=_round_vec(fixed_q, 4),
                failure_reason=None if fixed_valid else fixed_reason,
            )
            if fixed_valid:
                _check_deadline(deadline, "fixed_gateway_ompl")
                gateway_trajectory, gateway_info = planner.plan(
                    start_q=virtual_q,
                    goal_q=fixed_q,
                    time_limit=_remaining_probe_segment_time(deadline),
                    planner_name=DEFAULT_PLANNER_NAME,
                    ignored_body_names=ignored_body_names,
                    fragile_mode=False,
                )
                if gateway_trajectory is not None:
                    segment_count += 1
                    virtual_q = np.asarray(gateway_trajectory[-1], dtype=float).copy()
                    path_length += float(gateway_info.get("path_length_joint_space", 0.0))
                    waypoint_count += int(gateway_info.get("num_waypoints", len(gateway_trajectory)))
                    result["gateway_mode"] = "fixed_q"
                else:
                    fixed_valid = False
                    result["fixed_gateway_validation"] = "ompl_no_path"
            if not fixed_valid:
                result["gateway_mode"] = "cartesian"
                route_targets = [
                    (f"wall gateway {side} {index} for {label}", waypoint)
                    for index, waypoint in enumerate(cartesian_gateways, start=1)
                ] + route_targets

        for route_index, (segment_label, segment_target) in enumerate(route_targets):
            segment_index = segment_count + 1
            _check_deadline(deadline, f"segment_{segment_index}_ik", result["attempted_seeds"])
            attempts = _ranked_ik_goals(
                segment_target,
                cube_null_ref(segment_target),
                segment_label,
                ignored_body_names=ignored_body_names,
                max_valid_candidates=2,
                start_q=virtual_q,
                target_rotation=(
                    target_rotation if route_index == len(route_targets) - 1 else None
                ),
                deadline=deadline,
            )
            result["attempted_seeds"] += len(attempts)
            valid_attempts = [
                attempt
                for attempt in attempts
                if attempt.failure_reason == IK_SUCCESS
            ]
            if not valid_attempts:
                result["failure_reason"] = (
                    attempts[0].failure_reason if attempts else "ik_unreachable"
                )
                result["failed_segment"] = segment_index
                if attempts:
                    best = attempts[0]
                    result.update(
                        position_error_m=float(best.pos_err),
                        orientation_error_rad=float(best.ori_err),
                        joint_limit_valid=bool(best.joint_limit_valid),
                        collision_valid=bool(best.state_valid),
                        collision_reason=best.state_invalid_reason,
                    )
                return result

            result["ik_success"] = True

            trajectory = None
            info = {}
            selected_attempt = None
            for attempt in valid_attempts:
                _check_deadline(deadline, f"segment_{segment_index}_ompl", result["attempted_seeds"])
                trajectory, info = planner.plan(
                    start_q=virtual_q,
                    goal_q=attempt.q,
                    time_limit=_remaining_probe_segment_time(deadline),
                    planner_name=DEFAULT_PLANNER_NAME,
                    ignored_body_names=ignored_body_names,
                    fragile_mode=False,
                )
                if trajectory is not None:
                    selected_attempt = attempt
                    break
            if trajectory is None:
                result["failure_reason"] = "ompl_no_path_found"
                result["failed_segment"] = segment_index
                return result

            virtual_q = np.asarray(trajectory[-1], dtype=float).copy()
            if selected_attempt is not None:
                result.update(
                    position_error_m=float(selected_attempt.pos_err),
                    orientation_error_rad=float(selected_attempt.ori_err),
                    joint_limit_valid=bool(selected_attempt.joint_limit_valid),
                    collision_valid=bool(selected_attempt.state_valid),
                    collision_reason=selected_attempt.state_invalid_reason,
                )
            path_length += float(info.get("path_length_joint_space", 0.0))
            waypoint_count += int(info.get("num_waypoints", len(trajectory)))
            segment_count = segment_index

        result.update(
            success=True,
            ik_success=True,
            ompl_success=True,
            path_length=path_length,
            failure_reason=None,
            planner_name=info.get("planner_name", DEFAULT_PLANNER_NAME),
            waypoint_count=waypoint_count,
            segment_count=segment_count,
        )
        return result
    except MotionProbeTimeout as exc:
        result.update(
            success=False,
            ik_success=False,
            ompl_success=False,
            failure_reason="motion_probe_timeout",
            timeout_phase=exc.phase,
            attempted_seeds=max(result.get("attempted_seeds", 0), exc.attempted_seeds),
        )
        return result
    finally:
        result["planning_time"] = round(time.perf_counter() - started, 4)
        mutated = not (
            np.array_equal(data.qpos, qpos_before)
            and np.array_equal(data.qvel, qvel_before)
            and np.array_equal(data.ctrl, ctrl_before)
        )
        if mutated:
            data.qpos[:] = qpos_before
            data.qvel[:] = qvel_before
            data.ctrl[:] = ctrl_before
            mujoco.mj_forward(model, data)
            raise RuntimeError("motion probe mutated live simulation state")


# =============================
# SAFETY / GRIPPER HELPERS
# =============================

def _recover_to_safe_hover(ignored_body_names: Optional[Sequence[str]] = None) -> None:
    """Move away from table/contact states before retrying another pick."""
    log_event("RECOVERY", "START", phase="safe_hover", ignored_body_names=list(ignored_body_names or []))
    ok = _move_with_ompl(
        goal_q=GRASP_READY,
        grip=OPEN_GRIP,
        ignored_body_names=ignored_body_names,
        planner_name=DEFAULT_PLANNER_NAME,
        time_limit=max(DEFAULT_TIME_LIMIT, 2.0),
        label="recovery safe_hover",
        settle_steps_per_wp=DEFAULT_SETTLE_STEPS_PER_WP,
        final_settle_steps=DEFAULT_FINAL_SETTLE_STEPS,
    )
    if ok:
        log_event("RECOVERY", "OK", phase="safe_hover")
        return

    # Last-resort controller recovery. This is only used after releasing an
    # object, where the main risk is repeatedly starting near table contact.
    log_event("RECOVERY", "DIRECT_CTRL", phase="safe_hover")
    _step_sim(260, q=GRASP_READY, grip=OPEN_GRIP)
    log_event("RECOVERY", "OK", phase="safe_hover_direct")


def _move_to_grasp_ready(
    reason: str,
    grip: float = OPEN_GRIP,
    ignored_body_names: Optional[Sequence[str]] = None,
) -> bool:
    if float(np.linalg.norm(current_q() - GRASP_READY)) < 0.05:
        return True
    log_event("TRANSIT", "START", phase=reason, target="GRASP_READY")
    grasp_ready_xyz = _mujoco_fk_xyz(GRASP_READY)
    bypass = _grouped_wall_bypass_waypoints(grasp_ready_xyz)
    if bypass is not None:
        side, cartesian_gateways = bypass
        planner = _get_ompl_planner()
        fixed_valid, _, fixed_q = (
            _validated_wall_right_q(planner, ignored_body_names)
            if planner is not None
            else (False, "ompl_unavailable", _WALL_RIGHT_REF)
        )
        route_ok = fixed_valid and _move_with_ompl(
            goal_q=fixed_q,
            grip=grip,
            ignored_body_names=ignored_body_names,
            planner_name=DEFAULT_PLANNER_NAME,
            time_limit=DEFAULT_TIME_LIMIT,
            label=f"wall gateway_q {side} for transit {reason}",
            settle_steps_per_wp=DEFAULT_SETTLE_STEPS_PER_WP,
            final_settle_steps=DEFAULT_FINAL_SETTLE_STEPS,
        )
        if not route_ok:
            route_ok = all(
                _move_pose_safe(
                    waypoint,
                    grip=grip,
                    null_ref=cube_null_ref(waypoint),
                    ignored_body_names=ignored_body_names,
                    label=f"wall gateway {side} {index} for transit {reason}",
                    allow_wall_bypass=False,
                )
                for index, waypoint in enumerate(cartesian_gateways, start=1)
            )
        if not route_ok:
            log_event(
                "TRANSIT",
                "FAILED",
                phase=reason,
                target="GRASP_READY",
                failed_gateway="gateway_route",
            )
            return False
    ok = _move_with_ompl(
        goal_q=GRASP_READY,
        grip=grip,
        ignored_body_names=ignored_body_names,
        planner_name=DEFAULT_PLANNER_NAME,
        time_limit=max(DEFAULT_TIME_LIMIT, 3.0),
        label=f"transit {reason}",
        settle_steps_per_wp=DEFAULT_SETTLE_STEPS_PER_WP,
        final_settle_steps=DEFAULT_FINAL_SETTLE_STEPS,
    )
    log_event("TRANSIT", "OK" if ok else "FAILED", phase=reason, target="GRASP_READY")
    return ok


def set_grip(target: float, steps: int = 200):
    log_event("GRIPPER", "SET", grip=target, steps=steps)
    for _ in range(steps):
        data.ctrl[finger_ctrl_adr] = float(target)
        mujoco.mj_step(model, data)
        viewer.sync()


def drop(ignored_body_name: Optional[str] = None):
    """Emergency release at the current arm position."""
    global _held_grip_target
    released_object = _held_object_name or ignored_body_name
    ignored = [ignored_body_name] if _held_object_name is not None and ignored_body_name else None
    _log_arm_state("DROP", "START", object_id=released_object, ignored_body_names=ignored or [])
    set_grip(OPEN_GRIP, steps=300)
    _set_held_object(None)
    for _ in range(120):
        mujoco.mj_step(model, data)
        viewer.sync()
    _recover_to_safe_hover(ignored_body_names=ignored)
    _held_grip_target = 0.015
    _log_arm_state("DROP", "OK", object_id=released_object)


# =============================
# STARTUP WARMUP
# =============================

_initialize_available_arms()
_initialize_ik_backend()

# Phase 1: HOME.
log_event("WARMUP", "START", phase="home")
data.qpos[arm_qpos_adr] = HOME
_set_arm_ctrl(HOME, OPEN_GRIP)
for _ in range(250):
    mujoco.mj_step(model, data)
    viewer.sync()
log_event("WARMUP", "OK", phase="home")

# Phase 2: Transition to a grasp-friendly pose.
log_event("WARMUP", "START", phase="grasp_ready")
for _ in range(350):
    data.ctrl[arm_ctrl_adr] += 0.01 * (GRASP_READY - data.ctrl[arm_ctrl_adr])
    data.ctrl[finger_ctrl_adr] = OPEN_GRIP
    mujoco.mj_step(model, data)
    viewer.sync()
log_event("WARMUP", "OK", phase="grasp_ready")

_init_obstacle_monitoring()


# =============================
# HIGH-LEVEL ACTIONS
# =============================

_held_object_name: Optional[str] = None
_held_grip_target: float = 0.015
_pick_call_counts: dict[str, int] = {}


def _set_live_carry_constraint(object_name: str, active: bool) -> None:
    equality_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        f"carry_{object_name}",
    )
    if equality_id < 0:
        return
    if active:
        parent_id = int(model.body(f"{_arm_prefix(ACTIVE_ARM)}hand").id)
        body_id = int(model.body(object_name).id)
        mujoco.mj_forward(model, data)
        parent_rotation = data.xmat[parent_id].reshape(3, 3).copy()
        body_rotation = data.xmat[body_id].reshape(3, 3).copy()
        relative_position = parent_rotation.T @ (
            data.xpos[body_id] - data.xpos[parent_id]
        )
        relative_rotation = parent_rotation.T @ body_rotation
        relative_quaternion = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(
            relative_quaternion,
            relative_rotation.reshape(-1),
        )
        # Weld equality data: anchor[0:3], relative pose[3:10], torquescale[10].
        model.eq_data[equality_id, 3:6] = relative_position
        model.eq_data[equality_id, 6:10] = relative_quaternion
    data.eq_active[equality_id] = bool(active)
    mujoco.mj_forward(model, data)


def _set_held_object(object_name: Optional[str]) -> None:
    global _held_object_name
    previous = _held_object_name
    if previous == object_name:
        return
    if previous is not None:
        _set_live_carry_constraint(previous, False)
        if _live_collision_policy is not None:
            _live_collision_policy.detach_body(previous)
        for planner in _planner_by_arm.values():
            planner.detach_body(previous)
    _held_object_name = object_name
    if object_name is not None:
        _set_live_carry_constraint(object_name, True)
        if _live_collision_policy is not None:
            prefix = _arm_prefix(ACTIVE_ARM)
            _live_collision_policy.attach_body(
                object_name,
                allowed_robot_body_names=(
                    f"{prefix}hand",
                    f"{prefix}left_finger",
                    f"{prefix}right_finger",
                ),
            )
        parent_body_name = f"{_arm_prefix(ACTIVE_ARM)}hand"
        for planner in _planner_by_arm.values():
            planner.attach_body(object_name, parent_body_name)


def shutdown_runtime() -> None:
    """Best-effort release for native simulator/planner objects before CLI exit."""
    global _ompl_planner, _planner_by_arm, _live_collision_policy
    global _PINOCCHIO_ROBOT, _PINOCCHIO_MODEL, _PINOCCHIO_DATA, _PINOCCHIO_FRAME_ID
    global _hint_cache

    try:
        close = getattr(viewer, "close", None)
        if callable(close):
            close()
    except Exception:
        pass

    _planner_by_arm.clear()
    _ompl_planner = None
    _live_collision_policy = None
    _PINOCCHIO_ROBOT = None
    _PINOCCHIO_MODEL = None
    _PINOCCHIO_DATA = None
    _PINOCCHIO_FRAME_ID = None
    _hint_cache = None


def pick(
    obj,
    additional_ignored_body_names: Optional[Sequence[str]] = None,
):
    global _held_object_name, _held_grip_target

    print(f"[exec] pick({obj})")
    _log_arm_state("PICK", "START", object_id=obj)
    if obj not in name_to_cube:
        print(f"[exec] unknown object: {obj}")
        _log_arm_state("PICK", "FAILED", object_id=obj, failure_reason="unknown_object")
        return

    # Read approximate object position early for hint context (pre-settle).
    cube_id = name_to_cube[obj]
    mujoco.mj_forward(model, data)
    _hint_cube_pos = data.xpos[cube_id].copy()
    _hint_context["reach_dist"] = float(np.linalg.norm(_hint_cube_pos[:2] - BASE_XY))
    _hint_context["obstacle_dist"] = _min_obstacle_xy_distance(_hint_cube_pos)
    _hint_context["obj_class"] = "circle" if obj.startswith("circle") else "cube"

    if not _move_to_grasp_ready(f"before pick({obj})", grip=OPEN_GRIP):
        _log_arm_state("PICK", "FAILED", object_id=obj, phase="transit", failure_reason="move_to_grasp_ready_failed")
        return

    call_count = _pick_call_counts.get(obj, 0)
    profile_index = min(call_count, len(PICK_GRIP_SEQUENCE) - 1)
    _pick_call_counts[obj] = call_count + 1

    grip_target = PICK_GRIP_SEQUENCE[profile_index]
    grasp_offset = PICK_GRASP_OFFSET_SEQUENCE[profile_index]
    clearance_bonus = PICK_CLEARANCE_BONUS_SEQUENCE[profile_index]
    is_circle = obj.startswith("circle")
    if is_circle:
        # Cylinders skip profile 0 (grasp_offset=0.105 overshoots the cylinder body).
        # Start at profile 1 (grasp_offset=0.095) which is already proven reliable.
        cyl_index = min(profile_index + 1, len(COMPACT_CYLINDER_PICK_GRIP_SEQUENCE) - 1)
        profile_index = cyl_index
        grip_target = COMPACT_CYLINDER_PICK_GRIP_SEQUENCE[cyl_index]
        grasp_offset = max(COMPACT_CYLINDER_PICK_GRASP_OFFSET_SEQUENCE[cyl_index], CYLINDER_RETRY_MIN_GRASP_OFFSET)
        clearance_bonus = PICK_CLEARANCE_BONUS_SEQUENCE[cyl_index]

    # Apply profile hint on the first pick attempt for this object.
    if _hint_cache is not None and call_count == 0:
        _hint_pi = _hint_cache.preferred_grasp_profile(
            obj_class="circle" if is_circle else "cube",
            reach_dist=_hint_context.get("reach_dist", 0.5),
        )
        if _hint_pi is not None and _hint_pi != profile_index:
            if is_circle:
                _ci = min(_hint_pi, len(COMPACT_CYLINDER_PICK_GRIP_SEQUENCE) - 1)
                profile_index = _ci
                grip_target = COMPACT_CYLINDER_PICK_GRIP_SEQUENCE[_ci]
                grasp_offset = max(COMPACT_CYLINDER_PICK_GRASP_OFFSET_SEQUENCE[_ci], CYLINDER_RETRY_MIN_GRASP_OFFSET)
                clearance_bonus = PICK_CLEARANCE_BONUS_SEQUENCE[_ci]
            else:
                _pi = min(_hint_pi, len(PICK_GRIP_SEQUENCE) - 1)
                profile_index = _pi
                grip_target = PICK_GRIP_SEQUENCE[_pi]
                grasp_offset = PICK_GRASP_OFFSET_SEQUENCE[_pi]
                clearance_bonus = PICK_CLEARANCE_BONUS_SEQUENCE[_pi]

    _log_arm_state(
        "PICK_PROFILE",
        "SELECT",
        object_id=obj,
        profile_index=profile_index,
        grip=grip_target,
        grasp_offset=grasp_offset,
        clearance_bonus=clearance_bonus,
    )

    # Let the system settle before planning a new motion.
    _step_sim(80, grip=OPEN_GRIP)
    mujoco.mj_forward(model, data)
    cube_pos = data.xpos[cube_id].copy()
    pose_failure_reason = _object_pose_failure_reason(cube_pos)
    if pose_failure_reason is not None:
        _log_arm_state(
            "PICK",
            "FAILED",
            object_id=obj,
            phase="precheck",
            object_xyz=_round_vec(cube_pos, 4),
            distance_to_target=_distance_xy_to_base(cube_pos),
            failure_reason=pose_failure_reason,
        )
        return
    if is_circle and float(cube_pos[2]) < CYLINDER_TIPPED_CENTER_Z:
        grasp_offset = min(grasp_offset, CYLINDER_TIPPED_GRASP_OFFSET)
        grip_target = min(grip_target, CYLINDER_TIPPED_GRIP)
        clearance_bonus += 0.02
        _log_arm_state(
            "PICK_PROFILE",
            "TIPPED_CYLINDER",
            object_id=obj,
            object_z=round(float(cube_pos[2]), 4),
            threshold=CYLINDER_TIPPED_CENTER_Z,
            grip=grip_target,
            grasp_offset=grasp_offset,
            clearance_bonus=clearance_bonus,
        )
    obstacle_distance = _min_obstacle_xy_distance(cube_pos)
    approach_clearance = APPROACH_CLEARANCE + clearance_bonus
    cautious_motion = False
    _log_arm_state(
        "PICK_PRECHECK",
        "START",
        object_id=obj,
        object_xyz=_round_vec(cube_pos, 4),
        obstacle_distance=obstacle_distance if np.isfinite(obstacle_distance) else None,
        distance_to_target=_distance_xy_to_base(cube_pos),
        approach_clearance=approach_clearance,
    )
    if obstacle_distance < MIN_PICK_OBSTACLE_CLEARANCE and any(
        name not in _wall_ids for name in _obstacle_ids
    ):
        print(
            f"[exec][OBSTACLE_AVOID] cancel pick({obj}): "
            f"distance={obstacle_distance:.3f}m threshold={MIN_PICK_OBSTACLE_CLEARANCE:.3f}m"
        )
        _log_arm_state(
            "PICK",
            "FAILED",
            object_id=obj,
            phase="precheck",
            failure_reason="object_too_close_to_obstacle",
            obstacle_distance=round(float(obstacle_distance), 4),
            threshold=MIN_PICK_OBSTACLE_CLEARANCE,
        )
        return
    if obstacle_distance < CAUTIOUS_OBSTACLE_CLEARANCE:
        cautious_motion = True
        approach_clearance += 0.06
        grip_target = min(
            grip_target,
            OBSTACLE_CAUTIOUS_CYLINDER_GRIP if is_circle else OBSTACLE_CAUTIOUS_CUBE_GRIP,
        )
        if not is_circle and obstacle_distance < 0.12:
            grip_target = min(grip_target, 0.014)
            _log_arm_state(
                "PICK_PROFILE",
                "TIGHT_OBSTACLE_CUBE",
                object_id=obj,
                obstacle_distance=round(float(obstacle_distance), 4),
                grip=grip_target,
                grasp_offset=grasp_offset,
            )
        print(
            f"[exec][OBSTACLE_AVOID] {obj} near obstacle "
            f"distance={obstacle_distance:.3f}m; using cautious high-clearance approach"
        )
        _log_arm_state(
            "OBSTACLE_AVOID",
            "NEAR_CAUTIOUS",
            object_id=obj,
            obstacle_distance=obstacle_distance,
            approach_clearance=approach_clearance,
            grip=grip_target,
        )
    else:
        _log_arm_state(
            "OBSTACLE_AVOID",
            "CLEAR",
            object_id=obj,
            obstacle_distance=obstacle_distance if np.isfinite(obstacle_distance) else None,
            approach_clearance=approach_clearance,
        )

    cube_ref = cube_null_ref(cube_pos)
    object_xy_distance = float(np.linalg.norm(cube_pos[:2] - BASE_XY))
    if is_circle and object_xy_distance > FAR_PICK_XY_DISTANCE:
        grip_target = max(grip_target, 0.014)
        grasp_offset = max(grasp_offset, 0.105)
        _log_arm_state(
            "PICK_PROFILE",
            "FAR_CYLINDER",
            object_id=obj,
            grip=grip_target,
            grasp_offset=grasp_offset,
            xy_distance=round(object_xy_distance, 4),
        )
    if object_xy_distance > FAR_PICK_XY_DISTANCE:
        grasp_offset += 0.02
    pregrasp_clearance = approach_clearance
    if object_xy_distance > FAR_PICK_XY_DISTANCE:
        pregrasp_clearance = min(pregrasp_clearance, 0.24)
        _log_arm_state(
            "PICK_PROFILE",
            "FAR_PREGRASP",
            object_id=obj,
            obstacle_distance=obstacle_distance if np.isfinite(obstacle_distance) else None,
            approach_clearance=pregrasp_clearance,
            xy_distance=round(object_xy_distance, 4),
        )
    grasp_candidates = _grasp_pose_candidates(
        cube_pos,
        grasp_offset,
        pregrasp_clearance,
        approach_clearance,
        lateral_enabled=not is_circle,
    )
    selected_grasp = _select_grasp_pose_candidate(obj, grasp_candidates, cube_ref)
    if selected_grasp is None:
        _log_arm_state(
            "PICK",
            "FAILED",
            object_id=obj,
            phase="grasp_candidate_selection",
            failure_reason="no_ik_feasible_collision_free_grasp_candidate",
        )
        return
    grasp_candidate_name = str(selected_grasp["name"])
    grasp_rotation = np.asarray(selected_grasp["rotation"], dtype=float)
    pregrasp_xyz = np.asarray(selected_grasp["pregrasp_xyz"], dtype=float)
    grasp_xyz = np.asarray(selected_grasp["grasp_xyz"], dtype=float)
    lift_xyz = np.asarray(selected_grasp["lift_xyz"], dtype=float)
    _log_arm_state(
        "PICK_TARGETS",
        "SET",
        object_id=obj,
        object_xyz=_round_vec(cube_pos, 4),
        pregrasp_xyz=_round_vec(pregrasp_xyz, 4),
        grasp_xyz=_round_vec(grasp_xyz, 4),
        lift_xyz=_round_vec(lift_xyz, 4),
        grasp_candidate=grasp_candidate_name,
        approach_vector=_round_vec(selected_grasp["approach"], 3),
        cautious_motion=cautious_motion,
    )

    # 1) Move above the cube.
    if not _move_pose_safe(
        pregrasp_xyz,
        grip=OPEN_GRIP,
        null_ref=cube_ref,
        ignored_body_names=None,
        label=f"pick({obj}) pregrasp {grasp_candidate_name}",
        cautious_motion=cautious_motion,
        target_rotation=grasp_rotation,
    ):
        _check_obstacles_fallen(f"pick({obj}) pregrasp")
        _log_arm_state("PICK", "FAILED", object_id=obj, phase="pregrasp", target_xyz=_round_vec(pregrasp_xyz, 4), failure_reason="move_pregrasp_failed")
        return

    # 2) Move to the grasp pose while ignoring the target cube. Recovery may
    # additionally allow contact with other cubes from an already-collapsed
    # suffix while the top cube is being disentangled. This override is never
    # used by normal task execution.
    grasp_ignored = list(
        dict.fromkeys([obj, *(additional_ignored_body_names or [])])
    )
    transit_ignored = list(dict.fromkeys(additional_ignored_body_names or []))
    if not _move_pose_safe(
        grasp_xyz,
        grip=OPEN_GRIP,
        null_ref=cube_ref,
        ignored_body_names=grasp_ignored,
        label=f"pick({obj}) grasp {grasp_candidate_name}",
        cautious_motion=cautious_motion,
        target_rotation=grasp_rotation,
    ):
        _check_obstacles_fallen(f"pick({obj}) grasp")
        _log_arm_state("PICK", "FAILED", object_id=obj, phase="grasp", target_xyz=_round_vec(grasp_xyz, 4), failure_reason="move_grasp_failed")
        return

    # 3) Close gripper and let contact settle.
    _log_arm_state("PICK", "GRIP_CLOSE", object_id=obj, phase="close_gripper", target_xyz=_round_vec(grasp_xyz, 4))
    set_grip(grip_target, steps=320 if cautious_motion else 260)
    for _ in range(110 if cautious_motion else 70):
        mujoco.mj_step(model, data)
        viewer.sync()

    finger_contacts = _finger_contacts_object(obj)
    if len(finger_contacts) < 2:
        _log_arm_state(
            "PICK",
            "FAILED",
            object_id=obj,
            phase="grasp_contact",
            contact_body_names=sorted(finger_contacts),
            failure_reason="bilateral_finger_contact_missing",
        )
        set_grip(OPEN_GRIP, steps=180)
        _recover_to_safe_hover(ignored_body_names=[obj])
        return

    _set_held_object(obj)
    _held_grip_target = grip_target

    # 4) Lift straight up using the same bounded recovery contact policy.
    if not _move_pose_safe(
        lift_xyz,
        grip=grip_target,
        null_ref=cube_ref,
        ignored_body_names=transit_ignored,
        label=f"pick({obj}) lift {grasp_candidate_name}",
        cautious_motion=cautious_motion,
        target_rotation=grasp_rotation,
    ):
        # If lift planning fails after grasping, release immediately and let the
        # closed-loop system replan from the table state.
        drop(obj)
        _set_held_object(None)
        _check_obstacles_fallen(f"pick({obj}) lift")
        _log_arm_state("PICK", "FAILED", object_id=obj, phase="lift", target_xyz=_round_vec(lift_xyz, 4), failure_reason="move_lift_failed")
        return

    _check_obstacles_fallen(f"pick({obj})")
    lifted, z = _object_lifted(obj)
    if not lifted:
        print(f"[exec][PICK_RETRY] {obj} not lifted after grasp z={z:.3f}; release and retry")
        _log_arm_state("PICK", "FAILED", object_id=obj, phase="lift_check", failure_reason="object_not_lifted", object_z=round(z, 4))
        drop(obj)
        _set_held_object(None)
        _check_obstacles_fallen(f"pick({obj}) lift_check")
        return
    _log_arm_state("PICK", "OK", object_id=obj, object_z=round(z, 4))



def place(
    x,
    y,
    obj=None,
    target_z: float = 0.83,
    release_lift: float = 0.06,
    post_place_ignored_body_names: Optional[Sequence[str]] = None,
):
    global _held_object_name, _held_grip_target

    print(f"[exec] place({x:.3f}, {y:.3f}, z={target_z:.3f})")
    _log_arm_state(
        "PLACE",
        "START",
        object_id=obj or _held_object_name,
        target_xyz=[round(float(x), 4), round(float(y), 4), round(float(target_z), 4)],
        release_lift=round(float(release_lift), 4),
    )
    if _held_object_name is None:
        print("[exec] no cube is currently held")
        _log_arm_state("PLACE", "FAILED", object_id=obj, failure_reason="no_object_held")
        return

    obj = _held_object_name

    # Settled object centre after release. target_z defaults to the table
    # placement height; stacking scripts pass a higher target_z for upper cubes.
    place_pos = np.array([x, y, target_z])
    release_pos = np.array([x, y, target_z + release_lift])
    mujoco.mj_forward(model, data)
    held_object_pos = data.xpos[name_to_cube[obj]].copy()
    held_offset_from_ee = held_object_pos - data.xpos[ee_id].copy()
    place_ref = cube_null_ref(place_pos)
    obstacle_distance = _min_obstacle_xy_distance(place_pos)
    approach_clearance = APPROACH_CLEARANCE
    cautious_motion = False
    _log_arm_state(
        "PLACE_PRECHECK",
        "START",
        object_id=obj,
        target_xyz=_round_vec(place_pos, 4),
        obstacle_distance=obstacle_distance if np.isfinite(obstacle_distance) else None,
        distance_to_target=round(float(np.linalg.norm(np.asarray([x, y]) - np.asarray(_object_xyz(obj)[:2]))), 4) if _object_xyz(obj) else None,
        approach_clearance=approach_clearance,
    )
    if obstacle_distance < CAUTIOUS_OBSTACLE_CLEARANCE:
        cautious_motion = True
        approach_clearance += 0.06
        print(
            f"[exec][OBSTACLE_AVOID] place target near obstacle "
            f"distance={obstacle_distance:.3f}m; using cautious high-clearance preplace"
        )
        _log_arm_state(
            "OBSTACLE_AVOID",
            "NEAR_CAUTIOUS",
            object_id=obj,
            phase="place",
            obstacle_distance=obstacle_distance,
            approach_clearance=approach_clearance,
        )

    # Compensate the measured grasp transform. Assuming a fixed GRASP_OFFSET
    # puts an off-centre or slightly tilted cube into the support even when the
    # hand itself reaches its IK target. XY and Z are both observed live.
    compensated_xy = release_pos[:2] - held_offset_from_ee[:2]
    preplace_xyz = np.array(
        [compensated_xy[0], compensated_xy[1], place_pos[2] + approach_clearance]
    )
    release_xyz = release_pos - held_offset_from_ee
    retreat_xyz = np.array(
        [
            compensated_xy[0],
            compensated_xy[1],
            release_pos[2] + approach_clearance,
        ]
    )
    _log_arm_state(
        "PLACE_TARGETS",
        "SET",
        object_id=obj,
        target_xyz=_round_vec(place_pos, 4),
        preplace_xyz=_round_vec(preplace_xyz, 4),
        release_xyz=_round_vec(release_xyz, 4),
        retreat_xyz=_round_vec(retreat_xyz, 4),
        held_offset_from_ee=_round_vec(held_offset_from_ee, 4),
        cautious_motion=cautious_motion,
    )

    bypass = _grouped_wall_bypass_waypoints(preplace_xyz)
    if bypass is not None:
        side, cartesian_gateways = bypass
        planner = _get_ompl_planner()
        fixed_valid, fixed_reason, fixed_q = (
            _validated_wall_right_q(planner, None)
            if planner is not None
            else (False, "ompl_unavailable", _WALL_RIGHT_REF)
        )
        _log_arm_state(
            "WALL_BYPASS",
            "SELECT",
            object_id=obj,
            phase=side,
            target_xyz=_round_vec(place_pos, 4),
            gateway_q=_round_vec(fixed_q, 4),
            gateway_mode="fixed_q" if fixed_valid else "cartesian",
            fixed_gateway_validation=fixed_reason,
        )
        route_ok = False
        if fixed_valid:
            route_ok = _move_with_ompl(
                goal_q=fixed_q,
                grip=_held_grip_target,
                ignored_body_names=None,
                planner_name=DEFAULT_PLANNER_NAME,
                time_limit=DEFAULT_TIME_LIMIT,
                label=f"grouped wall gateway_q {side}",
                settle_steps_per_wp=DEFAULT_SETTLE_STEPS_PER_WP,
                final_settle_steps=DEFAULT_FINAL_SETTLE_STEPS,
            )
        if not route_ok:
            route_ok = all(
                _move_pose_safe(
                    waypoint,
                    grip=_held_grip_target,
                    null_ref=cube_null_ref(waypoint),
                    ignored_body_names=None,
                    label=f"grouped wall gateway {side} {index}",
                    allow_wall_bypass=False,
                )
                for index, waypoint in enumerate(cartesian_gateways, start=1)
            )
        if not route_ok:
            drop(obj)
            _set_held_object(None)
            _log_arm_state(
                "WALL_BYPASS",
                "FAILED",
                object_id=obj,
                phase=side,
                failure_reason="wall_gateway_route_failed",
            )
            return
        still_held, object_z = _object_lifted(obj)
        object_xyz = np.asarray(_object_xyz(obj), dtype=float)
        grasp_distance = float(
            np.linalg.norm(object_xyz - np.asarray(_ee_xyz(), dtype=float))
        )
        if not still_held or grasp_distance > 0.18:
            drop(obj)
            _set_held_object(None)
            _log_arm_state(
                "WALL_BYPASS",
                "FAILED",
                object_id=obj,
                phase=side,
                object_z=round(object_z, 4),
                distance_to_target=round(grasp_distance, 4),
                failure_reason="object_lost_during_wall_bypass",
            )
            return
        _log_arm_state(
            "WALL_BYPASS",
            "OK",
            object_id=obj,
            phase=side,
        )

    # 1) Move above the goal while still holding the cube.
    if not _move_pose_safe(
        preplace_xyz,
        grip=_held_grip_target,
        null_ref=place_ref,
        ignored_body_names=None,
        label=f"place({x:.3f}, {y:.3f}, {target_z:.3f}) preplace",
        cautious_motion=cautious_motion,
    ):
        drop(obj)
        _set_held_object(None)
        _log_arm_state("PLACE", "FAILED", object_id=obj, phase="preplace", target_xyz=_round_vec(preplace_xyz, 4), failure_reason="move_preplace_failed")
        return

    # The object can settle to a different pose inside the fingers during the
    # long preplace transit. Refresh the grasp transform immediately before
    # the short final descent so compensation is not based on stale geometry.
    mujoco.mj_forward(model, data)
    held_offset_from_ee = (
        data.xpos[name_to_cube[obj]].copy() - data.xpos[ee_id].copy()
    )
    release_xyz = release_pos - held_offset_from_ee
    retreat_xyz[:2] = release_xyz[:2]
    _log_arm_state(
        "PLACE_RELEASE_COMPENSATION",
        "UPDATED",
        object_id=obj,
        target_xyz=_round_vec(release_xyz, 4),
        held_offset_from_ee=_round_vec(held_offset_from_ee, 4),
    )

    # 2) Move to the release pose.
    if not _move_pose_safe(
        release_xyz,
        grip=_held_grip_target,
        null_ref=place_ref,
        ignored_body_names=None,
        label=f"place({x:.3f}, {y:.3f}, {target_z:.3f}) release",
        cautious_motion=cautious_motion,
    ):
        drop(obj)
        _set_held_object(None)
        _log_arm_state("PLACE", "FAILED", object_id=obj, phase="release", target_xyz=_round_vec(release_xyz, 4), failure_reason="move_release_failed")
        return

    # 3) Let the arm settle before opening.
    _log_arm_state(
        "PLACE",
        "SETTLE_BEFORE_OPEN",
        object_id=obj,
        target_xyz=_round_vec(release_xyz, 4),
        steps=CONFIG.grasp.place_settle_before_open_steps,
    )
    for _ in range(CONFIG.grasp.place_settle_before_open_steps):
        mujoco.mj_step(model, data)
        viewer.sync()

    print(f"[exec] finger before open: {_finger_pos():.4f}")
    _log_arm_state("PLACE", "GRIP_OPEN", object_id=obj, target_xyz=_round_vec(release_xyz, 4), finger_before=round(_finger_pos(), 4))
    guide_target = min(
        OPEN_GRIP,
        _finger_pos() + CONFIG.grasp.release_guide_clearance_m,
    )
    _log_arm_state(
        "PLACE",
        "GRIP_GUIDED_RELEASE",
        object_id=obj,
        target_xyz=_round_vec(place_pos, 4),
        grip=guide_target,
    )
    set_grip(guide_target, steps=CONFIG.grasp.release_guide_steps)
    _step_sim(CONFIG.grasp.release_guide_settle_steps, grip=guide_target)
    set_grip(OPEN_GRIP, steps=CONFIG.grasp.release_open_steps)
    _set_held_object(None)
    print(f"[exec] finger after open:  {_finger_pos():.4f}")
    _log_arm_state("PLACE", "GRIP_OPENED", object_id=obj, target_xyz=_round_vec(place_pos, 4), finger_after=round(_finger_pos(), 4))

    # 4) Let the cube fall and settle.
    for _ in range(CONFIG.grasp.release_post_open_settle_steps):
        mujoco.mj_step(model, data)
        viewer.sync()

    post_release_ignored = [obj]
    if post_place_ignored_body_names is None:
        contact_ignored = _current_robot_movable_contacts(post_release_ignored)
        post_release_ignored = _merge_ignored_body_names(
            post_release_ignored,
            contact_ignored,
        )
        if contact_ignored:
            _log_arm_state(
                "PLACE",
                "POST_RELEASE_CONTACTS",
                object_id=obj,
                ignored_body_names=post_release_ignored,
                contact_body_names=contact_ignored,
            )

    # 5) Retreat upward using OMPL.
    if not _move_pose_safe(
        retreat_xyz,
        grip=OPEN_GRIP,
        null_ref=place_ref,
        ignored_body_names=post_release_ignored,
        label=f"place({x:.3f}, {y:.3f}, {target_z:.3f}) retreat",
        cautious_motion=cautious_motion,
    ):
        # Retreat failure after release is not fatal, but we still keep the
        # controller in a safe open state.
        print(f"[exec] retreat failed after place({x:.3f}, {y:.3f}, {target_z:.3f})")
        _log_arm_state("PLACE", "RETREAT_FAILED", object_id=obj, target_xyz=_round_vec(retreat_xyz, 4), failure_reason="retreat_failed_after_release")

    _set_held_object(None)
    _held_grip_target = 0.015
    post_place_ignored = (
        post_release_ignored
        if post_place_ignored_body_names is None
        else list(post_place_ignored_body_names)
    )
    _move_to_grasp_ready(
        f"after place({x:.3f}, {y:.3f}, {target_z:.3f})",
        grip=OPEN_GRIP,
        ignored_body_names=post_place_ignored or None,
    )
    _check_obstacles_fallen(f"place({x:.3f}, {y:.3f}, {target_z:.3f})")
    _log_arm_state(
        "PLACE",
        "OK",
        object_id=obj,
        target_xyz=[round(float(x), 4), round(float(y), 4), round(float(target_z), 4)],
    )
