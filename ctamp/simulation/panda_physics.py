"""Actuator-driven Panda execution and contact-based grasp validation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .mujoco_backend import MuJoCoBackend
from .panda_ik import PandaIKSolver


@dataclass(frozen=True)
class PhysicalGraspResult:
    success: bool
    bilateral_contact: bool
    left_contact: bool
    right_contact: bool
    initial_cube_z: float
    lifted_cube_z: float
    lift_height: float
    arm_tracking_error: float
    reason: str | None = None


class PandaPhysicsExecutor:
    """Drive the menagerie Panda position actuators without teleporting arm joints."""

    def __init__(
        self,
        backend: MuJoCoBackend,
        viewer=None,
        realtime_factor: float = 1.0,
    ) -> None:
        if realtime_factor <= 0:
            raise ValueError("realtime_factor must be positive")
        self.backend = backend
        self.model = backend.model
        self.data = backend.data
        self.mujoco = backend._mujoco()
        self.viewer = viewer
        self.realtime_factor = float(realtime_factor)
        self.ik = PandaIKSolver(backend)
        if self.model.nu < 8:
            raise RuntimeError("Panda physics executor requires 8 position actuators")
        self.arm_actuators = list(range(7))
        self.gripper_actuator = 7
        self.data.ctrl[self.arm_actuators] = self.ik.current_qpos()
        self.data.ctrl[self.gripper_actuator] = 0.05
        self.mujoco.mj_forward(self.model, self.data)

    def _step(self) -> None:
        self.mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
            time.sleep(float(self.model.opt.timestep) / self.realtime_factor)

    def settle(self, steps: int = 200) -> None:
        for _ in range(steps):
            self._step()

    def initialize_arm(self, qpos) -> None:
        """Set the collision-free initial state before physics execution begins."""
        self.ik.set_qpos(qpos)
        self.data.ctrl[self.arm_actuators] = np.asarray(qpos, dtype=float)
        self.mujoco.mj_forward(self.model, self.data)

    def command_arm(
        self, target, tolerance: float = 0.025, max_steps: int = 350,
    ) -> tuple[bool, float]:
        target_array = np.asarray(target, dtype=float)
        self.data.ctrl[self.arm_actuators] = target_array
        error = float("inf")
        for _ in range(max_steps):
            self._step()
            error = float(np.max(np.abs(self.ik.current_qpos() - target_array)))
            if error <= tolerance:
                return True, error
        return False, error

    def follow_joint_path(self, waypoints, max_joint_step: float = 0.035) -> tuple[bool, float]:
        worst_error = 0.0
        for waypoint in waypoints:
            start = self.ik.current_qpos()
            goal = np.asarray(waypoint, dtype=float)
            count = max(1, int(np.ceil(np.max(np.abs(goal - start)) / max_joint_step)))
            for alpha in np.linspace(1.0 / count, 1.0, count):
                target = start + alpha * (goal - start)
                success, error = self.command_arm(
                    target, tolerance=0.025, max_steps=320,
                )
                worst_error = max(worst_error, error)
                if not success:
                    return False, worst_error
        return True, worst_error

    def open_gripper(self, steps: int = 250) -> None:
        self.data.ctrl[self.gripper_actuator] = 0.05
        self.settle(steps)

    def close_gripper(self, width: float = 0.052, steps: int = 500) -> None:
        # The actuator controls one finger joint in meters; equality mirrors it.
        self.data.ctrl[self.gripper_actuator] = float(np.clip(width / 2.0, 0.0, 0.05))
        self.settle(steps)

    def set_carry_constraint(self, object_id: str, active: bool) -> None:
        """Activate transport weld only after contact-validated acquisition."""
        equality_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_EQUALITY, f"carry_{object_id}",
        )
        if equality_id < 0:
            raise RuntimeError(f"carry equality missing for {object_id}")
        if active:
            hand_id = self.model.body("hand").id
            cube_id = self.model.body(f"cube_{object_id}").id
            self.mujoco.mj_forward(self.model, self.data)
            hand_rotation = self.data.xmat[hand_id].reshape(3, 3).copy()
            cube_rotation = self.data.xmat[cube_id].reshape(3, 3).copy()
            relative_position = hand_rotation.T @ (
                self.data.xpos[cube_id] - self.data.xpos[hand_id]
            )
            relative_rotation = hand_rotation.T @ cube_rotation
            relative_quaternion = np.zeros(4, dtype=float)
            self.mujoco.mju_mat2Quat(
                relative_quaternion, relative_rotation.reshape(-1),
            )
            self.model.eq_data[equality_id, 3:6] = relative_position
            self.model.eq_data[equality_id, 6:10] = relative_quaternion
        self.data.eq_active[equality_id] = bool(active)
        self.mujoco.mj_forward(self.model, self.data)

    def finger_contacts(self, object_id: str) -> tuple[bool, bool]:
        cube_body = self.model.body(f"cube_{object_id}").id
        left_body = self.model.body("left_finger").id
        right_body = self.model.body("right_finger").id
        left = right = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            bodies = {
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            }
            left |= cube_body in bodies and left_body in bodies
            right |= cube_body in bodies and right_body in bodies
        return left, right

    def servo_site_position(
        self,
        target,
        tolerance: float = 0.004,
        iterations: int = 30,
    ) -> tuple[bool, float]:
        """Live Jacobian correction through actuators; never writes arm qpos."""
        target_array = np.asarray(target, dtype=float)
        jacobian_full = np.zeros((3, self.model.nv))
        error_norm = float("inf")
        for _ in range(iterations):
            self.mujoco.mj_forward(self.model, self.data)
            error = target_array - self.ik.site_position()
            error_norm = float(np.linalg.norm(error))
            if error_norm <= tolerance:
                return True, error_norm
            jacobian_full.fill(0.0)
            self.mujoco.mj_jacSite(
                self.model, self.data, jacobian_full, None, self.ik.site_id,
            )
            jacobian = jacobian_full[:, self.ik.dof_indices]
            system = jacobian @ jacobian.T + 0.004 * np.eye(3)
            delta = jacobian.T @ np.linalg.solve(system, error)
            delta = np.clip(delta, -0.025, 0.025)
            q_target = np.clip(
                self.ik.current_qpos() + delta,
                self.ik.lower + 1e-5,
                self.ik.upper - 1e-5,
            )
            self.command_arm(q_target, tolerance=0.012, max_steps=100)
        return False, error_norm

    def cube_z(self, object_id: str) -> float:
        return float(self.data.xpos[self.model.body(f"cube_{object_id}").id, 2])

    def validate_grasp_and_lift(
        self,
        object_id: str,
        approach_qpos,
        lift_qpos,
        grasp_site_target=None,
        grip_width: float = 0.052,
        minimum_lift: float = 0.04,
    ) -> PhysicalGraspResult:
        initial_z = self.cube_z(object_id)
        _, error = self.follow_joint_path([approach_qpos], max_joint_step=0.012)
        if grasp_site_target is not None:
            _, servo_error = self.servo_site_position(grasp_site_target)
            error = max(error, servo_error)
        self.close_gripper(grip_width)
        left, right = self.finger_contacts(object_id)
        if left and right:
            self.set_carry_constraint(object_id, True)
        tracked, lift_error = self.command_arm(lift_qpos, max_steps=600)
        self.settle(150)
        lifted_z = self.cube_z(object_id)
        lift_height = lifted_z - initial_z
        success = tracked and left and right and lift_height >= minimum_lift
        reason = None
        if not left or not right:
            reason = "bilateral finger contact not established"
        elif lift_height < minimum_lift:
            reason = "cube did not remain in gripper during lift"
        elif not tracked:
            reason = "arm failed to track lift"
        return PhysicalGraspResult(success, left and right, left, right,
                                   initial_z, lifted_z, lift_height,
                                   max(error, lift_error), reason)
