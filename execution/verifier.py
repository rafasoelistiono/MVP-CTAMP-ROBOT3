from __future__ import annotations

import math
from typing import Protocol

from configuration import VerificationConfig


class PoseProvider(Protocol):
    def object_pose(self, object_id: str) -> tuple[float, float, float]: ...

    def all_object_poses(self) -> dict[str, tuple[float, float, float]]: ...

    def held_object_name(self) -> str | None: ...


class ObservedPredicateVerifier:
    TOLERANCES = {
        "at_x_m": 0.055,
        "at_y_m": 0.035,
        "at_z_m": 0.050,
        "on_xy_m": 0.045,
        "on_z_m": 0.035,
        "pick_z_m": 0.90,
    }

    def __init__(
        self,
        provider: PoseProvider,
        layer_height_m: float = 0.06,
        config: VerificationConfig | None = None,
    ):
        self.provider = provider
        self.layer_height_m = float(layer_height_m)
        configured = config or VerificationConfig()
        self.tolerances = {
            "at_x_m": configured.at_x_m,
            "at_y_m": configured.at_y_m,
            "at_z_m": configured.at_z_m,
            "on_xy_m": configured.on_xy_m,
            "on_z_m": configured.on_z_m,
            "pick_z_m": configured.pick_z_m,
            "stack_max_tilt_rad": configured.stack_max_tilt_rad,
            "stack_max_linear_velocity_mps": configured.stack_max_linear_velocity_mps,
            "stack_max_angular_velocity_radps": configured.stack_max_angular_velocity_radps,
        }

    def check_at(
        self,
        obj_id: str,
        slot_pose: tuple[float, float, float],
    ) -> bool:
        actual = self.provider.object_pose(obj_id)
        return (
            abs(actual[0] - slot_pose[0]) <= self.tolerances["at_x_m"]
            and abs(actual[1] - slot_pose[1]) <= self.tolerances["at_y_m"]
            and abs(actual[2] - slot_pose[2]) <= self.tolerances["at_z_m"]
        )

    def check_on(self, upper_id: str, lower_id: str) -> bool:
        upper = self.provider.object_pose(upper_id)
        lower = self.provider.object_pose(lower_id)
        xy_error = math.dist(upper[:2], lower[:2])
        expected_height = self.layer_height_m
        extent = getattr(self.provider, "object_vertical_half_extent", None)
        if callable(extent):
            expected_height = float(extent(upper_id)) + float(extent(lower_id))
        z_error = abs((upper[2] - lower[2]) - expected_height)
        return (
            xy_error <= self.tolerances["on_xy_m"]
            and z_error <= self.tolerances["on_z_m"]
        )

    def check_stable(self, obj_id: str, *, include_velocity: bool = False) -> bool:
        return self.stability_failure_reason(
            obj_id,
            include_velocity=include_velocity,
        ) is None

    def stability_failure_reason(
        self,
        obj_id: str,
        *,
        include_velocity: bool = False,
    ) -> str | None:
        orientation_fn = getattr(self.provider, "object_orientation", None)
        if callable(orientation_fn):
            w, x, y, z = orientation_fn(obj_id)
            norm = math.sqrt(w * w + x * x + y * y + z * z)
            if norm <= 1e-12:
                return "invalid_orientation"
            w /= norm
            x /= norm
            y /= norm
            z /= norm
            # A cube is stable when any one of its three equivalent local
            # axes is aligned with world Z. Using only local Z incorrectly
            # rejects a perfectly flat cube rotated by 90 degrees.
            world_z_alignments = (
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            )
            world_z_alignment = max(
                abs(max(-1.0, min(1.0, value)))
                for value in world_z_alignments
            )
            tilt = math.acos(world_z_alignment)
            if tilt > self.tolerances["stack_max_tilt_rad"]:
                return f"tilt:{tilt:.4f}"

        velocity_fn = getattr(self.provider, "object_velocity", None)
        if include_velocity and callable(velocity_fn):
            linear, angular = velocity_fn(obj_id)
            linear_speed = math.sqrt(sum(value * value for value in linear))
            angular_speed = math.sqrt(sum(value * value for value in angular))
            if linear_speed > self.tolerances["stack_max_linear_velocity_mps"]:
                return f"linear_velocity:{linear_speed:.4f}"
            if angular_speed > self.tolerances["stack_max_angular_velocity_radps"]:
                return f"angular_velocity:{angular_speed:.4f}"
        return None

    def check_clear(self, obj_id: str) -> bool:
        base = self.provider.object_pose(obj_id)
        for other_id, pose in self.provider.all_object_poses().items():
            if other_id == obj_id:
                continue
            if (
                math.dist(base[:2], pose[:2]) <= self.tolerances["on_xy_m"]
                and pose[2] > base[2] + self.layer_height_m / 2.0
            ):
                return False
        return True

    def check_handempty(self) -> bool:
        return self.provider.held_object_name() is None

    def check_holding(self, obj_id: str) -> bool:
        return (
            self.provider.held_object_name() == obj_id
            and self.provider.object_pose(obj_id)[2] > self.tolerances["pick_z_m"]
        )

    def verify_group_row_alignment(
        self,
        object_ids: tuple[str, ...],
        axis: str,
        tolerance_m: float | None = None,
    ) -> bool:
        """Check that objects are aligned in a row along the given axis."""
        if len(object_ids) < 2:
            return True
        tol = tolerance_m if tolerance_m is not None else self.tolerances["at_y_m"]
        poses = [self.provider.object_pose(oid) for oid in object_ids]
        if axis == "x":
            fixed_coords = [p[1] for p in poses]
        else:
            fixed_coords = [p[0] for p in poses]
        mean = sum(fixed_coords) / len(fixed_coords)
        return all(abs(c - mean) <= tol for c in fixed_coords)

    def verify_group_spacing(
        self,
        object_ids: tuple[str, ...],
        expected_spacing: float,
        axis: str,
        tolerance_m: float | None = None,
    ) -> bool:
        """Check that objects in a group have correct spacing along axis."""
        if len(object_ids) < 2:
            return True
        tol = tolerance_m if tolerance_m is not None else self.tolerances["at_x_m"]
        poses = [self.provider.object_pose(oid) for oid in object_ids]
        if axis == "x":
            moving_coords = sorted(p[0] for p in poses)
        else:
            moving_coords = sorted(p[1] for p in poses)
        for i in range(len(moving_coords) - 1):
            if abs((moving_coords[i + 1] - moving_coords[i]) - expected_spacing) > tol:
                return False
        return True

    def verify_group_color_assignment(
        self,
        object_ids: tuple[str, ...],
        expected_color: str,
        object_colors: dict[str, str],
    ) -> bool:
        """Check that all objects in a group have the expected color."""
        return all(
            object_colors.get(oid) == expected_color for oid in object_ids
        )

    def verify_no_grouped_slot_overlap(
        self,
        slots: dict[str, tuple[float, float, float]],
        gt=None,
        min_distance: float = 0.066,
    ) -> bool:
        """Check that no two grouped slots physically overlap."""
        poses = list(slots.values())
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                if math.dist(poses[i][:2], poses[j][:2]) < min_distance:
                    return False
        return True

    def verify_all_grouped_objects_stable(
        self,
        object_ids: tuple[str, ...],
        include_velocity: bool = True,
    ) -> bool:
        """Check that all objects in a group are stable."""
        return all(
            self.check_stable(oid, include_velocity=include_velocity)
            for oid in object_ids
        )

    def evaluate(
        self,
        predicate: dict,
        slots: dict[str, tuple[float, float, float]],
    ) -> bool:
        name = predicate.get("name")
        args = predicate.get("args", [])
        if name == "at" and len(args) == 2:
            slot_id = args[1]
            return slot_id in slots and self.check_at(args[0], slots[slot_id])
        if name == "on" and len(args) == 2:
            return self.check_on(args[0], args[1])
        if name == "clear" and len(args) == 1:
            return self.check_clear(args[0])
        if name == "handempty" and not args:
            return self.check_handempty()
        if name == "holding" and len(args) == 1:
            return self.check_holding(args[0])
        return False
