from __future__ import annotations

from execution.verifier import ObservedPredicateVerifier


class FakeProvider:
    def __init__(self, poses, held=None):
        self.poses = poses
        self.held = held

    def object_pose(self, object_id):
        return self.poses[object_id]

    def all_object_poses(self):
        return dict(self.poses)

    def held_object_name(self):
        return self.held


class DynamicProvider(FakeProvider):
    def __init__(self, poses, orientations, velocities):
        super().__init__(poses)
        self.orientations = orientations
        self.velocities = velocities

    def object_orientation(self, object_id):
        return self.orientations[object_id]

    def object_velocity(self, object_id):
        return self.velocities[object_id]


def test_at_inside_tolerance():
    verifier = ObservedPredicateVerifier(
        FakeProvider({"cube1": (0.23, -0.05, 0.83)})
    )
    assert verifier.check_at("cube1", (0.22, -0.06, 0.83))


def test_at_outside_tolerance():
    verifier = ObservedPredicateVerifier(
        FakeProvider({"cube1": (0.40, -0.05, 0.83)})
    )
    assert not verifier.check_at("cube1", (0.22, -0.06, 0.83))


def test_holding_requires_live_held_identity_and_height():
    verifier = ObservedPredicateVerifier(
        FakeProvider({"cube1": (0.1, 0.1, 0.95)}, held="cube1")
    )
    assert verifier.check_holding("cube1")
    assert not verifier.check_handempty()


def test_stability_rejects_tilted_or_moving_cube():
    poses = {
        "cube1": (0.22, -0.06, 0.83),
        "cube2": (0.22, -0.06, 0.89),
    }
    still = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    provider = DynamicProvider(
        poses,
        {
            "cube1": (1.0, 0.0, 0.0, 0.0),
            "cube2": (0.9239, 0.3827, 0.0, 0.0),
        },
        {"cube1": still, "cube2": still},
    )
    verifier = ObservedPredicateVerifier(provider)
    assert not verifier.check_stable("cube2")

    provider.orientations["cube2"] = (0.7071, 0.7071, 0.0, 0.0)
    assert verifier.check_stable("cube2")

    provider.orientations["cube2"] = (1.0, 0.0, 0.0, 0.0)
    provider.velocities["cube2"] = ((0.05, 0.0, 0.0), (0.0, 0.0, 0.0))
    verifier = ObservedPredicateVerifier(provider)
    assert not verifier.check_stable("cube2", include_velocity=True)
