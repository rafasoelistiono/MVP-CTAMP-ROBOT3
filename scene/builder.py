from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Protocol


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
GENERATED_DIR = MODELS_DIR / "generated"

SCENE_ALIASES = {
    "group_no_obs": "group_no_obs",
    "group-no-obs": "group_no_obs",
    "group no obs": "group_no_obs",
    "group": "group_no_obs",
    "ungroup_no_obs": "ungroup_no_obs",
    "ungroup-no-obs": "ungroup_no_obs",
    "ungroup no obs": "ungroup_no_obs",
    "ungroup": "ungroup_no_obs",
    "group_obs": "group_obs",
    "group-obs": "group_obs",
    "group obs": "group_obs",
    "group_long_obs": "group_long_obs",
    "group-long-obs": "group_long_obs",
    "group long obs": "group_long_obs",
    "group_long": "group_long_obs",
    "group long": "group_long_obs",
    "ungroup_obs": "ungroup_obs",
    "ungroup-obs": "ungroup_obs",
    "ungroup obs": "ungroup_obs",
    "ungroup_long_obs": "ungroup_long_obs",
    "ungroup-long-obs": "ungroup_long_obs",
    "ungroup long obs": "ungroup_long_obs",
    "ungroup_long": "ungroup_long_obs",
    "ungroup long": "ungroup_long_obs",
}

GOAL_CENTER = (0.22, -0.06, 0.806)
GOAL_HALF_SIZE_X = 0.26
GOAL_HALF_SIZE_Y = 0.20
GOAL_EXCLUSION_MARGIN = 0.04
COMPACT_CYLINDER_CENTER_Z = 0.84
COMPACT_CYLINDER_SIZE = (0.026, 0.04)
# Every generated world uses the same true 6.6 cm cube geometry. It gives the
# gripper and each stack layer a 21% larger contact area than the old 6 cm
# cube while remaining below the Panda gripper's maximum opening.
CUBE_HALF_EXTENTS = (0.033, 0.033, 0.033)
CUBE_CENTER_Z = 0.833
DEFAULT_OBSTACLE_HALF_HEIGHT = 0.085
LONG_OBSTACLE_HALF_HEIGHT = 0.32

VARIANT_OBJECTS = {
    "group_no_obs": {
        "cube1": (-0.02, -0.46, CUBE_CENTER_Z),
        "cube2": (0.10, -0.46, CUBE_CENTER_Z),
        "cube3": (0.22, -0.40, CUBE_CENTER_Z),
        "cube4": (0.32, -0.34, CUBE_CENTER_Z),
        "circle1": (-0.02, 0.24, COMPACT_CYLINDER_CENTER_Z),
        "circle2": (0.10, 0.32, COMPACT_CYLINDER_CENTER_Z),
        "circle3": (0.22, 0.26, COMPACT_CYLINDER_CENTER_Z),
        "circle4": (0.32, 0.34, COMPACT_CYLINDER_CENTER_Z),
    },
    "ungroup_no_obs": {
        "cube1": (-0.02, -0.46, CUBE_CENTER_Z),
        "circle1": (0.12, -0.42, COMPACT_CYLINDER_CENTER_Z),
        "cube2": (0.28, -0.40, CUBE_CENTER_Z),
        "circle2": (0.34, -0.32, COMPACT_CYLINDER_CENTER_Z),
        "cube3": (0.00, 0.24, CUBE_CENTER_Z),
        "circle3": (0.16, 0.32, COMPACT_CYLINDER_CENTER_Z),
        "cube4": (0.30, 0.24, CUBE_CENTER_Z),
        "circle4": (0.34, 0.34, COMPACT_CYLINDER_CENTER_Z),
    },
    "group_obs": {
        "cube1": (-0.02, -0.46, CUBE_CENTER_Z),
        "cube2": (0.10, -0.52, CUBE_CENTER_Z),
        "cube3": (0.22, -0.48, CUBE_CENTER_Z),
        "cube4": (0.32, -0.38, CUBE_CENTER_Z),
        "circle1": (-0.02, 0.24, COMPACT_CYLINDER_CENTER_Z),
        "circle2": (0.10, 0.32, COMPACT_CYLINDER_CENTER_Z),
        "circle3": (0.21, 0.20, COMPACT_CYLINDER_CENTER_Z),
        "circle4": (0.22, 0.42, COMPACT_CYLINDER_CENTER_Z),
    },
    "ungroup_obs": {
        # Keep the semantic ungrouped layout: cubes and cylinders remain
        # irregularly interleaved on both sides of the goal.  The cube poses
        # are inside the Panda's reliable annulus and outside the cautious
        # obstacle band so grasping does not start from a near-singular reach.
        "cube1": (-0.16, -0.42, CUBE_CENTER_Z),
        "circle1": (0.00, -0.54, COMPACT_CYLINDER_CENTER_Z),
        "cube2": (0.10, -0.54, CUBE_CENTER_Z),
        "circle2": (0.28, -0.48, COMPACT_CYLINDER_CENTER_Z),
        "cube3": (-0.10, 0.28, CUBE_CENTER_Z),
        "circle3": (0.06, 0.40, COMPACT_CYLINDER_CENTER_Z),
        "cube4": (0.12, 0.20, CUBE_CENTER_Z),
        "circle4": (0.28, 0.42, COMPACT_CYLINDER_CENTER_Z),
    },
}

VARIANT_OBJECTS["group_long_obs"] = dict(VARIANT_OBJECTS["group_obs"])
VARIANT_OBJECTS["ungroup_long_obs"] = dict(VARIANT_OBJECTS["ungroup_obs"])

OBSTACLE_POSITIONS = {
    # Refactor 3 obstacle mapping: obstacle1 and obstacle2 intentionally sit on
    # different sides of the object clusters. obstacle2 stays on the far end and
    # is moved slightly backward; obstacle1 challenges the front cube/cylinder
    # approach without entering the hard TOO_CLOSE band.
    "obstacle1": (0.11, -0.30, 0.89),
    "obstacle2": (0.350, 0.27, 0.89),
}


class _SceneObject(Protocol):
    id: str
    cls: str
    pose: tuple[float, float, float]
    rgba: tuple[float, float, float, float] | None


class _SceneObstacle(Protocol):
    id: str
    pose: tuple[float, float, float]
    height: str


def normalize_scene_key(raw: str | Iterable[str] | None) -> str:
    if raw is None:
        return "group_no_obs"
    if isinstance(raw, str):
        key = raw
    else:
        key = " ".join(raw)
    key = " ".join(key.strip().lower().replace("-", " ").replace("_", " ").split())
    normalized = SCENE_ALIASES.get(key) or SCENE_ALIASES.get(key.replace(" ", "_"))
    if normalized is None:
        valid = ", ".join(sorted({
            "group no obs",
            "ungroup no obs",
            "group obs",
            "ungroup obs",
            "group long obs",
            "ungroup long obs",
        }))
        raise ValueError(f"unknown --object '{key}'. Valid: {valid}")
    return normalized


def obstacle_mode_for_scene(scene_key: str) -> str:
    if scene_key.endswith("_no_obs"):
        return "no_obs"
    if scene_key.endswith("_obs"):
        return "obs"
    return "unknown"


def prepare_scene_variant(
    raw: str | Iterable[str] | None,
    *,
    base_model_file: str | Path | None = None,
    object_states: Iterable[_SceneObject] | None = None,
    obstacle_states: Iterable[_SceneObstacle] | None = None,
) -> Path:
    scene_key = normalize_scene_key(raw)
    _validate_variant(scene_key)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_model_file) if base_model_file else MODELS_DIR / "panda.xml"
    if not base_path.is_absolute():
        base_path = ROOT_DIR / base_path
    out_path = GENERATED_DIR / f"{base_path.stem}_{scene_key}.xml"

    tree = ET.parse(base_path)
    root = tree.getroot()
    _rebase_compiler_directories(
        root,
        source_directory=base_path.parent,
        output_directory=out_path.parent,
    )
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("models/panda.xml has no worldbody")

    removable_prefixes = ("cube", "circle", "obstacle", "vase", "glass", "ceramic")
    for body in list(worldbody.findall("body")):
        name = body.get("name", "")
        if name == "goal_area" or name.startswith(removable_prefixes):
            worldbody.remove(body)

    link0_index = 0
    for idx, body in enumerate(list(worldbody)):
        if body.tag == "body" and body.get("name") == "link0":
            link0_index = idx
            break

    inserts = [_goal_area_body()]
    if object_states is None:
        for object_name, pos in VARIANT_OBJECTS[scene_key].items():
            inserts.append(_movable_body(object_name, pos))
    else:
        for state in object_states:
            inserts.append(
                _movable_body(
                    state.id,
                    state.pose,
                    cls=state.cls,
                    rgba_override=getattr(state, "rgba", None),
                )
            )

    if obstacle_mode_for_scene(scene_key) == "obs":
        fixed = scene_key.endswith("_long_obs")
        if obstacle_states is None:
            half_height = _obstacle_half_height_for_scene(scene_key)
            for obstacle_name, pos in _obstacle_positions_for_scene(scene_key).items():
                inserts.append(_obstacle_body(obstacle_name, pos, half_height=half_height, fixed=fixed))
        else:
            for state in obstacle_states:
                half_height = (
                    LONG_OBSTACLE_HALF_HEIGHT
                    if state.height == "long"
                    else DEFAULT_OBSTACLE_HALF_HEIGHT
                )
                inserts.append(
                    _obstacle_body(state.id, state.pose, half_height=half_height, fixed=fixed)
                )

    for offset, body in enumerate(inserts):
        worldbody.insert(link0_index + offset, body)

    _indent(root)
    tmp_path = out_path.with_suffix(f".{os.getpid()}.tmp")
    tree.write(tmp_path, encoding="utf-8", xml_declaration=False)
    tmp_path.replace(out_path)
    return out_path


def _rebase_compiler_directories(
    root: ET.Element,
    *,
    source_directory: Path,
    output_directory: Path,
) -> None:
    """Preserve compiler asset paths when a scene XML is written elsewhere."""
    compiler = root.find("compiler")
    if compiler is None:
        return
    for attribute in ("assetdir", "meshdir", "texturedir"):
        raw_value = compiler.get(attribute)
        if not raw_value:
            continue
        source_path = Path(raw_value)
        if not source_path.is_absolute():
            source_path = (source_directory / source_path).resolve()
        relative_path = os.path.relpath(source_path, output_directory.resolve())
        compiler.set(attribute, Path(relative_path).as_posix())


def _obstacle_half_height_for_scene(scene_key: str) -> float:
    return LONG_OBSTACLE_HALF_HEIGHT if scene_key.endswith("_long_obs") else DEFAULT_OBSTACLE_HALF_HEIGHT


def _obstacle_positions_for_scene(scene_key: str) -> dict[str, tuple[float, float, float]]:
    if not scene_key.endswith("_long_obs"):
        return dict(OBSTACLE_POSITIONS)
    half_height = _obstacle_half_height_for_scene(scene_key)
    center_z = GOAL_CENTER[2] + half_height
    return {
        name: (float(pos[0]), float(pos[1]), center_z)
        for name, pos in OBSTACLE_POSITIONS.items()
    }


def _validate_variant(scene_key: str) -> None:
    for name, pos in VARIANT_OBJECTS[scene_key].items():
        if _inside_goal_area(pos[0], pos[1]):
            raise RuntimeError(
                f"{scene_key}: initial object {name} is inside goal area at {(pos[0], pos[1])}"
            )


def _inside_goal_area(x: float, y: float) -> bool:
    gx, gy, _ = GOAL_CENTER
    return (
        gx - GOAL_HALF_SIZE_X - GOAL_EXCLUSION_MARGIN
        <= x
        <= gx + GOAL_HALF_SIZE_X + GOAL_EXCLUSION_MARGIN
        and gy - GOAL_HALF_SIZE_Y - GOAL_EXCLUSION_MARGIN
        <= y
        <= gy + GOAL_HALF_SIZE_Y + GOAL_EXCLUSION_MARGIN
    )


def _goal_area_body() -> ET.Element:
    return ET.fromstring(
        """
        <body name="goal_area" pos="0.22 -0.06 0.806">
          <geom name="goal_area_base" type="box" size="0.26 0.20 0.003" rgba="0.05 0.35 0.95 0.22" contype="0" conaffinity="0"/>
          <geom name="goal_square_row" type="box" size="0.24 0.025 0.004" pos="0 -0.065 0.004" rgba="0.95 0.25 0.15 0.35" contype="0" conaffinity="0"/>
          <geom name="goal_circle_row" type="box" size="0.24 0.025 0.004" pos="0 0.065 0.004" rgba="0.1 0.8 0.45 0.35" contype="0" conaffinity="0"/>
        </body>
        """
    )


def _movable_body(
    name: str,
    pos: tuple[float, float, float],
    *,
    cls: str | None = None,
    rgba_override: tuple[float, float, float, float] | None = None,
) -> ET.Element:
    rgba = " ".join(str(value) for value in rgba_override) if rgba_override else {
        "cube1": "1 0 0 1",
        "cube2": "0 1 0 1",
        "cube3": "0 0 1 1",
        "cube4": "1 1 0 1",
        "cube5": "1 0.4 0 1",
        "cube6": "0.6 0 1 1",
        "cube7": "0 0.65 1 1",
        "cube8": "0.9 0.2 0.6 1",
        "cube9": "0.55 0.35 0.1 1",
        "cube10": "0.8 0.8 0.8 1",
        "circle1": "0.0 0.9 0.9 1",
        "circle2": "0.1 0.7 1.0 1",
        "circle3": "0.2 1.0 0.45 1",
        "circle4": "0.3 0.9 0.2 1",
    }.get(name, "1 1 1 1")
    object_class = cls or ("cube" if name.startswith("cube") else "cylinder")
    if object_class == "cube":
        half_x, half_y, half_z = CUBE_HALF_EXTENTS
        geom = f'<geom type="box" size="{half_x} {half_y} {half_z}" mass="0.1" friction="2 1 0.5" contype="1" conaffinity="1" rgba="{rgba}"/>'
    else:
        radius, half_height = COMPACT_CYLINDER_SIZE
        geom = f'<geom type="cylinder" size="{radius} {half_height}" mass="0.08" friction="3 1.5 0.8" contype="1" conaffinity="1" rgba="{rgba}"/>'
    return ET.fromstring(
        f"""
        <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
          <joint type="free"/>
          {geom}
        </body>
        """
    )


def _obstacle_body(
    name: str,
    pos: tuple[float, float, float],
    half_height: float = DEFAULT_OBSTACLE_HALF_HEIGHT,
    fixed: bool = False,
) -> ET.Element:
    joint = "" if fixed else '<joint type="free"/>'
    return ET.fromstring(
        f"""
        <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
          {joint}
          <geom type="cylinder" size="0.035 {half_height}" mass="0.4" friction="2 1 0.5" rgba="0.9 0.75 0.2 0.75" contype="1" conaffinity="1"/>
        </body>
        """
    )


def _indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
