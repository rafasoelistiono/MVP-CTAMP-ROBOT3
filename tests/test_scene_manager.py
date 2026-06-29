from __future__ import annotations

import csv
import xml.etree.ElementTree as ET

import pytest

from scene import (
    CUBE_HALF_EXTENTS,
    LONG_OBSTACLE_HALF_HEIGHT,
    normalize_scene_key,
    prepare_scene_variant,
)
from telemetry import write_summary_csv


def test_scene_alias_normalization():
    assert normalize_scene_key("group no obs") == "group_no_obs"
    assert normalize_scene_key(["ungroup", "long", "obs"]) == "ungroup_long_obs"
    with pytest.raises(ValueError, match="unknown"):
        normalize_scene_key("teleport")


def test_generated_scene_is_written_to_generated_directory():
    path = prepare_scene_variant("group_no_obs")
    assert path.parent.name == "generated"
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    mesh_directory = (path.parent / compiler.get("meshdir")).resolve()
    assert mesh_directory == (path.parents[2] / "assets").resolve()
    names = {body.get("name") for body in root.findall(".//body")}
    assert {"cube1", "cube2", "cube3", "cube4"} <= names
    assert "goal_area" in names


def test_all_generated_cubes_have_the_same_cubic_size():
    path = prepare_scene_variant("ungroup_obs")
    root = ET.parse(path).getroot()
    sizes = []
    for index in range(1, 5):
        geom = root.find(f".//body[@name='cube{index}']/geom")
        assert geom is not None
        sizes.append(tuple(float(value) for value in geom.get("size").split()))

    assert CUBE_HALF_EXTENTS == (0.033, 0.033, 0.033)
    assert sizes == [CUBE_HALF_EXTENTS] * 4


def test_long_obstacles_are_fixed_and_use_configured_height():
    path = prepare_scene_variant("group_long_obs")
    root = ET.parse(path).getroot()
    obstacle = root.find(".//body[@name='obstacle1']")
    assert obstacle is not None
    assert obstacle.find("joint") is None
    geom = obstacle.find("geom")
    assert geom is not None
    assert float(geom.get("size").split()[1]) == pytest.approx(
        LONG_OBSTACLE_HALF_HEIGHT
    )


def test_summary_records_runtime_provenance(tmp_path):
    path = write_summary_csv(
        "stack",
        "group_no_obs",
        {
            "success": True,
            "objects_moved": 1,
            "objects_total": 1,
            "failed": [],
            "runtime_profile": "conservative",
            "run_manifest": "manifest.json",
            "plan_source": "original_no_llm",
            "benchmark_role": "reference",
            "benchmark_label": "original-v1",
            "experiment_label": "Qwen 3 Coder",
            "run_id": "20260622_120000",
        },
        log_dir=tmp_path,
    )
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["runtime_profile"] == "conservative"
    assert row["run_manifest"] == "manifest.json"
    assert row["completion_percent"] == "100.0"
    assert row["plan_source"] == "original_no_llm"
    assert row["benchmark_role"] == "reference"
    assert row["benchmark_label"] == "original-v1"
    assert row["reference_100_percent"] == "true"
    assert row["run_id"] == "20260622_120000_qwen_3_coder"
    assert row["experiment_label"] == "qwen_3_coder"
    assert path.name == "stack_group_no_obs_20260622_120000_qwen_3_coder.csv"
