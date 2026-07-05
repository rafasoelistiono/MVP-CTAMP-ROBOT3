from __future__ import annotations

import json

from configuration import load_runtime_config
from telemetry import sha256_file, write_run_manifest


def test_run_manifest_records_full_resolved_config_and_hashes(tmp_path):
    plan = tmp_path / "plan.json"
    context = tmp_path / "context.md"
    plan.write_text('{"plan": true}\n', encoding="utf-8")
    context.write_text("# context\n", encoding="utf-8")
    config = load_runtime_config("obstacle", enable_viewer=False)
    output = write_run_manifest(
        tmp_path / "manifest.json",
        run_id="run-1",
        config=config,
        plan_file=plan,
        context_file=context,
        scene_id="scene-1",
        scene_variant="group_obs",
        task="align",
        plugin_package="plugins",
        plan_source="original_no_llm",
        benchmark_role="reference",
        benchmark_label="original-v1",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "ctamp-run-manifest/v1"
    assert payload["plan"]["sha256"] == sha256_file(plan)
    assert payload["runtime_config"]["motion"]["time_limit_s"] == 8.0
    assert payload["runtime_config"]["model"]["xml_path"].endswith("panda.xml")
    assert payload["benchmark"] == {
        "plan_source": "original_no_llm",
        "role": "reference",
        "label": "original-v1",
    }
