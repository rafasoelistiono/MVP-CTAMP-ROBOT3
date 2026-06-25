from __future__ import annotations

import pytest

from plugins.registry import DEFAULT_REGISTRY, PluginRegistry
from configuration import load_runtime_config


def test_default_registry_contains_align_stack_and_pyramid():
    assert DEFAULT_REGISTRY.names() == ("align", "pyramid", "stack")
    assert DEFAULT_REGISTRY.get("align").name == "align"
    assert DEFAULT_REGISTRY.get("pyramid").name == "pyramid"
    assert DEFAULT_REGISTRY.get("stack").name == "stack"


def test_unknown_task_error_is_descriptive():
    with pytest.raises(ValueError, match="tidak terdaftar"):
        DEFAULT_REGISTRY.get("teleport")


def test_incompatible_plugin_api_is_rejected():
    class BadPlugin:
        api_version = "ctamp-task/v999"
        name = "bad"
        supported_actions = set()

    with pytest.raises(ValueError, match="unsupported API"):
        PluginRegistry().register(BadPlugin())


def test_stack_plugin_owns_stack_specific_ready_pose():
    plugin = DEFAULT_REGISTRY.get("stack")
    config = load_runtime_config("conservative")
    tuned = plugin.configure_runtime(None, None, config)
    assert tuned.model.grasp_ready_q == tuned.model.home_q
    assert config.model.grasp_ready_q != config.model.home_q
