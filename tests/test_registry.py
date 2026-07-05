from __future__ import annotations

import pytest

from plugins.registry import DEFAULT_REGISTRY, PluginRegistry


def test_default_registry_contains_align_only():
    assert DEFAULT_REGISTRY.names() == ("align",)
    assert DEFAULT_REGISTRY.get("align").name == "align"


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
