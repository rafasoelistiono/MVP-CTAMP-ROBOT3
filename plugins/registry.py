from __future__ import annotations

from plugins.align_task import PLUGIN as ALIGN_PLUGIN
from plugins.protocol import TaskPlugin
from plugins.pyramid_task import PLUGIN as PYRAMID_PLUGIN
from plugins.stack_task import PLUGIN as STACK_PLUGIN


TASK_PLUGIN_API_VERSION = "ctamp-task/v2"


class PluginRegistry:
    def __init__(self):
        self._plugins: dict[str, TaskPlugin] = {}

    def register(self, plugin: TaskPlugin) -> None:
        if getattr(plugin, "api_version", None) != TASK_PLUGIN_API_VERSION:
            raise ValueError(
                f"task plugin {getattr(plugin, 'name', '<unknown>')!r} uses "
                f"unsupported API {getattr(plugin, 'api_version', None)!r}; "
                f"expected {TASK_PLUGIN_API_VERSION!r}"
            )
        if not getattr(plugin, "name", ""):
            raise ValueError("task plugin name must not be empty")
        required_methods = (
            "validate_plan",
            "make_slot_config",
            "configure_runtime",
            "assess_progress",
            "verify_goal",
        )
        missing = [name for name in required_methods if not callable(getattr(plugin, name, None))]
        if missing:
            raise ValueError(
                f"task plugin {plugin.name!r} is missing methods: {', '.join(missing)}"
            )
        if plugin.name in self._plugins:
            raise ValueError(f"task plugin {plugin.name!r} is already registered")
        self._plugins[plugin.name] = plugin

    def get(self, task_name: str) -> TaskPlugin:
        if task_name not in self._plugins:
            raise ValueError(
                f"Task {task_name!r} tidak terdaftar. "
                f"Tersedia: {sorted(self._plugins)}. "
                "Task yang didukung: align, pyramid, stack."
            )
        return self._plugins[task_name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))


def _default_registry() -> PluginRegistry:
    registry = PluginRegistry()
    for plugin in (ALIGN_PLUGIN, PYRAMID_PLUGIN, STACK_PLUGIN):
        registry.register(plugin)
    return registry


DEFAULT_REGISTRY = _default_registry()
