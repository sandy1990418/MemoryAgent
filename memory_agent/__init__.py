"""The canonical, framework-free chat-memory API.

Optional framework and evaluation integrations intentionally live below
``memory_agent.adapters`` and ``evaluation``.  Keeping the package root to
three names makes the supported product surface unambiguous and prevents
optional dependencies from leaking into a plain chat import.
"""

from __future__ import annotations

_EXPORTS = {
    "build_chat_memory": ("memory_agent.application.chat", "build_chat_memory"),
    "ChatMemory": ("memory_agent.application.chat", "ChatMemory"),
    "Turn": ("memory_agent.core.transcript", "Turn"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
