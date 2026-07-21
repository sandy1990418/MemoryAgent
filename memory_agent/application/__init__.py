from .structured_service import StructuredMemoryService, StructuredUpdateResult

__all__ = [
    "ChatMemory",
    "build_chat_memory",
    "StructuredMemoryService",
    "StructuredUpdateResult",
]


def __getattr__(name: str):
    if name in {"ChatMemory", "build_chat_memory"}:
        from .chat import ChatMemory, build_chat_memory

        value = {"ChatMemory": ChatMemory, "build_chat_memory": build_chat_memory}[name]
        globals()[name] = value
        return value
    raise AttributeError(name)
