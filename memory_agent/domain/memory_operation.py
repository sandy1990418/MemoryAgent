from dataclasses import dataclass
from enum import Enum

from .memory_entry import MemoryEntry


class MemoryOperationType(str, Enum):
    ADD = "add"
    UPDATE = "update"
    SUPERSEDE = "supersede"
    NOOP = "noop"


@dataclass(frozen=True)
class MemoryOperation:
    operation: MemoryOperationType
    entry: MemoryEntry | None = None
    target_memory_id: str | None = None
    reason: str = ""
