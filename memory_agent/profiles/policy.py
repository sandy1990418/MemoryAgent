from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from memory_agent.domain import MemoryEntry, MemoryEvent, MemoryType


@dataclass(frozen=True)
class MemoryCandidate:
    event: MemoryEvent
    content: str
    suggested_type: MemoryType | None = None
    subject: str = ""


@runtime_checkable
class MemoryPolicy(Protocol):
    name: str

    def should_store(self, candidate: MemoryCandidate) -> bool: ...
    def classify(self, candidate: MemoryCandidate) -> MemoryType: ...
    def importance(self, candidate: MemoryCandidate) -> float: ...
    def retention_priority(self, entry: MemoryEntry) -> float: ...
