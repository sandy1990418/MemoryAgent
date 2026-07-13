"""Policy contract for generic event-memory ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from memory_agent.domain import MemoryEntry, MemoryEvent, MemoryScope, MemoryType


@dataclass(frozen=True)
class EventMemoryCandidate:
    """A generic event prepared for workload-specific retention decisions."""

    event: MemoryEvent
    content: str
    suggested_type: MemoryType | None = None
    subject: str = ""


@runtime_checkable
class EventMemoryPolicy(Protocol):
    """Policy contract for the future event-memory ingestion boundary."""

    name: str

    def should_store(self, candidate: EventMemoryCandidate) -> bool: ...
    def classify(self, candidate: EventMemoryCandidate) -> MemoryType: ...
    def importance(self, candidate: EventMemoryCandidate) -> float: ...
    def retention_priority(self, entry: MemoryEntry) -> float: ...
    def scope_for(self, candidate: EventMemoryCandidate) -> MemoryScope: ...
