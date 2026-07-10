from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memory_agent.domain import MemoryEntry, MemoryEvent, MemoryStatus


@dataclass(frozen=True)
class AgentEvaluationResult:
    critical_state_recall: float
    retained_entries: int


class AgentMemoryEvaluator(Protocol):
    def evaluate(self, events: list[MemoryEvent], entries: list[MemoryEntry]) -> AgentEvaluationResult: ...


class CriticalStateEvaluator:
    """Minimal extension point; not a complete agent benchmark."""

    def evaluate(self, events: list[MemoryEvent], entries: list[MemoryEntry]) -> AgentEvaluationResult:
        expected = [event for event in events if event.metadata.get("critical")]
        active = [entry for entry in entries if entry.status == MemoryStatus.ACTIVE]
        recalled = sum(any(event.event_id == ref.event_id for entry in active for ref in entry.provenance) for event in expected)
        return AgentEvaluationResult(recalled / len(expected) if expected else 1.0, len(active))
