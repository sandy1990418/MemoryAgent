from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from memory_agent.domain import EventSourceType, MemoryEvent


class AgentTraceAdapter(Protocol):
    def adapt(self, records: Iterable[Mapping[str, Any]], *, task_id: str | None = None) -> list[MemoryEvent]: ...


class AgentEventAdapter:
    def adapt(self, records: Iterable[Mapping[str, Any]], *, task_id: str | None = None) -> list[MemoryEvent]:
        events = []
        for index, record in enumerate(records):
            source = EventSourceType(str(record["source_type"]))
            events.append(MemoryEvent(
                event_id=str(record.get("event_id") or f"agent-{index + 1}"),
                source_type=source,
                actor=str(record.get("actor") or "agent"),
                content=str(record.get("content", "")),
                task_id=task_id or record.get("task_id"),
                metadata=dict(record.get("metadata") or {}),
            ))
        return events
