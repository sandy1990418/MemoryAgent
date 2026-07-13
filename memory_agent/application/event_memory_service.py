"""Event-memory application boundary shared by chat and agent adapters."""

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

from memory_agent.domain import MemoryEntry, MemoryEvent, MemoryStatus, ProvenanceRef
from memory_agent.policies.event import EventMemoryCandidate, EventMemoryPolicy


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class EventMemoryService:
    """In-memory application boundary for generic chat and agent events.

    This service is intentionally event-facing. It is not a second structured
    memory core and currently acts as the evolution boundary for agent-event
    ingestion.
    """

    def __init__(self, policy: EventMemoryPolicy) -> None:
        self.policy = policy
        self.entries: dict[str, MemoryEntry] = {}
        self._next_id = 1
        self._processed_event_ids: set[str] = set()
        self._lock = RLock()

    def ingest_events(self, events: Iterable[MemoryEvent]) -> list[MemoryEntry]:
        stored = []
        with self._lock:
            for event in events:
                # Agent traces are commonly replayed or delivered at least
                # once. Stable event ids make ingestion idempotent.
                if event.event_id in self._processed_event_ids:
                    continue
                content = str(event.metadata.get("durable_summary") or event.content)
                candidate = EventMemoryCandidate(
                    event=event,
                    content=content,
                    subject=str(
                        event.metadata.get("subject")
                        or event.task_id
                        or event.actor
                    ),
                )
                if not self.policy.should_store(candidate):
                    self._processed_event_ids.add(event.event_id)
                    continue
                memory_type = self.policy.classify(candidate)
                subject = candidate.subject
                for prior in self.entries.values():
                    if (
                        prior.status == MemoryStatus.ACTIVE
                        and prior.subject == subject
                        and prior.memory_type == memory_type
                    ):
                        prior.status = MemoryStatus.SUPERSEDED
                entry = MemoryEntry(
                    memory_id=f"M{self._next_id}",
                    memory_type=memory_type,
                    scope=self.policy.scope_for(candidate),
                    subject=subject,
                    content=content,
                    provenance=[
                        ProvenanceRef(event.event_id, event.source_type.value)
                    ],
                    importance=self.policy.importance(candidate),
                    metadata={
                        "task_id": event.task_id,
                        "session_id": event.session_id,
                    },
                )
                self._next_id += 1
                self.entries[entry.memory_id] = entry
                self._processed_event_ids.add(event.event_id)
                stored.append(entry)
        return stored

    def retrieve_memory(self, *, max_tokens: int, scope=None) -> list[MemoryEntry]:
        with self._lock:
            candidates = [
                entry
                for entry in self.entries.values()
                if entry.status == MemoryStatus.ACTIVE
                and (scope is None or entry.scope == scope)
            ]
            candidates.sort(key=self.policy.retention_priority, reverse=True)
            selected, used = [], 0
            for entry in candidates:
                cost = _estimate_tokens(entry.content)
                if used + cost <= max_tokens:
                    selected.append(entry)
                    used += cost
            return selected

    def build_context(self, *, max_tokens: int, scope=None) -> str:
        return "\n".join(f"- [{entry.memory_type.value}] {entry.content}" for entry in self.retrieve_memory(max_tokens=max_tokens, scope=scope))
