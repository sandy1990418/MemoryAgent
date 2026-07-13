"""Adapt chat messages into generic memory events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from memory_agent.domain import EventSourceType, MemoryEvent


class ChatEventAdapter:
    """Maps chat-provider messages at the boundary; core sees only events."""

    def adapt(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        session_id: str | None = None,
    ) -> list[MemoryEvent]:
        events = []
        for index, message in enumerate(messages):
            role = str(message.get("role", ""))
            if role not in {"user", "assistant", "system"}:
                raise ValueError(f"unsupported chat role: {role}")
            metadata = dict(message.get("metadata") or {})
            metadata["chat_role"] = role
            events.append(MemoryEvent(
                event_id=str(message.get("id") or f"chat-{index + 1}"),
                source_type=EventSourceType.CHAT_MESSAGE,
                actor=role,
                content=str(message.get("content", "")),
                session_id=session_id,
                metadata=metadata,
            ))
        return events
