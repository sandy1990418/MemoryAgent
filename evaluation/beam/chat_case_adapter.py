from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from memory_agent.domain import MemoryEvent
from memory_agent.adapters.events.chat import ChatEventAdapter


class BeamChatCaseAdapter:
    """Keeps BEAM schema and metadata outside the production domain."""

    def __init__(self) -> None:
        self._chat = ChatEventAdapter()

    def adapt_messages(self, messages: Iterable[Mapping[str, Any]], *, case_id: str, session_id: str | None = None) -> list[MemoryEvent]:
        normalized = []
        for message in messages:
            normalized.append({
                "id": f"beam-{case_id}-{message.get('id', len(normalized) + 1)}",
                "role": message.get("role"),
                "content": message.get("content", ""),
                "metadata": {"dataset": "BEAM", "case_id": case_id, "beam_chat_id": message.get("id"), "beam_index": message.get("index")},
            })
        return self._chat.adapt(normalized, session_id=session_id or f"beam-{case_id}")
