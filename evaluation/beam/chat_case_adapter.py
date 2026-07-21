"""BEAM JSON to public chat-turn conversion.

BEAM metadata remains an evaluation concern and is never represented as a
production ``MemoryEvent``.  The adapter emits the same ``Turn`` objects used
by :func:`memory_agent.build_chat_memory`; callers can keep dataset metadata
alongside the returned turns when they need it for reports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from memory_agent.core.transcript import Turn


class BeamChatCaseAdapter:
    """Normalize one BEAM message batch to public chat turns."""

    def adapt_messages(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        case_id: str,
        session_id: str | None = None,
    ) -> list[Turn]:
        del session_id  # retained only as a report-level caller hint
        turns: list[Turn] = []
        for index, message in enumerate(messages, start=1):
            role = str(message.get("role", "unknown"))
            if role not in {"user", "assistant"}:
                continue
            # Turn ids are local to this adapted batch.  BEAM's opaque ids are
            # retained by metadata at the evaluation edge, not persisted in
            # production memory state.
            turns.append(
                Turn(
                    id=index,
                    role=role,
                    content=str(message.get("content", "")),
                )
            )
        return turns
