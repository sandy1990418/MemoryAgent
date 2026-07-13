"""Rendering for an already-selected answer-memory set."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from memory_agent.core.models import MemoryEntry
from memory_agent.core.store import Memory


@dataclass(frozen=True)
class AnswerContext:
    selected_ids: tuple[str, ...]
    rendered_context: str


def build_answer_memory_context(
    *,
    memory: Memory,
    entries: Sequence[MemoryEntry],
) -> AnswerContext:
    """Render entries selected by the caller's retrieval policy."""
    return AnswerContext(
        selected_ids=tuple(entry.id for entry in entries),
        rendered_context=memory.render(entries=list(entries)),
    )
