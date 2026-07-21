"""Structural recency/budget selection for answer-memory context."""

from __future__ import annotations

from typing import Callable

from memory_agent.core.models import MemoryEntry, SelectedMemory
from memory_agent.core.store import Memory
from memory_agent.policies.structured import StructuredMemoryPolicy


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


class MemorySelector:
    """Select memory by status/recency and hard token budget only.

    ``query`` is accepted at the API edge for callers that already provide it,
    but it never changes selection. Semantic ranking belongs to an explicit
    retrieval provider, not this bounded in-process fallback.
    """

    def __init__(
        self,
        token_estimator: Callable[[str], int] | None = None,
        policy: StructuredMemoryPolicy | None = None,
    ) -> None:
        self.token_estimator = token_estimator or _default_token_estimator
        self.policy = policy

    def select(
        self,
        memory: Memory,
        query: str = "",
        max_tokens: int | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        return [item.entry for item in self.select_with_scores(
            memory=memory,
            query=query,
            max_tokens=max_tokens,
            include_superseded=include_superseded,
        )]

    def select_for_answer(
        self,
        memory: Memory,
        query: str = "",
        budget: int | None = None,
    ) -> list[MemoryEntry]:
        return self.select(memory=memory, query=query, max_tokens=budget)

    def select_with_scores(
        self,
        memory: Memory,
        query: str = "",
        max_tokens: int | None = None,
        include_superseded: bool = False,
    ) -> list[SelectedMemory]:
        entries = [
            entry for entry in memory.entries.values()
            if include_superseded or entry.status == "active"
        ]
        entries.sort(key=lambda entry: (
            entry.status != "active",
            -(max(entry.provenance) if entry.provenance else -1),
            entry.id,
        ))
        selected: list[SelectedMemory] = []
        selected_entries: list[MemoryEntry] = []
        for entry in entries:
            candidate_entries = [*selected_entries, entry]
            if max_tokens is not None:
                rendered = memory.render(entries=candidate_entries)
                if self.token_estimator(rendered) > max_tokens:
                    continue
            recency = min(max(entry.provenance), 1000) / 1000.0 if entry.provenance else 0.0
            selected.append(SelectedMemory(
                entry=entry,
                score=(1.0 if entry.status == "active" else 0.0) + recency,
                reasons=(
                    ("active",) if entry.status == "active" else ("superseded",)
                ) + (("recency",) if entry.provenance else ()),
            ))
            selected_entries.append(entry)
        return selected
