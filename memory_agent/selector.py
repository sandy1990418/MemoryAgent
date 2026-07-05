"""Session-local memory selection for prompt injection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from memory_agent.memory import Memory, MemoryEntry

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


@dataclass(frozen=True)
class SelectedMemory:
    entry: MemoryEntry
    score: float
    reasons: tuple[str, ...]


class MemorySelector:
    """Select active memory entries for the current in-session prompt.

    This is intentionally deterministic. It is not semantic search and does not
    need embeddings. The goal is to stop blindly rendering every active memory
    entry once a single session grows large.
    """

    DEFAULT_SECTION_PRIORITIES: dict[str, float] = {
        "preferences": 100.0,
        "goal": 95.0,
        "progress": 90.0,
        "open_questions": 85.0,
        "decisions": 80.0,
        "tool_facts": 70.0,
        "facts": 60.0,
        "failed_attempts": 35.0,
    }

    def __init__(
        self,
        section_priorities: dict[str, float] | None = None,
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        self.section_priorities = dict(self.DEFAULT_SECTION_PRIORITIES)
        if section_priorities:
            self.section_priorities.update(section_priorities)
        self.token_estimator = token_estimator or _default_token_estimator

    def select(
        self,
        memory: Memory,
        query: str = "",
        max_tokens: int | None = None,
    ) -> list[MemoryEntry]:
        selected = self.select_with_scores(memory=memory, query=query, max_tokens=max_tokens)
        return [item.entry for item in selected]

    def select_with_scores(
        self,
        memory: Memory,
        query: str = "",
        max_tokens: int | None = None,
    ) -> list[SelectedMemory]:
        query_tokens = _tokens(query)
        candidates = [
            self._score(entry, query_tokens)
            for entry in memory.entries.values()
            if entry.status == "active"
        ]
        candidates.sort(
            key=lambda item: (
                -item.score,
                item.entry.section,
                -max(item.entry.provenance or [0]),
                item.entry.id,
            )
        )

        if max_tokens is None:
            return candidates

        selected: list[SelectedMemory] = []
        for candidate in candidates:
            projected_entries = [item.entry for item in selected] + [candidate.entry]
            rendered = memory.render(entries=projected_entries)
            if self.token_estimator(rendered) <= max_tokens:
                selected.append(candidate)
        return selected

    def _score(self, entry: MemoryEntry, query_tokens: set[str]) -> SelectedMemory:
        score = self.section_priorities.get(entry.section, 40.0)
        reasons = [f"section:{entry.section}"]

        entry_tokens = _tokens(entry.text)
        overlap = query_tokens & entry_tokens
        if overlap:
            score += len(overlap) * 12.0
            reasons.append(f"keyword_overlap:{len(overlap)}")

        if entry.provenance:
            score += min(max(entry.provenance), 1000) / 1000.0
            reasons.append("recency")

        return SelectedMemory(entry=entry, score=score, reasons=tuple(reasons))
