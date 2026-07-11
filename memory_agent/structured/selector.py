"""Session-local memory selection for prompt injection."""

from __future__ import annotations

import re
from typing import Callable, Iterable

from memory_agent.models.memory import MemoryEntry, SelectedMemory
from memory_agent.models.policy import MemoryPolicy
from memory_agent.structured.memory import Memory

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_TEMPORAL_QUERY_TERMS = {
    "when",
    "date",
    "deadline",
    "days",
    "weeks",
    "between",
    "before",
    "after",
    "passed",
    "start",
    "end",
    "timeline",
}
_LATEST_VALUE_QUERY_TERMS = {
    "latest",
    "current",
    "updated",
    "update",
    "changed",
    "percentage",
    "percent",
    "count",
    "total",
    "how",
    "many",
}


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


class MemorySelector:
    """Select active memory entries for the current in-session prompt.

    This is intentionally deterministic. It is not semantic search and does not
    need embeddings. The goal is to stop blindly rendering every active memory
    entry once a single session grows large. Entries in pinned sections are
    always selected even when they exceed the nominal prompt budget.
    """

    # Status changes are rare, high-value contradiction/correction records.
    DEFAULT_PINNED_SECTIONS = frozenset({"preferences", "goal", "status_changes"})

    DEFAULT_SECTION_PRIORITIES: dict[str, float] = {
        "preferences": 100.0,
        "goal": 95.0,
        "status_changes": 94.0,
        "progress": 90.0,
        "timeline": 88.0,
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
        pinned_sections: Iterable[str] | None = None,
        policy: MemoryPolicy | None = None,
    ) -> None:
        self.section_priorities = dict(self.DEFAULT_SECTION_PRIORITIES)
        if section_priorities:
            self.section_priorities.update(section_priorities)
        self.token_estimator = token_estimator or _default_token_estimator
        self.pinned_sections = (
            self.DEFAULT_PINNED_SECTIONS
            if pinned_sections is None
            else frozenset(pinned_sections)
        )
        self.policy = policy

    def select(
        self,
        memory: Memory,
        query: str = "",
        max_tokens: int | None = None,
        include_superseded: bool = False,
        pinned_sections: Iterable[str] | None = None,
    ) -> list[MemoryEntry]:
        selected = self.select_with_scores(
            memory=memory,
            query=query,
            max_tokens=max_tokens,
            include_superseded=include_superseded,
            pinned_sections=pinned_sections,
        )
        return [item.entry for item in selected]

    def select_for_answer(
        self,
        memory: Memory,
        query: str = "",
        budget: int | None = None,
    ) -> list[MemoryEntry]:
        """Production answer selection; ``select`` remains API-compatible."""
        return self.select(memory=memory, query=query, max_tokens=budget)

    def select_with_scores(
        self,
        memory: Memory,
        query: str = "",
        max_tokens: int | None = None,
        include_superseded: bool = False,
        pinned_sections: Iterable[str] | None = None,
    ) -> list[SelectedMemory]:
        query_tokens = _tokens(query)
        candidates = [
            self._score(entry, query_tokens)
            for entry in memory.entries.values()
            if entry.status == "active" or include_superseded
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

        # Pinned sections are a hard guarantee: include them even if rendering
        # them alone exceeds max_tokens. Budget only gates non-pinned entries.
        effective_pinned = (
            self.pinned_sections if pinned_sections is None else frozenset(pinned_sections)
        )
        pinned_ids = {
            candidate.entry.id
            for candidate in candidates
            if candidate.entry.section in effective_pinned
        }
        selected_ids = set(pinned_ids)
        for candidate in candidates:
            if candidate.entry.id in pinned_ids:
                continue
            projected_ids = selected_ids | {candidate.entry.id}
            projected_entries = [
                item.entry for item in candidates if item.entry.id in projected_ids
            ]
            rendered = memory.render(entries=projected_entries)
            if self.token_estimator(rendered) <= max_tokens:
                selected_ids.add(candidate.entry.id)
        return [candidate for candidate in candidates if candidate.entry.id in selected_ids]

    def _score(self, entry: MemoryEntry, query_tokens: set[str]) -> SelectedMemory:
        score = self.section_priorities.get(entry.section, 40.0)
        reasons = [f"section:{entry.section}"]

        if query_tokens & _TEMPORAL_QUERY_TERMS:
            temporal_boosts = {
                "timeline": 55.0,
                "progress": 25.0,
                "status_changes": 20.0,
                "facts": 10.0,
            }
            boost = temporal_boosts.get(entry.section, 0.0)
            if boost:
                score += boost
                reasons.append("temporal_query")

        if query_tokens & _LATEST_VALUE_QUERY_TERMS:
            latest_boosts = {
                "status_changes": 45.0,
                "progress": 30.0,
                "facts": 25.0,
                "timeline": 15.0,
            }
            boost = latest_boosts.get(entry.section, 0.0)
            if boost:
                score += boost
                reasons.append("latest_value_query")

        entry_tokens = _tokens(entry.text)
        overlap = query_tokens & entry_tokens
        if overlap:
            score += len(overlap) * 12.0
            reasons.append(f"keyword_overlap:{len(overlap)}")

        if entry.provenance:
            score += min(max(entry.provenance), 1000) / 1000.0
            reasons.append("recency")

        return SelectedMemory(entry=entry, score=score, reasons=tuple(reasons))
