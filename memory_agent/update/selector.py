"""Bounded recency selection for update prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from memory_agent.core.models import MemoryEntry
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class UpdateMemoryMatch:
    entry: MemoryEntry
    score: float
    reasons: tuple[str, ...]
    score_components: tuple[tuple[str, float], ...]
    confidence: float


@dataclass(frozen=True)
class UpdateMemorySelection:
    matches: tuple[UpdateMemoryMatch, ...]
    visible_tokens: int
    fallback_used: bool = False
    fallback_reason: str | None = None
    required_overflow_tokens: int = 0

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(match.entry for match in self.matches)


class UpdateMemorySelector:
    """Select a bounded recency view without semantic matching."""

    def __init__(
        self,
        memory: Memory,
        token_estimator: Callable[[str], int] | None = None,
        max_candidate_entries: int = 8,
    ) -> None:
        self.memory = memory
        self.token_estimator = token_estimator or _default_token_estimator
        self.max_candidate_entries = max(1, max_candidate_entries)

    def select_for_update(
        self,
        turns: list[Turn],
        budget: int | None,
    ) -> UpdateMemorySelection:
        if budget == 0:
            return UpdateMemorySelection(matches=(), visible_tokens=0)
        candidates = sorted(
            self.memory.entries.values(),
            key=lambda entry: (
                entry.status != "active",
                -(max(entry.provenance) if entry.provenance else -1),
                entry.id,
            ),
        )[: self.max_candidate_entries]
        selected: list[UpdateMemoryMatch] = []
        visible_tokens = 0
        for entry in candidates:
            rendered = self.memory.render(include_superseded=True, entries=[entry])
            tokens = self.token_estimator(rendered)
            if budget is not None and visible_tokens + tokens > budget:
                continue
            active_score = 2.0 if entry.status == "active" else 0.0
            recency_score = (
                min(max(entry.provenance), 1000) / 1000.0
                if entry.provenance
                else 0.0
            )
            selected.append(UpdateMemoryMatch(
                entry=entry,
                score=active_score + recency_score,
                reasons=(
                    ("active",) if entry.status == "active" else ("superseded",)
                ),
                score_components=(("active", active_score), ("recency", recency_score)),
                confidence=1.0 if entry.status == "active" else 0.0,
            ))
            visible_tokens += tokens
        return UpdateMemorySelection(tuple(selected), visible_tokens)
