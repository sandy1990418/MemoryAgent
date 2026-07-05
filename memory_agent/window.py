"""Sliding working-window of recent turns kept in the LLM context."""

from __future__ import annotations

from typing import Callable

from memory_agent.transcript import Turn


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


class WorkingWindow:
    """Holds the turns that are currently "in context" for the chat LLM.

    When the estimated token total exceeds max_tokens, the oldest turns
    (excluding the most recent two) are candidates for eviction into the
    structured memory. Turns are only actually removed after a caller
    confirms the memory update succeeded (see MemorySession).
    """

    def __init__(
        self,
        max_tokens: int,
        evict_fraction: float = 0.5,
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.evict_fraction = evict_fraction
        self.token_estimator = token_estimator or _default_token_estimator
        self._turns: list[Turn] = []

    def add(self, turn: Turn) -> None:
        self._turns.append(turn)

    def turns(self) -> list[Turn]:
        return list(self._turns)

    def total_tokens(self) -> int:
        return sum(self.token_estimator(t.content) for t in self._turns)

    def needs_eviction(self, extra_tokens: int = 0, max_tokens: int | None = None) -> bool:
        limit = self.max_tokens if max_tokens is None else max_tokens
        return self.total_tokens() + extra_tokens > limit

    def eviction_batch(self, extra_tokens: int = 0, max_tokens: int | None = None) -> list[Turn]:
        """Oldest turns to evict, keeping the most recent 2 turns untouched,
        stopping once the remaining total would drop to or below
        max_tokens * (1 - evict_fraction).
        """
        if len(self._turns) <= 2:
            return []

        limit = self.max_tokens if max_tokens is None else max_tokens
        target = limit * (1 - self.evict_fraction)
        evictable = self._turns[:-2]

        remaining_total = self.total_tokens() + extra_tokens
        batch: list[Turn] = []
        for turn in evictable:
            if remaining_total <= target:
                break
            batch.append(turn)
            remaining_total -= self.token_estimator(turn.content)

        return batch

    def remove(self, turns: list[Turn]) -> None:
        """Remove the given turns from the window. Call only after a
        successful memory update for those turns.
        """
        ids_to_remove = {t.id for t in turns}
        self._turns = [t for t in self._turns if t.id not in ids_to_remove]
