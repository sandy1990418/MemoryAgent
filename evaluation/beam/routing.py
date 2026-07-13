"""BEAM routing modes and the explicitly diagnostic oracle adapter."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from memory_agent.core.store import Memory
from memory_agent.retrieval.selector import MemorySelector


class RoutingMode(str, Enum):
    PRODUCTION = "production"
    ORACLE = "oracle"


def build_oracle_memory_context(
    *,
    query: str,
    question_type: str,
    memory: Memory,
    selector: MemorySelector,
    max_tokens: int,
) -> tuple[tuple[str, ...], str, str]:
    """Preserve historical BEAM-aware routing for diagnostic comparison only."""

    pinned_by_type: dict[str, Iterable[str]] = {
        "instruction_following": {"preferences"},
        "preference_following": {"preferences"},
        "contradiction_resolution": {"status_changes"},
        "summarization": {"preferences", "goal", "status_changes"},
    }
    budgets = {
        "instruction_following": min(max_tokens, 3500),
        "contradiction_resolution": min(max_tokens, 3000),
        "summarization": max_tokens,
    }
    budget = budgets.get(question_type, min(max_tokens, 2500))
    include_superseded = question_type == "contradiction_resolution"
    entries = selector.select(
        memory=memory,
        query=query,
        max_tokens=budget,
        include_superseded=include_superseded,
        pinned_sections=pinned_by_type.get(question_type, frozenset()),
    )
    rendered = memory.render(
        entries=entries,
        include_superseded=include_superseded,
        max_tokens=budget,
    )
    chronology = ""
    if question_type in {"summarization", "contradiction_resolution"}:
        chronology = memory.render_chronological(
            entries=None if question_type == "summarization" else entries,
            max_tokens=budget // 2,
            exclude_sections={"exact_values"},
        )
    return tuple(entry.id for entry in entries), rendered, chronology
