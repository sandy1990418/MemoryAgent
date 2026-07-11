"""Production answer-memory selection and rendering."""

from __future__ import annotations

from dataclasses import dataclass

from memory_agent.structured.memory import Memory
from memory_agent.structured.selector import MemorySelector


@dataclass(frozen=True)
class AnswerContextConfig:
    """Production selection policy, independent of evaluation metadata."""

    selector: MemorySelector


@dataclass(frozen=True)
class AnswerContextBudget:
    max_tokens: int | None


@dataclass(frozen=True)
class AnswerContext:
    selected_ids: tuple[str, ...]
    rendered_context: str


def build_answer_memory_context(
    *,
    query: str,
    memory: Memory,
    config: AnswerContextConfig,
    budget: AnswerContextBudget,
) -> AnswerContext:
    """Return the exact production-selected IDs and rendered memory context."""

    entries = config.selector.select(
        memory=memory,
        query=query,
        max_tokens=budget.max_tokens,
    )
    return AnswerContext(
        selected_ids=tuple(entry.id for entry in entries),
        rendered_context=memory.render(entries=entries),
    )
