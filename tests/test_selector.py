"""Structural chat answer-memory selection contracts."""

from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.retrieval.selector import MemorySelector


def _memory() -> Memory:
    memory = Memory(sections=CHAT_SECTIONS)
    applied, rejected = memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": "First durable fact.",
                "provenance": [1],
            },
            {
                "op": "ADD",
                "section": "facts",
                "text": "Second durable fact.",
                "provenance": [2],
            },
            {
                "op": "ADD",
                "section": "facts",
                "text": "Newest durable fact.",
                "provenance": [3],
            },
        ]
    )
    assert rejected == []
    assert len(applied) == 3
    return memory


def _entry_tokens(text: str) -> int:
    return text.count("- [")


def test_selector_returns_newest_active_entries_within_hard_budget():
    memory = _memory()
    selector = MemorySelector(token_estimator=_entry_tokens)

    selected = selector.select(
        memory=memory,
        query="an unrelated question does not change bounded selection",
        max_tokens=2,
    )

    assert [entry.id for entry in selected] == ["F3", "F2"]
    assert selector.token_estimator(memory.render(entries=selected)) <= 2


def test_selector_does_not_use_query_or_benchmark_metadata_for_ordering():
    memory = _memory()
    selector = MemorySelector(token_estimator=_entry_tokens)

    first = selector.select(memory, query="first fact", max_tokens=2)
    second = selector.select(memory, query="database migration", max_tokens=2)
    third = selector.select(memory, query="", max_tokens=2, pinned_sections=frozenset())

    assert [entry.id for entry in first] == ["F3", "F2"]
    assert [entry.id for entry in second] == ["F3", "F2"]
    assert [entry.id for entry in third] == ["F3", "F2"]


def test_selector_excludes_superseded_entries_by_default():
    memory = _memory()
    memory.apply_ops([{"op": "SUPERSEDE", "id": "F2", "reason": "corrected"}])

    selected = MemorySelector(token_estimator=_entry_tokens).select(
        memory, query="anything", max_tokens=10
    )

    assert [entry.id for entry in selected] == ["F3", "F1"]


def test_selector_can_include_superseded_entries_for_history():
    memory = _memory()
    memory.apply_ops([{"op": "SUPERSEDE", "id": "F2", "reason": "corrected"}])

    selected = MemorySelector(token_estimator=_entry_tokens).select(
        memory, query="anything", max_tokens=10, include_superseded=True
    )

    # Active entries are always emitted before superseded history.
    assert [entry.id for entry in selected] == ["F3", "F1", "F2"]


def test_selector_skips_an_oversized_entry_without_slicing():
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops(
        [{"op": "ADD", "section": "facts", "text": "A very long fact.", "provenance": [1]}]
    )
    selector = MemorySelector(token_estimator=lambda _text: 10)

    assert selector.select(memory, query="anything", max_tokens=2) == []


def test_selector_scores_are_structural_active_and_recency_indicators():
    memory = _memory()
    selector = MemorySelector(token_estimator=_entry_tokens)

    scored = selector.select_with_scores(memory, query="irrelevant", max_tokens=2)

    assert [item.entry.id for item in scored] == ["F3", "F2"]
    assert scored[0].reasons == ("active", "recency")
    assert scored[0].score == 1.003
