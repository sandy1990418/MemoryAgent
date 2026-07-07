from memory_agent.models.sections import CHAT_SECTIONS
from memory_agent.structured.memory import Memory
from memory_agent.structured.selector import MemorySelector


def make_memory() -> Memory:
    memory = Memory(sections=CHAT_SECTIONS)
    applied, rejected = memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "preferences",
                "text": "prefers detailed paragraph answers",
                "provenance": [1],
            },
            {
                "op": "ADD",
                "section": "decisions",
                "text": "use file storage for the cache layer",
                "provenance": [2],
            },
            {
                "op": "ADD",
                "section": "facts",
                "text": "favorite color is green",
                "provenance": [3],
            },
            {
                "op": "ADD",
                "section": "open_questions",
                "text": "choose a production database",
                "provenance": [4],
            },
        ]
    )
    assert rejected == []
    assert len(applied) == 4
    return memory


def count_entries(text: str) -> int:
    return text.count("- [")


def test_selector_prefers_relevant_and_high_priority_entries_within_budget():
    memory = make_memory()
    selector = MemorySelector(token_estimator=count_entries)

    selected = selector.select(
        memory=memory,
        query="What did we decide about storage for the cache?",
        max_tokens=2,
    )

    selected_ids = [entry.id for entry in selected]
    assert selected_ids == ["D1", "U1"]


def test_selector_excludes_superseded_entries():
    memory = make_memory()
    memory.apply_ops([{"op": "SUPERSEDE", "id": "D1", "reason": "changed storage plan"}])
    selector = MemorySelector(token_estimator=count_entries)

    selected = selector.select(memory=memory, query="storage cache", max_tokens=10)

    assert "D1" not in [entry.id for entry in selected]


def test_render_can_render_only_selected_entries():
    memory = make_memory()
    selector = MemorySelector(token_estimator=count_entries, pinned_sections=frozenset())

    selected = selector.select(memory=memory, query="favorite color green is", max_tokens=1)
    rendered = memory.render(entries=selected)

    assert "favorite color is green" in rendered
    assert "use file storage" not in rendered


def test_pinned_preferences_win_budget_over_relevant_fact():
    memory = make_memory()
    selector = MemorySelector(token_estimator=count_entries)

    selected = selector.select(memory=memory, query="favorite color green is", max_tokens=1)

    selected_ids = [entry.id for entry in selected]
    assert "U1" in selected_ids
    assert "F1" not in selected_ids


def test_empty_pinned_sections_restores_pure_budget_behavior():
    memory = make_memory()
    selector = MemorySelector(token_estimator=count_entries, pinned_sections=frozenset())

    selected = selector.select(
        memory=memory,
        query="What did we decide about storage for the cache?",
        max_tokens=1,
    )

    assert [entry.id for entry in selected] == ["D1"]


def test_pinned_entry_is_included_even_when_over_budget_alone():
    memory = Memory(sections=CHAT_SECTIONS)
    applied, rejected = memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "preferences",
                "text": "prefers a very detailed explanation with all relevant tradeoffs",
                "provenance": [1],
            }
        ]
    )
    assert rejected == []
    assert len(applied) == 1
    selector = MemorySelector(token_estimator=count_entries)

    selected = selector.select(memory=memory, query="anything", max_tokens=0)

    assert [entry.id for entry in selected] == ["U1"]
