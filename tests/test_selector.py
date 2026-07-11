from memory_agent.models.sections import AGENT_SECTIONS, CHAT_SECTIONS
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


def test_selector_can_include_superseded_entries_for_conflict_history():
    memory = make_memory()
    memory.apply_ops([{"op": "SUPERSEDE", "id": "D1", "reason": "conflict"}])
    selector = MemorySelector(token_estimator=count_entries)
    selected = selector.select(
        memory=memory,
        query="storage cache",
        max_tokens=10,
        include_superseded=True,
    )
    assert "D1" in [entry.id for entry in selected]


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


def test_pinned_entry_cannot_exceed_hard_budget_alone():
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

    assert selected == []


def test_multiple_pinned_entries_are_prioritized_but_still_bounded():
    memory = make_memory()
    memory.apply_ops([{
        "op": "ADD", "section": "preferences",
        "text": "prefers short examples", "provenance": [5],
    }])
    selector = MemorySelector(token_estimator=count_entries)

    selected = selector.select(memory=memory, query="favorite color green", max_tokens=1)

    assert len(selected) == 1
    assert selected[0].section == "preferences"


def test_call_site_can_disable_default_pinned_sections():
    memory = make_memory()
    selector = MemorySelector(token_estimator=count_entries)
    selected = selector.select(
        memory=memory,
        query="favorite color is green",
        max_tokens=1,
        pinned_sections=frozenset(),
    )
    assert [entry.id for entry in selected] == ["F1"]


def test_temporal_query_boosts_timeline_entries():
    memory = Memory(sections=AGENT_SECTIONS)
    applied, rejected = memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": "OpenWeather API key is configured for the weather app.",
                "provenance": [1],
            },
            {
                "op": "ADD",
                "section": "timeline",
                "text": "OpenWeather API key obtained on March 10, 2024.",
                "provenance": [2],
            },
        ]
    )
    assert rejected == []
    assert len(applied) == 2
    selector = MemorySelector(token_estimator=count_entries, pinned_sections=frozenset())

    selected = selector.select(
        memory=memory,
        query="How many days passed after the API key date?",
        max_tokens=1,
    )

    assert [entry.id for entry in selected] == ["M1"]


def test_latest_value_query_boosts_status_changes():
    memory = Memory(sections=AGENT_SECTIONS)
    applied, rejected = memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": "OpenWeather API key has a daily call quota.",
                "provenance": [1],
            },
            {
                "op": "ADD",
                "section": "status_changes",
                "text": "API daily quota updated to 1,200 calls/day.",
                "provenance": [2],
            },
        ]
    )
    assert rejected == []
    assert len(applied) == 2
    selector = MemorySelector(token_estimator=count_entries, pinned_sections=frozenset())

    selected = selector.select(
        memory=memory,
        query="What is the updated daily quota?",
        max_tokens=1,
    )

    assert [entry.id for entry in selected] == ["C1"]
