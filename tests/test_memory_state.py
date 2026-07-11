"""Memory state serialization round-trip used by frozen-memory replay."""

import pytest

from memory_agent.models.memory import MemoryValue, SubjectIdentity
from memory_agent.models.policy import get_memory_policy
from memory_agent.models.sections import PRACTICAL_SECTIONS
from memory_agent.structured.answer_context import (
    AnswerContextBudget,
    AnswerContextConfig,
    build_answer_memory_context,
)
from memory_agent.structured.memory import Memory
from memory_agent.structured.selector import MemorySelector


def _populated_memory() -> Memory:
    policy = get_memory_policy("practical")
    memory = Memory(PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "API latency is 250ms.", "provenance": [1]},
        {
            "op": "ADD",
            "section": "facts",
            "text": "Service uses Postgres 16.",
            "provenance": [2],
            "subject_identity": SubjectIdentity("project", "service", "database"),
            "value": MemoryValue("Postgres", unit=None),
        },
        {"op": "ADD", "section": "preferences", "text": "User prefers short answers.", "provenance": [3]},
    ])
    facts_ids = [e.id for e in memory.entries.values() if e.section == "facts"]
    memory.apply_ops([{"op": "SUPERSEDE", "id": facts_ids[0], "reason": "latency changed"}])
    memory.set_narrative("Early project setup discussion.")
    return memory


def _restored(memory: Memory) -> Memory:
    clone = Memory(PRACTICAL_SECTIONS, policy=memory.policy)
    clone.load_state(memory.to_state())
    return clone


def test_state_round_trip_preserves_entries_narrative_and_render():
    memory = _populated_memory()
    restored = _restored(memory)

    assert restored.narrative == memory.narrative
    assert set(restored.entries) == set(memory.entries)
    for entry_id, entry in memory.entries.items():
        copy = restored.entries[entry_id]
        assert (copy.section, copy.text, copy.provenance, copy.status, copy.note) == (
            entry.section, entry.text, entry.provenance, entry.status, entry.note
        )
        assert copy.subject_identity == entry.subject_identity
        assert copy.value == entry.value
    assert restored.render(include_superseded=True) == memory.render(include_superseded=True)


def test_state_round_trip_survives_json_and_preserves_selection():
    import json

    memory = _populated_memory()
    restored = Memory(PRACTICAL_SECTIONS, policy=memory.policy)
    restored.load_state(json.loads(json.dumps(memory.to_state())))

    policy = get_memory_policy("practical")
    selector = MemorySelector(policy=policy, pinned_sections=frozenset())

    def selected_ids(target: Memory) -> tuple[str, ...]:
        return build_answer_memory_context(
            query="What database does the service use?",
            memory=target,
            config=AnswerContextConfig(selector),
            budget=AnswerContextBudget(200),
        ).selected_ids

    assert selected_ids(restored) == selected_ids(memory)


def test_load_state_restores_counters_so_new_ids_do_not_collide():
    memory = _populated_memory()
    restored = _restored(memory)

    applied, rejected = restored.apply_ops(
        [{"op": "ADD", "section": "facts", "text": "New fact.", "provenance": [9]}]
    )

    assert applied and not rejected
    assert len(restored.entries) == len(memory.entries) + 1


def test_load_state_recomputes_counters_when_missing():
    memory = _populated_memory()
    state = memory.to_state()
    del state["counters"]
    restored = Memory(PRACTICAL_SECTIONS, policy=memory.policy)
    restored.load_state(state)

    restored.apply_ops(
        [{"op": "ADD", "section": "facts", "text": "New fact.", "provenance": [9]}]
    )

    assert len(restored.entries) == len(memory.entries) + 1


def test_load_state_rejects_unknown_section():
    restored = Memory(PRACTICAL_SECTIONS, policy=get_memory_policy("practical"))

    with pytest.raises(ValueError, match="unknown section"):
        restored.load_state(
            {"entries": [{"id": "X1", "section": "nonexistent", "text": "x", "provenance": []}]}
        )
