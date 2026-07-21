"""Memory state serialization round-trip used by frozen-memory replay."""

import pytest

from memory_agent.core.models import MemoryValue, SubjectIdentity
from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.retrieval.context import build_answer_memory_context
from memory_agent.retrieval.selector import MemorySelector


def _populated_memory() -> Memory:
    policy = CHAT_POLICY
    memory = Memory(CHAT_SECTIONS, policy=policy)
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
    clone = Memory(CHAT_SECTIONS, policy=memory.policy)
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
    restored = Memory(CHAT_SECTIONS, policy=memory.policy)
    restored.load_state(json.loads(json.dumps(memory.to_state())))

    selector = MemorySelector(policy=CHAT_POLICY)

    def selected_ids(target: Memory) -> tuple[str, ...]:
        entries = selector.select_for_answer(
            memory=target,
            query="What database does the service use?",
            budget=200,
        )
        return build_answer_memory_context(
            memory=target,
            entries=entries,
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
    restored = Memory(CHAT_SECTIONS, policy=memory.policy)
    restored.load_state(state)

    restored.apply_ops(
        [{"op": "ADD", "section": "facts", "text": "New fact.", "provenance": [9]}]
    )

    assert len(restored.entries) == len(memory.entries) + 1


def test_load_state_rejects_unknown_section():
    restored = Memory(CHAT_SECTIONS, policy=CHAT_POLICY)

    with pytest.raises(ValueError, match="unknown section"):
        restored.load_state(
            {"entries": [{"id": "X1", "section": "nonexistent", "text": "x", "provenance": []}]}
        )
