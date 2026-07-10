import json

from memory_agent.models.policy import get_memory_policy
from memory_agent.models.sections import PRACTICAL_SECTIONS
from memory_agent.structured.compactor import MemoryCompactor
from memory_agent.structured.memory import Memory
from tests.fakes import ScriptedLLM
from tests.practical_cases import SUBJECT_COMPACTION_CASE


def test_compaction_reduces_active_entries_and_preserves_latest_truth():
    case = SUBJECT_COMPACTION_CASE
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    applied, rejected = memory.apply_ops(case["preload"])
    assert rejected == []
    assert applied
    active_before = sum(entry.status == "active" for entry in memory.entries.values())

    compactor = MemoryCompactor(
        llm=ScriptedLLM(
            lambda system, messages: json.dumps(case["compactor_ops"])
        ),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    applied, rejected = compactor.compact(memory)

    assert rejected == []
    assert applied
    active_after = sum(entry.status == "active" for entry in memory.entries.values())
    assert active_after < active_before
    active_texts = {
        entry.text for entry in memory.entries.values() if entry.status == "active"
    }
    assert active_texts == set(case["expected_active"])
    assert "summary-based" in next(iter(active_texts))
    assert "rather than mem0" in next(iter(active_texts))
    assert memory.entries["D1"].status == "superseded"
    assert memory.entries["D2"].status == "superseded"
    assert memory.entries["D3"].status == "superseded"


def test_compactor_rejects_reactivating_superseded_entry():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "decisions",
                "text": "Use mem0 for product memory.",
                "provenance": [1],
            },
            {"op": "SUPERSEDE", "id": "D1", "reason": "Decision changed."},
            {
                "op": "ADD",
                "section": "decisions",
                "text": "Use summary memory.",
                "provenance": [2],
            },
            {
                "op": "ADD",
                "section": "decisions",
                "text": "Do not use mem0.",
                "provenance": [3],
            },
        ]
    )
    ops = [
        {"op": "SUPERSEDE", "id": "D2", "reason": "Bad merge."},
        {"op": "SUPERSEDE", "id": "D3", "reason": "Bad merge."},
        {
            "op": "ADD",
            "section": "decisions",
            "text": "Use mem0 for product memory.",
            "provenance": [2, 3],
        },
    ]
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda system, messages: json.dumps(ops)),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )

    applied, rejected = compactor.compact(memory)

    assert applied == []
    assert rejected
    assert memory.entries["D2"].status == "active"
    assert memory.entries["D3"].status == "active"


def test_compactor_accepts_string_noop_from_small_models():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": "Project uses Flask.",
                "provenance": [1],
            }
        ]
    )
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda system, messages: '["NOOP"]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )

    applied, rejected = compactor.compact(memory)

    assert applied == []
    assert rejected == []


def test_compactor_normalizes_small_model_key_value_source_id_schema():
    policy = get_memory_policy("chat")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "API latency was 300ms.", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "API latency is now 250ms.", "provenance": [2]},
    ])
    response = json.dumps([
        {"op": "SUPERSEDE", "key": "facts", "source_provenance_ids": ["F1", "F2"], "reason": "Merged."},
        {"op": "ADD", "key": "facts", "value": "API latency is now 250ms.", "source_provenance_ids": ["F1", "F2"]},
    ])
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda system, messages: response),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    applied, rejected = compactor.compact(memory)
    assert rejected == []
    assert len(applied) == 3
    assert memory.entries["F1"].status == "superseded"
    assert memory.entries["F2"].status == "superseded"
    assert memory.entries["F3"].provenance == [1, 2]


def test_compactor_normalizes_camel_case_and_nested_value_schema():
    policy = get_memory_policy("chat")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "goal", "text": "Ship MVP by April 15.", "provenance": [1]},
        {"op": "ADD", "section": "goal", "text": "MVP includes login and analytics.", "provenance": [2]},
    ])
    response = json.dumps([
        {"op": "SUPERSEDE", "key": "goal", "sourceProvenanceIds": ["G1", "G2"]},
        {"op": "ADD", "section": "goal", "value": {"description": "Ship login and analytics MVP by April 15."}, "provenance_ids": ["G1", "G2"]},
    ])
    compactor = MemoryCompactor(
        llm=ScriptedLLM(lambda system, messages: response),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    applied, rejected = compactor.compact(memory)
    assert rejected == []
    assert len(applied) == 3
    assert memory.entries["G3"].text == "Ship login and analytics MVP by April 15."
    assert memory.entries["G3"].provenance == [1, 2]
