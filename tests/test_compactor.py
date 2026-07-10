import json
from pathlib import Path

from memory_agent.models.policy import get_memory_policy
from memory_agent.models.sections import PRACTICAL_SECTIONS
from memory_agent.structured.compactor import MemoryCompactor
from memory_agent.structured.memory import Memory
from tests.fakes import ScriptedLLM


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "practical_memory_cases.json"


def _compaction_case() -> dict:
    cases = json.loads(FIXTURE_PATH.read_text())
    return next(case for case in cases if case["category"] == "summarization/compaction")


def test_compaction_reduces_active_entries_and_preserves_latest_truth():
    case = _compaction_case()
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
