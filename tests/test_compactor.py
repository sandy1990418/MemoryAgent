"""Structural and token-budget contracts for chat compaction."""

import json

from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.update.compactor import CompactionCandidate, MemoryCompactor
from tests.fakes import ScriptedLLM


def _memory() -> Memory:
    return Memory(sections=CHAT_SECTIONS, policy=CHAT_POLICY)


def _compactor(response="[]", **kwargs) -> MemoryCompactor:
    return MemoryCompactor(
        llm=ScriptedLLM(lambda *_: response),
        sections=CHAT_SECTIONS,
        policy=CHAT_POLICY,
        **kwargs,
    )


def test_detect_candidates_uses_structural_section_chunks_and_recency():
    memory = _memory()
    memory.apply_ops(
        [
            {"op": "ADD", "section": "facts", "text": "First fact.", "provenance": [1]},
            {"op": "ADD", "section": "facts", "text": "Second fact.", "provenance": [2]},
            {"op": "ADD", "section": "preferences", "text": "Short replies.", "provenance": [3]},
        ]
    )

    candidates = _compactor().detect_candidates(memory)

    assert len(candidates) == 1
    assert candidates[0].subject_key == "facts:0"
    assert [entry.id for entry in candidates[0].entries] == ["F1", "F2"]
    assert candidates[0].reason in {"llm-candidate", "semantic-overlap"}


def test_compaction_applies_model_selected_replacement_atomically():
    memory = _memory()
    memory.apply_ops(
        [
            {"op": "ADD", "section": "facts", "text": "First fact.", "provenance": [1]},
            {"op": "ADD", "section": "facts", "text": "Second fact.", "provenance": [2]},
        ]
    )
    response = json.dumps(
        [
            {"op": "SUPERSEDE", "id": "F1", "reason": "Merged."},
            {"op": "SUPERSEDE", "id": "F2", "reason": "Merged."},
            {"op": "ADD", "section": "facts", "text": "Merged facts.", "provenance": [1, 2]},
        ]
    )
    compactor = _compactor(response)

    applied, rejected = compactor.compact(memory)

    assert rejected == []
    assert [op["op"] for op in applied] == ["SUPERSEDE", "SUPERSEDE", "ADD"]
    assert memory.entries["F1"].status == "superseded"
    assert memory.entries["F2"].status == "superseded"
    assert any(entry.text == "Merged facts." and entry.status == "active" for entry in memory.entries.values())


def test_compaction_rejects_hidden_ids_without_partial_commit():
    memory = _memory()
    memory.apply_ops(
        [
            {"op": "ADD", "section": "facts", "text": "First fact.", "provenance": [1]},
            {"op": "ADD", "section": "facts", "text": "Second fact.", "provenance": [2]},
        ]
    )
    compactor = _compactor(
        '[{"op":"SUPERSEDE","id":"F1","reason":"Merged."},'
        '{"op":"SUPERSEDE","id":"F9","reason":"Hidden."}]'
    )

    applied, rejected = compactor.compact(memory)

    assert applied == []
    assert rejected
    assert all(entry.status == "active" for entry in memory.entries.values())


def test_compaction_budget_rejects_oversized_candidate_without_transport():
    calls = []
    memory = _memory()
    memory.apply_ops(
        [
            {"op": "ADD", "section": "facts", "text": "A " * 200, "provenance": [1]},
            {"op": "ADD", "section": "facts", "text": "B " * 200, "provenance": [2]},
        ]
    )
    compactor = _compactor(
        "[]",
        max_candidate_tokens=1,
        token_estimator=lambda text: len(text),
    )
    compactor.llm = ScriptedLLM(lambda *_: calls.append(True) or "[]")

    applied, rejected = compactor.compact(memory)

    assert applied == []
    assert rejected
    assert calls == []
    assert all(entry.status == "active" for entry in memory.entries.values())


def test_compactor_prompt_is_chat_only():
    memory = _memory()
    memory.apply_ops(
        [
            {"op": "ADD", "section": "facts", "text": "First fact.", "provenance": [1]},
            {"op": "ADD", "section": "facts", "text": "Second fact.", "provenance": [2]},
        ]
    )
    candidate = CompactionCandidate(
        "facts:0",
        tuple(memory.entries.values()),
        "llm-candidate",
    )
    system, _ = _compactor()._build_prompt(memory, candidate)

    assert "SUPERSEDE every replaced active entry" in system
    assert "EVAL PROFILE" not in system
    assert "timeline" not in system.lower()
