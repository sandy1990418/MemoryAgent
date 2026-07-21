"""Regression coverage for bounded structural compactor review."""

import json

from memory_agent.application.structured_service import StructuredMemoryService
from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.update.compactor import MemoryCompactor
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def _memory() -> Memory:
    return Memory(sections=CHAT_SECTIONS)


def test_candidates_overlap_at_section_boundaries_with_bounded_windows():
    memory = _memory()
    memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "facts",
                "text": f"Fact {index}.",
                "provenance": [index],
            }
            for index in range(1, 10)
        ]
    )
    compactor = MemoryCompactor(
        ScriptedLLM(lambda *_: "[]"),
        CHAT_SECTIONS,
        policy=CHAT_POLICY,
        max_candidate_entries=8,
    )

    candidates = compactor.detect_candidates(memory)

    facts = [candidate for candidate in candidates if candidate.subject_key.startswith("facts:")]
    assert [entry.id for entry in facts[0].entries] == [f"F{index}" for index in range(1, 9)]
    assert any(
        {entry.id for entry in candidate.entries} >= {"F8", "F9"}
        for candidate in facts
    )
    assert any(
        [entry.id for entry in candidate.entries]
        == [f"F{index}" for index in range(2, 10)]
        for candidate in facts
    )
    assert all(len(candidate.entries) <= 8 for candidate in candidates)


def test_canonical_text_rejects_entry_ids_and_compaction_bookkeeping():
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
            {
                "op": "ADD",
                "section": "facts",
                "text": "Merged [F1] and [F2] (provenance: [1, 2]).",
                "provenance": [1, 2],
            },
        ]
    )
    compactor = MemoryCompactor(
        ScriptedLLM(lambda *_: response),
        CHAT_SECTIONS,
        policy=CHAT_POLICY,
    )

    applied, rejected = compactor.compact(memory)

    assert applied == []
    assert rejected
    assert "bookkeeping" in rejected[0]["detail"][0]["reason"]
    assert all(entry.status == "active" for entry in memory.entries.values())


def test_rejected_candidate_does_not_block_later_section_review():
    memory = _memory()
    memory.apply_ops(
        [
            {"op": "ADD", "section": "facts", "text": "Fact one.", "provenance": [1]},
            {"op": "ADD", "section": "facts", "text": "Fact two.", "provenance": [2]},
            {"op": "ADD", "section": "preferences", "text": "Pref one.", "provenance": [3]},
            {"op": "ADD", "section": "preferences", "text": "Pref two.", "provenance": [4]},
        ]
    )

    def estimate(rendered: str) -> int:
        return 100 if "## Facts" in rendered else 1

    def response(system: str, _messages: list[dict]) -> str:
        if "## User Preferences" in system:
            return json.dumps(
                [
                    {"op": "SUPERSEDE", "id": "U1", "reason": "Merged."},
                    {"op": "SUPERSEDE", "id": "U2", "reason": "Merged."},
                    {
                        "op": "ADD",
                        "section": "preferences",
                        "text": "The user has two durable preferences.",
                        "provenance": [3, 4],
                    },
                ]
            )
        raise AssertionError("oversized facts candidate should not call the LLM")

    compactor = MemoryCompactor(
        ScriptedLLM(response),
        CHAT_SECTIONS,
        policy=CHAT_POLICY,
        max_candidate_tokens=10,
        token_estimator=estimate,
    )
    service = StructuredMemoryService(
        memory=memory,
        updater=MemoryUpdater(ScriptedLLM(lambda *_: "[]"), CHAT_SECTIONS, policy=CHAT_POLICY),
        policy=CHAT_POLICY,
        compactor=compactor,
        compact_min_active_entries=1,
    )

    service.maybe_compact()

    assert compactor.metrics.attempted_calls == 1
    assert memory.entries["U1"].status == "superseded"
    assert memory.entries["U2"].status == "superseded"
    report = service.compaction_diagnostics()
    assert report["checks"][0]["candidate_count"] >= 2
    assert report["checks"][0]["eligible_candidate_count"] >= 2
