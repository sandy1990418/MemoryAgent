"""Structural verification only; semantic cues belong to the model."""

from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.update.verifier import MemoryUpdateVerifier


def make_verifier() -> MemoryUpdateVerifier:
    return MemoryUpdateVerifier()


def test_passes_without_ops_or_semantic_cues():
    result = make_verifier().verify(
        evicted_turns=[Turn(id=1, role="user", content="Just building a dashboard.")],
        applied_ops=[],
        rejected_ops=[],
        memory=Memory(sections=CHAT_SECTIONS),
    )

    assert result.passed
    assert result.errors == []


def test_fails_when_rejected_ops_exist():
    result = make_verifier().verify(
        evicted_turns=[Turn(id=1, role="user", content="plain text")],
        applied_ops=[],
        rejected_ops=[{"op": {"op": "UPDATE", "id": "F999"}, "reason": "unknown id"}],
        memory=Memory(sections=CHAT_SECTIONS),
    )

    assert not result.passed
    assert "Rejected ops exist" in result.errors[0]


def test_fails_when_add_uses_unknown_section_or_turn():
    result = make_verifier().verify(
        evicted_turns=[Turn(id=1, role="user", content="plain text")],
        applied_ops=[
            {"op": "ADD", "section": "timeline", "text": "bad", "provenance": [99]}
        ],
        rejected_ops=[],
        memory=Memory(sections=CHAT_SECTIONS),
    )

    assert not result.passed
    assert any("unknown section" in error for error in result.errors)
    assert any("unknown turn ids" in error for error in result.errors)


def test_passes_valid_add_without_inspecting_semantic_content():
    result = make_verifier().verify(
        evicted_turns=[Turn(id=58, role="user", content="A correction.")],
        applied_ops=[
            {
                "op": "ADD",
                "section": "status_changes",
                "text": "Model-selected correction.",
                "provenance": [58],
            }
        ],
        rejected_ops=[],
        memory=Memory(sections=CHAT_SECTIONS),
    )

    assert result.passed


def test_update_and_supersede_require_existing_memory_ids():
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops(
        [
            {"op": "ADD", "section": "facts", "text": "Original fact.", "provenance": [1]},
            {"op": "ADD", "section": "decisions", "text": "Original decision.", "provenance": [1]},
        ]
    )
    result = make_verifier().verify(
        evicted_turns=[Turn(id=2, role="user", content="Updated state.")],
        applied_ops=[
            {"op": "UPDATE", "id": "F1", "text": "Updated fact.", "provenance": [2]},
            {"op": "SUPERSEDE", "id": "D1", "reason": "Replaced."},
        ],
        rejected_ops=[],
        memory=memory,
    )

    assert result.passed


def test_unknown_ids_are_rejected_structurally():
    result = make_verifier().verify(
        evicted_turns=[Turn(id=2, role="user", content="Updated state.")],
        applied_ops=[
            {"op": "UPDATE", "id": "F9", "text": "Updated fact.", "provenance": [2]},
            {"op": "SUPERSEDE", "id": "D9", "reason": "Replaced."},
        ],
        rejected_ops=[],
        memory=Memory(sections=CHAT_SECTIONS),
    )

    assert not result.passed
    assert any("unknown id" in error for error in result.errors)
