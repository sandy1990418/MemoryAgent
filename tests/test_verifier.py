from memory_agent.models.sections import AGENT_SECTIONS
from memory_agent.models.transcript import Turn
from memory_agent.structured.memory import Memory
from memory_agent.structured.verifier import MemoryUpdateVerifier


DENIAL_TURN = Turn(
    id=58,
    role="user",
    content=(
        "I've never written any Flask routes or handled HTTP requests "
        "in this project, so I'm starting from scratch."
    ),
)


def make_verifier() -> MemoryUpdateVerifier:
    return MemoryUpdateVerifier()


def test_passes_without_cues_or_rejections():
    result = make_verifier().verify(
        evicted_turns=[Turn(id=1, role="user", content="Just building a dashboard.")],
        applied_ops=[],
        rejected_ops=[],
        memory=Memory(sections=AGENT_SECTIONS),
    )

    assert result.passed
    assert result.errors == []


def test_fails_when_rejected_ops_exist():
    result = make_verifier().verify(
        evicted_turns=[Turn(id=1, role="user", content="plain text")],
        applied_ops=[],
        rejected_ops=[{"op": {"op": "UPDATE", "id": "F999"}, "reason": "unknown id"}],
        memory=Memory(sections=AGENT_SECTIONS),
    )

    assert not result.passed


def test_fails_when_cue_is_unrecorded():
    result = make_verifier().verify(
        evicted_turns=[DENIAL_TURN],
        applied_ops=[],
        rejected_ops=[],
        memory=Memory(sections=AGENT_SECTIONS),
    )

    assert not result.passed
    assert "turn 58" in result.errors[0]


def test_passes_when_status_changes_add_is_applied():
    result = make_verifier().verify(
        evicted_turns=[DENIAL_TURN],
        applied_ops=[
            {
                "op": "ADD",
                "section": "status_changes",
                "text": "User stated: I've never written any Flask routes...",
                "provenance": [58],
            }
        ],
        rejected_ops=[],
        memory=Memory(sections=AGENT_SECTIONS),
    )

    assert result.passed


def test_passes_when_supersede_is_applied():
    result = make_verifier().verify(
        evicted_turns=[DENIAL_TURN],
        applied_ops=[{"op": "SUPERSEDE", "id": "F1", "reason": "user denied it"}],
        rejected_ops=[],
        memory=Memory(sections=AGENT_SECTIONS),
    )

    assert result.passed


def test_passes_when_cue_already_covered_by_active_entry():
    """Anti-infinite-retry escape hatch: a statement recorded by an earlier
    batch is deduped by the updater, so no fresh op will ever appear for it.
    Verification must treat the existing entry as satisfying the invariant,
    otherwise the evicted turns would be retried forever."""
    memory = Memory(sections=AGENT_SECTIONS)
    applied, rejected = memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "status_changes",
                "text": (
                    "User stated: I've never written any Flask routes or handled "
                    "HTTP requests in this project, so I'm starting from scratch."
                ),
                "provenance": [12],
            }
        ]
    )
    assert rejected == []

    result = make_verifier().verify(
        evicted_turns=[DENIAL_TURN],
        applied_ops=[],
        rejected_ops=[],
        memory=memory,
    )

    assert result.passed


def test_assistant_turns_do_not_trigger_the_cue_check():
    assistant_turn = Turn(
        id=7,
        role="assistant",
        content="You should never hardcode credentials.",
    )
    result = make_verifier().verify(
        evicted_turns=[assistant_turn],
        applied_ops=[],
        rejected_ops=[],
        memory=Memory(sections=AGENT_SECTIONS),
    )

    assert result.passed


def test_chinese_cue_triggers_the_check():
    turn = Turn(id=9, role="user", content="資料庫改成 PostgreSQL，不要再用 SQLite。")
    result = make_verifier().verify(
        evicted_turns=[turn],
        applied_ops=[],
        rejected_ops=[],
        memory=Memory(sections=AGENT_SECTIONS),
    )

    assert not result.passed
