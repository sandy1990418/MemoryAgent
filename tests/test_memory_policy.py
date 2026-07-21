"""Chat-only retention and structural safety contracts."""

import json

from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def _updater(script, *, llm=None):
    return MemoryUpdater(
        llm=llm or ScriptedLLM(script),
        sections=CHAT_SECTIONS,
        policy=CHAT_POLICY,
    )


def _memory():
    return Memory(sections=CHAT_SECTIONS, policy=CHAT_POLICY)


def test_chat_policy_is_the_only_runtime_policy():
    assert CHAT_POLICY.name == "chat"
    assert CHAT_POLICY.max_ops_per_batch is None
    assert "exact_values" in CHAT_POLICY.disallowed_sections
    assert "timeline" in CHAT_POLICY.disallowed_sections


def test_chat_sections_are_the_only_runtime_section_set():
    assert {section.key for section in CHAT_SECTIONS} == {
        "decisions",
        "preferences",
        "status_changes",
        "goal",
        "facts",
        "progress",
        "open_questions",
        "failed_attempts",
    }


def test_chat_policy_filters_unsupported_sections_without_dropping_valid_batch_ops():
    updater = _updater(
        lambda system, messages: json.dumps(
            [
                {
                    "op": "ADD",
                    "section": "timeline",
                    "text": "Release date is 2026-09-01.",
                    "provenance": [1],
                },
                {
                    "op": "ADD",
                    "section": "preferences",
                    "text": "User prefers concise answers.",
                    "provenance": [1],
                },
                {
                    "op": "ADD",
                    "section": "facts",
                    "text": "Project uses PostgreSQL.",
                    "provenance": [1],
                },
                {
                    "op": "ADD",
                    "section": "decisions",
                    "text": "Project will use server-side rendering.",
                    "provenance": [1],
                },
                {
                    "op": "ADD",
                    "section": "facts",
                    "text": "Project uses Redis.",
                    "provenance": [1],
                },
            ]
        )
    )
    memory = _memory()

    applied, rejected = updater.update(
        memory,
        [Turn(id=1, role="user", content="I prefer concise answers for this project.")],
    )

    assert applied == []
    assert len(rejected) == 1
    assert "timeline" in rejected[0]["reason"]
    assert memory.entries == {}

    uncapped = _updater(
        lambda system, messages: json.dumps(
            [
                {"op": "ADD", "section": "preferences", "text": "A", "provenance": [1]},
                {"op": "ADD", "section": "facts", "text": "B", "provenance": [1]},
                {"op": "ADD", "section": "decisions", "text": "C", "provenance": [1]},
                {"op": "ADD", "section": "facts", "text": "D", "provenance": [1]},
            ]
        )
    )
    uncapped_memory = _memory()
    uncapped_applied, uncapped_rejected = uncapped.update(
        uncapped_memory,
        [Turn(id=1, role="user", content="Several durable facts.")],
    )
    assert uncapped_rejected == []
    assert len(uncapped_applied) == 4


def test_ordinary_assistant_explanation_is_not_saved():
    updater = _updater(
        lambda system, messages: '[{"op":"NOOP"}]',
    )
    memory = _memory()

    applied, rejected = updater.update(
        memory,
        [
            Turn(id=1, role="user", content="How does Redis persistence work?"),
            Turn(id=2, role="assistant", content="Redis supports snapshots and AOF."),
        ],
    )

    assert applied == []
    assert rejected == []
    assert memory.entries == {}


def test_accepted_user_state_can_be_saved_without_evaluation_profile():
    updater = _updater(
        lambda system, messages: '[{"op":"ADD","section":"preferences",'
        '"text":"User prefers concise replies.","provenance":[1]}]'
    )
    memory = _memory()

    applied, rejected = updater.update(
        memory,
        [Turn(id=1, role="user", content="I prefer concise replies.")],
    )

    assert rejected == []
    assert applied
    assert any(entry.section == "preferences" for entry in memory.entries.values())


def test_chat_policy_does_not_expose_evaluation_prompt_rules():
    updater = _updater(lambda *_: '[{"op":"NOOP"}]')
    system, messages = updater._build_prompt(
        _memory(),
        [Turn(id=1, role="user", content="How does memory work?")],
    )

    assert "EVAL PROFILE" not in system
    assert "evaluation profile" not in system.lower()
    assert "information extraction" not in system.lower()
    assert "operation-count limit" not in system
    assert messages == [
        {
            "role": "user",
            "content": "Apply the rules above and return the ops JSON array for these turns.",
        }
    ]


def test_chat_entry_validation_quarantines_oversized_llm_entry():
    updater = _updater(
        lambda system, messages: json.dumps(
            [
                {
                    "op": "ADD",
                    "section": "facts",
                    "text": "Project uses Flask " + "with detailed configuration " * 20,
                    "provenance": [1],
                }
            ]
        )
    )
    memory = _memory()

    updater.update(memory, [Turn(id=1, role="user", content="My project uses Flask.")])

    assert memory.entries == {}


def test_chat_policy_keeps_explicit_correction_as_status_change():
    memory = _memory()
    memory.apply_ops(
        [
            {
                "op": "ADD",
                "section": "decisions",
                "text": "Use mem0 for product memory.",
                "provenance": [1],
            }
        ]
    )
    updater = _updater(
        lambda *_: json.dumps(
            [
                {"op": "SUPERSEDE", "id": "D1", "reason": "User corrected the approach."},
                {
                    "op": "ADD",
                    "section": "decisions",
                    "text": "Use summary-based memory rather than mem0.",
                    "provenance": [2],
                },
            ]
        )
    )

    applied, rejected = updater.update(
        memory,
        [Turn(id=2, role="user", content="Actually, use summary-based memory rather than mem0.")],
    )

    assert rejected == []
    assert any(op["op"] == "SUPERSEDE" for op in applied)
    assert memory.entries["D1"].status == "superseded"
    assert any(
        entry.status == "active" and "summary-based" in entry.text
        for entry in memory.entries.values()
    )
