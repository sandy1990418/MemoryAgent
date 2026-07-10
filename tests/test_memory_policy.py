import json
from pathlib import Path

import pytest

from memory_agent.agents.structured import build_structured_middleware
from memory_agent.models.config import StructuredAgentConfig
from memory_agent.models.policy import get_memory_policy
from memory_agent.models.sections import EVAL_SECTIONS, PRACTICAL_SECTIONS
from memory_agent.models.transcript import Turn
from memory_agent.structured.memory import Memory
from memory_agent.structured.updater import MemoryUpdater
from memory_agent.structured.verifier import MemoryUpdateVerifier
from tests.fakes import ScriptedLLM


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "practical_memory_cases.json"


def _cases() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.mark.parametrize(
    "case",
    [case for case in _cases() if "turns" in case],
    ids=lambda case: case["id"],
)
def test_practical_synthetic_retention(case):
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    if case.get("preload"):
        applied, rejected = memory.apply_ops(case["preload"])
        assert rejected == []
        assert applied

    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: json.dumps(case["llm_ops"])),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    turns = [
        Turn(id=turn["turn_id"], role=turn["role"], content=turn["content"])
        for turn in case["turns"]
    ]

    _applied, rejected = updater.update(memory, turns)

    assert rejected == []
    active_texts = {
        entry.text for entry in memory.entries.values() if entry.status == "active"
    }
    assert set(case["expected_active"]).issubset(active_texts)
    if not case["expected_active"]:
        assert active_texts == set()

    if case["category"] == "preference_following":
        assert any(
            entry.section == "preferences" and entry.status == "active"
            for entry in memory.entries.values()
        )
    if case["id"] == "durable-project-direction":
        assert any(
            entry.section == "decisions" and entry.status == "active"
            for entry in memory.entries.values()
        )
    if case["id"] == "active-blocker":
        assert any(
            entry.section == "open_questions" and entry.status == "active"
            for entry in memory.entries.values()
        )
    if case["id"] == "failed-attempt":
        assert any(
            entry.section == "failed_attempts" and entry.status == "active"
            for entry in memory.entries.values()
        )
    if case["category"] == "contradiction_resolution":
        assert memory.entries["D1"].status == "superseded"
        assert any(
            op["op"] == "SUPERSEDE"
            for op in _applied
        )
        assert any(op["op"] == "ADD" for op in _applied)


def test_practical_profile_filters_disallowed_sections_and_caps_batch():
    policy = get_memory_policy("practical")
    updater = MemoryUpdater(
        llm=ScriptedLLM(
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
        ),
        sections=[*PRACTICAL_SECTIONS],
        policy=policy,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)

    applied, rejected = updater.update(
        memory,
        [Turn(id=1, role="user", content="I prefer concise answers for this project.")],
    )

    assert rejected == []
    assert len(applied) == 3
    assert memory.entries["U1"].section == "preferences"
    assert {entry.section for entry in memory.entries.values()} == {
        "preferences",
        "decisions",
        "facts",
    }


def test_eval_profile_keeps_beam_style_details():
    policy = get_memory_policy("eval")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
        sections=EVAL_SECTIONS,
        policy=policy,
    )
    memory = Memory(sections=EVAL_SECTIONS, policy=policy)

    applied, rejected = updater.update(
        memory,
        [
            Turn(
                id=1,
                role="user",
                content=(
                    "The deployment deadline moved to March 15, 2024, and API "
                    "test coverage improved to 78%."
                ),
            )
        ],
    )

    assert rejected == []
    assert applied
    assert any(entry.section == "timeline" for entry in memory.entries.values())
    assert any(entry.section == "exact_values" for entry in memory.entries.values())


def test_practical_profile_keeps_subject_bound_latest_metric_without_exact_inventory():
    policy = get_memory_policy("practical")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)

    applied, rejected = updater.update(
        memory,
        [
            Turn(
                id=1,
                role="user",
                content="My application has dashboard API response time improved to 250ms.",
            )
        ],
    )

    assert rejected == []
    assert applied
    assert any(
        entry.section == "facts" and "250ms" in entry.text
        for entry in memory.entries.values()
    )
    assert all(entry.section != "exact_values" for entry in memory.entries.values())


def test_practical_profile_keeps_subject_bound_commit_total():
    policy = get_memory_policy("practical")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)

    applied, rejected = updater.update(
        memory,
        [
            Turn(
                id=1,
                role="user",
                content=(
                    "My repository has seen commits merged into the main branch, "
                    "which has now reached 165, and I want to review security."
                ),
            )
        ],
    )

    assert rejected == []
    assert applied
    assert any("165" in entry.text for entry in memory.entries.values())


def test_practical_profile_keeps_explicit_project_denial_in_question():
    policy = get_memory_policy("practical")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)

    applied, rejected = updater.update(
        memory,
        [
            Turn(
                id=1,
                role="user",
                content=(
                    "I've never integrated Flask-Login in this project. "
                    "Can you show me how?"
                ),
            )
        ],
    )

    assert rejected == []
    assert applied
    assert any(
        entry.section == "status_changes" and "never integrated Flask-Login" in entry.text
        for entry in memory.entries.values()
    )


def test_practical_prompt_does_not_include_eval_extraction_rules():
    policy = get_memory_policy("practical")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    system, _messages = updater._build_prompt(
        Memory(sections=PRACTICAL_SECTIONS, policy=policy),
        [Turn(id=1, role="user", content="How does memory work?")],
    )

    assert "PRACTICAL PROFILE" in system
    assert "For information extraction, keep granular" not in system
    assert "use up to 3-5 concise ops" not in system
    assert "Do not save generic assistant advice" in system


def test_memory_profile_loads_from_env(monkeypatch):
    monkeypatch.setenv("MEMORY_PROFILE", "eval")

    config = StructuredAgentConfig.from_env()

    assert config.memory_profile == "eval"
    assert get_memory_policy(config.memory_profile).name == "eval"


def test_unknown_memory_profile_is_rejected(monkeypatch):
    monkeypatch.setenv("MEMORY_PROFILE", "everything")

    with pytest.raises(ValueError, match="MEMORY_PROFILE"):
        StructuredAgentConfig.from_env()


def test_structured_builder_wires_practical_policy_to_components():
    middleware = build_structured_middleware(StructuredAgentConfig())

    assert middleware.policy.name == "practical"
    assert middleware.memory.policy is middleware.policy
    assert middleware.updater.policy is middleware.policy
    assert middleware.memory_selector.policy is middleware.policy
    assert {section.key for section in middleware.memory.sections} == {
        section.key for section in PRACTICAL_SECTIONS
    }


def test_practical_verifier_ignores_correction_words_in_ordinary_questions():
    policy = get_memory_policy("practical")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    verification = MemoryUpdateVerifier(policy=policy).verify(
        evicted_turns=[
            Turn(
                id=1,
                role="user",
                content="Is this actually how memory compaction works?",
            )
        ],
        applied_ops=[],
        rejected_ops=[],
        memory=memory,
    )

    assert verification.passed
