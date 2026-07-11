import json

import pytest

from memory_agent.agents.structured import build_structured_middleware
from memory_agent.models.config import StructuredAgentConfig
from memory_agent.models.policy import get_memory_policy, is_chat_policy
from memory_agent.models.sections import EVAL_SECTIONS, PRACTICAL_SECTIONS
from memory_agent.models.transcript import Turn
from memory_agent.structured.memory import Memory
from memory_agent.structured.updater import MemoryUpdater
from memory_agent.structured.verifier import MemoryUpdateVerifier
from tests.fakes import ScriptedLLM
from tests.practical_cases import PRACTICAL_RETENTION_CASES


@pytest.mark.parametrize(
    "case",
    PRACTICAL_RETENTION_CASES,
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


def test_chat_profile_deterministically_keeps_stable_instruction_under_batch_cap():
    policy = get_memory_policy("chat")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: json.dumps([
            {"op": "ADD", "section": "facts", "text": f"Fact {index}", "provenance": [1]}
            for index in range(5)
        ])),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    updater.update(memory, [Turn(id=1, role="user", content="Always include version numbers when I ask about libraries.")])
    assert any(
        entry.section == "preferences" and "version numbers" in entry.text
        for entry in memory.entries.values()
    )


def test_chat_profile_does_not_treat_descriptive_always_as_instruction():
    policy = get_memory_policy("chat")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op":"NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    updater.update(memory, [Turn(id=1, role="user", content="How do I ensure this query always returns a list?")])
    assert not any(entry.section == "preferences" for entry in memory.entries.values())


def test_chat_profile_skips_updater_llm_for_non_durable_batch():
    calls = []
    policy = get_memory_policy("chat")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: calls.append(messages) or '[{"op":"NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    applied, rejected = updater.update(
        memory,
        [Turn(id=1, role="user", content="How does Redis persistence work?")],
    )
    assert applied == []
    assert rejected == []
    assert calls == []


def test_chat_profile_quarantines_oversized_llm_entry_instead_of_slicing():
    policy = get_memory_policy("chat")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: json.dumps([{
            "op": "ADD",
            "section": "facts",
            "text": "Project uses Flask " + "with detailed configuration " * 20,
            "provenance": [1],
        }])),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    updater.update(memory, [Turn(id=1, role="user", content="My project uses Flask.")])
    assert memory.entries == {}


def test_chat_entry_validation_does_not_slice_long_subject_value():
    text = (
        "User stated: trying to improve authentication and authorization with many "
        "security details and deployment constraints while the repository main branch "
        "has now reached 165 commits and requires a final review before launch."
    )
    canonical = MemoryUpdater._canonical_chat_entry_text(text, "facts")
    assert canonical == text
    assert "165 commits" in canonical


def test_chat_profile_locally_consolidates_near_duplicate_facts():
    policy = get_memory_policy("chat")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op":"ADD","section":"facts","text":"Dashboard API improved to 300ms after SQL optimization and caching.","provenance":[1]},
        {"op":"ADD","section":"facts","text":"Dashboard API improved to 250ms after SQL optimization and caching tweaks.","provenance":[2]},
    ])
    applied = MemoryUpdater._consolidate_near_duplicates(memory)
    assert applied
    active = [entry for entry in memory.entries.values() if entry.status == "active"]
    assert len(active) == 1
    assert "250ms" in active[0].text
    assert sorted(active[0].provenance) == [1, 2]


def test_chat_profile_does_not_mutate_untouched_legacy_subject_values():
    policy = get_memory_policy("chat")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op":"ADD","section":"facts","text":"The batch worker queue depth is 150 items.","provenance":[1]},
        {"op":"ADD","section":"facts","text":"The batch worker queue depth is 165 items.","provenance":[2]},
    ])
    applied = MemoryUpdater._consolidate_latest_subject_values(memory)
    assert applied == []
    active = [entry for entry in memory.entries.values() if entry.status == "active"]
    assert len(active) == 2
    assert all(entry.subject_identity is None for entry in active)


def test_latest_value_consolidation_keeps_distinct_latency_subjects():
    policy = get_memory_policy("chat")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "The search API latency is 120 ms.", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "The billing API latency is 80 ms.", "provenance": [2]},
    ])

    assert MemoryUpdater._consolidate_latest_subject_values(memory) == []
    assert len([entry for entry in memory.entries.values() if entry.status == "active"]) == 2


def test_latest_value_consolidation_keeps_conditional_preferences_separate():
    from memory_agent.models.memory import MemoryValue, SubjectIdentity

    policy = get_memory_policy("chat")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    memory.apply_ops([
        {
            "op": "ADD", "section": "preferences", "text": "Use 2 workers when offline.", "provenance": [1],
            "subject_identity": SubjectIdentity("chat", "worker count", "preference", "when offline"),
            "value": MemoryValue("2", "workers"),
        },
        {
            "op": "ADD", "section": "preferences", "text": "Use 8 workers when online.", "provenance": [2],
            "subject_identity": SubjectIdentity("chat", "worker count", "preference", "when online"),
            "value": MemoryValue("8", "workers"),
        },
    ])

    assert MemoryUpdater._consolidate_latest_subject_values(memory) == []
    assert len([entry for entry in memory.entries.values() if entry.status == "active"]) == 2


def test_low_confidence_identity_never_mutates_latest_value():
    from memory_agent.models.memory import MemoryValue, SubjectIdentity

    policy = get_memory_policy("chat")
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    for turn_id, value in ((1, "2"), (2, "8")):
        memory.apply_ops([{
            "op": "ADD", "section": "facts", "text": f"Possible pool size {value}.", "provenance": [turn_id],
            "subject_identity": SubjectIdentity("chat", "pool", "size", confidence=0.4),
            "value": MemoryValue(value, "workers"),
        }])

    assert MemoryUpdater._consolidate_latest_subject_values(memory, confidence_threshold=0.85) == []


def test_chat_profile_does_not_require_domain_specific_count_extraction():
    policy = get_memory_policy("chat")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op":"NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    updater.update(memory, [Turn(id=1, role="user", content="The service worker count updated to 12 workers.")])
    assert memory.entries == {}


def test_chat_profile_deterministically_keeps_implementation_state():
    policy = get_memory_policy("chat")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op":"NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS, policy=policy)
    updater.update(memory, [Turn(
        id=1,
        role="user",
        content="I'm trying to implement the homepage route, and I've managed to return static HTML.",
    )])
    assert any(
        entry.section == "facts" and "homepage route" in entry.text and "static HTML" in entry.text
        for entry in memory.entries.values()
    )


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


def test_practical_profile_does_not_require_domain_specific_count_extraction():
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
                        "My service worker count updated to 12 workers, "
                        "and I want to review security."
                ),
            )
        ],
    )

    assert rejected == []
    assert applied == []
    assert memory.entries == {}


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


def test_structured_builder_wires_chat_policy_to_components():
    middleware = build_structured_middleware(StructuredAgentConfig())

    assert middleware.policy.name == "chat"
    assert is_chat_policy(middleware.policy)
    assert middleware.compactor is not None
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
