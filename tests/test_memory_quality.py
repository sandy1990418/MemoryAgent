from memory_agent.core.models import MemoryValue, SubjectIdentity
from memory_agent.core.sections import PRACTICAL_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.policies.structured import get_memory_policy
from memory_agent.retrieval.quality import memory_quality_report
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def test_quality_report_labels_heuristics_and_exposes_requested_indicators():
    memory = Memory(sections=PRACTICAL_SECTIONS)
    memory.apply_ops([
        {"op": "ADD", "section": "goal", "text": "Goal: ship the release", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "User asked to", "provenance": [2]},
        {"op": "ADD", "section": "facts", "text": "Ongoing state: build is…", "provenance": [3]},
    ])
    report = memory_quality_report(memory)
    assert report.canonical.count == 1
    assert report.raw_request.count == 1
    assert report.incomplete.count == 1
    assert report.future_usefulness.label == "heuristic"


def test_canonical_validation_quarantines_sliced_or_incomplete_model_output():
    from memory_agent.core.transcript import Turn
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"ADD","section":"goal","text":"User asked to…","provenance":[1]}]'),
        sections=PRACTICAL_SECTIONS,
        policy=get_memory_policy("chat"),
        max_retries=0,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS)
    applied, rejected = updater.update(memory, [Turn(1, "user", "I need to ship the release")])
    assert applied == []
    assert rejected == []
    assert memory.entries == {}


def test_raw_request_is_canonicalized_to_goal_without_character_slicing():
    text = "User asked to ship a complete production release with verified artifacts"
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: f'[{{"op":"ADD","section":"goal","text":"{text}","provenance":[1]}}]'),
        sections=PRACTICAL_SECTIONS,
        policy=get_memory_policy("chat"),
        max_retries=0,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS)
    from memory_agent.core.transcript import Turn
    updater.update(memory, [Turn(1, "user", "I need to ship a complete production release")])
    assert [entry.text for entry in memory.entries.values()] == [
        "Goal: ship a complete production release with verified artifacts"
    ]


def test_equal_typed_state_dimension_supersedes_older_state():
    memory = Memory()
    identity = SubjectIdentity("chat", "release", "state")
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "Ongoing state: release is building", "provenance": [1], "subject_identity": identity, "value": MemoryValue("building")},
        {"op": "ADD", "section": "facts", "text": "Completed state: release shipped", "provenance": [2], "subject_identity": identity, "value": MemoryValue("shipped")},
    ])
    MemoryUpdater._consolidate_latest_subject_values(memory)
    active = [e.text for e in memory.entries.values() if e.status == "active"]
    assert active == [
        "Completed state: release shipped "
        "Value history (earliest→latest): building → shipped."
    ]


def test_generic_goal_identity_never_merges_distinct_goals():
    memory = Memory()
    identity = SubjectIdentity("chat", "goal", "goal", confidence=0.9)
    memory.apply_ops([
        {
            "op": "ADD",
            "section": "facts",
            "text": "Emergency fund goal is $2,000",
            "provenance": [1],
            "subject_identity": identity,
            "value": MemoryValue("2000", "$"),
        },
        {
            "op": "ADD",
            "section": "facts",
            "text": "Family car goal is $5,000",
            "provenance": [2],
            "subject_identity": identity,
            "value": MemoryValue("5000", "$"),
        },
    ])

    assert MemoryUpdater._consolidate_latest_subject_values(memory) == []
    assert len([entry for entry in memory.entries.values() if entry.status == "active"]) == 2


def test_lifecycle_value_history_survives_repeated_consolidation():
    memory = Memory()
    identity = SubjectIdentity("chat", "monthly book budget", "budget", confidence=0.9)
    for turn_id, value in ((1, "35"), (2, "50"), (3, "35")):
        memory.apply_ops([{
            "op": "ADD",
            "section": "facts",
            "text": f"Monthly book budget is ${value}",
            "provenance": [turn_id],
            "subject_identity": identity,
            "value": MemoryValue(value, "$"),
        }])
        MemoryUpdater._consolidate_latest_subject_values(memory)

    active = [entry for entry in memory.entries.values() if entry.status == "active"]
    assert len(active) == 1
    assert active[0].provenance == [1, 2, 3]
    assert active[0].text.endswith(
        "Value history (earliest→latest): $35 → $50 → $35."
    )


def test_non_latin_typed_state_lifecycle_consolidates_like_english():
    from memory_agent.core.transcript import Turn

    updater = MemoryUpdater(
        llm=ScriptedLLM(
            lambda *_: (_ for _ in ()).throw(
                AssertionError("explicit state lifecycle should not call the LLM")
            )
        ),
        sections=PRACTICAL_SECTIONS,
        policy=get_memory_policy("chat"),
        enable_llm_gate=True,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS)

    updater.update(
        memory,
        [
            Turn(1, "user", "專案是規劃中。"),
            Turn(2, "user", "它目前是進行中。"),
            Turn(3, "user", "它已經完成。"),
        ],
    )

    active = [entry for entry in memory.entries.values() if entry.status == "active"]
    assert len(active) == 1
    assert active[0].text == (
        "Ongoing state: State: 專案 is complete. "
        "Value history (earliest→latest): planned → active → complete."
    )
    assert active[0].provenance == [1, 2, 3]


def test_state_pronoun_does_not_resolve_across_update_batches():
    from memory_agent.core.transcript import Turn

    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=get_memory_policy("chat"),
        enable_llm_gate=True,
    )
    memory = Memory(sections=PRACTICAL_SECTIONS)

    updater.update(memory, [Turn(1, "user", "專案是規劃中。")])
    updater.update(memory, [Turn(2, "user", "它已經完成。")])

    active = [entry for entry in memory.entries.values() if entry.status == "active"]
    assert [entry.text for entry in active] == [
        "Ongoing state: State: 專案 is planned."
    ]
    assert active[0].provenance == [1]


def test_assistant_suggestion_cannot_become_user_decision():
    from memory_agent.core.transcript import Turn
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"ADD","section":"goal","text":"Assistant suggested choosing Redis","provenance":[1]}]'),
        sections=Memory().sections,
        max_retries=0,
    )
    memory = Memory()
    updater.update(memory, [Turn(1, "assistant", "I suggest choosing Redis")])
    assert memory.entries == {}
