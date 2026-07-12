import pytest

from memory_agent.models.sections import (
    AGENT_SECTIONS,
    CHAT_SECTIONS,
    EXACT_VALUES,
    PRACTICAL_SECTIONS,
)
from memory_agent.models.policy import get_memory_policy
from memory_agent.models.transcript import Turn
from memory_agent.structured.memory import Memory
from memory_agent.structured.updater import MemoryUpdater, UpdateFailed
from tests.fakes import ScriptedLLM


def make_updater(script, **kwargs):
    llm = ScriptedLLM(script)
    return MemoryUpdater(llm=llm, sections=CHAT_SECTIONS, **kwargs)


def test_fenced_json_array_parsed_and_applied():
    response = (
        "```json\n"
        '[{"op": "ADD", "section": "decisions", "text": "use in-memory storage", '
        '"provenance": [3]}]\n'
        "```"
    )
    updater = make_updater(lambda system, messages: response)
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [Turn(id=3, role="user", content="DECISION: use in-memory storage")]

    applied, rejected = updater.update(mem, turns)

    assert rejected == []
    assert len(applied) == 1
    assert "D1" in mem.entries
    assert mem.entries["D1"].text == "use in-memory storage"


def test_section_prefix_is_normalized_to_section_key():
    response = '[{"op": "ADD", "section": "D", "text": "use postgres", "provenance": [1]}]'
    updater = make_updater(lambda system, messages: response)
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [Turn(id=1, role="user", content="DECISION: use postgres")]

    applied, rejected = updater.update(mem, turns)

    assert rejected == []
    assert len(applied) == 1
    assert mem.entries["D1"].section == "decisions"
    assert mem.entries["D1"].text == "use postgres"


def test_prefix_op_with_add_payload_is_normalized_to_add():
    response = '[{"op": "F", "text": "SQLite database is configured", "provenance": [1]}]'
    updater = make_updater(lambda system, messages: response)
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [Turn(id=1, role="user", content="SQLite database is configured")]

    applied, rejected = updater.update(mem, turns)

    assert rejected == []
    assert len(applied) == 1
    assert mem.entries["F1"].section == "facts"
    assert mem.entries["F1"].text == "SQLite database is configured"


def test_exact_values_are_not_in_default_prompt():
    updater = make_updater(lambda system, messages: '[{"op": "NOOP"}]')
    mem = Memory(sections=CHAT_SECTIONS)

    system, _messages = updater._build_prompt(
        mem,
        [Turn(id=1, role="user", content="The version is 1.2.3.")],
    )

    assert 'key="exact_values"' not in system
    assert "Do not create standalone memory entries just for isolated exact values" in system
    assert "value inventory" in system


def test_exact_values_add_applies_when_explicitly_configured():
    mem = Memory(sections=[*AGENT_SECTIONS, EXACT_VALUES])

    applied, rejected = mem.apply_ops_atomically(
        [
            {
                "op": "ADD",
                "section": "exact_values",
                "text": "Node version is 22.11.0",
                "provenance": [1],
            }
        ]
    )

    assert rejected == []
    assert len(applied) == 1
    assert mem.entries["V1"].section == "exact_values"
    assert mem.entries["V1"].text == "Node version is 22.11.0"


def test_exact_values_prefix_sections_are_normalized():
    responses = iter(
        [
            '[{"op": "ADD", "section": "V", "text": "/tmp/report.json", "provenance": [1]}]',
            '[{"op": "ADD", "section": "v", "text": "2026-07-07", "provenance": [2]}]',
        ]
    )
    sections = [*CHAT_SECTIONS, EXACT_VALUES]
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: next(responses)),
        sections=sections,
        max_retries=0,
        policy=get_memory_policy("eval"),
    )
    mem = Memory(sections=sections)

    applied, rejected = updater.update(
        mem,
        [Turn(id=1, role="user", content="The path is /tmp/report.json")],
    )
    assert rejected == []
    assert len(applied) == 1

    applied, rejected = updater.update(
        mem,
        [Turn(id=2, role="user", content="The date is 2026-07-07")],
    )
    assert rejected == []
    assert len(applied) == 1

    assert mem.entries["V1"].section == "exact_values"
    assert mem.entries["V1"].text == "/tmp/report.json"
    assert mem.entries["V2"].section == "exact_values"
    # Deterministic date extraction now prefixes the same-sentence subject.
    assert mem.entries["V2"].text == "The date is 2026-07-07"


def test_exact_values_are_not_extracted_by_default_when_llm_noops():
    response = '[{"op": "NOOP"}]'
    updater = make_updater(lambda system, messages: response)
    mem = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(
        mem,
        [
            Turn(
                id=1,
                role="user",
                content=(
                    "I'm using Flask 2.3.1, SQLite 3.39, and Bootstrap 5.3. "
                    "The homepage responds in 150ms by April 15, 2024."
                ),
            )
        ],
    )

    assert rejected == []
    assert applied == []
    assert mem.entries == {}


def test_exact_values_add_is_rejected_by_default_sections():
    response = (
        '[{"op": "ADD", "section": "exact_values", '
        '"text": "Flask 2.3.1", "provenance": [1]}]'
    )
    updater = make_updater(lambda system, messages: response)
    mem = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(
        mem,
        [Turn(id=1, role="user", content="The app uses Flask 2.3.1.")],
    )

    assert applied == []
    assert len(rejected) == 1
    assert mem.entries == {}


def test_status_change_snippets_are_extracted_with_agent_sections():
    response = '[{"op": "NOOP"}]'
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: response),
        sections=AGENT_SECTIONS,
    )
    mem = Memory(sections=AGENT_SECTIONS)

    applied, rejected = updater.update(
        mem,
        [
            Turn(
                id=58,
                role="user",
                content=(
                    "I've never written any Flask routes or handled HTTP requests "
                    "in this project, so I'm starting from scratch. ```python\n@app.route('/')\n```"
                ),
            )
        ],
    )

    assert rejected == []
    status_entries = [
        entry for entry in mem.entries.values() if entry.section == "status_changes"
    ]
    assert len(status_entries) == 1
    assert status_entries[0].text == (
        "User stated: I've never written any Flask routes or handled HTTP requests "
        "in this project, so I'm starting from scratch."
    )
    assert status_entries[0].provenance == [58]


def test_rejected_llm_batch_does_not_commit_deterministic_status_change():
    response = '[{"op": "UPDATE", "id": "C999", "text": "bad update", "provenance": [58]}]'
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: response),
        sections=AGENT_SECTIONS,
    )
    mem = Memory(sections=AGENT_SECTIONS)

    applied, rejected = updater.update(
        mem,
        [
            Turn(
                id=58,
                role="user",
                content=(
                    "I've never written any Flask routes or handled HTTP requests "
                    "in this project, so I'm starting from scratch."
                ),
            )
        ],
    )

    assert len(applied) == 1
    assert len(rejected) == 1
    status_entries = [
        entry for entry in mem.entries.values() if entry.section == "status_changes"
    ]
    assert status_entries == []


@pytest.mark.parametrize(
    "script",
    [
        lambda _system, _messages: (_ for _ in ()).throw(RuntimeError("network down")),
        lambda _system, _messages: "not JSON",
    ],
    ids=["transport", "parse"],
)
def test_failed_update_leaves_complete_live_state_unchanged(script):
    updater = MemoryUpdater(llm=ScriptedLLM(script), sections=AGENT_SECTIONS, max_retries=0)
    mem = Memory(sections=AGENT_SECTIONS)
    mem.apply_ops([{"op": "ADD", "section": "facts", "text": "existing", "provenance": [9]}])
    before = mem.to_state()

    with pytest.raises(UpdateFailed):
        updater.update(mem, [Turn(58, "user", "Actually, I have never deployed it.")])

    assert mem.to_state() == before


def test_retry_exhaustion_and_provenance_rejection_leave_live_state_unchanged():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"ADD","section":"facts","text":"bad","provenance":[999]}]'),
        sections=AGENT_SECTIONS,
        max_retries=1,
    )
    mem = Memory(sections=AGENT_SECTIONS)
    before = mem.to_state()

    _applied, rejected = updater.update(
        mem, [Turn(58, "user", "Actually, I have never deployed it.")]
    )

    assert rejected
    assert len(updater.token_reports) == 2
    assert mem.to_state() == before


def test_successful_prepared_update_commits_complete_state_once():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"ADD","section":"facts","text":"deployed","provenance":[1]}]'),
        sections=AGENT_SECTIONS,
    )
    mem = Memory(sections=AGENT_SECTIONS)
    prepared = updater.prepare_update(mem, [Turn(1, "user", "The project is deployed.")])
    assert mem.entries == {}
    assert prepared.trial_memory.entries

    prepared.commit(mem)
    committed = mem.to_state()
    with pytest.raises(RuntimeError, match="already committed"):
        prepared.commit(mem)
    assert mem.to_state() == committed


def test_prepared_update_rejects_commit_after_hidden_id_validation_failure():
    mem = Memory(sections=CHAT_SECTIONS)
    mem.apply_ops([
        {"op": "ADD", "section": "facts", "text": "visible deployment", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "hidden unrelated value", "provenance": [2]},
    ])
    updater = MemoryUpdater(
        llm=ScriptedLLM(
            lambda *_: '[{"op":"SUPERSEDE","id":"F2","reason":"hidden"}]'
        ),
        sections=CHAT_SECTIONS,
        max_retries=0,
        max_candidate_entries=1,
    )
    before = mem.to_state()

    prepared = updater.prepare_update(
        mem, [Turn(3, "user", "The deployment status changed.")]
    )

    assert prepared.rejected_ops[0]["reason"] == (
        "UPDATE/SUPERSEDE id was not visible to updater"
    )
    with pytest.raises(RuntimeError, match="rejected operations"):
        prepared.commit(mem)
    assert mem.to_state() == before


def test_prepared_update_detects_intervening_live_memory_change():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"ADD","section":"facts","text":"trial","provenance":[1]}]'),
        sections=CHAT_SECTIONS,
    )
    mem = Memory(sections=CHAT_SECTIONS)
    prepared = updater.prepare_update(mem, [Turn(1, "user", "Remember the trial fact.")])
    mem.apply_ops([{"op": "ADD", "section": "facts", "text": "concurrent", "provenance": [2]}])

    with pytest.raises(RuntimeError, match="memory changed"):
        prepared.commit(mem)

    assert [entry.text for entry in mem.entries.values()] == ["concurrent"]


def test_numeric_update_id_is_rejected_to_avoid_turn_id_confusion():
    responses = iter(
        [
            '[{"op": "ADD", "section": "decisions", "text": "ship MVP", "provenance": [1]}]',
            '[{"op": "UPDATE", "id": 1, "text": "ship MVP by Friday", "provenance": [2]}]',
        ]
    )
    updater = make_updater(lambda system, messages: next(responses), max_retries=0)
    mem = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(mem, [Turn(id=1, role="user", content="ship MVP")])
    assert rejected == []
    assert len(applied) == 1

    applied, rejected = updater.update(
        mem,
        [Turn(id=2, role="user", content="ship MVP by Friday")],
    )

    assert applied == []
    assert len(rejected) == 1
    assert rejected[0]["reason"] == (
        "unknown memory entry id: 1; UPDATE/SUPERSEDE ids must be exact current "
        "memory entry ids like F1, U2, or G3, not turn_id values"
    )
    assert mem.entries["D1"].text == "ship MVP"


def test_numeric_supersede_id_is_rejected_to_avoid_turn_id_confusion():
    response = '[{"op": "SUPERSEDE", "id": 3, "reason": "new preference statement"}]'
    updater = make_updater(lambda system, messages: response, max_retries=0)
    mem = Memory(sections=CHAT_SECTIONS)
    mem.apply_ops(
        [
            {
                "op": "ADD",
                "section": "preferences",
                "text": "User prefers terse answers",
                "provenance": [1],
            }
        ]
    )

    applied, rejected = updater.update(
        mem,
        [Turn(id=3, role="user", content="I prefer pragmatic security enhancements.")],
    )

    assert applied == []
    assert len(rejected) == 1
    assert "not turn_id values" in rejected[0]["reason"]
    assert mem.entries["U1"].status == "active"


def test_rejected_ops_are_retried_with_validation_feedback():
    responses = iter(
        [
            '[{"op": "SUPERSEDE", "id": "U28", "reason": "hallucinated id"}]',
            (
                '[{"op": "ADD", "section": "preferences", '
                '"text": "User wants pragmatic security best practices for auth.", '
                '"provenance": [185]}]'
            ),
        ]
    )
    captured_messages = []

    def script(system, messages):
        captured_messages.append(messages)
        return next(responses)

    updater = make_updater(script)
    mem = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(
        mem,
        [
            Turn(
                id=185,
                role="user",
                content="Always provide pragmatic security best practices for auth.",
            )
        ],
    )

    assert rejected == []
    assert len(applied) == 1
    assert "pragmatic security best practices for auth" in mem.entries["U1"].text
    assert len(captured_messages) == 2
    assert "rejected by validation" in captured_messages[1][-1]["content"]


def test_numeric_string_provenance_is_normalized_to_turn_ids():
    updater = make_updater(
        lambda system, messages: (
            '[{"op":"ADD","section":"facts","text":"Project uses Flask.",'
            '"provenance":["185"]}]'
        )
    )
    memory = Memory(sections=CHAT_SECTIONS)
    applied, rejected = updater.update(
        memory,
        [Turn(id=185, role="user", content="My project uses Flask.")],
    )
    assert rejected == []
    assert applied
    assert memory.entries["F1"].provenance == [185]


def test_exact_string_update_id_still_updates_entry():
    responses = iter(
        [
            '[{"op": "ADD", "section": "decisions", "text": "ship MVP", "provenance": [1]}]',
            '[{"op": "UPDATE", "id": "D1", "text": "ship MVP by Friday", "provenance": [2]}]',
        ]
    )
    updater = make_updater(lambda system, messages: next(responses))
    mem = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(mem, [Turn(id=1, role="user", content="ship MVP")])
    assert rejected == []
    assert len(applied) == 1

    applied, rejected = updater.update(
        mem,
        [Turn(id=2, role="user", content="ship MVP by Friday")],
    )

    assert rejected == []
    assert len(applied) == 1
    assert mem.entries["D1"].text == "ship MVP by Friday"


def test_unhashable_update_id_is_rejected_without_crashing():
    response = '[{"op": "UPDATE", "id": [1], "text": "bad id", "provenance": [2]}]'
    updater = make_updater(lambda system, messages: response)
    mem = Memory(sections=CHAT_SECTIONS)
    mem.apply_ops([{"op": "ADD", "section": "facts", "text": "existing", "provenance": [1]}])

    applied, rejected = updater.update(mem, [Turn(id=2, role="user", content="bad id")])

    assert applied == []
    assert len(rejected) == 1
    assert mem.entries["F1"].text == "existing"


def test_text_suffix_turns_are_normalized_to_provenance():
    response = '[{"op": "ADD", "section": "facts", "text": "SQLite is configured. (turns 2-3)"}]'
    updater = make_updater(lambda system, messages: response)
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [
        Turn(id=2, role="user", content="SQLite"),
        Turn(id=3, role="assistant", content="Configured"),
    ]

    applied, rejected = updater.update(mem, turns)

    assert rejected == []
    assert len(applied) == 1
    assert mem.entries["F1"].text == "SQLite is configured."
    assert mem.entries["F1"].provenance == [2, 3]


def test_text_suffix_turns_are_stripped_even_when_provenance_exists():
    response = (
        '[{"op": "ADD", "section": "facts", '
        '"text": "Chart.js is configured. (turns 7)", "provenance": [7]}]'
    )
    updater = make_updater(lambda system, messages: response)
    mem = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(
        mem,
        [Turn(id=7, role="user", content="Chart.js is configured")],
    )

    assert rejected == []
    assert len(applied) == 1
    assert mem.entries["F1"].text == "Chart.js is configured."
    assert mem.entries["F1"].provenance == [7]


def test_assistant_only_unaccepted_proposal_is_not_stored():
    updater = make_updater(
        lambda *_: (
            '[{"op":"ADD","section":"decisions","text":"Use Redis",'
            '"provenance":[2]}]'
        )
    )
    memory = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(
        memory,
        [
            Turn(1, "user", "How could caching work?"),
            Turn(2, "assistant", "I propose using Redis."),
        ],
    )

    assert rejected == []
    assert applied == []
    assert memory.entries == {}
    assert updater.update_token_usage()["write_suppression_reasons"] == {
        "assistant_only_proposal": 1
    }


def test_accepted_assistant_proposal_can_be_stored_with_assistant_provenance():
    updater = make_updater(
        lambda *_: pytest.fail("pure accepted proposal should not call the LLM"),
        enable_llm_gate=True,
    )
    memory = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(
        memory,
        [
            Turn(1, "assistant", "I propose using Redis."),
            Turn(2, "user", "Yes, let's do that."),
        ],
    )

    assert rejected == []
    assert len(applied) == 1
    assert memory.entries["D1"].text == "Accepted strategy: using Redis"
    assert memory.entries["D1"].provenance == [1, 2]
    assert updater.update_token_usage()["calls"] == 0


@pytest.mark.parametrize(
    ("proposal", "resolution", "expected"),
    [
        ("I propose PostgreSQL.", "Yes, go with that.", "Accepted strategy: PostgreSQL"),
        ("I propose Redis.", "No, reject that proposal.", "Rejected proposal: Redis"),
        ("我建議採用 PostgreSQL。", "同意，就這樣。", "Accepted strategy: PostgreSQL。"),
    ],
)
def test_pure_proposal_resolution_is_deterministic_without_llm(
    proposal, resolution, expected
):
    updater = make_updater(
        lambda *_: pytest.fail("pure proposal resolution should not call the LLM"),
        enable_llm_gate=True,
    )
    memory = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(
        memory,
        [Turn(1, "assistant", proposal), Turn(2, "user", resolution)],
    )

    assert rejected == []
    assert len(applied) == 1
    assert memory.entries["D1"].text == expected
    assert memory.entries["D1"].provenance == [1, 2]
    assert updater.update_token_usage()["calls"] == 0


def test_acceptance_with_an_additional_assertion_still_calls_llm():
    calls = []

    def respond(system, messages):
        calls.append((system, messages))
        return '[{"op":"NOOP"}]'

    updater = make_updater(respond, enable_llm_gate=True)
    memory = Memory(sections=CHAT_SECTIONS)

    updater.update(
        memory,
        [
            Turn(1, "assistant", "I propose PostgreSQL."),
            Turn(2, "user", "Yes, go with that. My deployment remains blocked."),
        ],
    )

    assert len(calls) == 1
    assert updater.update_token_usage()["decision_reasons"] == {
        "call:user_acceptance_ambiguous": 1
    }


def test_explicit_project_work_inside_a_question_is_retained_without_llm():
    updater = make_updater(
        lambda *_: pytest.fail("deterministic project work should not call the LLM"),
        enable_llm_gate=True,
    )
    memory = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(
        memory,
        [
            Turn(
                1,
                "user",
                "I'm trying to implement password hashing with Werkzeug.security, "
                "but I'm not sure how to verify passwords correctly. Can you help?",
            ),
            Turn(2, "assistant", "Use check_password_hash."),
        ],
    )

    assert rejected == []
    assert len(applied) == 1
    assert memory.entries["F1"].text == (
        "Ongoing state: I'm trying to implement password hashing with "
        "Werkzeug.security, but I'm not sure how to verify passwords correctly."
    )
    assert updater.update_token_usage()["calls"] == 0


def test_project_work_retention_is_bounded_to_newest_event_per_batch():
    calls = []

    def respond(system, messages):
        calls.append((system, messages))
        return '[{"op":"NOOP"}]'

    updater = make_updater(respond, enable_llm_gate=True)
    memory = Memory(sections=CHAT_SECTIONS)

    updater.update(
        memory,
        [
            Turn(1, "user", "I'm trying to implement password hashing. Can you help?"),
            Turn(2, "assistant", "Yes."),
            Turn(
                3,
                "user",
                "I'm working on project documentation in Confluence with API tables "
                "and architecture diagrams. Can you review it?",
            ),
            Turn(4, "assistant", "Yes."),
        ],
    )

    active = [entry for entry in memory.entries.values() if entry.status == "active"]
    assert len(active) == 1
    assert "Confluence" in active[0].text
    assert active[0].provenance == [3]
    assert len(calls) == 1


def test_generic_assistant_intro_is_not_treated_as_a_rejected_proposal():
    updater = make_updater(
        lambda *_: pytest.fail("non-durable rejection should not call the LLM"),
        enable_llm_gate=True,
    )
    memory = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(
        memory,
        [
            Turn(1, "assistant", "Certainly! Let's walk through the error."),
            Turn(2, "user", "No, that did not solve it."),
        ],
    )

    assert applied == []
    assert rejected == []
    assert memory.entries == {}


def test_exact_restatement_from_same_source_is_suppressed_before_write():
    updater = make_updater(
        lambda *_: (
            '[{"op":"ADD","section":"facts","text":"Project uses SQLite",'
            '"provenance":[1]}]'
        )
    )
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops_atomically([
        {
            "op": "ADD",
            "section": "facts",
            "text": "Project uses SQLite",
            "provenance": [1],
        }
    ])

    applied, rejected = updater.update(
        memory, [Turn(1, "user", "My project uses SQLite")]
    )

    assert rejected == []
    assert applied == []
    assert len(memory.entries) == 1
    assert updater.update_token_usage()["write_suppression_reasons"] == {
        "redundant_add": 1
    }


def test_garbage_response_raises_update_failed():
    updater = make_updater(lambda system, messages: "this is not json at all, sorry")
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [Turn(id=1, role="user", content="hello")]

    with pytest.raises(UpdateFailed):
        updater.update(mem, turns)


def test_mixed_valid_and_invalid_ops_rejected_atomically():
    response = (
        '[{"op": "ADD", "section": "decisions", "text": "valid decision", "provenance": [1]}, '
        '{"op": "ADD", "section": "not_a_section", "text": "invalid", "provenance": [1]}, '
        '{"op": "UPDATE", "id": "NOPE", "text": "invalid update", "provenance": [1]}]'
    )
    updater = make_updater(lambda system, messages: response)
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [Turn(id=1, role="user", content="something")]

    applied, rejected = updater.update(mem, turns)

    assert applied == []
    assert len(rejected) == 2
    assert mem.entries == {}


def test_provenance_must_reference_evicted_turns():
    response = (
        '[{"op": "ADD", "section": "decisions", "text": "valid decision", '
        '"provenance": [999]}]'
    )
    updater = make_updater(lambda system, messages: response)
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [Turn(id=1, role="user", content="something")]

    applied, rejected = updater.update(mem, turns)

    assert applied == []
    assert len(rejected) == 1
    assert mem.entries == {}


def test_prompt_requires_supersede_add_for_reversals():
    captured = {}

    def script(system, messages):
        captured["system"] = system
        return '[{"op": "NOOP"}]'

    updater = make_updater(script)
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [Turn(id=1, role="user", content="I changed my mind about the answer style.")]

    updater.update(mem, turns)

    system = captured["system"]
    assert "MUST SUPERSEDE the old active entry" in system
    assert "then ADD a new replacement entry" in system
    assert "Never use UPDATE for that case" in system
    assert "Never use a turn_id" in system
    assert '{"id": 3} is always invalid' in system
    assert "If no exact current entry id exists, use ADD instead" in system
    assert "If Current memory has no exact conflicting active entry id" in system


def test_prompt_guards_against_cross_subject_supersede_and_duplicates():
    captured = {}

    def script(system, messages):
        captured["system"] = system
        return '[{"op": "NOOP"}]'

    updater = make_updater(script)
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [Turn(id=1, role="user", content="Always include dependency versions.")]

    updater.update(mem, turns)

    system = captured["system"]
    assert "same semantic subject" in system
    assert "Do not supersede identity or background facts" in system
    assert "use UPDATE to merge/refine it instead of adding a duplicate" in system
    assert "dependency/version-number preferences" in system


def test_prompt_includes_memory_quality_rules():
    captured = {}

    def script(system, messages):
        captured["system"] = system
        return '[{"op": "NOOP"}]'

    updater = make_updater(script)
    mem = Memory(sections=CHAT_SECTIONS)

    updater.update(mem, [Turn(id=1, role="user", content="How do I add an index?")])

    system = captured["system"]
    assert "Keep entries concise but aggregated" in system
    assert "return 0-2 durable ops total" in system
    assert "Prefer one consolidated entry per subject" in system
    assert "Do not save generic assistant advice" in system
    assert "Do not infer missing details" in system
    assert "Do not turn every user request into an open question" in system
    assert "do not split them into a separate value inventory" in system
    assert "assistant directly creates a plan, schedule, milestone breakdown" in system
    assert "For information extraction, keep granular subject-bound facts" in system
    assert "For temporal reasoning, every explicit dated event" in system
    assert "For knowledge updates, keep the latest value active" in system
    assert "For cross-session counting, keep compact aggregate lists" in system
    assert "Use status_changes for explicit contradictions" in system
    assert "I never" in system
    assert "Use timeline only for explicitly stated dated or staged milestones" in system


def test_transport_error_raises_update_failed():
    def raising_script(system, messages):
        raise RuntimeError("network down")

    updater = make_updater(raising_script)
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [Turn(id=1, role="user", content="hello")]

    with pytest.raises(UpdateFailed):
        updater.update(mem, turns)


@pytest.mark.parametrize("first", ["transport", "parse"])
def test_transport_and_parse_failures_retry_inside_one_transaction(first):
    responses = iter([
        RuntimeError("temporary outage") if first == "transport" else "not JSON",
        '[{"op":"ADD","section":"facts","text":"retry succeeded","provenance":[1]}]',
    ])

    def script(_system, _messages):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    updater = MemoryUpdater(
        llm=ScriptedLLM(script), sections=CHAT_SECTIONS, max_retries=1
    )
    mem = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(mem, [Turn(1, "user", "Remember this result.")])

    assert rejected == []
    assert len(applied) == 1
    assert [entry.text for entry in mem.entries.values()] == ["retry succeeded"]
    assert len(updater.token_reports) == 2
    assert updater.token_reports[0].rejected_ops_count == 1


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "I joined a winter reading challenge aiming for 10 books by March 1.",
            "10 books",
        ),
        (
            "I spent $43 this month, which is over my $35 monthly book budget.",
            "$35 monthly book budget",
        ),
        (
            "The probate process now takes 5-7 months.",
            "5-7 months",
        ),
        (
            "The estate tax rate is 12% on assets above $200,000.",
            "12% on assets above $200,000",
        ),
    ],
)
def test_general_subject_bound_personal_values_are_retained(content, expected):
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        sections=CHAT_SECTIONS,
        policy=get_memory_policy("chat"),
        enable_llm_gate=True,
    )
    mem = Memory(sections=CHAT_SECTIONS, policy=get_memory_policy("chat"))

    updater.update(mem, [Turn(1, "user", content)])

    assert any(expected in entry.text for entry in mem.entries.values())


def test_counted_noun_values_are_retained_without_a_fixed_unit_vocabulary():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        sections=CHAT_SECTIONS,
        policy=get_memory_policy("chat"),
        enable_llm_gate=True,
    )
    mem = Memory(sections=CHAT_SECTIONS, policy=get_memory_policy("chat"))

    updater.update(
        mem,
        [Turn(1, "user", "My Zotero library now has 52 sources after the import.")],
    )

    assert any("52 sources" in entry.text for entry in mem.entries.values())


def test_conversational_frame_is_trimmed_but_values_and_cues_survive():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        sections=CHAT_SECTIONS,
        policy=get_memory_policy("chat"),
        enable_llm_gate=True,
    )
    mem = Memory(sections=CHAT_SECTIONS, policy=get_memory_policy("chat"))

    updater.update(
        mem,
        [
            Turn(
                1,
                "user",
                "I'm kinda worried that I spent $43 on books in January, which is "
                "$8 over my $35 monthly budget, can you help me find a way to cut back?",
            )
        ],
    )

    texts = [entry.text for entry in mem.entries.values()]
    assert any("$35 monthly budget" in text for text in texts)
    assert all("can you help" not in text for text in texts)
    assert all("worried" not in text for text in texts)


def test_trailing_request_clause_with_a_value_is_never_trimmed():
    from memory_agent.structured.heuristics import trim_conversational_frame

    text = (
        "The probate process was shortened, so can you tell me how the 12% rate "
        "on assets above $200,000 affects my planning"
    )
    assert trim_conversational_frame(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "The release is ready, but can you help without changing Project Atlas?",
        "The release is ready, but can you help with ACME?",
        "The release is ready, but can you help with the plan we approved?",
        "專案已完成，但你可以幫忙嗎？必須保留客戶的限制。",
    ],
)
def test_conversational_trimming_preserves_material_request_clauses(text):
    from memory_agent.structured.heuristics import trim_conversational_frame

    assert trim_conversational_frame(text) == text


def test_conversational_trimming_still_removes_non_material_request_clause():
    from memory_agent.structured.heuristics import trim_conversational_frame

    text = "The release is ready, can you help me think through it?"
    assert trim_conversational_frame(text) == "The release is ready"


def test_conversational_trimming_removes_emotion_and_question_around_durable_state():
    from memory_agent.structured.heuristics import trim_conversational_frame

    text = "I'm feeling worried that Project Atlas is blocked, what should I do next?"
    assert trim_conversational_frame(text) == "Project Atlas is blocked"


def test_exact_value_shapes_are_domain_neutral_and_bounded():
    values = MemoryUpdater._extract_exact_values(
        "AcmeDB 4.7.2 raised CustomQuotaError: limit with 12 workers."
    )

    assert "AcmeDB 4.7.2" in values
    assert "CustomQuotaError: limit" in values
    assert "12 workers" in values
    assert MemoryUpdater._extract_exact_values("We discussed version planning.") == []


def test_latest_typed_personal_budget_supersedes_older_value():
    policy = get_memory_policy("chat")
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        sections=PRACTICAL_SECTIONS,
        policy=policy,
        enable_llm_gate=True,
    )
    mem = Memory(sections=PRACTICAL_SECTIONS, policy=policy)

    updater.update(mem, [Turn(1, "user", "My monthly book budget is $35.")])
    updater.update(mem, [Turn(2, "user", "My monthly book budget is now $50.")])

    active = [entry for entry in mem.entries.values() if entry.status == "active"]
    assert len(active) == 1
    assert "$50" in active[0].text
    assert active[0].provenance == [1, 2]
    assert active[0].subject_identity is not None
    assert active[0].value.value == "50"


def test_status_change_snippet_truncation_keeps_late_cue_phrase():
    """A cue buried past char 170 of a run-on sentence must survive truncation.

    Regression: turn 108 of the BEAM 100K case buries "never actually
    integrated" ~150 chars into one long sentence; a head-anchored cut
    dropped the denial and stored only the sentence's unrelated opening.
    """
    padding = (
        "I'm trying to optimize the dashboard API response time, which has "
        "recently improved to 250ms after adding some caching tweaks, but I "
        "want to make sure I'm using the latest versions of my dependencies, "
        "like Flask-Login, which I've never actually integrated into this "
        "project, so I'm starting from scratch - can you help me implement "
        "user session management with Flask-Login 0.6.2"
    )
    snippet = MemoryUpdater._extract_status_change_snippet(padding)

    assert snippet is not None
    assert "never" in snippet
    assert "Flask-Login" in snippet


def _add_fact(mem: Memory, text: str, provenance: list[int]) -> None:
    applied, rejected = mem.apply_ops(
        [{"op": "ADD", "section": "facts", "text": text, "provenance": provenance}]
    )
    assert rejected == []


def test_update_context_selects_overlapping_and_always_sections():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
        sections=AGENT_SECTIONS,
        update_context_max_entries=3,
    )
    mem = Memory(sections=AGENT_SECTIONS)
    mem.apply_ops(
        [
            {
                "op": "ADD",
                "section": "timeline",
                "text": "Final deployment deadline is April 15, 2024.",
                "provenance": [9],
            },
            {
                "op": "ADD",
                "section": "preferences",
                "text": "User prefers terse answers.",
                "provenance": [2],
            },
        ]
    )
    _add_fact(mem, "The weather tool returns JSON payloads.", [30])
    _add_fact(mem, "Bootstrap grid uses twelve columns.", [31])

    turns = [
        Turn(
            id=100,
            role="user",
            content="I moved the final deployment deadline to March 15, 2024.",
        )
    ]
    selected = updater._select_update_context_entries(mem, turns)
    texts = [entry.text for entry in selected]

    assert "User prefers terse answers." in texts  # always-context section
    assert "Final deployment deadline is April 15, 2024." in texts  # overlap
    assert "The weather tool returns JSON payloads." not in texts
    assert "Bootstrap grid uses twelve columns." not in texts


def test_build_prompt_uses_targeted_context_when_memory_is_large():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
        sections=AGENT_SECTIONS,
        update_context_max_entries=2,
    )
    mem = Memory(sections=AGENT_SECTIONS)
    mem.apply_ops(
        [
            {
                "op": "ADD",
                "section": "timeline",
                "text": "Final deployment deadline is April 15, 2024.",
                "provenance": [9],
            }
        ]
    )
    _add_fact(mem, "The weather tool returns JSON payloads.", [30])
    _add_fact(mem, "Bootstrap grid uses twelve columns.", [31])

    turns = [
        Turn(
            id=100,
            role="user",
            content="I moved the final deployment deadline to March 15, 2024.",
        )
    ]
    system, _messages = updater._build_prompt(mem, turns)

    assert "Final deployment deadline is April 15, 2024." in system
    assert "The weather tool returns JSON payloads." not in system


def test_build_prompt_renders_full_memory_when_small():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
        sections=AGENT_SECTIONS,
        update_context_max_entries=40,
    )
    mem = Memory(sections=AGENT_SECTIONS)
    _add_fact(mem, "The weather tool returns JSON payloads.", [30])

    turns = [Turn(id=100, role="user", content="Unrelated topic entirely.")]
    system, _messages = updater._build_prompt(mem, turns)

    assert "The weather tool returns JSON payloads." in system


def test_subject_bound_dates_are_extracted_to_timeline_with_agent_sections():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
        sections=AGENT_SECTIONS,
    )
    mem = Memory(sections=AGENT_SECTIONS)

    applied, rejected = updater.update(
        mem,
        [
            Turn(
                id=1,
                role="user",
                content=(
                    "Great progress so far. The final deployment deadline is "
                    "March 15, 2024."
                ),
            )
        ],
    )

    assert rejected == []
    assert len(applied) == 1
    entry = mem.entries["M1"]
    assert entry.section == "timeline"
    assert "final deployment deadline" in entry.text
    assert "March 15, 2024" in entry.text


def test_subject_bound_values_are_extracted_to_progress_with_agent_sections():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
        sections=AGENT_SECTIONS,
    )
    mem = Memory(sections=AGENT_SECTIONS)

    applied, rejected = updater.update(
        mem,
        [
            Turn(
                id=1,
                role="user",
                content="API integration test coverage improved to 78% after adding 401 tests.",
            )
        ],
    )

    assert rejected == []
    assert len(applied) == 1
    entry = mem.entries["P1"]
    assert entry.section == "progress"
    assert "78%" in entry.text
    assert "API integration test coverage" in entry.text


def test_technical_subject_bound_value_extraction_stays_disabled_for_simple_chat_sections():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, messages: '[{"op": "NOOP"}]'),
        sections=CHAT_SECTIONS,
    )
    mem = Memory(sections=CHAT_SECTIONS)

    applied, rejected = updater.update(
        mem,
        [
            Turn(
                id=1,
                role="user",
                content="API integration test coverage improved to 78% after adding 401 tests.",
            )
        ],
    )

    assert rejected == []
    assert applied == []
    assert mem.entries == {}
