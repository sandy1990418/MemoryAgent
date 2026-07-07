import pytest

from memory_agent.models.sections import AGENT_SECTIONS, CHAT_SECTIONS
from memory_agent.models.transcript import Turn
from memory_agent.structured.memory import Memory
from memory_agent.structured.updater import MemoryUpdater, UpdateFailed
from tests.fakes import ScriptedLLM


def make_updater(script):
    llm = ScriptedLLM(script)
    return MemoryUpdater(llm=llm, sections=CHAT_SECTIONS)


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


def test_exact_values_rule_is_included_in_prompt():
    updater = make_updater(lambda system, messages: '[{"op": "NOOP"}]')
    mem = Memory(sections=CHAT_SECTIONS)

    system, _messages = updater._build_prompt(
        mem,
        [Turn(id=1, role="user", content="The version is 1.2.3.")],
    )

    assert "MUST be captured verbatim in the exact_values section" in system


def test_exact_values_add_applies_with_agent_sections():
    mem = Memory(sections=AGENT_SECTIONS)

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
    updater = make_updater(lambda system, messages: next(responses))
    mem = Memory(sections=CHAT_SECTIONS)

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
    assert mem.entries["V2"].text == "2026-07-07"


def test_numeric_update_id_is_normalized_when_unique():
    responses = iter(
        [
            '[{"op": "ADD", "section": "decisions", "text": "ship MVP", "provenance": [1]}]',
            '[{"op": "UPDATE", "id": 1, "text": "ship MVP by Friday", "provenance": [2]}]',
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


def test_transport_error_raises_update_failed():
    def raising_script(system, messages):
        raise RuntimeError("network down")

    updater = make_updater(raising_script)
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [Turn(id=1, role="user", content="hello")]

    with pytest.raises(UpdateFailed):
        updater.update(mem, turns)
