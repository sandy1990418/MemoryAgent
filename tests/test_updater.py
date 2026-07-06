import pytest

from memory_agent.memory import Memory
from memory_agent.sections import CHAT_SECTIONS
from memory_agent.transcript import Turn
from memory_agent.updater import MemoryUpdater, UpdateFailed
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


def test_transport_error_raises_update_failed():
    def raising_script(system, messages):
        raise RuntimeError("network down")

    updater = make_updater(raising_script)
    mem = Memory(sections=CHAT_SECTIONS)
    turns = [Turn(id=1, role="user", content="hello")]

    with pytest.raises(UpdateFailed):
        updater.update(mem, turns)
