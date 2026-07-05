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
