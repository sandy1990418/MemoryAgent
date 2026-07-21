"""Structural and token-safety contracts for the chat updater."""

import pytest

from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.update.operations import UpdateFailed
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def _memory() -> Memory:
    return Memory(sections=CHAT_SECTIONS, policy=CHAT_POLICY)


def _updater(response, **kwargs) -> MemoryUpdater:
    return MemoryUpdater(
        llm=ScriptedLLM(lambda *_: response),
        sections=CHAT_SECTIONS,
        policy=CHAT_POLICY,
        **kwargs,
    )


def test_fenced_json_array_is_parsed_and_applied():
    updater = _updater(
        "```json\n"
        '[{"op":"ADD","section":"facts","text":"SQLite database is configured",'
        '"provenance":[1]}]\n```'
    )
    memory = _memory()

    applied, rejected = updater.update(
        memory,
        [Turn(1, "user", "The SQLite database is configured.")],
    )

    assert rejected == []
    assert applied == [
        {
            "op": "ADD",
            "section": "facts",
            "text": "SQLite database is configured",
            "provenance": [1],
        }
    ]
    assert memory.entries["F1"].text == "SQLite database is configured"


def test_chat_updater_accepts_model_selected_assistant_provenance():
    updater = _updater(
        '[{"op":"ADD","section":"progress","text":"Discussion covered two approaches.",'
        '"provenance":[1,2]}]'
    )
    memory = _memory()

    applied, rejected = updater.update(
        memory,
        [Turn(1, "user", "Compare two approaches."), Turn(2, "assistant", "Approach A is faster.")],
    )

    assert rejected == []
    assert applied[0]["provenance"] == [1, 2]
    assert memory.entries["P1"].section == "progress"


def test_empty_batch_skips_llm_call():
    calls = []
    updater = _updater("[]")
    updater.llm = ScriptedLLM(lambda *_: calls.append(True) or "[]")

    assert updater.update(_memory(), []) == ([], [])
    assert calls == []
    assert updater.decision_reasons["skip:empty_batch"] == 1


def test_unknown_section_is_structurally_rejected_without_commit():
    updater = _updater(
        '[{"op":"ADD","section":"timeline","text":"Release date is fixed",'
        '"provenance":[1]}]'
    )
    memory = _memory()

    applied, rejected = updater.update(
        memory,
        [Turn(1, "user", "The release date is fixed.")],
    )

    assert applied == []
    assert rejected
    assert "disallowed section" in rejected[0]["reason"]
    assert memory.entries == {}


def test_oversized_entry_is_rejected_without_slicing():
    text = "important " * 80
    updater = _updater("[]")
    updater.llm = ScriptedLLM(
        lambda *_: (
            '[{"op":"ADD","section":"facts","text":'
            + repr(text).replace("'", '"')
            + ',"provenance":[1]}]'
        )
    )
    memory = _memory()

    applied, rejected = updater.update(memory, [Turn(1, "user", "Remember this.")])

    assert applied == []
    assert rejected
    assert "exceeds 500 characters" in rejected[0]["reason"]
    assert memory.entries == {}


def test_invalid_provenance_is_rejected_atomically():
    updater = _updater(
        '[{"op":"ADD","section":"facts","text":"valid", "provenance":[1]},'
        '{"op":"ADD","section":"facts","text":"invalid", "provenance":[99]}]'
    )
    memory = _memory()

    applied, rejected = updater.update(memory, [Turn(1, "user", "A durable fact.")])

    assert applied == []
    assert rejected
    assert memory.entries == {}


def test_numeric_update_and_supersede_ids_are_rejected():
    updater = _updater(
        '[{"op":"UPDATE","id":1,"section":"facts","text":"changed", "provenance":[1]}]'
    )
    memory = _memory()
    memory.apply_ops([{"op": "ADD", "section": "facts", "text": "original", "provenance": [1]}])

    applied, rejected = updater.update(memory, [Turn(1, "user", "Change it.")])

    assert applied == []
    assert rejected
    assert memory.entries["F1"].text == "original"


def test_reversal_requires_visible_active_id_and_commits_atomically():
    memory = _memory()
    memory.apply_ops([{"op": "ADD", "section": "decisions", "text": "Use Redis.", "provenance": [1]}])
    updater = _updater(
        '[{"op":"SUPERSEDE","id":"D1","reason":"Decision changed"},'
        '{"op":"ADD","section":"decisions","text":"Use Postgres.","provenance":[2]}]'
    )

    applied, rejected = updater.update(memory, [Turn(2, "user", "Use Postgres instead.")])

    assert rejected == []
    assert [op["op"] for op in applied] == ["SUPERSEDE", "ADD"]
    assert memory.entries["D1"].status == "superseded"
    assert memory.entries["D2"].text == "Use Postgres."


def test_structural_batch_accepts_all_valid_operations():
    updater = _updater(
        "["
        '{"op":"ADD","section":"preferences","text":"A","provenance":[1]},'
        '{"op":"ADD","section":"facts","text":"B","provenance":[1]},'
        '{"op":"ADD","section":"goal","text":"C","provenance":[1]},'
        '{"op":"ADD","section":"facts","text":"D","provenance":[1]}'
        "]"
    )
    applied, rejected = updater.update(_memory(), [Turn(1, "user", "Several durable facts.")])

    assert rejected == []
    assert len(applied) == 4


def test_update_retries_transport_failure_inside_one_transaction():
    responses = iter(
        [RuntimeError("temporary outage"), '[{"op":"ADD","section":"facts",'
         '"text":"retry succeeded","provenance":[1]}]']
    )

    def script(*_):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    updater = MemoryUpdater(
        llm=ScriptedLLM(script),
        sections=CHAT_SECTIONS,
        policy=CHAT_POLICY,
        max_retries=1,
    )
    memory = _memory()

    applied, rejected = updater.update(memory, [Turn(1, "user", "Retry this fact.")])

    assert rejected == []
    assert applied[0]["text"] == "retry succeeded"
    assert memory.entries["F1"].text == "retry succeeded"


def test_transport_failure_raises_after_retry_exhaustion():
    updater = _updater("[]", max_retries=0)
    updater.llm = ScriptedLLM(lambda *_: (_ for _ in ()).throw(RuntimeError("network down")))

    with pytest.raises(UpdateFailed, match="network down"):
        updater.update(_memory(), [Turn(1, "user", "Remember this.")])


def test_prompt_omits_code_payload_and_bounds_turns():
    updater = _updater("[]")
    content = "User completed login.\n```python\n" + ("print('noise')\n" * 3000) + "```\nFinal constraint."
    system, _messages = updater._build_prompt(_memory(), [Turn(1, "user", content)])

    assert "[code block omitted from memory extraction]" in system
    assert "print('noise')" not in system
    assert "User completed login" in system
    assert "Final constraint" in system


def test_turn_budget_keeps_complete_newest_groups():
    updater = _updater("[]", evicted_turn_token_budget=3, token_estimator=lambda _text: 1)
    turns = [
        Turn(1, "user", "first"),
        Turn(2, "assistant", "second"),
        Turn(3, "user", "third"),
        Turn(4, "assistant", "fourth"),
    ]

    selected = updater._turns_within_budget(turns)

    assert selected == turns[-2:]
    assert updater.turn_selection_reports[-1]["selection_is_contiguous"] is True


def test_larger_batch_budget_amortizes_schema_without_dropping_exchange_groups():
    turns = [
        Turn(index, "user" if index % 2 else "assistant", "x" * 2800)
        for index in range(1, 7)
    ]
    old_budget = _updater(
        "[]", evicted_turn_token_budget=1200,
    )._plan_turn_batches(turns)
    new_updater = _updater(
        "[]", evicted_turn_token_budget=3600,
    )
    new_budget = new_updater._plan_turn_batches(turns)

    assert len(old_budget) == 3
    assert len(new_budget) == 2
    assert [turn.id for batch in new_budget for turn in batch] == [turn.id for turn in turns]
    assert new_updater.turn_selection_reports[-1]["dropped_turn_ids"] == []
    assert new_updater.turn_selection_reports[-1]["deferred_turn_ids"] == []


def test_update_token_usage_reports_provider_independent_estimates():
    updater = _updater("[]")
    updater.update(_memory(), [Turn(1, "user", "Remember this fact.")])

    usage = updater.update_token_usage()

    assert usage["source"] == "estimator"
    assert usage["calls"] == 1
    assert usage["system_tokens"] > 0
    assert usage["evicted_turn_tokens"] > 0
    assert usage["output_tokens"] > 0


def test_prompt_is_chat_only_and_has_no_evaluation_profile_rules():
    system, _ = _updater("[]")._build_prompt(
        _memory(), [Turn(1, "user", "A durable project result.")]
    )

    assert "CHAT POLICY rules" in system
    assert "operation-count limit" not in system
    assert "EVAL PROFILE" not in system
    assert "timeline/progress" not in system
