from memory_agent.models.policy import get_memory_policy
from memory_agent.models.sections import CHAT_SECTIONS
from memory_agent.models.transcript import Turn
from memory_agent.structured.memory import Memory
from memory_agent.structured.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def _run(turns, preload=()):
    calls = []
    llm = ScriptedLLM(lambda system, messages: calls.append(system) or '[{"op":"NOOP"}]')
    updater = MemoryUpdater(
        llm=llm, sections=CHAT_SECTIONS, policy=get_memory_policy("chat"),
        max_retries=0, enable_llm_gate=True,
    )
    memory = Memory(sections=CHAT_SECTIONS, policy=get_memory_policy("chat"))
    if preload:
        memory.apply_ops(list(preload))
    updater.update(memory, turns)
    return updater, memory, calls


def test_gate_skip_no_durable_assertion():
    updater, _memory, calls = _run([Turn(1, "user", "How do I sort a list?")])
    assert calls == []
    assert updater.decision_reasons == {"skip:no_durable_assertion": 1}


def test_gate_skip_deterministic_ops_fully_cover_batch():
    updater, memory, calls = _run([
        Turn(1, "user", "Always provide pragmatic security best practices for auth.")
    ])
    assert calls == []
    assert any(e.section == "preferences" for e in memory.entries.values())
    assert updater.decision_reasons == {"skip:deterministic_ops_fully_cover_batch": 1}


def test_gate_calls_on_unresolved_subject_conflict():
    updater, _memory, calls = _run([
        Turn(2, "user", "Correction: we no longer use Redis; we use Postgres.")
    ])
    assert len(calls) == 1
    assert updater.decision_reasons == {"call:unresolved_subject_conflict": 1}


def test_gate_calls_on_ambiguous_user_acceptance():
    updater, _memory, calls = _run([
        Turn(1, "assistant", "I suggest Redis."), Turn(2, "user", "Sounds good.")
    ])
    assert len(calls) == 1
    assert updater.decision_reasons == {"call:user_acceptance_ambiguous": 1}


def test_gate_calls_on_possible_durable_assertion():
    updater, _memory, calls = _run([Turn(1, "user", "We decided to use Postgres.")])
    assert len(calls) == 1
    assert updater.decision_reasons == {"call:possible_durable_assertion": 1}


def test_gate_does_not_confirm_assistant_only_proposal():
    updater, memory, calls = _run([Turn(1, "assistant", "I recommend Redis.")])
    assert calls == []
    assert memory.entries == {}
    assert updater.decision_reasons == {"skip:no_durable_assertion": 1}

