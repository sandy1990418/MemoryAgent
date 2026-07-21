"""The updater gate is structural/token safety, not semantic profiling."""

from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.update.updater import MemoryUpdater
from tests.fakes import ScriptedLLM


def _run(turns):
    calls = []
    llm = ScriptedLLM(lambda system, messages: calls.append(system) or '[{"op":"NOOP"}]')
    updater = MemoryUpdater(
        llm=llm,
        sections=CHAT_SECTIONS,
        policy=CHAT_POLICY,
        max_retries=0,
    )
    memory = Memory(sections=CHAT_SECTIONS, policy=CHAT_POLICY)
    updater.update(memory, turns)
    return updater, memory, calls


def test_gate_does_not_skip_ordinary_turns_or_select_a_profile():
    updater, memory, calls = _run([Turn(1, "user", "How do I sort a list?")])

    assert len(calls) == 1
    assert updater.decision_reasons == {"call:llm_chat_update": 1}
    assert memory.entries == {}


def test_gate_does_not_add_semantic_ops_when_model_returns_noop():
    updater, memory, calls = _run(
        [Turn(1, "user", "We decided to use Postgres.")]
    )

    assert len(calls) == 1
    assert updater.decision_reasons["call:llm_chat_update"] == 1
    assert memory.entries == {}


def test_empty_batch_is_the_only_pre_llm_skip():
    updater, memory, calls = _run([])

    assert calls == []
    assert updater.decision_reasons == {"skip:empty_batch": 1}
    assert memory.entries == {}
