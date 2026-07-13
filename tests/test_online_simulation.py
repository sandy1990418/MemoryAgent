from evaluation.memory.final_report import build_final_report
import pytest

from evaluation.memory.online_simulation import (
    OnlineSimulation, SimulationMode, TranscriptExchange,
)
from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.retrieval.selector import MemorySelector
from memory_agent.update.updater import MemoryUpdater


class RecordingLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0

    def complete(self, system, messages, model=None):
        self.calls += 1
        return next(self.replies)


def test_token_only_uses_recorded_assistant_and_cannot_read_self_or_future_memory():
    updater_llm = RecordingLLM([
        '[{"op":"ADD","section":"facts","text":"User stated: alpha","provenance":[1]}]',
        '[{"op":"ADD","section":"facts","text":"User stated: beta","provenance":[3]}]',
    ])
    updater = MemoryUpdater(updater_llm, CHAT_SECTIONS, policy=CHAT_POLICY, max_retries=0)
    runner = OnlineSimulation(
        memory=Memory(CHAT_SECTIONS, policy=CHAT_POLICY), updater=updater,
        answer_selector=MemorySelector(policy=CHAT_POLICY),
        answer_memory_budget=100, max_window_tokens=1,
        token_estimator=lambda text: len(text.split()) if text else 0,
    )
    report = runner.run([
        TranscriptExchange("remember alpha", "recorded one"),
        TranscriptExchange("remember beta", "recorded two"),
        TranscriptExchange("what is alpha?", "recorded three"),
    ])

    assert report["answer_calls"] == 0
    assert runner.transcript.all()[1].content == "recorded one"
    assert runner.turns[0].selected_ids == ()
    assert not set(runner.turns[0].selected_ids) & set(runner.turns[0].memory_ids_after_turn)
    assert all(entry.id not in runner.turns[0].selected_ids for entry in runner.memory.entries.values())
    for turn in runner.turns:
        assert set(turn.selected_ids) <= set(turn.memory_ids_before_turn)
        newly_written = set(turn.memory_ids_after_turn) - set(turn.memory_ids_before_turn)
        assert not newly_written & set(turn.selected_ids)


def test_live_mode_requires_explicit_answer_client_and_calls_it_once_per_turn():
    updater = MemoryUpdater(RecordingLLM([]), CHAT_SECTIONS, policy=CHAT_POLICY)
    common = dict(
        memory=Memory(CHAT_SECTIONS, policy=CHAT_POLICY), updater=updater,
        answer_selector=MemorySelector(policy=CHAT_POLICY),
        answer_memory_budget=100, max_window_tokens=1000, mode=SimulationMode.LIVE,
    )
    with pytest.raises(ValueError, match="explicit answer_llm"):
        OnlineSimulation(**common)
    answer_llm = RecordingLLM(["live answer"])
    runner = OnlineSimulation(**common, answer_llm=answer_llm)
    report = runner.run([TranscriptExchange("question", "recorded answer")])
    assert report["answer_calls"] == 1
    assert answer_llm.calls == 1
    assert runner.transcript.all()[-1].content == "live answer"


def test_simulation_reports_cumulative_distribution_and_updater_attribution():
    updater = MemoryUpdater(RecordingLLM([]), CHAT_SECTIONS, policy=CHAT_POLICY)
    runner = OnlineSimulation(
        memory=Memory(CHAT_SECTIONS, policy=CHAT_POLICY), updater=updater,
        answer_selector=MemorySelector(policy=CHAT_POLICY),
        answer_memory_budget=100, max_window_tokens=1000,
        token_estimator=lambda text: len(text) if text else 0,
    )
    report = runner.run([TranscriptExchange("one", "a"), TranscriptExchange("two", "b")])
    injection = report["injection"]
    assert injection["cumulative_tokens"] == sum(turn.injection_tokens for turn in runner.turns)
    assert injection["zero_injection_turns"] == 2
    assert set(injection) >= {"average_tokens", "p50_tokens", "p95_tokens", "max_tokens"}
    answer_input = report["answer_input"]
    assert answer_input["cumulative_tokens"] == sum(
        turn.answer_input_tokens for turn in runner.turns
    )
    assert all(
        turn.answer_input_tokens >= turn.injection_tokens + turn.working_window_tokens
        for turn in runner.turns
    )
    assert report["updater"]["visible_memory_tokens"] == 0


def test_long_replay_keeps_memory_injection_bounded_while_reporting_total_input_growth():
    memory = Memory(CHAT_SECTIONS, policy=CHAT_POLICY)
    for index in range(10):
        memory.apply_ops([{
            "op": "ADD", "section": "preferences",
            "text": f"Preference {index} has enough text to consume budget", "provenance": [index + 1],
        }])
    runner = OnlineSimulation(
        memory=memory,
        updater=MemoryUpdater(RecordingLLM([]), CHAT_SECTIONS, policy=CHAT_POLICY),
        answer_selector=MemorySelector(policy=CHAT_POLICY),
        answer_memory_budget=20, max_window_tokens=10_000,
        token_estimator=lambda text: len(text.split()) if text else 0,
    )

    report = runner.run(
        TranscriptExchange(f"question {index}", f"answer {index}") for index in range(20)
    )

    assert report["injection"]["max_tokens"] <= 20
    assert report["answer_input"]["cumulative_tokens"] > report["injection"]["cumulative_tokens"]
    assert report["answer_input"]["max_tokens"] > report["answer_input"]["p50_tokens"]


def test_final_report_schema_separates_estimates_provider_and_offline_ingestion():
    report = build_final_report(
        candidate={"injection": {"average": 4, "p50": 4, "p95": 6, "max": 6,
                                  "cumulative": 12, "zero_injection_turns": 0}},
        token_estimates={"online_injection": 12},
        provider_usage={"answer": {"input_tokens": 9}},
        offline_ingestion={"estimated_tokens": 50},
    )
    assert report["tokens"]["estimates"] == {"online_injection": 12}
    assert report["tokens"]["provider_usage"] == {"answer": {"input_tokens": 9}}
    assert report["offline_ingestion"] == {"estimated_tokens": 50}
    assert report["baseline"]["routing"]["status"] == "unavailable"
    assert report["baseline"]["routing"]["reason"]
    assert set(report["candidate"]) == {
        "routing", "quality", "updater", "injection", "compactor", "holdout", "adversarial"
    }
    assert set(report["failures"]) == {
        "routing", "memory_write", "update_selection", "answer_selection", "compactor"
    }
