from memory_agent.core.sections import CHAT_SECTIONS
from memory_agent.core.store import Memory
from memory_agent.core.transcript import Turn
from memory_agent.policies.structured import CHAT_POLICY
from memory_agent.update.selector import UpdateMemorySelector
from memory_agent.update.updater import MemoryUpdater
from memory_agent.models.config import ProductMemoryConfig
from tests.fakes import ScriptedLLM


def _memory() -> Memory:
    memory = Memory(sections=CHAT_SECTIONS)
    memory.apply_ops([
        {"op": "ADD", "section": "facts", "text": "Deployment deadline is April 15", "provenance": [1]},
        {"op": "ADD", "section": "facts", "text": "Weather API returns JSON", "provenance": [2]},
    ])
    return memory


def test_update_selector_returns_bounded_recency_view():
    selection = UpdateMemorySelector(_memory(), token_estimator=lambda _text: 1).select_for_update(
        [Turn(id=3, role="user", content="Deployment deadline moved to May 1")], 2
    )
    assert [entry.id for entry in selection.entries] == ["F2", "F1"]
    assert selection.matches[0].reasons == ("active",)
    assert dict(selection.matches[0].score_components)["active"] == 2.0
    assert selection.matches[0].confidence == 1.0


def test_update_selector_ignores_turn_content_when_bounding_context():
    selection = UpdateMemorySelector(_memory()).select_for_update(
        [Turn(id=3, role="user", content="Tell me a joke about penguins")], 100
    )
    assert [entry.id for entry in selection.entries] == ["F2", "F1"]
    assert selection.visible_tokens > 0


def test_updater_rejects_existing_id_outside_visible_context():
    memory = _memory()
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda _system, _messages: '[{"op":"SUPERSEDE","id":"Z99","reason":"unknown"}]'),
        sections=CHAT_SECTIONS,
        max_retries=0,
    )
    applied, rejected = updater.update(
        memory, [Turn(id=3, role="user", content="Deployment deadline moved")]
    )
    assert applied == []
    assert rejected[0]["reason"] == "unknown SUPERSEDE id: Z99"
    assert memory.entries["F2"].status == "active"


def test_update_token_report_separates_prompt_components():
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda _system, _messages: '[{"op":"NOOP"}]'),
        sections=CHAT_SECTIONS,
        max_retries=0,
    )
    updater.update(_memory(), [Turn(id=3, role="user", content="Deployment deadline moved")])
    usage = updater.update_token_usage()
    assert usage["source"] == "estimator"
    assert usage["calls"] == 1
    assert usage["system_tokens"] > 0
    assert usage["visible_memory_tokens"] > 0
    assert usage["evicted_turn_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["average_tokens_per_call"] == usage["total_tokens"]


def test_evicted_turn_budget_keeps_only_complete_newest_turns():
    captured = {}
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda system, _messages: captured.setdefault("system", system) and '[{"op":"NOOP"}]'),
        sections=CHAT_SECTIONS,
        max_retries=0,
        evicted_turn_token_budget=2,
        token_estimator=lambda _text: 1,
    )
    updater.update(_memory(), [
        Turn(id=3, role="user", content="older turn"),
        Turn(id=4, role="user", content="newer turn"),
        Turn(id=5, role="user", content="newest turn"),
    ])
    assert "older turn" not in captured["system"]
    assert "newer turn" in captured["system"]
    assert "newest turn" in captured["system"]


def test_product_config_loads_separate_memory_budgets(tmp_path, monkeypatch):
    path = tmp_path / "product.yaml"
    path.write_text(
        "answer_memory_token_budget: 101\n"
        "updater:\n"
        "  max_visible_memory_tokens: 202\n"
        "  max_evicted_turn_tokens: 303\n"
    )
    monkeypatch.setenv("UPDATE_MEMORY_TOKEN_BUDGET", "222")
    config = ProductMemoryConfig.from_yaml_env(path)
    assert config.answer_memory_token_budget == 101
    assert config.update_memory_token_budget == 222
    assert config.evicted_turn_token_budget == 303


def test_product_config_loads_nested_updater_budget(tmp_path):
    path = tmp_path / "product.yaml"
    path.write_text(
        "updater:\n"
        "  max_visible_memory_tokens: 111\n"
        "  max_evicted_turn_tokens: 222\n"
        "  max_candidate_entries: 7\n"
        "  max_legacy_candidate_entries: 3\n"
    )
    config = ProductMemoryConfig.from_yaml_env(path)
    assert config.update_memory_token_budget == 111
    assert config.evicted_turn_token_budget == 222
    assert config.updater_max_candidate_entries == 7
    assert not hasattr(config, "updater_max_legacy_candidate_entries")


def test_evicted_budget_preserves_complete_multiformat_turns():
    turns = [
        Turn(1, "user", "English durable statement"),
        Turn(2, "user", "中文完整句子"),
        Turn(3, "user", "mixed 中文 JSON: {\"ok\": true}"),
        Turn(4, "assistant", "```python\nprint('完整')\n```"),
        Turn(5, "assistant", "Result rows: [1, 2, 3].")
    ]
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'), sections=CHAT_SECTIONS,
        evicted_turn_token_budget=3, token_estimator=lambda _text: 1,
    )
    selected = updater._turns_within_budget(turns)
    assert selected == turns[-3:]
    assert selected[0].content == "mixed 中文 JSON: {\"ok\": true}"


def test_evicted_budget_never_orphans_assistant_and_records_mandatory_overflow():
    turns = [
        Turn(1, "user", "x " * 3000),
        Turn(2, "assistant", "I will implement the complete requirement."),
    ]
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'), sections=CHAT_SECTIONS,
        evicted_turn_token_budget=1200,
    )

    assert updater._turns_within_budget(turns) == turns
    report = updater.turn_selection_reports[-1]
    assert report["selected_turn_ids"] == [1, 2]
    assert report["dropped_turn_ids"] == []
    assert report["oversized_mandatory_group"] is True
    assert report["mandatory_overflow_tokens"] > 0
    assert report["selection_is_contiguous"] is True


def test_chat_budget_keeps_complete_newest_exchange_groups():
    policy = CHAT_POLICY
    turns = [
        Turn(1, "user", "Project Alpha is blocked."),
        Turn(2, "assistant", "Long generic advice " * 200),
        Turn(3, "user", "Project Beta is active."),
        Turn(4, "assistant", "More generic advice " * 200),
    ]
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        sections=CHAT_SECTIONS,
        policy=policy,
        evicted_turn_token_budget=20,
        token_estimator=lambda text: max(1, len(text) // 20),
    )

    selected = updater._turns_within_budget(turns)

    assert [turn.id for turn in selected] == [3, 4]


def test_chat_budget_keeps_complete_acceptance_exchange():
    policy = CHAT_POLICY
    turns = [
        Turn(1, "assistant", "I propose using a weekly release train."),
        Turn(2, "user", "Yes, let's do that."),
    ]
    updater = MemoryUpdater(
        llm=ScriptedLLM(lambda *_: '[{"op":"NOOP"}]'),
        sections=CHAT_SECTIONS,
        policy=policy,
        evicted_turn_token_budget=1,
        token_estimator=lambda _text: 1,
    )

    assert updater._turns_within_budget(turns) == turns[-1:]
